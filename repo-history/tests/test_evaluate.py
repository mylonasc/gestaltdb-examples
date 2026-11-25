"""End-to-end demonstrator evaluation: synthetic world, recorded evidence.

The synthetic world is fully offline (in-process pygit2 repository plus an
``httpx.MockTransport`` GitHub): no secrets, no network.  The recorded run
replays scrubbed live responses from ``tests/data/github``.  Nothing here
claims unmeasured accuracy; quality is scored only on the annotated fixture.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pygit2
import pytest

from repo_history.budgets import BudgetLedger
from repo_history.config import load_config
from repo_history.content import ContentExtractor
from repo_history.enrichment import EnrichmentCache
from repo_history.evaluate import (
    EvaluationReport,
    Timer,
    score_topic_decisions,
    storage_bytes,
    summarize_ledger,
)
from repo_history.events import EventKind
from repo_history.git import GitHistory, GitIndex, GitSelection
from repo_history.github import (
    EvidenceCache,
    GitHubCredentials,
    GitHubDiscoverer,
    GitHubSettings,
)
from repo_history.graph import (
    GraphSink,
    map_classes,
    map_diffs,
    map_discovery,
    map_folders,
    map_git_index,
    map_imports,
    map_registry,
    map_topic_assignments,
)
from repo_history.ids import IdFactory
from repo_history.inspection import Inspector
from repo_history.labeling import (
    DeterministicLabeler,
    extract_snapshot_features,
    extract_work_item_features,
    seed_registry,
)
from repo_history.linking import (
    infer_topical_links,
    link_edge_writes,
    resolve_provider_links,
    resolve_textual_links,
    score_links,
)
from repo_history.models import Budgets, GraphSettings, TimeWindow
from repo_history.orchestration import (
    JournalManifest,
    StageRunner,
    TemporalCoordinator,
    git_selection_events,
)
from repo_history.worker import JobQueue, WorkJob, make_job_id, run_job_file

UTC = timezone.utc
NOW = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)
DAY_ONE = datetime(2026, 9, 14, 10, 0, tzinfo=UTC)
DAY_TWO = datetime(2026, 9, 14, 11, 0, tzinfo=UTC)
WINDOW = TimeWindow(
    start=datetime(2026, 9, 13, tzinfo=UTC),
    end=datetime(2026, 9, 15, tzinfo=UTC),
)
REPOSITORY_URL = "https://github.com/mylonasc/tf_gnns"
RUN_ID = "e2e"
MERGE = "m" * 40


def seconds(value: datetime) -> int:
    return int(value.timestamp())


# ---------------------------------------------------------------------------
# Synthetic world builders
# ---------------------------------------------------------------------------


def build_repo(path: Path) -> tuple[GitHistory, dict[str, str]]:
    repo = pygit2.init_repository(str(path), bare=False)
    history = GitHistory(repo, REPOSITORY_URL)

    def tree(spec: dict) -> pygit2.Oid:
        builder = repo.TreeBuilder()
        for name, value in spec.items():
            if isinstance(value, dict):
                builder.insert(name, tree(value), int(pygit2.enums.FileMode.TREE))
            else:
                content, mode = value
                blob = repo.create_blob(content.encode())
                builder.insert(name, blob, int(mode))
        return builder.write()

    def commit(label: str, spec: dict, message: str, *, parents=(), at=DAY_ONE) -> str:
        author = pygit2.Signature("Author", "a@t", seconds(at), 0)
        committer = pygit2.Signature("Committer", "c@t", seconds(at), 0)
        oid = repo.create_commit(
            "refs/heads/main", author, committer, message, tree(spec),
            [pygit2.Oid(hex=parent) for parent in parents],
        )
        return str(oid)

    root = commit(
        "root",
        {"README.md": ("# demo\n", 0o100644),
         "docs": {"guide.md": ("# Guide\nFixes #10.\n", 0o100644)}},
        "Start docs",
        at=DAY_ONE,
    )
    tip = commit(
        "tip",
        {"README.md": ("# demo\n", 0o100644),
         "docs": {"guide.md": ("# Guide\nFixes #10.\n", 0o100644)},
         "models": {"gcn.py": ("import torch\nfrom models import graphnet\n\nclass GraphConv:\n    pass\n", 0o100644),
                    "graphnet.py": ('"""Graph networks."""\n', 0o100644)}},
        "Add graph convolution; fixes #10",
        parents=(root,),
        at=DAY_TWO,
    )
    return history, {"root": root, "tip": tip}


def build_transport() -> httpx.MockTransport:
    base = "https://api.github.com/repos/mylonasc/tf_gnns"

    def issue(number: int, **overrides):
        payload = {
            "number": number,
            "title": f"Work {number}",
            "state": "open",
            "body": "See docs/guide.md.",
            "user": {"login": "mylonasc"},
            "created_at": "2026-09-14T10:00:00Z",
            "updated_at": "2026-09-14T11:00:00Z",
            "closed_at": None,
            "html_url": f"https://github.com/mylonasc/tf_gnns/issues/{number}",
            "labels": [{"id": 1, "name": "docs", "color": "fff", "default": False,
                        "url": f"{base}/labels/docs"}],
        }
        payload.update(overrides)
        return payload

    pull = {
        "number": 11, "title": "Work 11", "state": "closed", "merged": True,
        "merged_at": "2026-09-14T11:30:00Z", "merge_commit_sha": MERGE,
        "head": {"ref": "feat", "sha": "h" * 40, "repo": {"full_name": "mylonasc/tf_gnns"}},
        "base": {"ref": "main", "sha": "b" * 40, "repo": {"full_name": "mylonasc/tf_gnns"}},
        "html_url": "https://github.com/mylonasc/tf_gnns/pull/11",
    }
    routes = {
        f"{base}/issues/10": issue(10),
        f"{base}/issues/10/comments": [],
        f"{base}/issues/10/timeline": [],
        f"{base}/issues/10/sub_issues": [],
        f"{base}/issues/11": {**issue(11, title="Work 11"),
                              "pull_request": {"url": f"{base}/pulls/11"}},
        f"{base}/issues/11/comments": [],
        f"{base}/issues/11/timeline": [],
        f"{base}/issues/11/sub_issues": [],
        f"{base}/pulls/11": pull,
        f"{base}/pulls/11/commits": [],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        payload = routes.get(str(request.url))
        if payload is None:
            return httpx.Response(404, json={"message": "Not Found"})
        return httpx.Response(200, json=payload, headers={"etag": '"e2e"'})

    return httpx.MockTransport(handler)


# ---------------------------------------------------------------------------
# Helper units
# ---------------------------------------------------------------------------


def test_timer_storage_ledger_and_topic_scoring(tmp_path):
    with Timer() as timer:
        (tmp_path / "blob.bin").write_bytes(b"x" * 100)
    assert timer.seconds >= 0.0
    assert storage_bytes(tmp_path / "blob.bin") == 100
    assert storage_bytes(tmp_path) == 100
    assert storage_bytes(tmp_path / "missing") == 0

    ledger = BudgetLedger(Budgets())
    ledger.reserve_api(2)
    assert summarize_ledger(ledger)["api_requests"] == 2

    predicted = [("s1", "reuse", "topic:a"), ("s2", "propose", "new-thing")]
    assert score_topic_decisions(predicted, predicted)["accuracy"] == 1.0
    assert score_topic_decisions(predicted[:1], predicted)["accuracy"] == 0.5


def test_evaluation_report_marks_synthetic_and_unmeasured():
    report = EvaluationReport(
        run_id=RUN_ID, repository="mylonasc/tf_gnns", synthetic=True,
        seconds={"parse": 1.5}, counts={"commits": 2},
        budgets={"api_requests": 4}, storage={"graph": 1024},
        gaps=(), notes=("synthetic fixture",),
    )

    text = report.to_markdown()
    assert "synthetic fixture" in text
    assert "no quality claims" in text
    assert "api_requests" in text and "4" in text


# ---------------------------------------------------------------------------
# Recorded evidence replay
# ---------------------------------------------------------------------------


def test_recorded_github_evidence_replays_offline():
    root = Path(__file__).parent / "data"
    assert (root / "github").is_dir()
    cache = EvidenceCache(root, offline=True)
    discovery = GitHubDiscoverer(
        load_config().repository,
        GitHubSettings(),
        cache=cache,
        ledger=BudgetLedger(Budgets()),
    ).run()

    assert [item.number for item in discovery.work_items] == [10, 11, 12]
    assert all(item.kind == "pull_request" for item in discovery.work_items)
    assert {pull.merge_commit_sha for pull in discovery.pull_requests} == {
        "f787d50f5bb8fc27b2ed0a0d3a0aa0a729e91eeb",
        "3f7ea9879666c5b04fb5dfec8bab926a26079e42",
        "6324b89cd52dceb9963e44cb5c3e1716dd5a1698",
    }
    assert [gap.code for gap in discovery.gaps] == [
        "github-historical-revisions-unavailable"
    ]
    assert discovery.counters["api_requests"] == 0
    assert len(discovery.to_events(IdFactory(REPOSITORY_URL))) == 31


# ---------------------------------------------------------------------------
# Full synthetic end-to-end run
# ---------------------------------------------------------------------------


def test_synthetic_end_to_end_pipeline(tmp_path):
    repository = load_config().repository.model_copy(
        update={"allowed_work_items": (10, 11)}
    )
    ids = IdFactory(REPOSITORY_URL)
    ledger = BudgetLedger(Budgets(max_api_requests=50, max_work_items=10))
    timings: dict[str, float] = {}

    with Timer() as timer:
        history, shas = build_repo(tmp_path / "repo")
        selection = history.select(
            shas["tip"], WINDOW, max_depth=64, skew_tolerance_seconds=0
        )
    timings["git-select"] = timer.seconds
    assert selection.selected_shas == (shas["root"], shas["tip"])
    assert selection.boundary_shas == ()

    with Timer() as timer:
        index = history.index(selection, ledger=ledger)
    timings["git-index"] = timer.seconds
    assert index.counters["snapshots"] == 6

    with Timer() as timer:
        cache = EvidenceCache(tmp_path / "cache")
        discoverer = GitHubDiscoverer(
            repository, GitHubSettings(), ledger=ledger, cache=cache,
            credentials=GitHubCredentials(), transport=build_transport(),
            clock=lambda: NOW,
        )
        discovery = discoverer.run()
    timings["github"] = timer.seconds
    assert [item.number for item in discovery.work_items] == [10, 11]
    assert ledger.api_requests > 0

    with Timer() as timer:
        from repo_history.orchestration import EventJournal

        manifest = JournalManifest(
            run_id=RUN_ID, repository_id=ids.repository_id,
            repository_url=ids.repository_url, anchor=shas["tip"],
            window_start=WINDOW.start, window_end=WINDOW.end,
            config_fingerprint="e" * 64, software_version="test", created_at=NOW,
        )
        runner = StageRunner.resume_or_create(
            tmp_path / "journal.jsonl", tmp_path / "checkpoint.json",
            manifest, Budgets(max_api_requests=50, max_work_items=10),
        )
        calls: list[str] = []

        def git_stage():
            calls.append("git")
            return git_selection_events(selection, ids, observed_at=NOW), selection.gaps

        def github_stage():
            calls.append("github")
            return (
                discovery.to_events(ids),
                discovery.gaps,
            )

        runner.run_stage("git", git_stage)
        runner.run_stage("github", github_stage)
        resumed = StageRunner.resume_or_create(
            tmp_path / "journal.jsonl", tmp_path / "checkpoint.json",
            manifest, Budgets(max_api_requests=50, max_work_items=10),
        )
        resumed.run_stage("git", git_stage)
        resumed.run_stage("github", github_stage)
    timings["journal"] = timer.seconds
    assert calls == ["git", "github"]  # resume repeats nothing
    events = runner.journal.read().events
    coordinator = TemporalCoordinator(repository)
    first = coordinator.collect(events)
    assert coordinator.collect(events) == first
    assert first.deferred == ()
    assert first.effective

    with Timer() as timer:
        content = ContentExtractor.from_history(history).extract(index)
    timings["content"] = timer.seconds
    assert content.counters["parsed_contents"] >= 2

    with Timer() as timer:
        worker = DeterministicLabeler()
        registry = seed_registry(at=NOW)
        features = [
            extract_snapshot_features(snap, content.content_for(snap.snapshot_id),
                                      subject_id=snap.snapshot_id)
            for snap in content.snapshots
        ]
        first_pass = [
            worker.assign(f, registry, subject_id=f.subject_id,
                          run_id=RUN_ID, info_horizon=NOW)
            for f in features
        ]
        assert {assignment.action for assignment in first_pass} == {"propose"}
        for assignment, feat in zip(first_pass, features):
            name = assignment.result.value["proposed_name"]
            if registry.lookup(name) is None:
                registry, _ = registry.register(
                    name, at=NOW, created_by=RUN_ID, ids=ids, description=feat.text(),
                )
        second_pass = [
            worker.assign(f, registry, subject_id=f.subject_id,
                          run_id=RUN_ID, info_horizon=NOW)
            for f in features
        ]
        assert {assignment.action for assignment in second_pass} == {"reuse"}
        scored = score_topic_decisions(
            [(a.subject_id, a.action, a.topic_id) for a in second_pass],
            [(a.subject_id, "reuse", a.topic_id) for a in second_pass],
        )
    timings["labeling"] = timer.seconds
    assert scored["accuracy"] == 1.0

    item_features = {
        item.number: extract_work_item_features(
            item, subject_id=f"workitem:github:{item.number}"
        )
        for item in discovery.work_items
    }
    texts = [
        (ids.work_item("github", 10), "Fixes #10. See docs/guide.md.",
         "issue:10:body", "https://example.test/10"),
        (ids.commit(shas["tip"]), "Add graph convolution; fixes #10",
         f"commit:{shas['tip']}", "https://example.test/c"),
    ]
    text_report = resolve_textual_links(
        texts, ids=ids,
        items={n: ids.work_item("github", n) for n in (10, 11)},
        commits={shas["root"], shas["tip"], MERGE},
        files={"docs/guide.md"}, repository="mylonasc/tf_gnns",
        run_id=RUN_ID, horizon=NOW,
    )
    provider_report = resolve_provider_links(
        discovery, index, ids=ids, run_id=RUN_ID, horizon=NOW
    )
    topical_report = infer_topical_links(
        {f"workitem:github:{n}": [worker.assign(
            item_features[n], registry, subject_id=f"workitem:github:{n}",
            run_id=RUN_ID, info_horizon=NOW)] for n in (10, 11)},
        {f.subject_id: [worker.assign(
            f, registry, subject_id=f.subject_id, run_id=RUN_ID, info_horizon=NOW)]
         for f in features},
        ids=ids, run_id=RUN_ID, horizon=NOW,
    )
    predicted = [
        (link.source_id, link.relation.value, link.target_id)
        for link in (*text_report.links, *provider_report.links)
    ]
    expected = [
        (ids.work_item("github", 10), "INTENDS_TO_RESOLVE", ids.work_item("github", 10)),
        (ids.work_item("github", 10), "MENTIONS", ids.file("docs/guide.md")),
        (ids.commit(shas["tip"]), "INTENDS_TO_RESOLVE", ids.work_item("github", 10)),
        (ids.work_item("github", 11), "CLOSED_BY", ids.commit(MERGE)),
        (ids.work_item("github", 11), "RELATED_TO", ids.branch("feat")),
        (ids.work_item("github", 11), "RELATED_TO", ids.branch("main")),
    ]
    quality = score_links(predicted, expected)
    assert quality["recall"] == 1.0, quality
    assert quality["precision"] == 1.0, quality

    with Timer() as timer:
        git_mapped = map_git_index(index, ids, run_id=RUN_ID)
        gh_mapped = map_discovery(discovery, ids, run_id=RUN_ID)
        link_writes = link_edge_writes(
            [*text_report.links, *provider_report.links, *topical_report.links],
            run_id=RUN_ID,
        )
        folders_mapped = map_folders(
            [snapshot.path for snapshot in content.snapshots],
            ids, run_id=RUN_ID, window_start=WINDOW.start,
        )
        topics_mapped = map_registry(registry, ids, run_id=RUN_ID)
        about_mapped = map_topic_assignments(second_pass, ids, run_id=RUN_ID)
        classes_mapped = map_classes(content, index, ids, run_id=RUN_ID)
        imports_mapped = map_imports(content, index, ids, run_id=RUN_ID)
        diffs_mapped = map_diffs(content, index, ids, run_id=RUN_ID)
        sink = GraphSink.create(
            GraphSettings(path=str(tmp_path / "graph"), backend="lmdb"),
            run_id=RUN_ID, overwrite=True,
        )
        sink.write_entities(
            [*git_mapped.entities, *gh_mapped.entities, *topics_mapped.entities,
             *classes_mapped.entities, *folders_mapped.entities, *diffs_mapped.entities],
            analysis_run_id=RUN_ID,
        )
        sink.write_relationships(
            [*git_mapped.relationships, *gh_mapped.relationships, *link_writes,
             *folders_mapped.relationships, *about_mapped.relationships,
             *classes_mapped.relationships, *imports_mapped.relationships,
             *diffs_mapped.relationships],
            analysis_run_id=RUN_ID,
        )
        dup_entities = sink.write_entities(
            [*git_mapped.entities, *gh_mapped.entities, *topics_mapped.entities,
             *classes_mapped.entities, *folders_mapped.entities, *diffs_mapped.entities],
            analysis_run_id=RUN_ID,
        )
    timings["graph"] = timer.seconds
    assert dup_entities.entities_duplicate == (
        len(git_mapped.entities) + len(gh_mapped.entities) + len(topics_mapped.entities)
        + len(classes_mapped.entities) + len(folders_mapped.entities)
        + len(diffs_mapped.entities)
    )
    assert sink.label_counts()["Commit"] == 2
    assert sink.label_counts()["WorkItem"] == 2
    assert sink.label_counts()["Topic"] >= 1
    assert sink.label_counts()["Folder"] >= 1
    assert sink.label_counts()["Class"] >= 1
    assert sink.edge_counts()["DEFINED_AT"] >= 1
    assert sink.edge_counts()["IMPORTS"] >= 1
    assert sink.edge_counts()["TOUCHES"] >= 1
    assert sink.edge_counts()["BELONGS_TO"] >= 1
    assert sink.edge_counts()["ABOUT"] >= 1

    inspector = Inspector(
        index=index, discovery=discovery, content=content,
        links=[*text_report.links, *provider_report.links, *topical_report.links],
        assignments=[*first_pass, *second_pass], info_horizon=NOW, run_id=RUN_ID,
        repository_url=REPOSITORY_URL,
    )
    assert len(inspector.changes_in_range(shas["root"], shas["tip"]).commits) == 1
    assert inspector.work_items_for_commits([shas["tip"]])
    assert inspector.files_for_work_item(10, NOW)
    assert inspector.explain(ids.work_item("github", 10))
    assert inspector.coverage().counts["links"] == len(text_report.links) + len(
        provider_report.links) + len(topical_report.links)

    queue = JobQueue(tmp_path / "queue")
    job = WorkJob(
        job_id=make_job_id(REPOSITORY_URL, RUN_ID, 10),
        subject_id="workitem:github:10", work_item_number=10,
        subject_text=item_features[10].text(), features=item_features[10],
        candidates=(),
        registry=registry, info_horizon=NOW, run_id=RUN_ID,
    )
    queue.submit(job)
    summary = run_job_file(str(tmp_path / "queue"), job.job_id)
    assert summary["required_status"] == "accepted"

    report = EvaluationReport(
        run_id=RUN_ID, repository="mylonasc/tf_gnns", synthetic=True,
        seconds=timings,
        counts={"commits": 2, "work_items": 2, "links": len(predicted)},
        budgets=summarize_ledger(ledger),
        storage={
            "graph": storage_bytes(tmp_path / "graph"),
            "journal": storage_bytes(tmp_path / "journal.jsonl"),
        },
        quality={"links": quality, "topics": scored},
        gaps=(),
        notes=("synthetic end-to-end fixture",),
    )
    text = report.to_markdown()
    assert "synthetic fixture" in text
    sink.close()
    history.close()
