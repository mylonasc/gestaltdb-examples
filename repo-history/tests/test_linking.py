"""Evidence-backed cross-entity linking over an annotated fixture.

The fixture is a small synthetic repository slice with known texts, commits,
branches, and topics.  Explicit links must match exactly; inferred links are
measured with precision/recall; abstentions are recorded, not invented.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from repo_history.analysis import AnalysisResult, DecisionStatus, Provenance
from repo_history.git import (
    BranchObservation,
    CommitRecord,
    GitIndex,
    GitSelection,
    TreeEntryRecord,
)
from repo_history.github import (
    GitHubDiscovery,
    GitHubWorkItem,
    PullRequestCommit,
    PullRequestEvidence,
    RevisionAvailability,
)
from repo_history.graph import GraphSink
from repo_history.ids import IdFactory
from repo_history.labeling import TopicAssignment
from repo_history.linking import (
    LinkRelation,
    infer_topical_links,
    link_edge_writes,
    resolve_provider_links,
    resolve_textual_links,
    score_links,
)
from repo_history.models import EvidenceRef, GraphSettings, LinkingSettings, TimeWindow

UTC = timezone.utc
NOW = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)
DAY_ONE = datetime(2026, 9, 14, 10, 0, tzinfo=UTC)
DAY_TWO = datetime(2026, 9, 14, 11, 0, tzinfo=UTC)
REPOSITORY_URL = "https://github.com/mylonasc/tf_gnns"
REPOSITORY = "mylonasc/tf_gnns"
RUN_ID = "run-1"
WINDOW = TimeWindow(
    start=datetime(2026, 9, 13, tzinfo=UTC),
    end=datetime(2026, 9, 15, tzinfo=UTC),
)
C1 = "c1" * 20
C2 = "c2" * 20
MERGE = "m" * 40


def ids() -> IdFactory:
    return IdFactory(REPOSITORY_URL)


def item(number: int, kind: str = "issue", **overrides) -> GitHubWorkItem:
    data = {
        "number": number,
        "kind": kind,
        "title": f"Work {number}",
        "state": "open",
        "state_reason": None,
        "body": "",
        "author": "mylonasc",
        "created_at": DAY_ONE,
        "updated_at": DAY_ONE,
        "closed_at": None,
        "url": f"https://github.com/mylonasc/tf_gnns/issues/{number}",
        "labels": (),
        "comments": (),
        "timeline": (),
        "revisions": RevisionAvailability(work_item_number=number),
        "observed_at": NOW,
        "source_hash": "0" * 64,
    }
    data.update(overrides)
    return GitHubWorkItem(**data)


def commit_record(sha: str, **overrides) -> CommitRecord:
    data = {
        "sha": sha,
        "parents": (),
        "tree_sha": "t" * 40,
        "message": f"commit {sha[:8]}",
        "author_name": "Author",
        "author_email": "author@example.test",
        "authored_at": DAY_ONE,
        "committer_name": "Committer",
        "committer_email": "committer@example.test",
        "committed_at": DAY_ONE,
        "source_url": f"{REPOSITORY_URL}/commit/{sha}",
        "is_merge": False,
    }
    data.update(overrides)
    return CommitRecord(**data)


def assignment(
    subject_id: str, topic_id: str, topic_name: str, evidence_id: str
) -> TopicAssignment:
    return TopicAssignment(
        subject_id=subject_id,
        kind="derived",
        action="reuse",
        topic_id=topic_id,
        topic_name=topic_name,
        confidence=0.9,
        rationale="fixture",
        registry_version=1,
        result=AnalysisResult.create(
            subject_id=subject_id,
            kind="topic",
            status=DecisionStatus.ACCEPTED,
            value={"action": "reuse", "topic_id": topic_id},
            provenance=Provenance(
                method="deterministic-tfidf-labeler",
                method_version="1",
                run_id=RUN_ID,
                info_horizon=NOW,
            ),
            evidence=(
                EvidenceRef(
                    evidence_id=evidence_id,
                    source="labeling",
                    source_id=subject_id,
                ),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Textual references
# ---------------------------------------------------------------------------


def test_fixture_text_resolves_exact_explicit_links():
    known = {10: ids().work_item("github", 10), 11: ids().work_item("github", 11),
             12: ids().work_item("github", 12)}
    report = resolve_textual_links(
        [
            (
                known[10],
                "Fixes #10 and see #12. Also mylonasc/tf_gnns#11.",
                "issue:10:body",
                "https://github.com/mylonasc/tf_gnns/issues/10",
            ),
            (
                ids().commit(C1),
                f"Merge pull request #11 from mylonasc/feat. "
                f"See https://github.com/mylonasc/tf_gnns/issues/12, "
                f"other/repo#5, closes #99, commit {C2}, deadbee5, docs/guide.md.",
                "commit:c1",
                f"{REPOSITORY_URL}/commit/{C1}",
            ),
        ],
        ids=ids(),
        items=known,
        commits={C1, C2, MERGE},
        files={"docs/guide.md"},
        repository=REPOSITORY,
        run_id=RUN_ID,
        horizon=NOW,
    )

    triples = {
        (link.source_id, link.relation.value, link.target_id)
        for link in report.links
    }
    assert triples == {
        (known[10], "INTENDS_TO_RESOLVE", known[10]),
        (known[10], "MENTIONS", known[12]),
        (known[10], "MENTIONS", known[11]),
        (ids().commit(C1), "MENTIONS", known[11]),
        (ids().commit(C1), "MENTIONS", known[12]),
        (ids().commit(C1), "MENTIONS", ids().commit(C2)),
        (ids().commit(C1), "MENTIONS", ids().file("docs/guide.md")),
    }
    # Merge-PR mentions are never mistaken for intent.
    assert not any(
        link.relation is LinkRelation.INTENDS_TO_RESOLVE
        and link.source_id == ids().commit(C1)
        for link in report.links
    )
    # Everything unresolved or foreign is reported, never linked.
    unresolved = {
        (ref.reference.target, ref.reason) for ref in report.references if not ref.resolved
    }
    assert any(target == "5" and "foreign" in reason for target, reason in unresolved)
    assert any(target == "99" and "not in the known selection" in reason
               for target, reason in unresolved)
    assert any(target == "deadbee5" and "no known commit" in reason
               for target, reason in unresolved)
    # Every link carries inspectable evidence and a rationale.
    assert all(link.evidence and link.rationale for link in report.links)
    assert all(link.run_id == RUN_ID and link.info_horizon == NOW
               for link in report.links)

    scores = score_links(
        [(source, relation, target) for source, relation, target in triples],
        [
            (known[10], "INTENDS_TO_RESOLVE", known[10]),
            (known[10], "MENTIONS", known[12]),
            (known[10], "MENTIONS", known[11]),
            (ids().commit(C1), "MENTIONS", known[11]),
            (ids().commit(C1), "MENTIONS", known[12]),
            (ids().commit(C1), "MENTIONS", ids().commit(C2)),
            (ids().commit(C1), "MENTIONS", ids().file("docs/guide.md")),
        ],
    )
    assert scores == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "true_positives": 7.0}


def test_ambiguous_sha_prefixes_do_not_link():
    report = resolve_textual_links(
        [(ids().commit(C1), f"see {C1[:7]}", "commit:c1", None)],
        ids=ids(),
        items={},
        commits={C1, C1[:7] + "ff" + "0" * 30},
        files=set(),
        repository=REPOSITORY,
        run_id=RUN_ID,
        horizon=NOW,
    )

    assert report.links == ()
    assert len(report.references) == 1
    assert report.references[0].resolved is False
    assert "ambiguous" in report.references[0].reason


def test_resolution_is_deterministic():
    texts = [
        (
            ids().work_item("github", 10),
            "Fixes #10, see #12 and docs/guide.md.",
            "issue:10:body",
            None,
        )
    ]
    arguments = {
        "ids": ids(),
        "items": {10: ids().work_item("github", 10), 12: ids().work_item("github", 12)},
        "commits": {C1},
        "files": {"docs/guide.md"},
        "repository": REPOSITORY,
        "run_id": RUN_ID,
        "horizon": NOW,
    }

    assert resolve_textual_links(texts, **arguments) == resolve_textual_links(
        texts, **arguments
    )


# ---------------------------------------------------------------------------
# Provider relationships
# ---------------------------------------------------------------------------


def fixture_index() -> GitIndex:
    selection = GitSelection(
        anchor="a" * 40,
        window=WINDOW,
        selected_commits=(commit_record(C1, committed_at=DAY_ONE),),
        boundary_commits=(commit_record(C2, committed_at=DAY_ONE),),
    )
    return GitIndex(
        selection=selection,
        branches=(
            BranchObservation(
                name="main", ref="refs/heads/main", tip_sha=C1,
                contains_anchor=True,
            ),
        ),
    )


def fixture_discovery() -> GitHubDiscovery:
    return GitHubDiscovery(
        repository=REPOSITORY,
        observed_at=NOW,
        offline=True,
        work_items=(
            item(10, body="Fixes #10."),
            item(11, kind="pull_request", state="closed"),
        ),
        pull_requests=(
            PullRequestEvidence(
                number=11,
                title="Work 11",
                state="closed",
                merged=True,
                merged_at=DAY_TWO,
                merge_commit_sha=MERGE,
                head_ref="feat",
                head_sha="h" * 40,
                base_ref="main",
                base_sha=C1,
                commits=(
                    PullRequestCommit(sha="p1" * 20, message="one"),
                    PullRequestCommit(sha="p2" * 20, message="two"),
                ),
                url="https://github.com/mylonasc/tf_gnns/pull/11",
            ),
        ),
        provider_labels=(),
        label_assignments=(),
        hierarchy_edges=(),
        absences=(),
        gaps=(),
        fetches=(),
        counters={},
        truncated=False,
    )


def test_provider_links_capture_closure_and_branch_status():
    report = resolve_provider_links(
        fixture_discovery(), fixture_index(), ids=ids(), run_id=RUN_ID, horizon=NOW
    )

    by_relation = {}
    for link in report.links:
        by_relation.setdefault(link.relation.value, []).append(link)

    closed = by_relation["CLOSED_BY"]
    assert len(closed) == 1
    assert closed[0].source_id == ids().work_item("github", 11)
    assert closed[0].target_id == ids().commit(MERGE)
    assert "merge style is non-head" in closed[0].rationale
    assert "2 PR commit(s)" in closed[0].rationale

    branches = {link.association: link for link in by_relation["RELATED_TO"]}
    assert branches["branch:head"].target_id == ids().branch("feat")
    assert branches["branch:head"].confidence == 0.5
    assert "deleted" in branches["branch:head"].rationale
    assert branches["branch:base"].target_id == ids().branch("main")
    assert branches["branch:base"].confidence == 0.9

    assert any(
        gap.code == "link-merge-commit-outside-selection" for gap in report.gaps
    )


def test_remote_tracking_branches_resolve_pr_head_refs():
    base = commit_record(C2, committed_at=DAY_ONE)
    selection = GitSelection(
        anchor="a" * 40, window=WINDOW, selected_commits=(),
        boundary_commits=(base,),
    )
    remote_index = GitIndex(
        selection=selection,
        branches=(
            BranchObservation(
                name="origin/feat", ref="refs/remotes/origin/feat",
                tip_sha=C2, contains_anchor=False, source="remote",
            ),
        ),
    )
    pull = PullRequestEvidence(
        number=11, title="Work 11", state="open", merged=False,
        head_ref="feat", head_sha="h" * 40, base_ref="main", base_sha=C2,
        url="https://github.com/mylonasc/tf_gnns/pull/11",
    )
    discovery = GitHubDiscovery(
        repository=REPOSITORY, observed_at=NOW, offline=True,
        work_items=(item(11, kind="pull_request"),),
        pull_requests=(pull,),
        provider_labels=(), label_assignments=(), hierarchy_edges=(),
        absences=(), gaps=(), fetches=(), counters={}, truncated=False,
    )
    report = resolve_provider_links(
        discovery, remote_index, ids=ids(), run_id=RUN_ID, horizon=NOW
    )

    heads = [link for link in report.links if link.association == "branch:head"]
    bases = [link for link in report.links if link.association == "branch:base"]
    assert len(heads) == 1 and heads[0].confidence == 0.9
    assert "deleted" not in heads[0].rationale
    # The unknown base branch is reported as deleted, not resolved.
    assert len(bases) == 1 and bases[0].confidence == 0.5


def test_link_records_convert_to_results_and_graph_writes(tmp_path):
    report = resolve_provider_links(
        fixture_discovery(), fixture_index(), ids=ids(), run_id=RUN_ID, horizon=NOW
    )

    results = [link.to_result() for link in report.links]
    assert all(result.kind == "link" for result in results)
    assert all(result.status is DecisionStatus.ACCEPTED for result in results)
    assert len({result.result_id for result in results}) == len(results)

    writes = link_edge_writes(report.links, run_id=RUN_ID)
    sink = GraphSink.create(
        GraphSettings(path=str(tmp_path / "graph"), backend="lmdb"),
        run_id=RUN_ID,
        overwrite=True,
    )
    summary = sink.write_relationships(writes, analysis_run_id=RUN_ID)
    assert summary.relationships_written == len(writes)
    counts = sink.edge_counts(["CLOSED_BY", "RELATED_TO"])
    assert counts == {"CLOSED_BY": 1, "RELATED_TO": 2}
    sink.close()


# ---------------------------------------------------------------------------
# Inference and abstention
# ---------------------------------------------------------------------------


def test_topical_inference_links_shared_topics_and_abstains_weakly():
    topic_one = ids().topic("sparse-training")
    topic_two = ids().topic("graph-convolution")
    topic_three = ids().topic("docs-style")
    item_key = ids().work_item("github", 10)
    file_key = ids().file("docs/guide.md")
    weak_key = ids().file("other.py")

    report = infer_topical_links(
        {
            item_key: [
                assignment(item_key, topic_one, "sparse-training", "ev:1"),
                assignment(item_key, topic_two, "graph-convolution", "ev:2"),
            ]
        },
        {
            file_key: [
                assignment(file_key, topic_two, "graph-convolution", "ev:3"),
                assignment(file_key, topic_three, "docs-style", "ev:4"),
            ],
            weak_key: [
                assignment(weak_key, topic_three, "docs-style", "ev:5"),
            ],
        },
        ids=ids(),
        settings=LinkingSettings(min_inference_score=0.3),
        run_id=RUN_ID,
        horizon=NOW,
    )

    assert len(report.links) == 1
    link = report.links[0]
    assert (link.source_id, link.target_id) == (item_key, file_key)
    assert link.relation is LinkRelation.RELATED_TO
    assert link.confidence == pytest.approx(1 / 3)
    assert "graph-convolution" in link.rationale
    assert {ref.evidence_id for ref in link.evidence} == {"ev:2", "ev:3"}

    # The weak pair shares nothing and stays silent; a near miss would abstain.
    assert report.abstained == ()

    near = infer_topical_links(
        {item_key: [assignment(item_key, topic_one, "sparse-training", "ev:1")]},
        {
            file_key: [
                assignment(file_key, topic_one, "sparse-training", "ev:6"),
                assignment(file_key, topic_two, "graph-convolution", "ev:7"),
                assignment(file_key, topic_three, "docs-style", "ev:8"),
                assignment(file_key, ids().topic("extra"), "extra", "ev:9"),
            ]
        },
        ids=ids(),
        settings=LinkingSettings(min_inference_score=0.3),
        run_id=RUN_ID,
        horizon=NOW,
    )
    assert near.links == ()
    assert len(near.abstained) == 1
    assert near.abstained[0].score == pytest.approx(0.25)
    assert "below" in near.abstained[0].reason


def test_inference_ignores_provider_only_assignments():
    lonely = TopicAssignment(
        subject_id="s",
        kind="provider",
        action="preserve",
        provider_label="bug",
        confidence=1.0,
        rationale="preserved",
        registry_version=-1,
        result=assignment("s", ids().topic("x"), "x", "ev:0").result,
    )
    report = infer_topical_links(
        {"a": [lonely]},
        {"b": [lonely]},
        ids=ids(),
        run_id=RUN_ID,
        horizon=NOW,
    )
    assert report.links == ()
    assert report.abstained == ()


def test_score_links_measures_quality():
    expected = [("a", "MENTIONS", "b"), ("a", "MENTIONS", "c")]
    assert score_links(expected, expected)["f1"] == 1.0
    partial = score_links([("a", "MENTIONS", "b")], expected)
    assert partial["precision"] == 1.0
    assert partial["recall"] == 0.5
    assert score_links([], expected)["recall"] == 0.0


def test_linking_settings_load_from_toml():
    from repo_history.config import load_config

    assert load_config().linking.min_inference_score == 0.3
    with pytest.raises(Exception):
        LinkingSettings(min_inference_score=1.5)
