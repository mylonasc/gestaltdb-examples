"""Single-writer GestaltDB temporal graph persistence.

Mapping tests are pure (no database); sink tests use throwaway LMDB graphs
under ``tmp_path``.  No network, no Git CLI, no model credentials.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from repo_history.content import ContentIndex, SnapshotContent
from repo_history.git import (
    BranchObservation,
    CommitRecord,
    FileChangeRecord,
    GitIndex,
    GitSelection,
    TreeEntryRecord,
)
from repo_history.github import (
    GitHubDiscovery,
    GitHubWorkItem,
    HierarchyEdge,
    LabelAssignment,
    ProviderLabel,
    PullRequestEvidence,
    RevisionAvailability,
)
from repo_history.graph import (
    ABOUT,
    BELONGS_TO,
    CHANGES,
    CHILD_OF,
    HAS_BRANCH,
    HAS_FOLDER,
    HAS_LABEL,
    IMPORTS,
    MEMBER_OF,
    MERGED_AS,
    OF_FILE,
    PARENT_COMMIT,
    TIP_OF,
    TOUCHES,
    EntityWrite,
    GraphClosedError,
    GraphError,
    GraphSink,
    map_diffs,
    map_discovery,
    map_folders,
    map_git_index,
    map_imports,
    map_registry,
    map_repository_branches,
    map_topic_assignments,
)
from repo_history.ids import IdFactory
from repo_history.labeling import TopicAssignment, seed_registry
from repo_history.models import GraphSettings, TimeWindow
from repo_history.protocols import GraphSink as GraphSinkProtocol

UTC = timezone.utc
NOW = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)
T0 = datetime(2026, 9, 14, 9, 0, tzinfo=UTC)
DAY_ONE = datetime(2026, 9, 14, 10, 0, tzinfo=UTC)
DAY_TWO = datetime(2026, 9, 14, 11, 0, tzinfo=UTC)
REPOSITORY_URL = "https://github.com/mylonasc/tf_gnns"
WINDOW = TimeWindow(
    start=datetime(2026, 9, 13, tzinfo=UTC),
    end=datetime(2026, 9, 15, tzinfo=UTC),
)
RUN_ID = "run-1"


def ids() -> IdFactory:
    return IdFactory(REPOSITORY_URL)


def commit(
    sha: str, *, parents: tuple[str, ...] = (), committed_at: datetime = DAY_ONE
) -> CommitRecord:
    return CommitRecord(
        sha=sha,
        parents=parents,
        tree_sha="t" * 40,
        message=f"commit {sha[:8]}",
        author_name="Author",
        author_email="author@example.test",
        authored_at=committed_at,
        committer_name="Committer",
        committer_email="committer@example.test",
        committed_at=committed_at,
        source_url=f"{REPOSITORY_URL}/commit/{sha}",
        is_merge=len(parents) > 1,
    )


def selection(
    selected: tuple[CommitRecord, ...], boundary: tuple[CommitRecord, ...] = ()
) -> GitSelection:
    return GitSelection(
        anchor="a" * 40, window=WINDOW, selected_commits=selected,
        boundary_commits=boundary,
    )


def git_index(selection_: GitSelection) -> GitIndex:
    snapshots = tuple(
        TreeEntryRecord(
            snapshot_id=f"snap:{record.sha[:8]}:x.py",
            commit_sha=record.sha,
            path="x.py",
            object_id="o" * 40,
            mode=0o100644,
            kind="blob",
            size=10,
            is_binary=False,
            is_lfs_pointer=False,
            content_status="inspected",
            is_baseline=record.sha == selection_.boundary_commits[0].sha
            if selection_.boundary_commits
            else False,
            source_url=f"{REPOSITORY_URL}/blob/{record.sha}/x.py",
        )
        for record in (*selection_.selected_commits, *selection_.boundary_commits)
    )
    branches = (
        BranchObservation(
            name="main",
            ref="refs/heads/main",
            tip_sha=selection_.selected_commits[-1].sha,
            contains_anchor=True,
        ),
    )
    return GitIndex(selection=selection_, snapshots=snapshots, branches=branches)


def work_item(number: int, **overrides) -> GitHubWorkItem:
    data = {
        "number": number,
        "kind": "pull_request",
        "title": f"Work {number}",
        "state": "closed",
        "state_reason": None,
        "body": f"Body {number}",
        "author": "mylonasc",
        "created_at": DAY_ONE,
        "updated_at": DAY_TWO,
        "closed_at": DAY_TWO,
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


def discovery(**overrides) -> GitHubDiscovery:
    data = {
        "repository": "mylonasc/tf_gnns",
        "observed_at": NOW,
        "offline": True,
        "work_items": (),
        "pull_requests": (),
        "provider_labels": (),
        "label_assignments": (),
        "hierarchy_edges": (),
        "absences": (),
        "gaps": (),
        "fetches": (),
        "counters": {},
        "truncated": False,
    }
    data.update(overrides)
    return GitHubDiscovery(**data)


def sink_at(tmp_path, run_id: str = RUN_ID) -> GraphSink:
    return GraphSink.create(
        GraphSettings(path=str(tmp_path / "graph"), backend="lmdb"),
        run_id=run_id,
        overwrite=True,
    )


def topic_assignment(
    subject_id: str,
    topic_id: str,
    topic_name: str,
    *,
    kind: str = "derived",
    action: str = "reuse",
    horizon: datetime = NOW,
) -> TopicAssignment:
    from repo_history.analysis import AnalysisResult, DecisionStatus, Provenance

    derived = kind == "derived"
    return TopicAssignment(
        subject_id=subject_id,
        kind=kind,  # type: ignore[arg-type]
        action=action,  # type: ignore[arg-type]
        topic_id=topic_id if derived else None,
        topic_name=topic_name if derived else None,
        provider_label=None if derived else topic_name,
        confidence=0.9,
        rationale="fixture",
        registry_version=3,
        result=AnalysisResult.create(
            subject_id=subject_id,
            kind="topic",
            status=DecisionStatus.ACCEPTED,
            value={"action": action, "topic_id": topic_id},
            provenance=Provenance(
                method="deterministic-tfidf-labeler",
                method_version="1",
                run_id=RUN_ID,
                info_horizon=horizon,
            ),
            evidence=(),
        ),
    )


# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------


def test_map_git_index_covers_commits_snapshots_files_branches():
    boundary = commit("b" * 40, committed_at=T0)
    tip = commit("c" * 40, parents=(boundary.sha,), committed_at=DAY_ONE)
    mapped = map_git_index(git_index(selection((tip,), (boundary,))), ids(), run_id=RUN_ID)

    assert mapped.skipped == ()
    by_id = {entity.logical_id: entity for entity in mapped.entities}
    assert by_id[ids().commit(tip.sha)].labels == ("Commit",)
    assert by_id[ids().commit(tip.sha)].valid_from == DAY_ONE
    assert by_id[ids().commit(tip.sha)].info_horizon == DAY_ONE
    assert by_id[ids().commit(boundary.sha)].valid_from == T0

    parents = [edge for edge in mapped.relationships if edge.edge_type == PARENT_COMMIT]
    assert len(parents) == 1
    assert (parents[0].source, parents[0].target) == (
        ids().commit(tip.sha), ids().commit(boundary.sha),
    )

    members = [edge for edge in mapped.relationships if edge.edge_type == MEMBER_OF]
    assert {edge.target for edge in members} == {
        ids().commit(tip.sha), ids().commit(boundary.sha),
    }
    assert all(edge.source.startswith("snap:") for edge in members)
    files = [edge for edge in mapped.relationships if edge.edge_type == OF_FILE]
    assert {edge.target for edge in files} == {ids().file("x.py")}
    assert by_id[ids().file("x.py")].labels == ("File",)

    tips = [edge for edge in mapped.relationships if edge.edge_type == TIP_OF]
    assert len(tips) == 1
    assert tips[0].source == ids().branch("main")
    assert tips[0].target == ids().commit(tip.sha)
    assert by_id[ids().branch("main")].properties["tip_sha"] == tip.sha


def test_map_git_index_skips_out_of_scope_parents():
    tip = commit("c" * 40, parents=("d" * 40,), committed_at=DAY_ONE)
    mapped = map_git_index(git_index(selection((tip,))), ids(), run_id=RUN_ID)

    assert not [edge for edge in mapped.relationships if edge.edge_type == PARENT_COMMIT]
    assert len(mapped.skipped) == 1
    assert "out of scope" in mapped.skipped[0]


def test_map_discovery_covers_items_labels_hierarchy_and_merges():
    item = work_item(10, labels=("bug",))
    pull = PullRequestEvidence(
        number=10,
        title="Work 10",
        state="closed",
        merged=True,
        merged_at=DAY_TWO,
        merge_commit_sha="m" * 40,
        head_ref="feature",
        head_sha="h" * 40,
        base_ref="main",
        base_sha="b" * 40,
        url="https://github.com/mylonasc/tf_gnns/pull/10",
    )
    mapped = map_discovery(
        discovery(
            work_items=(item, work_item(11, kind="issue")),
            provider_labels=(ProviderLabel(name="bug", color="ff0000"),),
            label_assignments=(
                LabelAssignment(
                    work_item_number=10,
                    label_name="bug",
                    observed_at=NOW,
                    evidence_id="evidence:1",
                    source_url=item.url,
                ),
            ),
            hierarchy_edges=(
                HierarchyEdge(
                    parent_number=11,
                    child_number=10,
                    relation="sub-issue",
                    in_scope=True,
                    evidence_id="evidence:2",
                ),
                HierarchyEdge(
                    parent_number=10,
                    child_number=99,
                    relation="sub-issue",
                    in_scope=False,
                    evidence_id="evidence:3",
                ),
            ),
            pull_requests=(pull,),
        ),
        ids(),
        run_id=RUN_ID,
    )

    by_id = {entity.logical_id: entity for entity in mapped.entities}
    assert by_id[ids().work_item("github", 10)].properties["state"] == "closed"
    assert by_id[ids().work_item("github", 10)].valid_from == DAY_ONE
    assert by_id[ids().work_item("github", 10)].info_horizon == NOW
    assert by_id[ids().provider_label("github", "bug")].labels == ("ProviderLabel",)

    labels = [edge for edge in mapped.relationships if edge.edge_type == HAS_LABEL]
    assert len(labels) == 1
    assert (labels[0].source, labels[0].target) == (
        ids().work_item("github", 10), ids().provider_label("github", "bug"),
    )
    assert labels[0].valid_from == NOW

    children = [edge for edge in mapped.relationships if edge.edge_type == CHILD_OF]
    assert len(children) == 1
    assert (children[0].source, children[0].target) == (
        ids().work_item("github", 10), ids().work_item("github", 11),
    )

    merges = [edge for edge in mapped.relationships if edge.edge_type == MERGED_AS]
    assert len(merges) == 1
    assert merges[0].target == ids().commit("m" * 40)
    assert merges[0].valid_from == DAY_TWO

    assert len(mapped.skipped) == 1
    assert "out of scope" in mapped.skipped[0]


def test_map_git_index_has_no_direct_commit_file_edges():
    # Commits reach files only through diffs (see map_diffs); the Git
    # selection maps ancestry, snapshots, files, and branches.
    boundary = commit("b" * 40, committed_at=T0)
    tip = commit("c" * 40, parents=(boundary.sha,), committed_at=DAY_ONE)
    mapped = map_git_index(git_index(selection((tip,), (boundary,))), ids(), run_id=RUN_ID)

    kinds = {edge.edge_type for edge in mapped.relationships}
    assert "TOUCHES" not in kinds and "CHANGES" not in kinds
    assert {"PARENT_COMMIT", "MEMBER_OF", "OF_FILE", "TIP_OF"} <= kinds


def test_map_diffs_links_commits_to_files_through_diff_nodes():
    from repo_history.content import CodeDocument, ImportRecord, ParsedContent
    from repo_history.git import TreeEntryRecord

    tip, base = "c" * 40, "b" * 40
    content = ContentIndex(
        contents=(
            ParsedContent(
                content_id="c:x", blob_id="b:x", oid="o" * 40, media_type="text",
                status="parsed", text="hello\n",
            ),
        ),
        snapshots=(
            SnapshotContent(
                snapshot_id="snap:tip", commit_sha=tip, path="x.py",
                oid="o" * 40, content_id="c:x", status="parsed",
                media_type="text", language="text", module=None,
                is_baseline=False, source_url="u",
            ),
        ),
    )
    index = GitIndex(
        selection=selection(
            (commit(tip, parents=(base,), committed_at=DAY_ONE),),
            (commit(base, committed_at=T0),),
        ),
        snapshots=(
            TreeEntryRecord(
                snapshot_id="snap:tip", commit_sha=tip, path="x.py",
                object_id="o" * 40, mode=0o100644, kind="blob", size=10,
                is_binary=False, is_lfs_pointer=False, content_status="inspected",
                source_url="u",
            ),
        ),
        changes=(
            FileChangeRecord(
                change_id="change:1", commit_sha=tip, parent_sha=base,
                comparison="first-parent", path="x.py", status="added",
                source_url="u",
            ),
        ),
    )
    mapped = map_diffs(content, index, ids(), run_id=RUN_ID)

    assert len(mapped.entities) == 1
    (entity,) = [e for e in mapped.entities if e.valid_from == DAY_ONE]
    assert entity.labels == ("Diff",)
    assert entity.properties["status"] == "added"
    assert entity.properties["preview"] == "hello"
    assert entity.valid_from == DAY_ONE

    touches = [edge for edge in mapped.relationships if edge.edge_type == TOUCHES]
    changes = [edge for edge in mapped.relationships if edge.edge_type == CHANGES]
    assert len(touches) == 1 and len(changes) == 1
    assert (touches[0].source, touches[0].target) == (
        ids().commit(tip), entity.logical_id,
    )
    assert (changes[0].source, changes[0].target) == (
        entity.logical_id, ids().file("x.py"),
    )
    assert all(edge.properties.get("hierarchical") is True for edge in mapped.relationships)


def test_map_imports_resolves_absolute_relative_and_package_imports():
    from repo_history.content import CodeDocument, ImportRecord, ParsedContent
    from repo_history.git import TreeEntryRecord
    from repo_history.graph import IMPORTS, map_imports

    def parsed(content_id, *records):
        return ParsedContent(
            content_id=content_id, blob_id=f"blob:{content_id}", oid="o" * 40,
            media_type="python", status="parsed", language="python",
            code=CodeDocument(language="python", imports=tuple(records)),
        )

    def tree(path):
        return TreeEntryRecord(
            snapshot_id=f"snap:{path}", commit_sha="c" * 40, path=path,
            object_id="o" * 40, mode=0o100644, kind="blob", size=10,
            is_binary=False, is_lfs_pointer=False, content_status="inspected",
            source_url=f"{REPOSITORY_URL}/blob/{'c' * 40}/{path}",
        )

    paths = [
        "models/gcn.py", "tests/t.py", "models/__init__.py",
        "models/graphnet.py", "tf_gnns/models/gcn.py", "tf_gnns/__init__.py",
    ]
    content_ids = {
        "models/gcn.py": "c:gcn", "tests/t.py": "c:t", "models/__init__.py": "c:init",
    }
    content = ContentIndex(
        contents=(
            parsed(
                "c:gcn",
                ImportRecord(kind="import", module=None, names=("torch",), line=1),
                ImportRecord(kind="from", module="models", names=("graphnet",), line=2),
                ImportRecord(kind="import", module=None, names=("os",), line=3),
            ),
            parsed(
                "c:t",
                ImportRecord(kind="from", module="tf_gnns.models", names=("gcn",), line=1),
                ImportRecord(kind="from", module="tf_gnns", names=(), line=2),
                ImportRecord(kind="import", module=None, names=("sys",), line=3),
            ),
            parsed(
                "c:init",
                ImportRecord(kind="from", module=None, names=("gcn",), line=1, level=1),
            ),
        ),
        snapshots=tuple(
            SnapshotContent(
                snapshot_id=f"snap:{path}", commit_sha="c" * 40, path=path,
                oid="o" * 40,
                content_id=content_ids.get(path, f"c:{path}"),
                status="parsed", media_type="python", language="python",
                module=None, is_baseline=False, source_url="u",
            )
            for path in paths
        ),
    )
    index = GitIndex(
        selection=selection(
            (commit("c" * 40, committed_at=DAY_ONE),),
            (commit("b" * 40, committed_at=T0),),
        ),
        snapshots=tuple(tree(path) for path in paths),
    )

    mapped = map_imports(content, index, ids(), run_id=RUN_ID)

    targets = {(edge.source, edge.target) for edge in mapped.relationships}
    assert (ids().file("models/gcn.py"), ids().file("models/graphnet.py")) in targets
    assert (ids().file("tests/t.py"), ids().file("tf_gnns/models/gcn.py")) in targets
    assert (ids().file("tests/t.py"), ids().file("tf_gnns/__init__.py")) in targets
    assert (ids().file("models/__init__.py"), ids().file("models/gcn.py")) in targets
    # Third-party and stdlib imports are reported, never invented.
    assert any("imports-external" in gap for gap in mapped.skipped)
    assert all(edge.edge_type == IMPORTS for edge in mapped.relationships)
    # Same import seen twice maps once per commit.
    assert len({edge.edge_id for edge in mapped.relationships}) == len(mapped.relationships)


def test_map_classes_creates_class_entities_and_defined_at_edges():
    from repo_history.content import CodeDocument, CodeSymbol, ImportRecord, ParsedContent
    from repo_history.graph import DEFINED_AT, map_classes

    def parsed(content_id, *symbols):
        return ParsedContent(
            content_id=content_id, blob_id=f"blob:{content_id}", oid="o" * 40,
            media_type="python", status="parsed", language="python",
            code=CodeDocument(language="python", imports=(), symbols=tuple(symbols)),
        )

    def sym(name, qualname, kind="class", line=1):
        return CodeSymbol(
            symbol_id=f"s:{qualname}", kind=kind, name=name, qualname=qualname,
            line=line, column=0, end_line=line + 5,
        )

    snapshots = [
        SnapshotContent(
            snapshot_id="snap:a", commit_sha="c" * 40, path="models/gcn.py",
            oid="o" * 40, content_id="c:a", status="parsed", media_type="python",
            language="python", module="models.gcn", is_baseline=False, source_url="u",
        ),
    ]
    content = ContentIndex(
        contents=(
            parsed("c:a", sym("GraphConv", "models.GraphConv"),
                   sym("helper", "models.helper", kind="function")),
        ),
        snapshots=tuple(snapshots),
    )
    index = GitIndex(
        selection=selection(
            (commit("c" * 40, committed_at=DAY_ONE),),
            (commit("b" * 40, committed_at=T0),),
        ),
        snapshots=(
            TreeEntryRecord(
                snapshot_id="snap:a", commit_sha="c" * 40, path="models/gcn.py",
                object_id="o" * 40, mode=0o100644, kind="blob", size=10,
                is_binary=False, is_lfs_pointer=False, content_status="inspected",
                source_url="u",
            ),
        ),
    )

    mapped = map_classes(content, index, ids(), run_id=RUN_ID)

    by_id = {entity.logical_id: entity for entity in mapped.entities}
    assert len(by_id) == 1  # functions are not nodes
    class_id = next(iter(by_id))
    assert by_id[class_id].labels == ("Class",)
    assert by_id[class_id].properties["qualname"] == "models.GraphConv"
    assert by_id[class_id].properties["line"] == 1
    assert len(mapped.relationships) == 1
    edge = mapped.relationships[0]
    assert edge.edge_type == DEFINED_AT
    assert edge.source == class_id
    assert edge.target == ids().file("models/gcn.py")
    assert edge.valid_from == DAY_ONE


def stable_class_id(path, qualname):
    from repo_history.ids import stable_id

    return stable_id("class", ids().repository_id, path, qualname)


def test_map_registry_and_assignments_cover_topics_and_about_edges():
    registry = seed_registry(at=NOW)
    registry, first = registry.register(
        "sparse-training", at=NOW, created_by="test", ids=ids(),
        description="sparse training graphs",
    )
    registry, second = registry.register(
        "graph-convolution", at=NOW, created_by="test", ids=ids()
    )
    merged_registry, merged = registry.merge(
        [first.topic_id, second.topic_id], "graph-learning",
        at=NOW, created_by="test", ids=ids(),
    )

    mapped = map_registry(merged_registry, ids(), run_id=RUN_ID)

    by_id = {entity.logical_id: entity for entity in mapped.entities}
    assert set(by_id) == {first.topic_id, second.topic_id, merged.topic_id}
    assert all(entity.labels == ("Topic",) for entity in by_id.values())
    assert by_id[first.topic_id].properties["merged_into"] == merged.topic_id
    assert by_id[merged.topic_id].properties["description"].startswith("sparse")
    assert by_id[merged.topic_id].properties["supersedes"] == sorted(
        [first.topic_id, second.topic_id]
    )

    assignments = [
        topic_assignment("snap:guide", first.topic_id, "sparse-training"),
        topic_assignment("workitem:github:10", merged.topic_id, "graph-learning"),
        topic_assignment(
        "workitem:github:10", "bug", "bug",
        kind="provider", action="preserve",
    ),
    ]
    linked = map_topic_assignments(assignments, ids(), run_id=RUN_ID)

    assert len(linked.relationships) == 2
    assert {edge.edge_type for edge in linked.relationships} == {ABOUT}
    by_source = {edge.source: edge for edge in linked.relationships}
    assert by_source["snap:guide"].target == first.topic_id
    assert by_source[ids().work_item("github", 10)].target == merged.topic_id
    assert all(edge.run_id == RUN_ID for edge in linked.relationships)
    assert "no derived topic" in linked.skipped[0]


def test_map_folders_creates_hierarchy_with_repo_root():
    from repo_history.graph import BELONGS_TO, HAS_BRANCH, map_folders, map_repository_branches

    mapped = map_folders(
        ["models/gcn.py", "models/graphnet.py", "docs/guide.md", "README.md"],
        ids(), run_id=RUN_ID, window_start=WINDOW.start,
    )
    by_id = {entity.logical_id: entity for entity in mapped.entities}
    folders = {nid for nid, entity in by_id.items() if entity.labels == ("Folder",)}
    assert len(folders) == 2
    assert by_id[ids().repository_id].labels == ("Repository",)
    folders_by_path = {
        entity.properties["path"]: nid
        for nid, entity in by_id.items()
        if entity.labels == ("Folder",)
    }
    assert set(folders_by_path) == {"models", "docs"}
    belongs = [edge for edge in mapped.relationships if edge.edge_type == BELONGS_TO]
    # Files link only to their immediate parent; folders chain upward.
    assert {(edge.source, edge.target) for edge in belongs} == {
        (ids().file("models/gcn.py"), folders_by_path["models"]),
        (ids().file("models/graphnet.py"), folders_by_path["models"]),
        (ids().file("docs/guide.md"), folders_by_path["docs"]),
        (ids().file("README.md"), ids().repository_id),
    }
    assert all(edge.properties.get("hierarchical") is True for edge in belongs)
    has_folders = [edge for edge in mapped.relationships if edge.edge_type == "HAS_FOLDER"]
    assert {(edge.source, edge.target) for edge in has_folders} == {
        (ids().repository_id, folders_by_path["models"]),
        (ids().repository_id, folders_by_path["docs"]),
    }

    nested = map_folders(["a/b/c.py"], ids(), run_id=RUN_ID, window_start=WINDOW.start)
    nested_folders = {
        entity.properties["path"]: nid
        for nid, entity in {
            entity.logical_id: entity for entity in nested.entities if entity.labels == ("Folder",)
        }.items()
    }
    assert set(nested_folders) == {"a", "a/b"}
    nested_belongs = {
        (edge.source, edge.target)
        for edge in nested.relationships
        if edge.edge_type == "BELONGS_TO"
    }
    assert nested_belongs == {
        (ids().file("a/b/c.py"), nested_folders["a/b"]),
        (nested_folders["a/b"], nested_folders["a"]),
    }
    assert nested.entities[0].labels == ("Repository",)

    branches = map_repository_branches(["main", "feat"], ids(), run_id=RUN_ID, window_start=WINDOW.start)
    assert all(edge.edge_type == HAS_BRANCH for edge in branches.relationships)
    assert {(edge.source, edge.target) for edge in branches.relationships} == {
        (ids().repository_id, ids().branch("feat")),
        (ids().repository_id, ids().branch("main")),
    }
    assert all(edge.properties.get("hierarchical") is True for edge in branches.relationships)


def test_about_and_diff_edges_persist_and_count(tmp_path):
    from repo_history.content import CodeDocument, ImportRecord, ParsedContent

    boundary = commit("b" * 40, committed_at=T0)
    tip = commit("c" * 40, parents=(boundary.sha,), committed_at=DAY_ONE)
    index = git_index(selection((tip,), (boundary,)))
    mapped = map_git_index(index, ids(), run_id=RUN_ID)
    registry = seed_registry(at=NOW)
    registry, topic = registry.register(
        "sparse-training", at=NOW, created_by="test", ids=ids()
    )
    topics = map_registry(registry, ids(), run_id=RUN_ID)
    about = map_topic_assignments(
        [topic_assignment("snap:guide", topic.topic_id, "sparse-training")],
        ids(), run_id=RUN_ID,
    )
    content = ContentIndex(
        contents=(
            ParsedContent(
                content_id="c:x", blob_id="b:x", oid="o" * 40, media_type="python",
                status="parsed", language="python", text="import os\n",
                code=CodeDocument(
                    language="python",
                    imports=(ImportRecord(kind="import", module=None, names=("os",), line=1),),
                ),
            ),
        ),
        snapshots=(
            SnapshotContent(
                snapshot_id="snap:tip", commit_sha=tip.sha, path="x.py",
                oid="o" * 40, content_id="c:x", status="parsed",
                media_type="python", language="python", module=None,
                is_baseline=False, source_url="u",
            ),
        ),
    )
    diffs = map_diffs(content, index, ids(), run_id=RUN_ID)

    sink = sink_at(tmp_path)
    sink.write_entities(
        [*mapped.entities, *topics.entities, *diffs.entities], analysis_run_id=RUN_ID
    )
    sink.write_relationships(
        [*mapped.relationships, *about.relationships, *diffs.relationships],
        analysis_run_id=RUN_ID,
    )

    assert sink.label_counts()["Topic"] == 1
    assert sink.label_counts()["Diff"] == 2
    assert sink.edge_counts()["TOUCHES"] == 2
    assert sink.edge_counts()["CHANGES"] == 2
    assert sink.edge_counts()["ABOUT"] == 1
    found = sink.find_nodes("Topic", "name", "sparse-training")
    assert len(found) == 1
    assert found[0]["prov_run_id"] == RUN_ID
    sink.close()


# ---------------------------------------------------------------------------
# Sink writes and reads
# ---------------------------------------------------------------------------


def test_file_entities_carry_language_and_edges_flag_hierarchy(tmp_path):
    boundary = commit("b" * 40, committed_at=T0)
    tip = commit("c" * 40, parents=(boundary.sha,), committed_at=DAY_ONE)
    mapped = map_git_index(git_index(selection((tip,), (boundary,))), ids(), run_id=RUN_ID)

    files = [entity for entity in mapped.entities if entity.labels == ("File",)]
    assert files and all(entity.properties["path"] == "x.py" for entity in files)
    assert all(entity.properties["language"] == "python" for entity in files)
    assert all(entity.properties["media_type"] == "python" for entity in files)

    hierarchical = {
        edge.edge_type
        for edge in mapped.relationships
        if edge.properties.get("hierarchical") is True
    }
    assert {"PARENT_COMMIT", "MEMBER_OF", "OF_FILE", "TIP_OF"} <= hierarchical
    assert "TOUCHES" not in {edge.edge_type for edge in mapped.relationships}


def test_sink_round_trip_with_provenance(tmp_path):
    boundary = commit("b" * 40, committed_at=T0)
    tip = commit("c" * 40, parents=(boundary.sha,), committed_at=DAY_ONE)
    mapped = map_git_index(git_index(selection((tip,), (boundary,))), ids(), run_id=RUN_ID)

    sink = sink_at(tmp_path)
    entities = sink.write_entities(mapped.entities, analysis_run_id=RUN_ID)
    relationships = sink.write_relationships(mapped.relationships, analysis_run_id=RUN_ID)

    assert entities.entities_written == len(mapped.entities)
    assert relationships.relationships_written == len(mapped.relationships)

    stored = sink.node_as_of(ids().commit(tip.sha), DAY_TWO)
    assert stored is not None
    assert stored["sha"] == tip.sha
    assert stored["prov_run_id"] == RUN_ID
    assert stored["prov_origin"] == "git"
    assert stored["prov_info_horizon"] == DAY_ONE.isoformat()
    assert stored["prov_evidence_ids"] == [
        ids().evidence("git", tip.sha, f"{REPOSITORY_URL}/commit/{tip.sha}")
    ]

    assert sink.node_as_of(ids().commit(tip.sha), T0) is None
    assert sink.current_node(ids().commit(tip.sha))["sha"] == tip.sha
    assert sink.find_nodes("Commit", "sha", tip.sha)[0]["sha"] == tip.sha
    assert sink.label_counts()["Commit"] == 2
    assert sink.label_counts()["Snapshot"] == 2
    assert sink.edge_counts()[PARENT_COMMIT] == 1
    edge_id = next(
        edge.edge_id for edge in mapped.relationships if edge.edge_type == PARENT_COMMIT
    )
    assert sink.edge_as_of(edge_id, DAY_TWO)["type"] == PARENT_COMMIT
    sink.close()


def test_duplicate_ingestion_is_idempotent(tmp_path):
    tip = commit("c" * 40, committed_at=DAY_ONE)
    mapped = map_git_index(git_index(selection((tip,))), ids(), run_id=RUN_ID)

    sink = sink_at(tmp_path)
    first = sink.write_entities(mapped.entities, analysis_run_id=RUN_ID)
    sink.write_relationships(mapped.relationships, analysis_run_id=RUN_ID)
    versions_before = len(sink.node_history(ids().commit(tip.sha)))

    second = sink.write_entities(mapped.entities, analysis_run_id=RUN_ID)
    relationships = sink.write_relationships(mapped.relationships, analysis_run_id=RUN_ID)

    assert (first.entities_written, first.entities_duplicate) == (
        len(mapped.entities), 0,
    )
    assert second.entities_written == 0
    assert second.entities_duplicate == len(mapped.entities)
    assert relationships.relationships_duplicate == len(mapped.relationships)
    assert len(sink.node_history(ids().commit(tip.sha))) == versions_before
    assert sink.current_node(ids().commit(tip.sha))["sha"] == tip.sha
    sink.close()


def test_sink_rejects_writes_for_another_run(tmp_path):
    tip = commit("c" * 40, committed_at=DAY_ONE)
    mapped = map_git_index(git_index(selection((tip,))), ids(), run_id=RUN_ID)

    sink = sink_at(tmp_path)
    with pytest.raises(GraphError):
        sink.write_entities(mapped.entities, analysis_run_id="run-2")
    with pytest.raises(GraphError):
        sink.write_relationships(mapped.relationships, analysis_run_id="run-2")
    sink.close()


def test_correction_retains_old_evidence(tmp_path):
    write = EntityWrite(
        logical_id=ids().work_item("github", 10),
        labels=("WorkItem",),
        properties={"number": 10, "state": "open"},
        valid_from=DAY_ONE,
        origin="github",
        origin_id="10",
        event_id="event-open",
        run_id=RUN_ID,
        info_horizon=DAY_ONE,
        evidence_ids=(),
    )
    sink = sink_at(tmp_path)
    sink.write_entities([write], analysis_run_id=RUN_ID)

    closed = write.model_copy(
        update={
            "properties": {"number": 10, "state": "closed"},
            "event_id": "event-closed",
            "valid_from": DAY_TWO,
            "info_horizon": DAY_TWO,
        }
    )
    history = sink.node_history(write.logical_id)
    assert len(history) == 1
    sink.correct_entity(
        closed, supersedes_version_id=history[0].version_id, valid_from=DAY_TWO
    )

    assert sink.node_as_of(write.logical_id, DAY_ONE)["state"] == "open"
    assert sink.node_as_of(write.logical_id, DAY_TWO)["state"] == "closed"
    corrected = sink.node_history(write.logical_id)
    assert len(corrected) == 2
    assert corrected[1].supersedes_version_id == history[0].version_id
    assert corrected[1].properties["state"] == "closed"
    assert corrected[0].properties["state"] == "open"
    assert sink.current_node(write.logical_id)["state"] == "closed"
    sink.close()


def test_deferred_indexes_rebuild_before_indexed_reads(tmp_path):
    tip = commit("c" * 40, committed_at=DAY_ONE)
    mapped = map_git_index(git_index(selection((tip,))), ids(), run_id=RUN_ID)

    sink = sink_at(tmp_path)
    sink.write_entities(mapped.entities, analysis_run_id=RUN_ID, defer_indexes=True)
    assert sink.stale_indexes() != ()

    assert sink.find_nodes("Commit", "sha", tip.sha)[0]["sha"] == tip.sha
    assert sink.stale_indexes() == ()
    sink.close()


def test_temporal_reads_rebuild_stale_indexes_first(tmp_path):
    tip = commit("c" * 40, committed_at=DAY_ONE)
    mapped = map_git_index(git_index(selection((tip,))), ids(), run_id=RUN_ID)

    sink = sink_at(tmp_path)
    sink.write_entities(mapped.entities, analysis_run_id=RUN_ID, defer_indexes=True)
    sink.write_relationships(mapped.relationships, analysis_run_id=RUN_ID, defer_indexes=True)
    assert sink.stale_indexes() != ()

    assert sink.node_as_of(ids().commit(tip.sha), DAY_TWO)["sha"] == tip.sha
    assert len(sink.node_history(ids().commit(tip.sha))) == 1
    assert sink.stale_indexes() == ()
    sink.close()


def test_persisted_graph_reopens_correctly(tmp_path):
    tip = commit("c" * 40, committed_at=DAY_ONE)
    mapped = map_git_index(git_index(selection((tip,))), ids(), run_id=RUN_ID)
    settings = GraphSettings(path=str(tmp_path / "graph"), backend="lmdb")

    sink = GraphSink.create(settings, run_id=RUN_ID, overwrite=True)
    sink.write_entities(mapped.entities, analysis_run_id=RUN_ID)
    sink.write_relationships(mapped.relationships, analysis_run_id=RUN_ID)
    sink.close()

    reopened = GraphSink.open(settings, run_id="run-2-resume")
    assert reopened.node_as_of(ids().commit(tip.sha), DAY_TWO)["sha"] == tip.sha
    assert reopened.current_node(ids().file("x.py"))["path"] == "x.py"
    assert reopened.find_nodes("Commit", "sha", tip.sha)
    reopened.close()


def test_current_projection_is_rebuildable_from_the_same_batch(tmp_path):
    boundary = commit("b" * 40, committed_at=T0)
    tip = commit("c" * 40, parents=(boundary.sha,), committed_at=DAY_ONE)
    mapped = map_git_index(git_index(selection((tip,), (boundary,))), ids(), run_id=RUN_ID)

    projections = []
    for name in ("first", "second"):
        sink = GraphSink.create(
            GraphSettings(path=str(tmp_path / name), backend="lmdb"),
            run_id=RUN_ID,
            overwrite=True,
        )
        sink.write_entities(mapped.entities, analysis_run_id=RUN_ID)
        sink.write_relationships(mapped.relationships, analysis_run_id=RUN_ID)
        projections.append(
            {
                entity.logical_id: sink.current_node(entity.logical_id)
                for entity in mapped.entities
            }
        )
        sink.close()

    assert projections[0] == projections[1]
    assert all(current is not None for current in projections[0].values())


def test_branch_reachability_through_parent_commits(tmp_path):
    root = commit("a" * 40, committed_at=T0)
    middle = commit("b" * 40, parents=(root.sha,), committed_at=DAY_ONE)
    tip = commit("c" * 40, parents=(middle.sha,), committed_at=DAY_TWO)
    mapped = map_git_index(git_index(selection((middle, tip), (root,))), ids(), run_id=RUN_ID)

    sink = sink_at(tmp_path)
    sink.write_entities(mapped.entities, analysis_run_id=RUN_ID)
    sink.write_relationships(mapped.relationships, analysis_run_id=RUN_ID)

    reachable = {ids().commit(tip.sha)}
    frontier = [ids().commit(tip.sha)]
    while frontier:
        current = frontier.pop()
        for parent in sink.neighbors(current, PARENT_COMMIT, direction="out"):
            if parent not in reachable:
                reachable.add(parent)
                frontier.append(parent)

    assert reachable == {
        ids().commit(tip.sha), ids().commit(middle.sha), ids().commit(root.sha),
    }
    branch_id = ids().branch("main")
    assert sink.neighbors(branch_id, TIP_OF, direction="out") == [ids().commit(tip.sha)]
    sink.close()


def test_closed_sink_refuses_operations_but_closes_safely(tmp_path):
    tip = commit("c" * 40, committed_at=DAY_ONE)
    mapped = map_git_index(git_index(selection((tip,))), ids(), run_id=RUN_ID)

    sink = sink_at(tmp_path)
    sink.close()
    sink.close()
    assert sink.is_closed

    with pytest.raises(GraphClosedError):
        sink.write_entities(mapped.entities, analysis_run_id=RUN_ID)
    with pytest.raises(GraphClosedError):
        sink.write_relationships(mapped.relationships, analysis_run_id=RUN_ID)
    with pytest.raises(GraphClosedError):
        sink.node_as_of(ids().commit(tip.sha), DAY_TWO)
    with pytest.raises(GraphClosedError):
        sink.find_nodes("Commit", "sha", tip.sha)

    with GraphSink.create(
        GraphSettings(path=str(tmp_path / "other"), backend="lmdb"),
        run_id=RUN_ID,
        overwrite=True,
    ) as managed:
        assert not managed.is_closed
    assert managed.is_closed


def test_sink_satisfies_the_graph_sink_protocol(tmp_path):
    assert isinstance(sink_at(tmp_path), GraphSinkProtocol)
    sink_at(tmp_path).close()


def test_graph_settings_have_bounded_defaults_and_load_from_toml():
    settings = GraphSettings()
    assert settings.path == "data/graph"
    assert settings.backend == "lmdb"
    assert settings.overwrite is False
    assert settings.map_size_bytes == 256 * 1024 * 1024
    assert settings.backend_options() == {"map_size": 256 * 1024 * 1024}
    assert GraphSettings(backend="pyrex").backend_options() == {}
    with pytest.raises(Exception):
        GraphSettings(path="  ")
    with pytest.raises(Exception):
        GraphSettings(backend="sqlite")
    with pytest.raises(Exception):
        GraphSettings(map_size_bytes=1024)

    from repo_history.config import load_config

    config = load_config()
    assert config.graph.path == "data/graph"
    assert config.graph.backend == "lmdb"
    assert config.graph.overwrite is False
    assert config.graph.map_size_bytes == 256 * 1024 * 1024


def test_unmapped_evidence_stays_out_of_the_graph(tmp_path):
    mapped = map_discovery(
        discovery(
            hierarchy_edges=(
                HierarchyEdge(
                    parent_number=10,
                    child_number=99,
                    relation="sub-issue",
                    in_scope=False,
                    evidence_id="evidence:9",
                ),
            ),
        ),
        ids(),
        run_id=RUN_ID,
    )

    assert mapped.entities == ()
    assert mapped.relationships == ()
    assert len(mapped.skipped) == 1

    sink = sink_at(tmp_path)
    report = sink.write_relationships(mapped.relationships, analysis_run_id=RUN_ID)
    assert report.relationships_written == 0
    assert sink.edge_counts()[CHILD_OF] == 0
    sink.close()
