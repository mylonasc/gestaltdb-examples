"""Single-writer GestaltDB temporal graph persistence.

Mapping tests are pure (no database); sink tests use throwaway LMDB graphs
under ``tmp_path``.  No network, no Git CLI, no model credentials.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

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
    HierarchyEdge,
    LabelAssignment,
    ProviderLabel,
    PullRequestEvidence,
    RevisionAvailability,
)
from repo_history.graph import (
    CHILD_OF,
    HAS_LABEL,
    MEMBER_OF,
    MERGED_AS,
    OF_FILE,
    PARENT_COMMIT,
    TIP_OF,
    EntityWrite,
    GraphClosedError,
    GraphError,
    GraphSink,
    map_discovery,
    map_git_index,
)
from repo_history.ids import IdFactory
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


# ---------------------------------------------------------------------------
# Sink writes and reads
# ---------------------------------------------------------------------------


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
