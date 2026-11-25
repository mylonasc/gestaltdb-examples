"""D3 viewer data export and page rendering over a synthetic slice."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from repo_history.analysis import AnalysisResult, DecisionStatus, Provenance
from repo_history.content import ContentIndex, SnapshotContent
from repo_history.git import BranchObservation, CommitRecord, GitIndex, GitSelection
from repo_history.github import GitHubDiscovery, GitHubWorkItem, RevisionAvailability
from repo_history.ids import IdFactory
from repo_history.labeling import TopicAssignment
from repo_history.linking import LinkRecord, LinkRelation
from repo_history.models import EvidenceRef, TimeWindow
from repo_history.viz_d3 import build_viewer_data, render_viewer_html

UTC = timezone.utc
NOW = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)
EARLY = datetime(2026, 9, 14, 9, 0, tzinfo=UTC)
DAY_ONE = datetime(2026, 9, 14, 10, 0, tzinfo=UTC)
DAY_TWO = datetime(2026, 9, 14, 11, 0, tzinfo=UTC)
REPOSITORY_URL = "https://github.com/mylonasc/tf_gnns"
RUN_ID = "run-1"
WINDOW = TimeWindow(
    start=datetime(2026, 9, 13, tzinfo=UTC),
    end=datetime(2026, 9, 15, tzinfo=UTC),
)
ROOT, MID, TIP = "r" * 40, "m" * 40, "t" * 40


def ids() -> IdFactory:
    return IdFactory(REPOSITORY_URL)


def commit(sha: str, parents=(), at=DAY_ONE) -> CommitRecord:
    return CommitRecord(
        sha=sha, parents=tuple(parents), tree_sha="t" * 40,
        message=f"commit {sha[:4]}", author_name="A", author_email="a@t",
        authored_at=at, committer_name="C", committer_email="c@t",
        committed_at=at, source_url=f"{REPOSITORY_URL}/commit/{sha}",
        is_merge=len(parents) > 1,
    )


def guide_snapshot(commit_sha: str = TIP, path: str = "docs/guide.md",
                     snapshot_id: str = "snap:guide") -> SnapshotContent:
    return SnapshotContent(
        snapshot_id=snapshot_id, commit_sha=commit_sha, path=path,
        oid="o" * 40, content_id=f"c:{snapshot_id}", status="parsed",
        media_type="markdown", language="markdown", module=None,
        is_baseline=False, source_url=f"{REPOSITORY_URL}/blob/{commit_sha}/x",
    )


def fixture_index() -> GitIndex:
    selection = GitSelection(
        anchor=TIP, window=WINDOW,
        selected_commits=(
            commit(MID, parents=(ROOT,), at=DAY_ONE),
            commit(TIP, parents=(MID,), at=DAY_TWO),
        ),
        boundary_commits=(commit(ROOT, at=EARLY),),
    )
    return GitIndex(
        selection=selection,
        branches=(
            BranchObservation(
                name="main", ref="refs/heads/main", tip_sha=TIP,
                contains_anchor=True,
            ),
            BranchObservation(
                name="gone", ref="refs/heads/gone", tip_sha="z" * 40,
                contains_anchor=False,
            ),
        ),
    )


def fixture_discovery() -> GitHubDiscovery:
    return GitHubDiscovery(
        repository="mylonasc/tf_gnns", observed_at=NOW, offline=True,
        work_items=(
            GitHubWorkItem(
                number=10, kind="issue", title="Sparse bug", state="open",
                state_reason=None, body="", author="mylonasc",
                created_at=DAY_ONE, updated_at=DAY_ONE, closed_at=None,
                url="https://github.com/mylonasc/tf_gnns/issues/10",
                labels=(), comments=(), timeline=(),
                revisions=RevisionAvailability(work_item_number=10),
                observed_at=NOW, source_hash="0" * 64,
            ),
        ),
        pull_requests=(), provider_labels=(), label_assignments=(),
        hierarchy_edges=(), absences=(), gaps=(), fetches=(),
        counters={}, truncated=False,
    )


def assignment(subject_id: str, topic_id: str, name: str) -> TopicAssignment:
    return TopicAssignment(
        subject_id=subject_id, kind="derived", action="reuse",
        topic_id=topic_id, topic_name=name, confidence=0.9,
        rationale="fixture", registry_version=1,
        result=AnalysisResult.create(
            subject_id=subject_id, kind="topic", status=DecisionStatus.ACCEPTED,
            value={"action": "reuse", "topic_id": topic_id},
            provenance=Provenance(
                method="deterministic-tfidf-labeler", method_version="1",
                run_id=RUN_ID, info_horizon=NOW,
            ),
            evidence=(
                EvidenceRef(evidence_id="ev:1", source="labeling", source_id=subject_id),
            ),
        ),
    )


def link(
    source: str, target: str, relation: LinkRelation, method: str = "link-resolver"
) -> LinkRecord:
    return LinkRecord(
        link_id=f"link:{source}:{relation.value}:{target}",
        source_id=source, target_id=target, relation=relation,
        confidence=0.8, rationale="fixture",
        evidence=(EvidenceRef(evidence_id="ev:1", source="t", source_id=source),),
        method=method, run_id=RUN_ID, info_horizon=NOW,
    )


def fixture_inputs(**overrides):
    content = ContentIndex(snapshots=(guide_snapshot(),))
    arguments = {
        "index": fixture_index(),
        "discovery": fixture_discovery(),
        "content": content,
        "links": [
            link(ids().commit(MID), ids().commit(ROOT), LinkRelation.MENTIONS),
            link(ids().commit(TIP), ids().commit(MID), LinkRelation.MENTIONS),
            link(ids().work_item("github", 10), ids().file("docs/guide.md"),
                 LinkRelation.RELATED_TO, method="link-inference"),
        ],
        "assignments": [
            assignment("snap:guide", ids().topic("sparse"), "sparse-training"),
        ],
        "info_horizon": NOW,
        "run_id": RUN_ID,
        "repository_url": REPOSITORY_URL,
    }
    arguments.update(overrides)
    return arguments


def test_times_normalize_min_to_max():
    data = build_viewer_data(**fixture_inputs())
    by_id = {node["id"]: node for node in data["nodes"]}

    assert by_id[ids().commit(ROOT)]["t"] == 0.0
    assert by_id[ids().commit(TIP)]["t"] == 1.0
    assert by_id[ids().commit(MID)]["t"] == 0.5
    assert 0.0 <= by_id[ids().work_item("github", 10)]["t"] <= 1.0
    # The file and topic inherit source times, inside the range.
    assert by_id[ids().file("docs/guide.md")]["t"] == 1.0
    assert by_id[ids().topic("sparse")]["t"] == 1.0


def test_timeless_nodes_center_without_pull():
    data = build_viewer_data(**fixture_inputs())
    by_id = {node["id"]: node for node in data["nodes"]}

    assert by_id[ids().branch("gone")]["time"] is None
    assert by_id[ids().branch("gone")]["t"] is None


def test_single_timestamp_maps_everything_to_mid():
    index = fixture_index()
    only = GitSelection(
        anchor=MID, window=WINDOW,
        selected_commits=(commit(MID, parents=(), at=DAY_ONE),),
        boundary_commits=(),
    )
    only_index = GitIndex(selection=only, snapshots=(), branches=())
    data = build_viewer_data(
        **{**fixture_inputs(), "index": only_index, "links": [], "assignments": []}
    )

    assert {node["t"] for node in data["nodes"]} == {0.5}


def test_node_budget_keeps_core_and_reports_omissions():
    extra_paths = [f"docs/f{i}.py" for i in range(12)]
    arguments = fixture_inputs()
    arguments["content"] = ContentIndex(
        snapshots=tuple(
            [guide_snapshot()]
            + [
                guide_snapshot(path=path, snapshot_id=f"snap:{i}")
                for i, path in enumerate(extra_paths)
            ]
        )
    )
    commit_tip = ids().commit(TIP)
    arguments["links"] = [
        *arguments["links"],
        *[link(commit_tip, ids().file(path), LinkRelation.MENTIONS) for path in extra_paths],
    ]
    data = build_viewer_data(**arguments, max_nodes=10)

    kinds = {node["kind"] for node in data["nodes"]}
    assert {"Commit", "WorkItem", "Branch"} <= kinds
    assert len(data["nodes"]) <= 10
    assert data["meta"]["counts"]["omitted"] > 0
    node_ids = {node["id"] for node in data["nodes"]}
    assert all(
        edge["source"] in node_ids and edge["target"] in node_ids
        for edge in data["edges"]
    )


def test_edges_only_join_kept_nodes_and_flag_inference():
    data = build_viewer_data(**fixture_inputs(max_nodes=200))
    node_ids = {node["id"] for node in data["nodes"]}

    assert all(
        edge["source"] in node_ids and edge["target"] in node_ids
        for edge in data["edges"]
    )
    inferred = [edge for edge in data["edges"] if edge["inferred"]]
    assert len(inferred) == 1
    assert inferred[0]["relation"] == "RELATED_TO"


def test_snapshot_links_aggregate_to_files():
    target = ids().file("docs/guide.md")
    data = build_viewer_data(**fixture_inputs(
        links=[link(ids().work_item("github", 10), "snap:guide", LinkRelation.MENTIONS)],
    ))
    node_ids = {node["id"] for node in data["nodes"]}

    assert target in node_ids
    assert "snap:guide" not in node_ids
    assert any(
        edge["source"] == ids().work_item("github", 10) and edge["target"] == target
        for edge in data["edges"]
    )


def test_unobserved_branches_become_placeholders():
    missing = ids().branch("version_bump")
    data = build_viewer_data(**fixture_inputs(
        links=[link(ids().work_item("github", 10), missing, LinkRelation.RELATED_TO)],
        branch_names={missing: "version_bump"},
    ))
    by_id = {node["id"]: node for node in data["nodes"]}

    assert by_id[missing]["kind"] == "Branch"
    assert by_id[missing]["t"] is None
    assert by_id[missing]["props"]["source"] == "unobserved"
    assert any(edge["target"] == missing for edge in data["edges"])

    dropped = build_viewer_data(**fixture_inputs(
        links=[link(ids().work_item("github", 10), missing, LinkRelation.RELATED_TO)],
    ))
    assert missing not in {node["id"] for node in dropped["nodes"]}
    assert all(edge["target"] != missing for edge in dropped["edges"])


def test_export_is_deterministic():
    assert build_viewer_data(**fixture_inputs()) == build_viewer_data(**fixture_inputs())


def test_page_embeds_data_controls_and_temporal_forces():
    html = render_viewer_html(build_viewer_data(**fixture_inputs()), title="Demo")

    assert "https://cdn.jsdelivr.net/npm/d3@7" in html
    for control in ("tmin", "tmax", "anchor", "spread", "ldist", "charge",
                    "labels", "bytime", "search", "find", "pause", "reset",
                    "details", "legend", "kindFilters", "edgeFilters"):
        assert f'id="{control}"' in html
    for hook in ("temporalForce", "forceX", "forceY", "forceLink", "forceCollide"):
        assert hook in html
    assert "fallback" in html


def test_embedded_payload_round_trips_and_escapes_script_close():
    data = build_viewer_data(**fixture_inputs())
    data["nodes"][0]["label"] = 'x</script><script>alert(1)'
    html = render_viewer_html(data)

    assert "</script><script>" not in html.replace("<\\/script>", "")
    match = re.search(r"const DATA = (.*?);\n\(function", html, re.DOTALL)
    assert match is not None
    assert json.loads(match.group(1))["nodes"][0]["label"] == 'x</script><script>alert(1)'


def test_empty_inputs_render_an_empty_page():
    html = render_viewer_html(
        build_viewer_data(
            index=GitIndex(
                selection=GitSelection(anchor="a" * 40, window=WINDOW),
            ),
            discovery=GitHubDiscovery(
                repository="mylonasc/tf_gnns", observed_at=NOW, offline=True,
            ),
            content=ContentIndex(),
            links=[], assignments=[], info_horizon=NOW, run_id=RUN_ID,
            repository_url=REPOSITORY_URL,
        )
    )

    match = re.search(r"const DATA = (.*?);\n\(function", html, re.DOTALL)
    assert match is not None
    assert json.loads(match.group(1))["nodes"] == []
