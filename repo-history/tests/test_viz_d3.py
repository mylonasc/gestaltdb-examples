"""D3 viewer data export and page rendering over a synthetic slice."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import pytest

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
    from repo_history.content import ParsedContent

    content = ContentIndex(
        contents=(
            ParsedContent(
                content_id="c:snap:guide", blob_id="blob:guide", oid="o" * 40,
                media_type="markdown", status="parsed", language="markdown",
                text="# Guide\n\nSee the training guide.\n",
            ),
        ),
        snapshots=(guide_snapshot(),),
    )
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


def test_times_normalize_to_the_demo_window_and_clamp_outliers():
    data = build_viewer_data(**fixture_inputs())
    by_id = {node["id"]: node for node in data["nodes"]}

    # WINDOW spans 2026-09-13T00:00Z..2026-09-15T00:00Z (48h).
    assert by_id[ids().commit(ROOT)]["t"] == 33 / 48
    assert by_id[ids().commit(MID)]["t"] == 34 / 48
    assert by_id[ids().commit(TIP)]["t"] == 35 / 48
    assert all(0.0 <= node["t"] <= 1.0 for node in data["nodes"] if node["t"] is not None)

    # Out-of-window timestamps clamp to the edges instead of squeezing the slice.
    early = build_viewer_data(
        **fixture_inputs(assignments=[assignment("s", ids().topic("early"), "Early")])
    )
    assert all(
        0.0 <= node["t"] <= 1.0 for node in early["nodes"] if node["t"] is not None
    )
    assert by_id[ids().work_item("github", 10)]["t"] == 34 / 48
    # The file and topic inherit source times, inside the range.
    assert by_id[ids().file("docs/guide.md")]["t"] == 35 / 48
    assert by_id[ids().topic("sparse")]["t"] == 35 / 48


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


def test_class_nodes_show_class_names_not_file_paths():
    from repo_history.graph import EntityWrite

    arguments = fixture_inputs()
    arguments["extra_nodes"] = [
        EntityWrite(
            logical_id="class:abc", labels=("Class",),
            properties={"name": "GraphConv", "qualname": "models.GraphConv",
                        "path": "models/gcn.py", "line": 3},
            valid_from=DAY_ONE, origin="git", origin_id="models/gcn.py:GraphConv",
            event_id="e1", run_id=RUN_ID, info_horizon=NOW, evidence_ids=(),
        ),
    ]
    data = build_viewer_data(**arguments)
    by_id = {node["id"]: node for node in data["nodes"]}

    assert by_id["class:abc"]["label"] == "GraphConv"
    assert by_id["class:abc"]["title"] == "models.GraphConv"
    assert by_id["class:abc"]["t"] is not None


def test_hashed_subject_assignments_link_to_commit_nodes():
    from repo_history.graph import map_topic_assignments
    from repo_history.labeling import TopicAssignment

    hashed = ids().commit(TIP)
    base = fixture_inputs()["assignments"][0]
    assignment = TopicAssignment(
        subject_id=hashed, kind="derived", action="reuse",
        topic_id=base.topic_id, topic_name=base.topic_name, confidence=0.9,
        rationale="hashed", registry_version=1, result=base.result,
    )
    about = map_topic_assignments([assignment], ids(), run_id=RUN_ID)
    assert about.relationships[0].source == hashed

    data = build_viewer_data(**fixture_inputs(extra_edges=list(about.relationships)))
    assert any(
        edge["source"] == hashed and edge["relation"] == "ABOUT"
        for edge in data["edges"]
    )


def test_extra_structural_nodes_join_with_folder_times():
    from repo_history.graph import map_folders

    mapped = map_folders(
        ["docs/guide.md", "docs/other.md"],
        ids(), run_id=RUN_ID, window_start=WINDOW.start,
    )
    kinds = {entity.logical_id: entity.labels for entity in mapped.entities}
    assert kinds[ids().repository_id] == ("Repository",)
    assert set(kinds) == {ids().repository_id, stable_folder_id("docs")}
    targets = {(edge.source, edge.target) for edge in mapped.relationships}
    assert (ids().file("docs/guide.md"), stable_folder_id("docs")) in targets
    assert (ids().file("docs/other.md"), stable_folder_id("docs")) in targets
    assert all(edge.properties.get("hierarchical") is True for edge in mapped.relationships)

    data = build_viewer_data(
        **fixture_inputs(extra_nodes=tuple(mapped.entities), max_nodes=200)
    )
    by_id = {node["id"]: node for node in data["nodes"]}
    assert by_id[stable_folder_id("docs")]["kind"] == "Folder"
    assert by_id[stable_folder_id("docs")]["t"] == pytest.approx(35 / 48)
    assert by_id[ids().repository_id]["kind"] == "Repository"
    assert by_id[ids().repository_id]["t"] is None


def stable_folder_id(path: str) -> str:
    from repo_history.ids import stable_id

    return stable_id("folder", ids().repository_id, path)


def writes_edge(
    source: str, target: str, edge_type: str, origin: str = "git",
) -> RelationshipWrite:
    from repo_history.graph import RelationshipWrite

    return RelationshipWrite(
        edge_id=f"write:{source}:{edge_type}:{target}",
        edge_type=edge_type,
        source=source,
        target=target,
        properties={"rationale": f"{edge_type} edge"},
        valid_from=DAY_ONE,
        origin=origin,
        origin_id=f"{source}:{target}",
        event_id=f"event:{source}:{target}",
        run_id=RUN_ID,
        info_horizon=NOW,
        evidence_ids=(),
    )


def test_extra_edges_join_kept_nodes_with_factual_default():
    arguments = fixture_inputs()
    arguments["extra_edges"] = [
        writes_edge(ids().commit(MID), ids().commit(ROOT), "PARENT_COMMIT"),
        writes_edge(ids().commit(TIP), ids().file("docs/guide.md"), "TOUCHES"),
        writes_edge(
            ids().work_item("github", 10), ids().topic("sparse"), "ABOUT",
            origin="labeling",
        ),
        writes_edge("snap:guide", ids().topic("sparse"), "ABOUT", origin="labeling"),
    ]
    data = build_viewer_data(**arguments)

    by_pair = {(edge["source"], edge["target"]): edge for edge in data["edges"]}
    assert by_pair[(ids().commit(MID), ids().commit(ROOT))]["relation"] == "PARENT_COMMIT"
    assert by_pair[(ids().commit(TIP), ids().file("docs/guide.md"))]["relation"] == "TOUCHES"
    pair = by_pair[(ids().work_item("github", 10), ids().topic("sparse"))]
    assert pair["relation"] == "ABOUT"
    # The snapshot-subject ABOUT aggregates to the same file edge.
    assert ("snap:guide", ids().topic("sparse")) not in by_pair
    assert by_pair[(ids().commit(MID), ids().commit(ROOT))]["inferred"] is False
    assert by_pair[(ids().commit(TIP), ids().file("docs/guide.md"))]["inferred"] is False
    assert by_pair[(ids().work_item("github", 10), ids().topic("sparse"))]["inferred"] is False


def test_extra_linking_related_to_edges_are_flagged_inferred():
    arguments = fixture_inputs(links=[])
    arguments["extra_edges"] = [
        writes_edge(
            ids().work_item("github", 10), ids().topic("sparse"),
            "RELATED_TO", origin="linking",
        ),
        writes_edge(
            ids().work_item("github", 10), ids().file("docs/guide.md"),
            "RELATED_TO", origin="github",
        ),
    ]
    data = build_viewer_data(**arguments)

    by_target = {edge["target"]: edge for edge in data["edges"]}
    assert by_target[ids().topic("sparse")]["inferred"] is True
    assert by_target[ids().file("docs/guide.md")]["inferred"] is False


def test_extra_edges_carry_hierarchical_flags():
    arguments = fixture_inputs()
    arguments["extra_edges"] = [
        writes_edge(ids().commit(MID), ids().commit(ROOT), "PARENT_COMMIT"),
        writes_edge(ids().work_item("github", 10), ids().topic("sparse"), "ABOUT"),
    ]
    data = build_viewer_data(**arguments)

    by_pair = {(edge["source"], edge["target"]): edge for edge in data["edges"]}
    assert by_pair[(ids().commit(MID), ids().commit(ROOT))]["hierarchical"] is True
    pair = by_pair[(ids().work_item("github", 10), ids().topic("sparse"))]
    assert pair["relation"] == "ABOUT"
    assert pair["hierarchical"] is False


def test_file_nodes_carry_a_distinguishing_language():
    data = build_viewer_data(**fixture_inputs())
    by_id = {node["id"]: node for node in data["nodes"]}

    assert by_id[ids().file("docs/guide.md")]["language"] == "markdown"
    assert by_id[ids().file("docs/guide.md")]["kind"] == "File"


def test_viewer_page_has_hierarchy_file_and_time_controls():
    html = render_viewer_html(build_viewer_data(**fixture_inputs()))

    for control in ("hier", "rank", "lpull", "filecolors", "tmin", "tmax"):
        assert f'id="{control}"' in html
    for hook in ("TIERS", "rankY", "hierMult", "langColor", "fileTypeColors"):
        assert hook in html
    for gone in ("labelForce", "lForce.enabled", "labspread", "nolap", "separate labels"):
        assert gone not in html
    assert "color files by type" in html


def test_viewer_synthesizes_commit_file_touches_and_file_details():
    from repo_history.git import TreeEntryRecord

    arguments = fixture_inputs()
    index = arguments["index"]
    arguments["index"] = index.model_copy(update={"snapshots": (
        TreeEntryRecord(
            snapshot_id="snap:guide", commit_sha=TIP, path="docs/guide.md",
            object_id="o" * 40, mode=0o100644, kind="blob", size=10,
            is_binary=False, is_lfs_pointer=False, content_status="inspected",
            source_url="u",
        ),
    )})
    data = build_viewer_data(**arguments)

    touches = [
        edge for edge in data["edges"]
        if edge["relation"] == "TOUCHES"
        and edge["source"] == ids().commit(TIP)
        and edge["target"] == ids().file("docs/guide.md")
    ]
    assert len(touches) == 1
    assert touches[0]["hierarchical"] is True
    assert touches[0]["inferred"] is False

    by_id = {node["id"]: node for node in data["nodes"]}
    doc = by_id[ids().file("docs/guide.md")]
    assert doc["kind"] == "File"
    assert doc["t"] is not None
    assert doc["props"]["history"]
    assert doc["props"]["preview"].startswith("# Guide")


def test_class_nodes_rank_below_files_and_carry_qualified_names():
    arguments = fixture_inputs()
    arguments["extra_nodes"] = [
        __import__("repo_history.graph", fromlist=["EntityWrite"]).EntityWrite(
            logical_id="class:1", labels=("Class",),
            properties={"name": "GraphConv", "qualname": "models.GraphConv",
                        "path": "models/gcn.py", "line": 3},
            valid_from=DAY_ONE, origin="git", origin_id="models/gcn.py:GraphConv",
            event_id="e1", run_id=RUN_ID, info_horizon=NOW, evidence_ids=(),
        )
    ]
    data = build_viewer_data(**arguments)
    by_id = {node["id"]: node for node in data["nodes"]}

    assert by_id["class:1"]["label"] == "GraphConv"
    assert by_id["class:1"]["kind"] == "Class"


def test_export_is_deterministic():
    assert build_viewer_data(**fixture_inputs()) == build_viewer_data(**fixture_inputs())


def test_extra_structural_nodes_join_with_hierarchy_times():
    from repo_history.graph import EntityWrite

    arguments = fixture_inputs()
    arguments["extra_nodes"] = [
        EntityWrite(
            logical_id="folder:docs", labels=("Folder",),
            properties={"path": "docs", "file_count": 1},
            valid_from=WINDOW.start, origin="git", origin_id="folder:docs",
            event_id="e1", run_id=RUN_ID, info_horizon=NOW, evidence_ids=(),
        ),
        EntityWrite(
            logical_id="repo:x", labels=("Repository",),
            properties={"url": "https://example.test/r"},
            valid_from=WINDOW.start, origin="git", origin_id="repo",
            event_id="e2", run_id=RUN_ID, info_horizon=NOW, evidence_ids=(),
        ),
    ]
    data = build_viewer_data(**arguments)
    by_id = {node["id"]: node for node in data["nodes"]}

    assert by_id["folder:docs"]["kind"] == "Folder"
    assert by_id["folder:docs"]["t"] == pytest.approx(35 / 48)
    assert by_id["repo:x"]["t"] is None
    assert "Repository" in {node["kind"] for node in data["nodes"]}


def test_file_nodes_show_filenames_with_clickable_definitions():
    data = build_viewer_data(**fixture_inputs())

    files = [node for node in data["nodes"] if node["kind"] == "File"]
    assert files
    for node in files:
        assert "/" not in node["label"], node["label"]
        assert node["props"]["definition_url"].startswith("https://"), node["props"]
        assert node["props"]["path"].endswith(node["label"])
    assert data["meta"]["repository"] == {
        "name": "mylonasc/tf_gnns",
        "url": "https://github.com/mylonasc/tf_gnns",
    }


def test_details_panel_offers_clickable_repo_and_definition_links():
    html = render_viewer_html(build_viewer_data(**fixture_inputs()))

    assert 'id="repolink"' in html
    assert "definition_url" in html
    assert "target=" in html and "_blank" in html


def test_viewer_has_rank_force_and_no_label_force():
    html = render_viewer_html(build_viewer_data(**fixture_inputs()))

    assert "rankY" in html and "TIERS" in html
    assert 'id="rank"' in html
    assert "labelForce" not in html and "nolap" not in html
    assert "hierarchical pull" in html


def test_page_embeds_data_controls_and_temporal_forces():
    html = render_viewer_html(build_viewer_data(**fixture_inputs()), title="Demo")

    assert "https://cdn.jsdelivr.net/npm/d3@7" in html
    for control in ("tmin", "tmax", "anchor", "spread", "rank", "ldist", "charge",
                    "labels", "bytime", "filecolors", "search", "find", "pause", "reset",
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
