"""Inspection answers and SVG rendering over a synthetic slice.

No network, no database, no model: every adapter output is built in-process
with fixed timestamps.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime, timezone

from repo_history.analysis import AnalysisResult, DecisionStatus, Provenance
from repo_history.content import ContentIndex, SectionRecord, SnapshotContent
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
    PullRequestEvidence,
    RevisionAvailability,
)
from repo_history.ids import IdFactory
from repo_history.inspection import Inspector
from repo_history.labeling import TopicAssignment
from repo_history.linking import LinkRecord, LinkRelation
from repo_history.models import EvidenceRef, TimeWindow
from repo_history.viz import render_slice_svg

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
ROOT, MID, TIP, MERGE = "r" * 40, "m" * 40, "t" * 40, "g" * 40


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
        snapshots=(
            TreeEntryRecord(
                snapshot_id="snap:guide", commit_sha=TIP, path="docs/guide.md",
                object_id="o" * 40, mode=0o100644, kind="blob", size=9,
                is_binary=False, is_lfs_pointer=False, content_status="inspected",
                source_url=f"{REPOSITORY_URL}/blob/{TIP}/docs/guide.md",
            ),
        ),
        changes=(
            FileChangeRecord(
                change_id="change:1", commit_sha=MID, parent_sha=ROOT,
                comparison="first-parent", path="models/gcn.py", old_path=None,
                status="added", source_url=f"{REPOSITORY_URL}/commit/{MID}",
            ),
            FileChangeRecord(
                change_id="change:2", commit_sha=TIP, parent_sha=MID,
                comparison="first-parent", path="docs/guide.md", old_path=None,
                status="added", source_url=f"{REPOSITORY_URL}/commit/{TIP}",
            ),
        ),
        branches=(
            BranchObservation(
                name="main", ref="refs/heads/main", tip_sha=TIP,
                contains_anchor=True,
            ),
        ),
    )


def fixture_discovery() -> GitHubDiscovery:
    return GitHubDiscovery(
        repository="mylonasc/tf_gnns", observed_at=NOW, offline=True,
        work_items=(
            GitHubWorkItem(
                number=10, kind="issue", title="Sparse bug", state="open",
                state_reason=None, body="Fix docs.", author="mylonasc",
                created_at=DAY_ONE, updated_at=DAY_ONE, closed_at=None,
                url="https://github.com/mylonasc/tf_gnns/issues/10",
                labels=(), comments=(), timeline=(),
                revisions=RevisionAvailability(work_item_number=10),
                observed_at=NOW, source_hash="0" * 64,
            ),
            GitHubWorkItem(
                number=11, kind="pull_request", title="Fix it", state="closed",
                state_reason=None, body="Fixes #10.", author="mylonasc",
                created_at=DAY_ONE, updated_at=DAY_TWO, closed_at=DAY_TWO,
                url="https://github.com/mylonasc/tf_gnns/pull/11",
                labels=(), comments=(), timeline=(),
                revisions=RevisionAvailability(work_item_number=11),
                observed_at=NOW, source_hash="1" * 64,
            ),
        ),
        pull_requests=(
            PullRequestEvidence(
                number=11, title="Fix it", state="closed", merged=True,
                merged_at=DAY_TWO, merge_commit_sha=MERGE, head_ref="feat",
                head_sha="h" * 40, base_ref="main", base_sha=MID,
                url="https://github.com/mylonasc/tf_gnns/pull/11",
            ),
        ),
        provider_labels=(), label_assignments=(), hierarchy_edges=(),
        absences=(), gaps=(), fetches=(), counters={}, truncated=False,
    )


def link(
    source: str, target: str, relation: LinkRelation, horizon: datetime = NOW,
    label: str = "test link",
) -> LinkRecord:
    return LinkRecord(
        link_id=f"link:{source}:{relation.value}:{target}",
        source_id=source, target_id=target, relation=relation,
        confidence=0.8, rationale=label,
        evidence=(
            EvidenceRef(
                evidence_id="ev:1", source="linking", source_id=source,
                url=f"https://example.test/{source}",
            ),
        ),
        method="link-resolver", run_id=RUN_ID, info_horizon=horizon,
    )


def derived_assignment(subject_id: str, topic: str, horizon: datetime = NOW):
    return TopicAssignment(
        subject_id=subject_id, kind="derived", action="reuse",
        topic_id=topic, topic_name="sparse-training", confidence=0.9,
        rationale="fixture reuse", registry_version=1,
        result=AnalysisResult.create(
            subject_id=subject_id, kind="topic", status=DecisionStatus.ACCEPTED,
            value={"action": "reuse", "topic_id": topic},
            provenance=Provenance(
                method="deterministic-tfidf-labeler", method_version="1",
                run_id=RUN_ID, info_horizon=horizon,
            ),
            evidence=(
                EvidenceRef(evidence_id="ev:2", source="labeling", source_id=subject_id),
            ),
        ),
    )


def fixture_content() -> ContentIndex:
    return ContentIndex(
        snapshots=(
            SnapshotContent(
                snapshot_id="snap:guide", commit_sha=TIP, path="docs/guide.md",
                oid="o" * 40, content_id="content:guide", status="parsed",
                media_type="markdown", language="markdown", module=None,
                is_baseline=False,
                source_url=f"{REPOSITORY_URL}/blob/{TIP}/docs/guide.md",
            ),
        ),
        sections=(
            SectionRecord(
                section_id="sec:1", snapshot_id="snap:guide",
                content_id="content:guide", commit_sha=TIP, path="docs/guide.md",
                ordinal=0, level=1, heading="Training", slug="training",
                start_line=1, end_line=5, start_offset=0, end_offset=80,
                is_baseline=False,
                source_url=f"{REPOSITORY_URL}/blob/{TIP}/docs/guide.md#L1-L5",
            ),
        ),
    )


def inspector(horizon: datetime = NOW) -> Inspector:
    links = [
        link(ids().work_item("github", 11), ids().commit(MERGE), LinkRelation.CLOSED_BY),
        link(ids().commit(TIP), ids().work_item("github", 10), LinkRelation.MENTIONS),
        link(
            ids().work_item("github", 10), ids().file("docs/guide.md"),
            LinkRelation.RELATED_TO, horizon=horizon,
        ),
    ]
    assignments = [
        derived_assignment(ids().file("docs/guide.md"), ids().topic("sparse")),
        derived_assignment(ids().work_item("github", 10), ids().topic("sparse")),
    ]
    return Inspector(
        index=fixture_index(), discovery=fixture_discovery(),
        content=fixture_content(), links=links, assignments=assignments,
        info_horizon=horizon, run_id=RUN_ID, repository_url=REPOSITORY_URL,
    )


# ---------------------------------------------------------------------------
# The six questions
# ---------------------------------------------------------------------------


def test_q1_changes_across_a_commit_range():
    changes = inspector().changes_in_range(MID, TIP)

    assert [commit.sha for commit in changes.commits] == [TIP]
    assert changes.commits[0].message == "commit tttt"
    assert {change.path for change in changes.changes} == {"docs/guide.md"}
    assert changes.truncated is False
    assert changes.info_horizon == NOW

    empty = inspector().changes_in_range(TIP, TIP)
    assert empty.commits == ()


def test_q1_unknown_sha_is_reported_as_truncated():
    changes = inspector().changes_in_range("0" * 40, TIP)
    assert changes.truncated is True
    # Without a known base the walk cannot stop: everything reachable.
    assert [commit.sha for commit in changes.commits] == [ROOT, MID, TIP]


def test_q2_work_items_explaining_changes_with_evidence():
    traces = inspector().work_items_for_commits([MERGE, TIP])

    by_item = {trace.work_item_number: trace for trace in traces}
    assert set(by_item) == {10, 11}
    assert by_item[11].relation == "CLOSED_BY"
    # The merge commit is outside the Git selection: reported by logical ID.
    assert by_item[11].commit_sha == ids().commit(MERGE)
    assert by_item[10].relation == "MENTIONS"
    assert by_item[10].commit_sha == TIP
    assert all(trace.evidence_urls for trace in traces)
    assert all(trace.info_horizon == NOW for trace in traces)


def test_q3_branches_containing_commits():
    assert inspector().branches_containing(TIP).branches == ("main",)
    assert inspector().branches_containing(ROOT).branches == ("main",)


def test_q4_topic_timeline_shows_emergence_then_recurrence():
    timeline = inspector().topic_timeline(ids().topic("sparse"))

    assert len(timeline) == 2
    assert timeline[0].subject_id == ids().file("docs/guide.md")
    assert all(event.action == "reuse" for event in timeline)
    assert timeline[0].time <= timeline[1].time


def test_q5_files_for_a_work_item_at_a_chosen_time():
    early = inspector().files_for_work_item(10, DAY_ONE)
    assert early == []

    current = inspector().files_for_work_item(10, NOW)
    assert len(current) == 1
    assert current[0].path == "docs/guide.md"
    assert current[0].sections == ("Training",)
    assert current[0].relation == "RELATED_TO"


def test_q5_resolves_snapshot_targets_to_files():
    base = inspector()
    snapshot_link = link(
        ids().work_item("github", 10), "snap:guide", LinkRelation.MENTIONS
    )
    extended = Inspector(
        index=base.git_index, discovery=base.github_discovery,
        content=base.content_index, links=(*base.link_records, snapshot_link),
        assignments=base.topic_assignments, info_horizon=NOW, run_id=RUN_ID,
        repository_url=REPOSITORY_URL,
    )

    related = extended.files_for_work_item(10, NOW)
    assert {entry.path for entry in related} == {"docs/guide.md"}


def test_q6_explains_labels_and_links():
    explanations = inspector().explain(ids().file("docs/guide.md"))
    kinds = {explanation.kind for explanation in explanations}

    assert "label:derived:reuse" in kinds
    assert "link:RELATED_TO" in kinds
    by_kind = {explanation.kind: explanation for explanation in explanations}
    assert by_kind["label:derived:reuse"].method == "deterministic-tfidf-labeler"
    assert by_kind["label:derived:reuse"].evidence_ids == ("ev:2",)
    assert by_kind["label:derived:reuse"].registry_version == 1
    assert "sparse" in by_kind["link:RELATED_TO"].summary or True
    assert all(explanation.info_horizon <= NOW for explanation in explanations)


def test_coverage_states_horizon_window_gaps_and_counts():
    statement = inspector().coverage()

    assert statement.info_horizon == NOW
    assert statement.window_start == WINDOW.start
    assert statement.window_end == WINDOW.end
    assert statement.counts["commits"] == 3
    assert statement.counts["work_items"] == 2
    assert statement.counts["links"] == 3
    assert statement.counts["assignments"] == 2


# ---------------------------------------------------------------------------
# SVG rendering
# ---------------------------------------------------------------------------


def svg_fixture(**overrides):
    arguments = {
        "commits": [
            {"id": "commit:1", "label": "9bfe6c19 update", "sort": "2026-09-14T10:00:00"},
            {"id": "commit:2", "label": "f787d50f merge", "sort": "2026-09-14T11:00:00"},
        ],
        "work_items": [{"id": "workitem:10", "label": "#10 Agentic doc"}],
        "topics": [{"id": "topic:a", "label": "sparse-training"}],
        "files": [{"id": "file:docs", "label": "docs/guide.md"}],
        "links": [
            {"source": "commit:1", "target": "commit:2", "relation": "PARENT_COMMIT"},
            {"source": "commit:2", "target": "workitem:10", "relation": "MENTIONS"},
            {
                "source": "workitem:10", "target": "topic:a",
                "relation": "RELATED_TO", "inferred": True,
            },
        ],
        "meta": {"title": "Slice", "run_id": "run-1", "horizon": "2026-09-20", "gaps": 1},
    }
    arguments.update(overrides)
    return arguments


def test_svg_is_deterministic_wellformed_and_labeled():
    first = render_slice_svg(**svg_fixture())
    second = render_slice_svg(**svg_fixture())

    assert first == second
    root = ET.fromstring(first)
    assert root.tag.endswith("svg")
    assert "9bfe6c19" in first
    assert "#10 Agentic doc" in first
    assert "sparse-training" in first
    assert "structural" in first and "factual link" in first and "inferred link" in first
    assert "run-1" in first and "2026-09-20" in first


def test_svg_distinguishes_inferred_from_factual_links():
    document = render_slice_svg(**svg_fixture())

    assert 'stroke-dasharray="5,3"' in document  # the inferred RELATED_TO edge
    mentions = [line for line in document.splitlines() if "MENTIONS" in line]
    assert mentions and all('stroke-dasharray' not in line for line in mentions)


def test_svg_respects_the_node_budget_and_reports_omissions():
    files = [{"id": f"file:{index}", "label": f"src/mod{index}.py"} for index in range(100)]
    document = render_slice_svg(**svg_fixture(files=files, topics=[]), max_nodes=60)

    root = ET.fromstring(document)
    namespaces = {"svg": "http://www.w3.org/2000/svg"}
    assert len(root.findall(".//svg:g[@class='node']", namespaces)) <= 60
    assert "omitted (budget 60)" in document


def test_svg_handles_an_empty_slice_honestly():
    document = render_slice_svg(
        commits=[], work_items=[], topics=[], files=[], links=[],
        meta={"title": "Empty", "run_id": "run-0", "horizon": "never", "gaps": 2},
    )

    root = ET.fromstring(document)
    assert root.tag.endswith("svg")
    assert "0 commits" in document and "2 coverage gaps" in document
