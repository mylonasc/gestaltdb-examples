"""Python inspection API answering the six user-visible questions.

The :class:`Inspector` works from adapter outputs (no database, no network),
so every answer is deterministic and replayable.  Each answer carries its
information horizon and the coverage gaps behind it; provider facts,
deterministic extraction, and inference stay labeled, and inferred links
(``RELATED_TO``) are never mixed with factual ones (``CLOSED_BY``,
``MENTIONS``, ``INTENDS_TO_RESOLVE``, structural edges).
"""

from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Any, Sequence

from .content import ContentIndex
from .git import GitIndex
from .github import GitHubDiscovery
from .ids import IdFactory
from .labeling import TopicAssignment
from .linking import LinkRecord, LinkRelation
from .models import CoverageGap, FrozenModel, ensure_utc

__all__ = [
    "AssignmentExplanation",
    "BranchCoverage",
    "CoverageStatement",
    "Inspector",
    "RangeChanges",
    "RelatedFile",
    "SummarizedChange",
    "SummarizedCommit",
    "TopicEvent",
    "WorkItemTrace",
]


class SummarizedCommit(FrozenModel):
    sha: str
    message: str
    committed_at: datetime
    parents: tuple[str, ...] = ()
    source_url: str = ""


class SummarizedChange(FrozenModel):
    commit_sha: str
    path: str
    status: str
    comparison: str = ""
    source_url: str = ""


class RangeChanges(FrozenModel):
    from_sha: str
    to_sha: str
    commits: tuple[SummarizedCommit, ...] = ()
    changes: tuple[SummarizedChange, ...] = ()
    truncated: bool = False
    info_horizon: datetime


class WorkItemTrace(FrozenModel):
    commit_sha: str
    work_item_number: int
    work_item_kind: str
    title: str
    relation: str
    rationale: str
    evidence_urls: tuple[str, ...] = ()
    info_horizon: datetime


class BranchCoverage(FrozenModel):
    commit_sha: str
    branches: tuple[str, ...] = ()
    info_horizon: datetime


class TopicEvent(FrozenModel):
    topic_id: str
    topic_name: str | None = None
    time: datetime
    subject_id: str = ""
    action: str = ""
    confidence: float = 0.0


class RelatedFile(FrozenModel):
    path: str
    relation: str
    confidence: float = 1.0
    sections: tuple[str, ...] = ()
    info_horizon: datetime


class AssignmentExplanation(FrozenModel):
    subject_id: str
    kind: str
    summary: str
    rationale: str
    method: str
    confidence: float = 0.0
    evidence_ids: tuple[str, ...] = ()
    registry_version: int | None = None
    info_horizon: datetime


class CoverageStatement(FrozenModel):
    info_horizon: datetime
    window_start: datetime
    window_end: datetime
    gaps: tuple[CoverageGap, ...] = ()
    counts: dict[str, int] = {}


class Inspector:
    """Answer inspection questions from adapter outputs and link reports."""

    def __init__(
        self,
        *,
        index: GitIndex,
        discovery: GitHubDiscovery,
        content: ContentIndex,
        links: Sequence[LinkRecord],
        assignments: Sequence[TopicAssignment] = (),
        info_horizon: datetime,
        run_id: str = "",
        repository_url: str = "",
    ) -> None:
        self._index = index
        self._discovery = discovery
        self._content = content
        self._links = tuple(links)
        self._assignments = tuple(assignments)
        self._horizon = ensure_utc(info_horizon)
        self._run_id = run_id
        self._ids = IdFactory(repository_url) if repository_url else None
        self._commit_times = {
            record.sha: record.committed_at
            for record in (
                *index.selection.selected_commits, *index.selection.boundary_commits
            )
        }
        self._commit_messages = {
            record.sha: record for record in (
                *index.selection.selected_commits, *index.selection.boundary_commits
            )
        }
        self._items = {item.number: item for item in discovery.work_items}

    @property
    def info_horizon(self) -> datetime:
        return self._horizon

    @property
    def git_index(self) -> GitIndex:
        return self._index

    @property
    def github_discovery(self) -> GitHubDiscovery:
        return self._discovery

    @property
    def content_index(self) -> ContentIndex:
        return self._content

    @property
    def link_records(self) -> tuple[LinkRecord, ...]:
        return self._links

    @property
    def topic_assignments(self) -> tuple[TopicAssignment, ...]:
        return self._assignments

    # -- Q1: what changed across a commit range ---------------------------
    def changes_in_range(self, from_sha: str, to_sha: str) -> RangeChanges:
        """Commits in ``(from_sha, to_sha]``: reachable from ``to_sha``.

        Git-range semantics: the base commit itself is excluded, so a
        degenerate ``(sha, sha)`` range is empty.  Traversal stops at
        ``from_sha`` and never walks past it.
        """

        parents: dict[str, tuple[str, ...]] = {
            sha: tuple(self._commit_messages[sha].parents)
            for sha in self._commit_times
            if sha in self._commit_messages
        }
        reachable: list[str] = []
        seen = {from_sha}
        queue: deque[str] = deque([to_sha])
        truncated = to_sha not in self._commit_times or from_sha not in self._commit_times
        while queue:
            sha = queue.popleft()
            if sha in seen:
                continue
            seen.add(sha)
            record = self._commit_messages.get(sha)
            if record is None:
                truncated = True
                continue
            reachable.append(sha)
            queue.extend(
                parent for parent in record.parents
                if parent not in seen and parent != from_sha
            )
        reachable_sorted = sorted(
            reachable, key=lambda sha: (self._commit_times[sha], sha)
        )
        commits = tuple(
            SummarizedCommit(
                sha=sha,
                message=self._commit_messages[sha].message,
                committed_at=self._commit_times[sha],
                parents=parents.get(sha, ()),
                source_url=self._commit_messages[sha].source_url,
            )
            for sha in reachable_sorted
        )
        wanted = set(reachable_sorted)
        changes = tuple(
            SummarizedChange(
                commit_sha=change.commit_sha,
                path=change.path,
                status=change.status,
                comparison=change.comparison,
                source_url=change.source_url,
            )
            for change in self._index.changes
            if change.commit_sha in wanted
        )
        return RangeChanges(
            from_sha=from_sha,
            to_sha=to_sha,
            commits=commits,
            changes=changes,
            truncated=truncated,
            info_horizon=self._horizon,
        )

    # -- Q2: which work items explain changes ------------------------------
    def work_items_for_commits(self, shas: Sequence[str]) -> list[WorkItemTrace]:
        """Trace commits to work items through closure and mention links."""

        traces: list[WorkItemTrace] = []
        wanted = set(shas)
        commit_ids = (
            {self._ids.commit(sha) for sha in wanted} if self._ids is not None else set()
        )
        for link in self._links:
            if link.relation is LinkRelation.CLOSED_BY and link.target_id in commit_ids:
                number = self._number_of(link.source_id)
                item = self._items.get(number, None) if number is not None else None
                if item is None:
                    continue
                traces.append(
                    WorkItemTrace(
                        commit_sha=self._sha_of(link.target_id),
                        work_item_number=item.number,
                        work_item_kind=item.kind,
                        title=item.title,
                        relation=link.relation.value,
                        rationale=link.rationale,
                        evidence_urls=tuple(
                            ref.url for ref in link.evidence if ref.url
                        ),
                        info_horizon=min(self._horizon, link.info_horizon),
                    )
                )
            elif link.relation in (LinkRelation.MENTIONS, LinkRelation.INTENDS_TO_RESOLVE):
                if link.source_id in commit_ids:
                    number = self._number_of(link.target_id)
                elif link.target_id in commit_ids:
                    number = self._number_of(link.source_id)
                else:
                    continue
                item = self._items.get(number, None) if number is not None else None
                if item is None:
                    continue
                traces.append(
                    WorkItemTrace(
                        commit_sha=self._sha_of(
                            link.source_id if link.source_id in commit_ids
                            else link.target_id
                        ),
                        work_item_number=item.number,
                        work_item_kind=item.kind,
                        title=item.title,
                        relation=link.relation.value,
                        rationale=link.rationale,
                        evidence_urls=tuple(
                            ref.url for ref in link.evidence if ref.url
                        ),
                        info_horizon=min(self._horizon, link.info_horizon),
                    )
                )
        traces.sort(key=lambda trace: (trace.commit_sha, trace.work_item_number))
        return traces

    def _number_of(self, work_item_id: str) -> int | None:
        for number in self._items:
            if self._ids is not None and work_item_id == self._ids.work_item("github", number):
                return number
            if work_item_id in (f"workitem:github:{number}", str(number)):
                return number
            if work_item_id.endswith(f":{number}"):
                return number
        return None

    def _sha_of(self, commit_id: str) -> str:
        for sha in self._commit_times:
            if self._ids is not None and commit_id == self._ids.commit(sha):
                return sha
        return commit_id

    # -- Q3: which branches contain commits ---------------------------------
    def branches_containing(self, sha: str) -> BranchCoverage:
        """Branches whose tips reach ``sha`` through parent edges."""

        branches = sorted(
            branch.name
            for branch in self._index.branches
            if sha in self._ancestors_including(branch.tip_sha)
        )
        return BranchCoverage(
            commit_sha=sha, branches=tuple(branches), info_horizon=self._horizon
        )

    def _ancestors_including(self, sha: str) -> set[str]:
        seen: set[str] = set()
        queue: deque[str] = deque([sha])
        while queue:
            current = queue.popleft()
            if current in seen:
                continue
            seen.add(current)
            record = self._commit_messages.get(current)
            if record is not None:
                queue.extend(record.parents)
        return seen

    # -- Q4: how did a topic emerge or recur -------------------------------
    def topic_timeline(self, topic_id: str) -> list[TopicEvent]:
        """Assignments for one topic in horizon order: emergence then recurrence."""

        events: list[TopicEvent] = []
        for assignment in self._assignments:
            if assignment.kind != "derived" or assignment.topic_id != topic_id:
                continue
            moment = assignment.result.provenance.info_horizon
            if moment > self._horizon:
                continue
            events.append(
                TopicEvent(
                    topic_id=topic_id,
                    topic_name=assignment.topic_name,
                    time=moment,
                    subject_id=assignment.subject_id,
                    action=assignment.action,
                    confidence=assignment.confidence,
                )
            )
        events.sort(key=lambda event: (event.time, event.subject_id))
        return events

    # -- Q5: files and sections for a work item at a chosen time ----------
    def files_for_work_item(
        self, number: int, at: datetime
    ) -> list[RelatedFile]:
        """Files linked to a work item with evidence visible at time ``at``."""

        moment = ensure_utc(at)
        candidates = {str(number), f"workitem:github:{number}"}
        if self._ids is not None:
            candidates.add(self._ids.work_item("github", number))
        related: list[RelatedFile] = []
        for link in self._links:
            if link.source_id not in candidates:
                continue
            if link.info_horizon > moment:
                continue
            if link.relation not in (LinkRelation.MENTIONS, LinkRelation.RELATED_TO):
                continue
            path = self._path_of(link.target_id)
            if path is None:
                continue
            sections = self._sections_for(path, moment)
            related.append(
                RelatedFile(
                    path=path,
                    relation=link.relation.value,
                    confidence=link.confidence,
                    sections=tuple(sections),
                    info_horizon=min(moment, link.info_horizon),
                )
            )
        related.sort(key=lambda entry: (-entry.confidence, entry.path))
        return related

    def _path_of(self, file_id: str) -> str | None:
        for snapshot in self._content.snapshots:
            if self._ids is not None and file_id == self._ids.file(snapshot.path):
                return snapshot.path
            # Inferred topical links target snapshot occurrences; resolve
            # them to their file path for the file-level answer.
            if file_id == snapshot.snapshot_id:
                return snapshot.path
        return None

    def _sections_for(self, path: str, moment: datetime) -> list[str]:
        known = {
            snapshot.snapshot_id
            for snapshot in self._content.snapshots
            if snapshot.path == path
            and self._commit_times.get(snapshot.commit_sha, moment) <= moment
        }
        return sorted(
            {
                section.heading or section.fragment
                for section in self._content.sections
                if section.snapshot_id in known and section.heading
            }
        )

    # -- Q6: why was an entity labeled or connected -------------------------
    def explain(self, entity_id: str) -> list[AssignmentExplanation]:
        """Attribution for labels and links touching one entity."""

        explanations: list[AssignmentExplanation] = []
        for assignment in self._assignments:
            if assignment.subject_id != entity_id:
                continue
            if assignment.result.provenance.info_horizon > self._horizon:
                continue
            explanations.append(
                AssignmentExplanation(
                    subject_id=entity_id,
                    kind=f"label:{assignment.kind}:{assignment.action}",
                    summary=(
                        assignment.topic_name or assignment.provider_label or ""
                    ),
                    rationale=assignment.rationale,
                    method=assignment.result.provenance.method,
                    confidence=assignment.confidence,
                    evidence_ids=tuple(
                        ref.evidence_id for ref in assignment.result.evidence
                    ),
                    registry_version=(
                        assignment.registry_version
                        if assignment.registry_version >= 0
                        else None
                    ),
                    info_horizon=assignment.result.provenance.info_horizon,
                )
            )
        for link in self._links:
            if link.info_horizon > self._horizon:
                continue
            if link.source_id != entity_id and link.target_id != entity_id:
                continue
            explanations.append(
                AssignmentExplanation(
                    subject_id=entity_id,
                    kind=f"link:{link.relation.value}",
                    summary=link.target_id,
                    rationale=link.rationale,
                    method=link.method,
                    confidence=link.confidence,
                    evidence_ids=tuple(
                        ref.evidence_id for ref in link.evidence
                    ),
                    registry_version=None,
                    info_horizon=link.info_horizon,
                )
            )
        explanations.sort(key=lambda item: (item.info_horizon, item.kind, item.summary))
        return explanations

    # -- coverage ------------------------------------------------------------
    def coverage(self) -> CoverageStatement:
        gaps = (
            *self._index.selection.gaps,
            *self._index.gaps,
            *self._discovery.gaps,
            *self._content.gaps,
        )
        counts = {
            "commits": len(self._commit_times),
            "changes": len(self._index.changes),
            "snapshots": len(self._index.snapshots),
            "work_items": len(self._items),
            "sections": len(self._content.sections),
            "links": len(self._links),
            "assignments": len(self._assignments),
        }
        window = self._index.selection.window
        return CoverageStatement(
            info_horizon=self._horizon,
            window_start=window.start,
            window_end=window.end,
            gaps=tuple(gaps),
            counts=counts,
        )
