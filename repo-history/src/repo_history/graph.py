"""Single-writer GestaltDB temporal graph persistence.

The sink keeps two coordinated layers, matching the driver's model:

- immutable temporal versions (``put_node_version`` / ``put_edge_version``)
  for the audit trail, read back with ``*_as_of`` and ``iter_*_versions``;
- a mutable current projection (``put_node`` / ``put_edge``) for indexed
  reads and traversal, rebuildable at any time by replaying the journal.

Every version carries provenance properties (run, origin event, information
horizon, evidence IDs), and every temporal commit records the run in its
metadata.  Version IDs are deterministic UUIDs, so ingesting the same batch
twice is idempotent: the driver rejects the duplicate ID and the sink skips
it after verifying the stored payload matches.  Corrections append new
versions that supersede earlier ones; old evidence is never rewritten.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Any

from gestaltdb.graphdb import (
    Edge,
    GraphDB,
    IndexMaintenanceMode,
    Node,
    TemporalVersionError,
)

from .git import GitIndex
from .github import GitHubDiscovery
from .ids import IdFactory, stable_id
from .labeling import TopicAssignment, TopicRegistry
from .models import CoverageGap, CoverageStatus, FrozenModel, GraphSettings, ensure_utc
from .protocols import GraphSink as GraphSinkProtocol

__all__ = [
    "ABOUT",
    "CHILD_OF",
    "HAS_LABEL",
    "MEMBER_OF",
    "MERGED_AS",
    "OF_FILE",
    "PARENT_COMMIT",
    "TIP_OF",
    "TOUCHES",
    "EntityWrite",
    "GraphClosedError",
    "GraphError",
    "GraphSink",
    "MappedGraph",
    "NODE_INDEXES",
    "EDGE_INDEXES",
    "RelationshipWrite",
    "VersionRecord",
    "WriteReport",
    "map_discovery",
    "map_git_index",
    "map_registry",
    "map_topic_assignments",
]

# Edge types used by typed traversal.  The driver reads the edge type from
# the ``type`` property, so the sink always sets it.
ABOUT = "ABOUT"
PARENT_COMMIT = "PARENT_COMMIT"
MEMBER_OF = "MEMBER_OF"
OF_FILE = "OF_FILE"
TOUCHES = "TOUCHES"
HAS_LABEL = "HAS_LABEL"
CHILD_OF = "CHILD_OF"
TIP_OF = "TIP_OF"
MERGED_AS = "MERGED_AS"

NODE_INDEXES = ("sha", "path", "number", "name", "prov_run_id")
EDGE_INDEXES = ("run_id",)

_VERSION_NAMESPACE = uuid.NAMESPACE_URL


class GraphError(RuntimeError):
    """Base class for sink failures that keep the database handle safe."""


class GraphClosedError(GraphError):
    def __init__(self, operation: str) -> None:
        super().__init__(f"GraphSink is closed; cannot {operation}")
        self.operation = operation


class EntityWrite(FrozenModel):
    logical_id: str
    labels: tuple[str, ...]
    properties: dict[str, Any]
    valid_from: datetime
    origin: str
    origin_id: str
    event_id: str
    run_id: str
    info_horizon: datetime
    evidence_ids: tuple[str, ...] = ()


class RelationshipWrite(FrozenModel):
    edge_id: str
    edge_type: str
    source: str
    target: str
    properties: dict[str, Any]
    valid_from: datetime
    origin: str
    origin_id: str
    event_id: str
    run_id: str
    info_horizon: datetime
    evidence_ids: tuple[str, ...] = ()


class MappedGraph(FrozenModel):
    entities: tuple[EntityWrite, ...] = ()
    relationships: tuple[RelationshipWrite, ...] = ()
    skipped: tuple[str, ...] = ()


class WriteReport(FrozenModel):
    run_id: str
    entities_written: int = 0
    entities_duplicate: int = 0
    relationships_written: int = 0
    relationships_duplicate: int = 0


class VersionRecord(FrozenModel):
    version_id: str
    logical_id: str
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    operation: str = ""
    supersedes_version_id: str | None = None
    properties: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Mapping from adapter evidence to graph writes
# ---------------------------------------------------------------------------


def _provenance(write: EntityWrite | RelationshipWrite) -> dict[str, Any]:
    return {
        "prov_run_id": write.run_id,
        "prov_origin": write.origin,
        "prov_origin_id": write.origin_id,
        "prov_event_id": write.event_id,
        "prov_info_horizon": write.info_horizon.isoformat(),
        "prov_evidence_ids": sorted(write.evidence_ids),
    }


def map_git_index(
    index: GitIndex, ids: IdFactory, *, run_id: str
) -> MappedGraph:
    """Map commits, snapshots, files, and branches to graph writes."""

    entities: list[EntityWrite] = []
    relationships: list[RelationshipWrite] = []
    skipped: list[str] = []
    commit_times = {
        record.sha: record.committed_at
        for record in (*index.selection.selected_commits, *index.selection.boundary_commits)
    }
    in_scope = set(commit_times)

    for record in (*index.selection.selected_commits, *index.selection.boundary_commits):
        observed = record.committed_at
        logical_id = ids.commit(record.sha)
        entities.append(
            EntityWrite(
                logical_id=logical_id,
                labels=("Commit",),
                properties={
                    "sha": record.sha,
                    "message": record.message,
                    "author_name": record.author_name,
                    "author_email": record.author_email,
                    "committed_at": record.committed_at.isoformat(),
                    "parents": list(record.parents),
                    "is_merge": record.is_merge,
                    "tree_sha": record.tree_sha,
                    "source_url": record.source_url,
                },
                valid_from=record.committed_at,
                origin="git",
                origin_id=record.sha,
                event_id=stable_id("graph-write", ids.repository_id, "git-commit", record.sha),
                run_id=run_id,
                info_horizon=observed,
                evidence_ids=(ids.evidence("git", record.sha, record.source_url),),
            )
        )
        for parent in record.parents:
            if parent not in in_scope:
                skipped.append(f"parent-commit:{record.sha[:8]}->{parent[:8]} out of scope")
                continue
            relationships.append(
                RelationshipWrite(
                    edge_id=stable_id("edge", ids.repository_id, PARENT_COMMIT, record.sha, parent),
                    edge_type=PARENT_COMMIT,
                    source=logical_id,
                    target=ids.commit(parent),
                    properties={},
                    valid_from=record.committed_at,
                    origin="git",
                    origin_id=f"{record.sha}:{parent}",
                    event_id=stable_id(
                        "graph-write", ids.repository_id, "git-parent", record.sha, parent
                    ),
                    run_id=run_id,
                    info_horizon=observed,
                    evidence_ids=(ids.evidence("git", record.sha, record.source_url),),
                )
            )

    for entry in index.snapshots:
        committed_at = commit_times.get(entry.commit_sha)
        if committed_at is None:
            skipped.append(f"snapshot:{entry.snapshot_id} without commit time")
            continue
        snapshot_id = entry.snapshot_id
        entities.append(
            EntityWrite(
                logical_id=snapshot_id,
                labels=("Snapshot",),
                properties={
                    "commit_sha": entry.commit_sha,
                    "path": entry.path,
                    "object_id": entry.object_id,
                    "content_status": entry.content_status,
                    "is_baseline": entry.is_baseline,
                    "source_url": entry.source_url,
                },
                valid_from=committed_at,
                origin="git",
                origin_id=f"{entry.commit_sha}:{entry.path}",
                event_id=stable_id("graph-write", ids.repository_id, "git-snapshot", snapshot_id),
                run_id=run_id,
                info_horizon=committed_at,
                evidence_ids=(ids.evidence("git", entry.commit_sha, entry.source_url),),
            )
        )
        relationships.append(
            RelationshipWrite(
                edge_id=stable_id("edge", ids.repository_id, MEMBER_OF, snapshot_id),
                edge_type=MEMBER_OF,
                source=snapshot_id,
                target=ids.commit(entry.commit_sha),
                properties={},
                valid_from=committed_at,
                origin="git",
                origin_id=f"{entry.commit_sha}:{entry.path}:member",
                event_id=stable_id(
                    "graph-write", ids.repository_id, "git-member", snapshot_id
                ),
                run_id=run_id,
                info_horizon=committed_at,
                evidence_ids=(ids.evidence("git", entry.commit_sha, entry.source_url),),
            )
        )
        relationships.append(
            RelationshipWrite(
                edge_id=stable_id("edge", ids.repository_id, OF_FILE, snapshot_id),
                edge_type=OF_FILE,
                source=snapshot_id,
                target=ids.file(entry.path),
                properties={},
                valid_from=committed_at,
                origin="git",
                origin_id=f"{entry.commit_sha}:{entry.path}:file",
                event_id=stable_id("graph-write", ids.repository_id, "git-offile", snapshot_id),
                run_id=run_id,
                info_horizon=committed_at,
                evidence_ids=(),
            )
        )
        # Direct commit-to-file evidence: which files a commit touched.
        relationships.append(
            RelationshipWrite(
                edge_id=stable_id(
                    "edge", ids.repository_id, TOUCHES, entry.commit_sha, entry.path
                ),
                edge_type=TOUCHES,
                source=ids.commit(entry.commit_sha),
                target=ids.file(entry.path),
                properties={},
                valid_from=committed_at,
                origin="git",
                origin_id=f"{entry.commit_sha}:{entry.path}:touches",
                event_id=stable_id(
                    "graph-write", ids.repository_id, "git-touches",
                    entry.commit_sha, entry.path,
                ),
                run_id=run_id,
                info_horizon=committed_at,
                evidence_ids=(ids.evidence("git", entry.commit_sha, entry.source_url),),
            )
        )

    # One File entity per path, valid from its earliest observed commit.
    first_seen: dict[str, datetime] = {}
    for entry in index.snapshots:
        committed_at = commit_times.get(entry.commit_sha)
        if committed_at is None:
            continue
        previous = first_seen.get(entry.path)
        if previous is None or committed_at < previous:
            first_seen[entry.path] = committed_at
    for path, valid_from in sorted(first_seen.items()):
        file_id = ids.file(path)
        entities.append(
            EntityWrite(
                logical_id=file_id,
                labels=("File",),
                properties={"path": path},
                valid_from=valid_from,
                origin="git",
                origin_id=path,
                event_id=stable_id("graph-write", ids.repository_id, "git-file", path),
                run_id=run_id,
                info_horizon=valid_from,
                evidence_ids=(),
            )
        )

    for branch in index.branches:
        branch_id = ids.branch(branch.name)
        entities.append(
            EntityWrite(
                logical_id=branch_id,
                labels=("Branch",),
                properties={
                    "name": branch.name,
                    "ref": branch.ref,
                    "tip_sha": branch.tip_sha,
                    "contains_anchor": branch.contains_anchor,
                    "source": branch.source,
                },
                valid_from=index.selection.window.start,
                origin="git",
                origin_id=branch.name,
                event_id=stable_id("graph-write", ids.repository_id, "git-branch", branch.name),
                run_id=run_id,
                info_horizon=index.selection.window.start,
                evidence_ids=(),
            )
        )
        relationships.append(
            RelationshipWrite(
                edge_id=stable_id("edge", ids.repository_id, TIP_OF, branch.name, branch.tip_sha),
                edge_type=TIP_OF,
                source=branch_id,
                target=ids.commit(branch.tip_sha),
                properties={"contains_anchor": branch.contains_anchor},
                valid_from=index.selection.window.start,
                origin="git",
                origin_id=f"{branch.name}:{branch.tip_sha}",
                event_id=stable_id(
                    "graph-write", ids.repository_id, "git-tip", branch.name, branch.tip_sha
                ),
                run_id=run_id,
                info_horizon=index.selection.window.start,
                evidence_ids=(),
            )
        )

    return MappedGraph(
        entities=tuple(entities), relationships=tuple(relationships), skipped=tuple(skipped)
    )


def map_discovery(
    discovery: GitHubDiscovery, ids: IdFactory, *, run_id: str
) -> MappedGraph:
    """Map work items, labels, hierarchy, and merge evidence to writes."""

    entities: list[EntityWrite] = []
    relationships: list[RelationshipWrite] = []
    skipped: list[str] = []

    for item in discovery.work_items:
        logical_id = ids.work_item("github", item.number)
        entities.append(
            EntityWrite(
                logical_id=logical_id,
                labels=("WorkItem",),
                properties={
                    "number": item.number,
                    "kind": item.kind,
                    "title": item.title,
                    "state": item.state,
                    "author": item.author,
                    "created_at": item.created_at.isoformat(),
                    "updated_at": item.updated_at.isoformat(),
                    "closed_at": item.closed_at.isoformat() if item.closed_at else None,
                    "url": item.url,
                    "labels": list(item.labels),
                    "source_hash": item.source_hash,
                },
                valid_from=item.created_at,
                origin="github",
                origin_id=str(item.number),
                event_id=stable_id(
                    "graph-write", ids.repository_id, "github-item", str(item.number)
                ),
                run_id=run_id,
                info_horizon=item.observed_at,
                evidence_ids=(ids.evidence("github", str(item.number), item.url),),
            )
        )

    for label in discovery.provider_labels:
        entities.append(
            EntityWrite(
                logical_id=ids.provider_label("github", label.name),
                labels=("ProviderLabel",),
                properties={
                    "name": label.name,
                    "color": label.color,
                    "description": label.description,
                    "provider": "github",
                },
                valid_from=discovery.observed_at,
                origin="github",
                origin_id=f"label:{label.name}",
                event_id=stable_id(
                    "graph-write", ids.repository_id, "github-label", label.name
                ),
                run_id=run_id,
                info_horizon=discovery.observed_at,
                evidence_ids=(),
            )
        )

    for assignment in discovery.label_assignments:
        entities.append(
            EntityWrite(
                logical_id=ids.provider_label("github", assignment.label_name),
                labels=("ProviderLabel",),
                properties={"name": assignment.label_name, "provider": "github"},
                valid_from=assignment.observed_at,
                origin="github",
                origin_id=f"label:{assignment.label_name}",
                event_id=stable_id(
                    "graph-write", ids.repository_id, "github-label", assignment.label_name
                ),
                run_id=run_id,
                info_horizon=assignment.observed_at,
                evidence_ids=(assignment.evidence_id,),
            )
        )
        relationships.append(
            RelationshipWrite(
                edge_id=stable_id(
                    "edge", ids.repository_id, HAS_LABEL,
                    str(assignment.work_item_number), assignment.label_name,
                ),
                edge_type=HAS_LABEL,
                source=ids.work_item("github", assignment.work_item_number),
                target=ids.provider_label("github", assignment.label_name),
                properties={},
                valid_from=assignment.observed_at,
                origin="github",
                origin_id=f"{assignment.work_item_number}:{assignment.label_name}",
                event_id=stable_id(
                    "graph-write", ids.repository_id, "github-has-label",
                    str(assignment.work_item_number), assignment.label_name,
                ),
                run_id=run_id,
                info_horizon=assignment.observed_at,
                evidence_ids=(assignment.evidence_id,),
            )
        )

    for edge in discovery.hierarchy_edges:
        if not edge.in_scope:
            skipped.append(
                f"hierarchy:{edge.parent_number}->{edge.child_number} out of scope"
            )
            continue
        relationships.append(
            RelationshipWrite(
                edge_id=edge.evidence_id,
                edge_type=CHILD_OF,
                source=ids.work_item("github", edge.child_number),
                target=ids.work_item("github", edge.parent_number),
                properties={"relation": edge.relation},
                valid_from=discovery.observed_at,
                origin="github",
                origin_id=f"{edge.parent_number}:{edge.child_number}",
                event_id=stable_id(
                    "graph-write", ids.repository_id, "github-child",
                    str(edge.parent_number), str(edge.child_number),
                ),
                run_id=run_id,
                info_horizon=discovery.observed_at,
                evidence_ids=(edge.evidence_id,),
            )
        )

    pull_requests = {pull.number: pull for pull in discovery.pull_requests}
    for item in discovery.work_items:
        pull = pull_requests.get(item.number)
        if pull is None or not pull.merged or not pull.merge_commit_sha:
            continue
        merged_at = pull.merged_at or discovery.observed_at
        relationships.append(
            RelationshipWrite(
                edge_id=stable_id(
                    "edge", ids.repository_id, MERGED_AS,
                    str(item.number), pull.merge_commit_sha,
                ),
                edge_type=MERGED_AS,
                source=ids.work_item("github", item.number),
                target=ids.commit(pull.merge_commit_sha),
                properties={
                    "merge_commit_sha": pull.merge_commit_sha,
                    "head_sha": pull.head_sha,
                    "base_sha": pull.base_sha,
                },
                valid_from=merged_at,
                origin="github",
                origin_id=f"{item.number}:merged",
                event_id=stable_id(
                    "graph-write", ids.repository_id, "github-merged", str(item.number)
                ),
                run_id=run_id,
                info_horizon=merged_at,
                evidence_ids=(ids.evidence("github", str(item.number), pull.url),),
            )
        )

    return MappedGraph(
        entities=tuple(entities), relationships=tuple(relationships), skipped=tuple(skipped)
    )


def map_registry(
    registry: TopicRegistry, ids: IdFactory, *, run_id: str
) -> MappedGraph:
    """Map derived topics to ``Topic`` entities with lifecycle provenance."""

    entities: list[EntityWrite] = []
    for topic in registry.topics:
        entities.append(
            EntityWrite(
                logical_id=topic.topic_id,
                labels=("Topic",),
                properties={
                    "name": topic.name,
                    "aliases": list(topic.aliases),
                    "description": topic.description,
                    "created_by": topic.created_by,
                    "merged_into": topic.merged_into,
                    "supersedes": list(topic.supersedes),
                },
                valid_from=topic.created_at,
                origin="labeling",
                origin_id=topic.topic_id,
                event_id=stable_id(
                    "graph-write", ids.repository_id, "topic", topic.topic_id
                ),
                run_id=run_id,
                info_horizon=topic.created_at,
                evidence_ids=(),
            )
        )
    return MappedGraph(entities=tuple(entities))


def map_topic_assignments(
    assignments: Sequence[TopicAssignment], ids: IdFactory, *, run_id: str
) -> MappedGraph:
    """Map derived topic assignments to ``ABOUT`` entity-topic edges.

    Subject IDs follow the layers that produced them: snapshot occurrences
    stay snapshot IDs (the graph holds ``Snapshot`` nodes under those IDs)
    while ``workitem:github:N`` subjects resolve to ``WorkItem`` nodes.
    Provider-label preservation assignments carry no topic and are skipped.
    """

    relationships: list[RelationshipWrite] = []
    skipped: list[str] = []
    for assignment in assignments:
        if assignment.kind != "derived" or assignment.topic_id is None:
            skipped.append(f"assignment:{assignment.subject_id} has no derived topic")
            continue
        match = re.fullmatch(r"workitem:github:(\d+)", assignment.subject_id)
        source = (
            ids.work_item("github", int(match.group(1)))
            if match is not None
            else assignment.subject_id
        )
        horizon = assignment.result.provenance.info_horizon
        relationships.append(
            RelationshipWrite(
                edge_id=stable_id(
                    "edge", ids.repository_id, ABOUT,
                    assignment.subject_id, assignment.topic_id,
                ),
                edge_type=ABOUT,
                source=source,
                target=assignment.topic_id,
                properties={
                    "action": assignment.action,
                    "confidence": assignment.confidence,
                    "rationale": assignment.rationale,
                },
                valid_from=horizon,
                origin="labeling",
                origin_id=f"{assignment.subject_id}:{assignment.topic_id}",
                event_id=stable_id(
                    "graph-write", ids.repository_id, "topic-about",
                    assignment.subject_id, assignment.topic_id,
                ),
                run_id=run_id,
                info_horizon=horizon,
                evidence_ids=tuple(
                    ref.evidence_id for ref in assignment.result.evidence
                ),
            )
        )
    return MappedGraph(
        relationships=tuple(relationships), skipped=tuple(skipped)
    )


# ---------------------------------------------------------------------------
# Single-writer sink
# ---------------------------------------------------------------------------


def _version_id(logical_id: str, valid_from: datetime, event_id: str, fingerprint: str) -> str:
    material = "|".join(
        ["graph-v1", logical_id, valid_from.isoformat(), fingerprint, event_id]
    )
    return str(uuid.uuid5(_VERSION_NAMESPACE, material))


def _fingerprint(properties: dict[str, Any]) -> str:
    import json

    canonical = json.dumps(properties, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _instant_to_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return ensure_utc(value)
    to_datetime = getattr(value, "to_datetime", None)
    if callable(to_datetime):
        return ensure_utc(to_datetime())
    return None


def _valid_interval(valid_from: datetime) -> tuple[datetime, None]:
    return (ensure_utc(valid_from), None)


class GraphSink(GraphSinkProtocol):
    """Own the GestaltDB handle and serialize every temporal write."""

    def __init__(self, db: GraphDB, *, run_id: str) -> None:
        self._db = db
        self._run_id = run_id
        self._closed = False

    @classmethod
    def create(
        cls, settings: GraphSettings | None, *, run_id: str, overwrite: bool | None = None
    ) -> "GraphSink":
        resolved = settings or GraphSettings()
        db = GraphDB.create(
            resolved.path,
            backend=resolved.backend,
            overwrite=resolved.overwrite if overwrite is None else overwrite,
            backend_options=resolved.backend_options(),
        )
        sink = cls(db, run_id=run_id)
        sink.ensure_indexes()
        return sink

    @classmethod
    def open(
        cls, settings: GraphSettings | None, *, run_id: str
    ) -> "GraphSink":
        resolved = settings or GraphSettings()
        db = GraphDB.open(resolved.path, backend_options=resolved.backend_options())
        sink = cls(db, run_id=run_id)
        sink.ensure_indexes()
        return sink

    def __enter__(self) -> "GraphSink":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def is_closed(self) -> bool:
        return self._closed

    def _ensure_open(self, operation: str) -> None:
        if self._closed:
            raise GraphClosedError(operation)

    # -- schema -----------------------------------------------------------
    def ensure_indexes(self) -> None:
        """Declare explicit property indexes; safe to call on every open."""

        self._ensure_open("declare indexes")
        statistics = self._db.index_statistics()
        for name in NODE_INDEXES:
            if name not in statistics.get("indexed_node_properties", ()):
                self._db.create_node_property_index(name)
        for name in EDGE_INDEXES:
            if name not in statistics.get("indexed_edge_properties", ()):
                self._db.create_edge_property_index(name)

    def stale_indexes(self) -> tuple[str, ...]:
        self._ensure_open("inspect indexes")
        return tuple(self._db.stale_indexes())

    def rebuild_deferred(self) -> dict[str, int]:
        self._ensure_open("rebuild indexes")
        return dict(self._db.rebuild_deferred_indexes())

    def _ensure_ready(self) -> None:
        """Rebuild deferred indexes before any read.

        The driver refuses temporal and indexed queries while the temporal
        index is stale, so every read path funnels through here.
        """

        if self.stale_indexes():
            self.rebuild_deferred()

    # -- writes -----------------------------------------------------------
    def write_entities(
        self,
        entities: Iterable[EntityWrite],
        *,
        analysis_run_id: str,
        defer_indexes: bool = False,
    ) -> WriteReport:
        self._ensure_open("write entities")
        self._check_run(analysis_run_id)
        written = duplicates = 0
        mode = IndexMaintenanceMode.DEFER if defer_indexes else IndexMaintenanceMode.MAINTAIN
        for write in entities:
            if self._write_entity(write, mode):
                written += 1
            else:
                duplicates += 1
        return WriteReport(
            run_id=self._run_id,
            entities_written=written,
            entities_duplicate=duplicates,
        )

    def write_relationships(
        self,
        relationships: Iterable[RelationshipWrite],
        *,
        analysis_run_id: str,
        defer_indexes: bool = False,
    ) -> WriteReport:
        self._ensure_open("write relationships")
        self._check_run(analysis_run_id)
        written = duplicates = 0
        mode = IndexMaintenanceMode.DEFER if defer_indexes else IndexMaintenanceMode.MAINTAIN
        for write in relationships:
            if self._write_relationship(write, mode):
                written += 1
            else:
                duplicates += 1
        return WriteReport(
            run_id=self._run_id,
            relationships_written=written,
            relationships_duplicate=duplicates,
        )

    def _check_run(self, analysis_run_id: str) -> None:
        if analysis_run_id != self._run_id:
            raise GraphError(
                f"Single writer holds run {self._run_id!r}; "
                f"refusing writes for run {analysis_run_id!r}"
            )

    def _write_entity(self, write: EntityWrite, mode: IndexMaintenanceMode) -> bool:
        properties = {**write.properties, **_provenance(write)}
        version_id = _version_id(
            write.logical_id, write.valid_from, write.event_id, _fingerprint(properties)
        )
        node = Node(node_id=write.logical_id, labels=list(write.labels), properties=properties)
        try:
            self._db.put_node_version(
                node,
                valid=_valid_interval(write.valid_from),
                version_id=version_id,
                metadata={"run_id": self._run_id, "event_id": write.event_id},
                index_mode=mode,
            )
        except TemporalVersionError as exc:
            if "already exists" not in str(exc):
                raise
            existing = self._db.get_node_version(version_id)
            if existing is None or dict(existing.node.properties) != properties:
                raise GraphError(
                    f"Conflicting write for version {version_id} "
                    f"({write.logical_id}); stored payload differs"
                ) from exc
            self._db.put_node(node)
            return False
        self._db.put_node(node)
        return True

    def _write_relationship(self, write: RelationshipWrite, mode: IndexMaintenanceMode) -> bool:
        properties = {"type": write.edge_type, **write.properties, **_provenance(write)}
        version_id = _version_id(
            write.edge_id, write.valid_from, write.event_id, _fingerprint(properties)
        )
        edge = Edge(
            edge_id=write.edge_id,
            source=write.source,
            target=write.target,
            properties=properties,
        )
        try:
            self._db.put_edge_version(
                edge,
                valid=_valid_interval(write.valid_from),
                version_id=version_id,
                metadata={"run_id": self._run_id, "event_id": write.event_id},
                index_mode=mode,
            )
        except TemporalVersionError as exc:
            if "already exists" not in str(exc):
                raise
            existing = self._db.get_edge_version(version_id)
            if existing is None or dict(existing.edge.properties) != properties:
                raise GraphError(
                    f"Conflicting write for version {version_id} "
                    f"({write.edge_id}); stored payload differs"
                ) from exc
            self._db.put_edge(edge)
            return False
        self._db.put_edge(edge)
        return True

    def correct_entity(
        self,
        write: EntityWrite,
        *,
        supersedes_version_id: str,
        valid_from: datetime | None = None,
        defer_indexes: bool = False,
    ) -> str:
        """Append a correction version; the superseded version is retained."""

        self._ensure_open("correct an entity")
        self._check_run(write.run_id)
        properties = {**write.properties, **_provenance(write)}
        version_id = _version_id(
            write.logical_id,
            valid_from or write.valid_from,
            write.event_id,
            _fingerprint(properties),
        )
        node = Node(node_id=write.logical_id, labels=list(write.labels), properties=properties)
        mode = IndexMaintenanceMode.DEFER if defer_indexes else IndexMaintenanceMode.MAINTAIN
        self._db.correct_node_version(
            node,
            supersedes_version_id=supersedes_version_id,
            valid=_valid_interval(valid_from or write.valid_from),
            version_id=version_id,
            metadata={"run_id": self._run_id, "event_id": write.event_id},
            index_mode=mode,
        )
        self._db.put_node(node)
        return version_id

    def correct_relationship(
        self,
        write: RelationshipWrite,
        *,
        supersedes_version_id: str,
        valid_from: datetime | None = None,
        defer_indexes: bool = False,
    ) -> str:
        self._ensure_open("correct a relationship")
        self._check_run(write.run_id)
        properties = {"type": write.edge_type, **write.properties, **_provenance(write)}
        version_id = _version_id(
            write.edge_id,
            valid_from or write.valid_from,
            write.event_id,
            _fingerprint(properties),
        )
        edge = Edge(
            edge_id=write.edge_id,
            source=write.source,
            target=write.target,
            properties=properties,
        )
        mode = IndexMaintenanceMode.DEFER if defer_indexes else IndexMaintenanceMode.MAINTAIN
        self._db.correct_edge_version(
            edge,
            supersedes_version_id=supersedes_version_id,
            valid=_valid_interval(valid_from or write.valid_from),
            version_id=version_id,
            metadata={"run_id": self._run_id, "event_id": write.event_id},
            index_mode=mode,
        )
        self._db.put_edge(edge)
        return version_id

    # -- historical and current reads -------------------------------------
    # Every read funnels through _ensure_ready: the driver refuses temporal
    # and indexed queries while deferred writes left indexes stale.
    def node_as_of(self, logical_id: str, valid_time: datetime) -> dict[str, Any] | None:
        self._ensure_open("read a node version")
        self._ensure_ready()
        found = self._db.get_node_as_of(
            logical_id, valid_time=ensure_utc(valid_time)
        )
        return dict(found.node.properties) if found is not None else None

    def edge_as_of(self, edge_id: str, valid_time: datetime) -> dict[str, Any] | None:
        self._ensure_open("read an edge version")
        self._ensure_ready()
        found = self._db.get_edge_as_of(edge_id, valid_time=ensure_utc(valid_time))
        return dict(found.edge.properties) if found is not None else None

    def node_history(self, logical_id: str) -> tuple[VersionRecord, ...]:
        """Return every retained version, oldest system-time first."""

        self._ensure_open("read node history")
        self._ensure_ready()
        records: list[VersionRecord] = []
        for version in self._db.iter_node_versions(logical_id):
            valid = getattr(version, "valid", None)
            records.append(
                VersionRecord(
                    version_id=version.version_id,
                    logical_id=version.logical_id,
                    valid_from=_instant_to_datetime(getattr(valid, "start", None)),
                    valid_to=_instant_to_datetime(getattr(valid, "end", None)),
                    operation=str(getattr(version, "operation", "")),
                    supersedes_version_id=getattr(version, "supersedes_version_id", None),
                    properties=dict(version.node.properties),
                )
            )
        return tuple(records)

    def edge_history(self, edge_id: str) -> tuple[VersionRecord, ...]:
        self._ensure_open("read edge history")
        self._ensure_ready()
        records: list[VersionRecord] = []
        for version in self._db.iter_edge_versions(edge_id):
            valid = getattr(version, "valid", None)
            records.append(
                VersionRecord(
                    version_id=version.version_id,
                    logical_id=version.logical_id,
                    valid_from=_instant_to_datetime(getattr(valid, "start", None)),
                    valid_to=_instant_to_datetime(getattr(valid, "end", None)),
                    operation=str(getattr(version, "operation", "")),
                    supersedes_version_id=getattr(version, "supersedes_version_id", None),
                    properties=dict(version.edge.properties),
                )
            )
        return tuple(records)

    def current_node(self, logical_id: str) -> dict[str, Any] | None:
        """Return the rebuildable current projection for one entity."""

        self._ensure_open("read the current projection")
        found = self._db.get_node(self._db.node_key_to_bytes(logical_id))
        return dict(found.properties) if found is not None else None

    def find_nodes(
        self, label: str, property_name: str, value: Any
    ) -> list[dict[str, Any]]:
        """Indexed exact-match read; deferred indexes rebuild first."""

        self._ensure_open("run an indexed node read")
        self._ensure_ready()
        return [
            dict(node.properties)
            for node in self._db.nodes_by_label_property(label, property_name, value)
        ]

    def neighbors(
        self, logical_id: str, edge_type: str, direction: str = "out"
    ) -> list[str]:
        """Traverse the current projection; used for reachability queries."""

        self._ensure_open("traverse the current projection")
        self._ensure_ready()
        return [
            node.decode("utf-8") if isinstance(node, bytes) else str(node)
            for node in self._db.neighbors_by_edge_type(logical_id, edge_type, direction)
        ]

    def label_counts(self, labels: Sequence[str] | None = None) -> dict[str, int]:
        self._ensure_open("count entities")
        self._ensure_ready()
        if labels is None:
            labels = (
                "Commit", "File", "Snapshot", "WorkItem", "ProviderLabel", "Branch",
                "Topic",
            )
        return {label: self._db.count_nodes_by_label(label) for label in labels}

    def edge_counts(
        self, edge_types: Sequence[str] | None = None
    ) -> dict[str, int]:
        self._ensure_open("count relationships")
        self._ensure_ready()
        if edge_types is None:
            edge_types = (
                ABOUT, PARENT_COMMIT, MEMBER_OF, OF_FILE, TOUCHES, HAS_LABEL,
                CHILD_OF, TIP_OF, MERGED_AS, "MENTIONS", "INTENDS_TO_RESOLVE",
                "CLOSED_BY", "RELATED_TO",
            )
        return {edge_type: self._db.count_edges_by_type(edge_type) for edge_type in edge_types}

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._db.close()
