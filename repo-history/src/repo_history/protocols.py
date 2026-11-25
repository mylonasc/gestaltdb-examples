"""Small extension interfaces; plugins return data and never own the graph writer."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, Protocol, Sequence, runtime_checkable

from .events import IngestEvent
from .models import EvidenceRef, RepositorySpec


@runtime_checkable
class RepositorySource(Protocol):
    @property
    def repository(self) -> RepositorySpec: ...

    def discover(self) -> Iterable[IngestEvent]: ...

    def open_snapshot(self, commit_sha: str, path: str) -> bytes | None: ...


@runtime_checkable
class WorkItemProvider(Protocol):
    def discover(self, repository: RepositorySpec) -> Iterable[IngestEvent]: ...


@runtime_checkable
class DocumentParser(Protocol):
    name: str

    def parse(self, content: bytes, *, path: str) -> Any: ...


@runtime_checkable
class CodeParser(Protocol):
    name: str

    def parse(self, content: bytes, *, path: str) -> Any: ...


@runtime_checkable
class Labeler(Protocol):
    name: str
    version: str

    def label(self, *, text: str, context: Any) -> Sequence[Any]: ...


@runtime_checkable
class RelationshipResolver(Protocol):
    name: str
    version: str

    def resolve(self, *, subject: Any, candidates: Sequence[Any], context: Any) -> Sequence[Any]: ...


@runtime_checkable
class GraphSink(Protocol):
    def write_entities(self, entities: Iterable[Any], *, analysis_run_id: str) -> Any: ...

    def write_relationships(self, relationships: Iterable[Any], *, analysis_run_id: str) -> Any: ...

    def close(self) -> None: ...


@runtime_checkable
class Clock(Protocol):
    def now(self) -> datetime: ...


def evidence_ids(refs: Sequence[EvidenceRef]) -> tuple[str, ...]:
    return tuple(sorted({ref.evidence_id for ref in refs}))
