"""Temporal replay and resumable orchestration over normalized events.

Every adapter (Git history, GitHub discovery, content extraction) emits
:class:`IngestEvent` records; this module turns those streams into a durable,
replayable history:

- :class:`EventJournal` persists events and coverage gaps as an append-only
  JSONL log with a manifest header, so an interrupted run can resume exactly
  where it stopped.
- :class:`CheckpointStore` records completed stages together with the budget
  ledger counters and a binding to the journal prefix, so retries and resumes
  continue spending the same budgets instead of resetting them.
- :class:`TemporalCoordinator` merges streams into a stable order, resolves
  conflicting observations causally, defers events whose dependencies have not
  arrived yet, and gates enrichment by information horizon.

Horizon semantics are the temporal firewall for later slices.  Immutable source
facts (Git commits) carry ``info_horizon == event_time``: the evidence
describes exactly what was true when the commit was made.  Mutable remote
observations (current issue bodies, labels, comments) carry
``info_horizon == observed_at``: they describe the provider state at fetch
time and must never be backdated.  A decision scoped at time ``T`` may only
use events with ``info_horizon <= T``, so future labels, docs, or async
results cannot seed past topic decisions.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator

from .budgets import BudgetLedger
from .events import EventKind, IngestEvent, now_utc
from .git import GitSelection
from .ids import IdFactory, stable_id
from .models import (
    Budgets,
    CoverageGap,
    CoverageStatus,
    FrozenModel,
    OrchestrationSettings,
    RepositorySpec,
    RunManifest,
    ensure_utc,
)

__all__ = [
    "Bootstrap",
    "Checkpoint",
    "CheckpointStore",
    "EventJournal",
    "JournalCorrupted",
    "JournalDiverged",
    "JournalManifest",
    "JournalMismatch",
    "JournalSnapshot",
    "OrchestrationError",
    "ReplayResult",
    "StageRunner",
    "TemporalCoordinator",
    "git_selection_events",
    "software_version",
]


def software_version() -> str:
    """Return the installed package version without importing providers."""

    try:
        return version("repo-history")
    except PackageNotFoundError:
        return "0.0.0+unknown"


class OrchestrationError(RuntimeError):
    """Base class for journal, checkpoint, and replay failures."""


class JournalCorrupted(OrchestrationError):
    def __init__(self, path: str | Path, lineno: int, detail: str) -> None:
        super().__init__(f"Journal {path} line {lineno} is unreadable: {detail}")
        self.path = str(path)
        self.lineno = lineno


class JournalMismatch(OrchestrationError):
    """The existing journal belongs to a different run configuration."""


class JournalDiverged(OrchestrationError):
    """The journal no longer extends the prefix a checkpoint was bound to."""


class CheckpointCorrupted(OrchestrationError):
    def __init__(self, path: str | Path, detail: str) -> None:
        super().__init__(f"Checkpoint {path} is unreadable: {detail}")
        self.path = str(path)


# ---------------------------------------------------------------------------
# Durable event journal
# ---------------------------------------------------------------------------


class JournalManifest(FrozenModel):
    run_id: str
    repository_id: str
    repository_url: str
    anchor: str
    window_start: datetime
    window_end: datetime
    config_fingerprint: str
    software_version: str
    created_at: datetime

    @field_validator("window_start", "window_end", "created_at")
    @classmethod
    def utc(cls, value: datetime) -> datetime:
        return ensure_utc(value)


class JournalSnapshot(FrozenModel):
    manifest: JournalManifest
    events: tuple[IngestEvent, ...] = ()
    gaps: tuple[CoverageGap, ...] = ()


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


class EventJournal:
    """Append-only JSONL log of normalized events and coverage gaps."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    @classmethod
    def create(cls, path: str | Path, manifest: JournalManifest) -> "EventJournal":
        journal = cls(path)
        if journal._path.exists():
            raise JournalMismatch(
                f"Journal {journal._path} already exists; refusing to overwrite it"
            )
        journal._path.parent.mkdir(parents=True, exist_ok=True)
        journal._write_line(
            {"record": "manifest", "manifest": manifest.model_dump(mode="json")}
        )
        return journal

    @classmethod
    def open_existing(
        cls,
        path: str | Path,
        *,
        repository_id: str,
        config_fingerprint: str,
    ) -> "EventJournal":
        journal = cls(path)
        snapshot = journal.read()
        manifest = snapshot.manifest
        if manifest.repository_id != repository_id:
            raise JournalMismatch(
                f"Journal {journal._path} belongs to repository "
                f"{manifest.repository_id!r}, not {repository_id!r}"
            )
        if manifest.config_fingerprint != config_fingerprint:
            raise JournalMismatch(
                f"Journal {journal._path} was written with a different configuration"
            )
        return journal

    def append(
        self,
        events: Iterable[IngestEvent],
        gaps: Iterable[CoverageGap] = (),
    ) -> dict[str, int]:
        """Append events and gaps; byte-identical duplicates are skipped."""

        known = {self._digest(event) for event in self.read().events}
        appended_events = 0
        appended_gaps = 0
        with self._path.open("a", encoding="utf-8") as handle:
            for event in events:
                digest = self._digest(event)
                if digest in known:
                    continue
                known.add(digest)
                handle.write(
                    _canonical({"record": "event", "event": event.model_dump(mode="json")})
                    + "\n"
                )
                appended_events += 1
            for gap in gaps:
                handle.write(
                    _canonical({"record": "gap", "gap": gap.model_dump(mode="json")}) + "\n"
                )
                appended_gaps += 1
            handle.flush()
        return {"events": appended_events, "gaps": appended_gaps}

    def read(self) -> JournalSnapshot:
        manifest: JournalManifest | None = None
        events: list[IngestEvent] = []
        gaps: list[CoverageGap] = []
        try:
            text = self._path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise JournalCorrupted(self._path, 0, "journal file is missing") from exc
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError as exc:
                raise JournalCorrupted(self._path, lineno, str(exc)) from exc
            try:
                kind = record["record"]
                if kind == "manifest":
                    if manifest is not None:
                        raise ValueError("duplicate manifest record")
                    manifest = JournalManifest.model_validate(record["manifest"])
                elif kind == "event":
                    events.append(IngestEvent.model_validate(record["event"]))
                elif kind == "gap":
                    gaps.append(CoverageGap.model_validate(record["gap"]))
                else:
                    raise ValueError(f"unknown record {kind!r}")
            except (KeyError, TypeError, ValueError) as exc:
                raise JournalCorrupted(self._path, lineno, str(exc)) from exc
        if manifest is None:
            raise JournalCorrupted(self._path, 0, "journal has no manifest record")
        return JournalSnapshot(manifest=manifest, events=tuple(events), gaps=tuple(gaps))

    def _write_line(self, record: dict[str, Any]) -> None:
        with self._path.open("w", encoding="utf-8") as handle:
            handle.write(_canonical(record) + "\n")
            handle.flush()

    @staticmethod
    def _digest(event: IngestEvent) -> tuple[str, str]:
        return (event.event_id, _canonical(event.model_dump(mode="json")))


# ---------------------------------------------------------------------------
# Checkpoints bound to the journal prefix and the budget ledger
# ---------------------------------------------------------------------------


class Checkpoint(FrozenModel):
    run_id: str
    completed_stages: tuple[str, ...] = ()
    commits: int = Field(default=0, ge=0)
    work_items: int = Field(default=0, ge=0)
    api_requests: int = Field(default=0, ge=0)
    text_bytes: int = Field(default=0, ge=0)
    llm_requests: int = Field(default=0, ge=0)
    llm_input_tokens: int = Field(default=0, ge=0)
    llm_output_tokens: int = Field(default=0, ge=0)
    estimated_input_tokens: int = Field(default=0, ge=0)
    estimated_output_tokens: int = Field(default=0, ge=0)
    exhausted: dict[str, str] = Field(default_factory=dict)
    journal_events: int = Field(default=0, ge=0)
    journal_digest: str = ""
    updated_at: datetime

    @field_validator("updated_at")
    @classmethod
    def utc(cls, value: datetime) -> datetime:
        return ensure_utc(value)

    def restore_ledger(self, limits: Budgets) -> BudgetLedger:
        """Rebuild a ledger that continues spending from this checkpoint."""

        ledger = BudgetLedger(limits)
        ledger.commits = self.commits
        ledger.work_items = self.work_items
        ledger.api_requests = self.api_requests
        ledger.text_bytes = self.text_bytes
        ledger.llm_requests = self.llm_requests
        ledger.llm_input_tokens = self.llm_input_tokens
        ledger.llm_output_tokens = self.llm_output_tokens
        ledger.estimated_input_tokens = self.estimated_input_tokens
        ledger.estimated_output_tokens = self.estimated_output_tokens
        ledger.exhausted = dict(self.exhausted)
        return ledger


def journal_digest(events: Iterable[IngestEvent]) -> str:
    """Hash the ordered event IDs so checkpoints bind to a journal prefix."""

    digest = hashlib.sha256()
    for event in events:
        digest.update(event.event_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


class CheckpointStore:
    """Atomically persisted stage progress bound to a journal prefix."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> Checkpoint | None:
        if not self._path.is_file():
            return None
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            return Checkpoint.model_validate(payload)
        except (OSError, ValueError) as exc:
            raise CheckpointCorrupted(self._path, str(exc)) from exc

    def save(self, checkpoint: Checkpoint) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(".tmp")
        temporary.write_text(
            _canonical(checkpoint.model_dump(mode="json")), encoding="utf-8"
        )
        temporary.replace(self._path)

    def record(
        self,
        run_id: str,
        stage: str,
        ledger: BudgetLedger,
        snapshot: JournalSnapshot,
    ) -> Checkpoint:
        """Mark ``stage`` complete against the current journal prefix."""

        if snapshot.manifest.run_id != run_id:
            raise JournalMismatch(
                f"Journal run {snapshot.manifest.run_id!r} does not match "
                f"checkpoint run {run_id!r}"
            )
        previous = self.load()
        if previous is not None:
            if previous.run_id != run_id:
                raise JournalMismatch(
                    f"Checkpoint run {previous.run_id!r} does not match run {run_id!r}"
                )
            prefix = snapshot.events[: previous.journal_events]
            if journal_digest(prefix) != previous.journal_digest:
                raise JournalDiverged(
                    f"Journal {snapshot.manifest.run_id!r} no longer extends the "
                    f"prefix checkpointed after stage {previous.completed_stages[-1]!r}"
                    if previous.completed_stages
                    else f"Journal {snapshot.manifest.run_id!r} diverged from its checkpoint"
                )
        completed = previous.completed_stages if previous is not None else ()
        if stage not in completed:
            completed = (*completed, stage)
        checkpoint = Checkpoint(
            run_id=run_id,
            completed_stages=completed,
            commits=ledger.commits,
            work_items=ledger.work_items,
            api_requests=ledger.api_requests,
            text_bytes=ledger.text_bytes,
            llm_requests=ledger.llm_requests,
            llm_input_tokens=ledger.llm_input_tokens,
            llm_output_tokens=ledger.llm_output_tokens,
            estimated_input_tokens=ledger.estimated_input_tokens,
            estimated_output_tokens=ledger.estimated_output_tokens,
            exhausted=dict(ledger.exhausted),
            journal_events=len(snapshot.events),
            journal_digest=journal_digest(snapshot.events),
            updated_at=now_utc(),
        )
        self.save(checkpoint)
        return checkpoint


# ---------------------------------------------------------------------------
# Git selection events
# ---------------------------------------------------------------------------


def _git_event_id(ids: IdFactory, scope: str, sha: str) -> str:
    return stable_id("git-event", ids.repository_id, scope, sha.lower())


def git_selection_events(
    selection: GitSelection,
    ids: IdFactory,
    *,
    observed_at: datetime,
) -> tuple[IngestEvent, ...]:
    """Emit baseline and commit events for a bounded Git selection.

    Boundary commits become ``BASELINE`` events (repository state at the
    window edge); selected commits become ``COMMIT`` events depending on
    their in-scope parents.  Commits are immutable source facts, so the
    information horizon equals the commit time itself.
    """

    observed_at = ensure_utc(observed_at)
    in_scope = {commit.sha: "commit" for commit in selection.selected_commits}
    in_scope.update({commit.sha: "baseline" for commit in selection.boundary_commits})
    skew = {
        item.commit_sha: item.delta_seconds for item in selection.clock_skews
    }

    events: list[IngestEvent] = []
    for commit in (*selection.boundary_commits, *selection.selected_commits):
        scope = in_scope[commit.sha]
        kind = EventKind.BASELINE if scope == "baseline" else EventKind.COMMIT
        dependencies = tuple(
            _git_event_id(ids, in_scope[parent], parent)
            for parent in commit.parents
            if parent in in_scope
        )
        missing_parents = tuple(
            parent for parent in commit.parents if parent not in in_scope
        )
        payload: dict[str, Any] = {
            "sha": commit.sha,
            "parents": list(commit.parents),
            "tree_sha": commit.tree_sha,
            "message": commit.message,
            "author": {"name": commit.author_name, "email": commit.author_email},
            "committer": {
                "name": commit.committer_name,
                "email": commit.committer_email,
            },
            "authored_at": commit.authored_at.isoformat(),
            "committed_at": commit.committed_at.isoformat(),
            "is_merge": commit.is_merge,
            "comparison": "root-empty-tree" if not commit.parents else "first-parent",
            "missing_parents": list(missing_parents),
            "source_url": commit.source_url,
        }
        if commit.sha in skew:
            payload["clock_skew_seconds"] = skew[commit.sha]
        events.append(
            IngestEvent(
                event_id=_git_event_id(ids, scope, commit.sha),
                kind=kind,
                source="git",
                source_id=f"{scope}:{commit.sha}",
                event_time=commit.committed_at,
                observed_at=observed_at,
                info_horizon=commit.committed_at,
                payload=payload,
                dependencies=dependencies,
            )
        )
    return tuple(sorted(events, key=lambda event: event.order_key))


# ---------------------------------------------------------------------------
# Temporal replay
# ---------------------------------------------------------------------------


class ReplayResult(FrozenModel):
    ordered: tuple[IngestEvent, ...] = ()
    effective: tuple[IngestEvent, ...] = ()
    superseded: tuple[str, ...] = ()
    deferred: tuple[str, ...] = ()
    gaps: tuple[CoverageGap, ...] = ()


class Bootstrap(FrozenModel):
    baseline: tuple[IngestEvent, ...] = ()
    ordered: tuple[IngestEvent, ...] = ()


class TemporalCoordinator:
    """Merge adapter streams into one deterministic, replayable history."""

    def __init__(
        self,
        repository: RepositorySpec,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.repository = repository
        self.ids = IdFactory(repository.url)
        self._clock = clock or now_utc

    def collect(self, events: Iterable[IngestEvent]) -> ReplayResult:
        """Order events stably, resolve conflicts, and defer unready events."""

        ordered = tuple(sorted(events, key=lambda event: event.order_key))
        winners: dict[tuple[str, str], IngestEvent] = {}
        superseded: list[str] = []
        for event in ordered:
            key = (event.source, event.source_id)
            current = winners.get(key)
            if current is None:
                winners[key] = event
                continue
            if self._supersedes(event, current):
                winners[key] = event
                superseded.append(current.event_id)
            else:
                superseded.append(event.event_id)

        available = {event.event_id for event in winners.values()}
        deferred: list[str] = []
        gaps: list[CoverageGap] = []
        deferred_ids: set[str] = set()
        changed = True
        while changed:
            changed = False
            for event in winners.values():
                if event.event_id in deferred_ids:
                    continue
                unmet = [
                    dependency
                    for dependency in event.dependencies
                    if dependency not in available or dependency in deferred_ids
                ]
                if unmet:
                    deferred.append(event.event_id)
                    deferred_ids.add(event.event_id)
                    gaps.append(
                        CoverageGap(
                            code="event-dependency-missing",
                            status=CoverageStatus.PARTIAL,
                            message=(
                                f"Event {event.source_id!r} waits on unavailable "
                                f"dependencies {sorted(set(unmet))}; it will apply "
                                "when the late evidence arrives."
                            ),
                            source="orchestration",
                            source_ids=(event.source_id, *sorted(set(unmet))),
                            retryable=True,
                        )
                    )
                    changed = True

        effective = tuple(
            event
            for event in ordered
            if winners.get((event.source, event.source_id)) is event
            and event.event_id not in deferred_ids
        )
        return ReplayResult(
            ordered=ordered,
            effective=effective,
            superseded=tuple(superseded),
            deferred=tuple(deferred),
            gaps=tuple(gaps),
        )

    @staticmethod
    def _supersedes(candidate: IngestEvent, current: IngestEvent) -> bool:
        """Later observations correct earlier ones; exact ties are stable."""

        if candidate.event_id == current.event_id and candidate == current:
            return False
        return (
            candidate.observed_at,
            candidate.event_time,
            candidate.sequence,
            candidate.event_id,
        ) > (
            current.observed_at,
            current.event_time,
            current.sequence,
            current.event_id,
        )

    def bootstrap(self, result: ReplayResult) -> Bootstrap:
        """Split per-boundary bootstrap state from in-window replay."""

        baseline = tuple(
            event for event in result.effective if event.kind is EventKind.BASELINE
        )
        ordered = tuple(
            event for event in result.effective if event.kind is not EventKind.BASELINE
        )
        return Bootstrap(baseline=baseline, ordered=ordered)

    @staticmethod
    def visible_at(
        events: Iterable[IngestEvent], horizon: datetime
    ) -> tuple[IngestEvent, ...]:
        """Return the events an enrichment scoped at ``horizon`` may use."""

        moment = ensure_utc(horizon)
        return tuple(
            event for event in events if event.info_horizon <= moment
        )

    def build_manifest(
        self,
        *,
        run_id: str,
        started_at: datetime,
        finished_at: datetime,
        selection: GitSelection | None,
        selected_work_items: tuple[int, ...],
        result: ReplayResult,
        ledger: BudgetLedger,
        gaps: Iterable[CoverageGap],
        config_fingerprint: str,
    ) -> RunManifest:
        """Assemble the run manifest from replay inputs and outcomes."""

        journal_gaps = list(gaps) + list(result.gaps)
        return RunManifest(
            run_id=run_id,
            repository_id=self.ids.repository_id,
            repository_url=self.ids.repository_url,
            anchor=self.repository.anchor,
            window=self.repository.window,
            started_at=started_at,
            finished_at=finished_at,
            software_version=software_version(),
            config_fingerprint=config_fingerprint,
            selected_commits=(
                selection.selected_shas if selection is not None else ()
            ),
            selected_work_items=tuple(selected_work_items),
            gaps=tuple(journal_gaps),
            counters={
                **ledger.counters(),
                "events": len(result.ordered),
                "effective_events": len(result.effective),
                "superseded_events": len(result.superseded),
                "deferred_events": len(result.deferred),
            },
            estimated_input_tokens=ledger.estimated_input_tokens,
            estimated_output_tokens=ledger.estimated_output_tokens,
        )


# ---------------------------------------------------------------------------
# Resumable stage runner
# ---------------------------------------------------------------------------


class StageRunner:
    """Run pipeline stages once, checkpointing progress and budgets."""

    def __init__(
        self,
        journal: EventJournal,
        store: CheckpointStore,
        ledger: BudgetLedger,
        *,
        run_id: str,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.journal = journal
        self.store = store
        self.ledger = ledger
        self.run_id = run_id
        self._clock = clock or now_utc

    @classmethod
    def resume_or_create(
        cls,
        journal_path: str | Path,
        checkpoint_path: str | Path,
        manifest: JournalManifest,
        limits: Budgets,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> "StageRunner":
        """Open an existing run or start a new one with matching identity."""

        journal_file = Path(journal_path)
        if journal_file.exists():
            journal = EventJournal.open_existing(
                journal_file,
                repository_id=manifest.repository_id,
                config_fingerprint=manifest.config_fingerprint,
            )
            snapshot = journal.read()
            if snapshot.manifest.run_id != manifest.run_id:
                raise JournalMismatch(
                    f"Journal run {snapshot.manifest.run_id!r} does not match "
                    f"requested run {manifest.run_id!r}"
                )
        else:
            journal = EventJournal.create(journal_file, manifest)
        store = CheckpointStore(checkpoint_path)
        checkpoint = store.load()
        if checkpoint is not None:
            if checkpoint.run_id != manifest.run_id:
                raise JournalMismatch(
                    f"Checkpoint run {checkpoint.run_id!r} does not match "
                    f"requested run {manifest.run_id!r}"
                )
            ledger = checkpoint.restore_ledger(limits)
        else:
            ledger = BudgetLedger(limits)
        return cls(journal, store, ledger, run_id=manifest.run_id, clock=clock)

    @property
    def completed_stages(self) -> tuple[str, ...]:
        checkpoint = self.store.load()
        if checkpoint is None:
            return ()
        return checkpoint.completed_stages

    def run_stage(
        self,
        name: str,
        produce: Callable[[], tuple[Iterable[IngestEvent], Iterable[CoverageGap]]],
    ) -> JournalSnapshot:
        """Run ``produce`` once; completed stages replay from the journal."""

        if name in self.completed_stages:
            return self.journal.read()
        events, gaps = produce()
        # A stage that raises (for example on budget exhaustion) leaves the
        # journal and checkpoint untouched, so a retry resumes this stage
        # with the ledger counters it already spent.
        snapshot_before = self.journal.read()
        counts = self.journal.append(events, gaps)
        snapshot = self.journal.read()
        assert len(snapshot.events) == len(snapshot_before.events) + counts["events"]
        self.store.record(self.run_id, name, self.ledger, snapshot)
        return snapshot


def default_settings() -> OrchestrationSettings:
    return OrchestrationSettings()
