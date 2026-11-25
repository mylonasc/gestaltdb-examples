"""Temporal replay and resumable orchestration over synthetic event streams.

No network, no Git CLI, and no database: every scenario builds its events
in-process with fixed timestamps, so interrupted replay, skew, late arrival,
and horizon gating are fully reproducible.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone

import pytest

from repo_history.budgets import BudgetExceeded, BudgetLedger
from repo_history.events import EventKind, IngestEvent
from repo_history.git import ClockSkew, CommitRecord, GitSelection
from repo_history.github import GitHubDiscovery
from repo_history.ids import IdFactory
from repo_history.models import (
    Budgets,
    CoverageGap,
    CoverageStatus,
    OrchestrationSettings,
    RepositorySpec,
    TimeWindow,
)
from repo_history.orchestration import (
    CheckpointStore,
    EventJournal,
    JournalCorrupted,
    JournalDiverged,
    JournalManifest,
    JournalMismatch,
    StageRunner,
    TemporalCoordinator,
    git_selection_events,
)

UTC = timezone.utc
NOW = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)
DAY_ONE = datetime(2026, 9, 14, 10, 0, tzinfo=UTC)
DAY_TWO = datetime(2026, 9, 14, 11, 0, tzinfo=UTC)
T0 = datetime(2026, 9, 14, 9, 0, tzinfo=UTC)
REPOSITORY_URL = "https://github.com/mylonasc/tf_gnns"
WINDOW = TimeWindow(
    start=datetime(2026, 9, 13, tzinfo=UTC),
    end=datetime(2026, 9, 15, tzinfo=UTC),
)


def repository(**overrides) -> RepositorySpec:
    data = {
        "owner": "mylonasc",
        "name": "tf_gnns",
        "url": REPOSITORY_URL,
        "anchor": "9bfe6c19c02723dbbf101b9e311de684a4728bb1",
        "window": WINDOW,
        "allowed_work_items": (10, 11),
    }
    data.update(overrides)
    return RepositorySpec(**data)


def commit(
    sha: str,
    *,
    parents: tuple[str, ...] = (),
    committed_at: datetime = DAY_ONE,
    merge: bool = False,
) -> CommitRecord:
    return CommitRecord(
        sha=sha,
        parents=parents,
        tree_sha="t" * 40,
        message=f"commit {sha}",
        author_name="Author",
        author_email="author@example.test",
        authored_at=committed_at,
        committer_name="Committer",
        committer_email="committer@example.test",
        committed_at=committed_at,
        source_url=f"{REPOSITORY_URL}/commit/{sha}",
        is_merge=merge,
    )


def selection(
    selected: tuple[CommitRecord, ...] = (),
    boundary: tuple[CommitRecord, ...] = (),
    skews: tuple[ClockSkew, ...] = (),
    gaps: tuple[CoverageGap, ...] = (),
) -> GitSelection:
    return GitSelection(
        anchor="a" * 40,
        window=WINDOW,
        selected_commits=selected,
        boundary_commits=boundary,
        clock_skews=skews,
        gaps=gaps,
    )


def event(
    source_id: str,
    *,
    source: str = "test",
    kind: EventKind = EventKind.WORK_ITEM_EVENT,
    event_time: datetime = DAY_ONE,
    observed_at: datetime = NOW,
    horizon: datetime | None = None,
    event_id: str | None = None,
    dependencies: tuple[str, ...] = (),
    sequence: int = 0,
    payload: dict | None = None,
) -> IngestEvent:
    return IngestEvent(
        event_id=event_id or f"test:{source}:{source_id}:{sequence}",
        kind=kind,
        source=source,
        source_id=source_id,
        event_time=event_time,
        observed_at=observed_at,
        info_horizon=horizon if horizon is not None else observed_at,
        payload=payload or {"source_id": source_id},
        dependencies=dependencies,
        sequence=sequence,
    )


def gap(code: str = "test-gap") -> CoverageGap:
    return CoverageGap(
        code=code,
        status=CoverageStatus.PARTIAL,
        message="synthetic gap",
        source="test",
        source_ids=("synthetic",),
    )


def manifest(run_id: str = "run-1") -> JournalManifest:
    return JournalManifest(
        run_id=run_id,
        repository_id="repo:abc123",
        repository_url=REPOSITORY_URL,
        anchor="a" * 40,
        window_start=WINDOW.start,
        window_end=WINDOW.end,
        config_fingerprint="f" * 64,
        software_version="0.1.0",
        created_at=NOW,
    )


# ---------------------------------------------------------------------------
# Git selection events
# ---------------------------------------------------------------------------


def test_git_selection_events_emit_baseline_and_commit_events():
    boundary = commit("b" * 40, committed_at=T0)
    first = commit("c" * 40, parents=(boundary.sha,), committed_at=DAY_ONE)
    merge = commit(
        "d" * 40, parents=(first.sha, "e" * 40), committed_at=DAY_TWO, merge=True
    )
    skew = ClockSkew(commit_sha=first.sha, parent_sha=boundary.sha, delta_seconds=-30)
    git_selection = selection(
        selected=(first, merge), boundary=(boundary,), skews=(skew,)
    )

    events = git_selection_events(
        git_selection, IdFactory(REPOSITORY_URL), observed_at=NOW
    )

    by_source = {item.source_id: item for item in events}
    assert set(by_source) == {
        f"baseline:{boundary.sha}",
        f"commit:{first.sha}",
        f"commit:{merge.sha}",
    }
    assert by_source[f"baseline:{boundary.sha}"].kind is EventKind.BASELINE
    assert by_source[f"commit:{first.sha}"].kind is EventKind.COMMIT
    # Immutable source facts: the horizon is the commit time itself.
    assert all(item.info_horizon == item.event_time for item in events)
    assert all(item.observed_at == NOW for item in events)
    assert by_source[f"commit:{first.sha}"].dependencies == (
        by_source[f"baseline:{boundary.sha}"].event_id,
    )
    assert by_source[f"commit:{first.sha}"].payload["clock_skew_seconds"] == -30
    assert by_source[f"commit:{merge.sha}"].payload["missing_parents"] == ["e" * 40]
    assert by_source[f"commit:{merge.sha}"].payload["comparison"] == "first-parent"
    root = git_selection_events(
        selection(selected=(commit("f" * 40),)),
        IdFactory(REPOSITORY_URL),
        observed_at=NOW,
    )
    assert root[0].dependencies == ()
    assert root[0].payload["comparison"] == "root-empty-tree"


def test_git_selection_events_are_deterministic():
    git_selection = selection(
        selected=(commit("c" * 40, parents=("b" * 40,)),),
        boundary=(commit("b" * 40, committed_at=T0),),
    )
    ids = IdFactory(REPOSITORY_URL)
    first = git_selection_events(git_selection, ids, observed_at=NOW)
    second = git_selection_events(git_selection, ids, observed_at=NOW)
    assert first == second
    assert [item.source_id for item in first] == sorted(
        item.source_id for item in first
    )


# ---------------------------------------------------------------------------
# Journal durability
# ---------------------------------------------------------------------------


def test_journal_round_trip_preserves_events_and_gaps(tmp_path):
    journal = EventJournal.create(tmp_path / "events.jsonl", manifest())
    events = (
        event("a", event_time=DAY_ONE),
        event("b", event_time=DAY_TWO),
    )
    assert journal.append(events, [gap()]) == {"events": 2, "gaps": 1}

    snapshot = journal.read()

    assert snapshot.manifest.run_id == "run-1"
    assert snapshot.events == events
    assert snapshot.gaps == (gap(),)


def test_journal_skips_identical_duplicates_but_keeps_corrections(tmp_path):
    journal = EventJournal.create(tmp_path / "events.jsonl", manifest())
    original = event("a")
    assert journal.append([original]) == {"events": 1, "gaps": 0}
    assert journal.append([original]) == {"events": 0, "gaps": 0}

    correction = original.model_copy(
        update={"observed_at": NOW + timedelta(hours=1)}
    )
    assert journal.append([correction]) == {"events": 1, "gaps": 0}
    assert [item.observed_at for item in journal.read().events] == [
        NOW,
        NOW + timedelta(hours=1),
    ]


def test_journal_rejects_corrupt_lines(tmp_path):
    path = tmp_path / "events.jsonl"
    EventJournal.create(path, manifest())
    with path.open("a", encoding="utf-8") as handle:
        handle.write("not json\n")

    with pytest.raises(JournalCorrupted):
        EventJournal(path).read()


def test_journal_rejects_a_manifest_from_another_configuration(tmp_path):
    path = tmp_path / "events.jsonl"
    EventJournal.create(path, manifest())

    with pytest.raises(JournalMismatch):
        EventJournal.open_existing(
            path, repository_id="repo:other", config_fingerprint="f" * 64
        )
    with pytest.raises(JournalMismatch):
        EventJournal.open_existing(
            path, repository_id="repo:abc123", config_fingerprint="0" * 64
        )
    with pytest.raises(JournalMismatch):
        EventJournal.create(path, manifest())


def test_journal_without_manifest_is_corrupt(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(JournalCorrupted):
        EventJournal(path).read()


# ---------------------------------------------------------------------------
# Checkpoints and budget continuity
# ---------------------------------------------------------------------------


def test_checkpoint_resume_restores_budget_counters(tmp_path):
    journal = EventJournal.create(tmp_path / "events.jsonl", manifest())
    journal.append([event("a")])
    store = CheckpointStore(tmp_path / "checkpoint.json")
    ledger = BudgetLedger(Budgets(max_api_requests=10, max_work_items=10))
    ledger.reserve_api(3)
    ledger.reserve_work_item(2)
    store.record("run-1", "github", ledger, journal.read())

    runner = StageRunner.resume_or_create(
        journal.path,
        store.path,
        manifest(),
        Budgets(max_api_requests=10, max_work_items=10),
    )

    assert runner.completed_stages == ("github",)
    assert runner.ledger.api_requests == 3
    assert runner.ledger.work_items == 2
    # Budgets continue from the checkpoint instead of resetting.
    with pytest.raises(BudgetExceeded):
        for _ in range(20):
            runner.ledger.reserve_api()


def test_checkpoint_detects_a_diverged_journal(tmp_path):
    journal = EventJournal.create(tmp_path / "events.jsonl", manifest())
    journal.append([event("a")])
    store = CheckpointStore(tmp_path / "checkpoint.json")
    store.record("run-1", "git", BudgetLedger(Budgets()), journal.read())

    other = EventJournal(tmp_path / "other.jsonl")
    EventJournal.create(other.path, manifest(run_id="run-1"))
    other.append([event("different")])
    diverged = other.read().model_copy(
        update={"manifest": other.read().manifest}
    )
    assert diverged.events != journal.read().events
    with pytest.raises(JournalDiverged):
        store.record("run-1", "github", BudgetLedger(Budgets()), diverged)


def test_checkpoint_rejects_a_foreign_run(tmp_path):
    store = CheckpointStore(tmp_path / "checkpoint.json")
    journal = EventJournal.create(tmp_path / "events.jsonl", manifest())
    with pytest.raises(JournalMismatch):
        store.record("run-2", "git", BudgetLedger(Budgets()), journal.read())


def test_checkpoint_round_trip_through_disk(tmp_path):
    store = CheckpointStore(tmp_path / "checkpoint.json")
    journal = EventJournal.create(tmp_path / "events.jsonl", manifest())
    journal.append([event("a"), event("b")])
    ledger = BudgetLedger(Budgets())
    ledger.reserve_api(4)
    saved = store.record("run-1", "git", ledger, journal.read())

    assert store.load() == saved
    assert store.load() is not None
    restored = store.load().restore_ledger(Budgets())
    assert restored is not None
    assert restored.api_requests == 4


# ---------------------------------------------------------------------------
# Stable ordering, conflicts, and dependencies
# ---------------------------------------------------------------------------


def test_collect_orders_events_stably_regardless_of_arrival_order():
    coordinator = TemporalCoordinator(repository())
    events = [
        event("c", event_time=DAY_TWO),
        event("a", event_time=DAY_ONE),
        event("b", event_time=DAY_ONE, source="aaa"),
        event("b", event_time=DAY_ONE, source="zzz"),
    ]
    shuffled = list(events)
    random.Random(7).shuffle(shuffled)

    first = coordinator.collect(shuffled)
    second = coordinator.collect(list(reversed(events)))

    assert first == second
    assert [item.order_key for item in first.ordered] == sorted(
        item.order_key for item in events
    )
    assert [(item.source, item.source_id) for item in first.ordered] == [
        ("aaa", "b"),
        ("test", "a"),
        ("zzz", "b"),
        ("test", "c"),
    ]
    assert first.effective == first.ordered
    assert first.superseded == ()
    assert first.deferred == ()


def test_collect_resolves_conflicting_observations_causally():
    coordinator = TemporalCoordinator(repository())
    stale = event("item", event_time=DAY_ONE, observed_at=NOW, event_id="same")
    correction = event(
        "item",
        event_time=DAY_ONE,
        observed_at=NOW + timedelta(hours=1),
        event_id="same",
        payload={"source_id": "item", "revised": True},
    )

    result = coordinator.collect([correction, stale])

    assert [item.payload for item in result.effective] == [
        {"source_id": "item", "revised": True}
    ]
    assert result.superseded == ("same",)
    assert len(result.ordered) == 2
    assert all(item in (stale, correction) for item in result.ordered)


def test_collect_breaks_exact_ties_deterministically():
    coordinator = TemporalCoordinator(repository())
    first = event("item", event_id="aaa")
    second = event("item", event_id="zzz")

    result = coordinator.collect([second, first])

    assert [item.event_id for item in result.effective] == ["zzz"]
    assert result.superseded == ("aaa",)


def test_collect_defers_events_until_dependencies_arrive():
    coordinator = TemporalCoordinator(repository())
    child = event("child", dependencies=("parent-event",))
    parent = event("parent", event_id="parent-event")

    partial = coordinator.collect([child])
    assert partial.effective == ()
    assert partial.deferred == (child.event_id,)
    assert partial.gaps[0].code == "event-dependency-missing"
    assert partial.gaps[0].retryable is True

    # The late evidence arrives; replaying the full journal applies the child.
    # Effective order follows the stable order key, not arrival order.
    complete = coordinator.collect([child, parent])
    assert [item.source_id for item in complete.effective] == ["child", "parent"]
    assert complete.deferred == ()


def test_collect_cascades_deferral_through_chains():
    coordinator = TemporalCoordinator(repository())
    grandchild = event("grandchild", dependencies=("child-event",))
    child = event("child", event_id="child-event", dependencies=("missing",))

    result = coordinator.collect([grandchild, child])

    assert result.effective == ()
    assert set(result.deferred) == {grandchild.event_id, child.event_id}
    assert len(result.gaps) == 2


def test_unavailable_content_is_a_gap_not_an_invented_event(tmp_path):
    journal = EventJournal.create(tmp_path / "events.jsonl", manifest())
    missing = CoverageGap(
        code="git-parent-missing",
        status=CoverageStatus.PARTIAL,
        message="parent tree is absent from the bounded history",
        source="git",
        source_ids=("deadbeef",),
        retryable=True,
    )
    journal.append([], [missing])
    snapshot = journal.read()

    result = TemporalCoordinator(repository()).collect(snapshot.events)

    assert result.effective == ()
    assert snapshot.gaps == (missing,)


# ---------------------------------------------------------------------------
# Bootstrap, horizons, and replay equivalence
# ---------------------------------------------------------------------------


def test_bootstrap_partitions_boundary_state_from_window_replay():
    coordinator = TemporalCoordinator(repository())
    baseline = event("base", kind=EventKind.BASELINE, event_time=T0)
    commit_event = event("tip", kind=EventKind.COMMIT, event_time=DAY_ONE)
    observed = event("item", kind=EventKind.WORK_ITEM_OBSERVED, event_time=NOW)

    bootstrap = coordinator.bootstrap(
        coordinator.collect([observed, commit_event, baseline])
    )

    assert [item.source_id for item in bootstrap.baseline] == ["base"]
    assert [item.source_id for item in bootstrap.ordered] == ["tip", "item"]


def test_visible_at_hides_future_information():
    coordinator = TemporalCoordinator(repository())
    past_label = event("label", event_time=DAY_ONE, observed_at=DAY_ONE)
    future_label = event(
        "label",
        event_time=NOW,
        observed_at=NOW,
        horizon=NOW,
        event_id="future-label",
    )
    events = [past_label, future_label]

    assert coordinator.visible_at(events, DAY_TWO) == (past_label,)
    assert coordinator.visible_at(events, NOW) == tuple(events)


def test_future_observations_cannot_seed_past_decisions():
    coordinator = TemporalCoordinator(repository())
    commit_event = event(
        "tip", kind=EventKind.COMMIT, event_time=DAY_ONE, horizon=DAY_ONE
    )
    future_label = event(
        "label",
        event_time=NOW,
        observed_at=NOW,
        horizon=NOW,
        payload={"label": "future-topic"},
    )

    horizon = DAY_TWO
    usable = coordinator.visible_at([commit_event, future_label], horizon)

    assert usable == (commit_event,)
    assert all(item.info_horizon <= horizon for item in usable)


def test_interrupted_replay_matches_a_single_replay(tmp_path):
    coordinator = TemporalCoordinator(repository())
    boundary = commit("b" * 40, committed_at=T0)
    first = commit("c" * 40, parents=(boundary.sha,), committed_at=DAY_ONE)
    second = commit("d" * 40, parents=(first.sha,), committed_at=DAY_TWO)
    git_events = git_selection_events(
        selection(selected=(first, second), boundary=(boundary,)),
        IdFactory(REPOSITORY_URL),
        observed_at=NOW,
    )
    remote = (
        event("item", event_time=DAY_ONE, observed_at=NOW),
        event("item", event_time=DAY_ONE, observed_at=NOW + timedelta(hours=1),
              event_id="remote-correction", payload={"revised": True}),
    )

    uninterrupted = EventJournal.create(tmp_path / "full.jsonl", manifest())
    uninterrupted.append([*git_events, *remote])

    interrupted = EventJournal.create(tmp_path / "split.jsonl", manifest())
    interrupted.append(git_events[:2])
    interrupted.append([*git_events[2:], *remote])

    whole = coordinator.collect(uninterrupted.read().events)
    resumed = coordinator.collect(interrupted.read().events)

    assert resumed == whole
    assert [item.event_id for item in resumed.effective] == [
        item.event_id for item in whole.effective
    ]
    # Replaying the same journal twice is idempotent.
    assert coordinator.collect(interrupted.read().events) == resumed


def test_skewed_commit_clocks_replay_deterministically():
    coordinator = TemporalCoordinator(repository())
    # The child claims an earlier timestamp than its parent: clocks disagree,
    # but the merged order must not depend on arrival order.
    parent = commit("p" * 40, committed_at=DAY_TWO)
    child = commit("c" * 40, parents=(parent.sha,), committed_at=DAY_ONE)
    git_events = git_selection_events(
        selection(selected=(parent, child)),
        IdFactory(REPOSITORY_URL),
        observed_at=NOW,
    )
    arrival = list(git_events)
    random.Random(3).shuffle(arrival)

    first = coordinator.collect(arrival)
    second = coordinator.collect(list(reversed(arrival)))

    assert first == second
    # Deterministic event-time order even though the child predates its parent.
    assert [item.source_id for item in first.effective] == [
        f"commit:{child.sha}",
        f"commit:{parent.sha}",
    ]


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------


def test_stage_runner_skips_completed_stages_on_resume(tmp_path):
    journal_path = tmp_path / "events.jsonl"
    checkpoint_path = tmp_path / "checkpoint.json"
    calls: list[str] = []

    def git_stage():
        calls.append("git")
        return [event("tip", kind=EventKind.COMMIT)], [gap("git-gap")]

    runner = StageRunner.resume_or_create(
        journal_path, checkpoint_path, manifest(), Budgets()
    )
    runner.run_stage("git", git_stage)
    assert calls == ["git"]

    resumed = StageRunner.resume_or_create(
        journal_path, checkpoint_path, manifest(), Budgets()
    )
    snapshot = resumed.run_stage("git", git_stage)

    assert calls == ["git"]
    assert [item.source_id for item in snapshot.events] == ["tip"]
    assert [item.code for item in snapshot.gaps] == ["git-gap"]


def test_stage_runner_resumes_after_a_budget_failure(tmp_path):
    journal_path = tmp_path / "events.jsonl"
    checkpoint_path = tmp_path / "checkpoint.json"
    limits = Budgets(max_api_requests=2)
    attempts: list[int] = []

    def flaky_stage():
        attempts.append(1)
        runner_ledger.reserve_api(2)
        # The second attempt exceeds the budget partway through.
        runner_ledger.reserve_api()
        return [event("tip")], []

    runner = StageRunner.resume_or_create(
        journal_path, checkpoint_path, manifest(), limits
    )
    runner_ledger = runner.ledger
    with pytest.raises(BudgetExceeded):
        runner.run_stage("github", flaky_stage)
    # A failed stage checkpoints nothing and appends nothing.
    assert runner.completed_stages == ()
    assert runner.journal.read().events == ()

    runner_ledger.exhausted.clear()
    runner_ledger.api_requests = 0

    def healthy_stage():
        runner_ledger.reserve_api()
        return [event("tip")], []

    snapshot = runner.run_stage("github", healthy_stage)
    assert [item.source_id for item in snapshot.events] == ["tip"]
    assert runner.completed_stages == ("github",)


def test_stage_runner_rejects_a_foreign_run_identity(tmp_path):
    journal_path = tmp_path / "events.jsonl"
    checkpoint_path = tmp_path / "checkpoint.json"
    runner = StageRunner.resume_or_create(
        journal_path, checkpoint_path, manifest(), Budgets()
    )
    runner.run_stage("git", lambda: ([event("tip")], []))

    with pytest.raises(JournalMismatch):
        StageRunner.resume_or_create(
            journal_path, checkpoint_path, manifest(run_id="run-2"), Budgets()
        )


def test_build_manifest_records_selection_counters_and_gaps():
    coordinator = TemporalCoordinator(repository())
    git_selection = selection(
        selected=(commit("c" * 40),),
        gaps=(
            CoverageGap(
                code="git-parent-missing",
                status=CoverageStatus.PARTIAL,
                message="parent absent",
                source="git",
                source_ids=("c" * 40,),
            ),
        ),
    )
    events = git_selection_events(
        git_selection, IdFactory(REPOSITORY_URL), observed_at=NOW
    )
    result = coordinator.collect(events)
    ledger = BudgetLedger(Budgets())
    ledger.reserve_commit(1)

    started = NOW - timedelta(minutes=5)
    manifest_result = coordinator.build_manifest(
        run_id="run-1",
        started_at=started,
        finished_at=NOW,
        selection=git_selection,
        selected_work_items=(10, 11),
        result=result,
        ledger=ledger,
        gaps=git_selection.gaps,
        config_fingerprint="f" * 64,
    )

    assert manifest_result.run_id == "run-1"
    assert manifest_result.anchor == repository().anchor
    assert manifest_result.selected_commits == ("c" * 40,)
    assert manifest_result.selected_work_items == (10, 11)
    assert manifest_result.started_at == started
    assert manifest_result.software_version
    assert manifest_result.counters["commits"] == 1
    assert manifest_result.counters["effective_events"] == 1
    assert [item.code for item in manifest_result.gaps] == ["git-parent-missing"]


def test_orchestration_settings_have_bounded_defaults():
    settings = OrchestrationSettings()
    assert settings.journal_file == "data/journal/events.jsonl"
    assert settings.checkpoint_file == "data/journal/checkpoint.json"
    with pytest.raises(Exception):
        OrchestrationSettings(journal_file="  ")


def test_settings_are_loaded_from_toml():
    from repo_history.config import load_config

    config = load_config()
    assert config.orchestration.journal_file == "data/journal/events.jsonl"
    assert config.orchestration.checkpoint_file == "data/journal/checkpoint.json"


def test_discovery_events_replay_through_the_coordinator():
    discovery = GitHubDiscovery(
        repository="mylonasc/tf_gnns",
        observed_at=NOW,
        offline=True,
        work_items=(),
        pull_requests=(),
        provider_labels=(),
        label_assignments=(),
        hierarchy_edges=(),
        absences=(),
        gaps=(),
        fetches=(),
        counters={},
        truncated=False,
    )
    events = discovery.to_events(IdFactory(REPOSITORY_URL))
    assert events == ()

    coordinator = TemporalCoordinator(repository())
    result = coordinator.collect(events)
    assert result.effective == ()
