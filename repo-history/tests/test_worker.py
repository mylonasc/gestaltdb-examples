"""Separate-process work-item analysis over synthetic jobs.

Process tests use real spawned interpreters (``spawn`` context): PIDs prove
separation, timeouts prove crash recovery, and files prove durability.  No
network, no models, no database.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from repo_history.analysis import DecisionStatus
from repo_history.budgets import BudgetExceeded, BudgetLedger
from repo_history.enrichment import LLMSettings
from repo_history.ids import IdFactory
from repo_history.labeling import ContentFeatures, seed_registry
from repo_history.models import Budgets
from repo_history.worker import (
    JOB_ENVELOPE_VERSION,
    JobQueue,
    WorkJob,
    WorkerPool,
    charge_usage,
    collect_results,
    make_job_id,
    run_job_file,
)

UTC = timezone.utc
NOW = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)
EARLY = datetime(2026, 9, 14, 12, 0, tzinfo=UTC)
LATE = datetime(2026, 9, 14, 18, 0, tzinfo=UTC)
REPOSITORY_URL = "https://github.com/mylonasc/tf_gnns"
RUN_ID = "run-1"


def features(subject: str = "workitem:github:10") -> ContentFeatures:
    return ContentFeatures(
        subject_id=subject,
        title_tokens=("sparse", "training", "fails"),
        body_tokens=("graph", "convolution", "overflows", "sparse"),
    )


def registry_at(moment: datetime):
    registry = seed_registry(at=moment)
    registry, _ = registry.register(
        "sparse-training", at=moment, created_by="test", ids=IdFactory(REPOSITORY_URL),
        description="sparse training graphs convolution",
    )
    return registry


def job(
    number: int = 10,
    *,
    horizon: datetime = EARLY,
    method: str = "deterministic",
    llm: LLMSettings | None = None,
    fake_replies: tuple[str, ...] | None = None,
) -> WorkJob:
    return WorkJob(
        job_id=make_job_id(REPOSITORY_URL, RUN_ID, number),
        subject_id=f"workitem:github:{number}",
        work_item_number=number,
        subject_text="Training fails on sparse graphs.",
        features=features(f"workitem:github:{number}"),
        candidates=({"topic_id": IdFactory(REPOSITORY_URL).topic("sparse-training"),
                     "name": "sparse-training"},),
        registry=registry_at(horizon),
        info_horizon=horizon,
        run_id=RUN_ID,
        method=method,  # type: ignore[arg-type]
        llm=llm,
        fake_replies=fake_replies,
    )


def llm_settings(**overrides) -> LLMSettings:
    data = {
        "enabled": True,
        "privacy_consent": True,
        "provider": "openai",
        "model": "gpt-4.1-mini",
        "temperature": 0.0,
        "max_attempts": 1,
        "prompt_version": "v1",
        "cache_file": "data/llm/cache.json",
    }
    data.update(overrides)
    return LLMSettings(**data)


# ---------------------------------------------------------------------------
# Queue durability
# ---------------------------------------------------------------------------


def test_job_submission_dedupes_by_id(tmp_path):
    queue = JobQueue(tmp_path / "queue")

    first = queue.submit(job(10))
    second = queue.submit(job(10))

    assert first == second == make_job_id(REPOSITORY_URL, RUN_ID, 10)
    assert queue.pending() == [first]
    assert queue.load(first).status == "pending"


def test_completed_jobs_are_returned_as_is(tmp_path):
    queue = JobQueue(tmp_path / "queue")
    job_id = queue.submit(job(10))

    claimed = queue.claim(job_id, worker_pid=1234)
    assert claimed.status == "running"
    queue.complete(job_id, queue.result(job_id) or _result(job_id))
    assert queue.load(job_id).status == "completed"
    assert queue.pending() == []
    # Claiming a completed job never resurrects it.
    assert queue.claim(job_id, worker_pid=9999).status == "completed"


def _result(job_id: str):
    from repo_history.worker import WorkResult
    from repo_history.analysis import AnalysisResult, Provenance

    return WorkResult(
        job_id=job_id,
        subject_id="workitem:github:10",
        worker_pid=1111,
        required=AnalysisResult.create(
            subject_id="workitem:github:10",
            kind="topic",
            status=DecisionStatus.ACCEPTED,
            value={"action": "reuse"},
            provenance=Provenance(
                method="m", method_version="1", run_id=RUN_ID, info_horizon=EARLY
            ),
        ),
        info_horizon=EARLY,
        started_at=EARLY,
        finished_at=EARLY,
    )


def test_result_envelope_version_is_checked(tmp_path):
    queue = JobQueue(tmp_path / "queue")
    job_id = queue.submit(job(10))

    bad = _result(job_id).model_copy(update={"envelope_version": 99})
    with pytest.raises(ValueError):
        queue.complete(job_id, bad)


def test_result_horizon_must_match_the_job(tmp_path):
    queue = JobQueue(tmp_path / "queue")
    job_id = queue.submit(job(10, horizon=EARLY))

    drifted = _result(job_id).model_copy(update={"info_horizon": LATE})
    with pytest.raises(ValueError):
        queue.complete(job_id, drifted)


def test_failed_jobs_retry_until_attempts_run_out(tmp_path):
    queue = JobQueue(tmp_path / "queue")
    submitted = job(10, horizon=EARLY).model_copy(update={"max_attempts": 1})
    job_id = queue.submit(submitted)

    queue.record_attempt(job_id)
    assert queue.fail(job_id, retryable=True).status == "failed"
    assert queue.fail(job_id, retryable=True).status == "failed"


# ---------------------------------------------------------------------------
# Separate processes
# ---------------------------------------------------------------------------


def test_worker_runs_in_a_separate_process(tmp_path):
    queue = JobQueue(tmp_path / "queue")
    job_id = queue.submit(job(10))
    pool = WorkerPool(queue, max_workers=1, task_timeout_seconds=300.0)

    summaries = pool.run_pending()

    assert summaries[job_id].get("done") is True
    result = queue.result(job_id)
    assert result is not None
    assert result.envelope_version == JOB_ENVELOPE_VERSION
    assert result.worker_pid != os.getpid()
    assert result.worker_pid > 0
    assert result.required.status is DecisionStatus.ACCEPTED
    assert result.required.value["action"] == "reuse"
    assert result.info_horizon == EARLY
    assert queue.load(job_id).status == "completed"


def test_repeated_runs_create_no_duplicate_assertions(tmp_path):
    queue = JobQueue(tmp_path / "queue")
    job_id = queue.submit(job(10))
    pool = WorkerPool(queue, max_workers=1, task_timeout_seconds=300.0)
    pool.run_pending()

    before = (tmp_path / "queue" / "results" / f"{job_id}.json").read_bytes()
    rerun = run_job_file(str(tmp_path / "queue"), job_id)

    after = (tmp_path / "queue" / "results" / f"{job_id}.json").read_bytes()
    assert rerun["repeated"] is True
    assert before == after
    assert len(queue.records()) == 1


def test_timeout_kill_loses_no_accepted_work(tmp_path):
    queue = JobQueue(tmp_path / "queue")
    early_id = queue.submit(job(10, horizon=EARLY))
    late_id = queue.submit(job(11, horizon=LATE))

    # A near-zero timeout kills workers during spawn startup every time.
    killer = WorkerPool(queue, max_workers=2, task_timeout_seconds=0.01)
    killed = killer.run_pending()

    assert set(killed) == {early_id, late_id}
    assert queue.result(early_id) is None
    assert queue.pending() == sorted([early_id, late_id])

    # Restart with a real budget of time: nothing accepted is lost or doubled.
    pool = WorkerPool(queue, max_workers=2, task_timeout_seconds=300.0)
    pool.run_pending()

    assert queue.result(early_id) is not None
    assert queue.result(late_id) is not None
    assert queue.pending() == []
    assert {record.job_id for record in queue.records()} == {early_id, late_id}


def test_out_of_order_completion_keeps_horizons(tmp_path):
    queue = JobQueue(tmp_path / "queue")
    early_id = queue.submit(job(10, horizon=EARLY))
    late_id = queue.submit(job(11, horizon=LATE))

    # Complete the later horizon first, in this process.
    run_job_file(str(tmp_path / "queue"), late_id)
    pool = WorkerPool(queue, max_workers=1, task_timeout_seconds=300.0)
    pool.run_pending()

    merged = collect_results(queue)

    assert [result.job_id for result in merged] == [early_id, late_id]
    assert [result.info_horizon for result in merged] == [EARLY, LATE]
    assert all(
        result.required.provenance.info_horizon == result.info_horizon
        for result in merged
    )


def test_optional_llm_failure_keeps_required_completion(tmp_path):
    queue = JobQueue(tmp_path / "queue")
    bad_llm = llm_settings(provider="mystery", model="m-1")
    job_id = queue.submit(
        job(10, method="deterministic+llm", llm=bad_llm)
    )
    pool = WorkerPool(queue, max_workers=1, task_timeout_seconds=300.0)
    pool.run_pending()

    result = queue.result(job_id)
    assert result is not None
    assert result.required.status is DecisionStatus.ACCEPTED
    assert result.optional is None
    assert result.optional_error is not None
    assert "mystery" in result.optional_error
    assert queue.load(job_id).status == "completed"


def test_optional_llm_success_reports_usage_for_the_ledger(tmp_path):
    import json as json_module

    reply = json_module.dumps(
        {
            "action": "reuse",
            "topic_id": IdFactory(REPOSITORY_URL).topic("sparse-training"),
            "confidence": 0.8,
            "rationale": "Matches sparse training.",
            "evidence_quotes": ["sparse"],
        }
    )
    queue = JobQueue(tmp_path / "queue")
    job_id = queue.submit(
        job(10, method="deterministic+llm", llm=llm_settings(), fake_replies=(reply,))
    )
    pool = WorkerPool(
        queue, max_workers=1, task_timeout_seconds=300.0,
        cache_dir=tmp_path / "llm-cache",
    )
    pool.run_pending()

    result = queue.result(job_id)
    assert result is not None
    assert result.optional is not None
    assert result.optional.status is DecisionStatus.ACCEPTED

    ledger = BudgetLedger(Budgets(max_llm_requests=1))
    charge_usage(ledger, result)
    assert ledger.llm_requests == 1
    with pytest.raises(BudgetExceeded):
        charge_usage(ledger, result)


def test_charge_usage_skips_jobs_without_optional_results(tmp_path):
    ledger = BudgetLedger(Budgets())
    charge_usage(ledger, _result("job:x"))
    assert ledger.llm_requests == 0
