"""Separate-process work-item analysis with durable, replayable jobs.

The coordinator submits :class:`WorkJob` envelopes whose inputs are fully
embedded (feature snapshot, candidate topics, registry snapshot, horizon), so
any worker — now or after a restart — replays the same job without hidden
state.  Workers run in spawned OS processes
(:func:`multiprocessing.get_context("spawn")`), record their PID in the
result, and return :class:`WorkResult` envelopes to the single writer.

Required enrichment (deterministic labeling) must succeed for a job to
complete; optional enrichment (LLM proposals) may abstain or fail while the
job still completes, and the two outcomes stay distinguishable.  Results carry
the job's horizon and are merged in ``(horizon, job_id)`` order, so an
out-of-order completion can never publish future context into an earlier
event.  Redelivered jobs produce byte-identical results keyed by ``job_id``,
so repeats create no duplicate assertions.
"""

from __future__ import annotations

import json
import multiprocessing as multiprocessing_module
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from .analysis import AnalysisResult, DecisionStatus
from .budgets import BudgetLedger
from .enrichment import (
    Enricher,
    EnrichmentCache,
    EnrichmentError,
    LLMSettings,
)
from .ids import IdFactory, stable_id
from .labeling import ContentFeatures, DeterministicLabeler, TopicRegistry
from .models import Budgets, FrozenModel, ensure_utc

__all__ = [
    "JOB_ENVELOPE_VERSION",
    "JobQueue",
    "JobRecord",
    "JobStatus",
    "WorkJob",
    "WorkResult",
    "WorkerPool",
    "collect_results",
    "run_job_file",
]

JOB_ENVELOPE_VERSION = 1

JobStatus = Literal["pending", "running", "completed", "failed"]


class WorkJob(FrozenModel):
    """Versioned, self-contained unit of work-item analysis."""

    envelope_version: int = JOB_ENVELOPE_VERSION
    job_id: str
    subject_id: str
    work_item_number: int
    subject_text: str
    features: ContentFeatures
    candidates: tuple[dict[str, str], ...] = ()
    registry: TopicRegistry
    info_horizon: datetime
    run_id: str
    method: Literal["deterministic", "deterministic+llm"] = "deterministic"
    llm: LLMSettings | None = None
    fake_replies: tuple[str, ...] | None = None
    attempts: int = 0
    max_attempts: int = 3
    status: JobStatus = "pending"
    worker_pid: int | None = None


class JobRecord(FrozenModel):
    job_id: str
    status: JobStatus
    attempts: int
    updated_at: datetime


class WorkResult(FrozenModel):
    """Versioned worker output; horizons echo the job, never the wall clock."""

    envelope_version: int = JOB_ENVELOPE_VERSION
    job_id: str
    subject_id: str
    worker_pid: int
    required: AnalysisResult
    optional: AnalysisResult | None = None
    optional_error: str | None = None
    info_horizon: datetime
    attempt: int = 0
    started_at: datetime
    finished_at: datetime


class JobQueue:
    """Durable job and result files; workers only read jobs and write results."""

    def __init__(self, directory: str | Path) -> None:
        self._root = Path(directory)
        self._jobs = self._root / "jobs"
        self._results = self._root / "results"
        self._jobs.mkdir(parents=True, exist_ok=True)
        self._results.mkdir(parents=True, exist_ok=True)

    @property
    def directory(self) -> Path:
        return self._root

    def _job_path(self, job_id: str) -> Path:
        return self._jobs / f"{job_id}.json"

    def _result_path(self, job_id: str) -> Path:
        return self._results / f"{job_id}.json"

    @staticmethod
    def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary.replace(path)

    def submit(self, job: WorkJob) -> str:
        """Store a job; resubmitting the same ID returns it unchanged."""

        path = self._job_path(job.job_id)
        if path.is_file():
            return job.job_id
        self._write_atomic(path, job.model_dump(mode="json"))
        return job.job_id

    def load(self, job_id: str) -> WorkJob:
        return WorkJob.model_validate(
            json.loads(self._job_path(job_id).read_text(encoding="utf-8"))
        )

    def _store(self, job: WorkJob) -> None:
        self._write_atomic(self._job_path(job.job_id), job.model_dump(mode="json"))

    def claim(self, job_id: str, worker_pid: int) -> WorkJob:
        job = self.load(job_id)
        if job.status == "completed":
            return job
        claimed = job.model_copy(
            update={"status": "running", "worker_pid": worker_pid}
        )
        self._store(claimed)
        return claimed

    def record_attempt(self, job_id: str) -> WorkJob:
        job = self.load(job_id)
        updated = job.model_copy(update={"attempts": job.attempts + 1})
        self._store(updated)
        return updated

    def complete(self, job_id: str, result: WorkResult) -> None:
        if result.envelope_version != JOB_ENVELOPE_VERSION:
            raise ValueError(
                f"Refusing result with envelope v{result.envelope_version}"
            )
        job = self.load(job_id)
        if result.info_horizon != job.info_horizon:
            raise ValueError(
                f"Result horizon {result.info_horizon} does not match "
                f"job horizon {job.info_horizon}"
            )
        self._write_atomic(
            self._result_path(job_id), result.model_dump(mode="json")
        )
        self._store(job.model_copy(update={"status": "completed"}))

    def fail(self, job_id: str, *, retryable: bool) -> WorkJob:
        job = self.load(job_id)
        if retryable and job.attempts < job.max_attempts:
            updated = job.model_copy(update={"status": "pending", "worker_pid": None})
        else:
            updated = job.model_copy(update={"status": "failed", "worker_pid": None})
        self._store(updated)
        return updated

    def reset(self, job_id: str) -> WorkJob:
        job = self.load(job_id)
        updated = job.model_copy(update={"status": "pending", "worker_pid": None})
        self._store(updated)
        return updated

    def pending(self) -> list[str]:
        ids: list[str] = []
        for path in sorted(self._jobs.glob("*.json")):
            job = WorkJob.model_validate(
                json.loads(path.read_text(encoding="utf-8"))
            )
            if job.status == "pending":
                ids.append(job.job_id)
        return ids

    def result(self, job_id: str) -> WorkResult | None:
        path = self._result_path(job_id)
        if not path.is_file():
            return None
        result = WorkResult.model_validate(
            json.loads(path.read_text(encoding="utf-8"))
        )
        if result.envelope_version != JOB_ENVELOPE_VERSION:
            raise ValueError(
                f"Refusing result with envelope v{result.envelope_version}"
            )
        return result

    def records(self) -> list[JobRecord]:
        records: list[JobRecord] = []
        for path in sorted(self._jobs.glob("*.json")):
            job = WorkJob.model_validate(
                json.loads(path.read_text(encoding="utf-8"))
            )
            records.append(
                JobRecord(
                    job_id=job.job_id,
                    status=job.status,
                    attempts=job.attempts,
                    updated_at=job.info_horizon,
                )
            )
        return records


def make_job_id(repository_url: str, run_id: str, work_item_number: int) -> str:
    ids = IdFactory(repository_url)
    return stable_id(
        "workjob", ids.repository_id, run_id, str(work_item_number)
    )


# ---------------------------------------------------------------------------
# Worker entry point (spawn-safe: module level, plain-data arguments)
# ---------------------------------------------------------------------------


def run_job_file(
    queue_dir: str,
    job_id: str,
    *,
    cache_dir: str | None = None,
) -> dict[str, Any]:
    """Execute one job in this process; returns a JSON-serializable summary."""

    queue = JobQueue(queue_dir)
    pid = os.getpid()
    job = queue.claim(job_id, pid)
    if job.status == "completed":
        existing = queue.result(job_id)
        assert existing is not None
        return {"job_id": job_id, "pid": pid, "repeated": True}
    job = queue.record_attempt(job_id)

    started = ensure_utc(datetime.now().astimezone())
    labeler = DeterministicLabeler()
    required = labeler.assign(
        job.features,
        job.registry,
        subject_id=job.subject_id,
        run_id=job.run_id,
        info_horizon=job.info_horizon,
    ).result
    optional: AnalysisResult | None = None
    optional_error: str | None = None
    usage = {"llm_requests": 0, "estimated_input": 0, "estimated_output": 0}

    if job.method == "deterministic+llm":
        try:
            optional, usage = _run_optional_llm(job, cache_dir)
        except Exception as exc:  # optional enrichment never fails the job
            optional = None
            optional_error = f"{type(exc).__name__}: {exc}"

    finished = ensure_utc(datetime.now().astimezone())
    result = WorkResult(
        job_id=job.job_id,
        subject_id=job.subject_id,
        worker_pid=pid,
        required=required,
        optional=optional,
        optional_error=optional_error,
        info_horizon=job.info_horizon,
        attempt=job.attempts,
        started_at=started,
        finished_at=finished,
    )
    if required.status is DecisionStatus.FAILED:
        queue.fail(job_id, retryable=True)
        return {"job_id": job_id, "pid": pid, "failed": True}
    queue.complete(job_id, result)
    return {
        "job_id": job_id,
        "pid": pid,
        "required_status": required.status.value,
        "optional_status": optional.status.value if optional is not None else None,
        "optional_error": optional_error,
        "usage": usage,
    }


def _run_optional_llm(
    job: WorkJob, cache_dir: str | None
) -> tuple[AnalysisResult, dict[str, int]]:
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    if job.llm is None:
        raise EnrichmentError("optional LLM enrichment requested without settings")
    if job.fake_replies is not None:
        model: Any = GenericFakeChatModel(
            messages=iter([AIMessage(content=reply) for reply in job.fake_replies])
        )
        model_id = "fake:scripted"
    else:
        from .enrichment import build_chat_model

        model = build_chat_model(job.llm)
        model_id = f"{job.llm.provider}:{job.llm.model}"
    del model_id
    cache_root = Path(cache_dir) if cache_dir else Path("data/llm")
    cache_root.mkdir(parents=True, exist_ok=True)
    ledger = BudgetLedger(Budgets())
    enricher = Enricher(
        model,
        job.llm,
        ledger=ledger,
        cache=EnrichmentCache(cache_root / f"{job.job_id}.json"),
        run_id=job.run_id,
    )
    result = enricher.enrich_topic(
        subject_id=job.subject_id,
        subject_text=job.subject_text,
        candidates=[dict(item) for item in job.candidates],
        registry_version=job.registry.version,
        info_horizon=job.info_horizon,
    )
    usage = {
        "llm_requests": ledger.llm_requests,
        "estimated_input": ledger.estimated_input_tokens,
        "estimated_output": ledger.estimated_output_tokens,
    }
    return result, usage


# ---------------------------------------------------------------------------
# Pool and collection
# ---------------------------------------------------------------------------


class WorkerPool:
    """Run queued jobs in spawned worker processes with bounded retries."""

    def __init__(
        self,
        queue: JobQueue,
        *,
        max_workers: int = 2,
        task_timeout_seconds: float = 120.0,
        cache_dir: str | Path | None = None,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        self._queue = queue
        self._max_workers = max_workers
        self._timeout = task_timeout_seconds
        self._cache_dir = str(cache_dir) if cache_dir is not None else None
        self._context = multiprocessing_module.get_context("spawn")

    def run_pending(self) -> dict[str, dict[str, Any]]:
        """Run every pending job; crashed or timed-out jobs stay resumable."""

        summaries: dict[str, dict[str, Any]] = {}
        pending = self._queue.pending()
        for index in range(0, len(pending), self._max_workers):
            batch = pending[index : index + self._max_workers]
            processes = [
                self._context.Process(
                    target=_spawn_entry,
                    args=(str(self._queue.directory), job_id, self._cache_dir),
                )
                for job_id in batch
            ]
            for process in processes:
                process.start()
            for process, job_id in zip(processes, batch):
                process.join(self._timeout)
                if process.is_alive():
                    process.terminate()
                    process.join(5)
                    self._queue.reset(job_id)
                    summaries[job_id] = {"job_id": job_id, "timeout": True}
                elif process.exitcode != 0:
                    self._queue.reset(job_id)
                    summaries[job_id] = {
                        "job_id": job_id,
                        "crashed": True,
                        "exitcode": process.exitcode,
                    }
                else:
                    summaries[job_id] = {"job_id": job_id, "done": True}
        return summaries

    @property
    def queue(self) -> JobQueue:
        return self._queue


def _spawn_entry(queue_dir: str, job_id: str, cache_dir: str | None) -> None:
    run_job_file(queue_dir, job_id, cache_dir=cache_dir)


def collect_results(queue: JobQueue) -> list[WorkResult]:
    """Merge completed results in horizon order for the single writer.

    Sorting by ``(info_horizon, job_id)`` keeps a late-arriving result at its
    own horizon: future context can never leak into an earlier event.
    """

    results: list[WorkResult] = []
    for path in sorted(queue.directory.glob("results/*.json")):
        result = queue.result(path.stem)
        if result is not None:
            results.append(result)
    results.sort(key=lambda item: (item.info_horizon, item.job_id))
    return results


def charge_usage(ledger: BudgetLedger, result: WorkResult) -> None:
    """Charge the coordinator ledger for a worker's reported LLM usage."""

    for attempt in (result.optional,):
        if attempt is None:
            continue
        ledger.reserve_llm(
            input_tokens=attempt.provenance.estimated_input_tokens,
            output_tokens=attempt.provenance.estimated_output_tokens,
            estimated_input_tokens=attempt.provenance.estimated_input_tokens,
            estimated_output_tokens=attempt.provenance.estimated_output_tokens,
        )
