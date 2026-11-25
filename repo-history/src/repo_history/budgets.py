"""Deterministic request/content/token accounting with fail-closed limits."""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import Budgets


class BudgetExceeded(RuntimeError):
    def __init__(self, dimension: str, requested: int, used: int, limit: int) -> None:
        super().__init__(
            f"Budget exceeded for {dimension}: requested {requested}, "
            f"used {used}, limit {limit}"
        )
        self.dimension = dimension
        self.requested = requested
        self.used = used
        self.limit = limit


@dataclass
class BudgetLedger:
    limits: Budgets
    api_requests: int = 0
    text_bytes: int = 0
    llm_requests: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    commits: int = 0
    work_items: int = 0
    exhausted: dict[str, str] = field(default_factory=dict)

    def _check(self, dimension: str, used: int, requested: int, limit: int) -> None:
        if used + requested > limit:
            self.exhausted[dimension] = (
                f"requested {requested}; used {used}; limit {limit}"
            )
            raise BudgetExceeded(dimension, requested, used, limit)

    def reserve_commit(self, count: int = 1) -> None:
        self._check("commits", self.commits, count, self.limits.max_commits)
        self.commits += count

    def reserve_work_item(self, count: int = 1) -> None:
        self._check("work_items", self.work_items, count, self.limits.max_work_items)
        self.work_items += count

    def reserve_api(self, count: int = 1) -> None:
        self._check("api_requests", self.api_requests, count, self.limits.max_api_requests)
        self.api_requests += count

    def reserve_text(self, byte_count: int) -> None:
        self._check("text_bytes", self.text_bytes, byte_count, self.limits.max_text_bytes)
        self.text_bytes += byte_count

    def reserve_llm(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        estimated_input_tokens: int | None = None,
        estimated_output_tokens: int | None = None,
    ) -> None:
        input_estimate = input_tokens if estimated_input_tokens is None else estimated_input_tokens
        output_estimate = output_tokens if estimated_output_tokens is None else estimated_output_tokens
        self._check("llm_requests", self.llm_requests, 1, self.limits.max_llm_requests)
        self._check(
            "llm_input_tokens", self.llm_input_tokens, input_tokens,
            self.limits.max_llm_input_tokens,
        )
        self._check(
            "llm_output_tokens", self.llm_output_tokens, output_tokens,
            self.limits.max_llm_output_tokens,
        )
        self.llm_requests += 1
        self.llm_input_tokens += input_tokens
        self.llm_output_tokens += output_tokens
        self.estimated_input_tokens += input_estimate
        self.estimated_output_tokens += output_estimate

    def assert_available(self, dimension: str) -> None:
        limits = {
            "api_requests": self.limits.max_api_requests,
            "text_bytes": self.limits.max_text_bytes,
            "llm_requests": self.limits.max_llm_requests,
            "llm_input_tokens": self.limits.max_llm_input_tokens,
            "llm_output_tokens": self.limits.max_llm_output_tokens,
            "commits": self.limits.max_commits,
            "work_items": self.limits.max_work_items,
        }
        if dimension not in limits:
            raise KeyError(dimension)
        used = getattr(self, dimension)
        if used >= limits[dimension]:
            self.exhausted.setdefault(
                dimension, f"used {used}; limit {limits[dimension]}"
            )
            raise BudgetExceeded(dimension, 0, used, limits[dimension])

    def counters(self) -> dict[str, int]:
        return {
            "commits": self.commits,
            "work_items": self.work_items,
            "api_requests": self.api_requests,
            "text_bytes": self.text_bytes,
            "llm_requests": self.llm_requests,
            "llm_input_tokens": self.llm_input_tokens,
            "llm_output_tokens": self.llm_output_tokens,
        }
