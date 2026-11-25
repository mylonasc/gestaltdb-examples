"""Validated domain contracts shared by adapters, analyzers, and the writer."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must include a UTC offset")
    return value.astimezone(timezone.utc)


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CoverageStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class CoverageGap(FrozenModel):
    code: str
    status: CoverageStatus
    message: str
    source: str
    source_ids: tuple[str, ...] = ()
    retryable: bool = False


class TimeWindow(FrozenModel):
    start: datetime
    end: datetime

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        if self.start >= self.end:
            raise ValueError("window start must precede end")
        return self

    @field_validator("start", "end")
    @classmethod
    def utc(cls, value: datetime) -> datetime:
        return ensure_utc(value)

    def contains(self, value: datetime) -> bool:
        return self.start <= ensure_utc(value) < self.end


class RepositorySpec(FrozenModel):
    provider: str = "github"
    owner: str
    name: str
    url: str
    anchor: str = Field(pattern=r"^[0-9a-fA-F]{40}$")
    window: TimeWindow
    allowed_work_items: tuple[int, ...] = ()
    excluded_work_items: tuple[int, ...] = ()

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        overlap = set(self.allowed_work_items) & set(self.excluded_work_items)
        if overlap:
            raise ValueError(f"work items cannot be both included and excluded: {sorted(overlap)}")
        return self

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.name}"


class Budgets(FrozenModel):
    max_commits: int = Field(default=32, ge=1, le=10_000)
    max_work_items: int = Field(default=25, ge=1, le=1_000)
    max_api_requests: int = Field(default=100, ge=1, le=10_000)
    max_text_bytes: int = Field(default=2 * 1024 * 1024, ge=1_024)
    max_llm_requests: int = Field(default=16, ge=0, le=1_000)
    max_llm_input_tokens: int = Field(default=64_000, ge=0)
    max_llm_output_tokens: int = Field(default=8_000, ge=0)


class LLMSettings(FrozenModel):
    enabled: bool = False
    provider: str = "openai"
    model: str = "gpt-4.1-mini"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    privacy_consent: bool = False
    max_attempts: int = Field(default=2, ge=1, le=5)


class RuntimeConfig(FrozenModel):
    repository: RepositorySpec
    budgets: Budgets = Field(default_factory=Budgets)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    data_root: str = "data"
    cache_dir: str = "data/cache"
    offline: bool = False

    @model_validator(mode="after")
    def validate_llm_consent(self) -> Self:
        if self.llm.enabled and not self.llm.privacy_consent:
            raise ValueError("enabling an LLM requires explicit privacy_consent")
        return self


class RunManifest(FrozenModel):
    run_id: str
    repository_id: str
    repository_url: str
    anchor: str
    window: TimeWindow
    started_at: datetime
    finished_at: datetime | None = None
    software_version: str
    config_fingerprint: str
    selected_commits: tuple[str, ...] = ()
    selected_work_items: tuple[int, ...] = ()
    gaps: tuple[CoverageGap, ...] = ()
    counters: dict[str, int] = Field(default_factory=dict)
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0

    @field_validator("started_at", "finished_at")
    @classmethod
    def utc(cls, value: datetime | None) -> datetime | None:
        return ensure_utc(value) if value is not None else None


class EvidenceRef(FrozenModel):
    evidence_id: str
    source: str
    source_id: str
    locator: str = ""
    url: str | None = None
    excerpt: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
