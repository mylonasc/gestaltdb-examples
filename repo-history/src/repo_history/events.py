"""Normalized, replayable events independent of any remote API shape."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Self

from pydantic import Field, field_validator, model_validator

from .models import FrozenModel, ensure_utc


class EventKind(StrEnum):
    BASELINE = "baseline"
    COMMIT = "commit"
    WORK_ITEM_OBSERVED = "work-item-observed"
    WORK_ITEM_EVENT = "work-item-event"


class IngestEvent(FrozenModel):
    event_id: str
    kind: EventKind
    source: str
    source_id: str
    event_time: datetime
    observed_at: datetime
    info_horizon: datetime
    payload: dict[str, Any]
    dependencies: tuple[str, ...] = ()
    sequence: int = 0

    @field_validator("event_time", "observed_at", "info_horizon")
    @classmethod
    def utc(cls, value: datetime) -> datetime:
        return ensure_utc(value)

    @model_validator(mode="after")
    def horizon_not_before_event(self) -> Self:
        if self.info_horizon < self.event_time:
            raise ValueError("information horizon cannot precede the event time")
        return self

    @property
    def order_key(self) -> tuple[datetime, str, str, int]:
        return self.event_time, self.source, self.source_id, self.sequence


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


class EventBatch(FrozenModel):
    events: tuple[IngestEvent, ...] = ()
    gaps: tuple[Any, ...] = Field(default_factory=tuple)
