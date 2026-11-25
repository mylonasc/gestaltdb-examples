"""Replayable event contracts."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from repo_history.events import EventBatch, EventKind, IngestEvent
from repo_history.ids import stable_id

UTC = timezone.utc
EVENT_TIME = datetime(2026, 9, 14, 20, 15, 31, tzinfo=UTC)


def _event(**overrides) -> IngestEvent:
    data = {
        "event_id": stable_id("event", "github", "pull:10"),
        "kind": EventKind.WORK_ITEM_OBSERVED,
        "source": "github",
        "source_id": "pull:10",
        "event_time": EVENT_TIME,
        "observed_at": EVENT_TIME + timedelta(minutes=5),
        "info_horizon": EVENT_TIME + timedelta(minutes=5),
        "payload": {"number": 10},
    }
    data.update(overrides)
    return IngestEvent(**data)


def test_event_keeps_source_and_observation_time_separate():
    event = _event()
    assert event.event_time == EVENT_TIME
    assert event.observed_at > event.event_time
    assert event.kind is EventKind.WORK_ITEM_OBSERVED


def test_event_horizon_cannot_precede_event_time():
    with pytest.raises(ValidationError):
        _event(info_horizon=EVENT_TIME - timedelta(seconds=1))


def test_event_requires_aware_timestamps():
    with pytest.raises(ValidationError):
        _event(event_time=datetime(2026, 9, 14, 20, 15, 31))


def test_order_key_is_deterministic_with_explicit_tie_breakers():
    first = _event(sequence=0)
    second = _event(sequence=1)
    assert first.order_key < second.order_key
    assert first.order_key[0] == EVENT_TIME
    assert first.order_key[1] == "github"


def test_event_batch_defaults_to_empty():
    batch = EventBatch()
    assert batch.events == ()
    with pytest.raises(ValidationError):
        batch.events = (_event(),)
