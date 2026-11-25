"""Validated domain model contracts."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from repo_history.models import (
    Budgets,
    CoverageGap,
    CoverageStatus,
    LLMSettings,
    RepositorySpec,
    RuntimeConfig,
    TimeWindow,
)

UTC = timezone.utc
START = datetime(2026, 9, 13, tzinfo=UTC)
END = datetime(2026, 9, 15, tzinfo=UTC)


def _repository(**overrides) -> RepositorySpec:
    data = {
        "owner": "mylonasc",
        "name": "tf_gnns",
        "url": "https://github.com/mylonasc/tf_gnns",
        "anchor": "9bfe6c19c02723dbbf101b9e311de684a4728bb1",
        "window": TimeWindow(start=START, end=END),
        "allowed_work_items": (10, 11, 12),
    }
    data.update(overrides)
    return RepositorySpec(**data)


def test_time_window_requires_ordered_aware_timestamps():
    assert TimeWindow(start=START, end=END).contains(datetime(2026, 9, 14, 12, tzinfo=UTC))
    assert not TimeWindow(start=START, end=END).contains(END)
    with pytest.raises(ValidationError):
        TimeWindow(start=END, end=START)
    with pytest.raises(ValidationError):
        TimeWindow(start=datetime(2026, 9, 13), end=END)


def test_time_window_normalizes_to_utc():
    shifted = TimeWindow(
        start=datetime(2026, 9, 13, 2, tzinfo=timezone(timedelta(hours=2))), end=END
    )
    assert shifted.start == START
    assert shifted.start.tzinfo is UTC


def test_repository_requires_sha_anchor_and_disjoint_selection():
    assert _repository().full_name == "mylonasc/tf_gnns"
    with pytest.raises(ValidationError):
        _repository(anchor="main")
    with pytest.raises(ValidationError):
        _repository(allowed_work_items=(10,), excluded_work_items=(10,))


def test_budget_defaults_match_epic_ceilings():
    budgets = Budgets()
    assert budgets.max_commits == 32
    assert budgets.max_work_items == 25
    assert budgets.max_api_requests == 100
    assert budgets.max_text_bytes == 2 * 1024 * 1024
    assert budgets.max_llm_requests == 16
    assert budgets.max_llm_input_tokens == 64_000
    assert budgets.max_llm_output_tokens == 8_000


@pytest.mark.parametrize(
    "field,value",
    [("max_commits", 0), ("max_work_items", 0), ("max_api_requests", 0), ("max_text_bytes", 1)],
)
def test_budgets_reject_non_positive_ceilings(field: str, value: int):
    with pytest.raises(ValidationError):
        Budgets(**{field: value})


def test_llm_requires_explicit_privacy_consent_when_enabled():
    with pytest.raises(ValidationError):
        RuntimeConfig(repository=_repository(), llm=LLMSettings(enabled=True))
    config = RuntimeConfig(
        repository=_repository(), llm=LLMSettings(enabled=True, privacy_consent=True)
    )
    assert config.llm.enabled


def test_runtime_config_defaults_to_offline_safe_import():
    config = RuntimeConfig(repository=_repository())
    assert config.offline is False
    assert config.llm.enabled is False
    assert config.data_root == "data"
    assert config.git.max_depth == 256


def test_coverage_gap_is_immutable_and_typed():
    gap = CoverageGap(
        code="issue-history",
        status=CoverageStatus.UNAVAILABLE,
        message="historical edit revisions are unavailable",
        source="github",
        source_ids=("issue:10",),
    )
    assert gap.status is CoverageStatus.UNAVAILABLE
    with pytest.raises(ValidationError):
        gap.message = "changed"
