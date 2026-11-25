"""Fail-closed budget accounting."""

from __future__ import annotations

import pytest

from repo_history.budgets import BudgetExceeded, BudgetLedger
from repo_history.models import Budgets


def _ledger(**overrides) -> BudgetLedger:
    return BudgetLedger(limits=Budgets(**overrides))


def test_reservations_accumulate_and_report_counters():
    ledger = _ledger()
    ledger.reserve_commit(16)
    ledger.reserve_work_item(3)
    ledger.reserve_api(12)
    ledger.reserve_text(4096)
    ledger.reserve_llm(input_tokens=1000, output_tokens=200)
    assert ledger.counters() == {
        "commits": 16,
        "work_items": 3,
        "api_requests": 12,
        "text_bytes": 4096,
        "llm_requests": 1,
        "llm_input_tokens": 1000,
        "llm_output_tokens": 200,
    }


def test_commit_limit_fails_closed_and_records_reason():
    ledger = _ledger(max_commits=2)
    ledger.reserve_commit(2)
    with pytest.raises(BudgetExceeded) as error:
        ledger.reserve_commit()
    assert error.value.dimension == "commits"
    assert ledger.commits == 2
    assert "limit 2" in ledger.exhausted["commits"]


def test_api_limit_counts_retries():
    ledger = _ledger(max_api_requests=2)
    ledger.reserve_api()
    ledger.reserve_api()
    with pytest.raises(BudgetExceeded):
        ledger.reserve_api()  # a retry still consumes the request budget
    assert ledger.api_requests == 2


def test_text_limit_is_in_bytes():
    ledger = _ledger(max_text_bytes=1024)
    ledger.reserve_text(1024)
    with pytest.raises(BudgetExceeded):
        ledger.reserve_text(1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_tokens": 11, "output_tokens": 1},
        {"input_tokens": 1, "output_tokens": 11},
    ],
)
def test_llm_token_ceilings_block_before_mutating(kwargs):
    ledger = _ledger(max_llm_input_tokens=10, max_llm_output_tokens=10)
    with pytest.raises(BudgetExceeded):
        ledger.reserve_llm(**kwargs)
    assert ledger.llm_requests == 0
    assert ledger.llm_input_tokens == 0
    assert ledger.llm_output_tokens == 0


def test_llm_request_ceiling_blocks_entirely():
    ledger = _ledger(max_llm_requests=1)
    ledger.reserve_llm(input_tokens=1, output_tokens=1)
    with pytest.raises(BudgetExceeded):
        ledger.reserve_llm(input_tokens=1, output_tokens=1)
    assert ledger.llm_requests == 1


def test_estimated_and_actual_usage_are_separate():
    ledger = _ledger()
    ledger.reserve_llm(
        input_tokens=80,
        output_tokens=20,
        estimated_input_tokens=100,
        estimated_output_tokens=25,
    )
    assert ledger.llm_input_tokens == 80
    assert ledger.estimated_input_tokens == 100
    assert ledger.llm_output_tokens == 20
    assert ledger.estimated_output_tokens == 25


def test_assert_available_marks_exhaustion():
    ledger = _ledger(max_commits=1)
    ledger.reserve_commit()
    with pytest.raises(BudgetExceeded):
        ledger.assert_available("commits")
    with pytest.raises(KeyError):
        ledger.assert_available("unknown")
