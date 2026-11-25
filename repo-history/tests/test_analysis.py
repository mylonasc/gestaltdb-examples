"""Structured analysis envelopes and decisions."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from repo_history.analysis import (
    AnalysisResult,
    DecisionStatus,
    Provenance,
    RelationshipDecision,
    TopicDecision,
)
from repo_history.models import EvidenceRef

UTC = timezone.utc
HORIZON = datetime(2026, 9, 14, 21, 15, 58, tzinfo=UTC)
EVIDENCE = (
    EvidenceRef(
        evidence_id="evidence:1",
        source="git",
        source_id="9bfe6c1",
        locator="README.md:1-20",
        url="https://example.test/README.md",
    ),
)


def _provenance(**overrides) -> Provenance:
    data = {
        "method": "deterministic-path-labeler",
        "method_version": "1",
        "run_id": "run:1",
        "info_horizon": HORIZON,
    }
    data.update(overrides)
    return Provenance(**data)


def test_result_identity_is_deterministic_for_identical_decisions():
    first = AnalysisResult.create(
        subject_id="commit:1",
        kind="topic",
        status=DecisionStatus.ACCEPTED,
        value={"topics": ["documentation"]},
        provenance=_provenance(),
        evidence=EVIDENCE,
    )
    second = AnalysisResult.create(
        subject_id="commit:1",
        kind="topic",
        status=DecisionStatus.ACCEPTED,
        value={"topics": ["documentation"]},
        provenance=_provenance(),
        evidence=EVIDENCE,
    )
    assert first.result_id == second.result_id
    assert first.decision_key == second.decision_key


def test_result_identity_changes_with_horizon_or_method():
    base = dict(
        subject_id="commit:1",
        kind="topic",
        status=DecisionStatus.ACCEPTED,
        value={"topics": ["documentation"]},
        evidence=EVIDENCE,
    )
    first = AnalysisResult.create(provenance=_provenance(), **base)
    later = AnalysisResult.create(
        provenance=_provenance(info_horizon=HORIZON.replace(hour=22)), **base
    )
    other = AnalysisResult.create(
        provenance=_provenance(method_version="2"), **base
    )
    assert first.result_id != later.result_id
    assert first.result_id != other.result_id


@pytest.mark.parametrize(
    "status",
    [DecisionStatus.INVALID, DecisionStatus.FAILED, DecisionStatus.BUDGET_EXHAUSTED],
)
def test_failure_statuses_require_an_error(status: DecisionStatus):
    with pytest.raises(ValidationError):
        AnalysisResult.create(
            subject_id="commit:1",
            kind="topic",
            status=status,
            value={},
            provenance=_provenance(),
            evidence=EVIDENCE,
        )
    result = AnalysisResult.create(
        subject_id="commit:1",
        kind="topic",
        status=status,
        value={},
        provenance=_provenance(),
        evidence=EVIDENCE,
        error="provider unavailable",
    )
    assert result.error == "provider unavailable"


def test_provenance_records_estimated_and_actual_usage():
    provenance = _provenance(
        method="langchain",
        model="openai:gpt-4.1-mini",
        prompt_version="topics-v1",
        schema_version="topic-decision-v1",
        estimated_input_tokens=900,
        estimated_output_tokens=120,
        actual_input_tokens=870,
        actual_output_tokens=96,
    )
    assert provenance.actual_input_tokens == 870
    assert provenance.estimated_input_tokens == 900


def test_topic_decision_requires_exactly_one_choice():
    assert TopicDecision(reuse_topic_id="topic:1", confidence=0.9, rationale="match").reuse_topic_id
    assert TopicDecision(proposed_name="Benchmarking", confidence=0.5, rationale="new").proposed_name
    with pytest.raises(ValidationError):
        TopicDecision(confidence=0.5, rationale="neither")
    with pytest.raises(ValidationError):
        TopicDecision(
            reuse_topic_id="topic:1", proposed_name="Benchmarking",
            confidence=0.5, rationale="both",
        )
    with pytest.raises(ValidationError):
        TopicDecision(proposed_name="X", confidence=1.5, rationale="bad score")


def test_relationship_decision_is_bounded_and_typed():
    decision = RelationshipDecision(
        relation="MENTIONS", target_id="workitem:10", confidence=0.7, rationale="text reference"
    )
    assert decision.relation == "MENTIONS"
    with pytest.raises(ValidationError):
        RelationshipDecision(
            relation="FIXES", target_id="workitem:10", confidence=0.7, rationale="x"
        )
