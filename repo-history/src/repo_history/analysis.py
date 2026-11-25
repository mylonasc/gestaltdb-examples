"""Structured analysis result envelopes and decision statuses."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from .ids import stable_id
from .models import EvidenceRef, FrozenModel, ensure_utc


class DecisionStatus(StrEnum):
    ACCEPTED = "accepted"
    PROPOSED = "proposed"
    ABSTAINED = "abstained"
    INVALID = "invalid"
    FAILED = "failed"
    BUDGET_EXHAUSTED = "budget-exhausted"


class Provenance(FrozenModel):
    method: str
    method_version: str
    run_id: str
    info_horizon: datetime
    model: str | None = None
    prompt_version: str | None = None
    schema_version: str | None = None
    estimated_input_tokens: int = Field(default=0, ge=0)
    estimated_output_tokens: int = Field(default=0, ge=0)
    actual_input_tokens: int = Field(default=0, ge=0)
    actual_output_tokens: int = Field(default=0, ge=0)

    @field_validator("info_horizon")
    @classmethod
    def utc(cls, value: datetime) -> datetime:
        return ensure_utc(value)


class AnalysisResult(FrozenModel):
    result_id: str
    subject_id: str
    kind: str
    status: DecisionStatus
    value: dict[str, Any]
    evidence: tuple[EvidenceRef, ...] = ()
    provenance: Provenance
    decision_key: str
    error: str | None = None

    @model_validator(mode="after")
    def failure_has_error(self) -> Self:
        if self.status in {DecisionStatus.INVALID, DecisionStatus.FAILED, DecisionStatus.BUDGET_EXHAUSTED}:
            if not self.error:
                raise ValueError("invalid, failed, and exhausted results require an error")
        return self

    @classmethod
    def create(
        cls,
        *,
        subject_id: str,
        kind: str,
        status: DecisionStatus,
        value: dict[str, Any],
        provenance: Provenance,
        evidence: tuple[EvidenceRef, ...] = (),
        error: str | None = None,
    ) -> Self:
        evidence_ids = tuple(sorted(ref.evidence_id for ref in evidence))
        decision_key = stable_id(
            "decision", subject_id, kind, provenance.method, provenance.method_version,
            provenance.info_horizon.isoformat(), status, *evidence_ids,
        )
        return cls(
            result_id=stable_id("result", decision_key),
            subject_id=subject_id,
            kind=kind,
            status=status,
            value=value,
            evidence=evidence,
            provenance=provenance,
            decision_key=decision_key,
            error=error,
        )


class TopicDecision(FrozenModel):
    reuse_topic_id: str | None = None
    proposed_name: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    evidence_text: str | None = None

    @model_validator(mode="after")
    def one_choice(self) -> Self:
        if bool(self.reuse_topic_id) == bool(self.proposed_name):
            raise ValueError("choose exactly one of reuse_topic_id or proposed_name")
        return self


class RelationshipDecision(FrozenModel):
    relation: Literal["MENTIONS", "INTENDS_TO_RESOLVE", "RELATED_TO"]
    target_id: str
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    evidence_text: str | None = None
