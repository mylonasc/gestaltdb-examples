"""LangChain LLM enrichment with deterministic fake chat models.

No network and no credentials: every model is an in-process
``GenericFakeChatModel`` with scripted replies, so retries, budgets, caching,
and validation are fully reproducible.  The single real-model smoke runs
outside this suite, explicitly enabled.
"""

from __future__ import annotations

import itertools
import json
from datetime import datetime, timezone

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from repo_history.budgets import BudgetLedger
from repo_history.enrichment import (
    Enricher,
    EnrichmentCache,
    EnrichmentConsentError,
    EnrichmentError,
    EnrichmentModelError,
    build_chat_model,
    check_model,
    estimate_tokens,
    gap_for_unavailable_history,
    require_consent,
)
from repo_history.analysis import DecisionStatus
from repo_history.models import Budgets, CoverageStatus, LLMSettings

UTC = timezone.utc
NOW = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)
HORIZON = datetime(2026, 9, 14, 12, 0, tzinfo=UTC)
CANDIDATES = ({"topic_id": "topic:sparse", "name": "sparse-training"},)


def fake_model(*contents: str, usage: dict | None = None) -> GenericFakeChatModel:
    messages = [
        AIMessage(
            content=content,
            **(
                {"usage_metadata": {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}}
                if usage
                else {}
            ),
        )
        for content in contents
    ]
    return GenericFakeChatModel(messages=iter(messages))


def cycling_model(*contents: str) -> GenericFakeChatModel:
    return GenericFakeChatModel(
        messages=itertools.cycle([AIMessage(content=c) for c in contents])
    )


class NamedFakeModel(GenericFakeChatModel):
    """Fake chat model with a stable, introspectable model name."""

    model_name: str = "fake-default"


def enricher(
    chat_model,
    tmp_path,
    *,
    budgets: Budgets | None = None,
    cache_path=None,
    **settings_overrides,
) -> tuple[Enricher, BudgetLedger]:
    settings_data = {"enabled": True, "privacy_consent": True}
    settings_data.update(settings_overrides)
    settings = LLMSettings(**settings_data)
    ledger = BudgetLedger(budgets or Budgets())
    cache = EnrichmentCache(
        cache_path if cache_path is not None else tmp_path / "cache.json"
    )
    return (
        Enricher(chat_model, settings, ledger=ledger, cache=cache, run_id="run-1"),
        ledger,
    )


def reuse_payload(topic_id: str = "topic:sparse") -> str:
    return json.dumps(
        {
            "action": "reuse",
            "topic_id": topic_id,
            "confidence": 0.9,
            "rationale": "The subject discusses sparse training.",
            "evidence_quotes": ["sparse training"],
        }
    )


# ---------------------------------------------------------------------------
# Consent and capabilities
# ---------------------------------------------------------------------------


def test_consent_is_required_before_anything_runs(tmp_path):
    with pytest.raises(EnrichmentConsentError):
        require_consent(LLMSettings())
    with pytest.raises(EnrichmentConsentError):
        require_consent(LLMSettings(enabled=True))
    require_consent(LLMSettings(enabled=True, privacy_consent=True))

    with pytest.raises(EnrichmentConsentError):
        Enricher(
            fake_model(reuse_payload()),
            LLMSettings(),
            ledger=BudgetLedger(Budgets()),
            cache=EnrichmentCache(tmp_path / "cache.json"),
            run_id="run-1",
        )


def test_capability_check_rejects_non_chat_models():
    with pytest.raises(EnrichmentModelError):
        check_model("not-a-model")
    with pytest.raises(EnrichmentModelError):
        check_model(object())

    assert "Fake" in check_model(fake_model(reuse_payload())) or check_model(
        fake_model(reuse_payload())
    )


def test_build_chat_model_rejects_unknown_providers():
    settings = LLMSettings(
        enabled=True, privacy_consent=True, provider="mystery", model="m-1"
    )
    with pytest.raises(EnrichmentModelError):
        build_chat_model(settings)


def test_build_chat_model_reports_a_missing_openai_extra(monkeypatch):
    settings = LLMSettings(enabled=True, privacy_consent=True)
    monkeypatch.setitem(__import__("sys").modules, "langchain_openai", None)
    with pytest.raises(EnrichmentModelError) as caught:
        build_chat_model(settings)
    assert "langchain-openai" in str(caught.value)


# ---------------------------------------------------------------------------
# Topic proposals
# ---------------------------------------------------------------------------


def test_topic_reuse_is_grounded_cached_and_budgeted(tmp_path):
    model = fake_model(reuse_payload())
    the_enricher, ledger = enricher(model, tmp_path)

    result = the_enricher.enrich_topic(
        subject_id="workitem:github:10",
        subject_text="Training fails on sparse graphs.",
        candidates=CANDIDATES,
        registry_version=3,
        info_horizon=HORIZON,
    )

    assert result.status is DecisionStatus.ACCEPTED
    assert result.kind == "llm-topic"
    assert result.value["proposal"]["topic_id"] == "topic:sparse"
    assert result.value["cached"] is False
    assert result.value["model"].startswith("openai:")
    assert result.value["prompt_version"] == "v1"
    assert result.value["schema_version"] == "1"
    assert result.provenance.model.startswith("openai:")
    assert result.provenance.prompt_version == "v1"
    assert result.provenance.schema_version == "1"
    assert result.provenance.info_horizon == HORIZON
    assert ledger.llm_requests == 1
    assert ledger.estimated_input_tokens > 0
    assert ledger.llm_input_tokens == ledger.estimated_input_tokens

    repeat = the_enricher.enrich_topic(
        subject_id="workitem:github:10",
        subject_text="Training fails on sparse graphs.",
        candidates=CANDIDATES,
        registry_version=3,
        info_horizon=HORIZON,
    )

    assert repeat.value["cached"] is True
    assert repeat.value["proposal"] == result.value["proposal"]
    assert the_enricher.invocations == 1
    assert ledger.llm_requests == 1  # cache hits spend no budget


def test_cache_invalidated_by_prompt_context_or_model(tmp_path):
    cache_path = tmp_path / "cache.json"
    first, _ = enricher(fake_model(reuse_payload()), tmp_path, cache_path=cache_path)
    first.enrich_topic(
        subject_id="s", subject_text="sparse graphs", candidates=CANDIDATES,
        registry_version=1, info_horizon=HORIZON,
    )
    assert first.invocations == 1

    changed_prompt, _ = enricher(
        fake_model(reuse_payload()), tmp_path, cache_path=cache_path,
        prompt_version="v2",
    )
    changed_prompt.enrich_topic(
        subject_id="s", subject_text="sparse graphs", candidates=CANDIDATES,
        registry_version=1, info_horizon=HORIZON,
    )
    assert changed_prompt.invocations == 1

    changed_context, _ = enricher(
        fake_model(reuse_payload()), tmp_path, cache_path=cache_path
    )
    changed_context.enrich_topic(
        subject_id="s", subject_text="dense graphs instead", candidates=CANDIDATES,
        registry_version=1, info_horizon=HORIZON,
    )
    assert changed_context.invocations == 1

    changed_model, _ = enricher(
        NamedFakeModel(
            messages=iter([AIMessage(content=reuse_payload())]),
            model_name="gpt-4.1",
        ),
        tmp_path,
        cache_path=cache_path,
        model="gpt-4.1",
    )
    changed_model.enrich_topic(
        subject_id="s", subject_text="sparse graphs", candidates=CANDIDATES,
        registry_version=1, info_horizon=HORIZON,
    )
    assert changed_model.invocations == 1


def test_invalid_output_is_retried_then_accepted(tmp_path):
    model = fake_model("not json at all", reuse_payload())
    enricher_, ledger = enricher(model, tmp_path)

    result = enricher_.enrich_topic(
        subject_id="s", subject_text="sparse graphs", candidates=CANDIDATES,
        registry_version=1, info_horizon=HORIZON,
    )

    assert result.status is DecisionStatus.ACCEPTED
    assert enricher_.invocations == 2
    assert ledger.llm_requests == 2  # retries spend budget too


def test_persistently_invalid_output_abstains_without_caching(tmp_path):
    model = cycling_model("garbage {{{")
    enricher_, ledger = enricher(model, tmp_path, max_attempts=2)

    first = enricher_.enrich_topic(
        subject_id="s", subject_text="sparse graphs", candidates=CANDIDATES,
        registry_version=1, info_horizon=HORIZON,
    )

    assert first.status is DecisionStatus.ABSTAINED
    assert first.value["proposal"] is None
    assert enricher_.invocations == 2

    second = enricher_.enrich_topic(
        subject_id="s", subject_text="sparse graphs", candidates=CANDIDATES,
        registry_version=1, info_horizon=HORIZON,
    )
    assert second.status is DecisionStatus.ABSTAINED
    assert enricher_.invocations == 4  # validation failures are not cached


def test_model_chosen_abstention_is_cached(tmp_path):
    abstain = json.dumps({"action": "abstain", "rationale": "Not enough evidence."})
    model = cycling_model(abstain)
    enricher_, _ = enricher(model, tmp_path)

    first = enricher_.enrich_topic(
        subject_id="s", subject_text="???", candidates=(),
        registry_version=1, info_horizon=HORIZON,
    )
    second = enricher_.enrich_topic(
        subject_id="s", subject_text="???", candidates=(),
        registry_version=1, info_horizon=HORIZON,
    )

    assert first.status is DecisionStatus.ABSTAINED
    assert second.value["cached"] is True
    assert enricher_.invocations == 1


def test_ungrounded_reuse_is_rejected_then_abstained(tmp_path):
    model = cycling_model(reuse_payload(topic_id="topic:invented"))
    enricher_, _ = enricher(model, tmp_path, max_attempts=2)

    result = enricher_.enrich_topic(
        subject_id="s", subject_text="sparse graphs", candidates=CANDIDATES,
        registry_version=1, info_horizon=HORIZON,
    )

    assert result.status is DecisionStatus.ABSTAINED
    assert enricher_.invocations == 2


def test_budget_exhaustion_returns_an_explicit_envelope(tmp_path):
    model = cycling_model(reuse_payload())
    enricher_, ledger = enricher(
        model, tmp_path, budgets=Budgets(max_llm_requests=1)
    )

    first = enricher_.enrich_topic(
        subject_id="s1", subject_text="sparse graphs", candidates=CANDIDATES,
        registry_version=1, info_horizon=HORIZON,
    )
    second = enricher_.enrich_topic(
        subject_id="s2", subject_text="dense graphs", candidates=CANDIDATES,
        registry_version=1, info_horizon=HORIZON,
    )

    assert first.status is DecisionStatus.ACCEPTED
    assert second.status is DecisionStatus.BUDGET_EXHAUSTED
    assert second.value["reason"]
    assert ledger.exhausted.get("llm_requests")

    repeat = enricher_.enrich_topic(
        subject_id="s1", subject_text="sparse graphs", candidates=CANDIDATES,
        registry_version=1, info_horizon=HORIZON,
    )
    assert repeat.value["cached"] is True  # served without spending


def test_usage_metadata_is_recorded_when_reported(tmp_path):
    model = fake_model(reuse_payload(), usage=True)
    enricher_, ledger = enricher(model, tmp_path)

    result = enricher_.enrich_topic(
        subject_id="s", subject_text="sparse graphs", candidates=CANDIDATES,
        registry_version=1, info_horizon=HORIZON,
    )

    assert result.provenance.actual_input_tokens == 7
    assert result.provenance.actual_output_tokens == 3


# ---------------------------------------------------------------------------
# Summaries and links
# ---------------------------------------------------------------------------


def test_summary_proposal_is_accepted(tmp_path):
    payload = json.dumps(
        {
            "summary": "Version bump for the release.",
            "key_points": ["bumps version", "touches one file"],
            "confidence": 0.8,
            "rationale": "The diff is tiny.",
        }
    )
    enricher_, _ = enricher(fake_model(payload), tmp_path)

    result = enricher_.enrich_summary(
        subject_id="workitem:github:11",
        title="updated version",
        text="Bumps the version number.",
        info_horizon=HORIZON,
    )

    assert result.status is DecisionStatus.ACCEPTED
    assert result.kind == "llm-summary"
    assert result.value["proposal"]["summary"].startswith("Version bump")


def test_link_to_unknown_target_is_rejected(tmp_path):
    payload = json.dumps(
        {
            "links": [
                {
                    "target_id": "workitem:github:999",
                    "relation": "INTENDS_TO_RESOLVE",
                    "rationale": "Seems related.",
                    "confidence": 0.6,
                }
            ],
            "rationale": "one link",
        }
    )
    enricher_, _ = enricher(
        cycling_model(payload), tmp_path, max_attempts=1
    )

    result = enricher_.enrich_links(
        subject_id="workitem:github:10",
        subject_text="Fixes the sparse bug.",
        candidate_targets={"workitem:github:11": "Version bump"},
        info_horizon=HORIZON,
    )

    assert result.status is DecisionStatus.ABSTAINED


def test_link_proposal_accepts_only_listed_targets(tmp_path):
    payload = json.dumps(
        {
            "links": [
                {
                    "target_id": "commit:abc",
                    "relation": "MENTIONS",
                    "rationale": "Mentions the commit.",
                    "confidence": 0.7,
                }
            ],
            "rationale": "one grounded link",
        }
    )
    enricher_, _ = enricher(fake_model(payload), tmp_path)

    result = enricher_.enrich_links(
        subject_id="workitem:github:10",
        subject_text="See commit abc.",
        candidate_targets={"commit:abc": "Fix commit"},
        info_horizon=HORIZON,
    )

    assert result.status is DecisionStatus.ACCEPTED
    assert result.value["proposal"]["links"][0]["target_id"] == "commit:abc"


def test_unavailable_history_becomes_a_gap_not_a_prompt():
    gap = gap_for_unavailable_history("workitem:github:10", "no timeline access")

    assert gap.code == "llm-history-unavailable"
    assert gap.status is CoverageStatus.UNAVAILABLE
    assert gap.source == "llm"


def test_estimate_tokens_is_deterministic_and_positive():
    assert estimate_tokens("") == 1
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcdefgh") == 2
    assert estimate_tokens("x" * 100) == estimate_tokens("x" * 100)


def test_llm_settings_load_new_fields_from_toml():
    from repo_history.config import load_config

    config = load_config()
    assert config.llm.prompt_version == "v1"
    assert config.llm.cache_file == "data/llm/cache.json"


def test_enrichment_errors_are_plain_runtime_errors():
    assert issubclass(EnrichmentConsentError, EnrichmentError)
    assert issubclass(EnrichmentModelError, EnrichmentError)
