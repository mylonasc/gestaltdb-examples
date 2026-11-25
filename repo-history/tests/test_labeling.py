"""Deterministic and classical-NLP topic labeling over synthetic fixtures.

No network, no models, no database: every scenario builds its features,
registry, and work items in-process with fixed timestamps.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from repo_history.analysis import DecisionStatus
from repo_history.content import (
    CodeDocument,
    CodeSymbol,
    ImportRecord,
    MarkdownDocument,
    MarkdownHeading,
    MarkdownLink,
    ParsedContent,
    SnapshotContent,
)
from repo_history.github import GitHubWorkItem, RevisionAvailability
from repo_history.ids import IdFactory
from repo_history.labeling import (
    ContentFeatures,
    DeterministicLabeler,
    TopicRegistry,
    extract_snapshot_features,
    extract_work_item_features,
    seed_registry,
)
from repo_history.models import LabelingSettings
from repo_history.protocols import Labeler

UTC = timezone.utc
NOW = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)
HORIZON_ONE = datetime(2026, 9, 14, 12, 0, tzinfo=UTC)
HORIZON_TWO = datetime(2026, 9, 14, 18, 0, tzinfo=UTC)
REPOSITORY_URL = "https://github.com/mylonasc/tf_gnns"


def ids() -> IdFactory:
    return IdFactory(REPOSITORY_URL)


def markdown_content() -> ParsedContent:
    return ParsedContent(
        content_id="content:md",
        blob_id="blob:md",
        oid="o" * 40,
        media_type="markdown",
        status="parsed",
        language="markdown",
        byte_size=120,
        text_bytes=120,
        line_count=4,
        sha256="s" * 64,
        markdown=MarkdownDocument(
            headings=(
                MarkdownHeading(
                    level=1, text="Graph Neural Networks", slug="graph-neural-networks",
                    line=1, end_line=1, offset=0, end_offset=22,
                ),
            ),
            links=(
                MarkdownLink(
                    kind="link", text="guide", target="docs/training-guide.md",
                    line=3, offset=0, end_offset=20,
                ),
            ),
        ),
    )


def code_content() -> ParsedContent:
    return ParsedContent(
        content_id="content:py",
        blob_id="blob:py",
        oid="p" * 40,
        media_type="python",
        status="parsed",
        language="python",
        byte_size=200,
        text_bytes=200,
        line_count=10,
        sha256="t" * 64,
        code=CodeDocument(
            language="python",
            imports=(
                ImportRecord(kind="import", module="torch_geometric.nn", names=(), line=1),
            ),
            symbols=(
                CodeSymbol(
                    symbol_id="sym:1", kind="class", name="GraphConv",
                    qualname="models.GraphConv", line=1, column=0, end_line=8,
                ),
            ),
        ),
    )


def snapshot(
    path: str, language: str | None = None, module: str | None = None
) -> SnapshotContent:
    return SnapshotContent(
        snapshot_id=f"snap:{path}",
        commit_sha="c" * 40,
        path=path,
        oid="o" * 40,
        content_id="content:x",
        status="parsed",
        media_type="markdown" if path.endswith(".md") else "python",
        language=language,
        module=module,
        is_baseline=False,
        source_url=f"{REPOSITORY_URL}/blob/{'c' * 40}/{path}",
    )


def work_item(number: int, **overrides) -> GitHubWorkItem:
    data = {
        "number": number,
        "kind": "issue",
        "title": "Training fails on sparse graphs",
        "state": "open",
        "state_reason": None,
        "body": "The graph convolution overflows on sparse adjacency batches.",
        "author": "mylonasc",
        "created_at": HORIZON_ONE,
        "updated_at": HORIZON_ONE,
        "closed_at": None,
        "url": f"https://github.com/mylonasc/tf_gnns/issues/{number}",
        "labels": (),
        "comments": (),
        "timeline": (),
        "revisions": RevisionAvailability(work_item_number=number),
        "observed_at": NOW,
        "source_hash": "0" * 64,
    }
    data.update(overrides)
    return GitHubWorkItem(**data)


def registry_with(*names: str, at: datetime = NOW) -> TopicRegistry:
    registry = seed_registry(at=at)
    for name in names:
        registry, _ = registry.register(name, at=at, created_by="test", ids=ids())
    return registry


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def test_snapshot_features_are_deterministic_and_never_execute_code():
    first = extract_snapshot_features(
        snapshot("docs/guide.md", language="markdown"), markdown_content(),
        subject_id="snap:docs/guide.md",
    )
    second = extract_snapshot_features(
        snapshot("docs/guide.md", language="markdown"), markdown_content(),
        subject_id="snap:docs/guide.md",
    )

    assert first == second
    assert "graph" in first.heading_tokens
    assert "networks" in first.heading_tokens
    assert "training-guide" in first.link_targets or "training" in first.link_targets
    assert first.path_tokens[:2] == ("docs", "guide")


def test_code_features_cover_language_module_imports_and_symbols():
    features = extract_snapshot_features(
        snapshot("models/gcn.py", language="python", module="models.gcn"),
        code_content(),
        subject_id="snap:models/gcn.py",
    )

    assert features.language == "python"
    assert features.module == "models.gcn"
    assert "torch" in features.import_modules or "geometric" in features.import_modules
    assert "graphconv" in features.symbol_tokens
    assert "models" in features.symbol_tokens


def test_missing_content_yields_path_only_features():
    features = extract_snapshot_features(
        snapshot("README.md"), None, subject_id="snap:README.md"
    )

    assert features.path_tokens == ("readme", "md")
    assert features.heading_tokens == ()
    assert features.text() == "readme md"


def test_work_item_features_keep_provider_labels_attached():
    features = extract_work_item_features(
        work_item(10, labels=("bug", "sparse-support")),
        subject_id="workitem:github:10",
    )

    assert "sparse" in features.body_tokens
    assert "training" in features.title_tokens
    assert features.provider_labels == ("bug", "sparse-support")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_starts_empty_and_versioned():
    registry = seed_registry(at=NOW)

    assert registry.version == 0
    assert registry.topics == ()
    assert registry.lookup("anything") is None


def test_registry_rejects_collisions_and_blank_names():
    registry = registry_with("graph-networks")

    with pytest.raises(ValueError):
        registry.register("graph-networks", at=NOW, created_by="test", ids=ids())
    with pytest.raises(ValueError):
        registry.register("Graph-Networks", at=NOW, created_by="test", ids=ids())
    with pytest.raises(ValueError):
        registry.register("  ", at=NOW, created_by="test", ids=ids())

    grown, record = registry.register(
        "sparse-training", at=NOW, created_by="test", ids=ids(),
        description="sparse training graphs",
    )
    assert grown.version == 2
    assert record.topic_id == ids().topic("sparse-training")
    assert "sparse" in record.text()
    assert registry.version == 1  # the original snapshot is unchanged


def test_registered_descriptions_make_proposals_reusable():
    labeler = DeterministicLabeler()
    features = ContentFeatures(
        subject_id="snap:models/gcn.py",
        path_tokens=("models", "gcn", "py"),
        language="python",
        symbol_tokens=("models", "graphconv"),
    )
    registry = seed_registry(at=NOW)
    first = labeler.assign(
        features, registry, subject_id=features.subject_id,
        run_id="run-1", info_horizon=HORIZON_ONE,
    )
    assert first.action == "propose"

    grown, record = registry.register(
        first.result.value["proposed_name"], at=NOW, created_by="run-1",
        ids=ids(), description=features.text(),
    )
    second = labeler.assign(
        features, grown, subject_id=features.subject_id,
        run_id="run-1", info_horizon=HORIZON_ONE,
    )
    assert second.action == "reuse"
    assert second.topic_id == record.topic_id


def test_registry_aliases_resolve_without_rewriting_topics():
    registry = registry_with("graph-neural-networks")
    original = registry.lookup("graph-neural-networks")

    grown, updated = registry.add_alias(original.topic_id, "GNN", at=NOW)

    assert grown.version == 2
    assert grown.lookup("gnn") == updated
    assert grown.lookup("graph-neural-networks") == updated
    assert "GNN" in updated.aliases
    with pytest.raises(ValueError):
        grown.add_alias("topic:missing", "other", at=NOW)
    other, _ = grown.register("sparse", at=NOW, created_by="test", ids=ids())
    with pytest.raises(ValueError):
        other.add_alias(other.topics[-1].topic_id, "gnn", at=NOW)


def test_registry_merge_supersedes_without_deleting():
    registry = registry_with("graph-networks", "graph-convolution")
    merged, record = registry.merge(
        [topic.topic_id for topic in registry.topics],
        "graph-learning",
        at=NOW,
        created_by="test",
        ids=ids(),
    )

    assert merged.version == 3
    assert record.supersedes == tuple(
        sorted(topic.topic_id for topic in registry.topics)
    )
    assert merged.lookup("graph-networks").merged_into == record.topic_id
    assert [topic.topic_id for topic in merged.active_topics()] == [record.topic_id]
    with pytest.raises(ValueError):
        merged.merge([registry.topics[0].topic_id], "again", at=NOW, created_by="t", ids=ids())


# ---------------------------------------------------------------------------
# Retrieval and decisions
# ---------------------------------------------------------------------------


def test_retrieval_ranks_the_matching_topic_first():
    registry = registry_with("sparse-training", "documentation-style")
    labeler = DeterministicLabeler()
    features = ContentFeatures(
        subject_id="workitem:github:10",
        title_tokens=("training", "fails", "sparse", "graphs"),
        body_tokens=("graph", "convolution", "overflows", "sparse", "training"),
    )

    first = labeler.retrieve(features, registry)
    second = labeler.retrieve(features, registry)

    assert first == second
    assert first[0].name == "sparse-training"
    assert first[0].score > 0
    assert first[0].matched_terms
    assert "sparse" in first[0].matched_terms or "training" in first[0].matched_terms


def test_retrieval_is_empty_without_candidates_or_vocabulary():
    labeler = DeterministicLabeler()

    assert labeler.retrieve(
        ContentFeatures(subject_id="s", title_tokens=("graph",)),
        seed_registry(at=NOW),
    ) == ()
    assert labeler.retrieve(
        ContentFeatures(subject_id="s"), registry_with("graph")
    ) == ()


def test_decide_reuses_a_strong_match_with_attribution():
    registry = registry_with("sparse-training")
    labeler = DeterministicLabeler()
    features = ContentFeatures(
        subject_id="s",
        title_tokens=("sparse", "training", "fails"),
        body_tokens=("sparse", "training", "graphs"),
    )
    candidates = labeler.retrieve(features, registry)

    decision, action, alias_of, confidence, matched = labeler.decide(
        features, candidates, registry
    )

    assert action == "reuse"
    assert decision is not None
    assert decision.reuse_topic_id == registry.topics[0].topic_id
    assert decision.proposed_name is None
    assert 0.0 < confidence <= 1.0
    assert matched
    assert "sparse-training" in decision.rationale


def test_decide_proposes_a_genuinely_new_topic():
    labeler = DeterministicLabeler()
    features = ContentFeatures(
        subject_id="snap:models/gcn.py",
        path_tokens=("models", "gcn", "py"),
        language="python",
        symbol_tokens=("models", "graphconv"),
    )

    decision, action, alias_of, confidence, matched = labeler.decide(
        features, (), seed_registry(at=NOW)
    )

    assert action == "propose"
    assert decision is not None
    assert decision.reuse_topic_id is None
    assert decision.proposed_name
    assert alias_of == ""
    assert "no registry candidate" in decision.rationale


def test_decide_suggests_aliases_for_near_matches():
    registry = registry_with("graph-neural-networks")
    labeler = DeterministicLabeler(
        LabelingSettings(reuse_threshold=0.99, alias_threshold=0.05, min_score=0.01)
    )
    features = ContentFeatures(
        subject_id="s", title_tokens=("graph", "neural", "networks", "guide")
    )
    candidates = labeler.retrieve(features, registry)
    assert candidates

    decision, action, alias_of, confidence, matched = labeler.decide(
        features, candidates, registry
    )

    assert action == "alias"
    assert decision is not None and decision.proposed_name
    assert alias_of == registry.topics[0].topic_id


def test_decide_abstains_without_usable_features():
    labeler = DeterministicLabeler()
    decision, action, alias_of, confidence, matched = labeler.decide(
        ContentFeatures(subject_id="s"), (), seed_registry(at=NOW)
    )

    assert action == "abstain"
    assert decision is None
    assert confidence == 0.0


def test_assign_wraps_decisions_in_horizon_scoped_envelopes():
    registry = registry_with("sparse-training")
    labeler = DeterministicLabeler()
    features = ContentFeatures(
        subject_id="workitem:github:10",
        title_tokens=("sparse", "training"),
        body_tokens=("sparse", "training"),
    )

    first = labeler.assign(
        features, registry, subject_id="workitem:github:10",
        run_id="run-1", info_horizon=HORIZON_ONE,
    )
    repeat = labeler.assign(
        features, registry, subject_id="workitem:github:10",
        run_id="run-1", info_horizon=HORIZON_ONE,
    )
    later = labeler.assign(
        features, registry, subject_id="workitem:github:10",
        run_id="run-1", info_horizon=HORIZON_TWO,
    )

    assert first == repeat  # repeatable without a model
    assert first.kind == "derived"
    assert first.action == "reuse"
    assert first.result.status is DecisionStatus.ACCEPTED
    assert first.registry_version == registry.version
    assert first.result.decision_key != later.result.decision_key
    assert later.result.provenance.info_horizon == HORIZON_TWO


def test_assign_abstention_is_an_explicit_envelope():
    labeler = DeterministicLabeler()
    assignment = labeler.assign(
        ContentFeatures(subject_id="empty"),
        seed_registry(at=NOW),
        subject_id="empty",
        run_id="run-1",
        info_horizon=HORIZON_ONE,
    )

    assert assignment.action == "abstain"
    assert assignment.result.status is DecisionStatus.ABSTAINED


def test_growing_the_vocabulary_does_not_relabel_old_decisions():
    labeler = DeterministicLabeler()
    registry_v1 = registry_with("documentation")
    docs = ContentFeatures(
        subject_id="snap:docs/guide.md",
        path_tokens=("docs", "guide", "md"),
        heading_tokens=("graph", "neural", "networks"),
    )
    original = labeler.assign(
        docs, registry_v1, subject_id="snap:docs/guide.md",
        run_id="run-1", info_horizon=HORIZON_ONE,
    )

    registry_v2, proposed = registry_v1.register(
        "sparse-training", at=HORIZON_TWO, created_by="run-1", ids=ids()
    )
    registry_v3, merged = registry_v2.merge(
        [topic.topic_id for topic in registry_v2.topics],
        "all-docs",
        at=HORIZON_TWO, created_by="run-1", ids=ids(),
    )

    assert original.result.value["registry_version"] == 1
    replayed = labeler.assign(
        docs, registry_v1, subject_id="snap:docs/guide.md",
        run_id="run-1", info_horizon=HORIZON_ONE,
    )
    assert replayed == original  # the old registry version decides identically
    assert merged.supersedes  # merges are tracked, not rewritten


def test_provider_labels_are_preserved_verbatim_and_kept_separate():
    labeler = DeterministicLabeler()
    item = work_item(10, labels=("bug", "sparse-support"))

    preserved = labeler.preserve_provider_labels(
        item, subject_id="workitem:github:10", run_id="run-1", info_horizon=NOW
    )
    derived = labeler.assign(
        extract_work_item_features(item, subject_id="workitem:github:10"),
        seed_registry(at=NOW),
        subject_id="workitem:github:10",
        run_id="run-1",
        info_horizon=NOW,
    )

    assert [assignment.provider_label for assignment in preserved] == ["bug", "sparse-support"]
    assert all(assignment.kind == "provider" for assignment in preserved)
    assert all(assignment.action == "preserve" for assignment in preserved)
    assert all(assignment.topic_id is None for assignment in preserved)
    assert derived.kind == "derived"
    assert derived.topic_id != "bug"  # derived topics never absorb provider names silently


def test_labeler_satisfies_the_labeler_protocol():
    labeler = DeterministicLabeler()
    assert isinstance(labeler, Labeler)

    assignments = labeler.label(
        text="sparse training fails",
        context={
            "features": ContentFeatures(
                subject_id="s", title_tokens=("sparse", "training")
            ),
            "registry": registry_with("sparse-training"),
            "subject_id": "s",
            "run_id": "run-1",
            "info_horizon": HORIZON_ONE,
            "evidence": (),
        },
    )

    assert len(assignments) == 1
    assert assignments[0].action == "reuse"
    with pytest.raises(ValueError):
        labeler.label(text="x", context="not-a-mapping")


def test_labeling_settings_have_thresholds_and_load_from_toml():
    settings = LabelingSettings()
    assert settings.reuse_threshold == 0.5
    assert settings.alias_threshold == 0.35
    assert settings.min_score == 0.1
    assert settings.top_k == 5
    with pytest.raises(Exception):
        LabelingSettings(alias_threshold=0.9, reuse_threshold=0.5)
    with pytest.raises(Exception):
        LabelingSettings(min_score=0.9)

    from repo_history.config import load_config

    config = load_config()
    assert config.labeling.reuse_threshold == 0.5
    assert config.labeling.top_k == 5
