"""Topic registry and deterministic/NLP labeling.

Derived topics live in a versioned :class:`TopicRegistry`; provider labels are
preserved verbatim as a separate assignment kind and never merged into the
registry.  Every decision is a pure function of (evidence, registry version,
information horizon):

- deterministic path/language/heading/reference features describe a subject;
- a classical TF-IDF baseline retrieves candidate topics by cosine similarity;
- the outcome is reuse, a genuinely new proposal, an alias suggestion, or an
  explicit abstention, each with matched terms as the rationale.

Assignments are :class:`AnalysisResult` envelopes, so the decision key already
binds subject, method, horizon, and evidence.  Growing the registry (new
topics, aliases, merges) never rewrites old assignments: they reference the
registry version they were decided against, and merges mark old topics as
superseded instead of deleting them.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Literal, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .analysis import AnalysisResult, DecisionStatus, Provenance, TopicDecision
from .content import ParsedContent, SnapshotContent
from .github import GitHubWorkItem
from .ids import IdFactory
from .models import EvidenceRef, FrozenModel, GraphSettings, LabelingSettings, ensure_utc

__all__ = [
    "CandidateTopic",
    "ContentFeatures",
    "DeterministicLabeler",
    "TopicAssignment",
    "TopicRecord",
    "TopicRegistry",
    "extract_snapshot_features",
    "extract_work_item_features",
    "seed_registry",
]

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    {
        "the", "and", "for", "with", "from", "that", "this", "are", "was",
        "were", "have", "has", "had", "will", "would", "should", "could",
        "into", "over", "under", "between", "through", "about", "after",
        "before", "such", "than", "then", "them", "they", "their", "there",
        "these", "those", "which", "while", "when", "where", "your", "you",
        "our", "out", "own", "same", "http", "https", "www", "com", "org",
    }
)


def _tokens(text: str | None, *, minimum: int = 2) -> tuple[str, ...]:
    """Lowercase alphanumeric tokens with stopwords removed, order kept."""

    if not text:
        return ()
    return tuple(
        token
        for token in TOKEN_PATTERN.findall(text.lower())
        if len(token) >= minimum and token not in STOPWORDS
    )


def _path_tokens(path: str) -> tuple[str, ...]:
    parts = re.split(r"[/_.\-]+", path.lower())
    tokens: list[str] = []
    for part in parts:
        tokens.extend(
            token for token in TOKEN_PATTERN.findall(part) if token not in STOPWORDS
        )
    return tuple(tokens)


class ContentFeatures(FrozenModel):
    subject_id: str
    path_tokens: tuple[str, ...] = ()
    language: str | None = None
    module: str | None = None
    heading_tokens: tuple[str, ...] = ()
    link_targets: tuple[str, ...] = ()
    import_modules: tuple[str, ...] = ()
    symbol_tokens: tuple[str, ...] = ()
    title_tokens: tuple[str, ...] = ()
    body_tokens: tuple[str, ...] = ()
    provider_labels: tuple[str, ...] = ()

    def text(self) -> str:
        """Canonical TF-IDF document: repetition preserves emphasis."""

        parts: list[str] = []
        parts.extend(self.path_tokens)
        if self.language:
            parts.append(self.language)
        if self.module:
            parts.extend(_path_tokens(self.module.replace(".", "/")))
        parts.extend(self.heading_tokens)
        parts.extend(self.symbol_tokens)
        parts.extend(self.import_modules)
        parts.extend(self.link_targets)
        parts.extend(self.title_tokens)
        parts.extend(self.body_tokens)
        parts.extend(label.lower() for label in self.provider_labels)
        return " ".join(parts)

    def token_set(self) -> frozenset[str]:
        return frozenset(self.text().split())


def extract_snapshot_features(
    snapshot: SnapshotContent,
    content: ParsedContent | None,
    *,
    subject_id: str,
) -> ContentFeatures:
    """Deterministic features for one content snapshot (never executes code)."""

    headings: list[str] = []
    links: list[str] = []
    imports: list[str] = []
    symbols: list[str] = []
    language = snapshot.language
    if content is not None and content.status == "parsed":
        language = content.language or language
        if content.markdown is not None:
            for heading in content.markdown.headings:
                headings.extend(_tokens(heading.text))
            for link in content.markdown.links:
                links.extend(_path_tokens(link.target))
        if content.code is not None:
            language = content.code.language
            for record in content.code.imports:
                if record.module:
                    imports.extend(_path_tokens(record.module.replace(".", "/")))
            for symbol in content.code.symbols:
                symbols.extend(_tokens(symbol.qualname.replace(".", " ")))
    return ContentFeatures(
        subject_id=subject_id,
        path_tokens=_path_tokens(snapshot.path),
        language=language,
        module=snapshot.module,
        heading_tokens=tuple(headings),
        link_targets=tuple(links),
        import_modules=tuple(imports),
        symbol_tokens=tuple(symbols),
    )


def extract_work_item_features(
    item: GitHubWorkItem, *, subject_id: str
) -> ContentFeatures:
    """Deterministic features for a work item; labels stay attached."""

    return ContentFeatures(
        subject_id=subject_id,
        title_tokens=_tokens(item.title),
        body_tokens=_tokens(item.body),
        provider_labels=tuple(item.labels),
    )


class TopicRecord(FrozenModel):
    topic_id: str
    name: str
    aliases: tuple[str, ...] = ()
    description: str = ""
    created_at: datetime
    created_by: str
    merged_into: str | None = None
    supersedes: tuple[str, ...] = ()

    @property
    def active(self) -> bool:
        return self.merged_into is None

    def text(self) -> str:
        return " ".join([self.name, *self.aliases, self.description])


class TopicRegistry(FrozenModel):
    version: int = 0
    topics: tuple[TopicRecord, ...] = ()
    updated_at: datetime

    def lookup(self, name_or_alias: str) -> TopicRecord | None:
        wanted = name_or_alias.strip().lower()
        for topic in self.topics:
            if topic.name.lower() == wanted or any(
                alias.lower() == wanted for alias in topic.aliases
            ):
                return topic
        return None

    def active_topics(self) -> tuple[TopicRecord, ...]:
        return tuple(topic for topic in self.topics if topic.active)

    def register(
        self,
        name: str,
        *,
        at: datetime,
        created_by: str,
        ids: IdFactory,
        description: str = "",
    ) -> tuple["TopicRegistry", TopicRecord]:
        """Add a genuinely new topic; collisions fail instead of merging.

        The description (usually the proposing subject's feature text) gives
        the topic retrievable substance beyond its name, so the same subject
        reuses its own topic instead of proposing a new name each time.
        """

        clean = name.strip()
        if not clean:
            raise ValueError("Topic names cannot be empty")
        if self.lookup(clean) is not None:
            raise ValueError(f"Topic {clean!r} already exists; use an alias")
        record = TopicRecord(
            topic_id=ids.topic(clean),
            name=clean,
            description=description.strip(),
            created_at=ensure_utc(at),
            created_by=created_by,
        )
        return (
            self.model_copy(
                update={
                    "version": self.version + 1,
                    "topics": (*self.topics, record),
                    "updated_at": ensure_utc(at),
                }
            ),
            record,
        )

    def add_alias(
        self, topic_id: str, alias: str, *, at: datetime
    ) -> tuple["TopicRegistry", TopicRecord]:
        """Attach a name variant; original assignments keep working."""

        clean = alias.strip()
        if not clean:
            raise ValueError("Topic aliases cannot be empty")
        existing = self.lookup(clean)
        if existing is not None and existing.topic_id != topic_id:
            raise ValueError(f"Alias {clean!r} already belongs to another topic")
        topics: list[TopicRecord] = []
        updated: TopicRecord | None = None
        for topic in self.topics:
            if topic.topic_id == topic_id:
                known = {topic.name.lower(), *(a.lower() for a in topic.aliases)}
                if clean.lower() not in known:
                    topic = topic.model_copy(
                        update={"aliases": (*topic.aliases, clean)}
                    )
                updated = topic
            topics.append(topic)
        if updated is None:
            raise ValueError(f"Unknown topic {topic_id!r}")
        return (
            self.model_copy(
                update={
                    "version": self.version + 1,
                    "topics": tuple(topics),
                    "updated_at": ensure_utc(at),
                }
            ),
            updated,
        )

    def merge(
        self,
        topic_ids: Sequence[str],
        name: str,
        *,
        at: datetime,
        created_by: str,
        ids: IdFactory,
    ) -> tuple["TopicRegistry", TopicRecord]:
        """Fold topics into a new canonical one; old IDs stay resolvable."""

        clean = name.strip()
        if not clean:
            raise ValueError("Topic names cannot be empty")
        if self.lookup(clean) is not None:
            raise ValueError(f"Topic {clean!r} already exists")
        wanted = set(topic_ids)
        members = [topic for topic in self.topics if topic.topic_id in wanted]
        if len(members) != len(wanted):
            raise ValueError("merge references unknown topics")
        if any(not topic.active for topic in members):
            raise ValueError("merge references already-merged topics")
        merged = TopicRecord(
            topic_id=ids.topic(clean),
            name=clean,
            description=" ".join(
                text for text in (member.description for member in members) if text
            ),
            created_at=ensure_utc(at),
            created_by=created_by,
            supersedes=tuple(sorted(wanted)),
        )
        topics: list[TopicRecord] = []
        for topic in self.topics:
            if topic.topic_id in wanted:
                topic = topic.model_copy(update={"merged_into": merged.topic_id})
            topics.append(topic)
        topics.append(merged)
        return (
            self.model_copy(
                update={
                    "version": self.version + 1,
                    "topics": tuple(topics),
                    "updated_at": ensure_utc(at),
                }
            ),
            merged,
        )


def seed_registry(*, at: datetime) -> TopicRegistry:
    """Start from an empty, versioned registry; nothing is pre-invented."""

    return TopicRegistry(version=0, topics=(), updated_at=ensure_utc(at))


class CandidateTopic(FrozenModel):
    topic_id: str
    name: str
    score: float
    matched_terms: tuple[str, ...] = ()


class TopicAssignment(FrozenModel):
    subject_id: str
    kind: Literal["provider", "derived"]
    action: Literal["preserve", "reuse", "propose", "alias", "abstain"]
    topic_id: str | None = None
    topic_name: str | None = None
    alias_of: str | None = None
    provider_label: str | None = None
    confidence: float = 0.0
    rationale: str = ""
    matched_terms: tuple[str, ...] = ()
    registry_version: int = 0
    result: AnalysisResult


def _matched_terms(
    subject: frozenset[str], topic: TopicRecord, limit: int = 8
) -> tuple[str, ...]:
    overlap = subject & set(_tokens(topic.text()))
    return tuple(sorted(overlap)[:limit])


class DeterministicLabeler:
    """Classical-NLP topic labeling with attributed, repeatable decisions."""

    name = "deterministic-tfidf-labeler"
    version = "1"

    def __init__(self, settings: LabelingSettings | None = None) -> None:
        self.settings = settings or LabelingSettings()

    def label(
        self, *, text: str, context: Any
    ) -> Sequence[TopicAssignment]:
        """Implement the :class:`Labeler` protocol for precomputed features."""

        if not isinstance(context, dict):
            raise ValueError("label context must provide features, registry, and run metadata")
        try:
            features = context["features"]
            registry = context["registry"]
            subject_id = context["subject_id"]
            run_id = context["run_id"]
            horizon = ensure_utc(context["info_horizon"])
            evidence = tuple(context.get("evidence", ()))
        except KeyError as exc:
            raise ValueError(f"label context is missing {exc}") from exc
        return (
            self.assign(
                features,
                registry,
                subject_id=subject_id,
                run_id=run_id,
                info_horizon=horizon,
                evidence=evidence,
            ),
        )

    def retrieve(
        self, features: ContentFeatures, registry: TopicRegistry
    ) -> tuple[CandidateTopic, ...]:
        """Rank active topics by TF-IDF cosine similarity, ties by topic ID."""

        subjects = [topic for topic in registry.active_topics()]
        documents = [topic.text() for topic in subjects] + [features.text()]
        try:
            matrix = TfidfVectorizer().fit_transform(documents)
        except ValueError:
            return ()
        if matrix.shape[0] <= 1:
            return ()
        similarities = cosine_similarity(matrix[-1], matrix[:-1])[0]
        subject_tokens = features.token_set()
        candidates = [
            CandidateTopic(
                topic_id=topic.topic_id,
                name=topic.name,
                score=float(score),
                matched_terms=_matched_terms(subject_tokens, topic),
            )
            for topic, score in zip(subjects, similarities)
            if float(score) >= self.settings.min_score
        ]
        candidates.sort(key=lambda item: (-item.score, item.topic_id))
        return tuple(candidates[: self.settings.top_k])

    def decide(
        self,
        features: ContentFeatures,
        candidates: Sequence[CandidateTopic],
        registry: TopicRegistry,
    ) -> tuple[TopicDecision | None, str, str, float, tuple[str, ...]]:
        """Return (decision, action, alias_of, confidence, matched terms)."""

        if not features.token_set():
            return None, "abstain", "", 0.0, ()
        best = candidates[0] if candidates else None
        if best is not None and best.score >= self.settings.reuse_threshold:
            rationale = (
                f"Reusing topic {best.name!r}: cosine {best.score:.3f} "
                f"over matched terms {list(best.matched_terms)}."
            )
            return (
                TopicDecision(
                    reuse_topic_id=best.topic_id,
                    confidence=min(1.0, max(0.0, best.score)),
                    rationale=rationale,
                    evidence_text=" ".join(best.matched_terms) or None,
                ),
                "reuse",
                "",
                min(1.0, max(0.0, best.score)),
                best.matched_terms,
            )
        if best is not None and best.score >= self.settings.alias_threshold:
            proposed = _suggest_name(features, registry)
            rationale = (
                f"Proposing {proposed!r} as an alias of {best.name!r}: "
                f"cosine {best.score:.3f} below the reuse threshold "
                f"{self.settings.reuse_threshold:.2f}."
            )
            return (
                TopicDecision(
                    proposed_name=proposed,
                    confidence=min(1.0, max(0.0, best.score)),
                    rationale=rationale,
                    evidence_text=" ".join(best.matched_terms) or None,
                ),
                "alias",
                best.topic_id,
                min(1.0, max(0.0, best.score)),
                best.matched_terms,
            )
        proposed = _suggest_name(features, registry)
        confidence = 1.0 - (best.score if best is not None else 0.0)
        if best is None:
            rationale = f"Proposing new topic {proposed!r}: no registry candidate."
        else:
            rationale = (
                f"Proposing new topic {proposed!r}: best candidate "
                f"{best.name!r} scored {best.score:.3f}."
            )
        matched = best.matched_terms if best is not None else ()
        return (
            TopicDecision(
                proposed_name=proposed,
                confidence=min(1.0, max(0.0, confidence)),
                rationale=rationale,
                evidence_text=" ".join(matched) or None,
            ),
            "propose",
            "",
            min(1.0, max(0.0, confidence)),
            matched,
        )

    def assign(
        self,
        features: ContentFeatures,
        registry: TopicRegistry,
        *,
        subject_id: str,
        run_id: str,
        info_horizon: datetime,
        evidence: Sequence[EvidenceRef] = (),
        method: str | None = None,
    ) -> TopicAssignment:
        """Decide one derived topic and wrap it in an analysis envelope."""

        horizon = ensure_utc(info_horizon)
        candidates = self.retrieve(features, registry)
        decision, action, alias_of, confidence, matched = self.decide(
            features, candidates, registry
        )
        status = DecisionStatus.ACCEPTED if decision is not None else DecisionStatus.ABSTAINED
        value: dict[str, Any] = {
            "action": action,
            "topic_id": decision.reuse_topic_id if decision else None,
            "proposed_name": decision.proposed_name if decision else None,
            "alias_of": alias_of or None,
            "confidence": confidence,
            "rationale": decision.rationale if decision else "No usable features; abstaining.",
            "matched_terms": list(matched),
            "registry_version": registry.version,
            "candidates": [
                {"topic_id": item.topic_id, "name": item.name, "score": item.score}
                for item in candidates
            ],
        }
        result = AnalysisResult.create(
            subject_id=subject_id,
            kind="topic",
            status=status,
            value=value,
            provenance=Provenance(
                method=method or self.name,
                method_version=self.version,
                run_id=run_id,
                info_horizon=horizon,
            ),
            evidence=tuple(evidence),
            error=None if decision is not None else "abstained: no usable features",
        )
        return TopicAssignment(
            subject_id=subject_id,
            kind="derived",
            action=action,  # type: ignore[arg-type]
            topic_id=value["topic_id"],
            topic_name=(
                decision.reuse_topic_id
                and next(
                    (
                        item.name
                        for item in candidates
                        if item.topic_id == decision.reuse_topic_id
                    ),
                    None,
                )
            )
            if decision and decision.reuse_topic_id
            else (decision.proposed_name if decision else None),
            alias_of=alias_of or None,
            confidence=confidence,
            rationale=value["rationale"],
            matched_terms=matched,
            registry_version=registry.version,
            result=result,
        )

    def preserve_provider_labels(
        self,
        item: GitHubWorkItem,
        *,
        subject_id: str,
        run_id: str,
        info_horizon: datetime,
    ) -> tuple[TopicAssignment, ...]:
        """Keep provider labels verbatim in a separate assignment namespace."""

        horizon = ensure_utc(info_horizon)
        assignments: list[TopicAssignment] = []
        for label in item.labels:
            result = AnalysisResult.create(
                subject_id=subject_id,
                kind="provider-label",
                status=DecisionStatus.ACCEPTED,
                value={
                    "action": "preserve",
                    "provider": "github",
                    "label": label,
                    "registry_version": None,
                },
                provenance=Provenance(
                    method="provider-label-preservation",
                    method_version="1",
                    run_id=run_id,
                    info_horizon=horizon,
                ),
                evidence=(),
            )
            assignments.append(
                TopicAssignment(
                    subject_id=subject_id,
                    kind="provider",
                    action="preserve",
                    provider_label=label,
                    confidence=1.0,
                    rationale=f"Preserving provider label {label!r} verbatim.",
                    registry_version=-1,
                    result=result,
                )
            )
        return tuple(assignments)


def _suggest_name(features: ContentFeatures, registry: TopicRegistry) -> str:
    """Propose a readable name from path/module/symbol tokens."""

    for tokens in (
        features.path_tokens[-2:],
        _path_tokens(features.module.replace(".", "/")) if features.module else (),
        features.symbol_tokens[:2],
        features.heading_tokens[:3],
        features.title_tokens[:3],
    ):
        candidate = "-".join(token for token in tokens if token)[:48]
        if candidate and registry.lookup(candidate) is None:
            return candidate
    base = features.subject_id.replace(":", "-")[-32:]
    candidate = f"topic-{base}"
    suffix = 2
    while registry.lookup(candidate) is not None:
        candidate = f"topic-{base}-{suffix}"
        suffix += 1
    return candidate
