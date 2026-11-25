"""Evidence-backed cross-entity linking.

Explicit links resolve textual references (``#123``, qualified
``owner/repo#123``, issue/PR URLs, commit SHAs, known file paths) and provider
relationships (merged PRs, head/base branches) against known entities only.
Unresolved numbers, foreign repositories, unknown SHAs, and deleted branches
are recorded as references, never invented into links.

Inferred links (topical overlap, branch association) carry a score with the
supporting evidence and abstain below ``LinkingSettings.min_inference_score``.
The four relations stay distinct: ``MENTIONS`` (textual reference),
``INTENDS_TO_RESOLVE`` (fix/close/resolve intent), ``CLOSED_BY`` (provider
closure evidence), and ``RELATED_TO`` (scored inference).  Squash/rebase
merges are described from PR commit evidence (``merge_style``) without
inventing ancestry.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal, Sequence

from .analysis import AnalysisResult, DecisionStatus, Provenance
from .git import GitIndex
from .github import GitHubDiscovery
from .graph import RelationshipWrite
from .ids import IdFactory, stable_id
from .labeling import TopicAssignment
from .models import CoverageGap, CoverageStatus, EvidenceRef, FrozenModel, LinkingSettings

__all__ = [
    "AbstainedLink",
    "LinkRecord",
    "LinkRelation",
    "LinkReport",
    "ResolvedReference",
    "TextReference",
    "extract_references",
    "infer_topical_links",
    "link_edge_writes",
    "resolve_provider_links",
    "resolve_textual_links",
    "score_links",
]


class LinkRelation(StrEnum):
    MENTIONS = "MENTIONS"
    INTENDS_TO_RESOLVE = "INTENDS_TO_RESOLVE"
    CLOSED_BY = "CLOSED_BY"
    RELATED_TO = "RELATED_TO"


class TextReference(FrozenModel):
    kind: Literal["work-item", "commit", "file", "external", "unknown"]
    target: str
    span: tuple[int, int]
    qualifier: str | None = None
    intent: bool = False
    snippet: str = ""


class ResolvedReference(FrozenModel):
    reference: TextReference
    resolved: bool
    target_id: str | None = None
    reason: str = ""


class LinkRecord(FrozenModel):
    link_id: str
    source_id: str
    target_id: str
    relation: LinkRelation
    confidence: float = 1.0
    rationale: str = ""
    evidence: tuple[EvidenceRef, ...] = ()
    method: str = "link-resolver"
    association: str | None = None
    run_id: str = ""
    info_horizon: datetime

    def to_result(self) -> AnalysisResult:
        return AnalysisResult.create(
            subject_id=self.source_id,
            kind="link",
            status=DecisionStatus.ACCEPTED,
            value={
                "relation": self.relation.value,
                "target_id": self.target_id,
                "confidence": self.confidence,
                "rationale": self.rationale,
                "association": self.association,
                "link_id": self.link_id,
                "method": self.method,
            },
            provenance=Provenance(
                method=self.method,
                method_version="1",
                run_id=self.run_id,
                info_horizon=self.info_horizon,
            ),
            evidence=self.evidence,
        )


class AbstainedLink(FrozenModel):
    source_id: str
    target_id: str
    relation: LinkRelation
    score: float
    reason: str


class LinkReport(FrozenModel):
    links: tuple[LinkRecord, ...] = ()
    abstained: tuple[AbstainedLink, ...] = ()
    references: tuple[ResolvedReference, ...] = ()
    gaps: tuple[CoverageGap, ...] = ()


NUMBER_PATTERN = re.compile(
    r"(?P<qualifier>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)?#(?P<number>\d{1,6})\b"
)
URL_PATTERN = re.compile(
    r"https?://github\.com/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)/"
    r"(?P<kind>issues|pull)/(?P<number>\d{1,6})\b",
    re.IGNORECASE,
)
SHA_PATTERN = re.compile(r"\b(?P<sha>[0-9a-f]{7,40})\b", re.IGNORECASE)
INTENT_PATTERN = re.compile(
    r"(fix(?:es|ed)?|close[sd]?|resolve[sd]?)\s+#?\d{1,6}\b", re.IGNORECASE
)
MERGE_PR_PATTERN = re.compile(r"merge pull request #(\d{1,6})\b", re.IGNORECASE)


def _intent_nearby(text: str, span: tuple[int, int]) -> bool:
    window = text[max(0, span[0] - 30) : span[0]].lower()
    if MERGE_PR_PATTERN.search(text[max(0, span[0] - 20) : span[1]]):
        return False
    return bool(re.search(r"(fix(?:es|ed)?|close[sd]?|resolve[sd]?)\s+#?$", window))


def extract_references(text: str) -> tuple[TextReference, ...]:
    """Find candidate references without resolving them against any entity."""

    if not text:
        return ()
    references: list[TextReference] = []
    claimed: list[tuple[int, int]] = []

    def overlaps(span: tuple[int, int]) -> bool:
        return any(start < span[1] and span[0] < end for start, end in claimed)

    for match in URL_PATTERN.finditer(text):
        span = (match.start(), match.end())
        claimed.append(span)
        references.append(
            TextReference(
                kind="work-item",
                target=match.group("number"),
                span=span,
                qualifier=f"{match.group('owner')}/{match.group('repo')}",
                intent=_intent_nearby(text, span),
                snippet=text[max(0, span[0] - 40) : span[1] + 20],
            )
        )
    for match in NUMBER_PATTERN.finditer(text):
        span = (match.start("number") - 1, match.end("number"))
        if overlaps(span):
            continue
        claimed.append(span)
        references.append(
            TextReference(
                kind="work-item",
                target=match.group("number"),
                span=span,
                qualifier=match.group("qualifier"),
                intent=_intent_nearby(text, span),
                snippet=text[max(0, span[0] - 40) : span[1] + 20],
            )
        )
    for match in SHA_PATTERN.finditer(text):
        span = (match.start(), match.end())
        if overlaps(span):
            continue
        sha = match.group("sha").lower()
        if len(sha) < 7 or not any(c.isdigit() for c in sha):
            continue
        claimed.append(span)
        references.append(
            TextReference(
                kind="commit",
                target=sha,
                span=span,
                snippet=text[max(0, span[0] - 40) : span[1] + 20],
            )
        )
    return tuple(sorted(references, key=lambda ref: ref.span))


def _link_id(
    ids: IdFactory, source: str, relation: LinkRelation, target: str,
    evidence: Sequence[EvidenceRef],
) -> str:
    fingerprint = ",".join(sorted(ref.evidence_id for ref in evidence))
    return stable_id("link", ids.repository_id, source, relation.value, target, fingerprint)


def _evidence(
    ids: IdFactory, source: str, locator: str, url: str | None = None
) -> EvidenceRef:
    return EvidenceRef(
        evidence_id=ids.evidence("linking", source, locator),
        source="linking",
        source_id=source,
        locator=locator,
        url=url,
    )


class _Resolver:
    """Resolve references against bounded known-entity sets."""

    def __init__(
        self,
        ids: IdFactory,
        items: dict[int, str],
        commits: set[str],
        files: set[str],
        repository: str,
        run_id: str,
        horizon: datetime,
    ) -> None:
        self._ids = ids
        self._items = items  # number -> work-item logical id
        self._commits = {sha.lower() for sha in commits}
        self._files = files
        self._repository = repository.lower()
        self._run_id = run_id
        self._horizon = horizon
        self._links: list[LinkRecord] = []
        self._references: list[ResolvedReference] = []

    def _emit(
        self,
        source: str,
        target: str,
        relation: LinkRelation,
        rationale: str,
        evidence: Sequence[EvidenceRef],
        *,
        confidence: float = 1.0,
        association: str | None = None,
        method: str = "link-resolver",
    ) -> LinkRecord:
        record = LinkRecord(
            link_id=_link_id(self._ids, source, relation, target, evidence),
            source_id=source,
            target_id=target,
            relation=relation,
            confidence=confidence,
            rationale=rationale,
            evidence=tuple(evidence),
            method=method,
            association=association,
            run_id=self._run_id,
            info_horizon=self._horizon,
        )
        self._links.append(record)
        return record

    def resolve_text(
        self,
        source_id: str,
        text: str,
        *,
        locator: str,
        url: str | None = None,
    ) -> None:
        for reference in extract_references(text):
            if reference.kind == "commit":
                matches = sorted(
                    sha for sha in self._commits if sha.startswith(reference.target)
                )
                if len(matches) != 1:
                    self._references.append(
                        ResolvedReference(
                            reference=reference, resolved=False,
                            reason=(
                                f"SHA {reference.target} matches no known commit"
                                if not matches
                                else f"SHA {reference.target} is ambiguous"
                            ),
                        )
                    )
                    continue
                match = matches[0]
                target = self._ids.commit(match)
                self._references.append(
                    ResolvedReference(
                        reference=reference, resolved=True, target_id=target
                    )
                )
                self._emit(
                    source_id, target, LinkRelation.MENTIONS,
                    f"Text references commit {match[:8]}.",
                    (_evidence(self._ids, source_id, locator, url),),
                )
                continue
            # Work-item shaped reference (bare, qualified, or URL).
            number = int(reference.target)
            if reference.qualifier is not None and (
                reference.qualifier.lower() != self._repository
                and "/" in (reference.qualifier or "")
            ):
                self._references.append(
                    ResolvedReference(
                        reference=reference,
                        resolved=False,
                        reason=(
                            "Reference points at foreign repository "
                            f"{reference.qualifier}"
                        ),
                    )
                )
                continue
            target = self._items.get(number)
            if target is None:
                self._references.append(
                    ResolvedReference(
                        reference=reference, resolved=False,
                        reason=f"Work item #{number} is not in the known selection",
                    )
                )
                continue
            self._references.append(
                ResolvedReference(
                    reference=reference, resolved=True, target_id=target
                )
            )
            relation = (
                LinkRelation.INTENDS_TO_RESOLVE
                if reference.intent
                else LinkRelation.MENTIONS
            )
            self._emit(
                source_id, target, relation,
                (
                    f"Text closes/resolves work item #{number}."
                    if reference.intent
                    else f"Text references work item #{number}."
                ),
                (_evidence(self._ids, source_id, locator, url),),
            )

    def resolve_file_mention(
        self,
        source_id: str,
        text: str,
        *,
        locator: str,
        url: str | None = None,
    ) -> None:
        lowered = text.lower()
        for path in sorted(self._files):
            needle = path.lower()
            index = lowered.find(needle)
            if index == -1:
                continue
            before = lowered[index - 1] if index > 0 else " "
            after = lowered[index + len(needle)] if index + len(needle) < len(lowered) else " "
            if before not in " \t\n\"'(`[" or after not in " \t\n\"'),.`;:]":
                continue
            reference = TextReference(
                kind="file", target=path, span=(index, index + len(needle)),
                snippet=text[max(0, index - 40) : index + len(needle) + 20],
            )
            target = self._ids.file(path)
            self._references.append(
                ResolvedReference(reference=reference, resolved=True, target_id=target)
            )
            self._emit(
                source_id, target, LinkRelation.MENTIONS,
                f"Text names file {path}.",
                (_evidence(self._ids, source_id, locator, url),),
            )


def resolve_textual_links(
    texts: Sequence[tuple[str, str, str, str | None]],
    *,
    ids: IdFactory,
    items: dict[int, str],
    commits: set[str],
    files: set[str],
    repository: str,
    run_id: str,
    horizon: datetime,
) -> LinkReport:
    """Resolve ``(source_id, text, locator, url)`` inputs into explicit links.

    Every reference is reported as resolved or not; only resolved references
    become links, so the report never invents targets.
    """

    resolver = _Resolver(ids, items, commits, files, repository, run_id, horizon)
    for source_id, text, locator, url in texts:
        resolver.resolve_text(source_id, text, locator=locator, url=url)
        resolver.resolve_file_mention(source_id, text, locator=locator, url=url)
    return LinkReport(
        links=tuple(resolver._links), references=tuple(resolver._references)
    )


def resolve_provider_links(
    discovery: GitHubDiscovery,
    index: GitIndex,
    *,
    ids: IdFactory,
    run_id: str,
    horizon: datetime,
) -> LinkReport:
    """Turn PR/issue provider evidence into closure and branch links."""

    links: list[LinkRecord] = []
    gaps: list[CoverageGap] = []
    known_commits = {
        record.sha
        for record in (*index.selection.selected_commits, *index.selection.boundary_commits)
    }
    known_branches: set[str] = set()
    for branch in index.branches:
        known_branches.add(branch.name)
        # Remote tracking refs (origin/<name>) resolve to the PR head ref
        # <name> of the same repository; anything else stays unresolved.
        if branch.source == "remote" and "/" in branch.name:
            known_branches.add(branch.name.split("/", 1)[1])
    pulls = {pull.number: pull for pull in discovery.pull_requests}

    for item in discovery.work_items:
        item_id = ids.work_item("github", item.number)
        pull = pulls.get(item.number)
        if pull is None:
            continue
        if pull.merged and pull.merge_commit_sha:
            merge_id = ids.commit(pull.merge_commit_sha)
            style = "head" if pull.merge_commit_sha == pull.head_sha else "non-head"
            if style == "non-head" and pull.merge_commit_sha not in known_commits:
                gaps.append(
                    CoverageGap(
                        code="link-merge-commit-outside-selection",
                        status=CoverageStatus.PARTIAL,
                        message=(
                            f"Merge commit {pull.merge_commit_sha[:8]} for PR "
                            f"#{item.number} is outside the bounded Git selection; "
                            "it is linked but not otherwise analyzed."
                        ),
                        source="linking",
                        source_ids=(str(item.number), pull.merge_commit_sha),
                    )
                )
            evidence = (
                EvidenceRef(
                    evidence_id=ids.evidence(
                        "github", str(item.number), f"{pull.url}#pull/{item.number}"
                    ),
                    source="github",
                    source_id=str(item.number),
                    locator=f"pull/{item.number}",
                    url=pull.url,
                ),
            )
            links.append(
                LinkRecord(
                    link_id=_link_id(ids, item_id, LinkRelation.CLOSED_BY, merge_id, evidence),
                    source_id=item_id,
                    target_id=merge_id,
                    relation=LinkRelation.CLOSED_BY,
                    confidence=1.0,
                    rationale=(
                        f"PR #{item.number} merged at {pull.merged_at}; "
                        f"merge style is {style} over {len(pull.commits)} PR commit(s)."
                    ),
                    evidence=evidence,
                    method="link-provider",
                    association=f"merge-style:{style}",
                    run_id=run_id,
                    info_horizon=horizon,
                )
            )
        for ref_name, ref_sha, role in (
            (pull.head_ref, pull.head_sha, "head"),
            (pull.base_ref, pull.base_sha, "base"),
        ):
            if not ref_name:
                continue
            branch_id = ids.branch(ref_name)
            deleted = ref_name not in known_branches
            branch_evidence = (
                EvidenceRef(
                    evidence_id=ids.evidence(
                        "github",
                        str(item.number),
                        f"{pull.url or item.url}#pull/{item.number}:{role}",
                    ),
                    source="github",
                    source_id=str(item.number),
                    locator=f"pull/{item.number}:{role}",
                    url=pull.url,
                ),
            )
            links.append(
                LinkRecord(
                    link_id=_link_id(
                        ids, item_id, LinkRelation.RELATED_TO, branch_id, branch_evidence
                    ),
                    source_id=item_id,
                    target_id=branch_id,
                    relation=LinkRelation.RELATED_TO,
                    confidence=0.9 if not deleted else 0.5,
                    rationale=(
                        f"PR #{item.number} targets {role} branch {ref_name!r}"
                        + (" (branch is deleted; tip not resolved)." if deleted else ".")
                    ),
                    evidence=branch_evidence,
                    method="link-provider",
                    association=f"branch:{role}",
                    run_id=run_id,
                    info_horizon=horizon,
                )
            )
    return LinkReport(links=tuple(links), gaps=tuple(gaps))


def infer_topical_links(
    item_topics: dict[str, list[TopicAssignment]],
    subject_topics: dict[str, list[TopicAssignment]],
    *,
    ids: IdFactory,
    settings: LinkingSettings | None = None,
    run_id: str,
    horizon: datetime,
) -> LinkReport:
    """Link subjects that share derived topics; weak overlap abstains."""

    resolved = settings or LinkingSettings()
    links: list[LinkRecord] = []
    abstained: list[AbstainedLink] = []

    def topic_ids(assignments: list[TopicAssignment]) -> dict[str, TopicAssignment]:
        found: dict[str, TopicAssignment] = {}
        for assignment in assignments:
            if assignment.kind == "derived" and assignment.topic_id is not None:
                found.setdefault(assignment.topic_id, assignment)
        return found

    item_maps = {key: topic_ids(value) for key, value in item_topics.items()}
    subject_maps = {key: topic_ids(value) for key, value in subject_topics.items()}

    for item_key, item_map in item_maps.items():
        for subject_key, subject_map in subject_maps.items():
            shared = set(item_map) & set(subject_map)
            if not shared:
                continue
            union = set(item_map) | set(subject_map)
            score = len(shared) / len(union) if union else 0.0
            evidence: list[EvidenceRef] = []
            for topic_id in sorted(shared):
                for assignment in (item_map[topic_id], subject_map[topic_id]):
                    evidence.extend(assignment.result.evidence)
            seen: set[str] = set()
            evidence = [
                ref for ref in evidence
                if ref.evidence_id not in seen and not seen.add(ref.evidence_id)
            ]
            names = sorted(
                {
                    assignment.topic_name or topic_id
                    for topic_id in shared
                    for assignment in (item_map[topic_id],)
                }
            )
            if score >= resolved.min_inference_score:
                links.append(
                    LinkRecord(
                        link_id=_link_id(
                            ids, item_key, LinkRelation.RELATED_TO, subject_key,
                            evidence,
                        ),
                        source_id=item_key,
                        target_id=subject_key,
                        relation=LinkRelation.RELATED_TO,
                        confidence=score,
                        rationale=(
                            f"Shared derived topics {names} "
                            f"(overlap {len(shared)}/{len(union)})."
                        ),
                        evidence=tuple(evidence),
                        method="link-inference",
                        association="topical",
                        run_id=run_id,
                        info_horizon=horizon,
                    )
                )
            else:
                abstained.append(
                    AbstainedLink(
                        source_id=item_key,
                        target_id=subject_key,
                        relation=LinkRelation.RELATED_TO,
                        score=score,
                        reason=(
                            f"Topical overlap {score:.2f} is below "
                            f"{resolved.min_inference_score:.2f}."
                        ),
                    )
                )
    return LinkReport(links=tuple(links), abstained=tuple(abstained))


def score_links(
    predicted: Sequence[tuple[str, str, str]],
    expected: Sequence[tuple[str, str, str]],
) -> dict[str, float]:
    """Measure (source, relation, target) link quality deterministically."""

    predicted_set = set(predicted)
    expected_set = set(expected)
    true_positives = len(predicted_set & expected_set)
    precision = true_positives / len(predicted_set) if predicted_set else 1.0
    recall = true_positives / len(expected_set) if expected_set else 1.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        ),
        "true_positives": float(true_positives),
    }


def link_edge_writes(
    links: Sequence[LinkRecord], *, run_id: str
) -> tuple[RelationshipWrite, ...]:
    """Map accepted links to single-writer relationship writes."""

    writes: list[RelationshipWrite] = []
    for link in links:
        properties: dict[str, Any] = {
            "confidence": link.confidence,
            "rationale": link.rationale,
        }
        if link.association is not None:
            properties["association"] = link.association
        writes.append(
            RelationshipWrite(
                edge_id=link.link_id,
                edge_type=link.relation.value,
                source=link.source_id,
                target=link.target_id,
                properties=properties,
                valid_from=link.info_horizon,
                origin="linking",
                origin_id=f"{link.source_id}:{link.target_id}",
                event_id=link.link_id,
                run_id=run_id,
                info_horizon=link.info_horizon,
                evidence_ids=tuple(ref.evidence_id for ref in link.evidence),
            )
        )
    return tuple(writes)
