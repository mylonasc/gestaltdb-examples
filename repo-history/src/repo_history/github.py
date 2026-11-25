"""Bounded GitHub REST discovery with saved evidence and explicit coverage.

The adapter is deliberately narrow: it reads only the work items named by the
repository selection, keeps the raw JSON responses it used, and never treats
GitHub's current state as a complete edit history.  Provider facts, observations,
and absences are separate values so later temporal replay cannot mistake an
observation timestamp for the time a body or label was originally written.

Every network attempt is charged to a :class:`BudgetLedger`.  Responses are saved
with fetch metadata (URL, status, hashes, ETag, rate-limit headers, and the next
pagination link) so an offline run can replay exactly the evidence it saw.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping
from urllib.parse import urlencode, urlsplit

import httpx
from pydantic import Field, ValidationError, field_validator

from .budgets import BudgetExceeded, BudgetLedger
from .events import EventKind, IngestEvent, now_utc
from .ids import IdFactory, stable_id
from .models import (
    Budgets,
    CoverageGap,
    CoverageStatus,
    FrozenModel,
    GitHubSettings,
    RepositorySpec,
    ensure_utc,
)

__all__ = [
    "AbsenceRecord",
    "CachedResponse",
    "EvidenceCache",
    "EvidenceMissing",
    "FetchMetadata",
    "GitHubAuthError",
    "GitHubClient",
    "GitHubCredentials",
    "GitHubDiscovery",
    "GitHubDiscoverer",
    "GitHubError",
    "GitHubPermissionError",
    "GitHubProvider",
    "GitHubRateLimitError",
    "GitHubResponseError",
    "GitHubResponseTooLarge",
    "GitHubScopeError",
    "GitHubWorkItem",
    "HierarchyEdge",
    "LabelAssignment",
    "PageResult",
    "ProviderLabel",
    "PullRequestCommit",
    "PullRequestEvidence",
    "RevisionAvailability",
    "TimelineEvent",
    "WorkItemComment",
    "parse_link_header",
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class GitHubError(RuntimeError):
    """Base class for provider failures that are safe to report."""


class GitHubAuthError(GitHubError):
    """The supplied credentials were rejected."""


class GitHubPermissionError(GitHubError):
    """The credentials authenticated but lacked a required field or scope."""


class GitHubRateLimitError(GitHubError):
    """A rate limit would require waiting longer than configured."""


class GitHubResponseTooLarge(GitHubError):
    def __init__(self, url: str, size: int, limit: int) -> None:
        super().__init__(
            f"GitHub response from {url} was {size} bytes, above the {limit}-byte limit"
        )
        self.url = url
        self.size = size
        self.limit = limit


class GitHubResponseError(GitHubError):
    def __init__(self, status_code: int, url: str, message: str = "") -> None:
        detail = f" ({message})" if message else ""
        super().__init__(f"GitHub returned HTTP {status_code} for {url}{detail}")
        self.status_code = status_code
        self.url = url
        self.message = message


class GitHubScopeError(GitHubError):
    """A URL or payload fell outside the configured repository/API origin."""


class EvidenceMissing(GitHubError):
    def __init__(self, url: str) -> None:
        super().__init__(f"No cached GitHub evidence is available for {url}")
        self.url = url


# ---------------------------------------------------------------------------
# Credentials and fetch evidence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GitHubCredentials:
    """A token that never appears in reprs, manifests, or cache files."""

    token: str | None = field(default=None, repr=False)
    token_env: str = "GITHUB_TOKEN"
    source: Literal["explicit", "environment", "none"] = "none"

    @classmethod
    def from_env(
        cls,
        settings: GitHubSettings | None = None,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> "GitHubCredentials":
        resolved = settings or GitHubSettings()
        env = os.environ if environ is None else environ
        token = env.get(resolved.token_env) or None
        return cls(
            token=token,
            token_env=resolved.token_env,
            source="environment" if token else "none",
        )

    @property
    def authenticated(self) -> bool:
        return bool(self.token)

    def __post_init__(self) -> None:
        if self.token is not None and not self.token.strip():
            raise ValueError("GitHub token must not be blank")


class FetchMetadata(FrozenModel):
    """What was requested, what came back, and how it can be revalidated."""

    url: str
    status: int
    fetched_at: datetime
    etag: str | None = None
    last_modified: str | None = None
    content_type: str | None = None
    body_sha256: str
    body_bytes: int
    from_cache: bool = False
    conditional_hit: bool = False
    request_count: int = 1
    scopes: tuple[str, ...] = ()
    rate_limit_remaining: int | None = None
    rate_limit_reset: datetime | None = None
    link_next: str | None = None

    @field_validator("fetched_at", "rate_limit_reset")
    @classmethod
    def utc(cls, value: datetime | None) -> datetime | None:
        return ensure_utc(value) if value is not None else None


class CachedResponse(FrozenModel):
    metadata: FetchMetadata
    body: str

    def json(self) -> Any:
        return json.loads(self.body)


class PageResult(FrozenModel):
    items: tuple[Any, ...] = ()
    fetches: tuple[FetchMetadata, ...] = ()
    truncated: bool = False


class EvidenceCache:
    """Raw response bodies plus fetch metadata, keyed by canonical request URL."""

    def __init__(self, root: str | Path, *, offline: bool = False) -> None:
        self.root = Path(root) / "github"
        self.offline = offline

    def path_for(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.root / f"{digest}.json"

    def load(self, url: str) -> CachedResponse | None:
        path = self.path_for(url)
        if not path.is_file():
            return None
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
            return CachedResponse.model_validate(envelope)
        except (OSError, ValueError, ValidationError):
            return None

    def store(self, url: str, response: CachedResponse) -> None:
        if self.offline:
            return
        path = self.path_for(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        envelope = {
            "metadata": response.metadata.model_dump(mode="json"),
            "body": response.body,
        }
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(envelope, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary.replace(path)


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------


def parse_link_header(value: str | None) -> dict[str, str]:
    """Return ``{rel: url}`` for an RFC 8288 ``Link`` header."""

    if not value:
        return {}
    links: dict[str, str] = {}
    for section in value.split(","):
        parts = [part.strip() for part in section.split(";") if part.strip()]
        if not parts:
            continue
        target = parts[0]
        if not (target.startswith("<") and target.endswith(">")):
            continue
        url = target[1:-1]
        for part in parts[1:]:
            if "=" not in part:
                continue
            key, raw = part.split("=", 1)
            if key.strip().lower() != "rel":
                continue
            for relation in raw.strip().strip('"').split():
                links[relation] = url
    return links


def _optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


class GitHubClient:
    """Small ``httpx`` wrapper with retries, conditional requests, and caching."""

    def __init__(
        self,
        settings: GitHubSettings | None = None,
        credentials: GitHubCredentials | None = None,
        *,
        cache: EvidenceCache | None = None,
        ledger: BudgetLedger | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.settings = settings or GitHubSettings()
        self.credentials = credentials or GitHubCredentials()
        self.cache = cache or EvidenceCache(Path("data/cache"))
        self.ledger = ledger or BudgetLedger(Budgets())
        self._sleep = sleep or time.sleep
        self._clock = clock or now_utc
        self._base = self.settings.api_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self._base,
            timeout=httpx.Timeout(self.settings.timeout_seconds),
            transport=transport,
            follow_redirects=False,
            headers=self._base_headers(),
        )

    def __enter__(self) -> "GitHubClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _base_headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": self.settings.user_agent,
        }
        if self.credentials.token:
            headers["Authorization"] = f"Bearer {self.credentials.token}"
        return headers

    def _resolve(self, url: str) -> str:
        if url.startswith(("https://", "http://")):
            resolved = url
        else:
            resolved = f"{self._base}/{url.lstrip('/')}"
        parts = urlsplit(resolved)
        base = urlsplit(self._base)
        if (parts.scheme, parts.netloc) != (base.scheme, base.netloc):
            raise GitHubScopeError(
                f"Refusing to fetch {resolved!r}: outside the configured API origin"
            )
        base_path = base.path.rstrip("/")
        if base_path and not parts.path.startswith(f"{base_path}/"):
            raise GitHubScopeError(
                f"Refusing to fetch {resolved!r}: outside the configured API path"
            )
        return resolved

    def _request_url(self, url: str, params: Mapping[str, Any] | None) -> str:
        resolved = self._resolve(url)
        if not params:
            return resolved
        query = urlencode(
            sorted((str(key), str(value)) for key, value in params.items() if value is not None)
        )
        if not query:
            return resolved
        separator = "&" if "?" in resolved else "?"
        return f"{resolved}{separator}{query}"

    def _scopes(self, response: httpx.Response) -> tuple[str, ...]:
        header = response.headers.get("x-oauth-scopes")
        if header is None:
            return ()
        return tuple(item.strip() for item in header.split(",") if item.strip())

    def _check_scopes(self, response: httpx.Response) -> None:
        required = set(self.settings.required_scopes)
        if not required:
            return
        header = response.headers.get("x-oauth-scopes")
        if header is None:
            raise GitHubPermissionError(
                "GitHub did not report token scopes; cannot verify required scopes "
                f"{sorted(required)}"
            )
        missing = required - set(self._scopes(response))
        if missing:
            raise GitHubPermissionError(
                f"GitHub token is missing required scopes: {sorted(missing)}"
            )

    def _metadata(
        self, request_url: str, response: httpx.Response, body: bytes
    ) -> FetchMetadata:
        reset = _optional_int(response.headers.get("x-ratelimit-reset"))
        return FetchMetadata(
            url=request_url,
            status=response.status_code,
            fetched_at=self._clock(),
            etag=response.headers.get("etag"),
            last_modified=response.headers.get("last-modified"),
            content_type=response.headers.get("content-type"),
            body_sha256=hashlib.sha256(body).hexdigest(),
            body_bytes=len(body),
            scopes=self._scopes(response),
            rate_limit_remaining=_optional_int(
                response.headers.get("x-ratelimit-remaining")
            ),
            rate_limit_reset=(
                datetime.fromtimestamp(reset, tz=timezone.utc)
                if reset is not None
                else None
            ),
            link_next=parse_link_header(response.headers.get("link")).get("next"),
        )

    def _is_rate_limited(self, response: httpx.Response) -> bool:
        if response.status_code == 429:
            return True
        if response.status_code != 403:
            return False
        if response.headers.get("x-ratelimit-remaining") == "0":
            return True
        body = response.text.lower()
        return "rate limit" in body or "abuse" in body

    def _rate_limit_wait(self, response: httpx.Response) -> float:
        retry_after = response.headers.get("retry-after")
        if retry_after is not None:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                pass
        reset = _optional_int(response.headers.get("x-ratelimit-reset"))
        if reset is not None:
            return max(0.0, reset - self._clock().timestamp())
        return 1.0

    def _backoff(self, attempt: int) -> float:
        return min(0.25 * (2**attempt), 4.0)

    def get(
        self,
        url: str,
        params: Mapping[str, Any] | None = None,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> CachedResponse:
        request_url = self._request_url(url, params)
        cached = self.cache.load(request_url)

        if self.cache.offline:
            if cached is None:
                raise EvidenceMissing(request_url)
            return cached.model_copy(
                update={
                    "metadata": cached.metadata.model_copy(
                        update={"from_cache": True, "conditional_hit": False}
                    )
                }
            )

        request_headers = dict(headers or {})
        if cached is not None:
            if cached.metadata.etag:
                request_headers["If-None-Match"] = cached.metadata.etag
            if cached.metadata.last_modified:
                request_headers["If-Modified-Since"] = cached.metadata.last_modified

        for attempt in range(self.settings.max_retries + 1):
            # Retries are budgeted too; never let a flaky endpoint look free.
            self.ledger.reserve_api()
            try:
                response = self._client.get(request_url, headers=request_headers)
            except httpx.HTTPError as exc:
                if attempt < self.settings.max_retries:
                    self._sleep(self._backoff(attempt))
                    continue
                raise GitHubError(
                    f"Transport failure for {request_url}: {type(exc).__name__}"
                ) from exc

            if response.status_code == 304:
                if cached is None:
                    raise GitHubResponseError(
                        304, request_url, "unexpected 304 without cached evidence"
                    )
                metadata = cached.metadata.model_copy(
                    update={
                        "status": 304,
                        "fetched_at": self._clock(),
                        "from_cache": True,
                        "conditional_hit": True,
                        "request_count": attempt + 1,
                    }
                )
                return cached.model_copy(update={"metadata": metadata})

            if response.status_code == 200:
                body = response.content
                if len(body) > self.settings.max_response_bytes:
                    raise GitHubResponseTooLarge(
                        request_url, len(body), self.settings.max_response_bytes
                    )
                self._check_scopes(response)
                metadata = self._metadata(request_url, response, body).model_copy(
                    update={"request_count": attempt + 1}
                )
                result = CachedResponse(
                    metadata=metadata, body=body.decode("utf-8", errors="strict")
                )
                self.cache.store(request_url, result)
                return result

            if response.status_code == 401:
                raise GitHubAuthError(
                    "GitHub rejected the supplied credentials (HTTP 401)"
                )

            if self._is_rate_limited(response):
                wait = self._rate_limit_wait(response)
                if wait > self.settings.max_rate_limit_wait_seconds:
                    raise GitHubRateLimitError(
                        f"Rate limit requires waiting {wait:.0f}s for {request_url}, "
                        f"above the configured {self.settings.max_rate_limit_wait_seconds:.0f}s"
                    )
                if attempt < self.settings.max_retries:
                    self._sleep(wait + 0.05)
                    continue
                raise GitHubRateLimitError(
                    f"Rate limit persisted for {request_url} after "
                    f"{self.settings.max_retries} retries"
                )

            if response.status_code == 403:
                raise GitHubPermissionError(
                    f"GitHub denied access to {request_url} (HTTP 403)"
                )

            if response.status_code >= 500:
                if attempt < self.settings.max_retries:
                    self._sleep(self._backoff(attempt))
                    continue
                raise GitHubResponseError(
                    response.status_code, request_url, "server error after retries"
                )

            raise GitHubResponseError(
                response.status_code, request_url, "unexpected response"
            )

        raise GitHubError(f"Request to {request_url} exhausted its retry budget")

    def get_json(
        self,
        url: str,
        params: Mapping[str, Any] | None = None,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> tuple[Any, FetchMetadata]:
        result = self.get(url, params, headers=headers)
        try:
            data = result.json()
        except ValueError as exc:
            raise GitHubResponseError(
                result.metadata.status, result.metadata.url, "invalid JSON"
            ) from exc
        return data, result.metadata

    def get_pages(
        self,
        url: str,
        params: Mapping[str, Any] | None = None,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> PageResult:
        page_params = dict(params or {})
        page_params.setdefault("per_page", self.settings.per_page)
        items: list[Any] = []
        fetches: list[FetchMetadata] = []
        current_url = url
        current_params: Mapping[str, Any] | None = page_params

        for _ in range(self.settings.max_pages):
            data, metadata = self.get_json(
                current_url, current_params, headers=headers
            )
            fetches.append(metadata)
            if not isinstance(data, list):
                raise GitHubResponseError(
                    metadata.status, metadata.url, "expected a JSON array"
                )
            items.extend(data)
            if not metadata.link_next:
                return PageResult(
                    items=tuple(items), fetches=tuple(fetches), truncated=False
                )
            current_url = metadata.link_next
            current_params = None

        return PageResult(
            items=tuple(items), fetches=tuple(fetches), truncated=True
        )


# ---------------------------------------------------------------------------
# Provider evidence models
# ---------------------------------------------------------------------------


class ProviderLabel(FrozenModel):
    name: str
    color: str | None = None
    description: str | None = None
    default: bool = False
    provider_id: int | None = None
    url: str | None = None


class LabelAssignment(FrozenModel):
    work_item_number: int
    label_name: str
    observed_at: datetime
    evidence_id: str
    source_url: str

    @field_validator("observed_at")
    @classmethod
    def utc(cls, value: datetime) -> datetime:
        return ensure_utc(value)


class HierarchyEdge(FrozenModel):
    parent_number: int
    child_number: int
    relation: Literal["parent", "sub-issue"] = "sub-issue"
    in_scope: bool = True
    source_url: str | None = None
    evidence_id: str


class AbsenceRecord(FrozenModel):
    kind: str
    subject: str
    status: CoverageStatus
    detail: str
    checked_urls: tuple[str, ...] = ()


class WorkItemComment(FrozenModel):
    comment_id: int
    author: str | None = None
    body: str = ""
    created_at: datetime
    updated_at: datetime | None = None
    url: str | None = None
    author_association: str | None = None

    @field_validator("created_at", "updated_at")
    @classmethod
    def utc(cls, value: datetime | None) -> datetime | None:
        return ensure_utc(value) if value is not None else None


class TimelineEvent(FrozenModel):
    event_id: str | None = None
    event: str
    actor: str | None = None
    created_at: datetime
    url: str | None = None
    label: str | None = None
    commit_id: str | None = None
    source: dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def utc(cls, value: datetime) -> datetime:
        return ensure_utc(value)


class PullRequestCommit(FrozenModel):
    sha: str
    message: str = ""
    author: str | None = None
    authored_at: datetime | None = None
    committer: str | None = None
    committed_at: datetime | None = None
    url: str | None = None
    parents: tuple[str, ...] = ()

    @field_validator("authored_at", "committed_at")
    @classmethod
    def utc(cls, value: datetime | None) -> datetime | None:
        return ensure_utc(value) if value is not None else None


class PullRequestEvidence(FrozenModel):
    number: int
    title: str
    state: str
    merged: bool = False
    merged_at: datetime | None = None
    merge_commit_sha: str | None = None
    head_ref: str | None = None
    head_sha: str
    head_repository: str | None = None
    base_ref: str | None = None
    base_sha: str
    base_repository: str | None = None
    commits: tuple[PullRequestCommit, ...] = ()
    commits_truncated: bool = False
    changed_files: int | None = None
    additions: int | None = None
    deletions: int | None = None
    draft: bool = False
    url: str
    created_at: datetime | None = None
    updated_at: datetime | None = None
    closed_at: datetime | None = None

    @field_validator("merged_at", "created_at", "updated_at", "closed_at")
    @classmethod
    def utc(cls, value: datetime | None) -> datetime | None:
        return ensure_utc(value) if value is not None else None


class RevisionAvailability(FrozenModel):
    work_item_number: int
    body_history: bool = False
    label_history: bool = False
    comments_history: bool = False
    timeline_history: bool = True
    detail: str = (
        "GitHub REST exposes current bodies, labels, and comments; earlier edit "
        "revisions are not available and were not backdated."
    )


class GitHubWorkItem(FrozenModel):
    number: int
    kind: Literal["issue", "pull_request"]
    title: str
    state: str
    state_reason: str | None = None
    body: str | None = None
    author: str | None = None
    created_at: datetime
    updated_at: datetime
    closed_at: datetime | None = None
    url: str
    labels: tuple[str, ...] = ()
    comments: tuple[WorkItemComment, ...] = ()
    timeline: tuple[TimelineEvent, ...] = ()
    timeline_available: bool = True
    comments_truncated: bool = False
    timeline_truncated: bool = False
    revisions: RevisionAvailability
    observed_at: datetime
    source_hash: str

    @field_validator("created_at", "updated_at", "closed_at", "observed_at")
    @classmethod
    def utc(cls, value: datetime | None) -> datetime | None:
        return ensure_utc(value) if value is not None else None

    @property
    def is_pull_request(self) -> bool:
        return self.kind == "pull_request"


class GitHubDiscovery(FrozenModel):
    repository: str
    observed_at: datetime
    offline: bool
    work_items: tuple[GitHubWorkItem, ...] = ()
    pull_requests: tuple[PullRequestEvidence, ...] = ()
    provider_labels: tuple[ProviderLabel, ...] = ()
    label_assignments: tuple[LabelAssignment, ...] = ()
    hierarchy_edges: tuple[HierarchyEdge, ...] = ()
    absences: tuple[AbsenceRecord, ...] = ()
    gaps: tuple[CoverageGap, ...] = ()
    fetches: tuple[FetchMetadata, ...] = ()
    counters: dict[str, int] = Field(default_factory=dict)
    truncated: bool = False

    @field_validator("observed_at")
    @classmethod
    def utc(cls, value: datetime) -> datetime:
        return ensure_utc(value)

    def to_events(self, ids: IdFactory) -> tuple[IngestEvent, ...]:
        """Normalize observations into replayable work-item events.

        Creation events use the source creation time.  Current-state observations
        use the fetch time, so today's body or labels are never backdated to the
        issue's creation.  Comments and timeline events keep their source times.
        """

        events: list[IngestEvent] = []
        pull_requests = {item.number: item for item in self.pull_requests}

        for work_item in self.work_items:
            source_base = f"{self.repository}#{work_item.number}"
            created_id = stable_id(
                "gh-event", ids.repository_id, source_base, "created"
            )
            events.append(
                IngestEvent(
                    event_id=created_id,
                    kind=EventKind.WORK_ITEM_EVENT,
                    source="github",
                    source_id=f"{source_base}:created",
                    event_time=work_item.created_at,
                    observed_at=work_item.observed_at,
                    info_horizon=max(work_item.observed_at, work_item.created_at),
                    payload={
                        "event": "created",
                        "number": work_item.number,
                        "kind": work_item.kind,
                        "title": work_item.title,
                        "author": work_item.author,
                        "url": work_item.url,
                        "created_at": work_item.created_at.isoformat(),
                    },
                    sequence=0,
                )
            )

            payload = work_item.model_dump(mode="json")
            pull_request = pull_requests.get(work_item.number)
            if pull_request is not None:
                payload["pull_request"] = pull_request.model_dump(mode="json")
            observed_id = stable_id(
                "gh-event", ids.repository_id, source_base, "observed"
            )
            events.append(
                IngestEvent(
                    event_id=observed_id,
                    kind=EventKind.WORK_ITEM_OBSERVED,
                    source="github",
                    source_id=source_base,
                    event_time=work_item.observed_at,
                    observed_at=work_item.observed_at,
                    info_horizon=work_item.observed_at,
                    payload=payload,
                    dependencies=(created_id,),
                    sequence=1,
                )
            )

            assignments = [
                assignment
                for assignment in self.label_assignments
                if assignment.work_item_number == work_item.number
            ]
            for index, assignment in enumerate(assignments):
                events.append(
                    IngestEvent(
                        event_id=stable_id(
                            "gh-event",
                            ids.repository_id,
                            source_base,
                            "label",
                            index,
                            assignment.label_name,
                        ),
                        kind=EventKind.WORK_ITEM_EVENT,
                        source="github",
                        source_id=f"{source_base}:label:{assignment.label_name}",
                        event_time=assignment.observed_at,
                        observed_at=work_item.observed_at,
                        info_horizon=max(
                            work_item.observed_at, assignment.observed_at
                        ),
                        payload={
                            "event": "label-observed",
                            "work_item": work_item.number,
                            "label": assignment.label_name,
                            "evidence_id": assignment.evidence_id,
                            "source_url": assignment.source_url,
                        },
                        dependencies=(created_id,),
                        sequence=2 + index,
                    )
                )

            for index, comment in enumerate(work_item.comments):
                events.append(
                    IngestEvent(
                        event_id=stable_id(
                            "gh-event",
                            ids.repository_id,
                            source_base,
                            "comment",
                            comment.comment_id,
                        ),
                        kind=EventKind.WORK_ITEM_EVENT,
                        source="github",
                        source_id=f"{source_base}:comment:{comment.comment_id}",
                        event_time=comment.created_at,
                        observed_at=work_item.observed_at,
                        info_horizon=max(
                            work_item.observed_at, comment.created_at
                        ),
                        payload={
                            "event": "comment",
                            **comment.model_dump(mode="json"),
                        },
                        dependencies=(created_id,),
                        sequence=100 + index,
                    )
                )

            for index, timeline_event in enumerate(work_item.timeline):
                events.append(
                    IngestEvent(
                        event_id=stable_id(
                            "gh-event",
                            ids.repository_id,
                            source_base,
                            "timeline",
                            timeline_event.event_id or index,
                        ),
                        kind=EventKind.WORK_ITEM_EVENT,
                        source="github",
                        source_id=(
                            f"{source_base}:timeline:"
                            f"{timeline_event.event_id or index}"
                        ),
                        event_time=timeline_event.created_at,
                        observed_at=work_item.observed_at,
                        info_horizon=max(
                            work_item.observed_at, timeline_event.created_at
                        ),
                        payload={
                            "event": f"timeline:{timeline_event.event}",
                            **timeline_event.model_dump(mode="json"),
                        },
                        dependencies=(created_id,),
                        sequence=200 + index,
                    )
                )

            if pull_request is not None and pull_request.merged_at is not None:
                events.append(
                    IngestEvent(
                        event_id=stable_id(
                            "gh-event", ids.repository_id, source_base, "merged"
                        ),
                        kind=EventKind.WORK_ITEM_EVENT,
                        source="github",
                        source_id=f"{source_base}:merged",
                        event_time=pull_request.merged_at,
                        observed_at=work_item.observed_at,
                        info_horizon=max(
                            work_item.observed_at, pull_request.merged_at
                        ),
                        payload={
                            "event": "merged",
                            "number": pull_request.number,
                            "merge_commit_sha": pull_request.merge_commit_sha,
                            "head_sha": pull_request.head_sha,
                            "base_sha": pull_request.base_sha,
                            "merged_at": pull_request.merged_at.isoformat(),
                        },
                        dependencies=(observed_id,),
                        sequence=300,
                    )
                )

        return tuple(sorted(events, key=lambda event: event.order_key))


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


class GitHubDiscoverer:
    """Read a bounded, approved set of GitHub work items and their evidence."""

    def __init__(
        self,
        repository: RepositorySpec,
        settings: GitHubSettings | None = None,
        *,
        ledger: BudgetLedger | None = None,
        cache: EvidenceCache | None = None,
        client: GitHubClient | None = None,
        credentials: GitHubCredentials | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.repository = repository
        self.settings = settings or GitHubSettings()
        self.ids = IdFactory(repository.url)
        self._clock = clock or now_utc
        self._owns_client = client is None
        if client is not None:
            self.client = client
            self.cache = client.cache
            self.ledger = client.ledger
        else:
            self.cache = cache or EvidenceCache(Path("data/cache"))
            self.ledger = ledger or BudgetLedger(Budgets())
            self.client = GitHubClient(
                self.settings,
                credentials,
                cache=self.cache,
                ledger=self.ledger,
                transport=transport,
                sleep=sleep,
                clock=clock,
            )
        self._fetches: list[FetchMetadata] = []
        self._gaps: list[CoverageGap] = []
        self._gap_keys: set[tuple[str, tuple[str, ...]]] = set()
        self._absences: list[AbsenceRecord] = []
        self._labels: dict[str, ProviderLabel] = {}
        self._assignments: list[LabelAssignment] = []
        self._edges: dict[tuple[int, int], HierarchyEdge] = {}
        self._truncated = False

    # -- helpers ------------------------------------------------------------
    @property
    def _owner(self) -> str:
        return self.repository.owner

    @property
    def _name(self) -> str:
        return self.repository.name

    @property
    def _api_repository(self) -> str:
        return f"/repos/{self._owner}/{self._name}"

    def _html(self, suffix: str) -> str:
        return f"https://github.com/{self.repository.full_name}{suffix}"

    def _add_gap(
        self,
        code: str,
        message: str,
        *,
        status: CoverageStatus = CoverageStatus.PARTIAL,
        source_ids: Iterable[str] = (),
        retryable: bool = False,
    ) -> None:
        ids = tuple(source_ids)
        key = (code, ids)
        if key in self._gap_keys:
            return
        self._gap_keys.add(key)
        self._gaps.append(
            CoverageGap(
                code=code,
                status=status,
                message=message,
                source="github",
                source_ids=ids,
                retryable=retryable,
            )
        )

    def _fetch_json(
        self, path: str, params: Mapping[str, Any] | None = None
    ) -> tuple[dict[str, Any], FetchMetadata]:
        data, metadata = self.client.get_json(path, params)
        self._fetches.append(metadata)
        if not isinstance(data, dict):
            raise GitHubResponseError(
                metadata.status, metadata.url, "expected a JSON object"
            )
        return data, metadata

    def _fetch_pages(
        self, path: str, params: Mapping[str, Any] | None = None
    ) -> PageResult:
        result = self.client.get_pages(path, params)
        self._fetches.extend(result.fetches)
        if result.truncated:
            self._truncated = True
            self._add_gap(
                "github-pagination-truncated",
                f"Pagination for {path} stopped at max_pages="
                f"{self.settings.max_pages}; later pages were not indexed.",
                source_ids=(path,),
            )
        return result

    def _in_scope(self, payload: Mapping[str, Any]) -> bool:
        repository_url = payload.get("repository_url")
        if isinstance(repository_url, str):
            expected = self._api_repository.lower().rstrip("/")
            if not repository_url.lower().rstrip("/").endswith(expected):
                return False
        html_url = payload.get("html_url")
        if isinstance(html_url, str):
            expected_html = self._html("/").lower()
            if not html_url.lower().startswith(expected_html):
                return False
        return True

    # -- item discovery -----------------------------------------------------
    def run(self) -> GitHubDiscovery:
        try:
            return self._run()
        finally:
            if self._owns_client:
                self.client.close()

    def _run(self) -> GitHubDiscovery:
        observed_at = self._clock()
        excluded = set(self.repository.excluded_work_items)
        selected = [
            number
            for number in self.repository.allowed_work_items
            if number not in excluded
        ]

        if not selected:
            self._absences.append(
                AbsenceRecord(
                    kind="work-item-selection",
                    subject=self.repository.full_name,
                    status=CoverageStatus.UNAVAILABLE,
                    detail=(
                        "No approved work items were configured; no GitHub work "
                        "items were fetched."
                    ),
                    checked_urls=(self._html("/issues"),),
                )
            )

        work_items: list[GitHubWorkItem] = []
        pull_requests: list[PullRequestEvidence] = []

        for number in selected:
            try:
                self.ledger.reserve_work_item()
            except BudgetExceeded as exc:
                self._truncated = True
                self._add_gap(
                    "github-work-item-budget",
                    f"Work-item budget exhausted before #{number}: {exc}",
                    source_ids=(str(number),),
                )
                break

            try:
                outcome = self._discover_one(number, observed_at)
            except BudgetExceeded as exc:
                self._truncated = True
                self._add_gap(
                    "github-request-budget",
                    f"API budget exhausted while reading #{number}: {exc}",
                    source_ids=(str(number),),
                )
                break
            except GitHubAuthError:
                raise
            except EvidenceMissing as exc:
                self._add_gap(
                    "github-evidence-missing",
                    f"Offline evidence is missing for #{number}: {exc}",
                    status=CoverageStatus.UNAVAILABLE,
                    source_ids=(str(number),),
                )
                continue
            except (
                GitHubPermissionError,
                GitHubResponseError,
                GitHubResponseTooLarge,
            ) as exc:
                self._add_gap(
                    "github-work-item-unavailable",
                    f"Work item #{number} could not be read: {exc}",
                    status=CoverageStatus.UNAVAILABLE,
                    source_ids=(str(number),),
                )
                continue

            if outcome is None:
                continue
            item, pull_request = outcome
            work_items.append(item)
            if pull_request is not None:
                pull_requests.append(pull_request)

        self._record_absences(work_items)

        return GitHubDiscovery(
            repository=self.repository.full_name,
            observed_at=observed_at,
            offline=self.cache.offline,
            work_items=tuple(work_items),
            pull_requests=tuple(pull_requests),
            provider_labels=tuple(
                self._labels[name] for name in sorted(self._labels)
            ),
            label_assignments=tuple(self._assignments),
            hierarchy_edges=tuple(
                self._edges[key] for key in sorted(self._edges)
            ),
            absences=tuple(self._absences),
            gaps=tuple(self._gaps),
            fetches=tuple(self._fetches),
            counters={
                "work_items": len(work_items),
                "pull_requests": len(pull_requests),
                "provider_labels": len(self._labels),
                "label_assignments": len(self._assignments),
                "hierarchy_edges": len(self._edges),
                "comments": sum(len(item.comments) for item in work_items),
                "timeline_events": sum(len(item.timeline) for item in work_items),
                "absences": len(self._absences),
                "gaps": len(self._gaps),
                "api_requests": self.ledger.api_requests,
                "fetches": len(self._fetches),
            },
            truncated=self._truncated,
        )

    def _discover_one(
        self, number: int, observed_at: datetime
    ) -> tuple[GitHubWorkItem, PullRequestEvidence | None] | None:
        issue_path = f"{self._api_repository}/issues/{number}"
        try:
            payload, metadata = self._fetch_json(issue_path)
        except GitHubResponseError as exc:
            if exc.status_code in (404, 410):
                self._add_gap(
                    "github-work-item-missing",
                    f"Work item #{number} is not retrievable (HTTP "
                    f"{exc.status_code}); no historical revision was invented.",
                    status=CoverageStatus.UNAVAILABLE,
                    source_ids=(str(number),),
                )
                self._absences.append(
                    AbsenceRecord(
                        kind="work-item",
                        subject=f"{self.repository.full_name}#{number}",
                        status=CoverageStatus.UNAVAILABLE,
                        detail=(
                            f"GitHub returned HTTP {exc.status_code} for the "
                            "selected work item."
                        ),
                        checked_urls=(exc.url,),
                    )
                )
                return None
            raise

        if not self._in_scope(payload):
            self._add_gap(
                "github-scope-violation",
                f"Payload for #{number} belongs to another repository; it was "
                "not indexed.",
                status=CoverageStatus.UNSUPPORTED,
                source_ids=(str(number),),
            )
            return None

        actual_number = int(payload.get("number", number))
        html_url = payload.get("html_url") or self._html(f"/issues/{number}")
        is_pull_request = bool(payload.get("pull_request"))

        comments, comments_truncated, comments_available = self._comments(
            actual_number
        )
        timeline, timeline_truncated, timeline_available = self._timeline(
            actual_number
        )
        labels = self._collect_labels(
            payload, actual_number, observed_at, html_url
        )

        revisions = RevisionAvailability(
            work_item_number=actual_number,
            comments_history=False,
            timeline_history=timeline_available,
            detail=(
                "GitHub REST exposes current bodies, labels, and comments; "
                "earlier edit revisions are unavailable. "
                + (
                    "Timeline events are available as observed."
                    if timeline_available
                    else "Timeline events were not accessible."
                )
            ),
        )

        pull_request: PullRequestEvidence | None = None
        if is_pull_request:
            try:
                pull_request = self._pull_request(actual_number, payload)
            except (EvidenceMissing, GitHubResponseError) as exc:
                self._add_gap(
                    "github-pull-request-missing",
                    f"Pull-request evidence for #{actual_number} is unavailable: {exc}",
                    status=CoverageStatus.UNAVAILABLE,
                    source_ids=(str(actual_number),),
                )

        user = payload.get("user") or {}
        item = GitHubWorkItem(
            number=actual_number,
            kind="pull_request" if is_pull_request else "issue",
            title=str(payload.get("title") or ""),
            state=str(payload.get("state") or "unknown"),
            state_reason=payload.get("state_reason"),
            body=payload.get("body"),
            author=user.get("login") if isinstance(user, dict) else None,
            created_at=payload["created_at"],
            updated_at=payload["updated_at"],
            closed_at=payload.get("closed_at"),
            url=html_url,
            labels=tuple(label.name for label in labels),
            comments=comments,
            timeline=timeline,
            timeline_available=timeline_available,
            comments_truncated=comments_truncated,
            timeline_truncated=timeline_truncated,
            revisions=revisions,
            observed_at=observed_at,
            source_hash=metadata.body_sha256,
        )

        self._hierarchy(actual_number)
        return item, pull_request

    def _comments(
        self, number: int
    ) -> tuple[tuple[WorkItemComment, ...], bool, bool]:
        path = f"{self._api_repository}/issues/{number}/comments"
        try:
            pages = self._fetch_pages(path)
        except (EvidenceMissing, GitHubPermissionError, GitHubResponseError) as exc:
            self._add_gap(
                "github-comments-unavailable",
                f"Comments for #{number} are unavailable: {exc}",
                status=CoverageStatus.UNAVAILABLE,
                source_ids=(str(number),),
            )
            return (), False, False

        comments: list[WorkItemComment] = []
        for raw in pages.items:
            try:
                comments.append(
                    WorkItemComment(
                        comment_id=int(raw["id"]),
                        author=(raw.get("user") or {}).get("login"),
                        body=str(raw.get("body") or ""),
                        created_at=raw["created_at"],
                        updated_at=raw.get("updated_at"),
                        url=raw.get("html_url"),
                        author_association=raw.get("author_association"),
                    )
                )
            except (KeyError, TypeError, ValueError, ValidationError):
                self._add_gap(
                    "github-comment-malformed",
                    f"A comment on #{number} lacked required fields and was skipped.",
                    source_ids=(str(number),),
                )
        return tuple(comments), pages.truncated, True

    def _timeline(
        self, number: int
    ) -> tuple[tuple[TimelineEvent, ...], bool, bool]:
        path = f"{self._api_repository}/issues/{number}/timeline"
        try:
            pages = self._fetch_pages(path)
        except (EvidenceMissing, GitHubPermissionError, GitHubResponseError) as exc:
            self._add_gap(
                "github-timeline-unavailable",
                f"Timeline evidence for #{number} is unavailable: {exc}",
                status=CoverageStatus.UNAVAILABLE,
                source_ids=(str(number),),
            )
            return (), False, False

        events: list[TimelineEvent] = []
        for raw in pages.items:
            event = raw.get("event")
            created_at = raw.get("created_at")
            # ``committed`` timeline entries embed the commit instead of a
            # top-level timestamp; use the committer (then author) date.
            if not created_at:
                committer = raw.get("committer") or {}
                author = raw.get("author") or {}
                if isinstance(committer, Mapping):
                    created_at = committer.get("date")
                if not created_at and isinstance(author, Mapping):
                    created_at = author.get("date")
            if not event or not created_at:
                self._add_gap(
                    "github-timeline-malformed",
                    f"A timeline event on #{number} lacked event/created_at.",
                    source_ids=(str(number),),
                )
                continue
            actor = raw.get("actor") or {}
            label = raw.get("label") or {}
            source = raw.get("source") or {}
            events.append(
                TimelineEvent(
                    event_id=(
                        str(raw["id"]) if raw.get("id") is not None else None
                    ),
                    event=str(event),
                    actor=actor.get("login") if isinstance(actor, dict) else None,
                    created_at=created_at,
                    url=raw.get("html_url"),
                    label=label.get("name") if isinstance(label, dict) else None,
                    commit_id=raw.get("commit_id") or raw.get("sha"),
                    source=source if isinstance(source, dict) else {},
                )
            )
        return tuple(events), pages.truncated, True

    def _collect_labels(
        self,
        payload: Mapping[str, Any],
        number: int,
        observed_at: datetime,
        issue_url: str,
    ) -> tuple[ProviderLabel, ...]:
        labels: list[ProviderLabel] = []
        for raw in payload.get("labels") or ():
            if isinstance(raw, str):
                name, color, description = raw, None, None
                provider_id, default, url = None, False, None
            elif isinstance(raw, Mapping):
                name = raw.get("name")
                color = raw.get("color")
                description = raw.get("description")
                provider_id = raw.get("id")
                default = bool(raw.get("default", False))
                url = raw.get("url")
            else:
                continue
            if not name:
                continue
            label = ProviderLabel(
                name=str(name),
                color=color,
                description=description,
                default=default,
                provider_id=provider_id,
                url=url,
            )
            existing = self._labels.get(label.name)
            if existing is not None and existing != label:
                self._add_gap(
                    "github-label-metadata-conflict",
                    f"Label {label.name!r} was reported with conflicting metadata; "
                    "the first observation was kept.",
                    source_ids=(str(number), label.name),
                )
            else:
                self._labels.setdefault(label.name, label)
            labels.append(label)
            self._assignments.append(
                LabelAssignment(
                    work_item_number=number,
                    label_name=label.name,
                    observed_at=observed_at,
                    evidence_id=self.ids.evidence(
                        "github", f"{number}:label:{label.name}", issue_url
                    ),
                    source_url=issue_url,
                )
            )
        return tuple(labels)

    def _hierarchy(self, number: int) -> None:
        parent_path = f"{self._api_repository}/issues/{number}/parent"
        try:
            parent, _ = self._fetch_json(parent_path)
        except EvidenceMissing:
            # A 404 online leaves no cache entry, so offline replay cannot
            # tell it apart from missing evidence.  Both mean the same thing
            # here — no parent edge — and staying silent keeps offline replay
            # identical to the online run.
            parent = None
        except (GitHubPermissionError, GitHubResponseError) as exc:
            if isinstance(exc, GitHubResponseError) and exc.status_code in (404, 410):
                parent = None  # No parent is a normal, complete answer.
            else:
                self._add_gap(
                    "github-parent-unavailable",
                    f"Parent evidence for #{number} is unavailable: {exc}",
                    status=CoverageStatus.UNAVAILABLE,
                    source_ids=(str(number),),
                )
                parent = None
        if parent is not None:
            parent_number = parent.get("number")
            if isinstance(parent_number, int):
                self._add_edge(
                    parent_number=parent_number,
                    child_number=number,
                    relation="parent",
                    source_url=parent.get("html_url"),
                )

        children_path = f"{self._api_repository}/issues/{number}/sub_issues"
        try:
            pages = self._fetch_pages(children_path)
        except (EvidenceMissing, GitHubPermissionError, GitHubResponseError) as exc:
            self._add_gap(
                "github-sub-issues-unavailable",
                f"Sub-issue evidence for #{number} is unavailable: {exc}",
                status=CoverageStatus.UNAVAILABLE,
                source_ids=(str(number),),
            )
            return

        for child in pages.items:
            child_number = child.get("number")
            if isinstance(child_number, int):
                self._add_edge(
                    parent_number=number,
                    child_number=child_number,
                    relation="sub-issue",
                    source_url=child.get("html_url"),
                )

    def _add_edge(
        self,
        *,
        parent_number: int,
        child_number: int,
        relation: Literal["parent", "sub-issue"],
        source_url: str | None,
    ) -> None:
        key = (parent_number, child_number)
        if key in self._edges:
            return
        in_scope = parent_number in self.repository.allowed_work_items and (
            child_number in self.repository.allowed_work_items
        )
        if not in_scope:
            self._add_gap(
                "github-hierarchy-out-of-scope",
                f"Hierarchy edge {parent_number} -> {child_number} leaves the "
                "approved work-item selection; the external item was not fetched.",
                source_ids=(str(parent_number), str(child_number)),
            )
        self._edges[key] = HierarchyEdge(
            parent_number=parent_number,
            child_number=child_number,
            relation=relation,
            in_scope=in_scope,
            source_url=source_url,
            evidence_id=self.ids.evidence(
                "github",
                f"{parent_number}:{child_number}",
                source_url or self._html(f"/issues/{parent_number}"),
            ),
        )

    def _pull_request(
        self, number: int, issue_payload: Mapping[str, Any]
    ) -> PullRequestEvidence:
        payload, _ = self._fetch_json(f"{self._api_repository}/pulls/{number}")
        commits, commits_truncated = self._pull_request_commits(number)
        head = payload.get("head") or {}
        base = payload.get("base") or {}
        head_repo = head.get("repo") or {}
        base_repo = base.get("repo") or {}
        return PullRequestEvidence(
            number=int(payload.get("number", number)),
            title=str(payload.get("title") or issue_payload.get("title") or ""),
            state=str(payload.get("state") or "unknown"),
            merged=bool(payload.get("merged", False)),
            merged_at=payload.get("merged_at"),
            merge_commit_sha=payload.get("merge_commit_sha"),
            head_ref=head.get("ref"),
            head_sha=str(head.get("sha") or ""),
            head_repository=head_repo.get("full_name"),
            base_ref=base.get("ref"),
            base_sha=str(base.get("sha") or ""),
            base_repository=base_repo.get("full_name"),
            commits=commits,
            commits_truncated=commits_truncated,
            changed_files=payload.get("changed_files"),
            additions=payload.get("additions"),
            deletions=payload.get("deletions"),
            draft=bool(payload.get("draft", False)),
            url=payload.get("html_url")
            or self._html(f"/pull/{number}"),
            created_at=payload.get("created_at"),
            updated_at=payload.get("updated_at"),
            closed_at=payload.get("closed_at"),
        )

    def _pull_request_commits(
        self, number: int
    ) -> tuple[tuple[PullRequestCommit, ...], bool]:
        path = f"{self._api_repository}/pulls/{number}/commits"
        try:
            pages = self._fetch_pages(path)
        except (EvidenceMissing, GitHubPermissionError, GitHubResponseError) as exc:
            self._add_gap(
                "github-pr-commits-unavailable",
                f"Commit evidence for PR #{number} is unavailable: {exc}",
                status=CoverageStatus.UNAVAILABLE,
                source_ids=(str(number),),
            )
            return (), False

        commits: list[PullRequestCommit] = []
        for raw in pages.items:
            sha = raw.get("sha")
            if not sha:
                self._add_gap(
                    "github-pr-commit-malformed",
                    f"A commit on PR #{number} lacked a SHA and was skipped.",
                    source_ids=(str(number),),
                )
                continue
            commit = raw.get("commit") or {}
            author = commit.get("author") or {}
            committer = commit.get("committer") or {}
            commits.append(
                PullRequestCommit(
                    sha=str(sha),
                    message=str(commit.get("message") or ""),
                    author=(raw.get("author") or {}).get("login"),
                    authored_at=author.get("date"),
                    committer=(raw.get("committer") or {}).get("login"),
                    committed_at=committer.get("date"),
                    url=raw.get("html_url"),
                    parents=tuple(
                        parent["sha"]
                        for parent in commit.get("parents") or ()
                        if isinstance(parent, Mapping) and parent.get("sha")
                    ),
                )
            )
        return tuple(commits), pages.truncated

    def _record_absences(self, work_items: list[GitHubWorkItem]) -> None:
        if work_items and not any(item.kind == "issue" for item in work_items):
            self._absences.append(
                AbsenceRecord(
                    kind="standalone-issue",
                    subject=self.repository.full_name,
                    status=CoverageStatus.UNAVAILABLE,
                    detail=(
                        "All selected work items are pull requests; no standalone "
                        "issues were observed in the scoped slice."
                    ),
                    checked_urls=(self._html("/issues"),),
                )
            )
        elif not work_items:
            self._absences.append(
                AbsenceRecord(
                    kind="work-item",
                    subject=self.repository.full_name,
                    status=CoverageStatus.UNAVAILABLE,
                    detail=(
                        "No selected work item could be observed; no issue or PR "
                        "content was invented."
                    ),
                    checked_urls=(self._html("/issues"),),
                )
            )

        if not self._labels:
            self._absences.append(
                AbsenceRecord(
                    kind="provider-label",
                    subject=self.repository.full_name,
                    status=CoverageStatus.UNAVAILABLE,
                    detail=(
                        "No provider labels were attached to the observed work "
                        "items; none were invented."
                    ),
                    checked_urls=(self._html("/labels"),),
                )
            )

        if not self._edges:
            self._absences.append(
                AbsenceRecord(
                    kind="hierarchy",
                    subject=self.repository.full_name,
                    status=CoverageStatus.UNAVAILABLE,
                    detail=(
                        "No native parent/sub-issue relationships were observed "
                        "among the selected work items."
                    ),
                    checked_urls=tuple(
                        self._html(f"/issues/{item.number}")
                        for item in work_items
                    ),
                )
            )

        if work_items:
            self._add_gap(
                "github-historical-revisions-unavailable",
                "GitHub REST exposes current issue bodies, labels, and comments; "
                "earlier edit revisions are unavailable and were not backdated.",
                status=CoverageStatus.UNAVAILABLE,
                source_ids=tuple(f"#{item.number}" for item in work_items),
            )


class GitHubProvider:
    """``WorkItemProvider`` facade that also exposes the full evidence bundle."""

    def __init__(
        self,
        settings: GitHubSettings | None = None,
        *,
        ledger: BudgetLedger | None = None,
        cache: EvidenceCache | None = None,
        credentials: GitHubCredentials | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.settings = settings or GitHubSettings()
        self.ledger = ledger or BudgetLedger(Budgets())
        self.cache = cache or EvidenceCache(Path("data/cache"))
        self.credentials = credentials
        self.transport = transport
        self.sleep = sleep
        self.clock = clock
        self.last_discovery: GitHubDiscovery | None = None

    def discover_full(self, repository: RepositorySpec) -> GitHubDiscovery:
        discoverer = GitHubDiscoverer(
            repository,
            self.settings,
            ledger=self.ledger,
            cache=self.cache,
            credentials=self.credentials,
            transport=self.transport,
            sleep=self.sleep,
            clock=self.clock,
        )
        self.last_discovery = discoverer.run()
        return self.last_discovery

    def discover(self, repository: RepositorySpec) -> tuple[IngestEvent, ...]:
        discovery = self.discover_full(repository)
        return discovery.to_events(IdFactory(repository.url))
