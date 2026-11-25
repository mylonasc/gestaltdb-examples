"""Bounded GitHub discovery against deterministic in-process HTTP fixtures.

No test opens a socket.  ``httpx.MockTransport`` serves exact JSON payloads and
headers, while a fake clock and fake ``sleep`` make retries, conditional
requests, and rate-limit waits fully reproducible.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from repo_history.budgets import BudgetLedger
from repo_history.events import EventKind
from repo_history.github import (
    EvidenceCache,
    EvidenceMissing,
    GitHubAuthError,
    GitHubClient,
    GitHubCredentials,
    GitHubDiscoverer,
    GitHubPermissionError,
    GitHubProvider,
    GitHubRateLimitError,
    GitHubResponseError,
    GitHubResponseTooLarge,
    GitHubScopeError,
    GitHubSettings,
    parse_link_header,
)
from repo_history.ids import IdFactory
from repo_history.models import Budgets, RepositorySpec, TimeWindow
from repo_history.protocols import WorkItemProvider

UTC = timezone.utc
NOW = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)
ANCHOR = "9bfe6c19c02723dbbf101b9e311de684a4728bb1"
REPOSITORY = RepositorySpec(
    provider="github",
    owner="mylonasc",
    name="tf_gnns",
    url="https://github.com/mylonasc/tf_gnns",
    anchor=ANCHOR,
    window=TimeWindow(
        start=datetime(2026, 9, 13, tzinfo=UTC),
        end=datetime(2026, 9, 15, tzinfo=UTC),
    ),
    allowed_work_items=(10, 11),
)


def repository(*, allowed=(10, 11), excluded=()) -> RepositorySpec:
    return REPOSITORY.model_copy(
        update={
            "allowed_work_items": tuple(allowed),
            "excluded_work_items": tuple(excluded),
        }
    )


def issue(
    number: int,
    *,
    pull_request: bool = False,
    labels=(),
    title: str | None = None,
    html_url: str | None = None,
    created_at: str = "2026-09-13T10:00:00Z",
):
    payload = {
        "number": number,
        "title": title or f"Work item {number}",
        "state": "closed",
        "state_reason": "completed",
        "body": f"Body for {number}",
        "user": {"login": "mylonasc"},
        "created_at": created_at,
        "updated_at": "2026-09-14T09:00:00Z",
        "closed_at": "2026-09-14T10:00:00Z",
        "html_url": html_url
        or f"https://github.com/mylonasc/tf_gnns/issues/{number}",
        "labels": list(labels),
    }
    if pull_request:
        payload["pull_request"] = {
            "url": f"https://api.github.com/repos/mylonasc/tf_gnns/pulls/{number}"
        }
    return payload


def label(name: str, *, label_id: int = 1, color: str = "ff0000", description=None):
    return {
        "id": label_id,
        "name": name,
        "color": color,
        "description": description,
        "default": False,
        "url": f"https://api.github.com/repos/mylonasc/tf_gnns/labels/{name}",
    }


def comment(comment_id: int, body: str = "A comment"):
    return {
        "id": comment_id,
        "user": {"login": "reviewer"},
        "body": body,
        "created_at": "2026-09-13T12:00:00Z",
        "updated_at": "2026-09-13T12:30:00Z",
        "html_url": f"https://github.com/mylonasc/tf_gnns/issues/10#issuecomment-{comment_id}",
        "author_association": "MEMBER",
    }


def timeline(event_id: int, event: str = "labeled", **extra):
    payload = {
        "id": event_id,
        "event": event,
        "actor": {"login": "mylonasc"},
        "created_at": "2026-09-13T13:00:00Z",
        "html_url": "https://github.com/mylonasc/tf_gnns/issues/10#event",
    }
    payload.update(extra)
    return payload


def pull_request(number: int = 11):
    return {
        "number": number,
        "title": f"Pull request {number}",
        "state": "closed",
        "merged": True,
        "merged_at": "2026-09-14T11:00:00Z",
        "merge_commit_sha": "merge-sha",
        "head": {
            "ref": "feature",
            "sha": "head-sha",
            "repo": {"full_name": "mylonasc/tf_gnns"},
        },
        "base": {
            "ref": "main",
            "sha": "base-sha",
            "repo": {"full_name": "mylonasc/tf_gnns"},
        },
        "changed_files": 2,
        "additions": 5,
        "deletions": 1,
        "draft": False,
        "html_url": f"https://github.com/mylonasc/tf_gnns/pull/{number}",
        "created_at": "2026-09-13T11:00:00Z",
        "updated_at": "2026-09-14T11:00:00Z",
        "closed_at": "2026-09-14T11:00:00Z",
    }


def pull_commit(sha: str = "commit-sha"):
    return {
        "sha": sha,
        "commit": {
            "message": "Add feature",
            "author": {"date": "2026-09-13T11:05:00Z"},
            "committer": {"date": "2026-09-13T11:06:00Z"},
            "parents": [{"sha": "parent-sha"}],
        },
        "author": {"login": "mylonasc"},
        "committer": {"login": "mylonasc"},
        "html_url": "https://github.com/mylonasc/tf_gnns/commit/commit-sha",
    }


class FakeGitHub:
    """A tiny router for the exact endpoints RH-04 is allowed to call."""

    def __init__(
        self,
        *,
        issues=None,
        comments=None,
        timeline=None,
        parents=None,
        sub_issues=None,
        pulls=None,
        commits=None,
        etag: bool = True,
    ) -> None:
        self.issues = {int(k): v for k, v in (issues or {}).items()}
        self.comments = {int(k): v for k, v in (comments or {}).items()}
        self.timeline = {int(k): v for k, v in (timeline or {}).items()}
        self.parents = {int(k): v for k, v in (parents or {}).items()}
        self.sub_issues = {int(k): v for k, v in (sub_issues or {}).items()}
        self.pulls = {int(k): v for k, v in (pulls or {}).items()}
        self.commits = {int(k): v for k, v in (commits or {}).items()}
        self.etag = etag
        self.requests: list[httpx.Request] = []
        self.overrides: dict[str, object] = {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        override = self.overrides.get(path)
        if override is not None:
            return override(request)  # type: ignore[operator]

        segments = [part for part in path.split("/") if part]
        if len(segments) >= 5 and segments[0] == "repos":
            kind, number = segments[3], int(segments[4])
            if kind == "issues" and len(segments) == 5:
                payload = self.issues.get(number)
                if payload is None:
                    return httpx.Response(404, json={"message": "Not Found"})
                return self._ok(request, payload)
            if kind == "issues" and len(segments) == 6:
                sub = segments[5]
                if sub == "comments":
                    return self._ok(request, self.comments.get(number, []))
                if sub == "timeline":
                    return self._ok(request, self.timeline.get(number, []))
                if sub == "sub_issues":
                    return self._ok(request, self.sub_issues.get(number, []))
                if sub == "parent":
                    payload = self.parents.get(number)
                    if payload is None:
                        return httpx.Response(404, json={"message": "Not Found"})
                    return self._ok(request, payload)
            if kind == "pulls" and len(segments) == 5:
                payload = self.pulls.get(number)
                if payload is None:
                    return httpx.Response(404, json={"message": "Not Found"})
                return self._ok(request, payload)
            if kind == "pulls" and len(segments) == 6 and segments[5] == "commits":
                return self._ok(request, self.commits.get(number, []))
        return httpx.Response(404, json={"message": "Not Found"})

    def _ok(self, request: httpx.Request, payload) -> httpx.Response:
        if self.etag:
            if request.headers.get("if-none-match") == '"v1"':
                return httpx.Response(304)
            return httpx.Response(200, json=payload, headers={"etag": '"v1"'})
        return httpx.Response(200, json=payload)


def transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def discover(
    tmp_path: Path,
    handler,
    *,
    settings: GitHubSettings | None = None,
    ledger: BudgetLedger | None = None,
    selection: RepositorySpec | None = None,
    credentials: GitHubCredentials | None = None,
    offline: bool = False,
    sleep=None,
    clock=lambda: NOW,
):
    sleeps: list[float] = []

    def record_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    discoverer = GitHubDiscoverer(
        selection or REPOSITORY,
        settings or GitHubSettings(),
        ledger=ledger or BudgetLedger(Budgets()),
        cache=EvidenceCache(tmp_path, offline=offline),
        credentials=credentials,
        transport=transport(handler),
        sleep=sleep or record_sleep,
        clock=clock,
    )
    return discoverer.run(), sleeps


def full_fixture() -> FakeGitHub:
    return FakeGitHub(
        issues={
            10: issue(10, labels=[label("bug", description="Something is broken")]),
            11: issue(
                11,
                pull_request=True,
                labels=[label("enhancement", label_id=2, color="00ff00")],
                title="Add feature",
            ),
        },
        comments={10: [comment(100)]},
        timeline={10: [timeline(500, label={"name": "bug"})]},
        sub_issues={
            10: [
                {
                    "number": 11,
                    "html_url": "https://github.com/mylonasc/tf_gnns/issues/11",
                }
            ]
        },
        pulls={11: pull_request(11)},
        commits={11: [pull_commit()]},
    )


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_full_discovery_collects_items_labels_hierarchy_and_pr_evidence(tmp_path):
    fake = full_fixture()
    discovery, _ = discover(tmp_path, fake)

    assert discovery.repository == "mylonasc/tf_gnns"
    assert discovery.observed_at == NOW
    assert discovery.offline is False
    assert [item.number for item in discovery.work_items] == [10, 11]
    assert [item.kind for item in discovery.work_items] == ["issue", "pull_request"]
    assert discovery.counters["work_items"] == 2
    assert discovery.counters["pull_requests"] == 1
    assert discovery.counters["provider_labels"] == 2
    assert discovery.counters["label_assignments"] == 2
    assert discovery.counters["hierarchy_edges"] == 1
    assert discovery.counters["comments"] == 1
    assert discovery.counters["timeline_events"] == 1
    assert discovery.counters["api_requests"] == len(fake.requests)
    assert len(discovery.fetches) <= discovery.counters["api_requests"]

    labels = {label.name: label for label in discovery.provider_labels}
    assert labels["bug"].description == "Something is broken"
    assert labels["enhancement"].color == "00ff00"
    assert [item.labels for item in discovery.work_items] == [
        ("bug",),
        ("enhancement",),
    ]

    edge = discovery.hierarchy_edges[0]
    assert (edge.parent_number, edge.child_number) == (10, 11)
    assert edge.relation == "sub-issue"
    assert edge.in_scope is True

    pr = discovery.pull_requests[0]
    assert pr.number == 11
    assert pr.merged is True
    assert pr.merge_commit_sha == "merge-sha"
    assert (pr.head_ref, pr.head_sha) == ("feature", "head-sha")
    assert (pr.base_ref, pr.base_sha) == ("main", "base-sha")
    assert pr.head_repository == "mylonasc/tf_gnns"
    assert len(pr.commits) == 1
    assert pr.commits[0].parents == ("parent-sha",)
    assert pr.commits[0].authored_at == datetime(2026, 9, 13, 11, 5, tzinfo=UTC)

    issue_item = discovery.work_items[0]
    assert issue_item.comments[0].author == "reviewer"
    assert issue_item.comments[0].created_at == datetime(
        2026, 9, 13, 12, tzinfo=UTC
    )
    assert issue_item.timeline[0].label == "bug"
    assert issue_item.timeline_available is True

    # Current bodies/labels are not historical revisions.
    assert all(item.revisions.body_history is False for item in discovery.work_items)
    assert all(item.revisions.label_history is False for item in discovery.work_items)
    assert any(
        gap.code == "github-historical-revisions-unavailable"
        for gap in discovery.gaps
    )
    assert discovery.truncated is False


def test_committed_timeline_events_use_the_commit_timestamp(tmp_path):
    fake = full_fixture()
    fake.timeline[10] = [
        {
            "event": "committed",
            "sha": "abc123",
            "message": "Add feature",
            "author": {"date": "2026-09-13T14:00:00Z"},
            "committer": {"date": "2026-09-13T14:05:00Z"},
            "html_url": "https://github.com/mylonasc/tf_gnns/commit/abc123",
        }
    ]

    discovery, _ = discover(tmp_path, fake)

    entry = discovery.work_items[0].timeline[0]
    assert entry.event == "committed"
    assert entry.created_at == datetime(2026, 9, 13, 14, 5, tzinfo=UTC)
    assert entry.commit_id == "abc123"
    assert not any(
        gap.code == "github-timeline-malformed" for gap in discovery.gaps
    )


def test_events_are_deterministic_and_never_backdate_current_state(tmp_path):
    discovery, _ = discover(tmp_path, full_fixture())
    ids = IdFactory(REPOSITORY.url)
    events = discovery.to_events(ids)

    assert events == discovery.to_events(ids)
    assert list(events) == sorted(events, key=lambda event: event.order_key)
    assert len({event.event_id for event in events}) == len(events)

    created = [event for event in events if event.payload.get("event") == "created"]
    observed = [
        event for event in events if event.kind is EventKind.WORK_ITEM_OBSERVED
    ]
    comments = [event for event in events if event.payload.get("event") == "comment"]
    merged = [event for event in events if event.payload.get("event") == "merged"]
    label_events = [
        event for event in events if event.payload.get("event") == "label-observed"
    ]

    assert len(created) == 2
    assert len(observed) == 2
    assert created[0].event_time == datetime(2026, 9, 13, 10, tzinfo=UTC)
    # Observations are stamped at fetch time, not the issue creation time.
    assert all(event.event_time == NOW for event in observed)
    assert all(event.info_horizon == NOW for event in observed)
    # Comments keep their source timestamp but record when they were observed.
    assert comments[0].event_time == datetime(2026, 9, 13, 12, tzinfo=UTC)
    assert comments[0].observed_at == NOW
    assert merged[0].event_time == datetime(2026, 9, 14, 11, tzinfo=UTC)
    assert merged[0].observed_at == NOW
    assert all(event.event_time == NOW for event in label_events)
    assert all(event.info_horizon >= event.event_time for event in events)
    assert all(
        dependency in {event.event_id for event in events}
        for event in events
        for dependency in event.dependencies
    )


def test_pagination_follows_link_headers(tmp_path):
    fake = full_fixture()
    first = comment(100, "first")
    second = comment(101, "second")

    def comments_page(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("page") == "2":
            return httpx.Response(200, json=[second])
        return httpx.Response(
            200,
            json=[first],
            headers={
                "link": (
                    "<https://api.github.com/repos/mylonasc/tf_gnns/issues/10/"
                    'comments?page=2>; rel="next"'
                )
            },
        )

    fake.comments[10] = []
    fake.overrides["/repos/mylonasc/tf_gnns/issues/10/comments"] = comments_page

    discovery, _ = discover(tmp_path, fake)
    item = discovery.work_items[0]
    assert [entry.body for entry in item.comments] == ["first", "second"]
    assert item.comments_truncated is False
    assert not any(
        gap.code == "github-pagination-truncated" for gap in discovery.gaps
    )


def test_pagination_truncation_is_an_explicit_gap(tmp_path):
    fake = full_fixture()
    fake.overrides["/repos/mylonasc/tf_gnns/issues/10/comments"] = (
        lambda request: httpx.Response(
            200,
            json=[comment(100)],
            headers={
                "link": (
                    "<https://api.github.com/repos/mylonasc/tf_gnns/issues/10/"
                    'comments?page=2>; rel="next"'
                )
            },
        )
    )

    discovery, _ = discover(
        tmp_path, fake, settings=GitHubSettings(max_pages=1)
    )
    assert discovery.truncated is True
    assert discovery.work_items[0].comments_truncated is True
    assert any(
        gap.code == "github-pagination-truncated" for gap in discovery.gaps
    )


def test_conditional_requests_reuse_cached_bodies(tmp_path):
    fake = full_fixture()
    first, _ = discover(tmp_path, fake)
    second, _ = discover(tmp_path, fake)

    assert first.work_items == second.work_items
    assert first.pull_requests == second.pull_requests
    assert first.label_assignments == second.label_assignments
    assert second.fetches
    assert all(fetch.conditional_hit for fetch in second.fetches)
    assert all(fetch.from_cache for fetch in second.fetches)
    # Conditional requests still consume the API budget.
    assert second.counters["api_requests"] > 0


def test_offline_replay_uses_only_saved_evidence(tmp_path):
    first, _ = discover(tmp_path, full_fixture())

    def refuse(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"offline discovery attempted {request.url}")

    second, _ = discover(tmp_path, refuse, offline=True)

    assert second.offline is True
    assert second.work_items == first.work_items
    assert second.pull_requests == first.pull_requests
    assert second.provider_labels == first.provider_labels
    assert second.hierarchy_edges == first.hierarchy_edges
    assert all(fetch.from_cache for fetch in second.fetches)


def test_offline_replay_without_evidence_reports_unavailability(tmp_path):
    def refuse(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"offline discovery attempted {request.url}")

    discovery, _ = discover(tmp_path, refuse, offline=True)

    assert discovery.work_items == ()
    assert discovery.truncated is False
    codes = {gap.code for gap in discovery.gaps}
    assert "github-evidence-missing" in codes
    kinds = {absence.kind for absence in discovery.absences}
    assert {"work-item", "provider-label", "hierarchy"} <= kinds


def test_missing_work_item_is_an_absence_not_invented_content(tmp_path):
    fake = full_fixture()
    del fake.issues[10]

    discovery, _ = discover(tmp_path, fake)

    assert [item.number for item in discovery.work_items] == [11]
    assert any(
        gap.code == "github-work-item-missing" for gap in discovery.gaps
    )
    assert any(
        absence.kind == "work-item" and "#10" in absence.subject
        for absence in discovery.absences
    )


def test_forbidden_optional_endpoint_keeps_the_work_item(tmp_path):
    fake = full_fixture()
    fake.overrides["/repos/mylonasc/tf_gnns/issues/10/comments"] = (
        lambda request: httpx.Response(403, json={"message": "Forbidden"})
    )

    discovery, _ = discover(tmp_path, fake)

    item = discovery.work_items[0]
    assert item.number == 10
    assert item.comments == ()
    assert any(
        gap.code == "github-comments-unavailable" for gap in discovery.gaps
    )


def test_payload_from_another_repository_is_rejected(tmp_path):
    fake = full_fixture()
    fake.issues[10] = issue(
        10, html_url="https://github.com/someone/else/issues/10"
    )

    discovery, _ = discover(tmp_path, fake)

    assert [item.number for item in discovery.work_items] == [11]
    gap = next(
        gap for gap in discovery.gaps if gap.code == "github-scope-violation"
    )
    assert gap.status.value == "unsupported"


def test_hierarchy_edges_outside_the_selection_are_flagged(tmp_path):
    fake = full_fixture()
    fake.parents[10] = {
        "number": 99,
        "html_url": "https://github.com/mylonasc/tf_gnns/issues/99",
    }

    discovery, _ = discover(tmp_path, fake)

    external = next(
        edge for edge in discovery.hierarchy_edges if edge.parent_number == 99
    )
    assert external.in_scope is False
    assert any(
        gap.code == "github-hierarchy-out-of-scope" for gap in discovery.gaps
    )
    # The out-of-scope parent is never fetched.
    assert not any(
        "/issues/99" in str(request.url) for request in fake.requests
    )


def test_absent_labels_and_hierarchy_are_explicit(tmp_path):
    fake = FakeGitHub(issues={10: issue(10)})

    discovery, _ = discover(tmp_path, fake)

    assert discovery.provider_labels == ()
    assert discovery.hierarchy_edges == ()
    kinds = {absence.kind for absence in discovery.absences}
    assert {"provider-label", "hierarchy"} <= kinds
    assert any(
        absence.kind == "standalone-issue" for absence in discovery.absences
    ) is False


def test_standalone_issues_are_reported_when_only_pull_requests_exist(tmp_path):
    fake = FakeGitHub(
        issues={11: issue(11, pull_request=True)},
        pulls={11: pull_request(11)},
    )

    discovery, _ = discover(tmp_path, fake, selection=repository(allowed=(11,)))

    assert any(
        absence.kind == "standalone-issue" for absence in discovery.absences
    )


def test_excluded_work_items_are_never_requested(tmp_path):
    fake = full_fixture()
    discovery, _ = discover(
        tmp_path, fake, selection=repository(allowed=(10, 11), excluded=(11,))
    )

    assert [item.number for item in discovery.work_items] == [10]
    assert not any("/pulls/11" in str(request.url) for request in fake.requests)


def test_empty_selection_fetches_nothing_and_records_absence(tmp_path):
    fake = full_fixture()
    discovery, _ = discover(tmp_path, fake, selection=repository(allowed=()))

    assert fake.requests == []
    assert discovery.work_items == ()
    assert any(
        absence.kind == "work-item-selection" for absence in discovery.absences
    )


def test_api_budget_stops_discovery_and_reports_a_gap(tmp_path):
    ledger = BudgetLedger(Budgets(max_api_requests=3, max_work_items=5))
    discovery, _ = discover(tmp_path, full_fixture(), ledger=ledger)

    assert ledger.api_requests == 3
    assert discovery.truncated is True
    assert any(gap.code == "github-request-budget" for gap in discovery.gaps)
    assert discovery.work_items == ()


def test_work_item_budget_stops_before_the_next_item(tmp_path):
    ledger = BudgetLedger(Budgets(max_work_items=1, max_api_requests=100))
    discovery, _ = discover(tmp_path, full_fixture(), ledger=ledger)

    assert ledger.work_items == 1
    assert [item.number for item in discovery.work_items] == [10]
    assert discovery.truncated is True
    assert any(gap.code == "github-work-item-budget" for gap in discovery.gaps)


def test_provider_facade_satisfies_the_work_item_contract(tmp_path):
    provider = GitHubProvider(
        cache=EvidenceCache(tmp_path),
        transport=transport(full_fixture()),
        clock=lambda: NOW,
    )
    assert isinstance(provider, WorkItemProvider)

    events = provider.discover(REPOSITORY)
    assert provider.last_discovery is not None
    assert events
    assert all(event.source == "github" for event in events)


# ---------------------------------------------------------------------------
# Client behaviour
# ---------------------------------------------------------------------------


def make_client(
    handler,
    tmp_path: Path,
    *,
    settings: GitHubSettings | None = None,
    credentials: GitHubCredentials | None = None,
    ledger: BudgetLedger | None = None,
    sleeps: list[float] | None = None,
    clock=lambda: NOW,
) -> GitHubClient:
    recorded = sleeps if sleeps is not None else []

    def fake_sleep(seconds: float) -> None:
        recorded.append(seconds)

    return GitHubClient(
        settings or GitHubSettings(),
        credentials or GitHubCredentials(),
        cache=EvidenceCache(tmp_path),
        ledger=ledger or BudgetLedger(Budgets()),
        transport=transport(handler),
        sleep=fake_sleep,
        clock=clock,
    )


def test_parse_link_header_handles_multiple_relations():
    header = (
        '<https://api.github.com/a?page=2>; rel="next", '
        '<https://api.github.com/a?page=5>; rel="last"'
    )
    assert parse_link_header(header) == {
        "next": "https://api.github.com/a?page=2",
        "last": "https://api.github.com/a?page=5",
    }
    assert parse_link_header(None) == {}
    assert parse_link_header("not a link") == {}


def test_client_retries_transient_server_errors(tmp_path):
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] == 1:
            return httpx.Response(500, json={"message": "boom"})
        return httpx.Response(200, json={"ok": True})

    sleeps: list[float] = []
    client = make_client(
        handler, tmp_path, settings=GitHubSettings(max_retries=1), sleeps=sleeps
    )
    data, metadata = client.get_json("/repos/mylonasc/tf_gnns/issues/10")

    assert data == {"ok": True}
    assert metadata.request_count == 2
    assert len(sleeps) == 1
    assert client.ledger.api_requests == 2


def test_client_reports_exhausted_server_errors(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"message": "boom"})

    client = make_client(
        handler, tmp_path, settings=GitHubSettings(max_retries=1)
    )
    with pytest.raises(GitHubResponseError) as caught:
        client.get_json("/repos/mylonasc/tf_gnns/issues/10")
    assert caught.value.status_code == 500
    assert client.ledger.api_requests == 2


def test_client_rejects_bad_credentials_without_echoing_them(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"message": "Bad credentials"})

    client = make_client(handler, tmp_path)
    with pytest.raises(GitHubAuthError):
        client.get_json("/repos/mylonasc/tf_gnns/issues/10")


def test_client_distinguishes_permission_from_rate_limit(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"message": "Forbidden"})

    client = make_client(handler, tmp_path)
    with pytest.raises(GitHubPermissionError):
        client.get_json("/repos/mylonasc/tf_gnns/issues/10")


def test_client_waits_for_a_short_rate_limit_and_retries(tmp_path):
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] == 1:
            return httpx.Response(
                429, headers={"retry-after": "2"}, json={"message": "slow down"}
            )
        return httpx.Response(200, json={"ok": True})

    sleeps: list[float] = []
    client = make_client(
        handler,
        tmp_path,
        settings=GitHubSettings(max_retries=1, max_rate_limit_wait_seconds=30),
        sleeps=sleeps,
    )
    data, _ = client.get_json("/repos/mylonasc/tf_gnns/issues/10")

    assert data == {"ok": True}
    assert sleeps and sleeps[0] >= 2
    assert client.ledger.api_requests == 2


def test_client_refuses_to_wait_past_the_rate_limit_ceiling(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429, headers={"retry-after": "600"}, json={"message": "slow down"}
        )

    client = make_client(
        handler,
        tmp_path,
        settings=GitHubSettings(max_retries=3, max_rate_limit_wait_seconds=60),
    )
    with pytest.raises(GitHubRateLimitError):
        client.get_json("/repos/mylonasc/tf_gnns/issues/10")


def test_client_checks_required_oauth_scopes(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True}, headers={"x-oauth-scopes": "read:org"})

    client = make_client(
        handler, tmp_path, settings=GitHubSettings(required_scopes=("repo",))
    )
    with pytest.raises(GitHubPermissionError):
        client.get_json("/repos/mylonasc/tf_gnns/issues/10")

    def scoped(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"ok": True}, headers={"x-oauth-scopes": "repo, read:org"}
        )

    client = make_client(
        scoped, tmp_path, settings=GitHubSettings(required_scopes=("repo",))
    )
    assert client.get_json("/repos/mylonasc/tf_gnns/issues/10")[0] == {"ok": True}


def test_client_enforces_the_response_size_limit(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 4096)

    client = make_client(
        handler, tmp_path, settings=GitHubSettings(max_response_bytes=1024)
    )
    with pytest.raises(GitHubResponseTooLarge):
        client.get_json("/repos/mylonasc/tf_gnns/issues/10")


def test_client_refuses_urls_outside_the_configured_origin(tmp_path):
    client = make_client(
        lambda request: httpx.Response(200, json={}), tmp_path
    )
    with pytest.raises(GitHubScopeError):
        client.get_json("https://evil.example/repos/mylonasc/tf_gnns/issues/10")


def test_client_rejects_invalid_json(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    client = make_client(handler, tmp_path)
    with pytest.raises(GitHubResponseError):
        client.get_json("/repos/mylonasc/tf_gnns/issues/10")


def test_client_rejects_304_without_cached_evidence(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(304)

    client = make_client(handler, tmp_path)
    with pytest.raises(GitHubResponseError):
        client.get_json("/repos/mylonasc/tf_gnns/issues/10")


def test_client_sends_a_token_but_never_persists_it(tmp_path):
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization", ""))
        return httpx.Response(200, json={"ok": True})

    credentials = GitHubCredentials(token="s3cret-token", source="explicit")
    assert "s3cret-token" not in repr(credentials)

    client = make_client(handler, tmp_path, credentials=credentials)
    client.get_json("/repos/mylonasc/tf_gnns/issues/10")

    assert seen == ["Bearer s3cret-token"]
    cached = client.cache.path_for(
        "https://api.github.com/repos/mylonasc/tf_gnns/issues/10"
    )
    assert cached.is_file()
    assert "s3cret-token" not in cached.read_text(encoding="utf-8")
    stored = json.loads(cached.read_text(encoding="utf-8"))
    assert stored["metadata"]["url"].endswith("/issues/10")
    assert stored["body"] == '{"ok":true}'


def test_credentials_load_from_the_configured_environment_variable():
    credentials = GitHubCredentials.from_env(
        GitHubSettings(token_env="MY_GH_TOKEN"),
        environ={"MY_GH_TOKEN": "abc", "GITHUB_TOKEN": "other"},
    )
    assert credentials.token == "abc"
    assert credentials.source == "environment"
    assert credentials.authenticated is True

    missing = GitHubCredentials.from_env(
        GitHubSettings(token_env="MY_GH_TOKEN"), environ={}
    )
    assert missing.token is None
    assert missing.authenticated is False


def test_offline_cache_raises_when_evidence_is_absent(tmp_path):
    client = GitHubClient(
        GitHubSettings(),
        GitHubCredentials(),
        cache=EvidenceCache(tmp_path, offline=True),
        transport=transport(lambda request: httpx.Response(500)),
    )
    with pytest.raises(EvidenceMissing):
        client.get_json("/repos/mylonasc/tf_gnns/issues/10")


def test_failed_retries_still_count_against_the_request_budget(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"message": "boom"})

    ledger = BudgetLedger(Budgets(max_api_requests=1))
    client = make_client(
        handler,
        tmp_path,
        settings=GitHubSettings(max_retries=3),
        ledger=ledger,
    )
    with pytest.raises(Exception):
        client.get_json("/repos/mylonasc/tf_gnns/issues/10")
    assert ledger.api_requests == 1
    assert ledger.exhausted.get("api_requests")
