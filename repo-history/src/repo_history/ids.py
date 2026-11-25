"""Stable, bounded identifiers for repository-qualified graph entities."""

from __future__ import annotations

import hashlib
import json
import re
from urllib.parse import urlsplit, urlunsplit


def canonical_repository_url(url: str) -> str:
    """Return a normalized HTTPS identity without credentials or a trailing slash."""

    value = url.strip()
    if "://" not in value:
        value = f"https://{value}"
    parts = urlsplit(value)
    host = (parts.hostname or "").lower()
    path = re.sub(r"\.git$", "", parts.path.rstrip("/"), flags=re.IGNORECASE)
    if not host or not path:
        raise ValueError(f"Repository URL must include a host and path: {url!r}")
    return urlunsplit(("https", host, path, "", ""))


def stable_id(kind: str, *parts: object, size: int = 24) -> str:
    """Create a deterministic repository-safe ID within GestaltDB key limits."""

    if not kind or any(char in kind for char in ":\n\r\t"):
        raise ValueError("kind must be a non-empty token without separators")
    if size < 12 or size > 64:
        raise ValueError("size must be between 12 and 64")
    material = json.dumps([str(part) for part in parts], ensure_ascii=True, separators=(",", ":"))
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:size]
    return f"{kind}:{digest}"


class IdFactory:
    """Repository-scoped identity factory used by every ingestion adapter."""

    def __init__(self, repository_url: str) -> None:
        self.repository_url = canonical_repository_url(repository_url)
        self.repository_id = stable_id("repo", self.repository_url)

    def commit(self, sha: str) -> str:
        return stable_id("commit", self.repository_id, sha.lower())

    def branch(self, name: str) -> str:
        return stable_id("branch", self.repository_id, name)

    def file(self, path: str) -> str:
        return stable_id("file", self.repository_id, path)

    def snapshot(self, commit_sha: str, path: str) -> str:
        return stable_id("snap", self.repository_id, commit_sha.lower(), path)

    def blob(self, content_hash: str) -> str:
        return stable_id("blob", content_hash.lower())

    def content(self, oid: str, kind: str) -> str:
        return stable_id("content", self.repository_id, oid.lower(), kind)

    def change(self, commit_sha: str, parent_sha: str | None, path: str) -> str:
        return stable_id("change", self.repository_id, commit_sha.lower(), parent_sha or "root", path)

    def section(self, commit_sha: str, path: str, ordinal: int) -> str:
        return stable_id("section", self.repository_id, commit_sha.lower(), path, ordinal)

    def work_item(self, provider: str, number: int) -> str:
        return stable_id("workitem", self.repository_id, provider.lower(), number)

    def provider_label(self, provider: str, name: str) -> str:
        return stable_id("providerlabel", self.repository_id, provider.lower(), name)

    def topic(self, canonical_name: str) -> str:
        normalized = re.sub(r"\s+", " ", canonical_name.strip().lower())
        if not normalized:
            raise ValueError("Topic names cannot be empty")
        return stable_id("topic", normalized, size=20)

    def analysis_run(self, run_key: str) -> str:
        return stable_id("run", self.repository_id, run_key)

    def evidence(self, source: str, source_id: str, locator: str = "") -> str:
        return stable_id("evidence", self.repository_id, source, source_id, locator)

    def worktree(self, identifier: str) -> str:
        return stable_id("worktree", self.repository_id, identifier)
