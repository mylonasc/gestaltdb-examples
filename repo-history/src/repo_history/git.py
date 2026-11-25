"""Library-only bounded Git history parsing.

The parser uses ``pygit2`` for every Git operation: no subprocess or CLI calls.
It selects commits from a pinned anchor and explicit time window, records full
trees for selected commits and immediate out-of-window boundary commits, and
derives change records with an explicit first-parent policy.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable, Iterator, Literal

import pygit2
from pydantic import Field

from .budgets import BudgetLedger
from .ids import IdFactory, canonical_repository_url
from .models import (
    CoverageGap,
    CoverageStatus,
    FrozenModel,
    GitSettings,
    TimeWindow,
)

LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
BINARY_SNIFF_BYTES = 8000

ChangeStatus = Literal[
    "added", "modified", "deleted", "renamed", "copied", "typechange",
    "unreadable", "unknown",
]
ContentStatus = Literal["inspected", "oversized", "unavailable", "gitlink"]

_STATUS_BY_CHAR: dict[str, ChangeStatus] = {
    "A": "added",
    "M": "modified",
    "D": "deleted",
    "R": "renamed",
    "C": "copied",
    "T": "typechange",
    "U": "unreadable",
}


def _git_time(timestamp: int) -> datetime:
    """Normalize a Git timestamp to UTC while preserving the instant.

    Git stores committer/author time as a Unix instant plus the author's local
    offset. The offset does not change the instant, so it is intentionally not
    applied: temporal replay must use one comparable clock.
    """

    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


class MissingParentObject(RuntimeError):
    """A selected commit's parent tree is absent from the local object database."""

    def __init__(self, parent_sha: str) -> None:
        super().__init__(f"parent commit object is unavailable: {parent_sha}")
        self.parent_sha = parent_sha


class CommitRecord(FrozenModel):
    sha: str
    parents: tuple[str, ...]
    tree_sha: str
    message: str
    author_name: str
    author_email: str
    authored_at: datetime
    committer_name: str
    committer_email: str
    committed_at: datetime
    source_url: str
    is_merge: bool


class ClockSkew(FrozenModel):
    commit_sha: str
    parent_sha: str
    delta_seconds: int


class GitSelection(FrozenModel):
    anchor: str
    window: TimeWindow
    selected_commits: tuple[CommitRecord, ...] = ()
    boundary_commits: tuple[CommitRecord, ...] = ()
    clock_skews: tuple[ClockSkew, ...] = ()
    gaps: tuple[CoverageGap, ...] = ()
    visited_commits: int = 0
    depth_limit: int = 0
    truncated: bool = False

    @property
    def selected_shas(self) -> tuple[str, ...]:
        return tuple(commit.sha for commit in self.selected_commits)

    @property
    def boundary_shas(self) -> tuple[str, ...]:
        return tuple(commit.sha for commit in self.boundary_commits)


class TreeEntryRecord(FrozenModel):
    snapshot_id: str
    commit_sha: str
    path: str
    object_id: str
    mode: int
    kind: Literal["blob", "symlink", "gitlink", "unknown"]
    size: int | None = None
    is_binary: bool | None = None
    is_lfs_pointer: bool | None = None
    content_status: ContentStatus
    is_baseline: bool = False
    source_url: str


class BlobRecord(FrozenModel):
    blob_id: str
    oid: str
    size: int
    is_binary: bool | None = None
    is_lfs_pointer: bool | None = None
    content_status: ContentStatus


class FileChangeRecord(FrozenModel):
    change_id: str
    commit_sha: str
    parent_sha: str | None
    comparison: Literal["first-parent", "root-empty-tree"]
    path: str
    old_path: str | None = None
    status: ChangeStatus
    old_oid: str | None = None
    new_oid: str | None = None
    old_mode: int | None = None
    new_mode: int | None = None
    similarity: int | None = None
    is_binary: bool | None = None
    source_url: str


class BranchObservation(FrozenModel):
    name: str
    ref: str
    tip_sha: str
    contains_anchor: bool
    source: Literal["local", "remote"] = "local"


class WorktreeObservation(FrozenModel):
    identifier: str
    path: str
    head_sha: str | None
    branch: str | None
    detached: bool
    bare: bool
    is_main: bool


class GitIndex(FrozenModel):
    selection: GitSelection
    snapshots: tuple[TreeEntryRecord, ...] = ()
    blobs: tuple[BlobRecord, ...] = ()
    changes: tuple[FileChangeRecord, ...] = ()
    branches: tuple[BranchObservation, ...] = ()
    worktrees: tuple[WorktreeObservation, ...] = ()
    gaps: tuple[CoverageGap, ...] = ()
    counters: dict[str, int] = Field(default_factory=dict)


class GitHistory:
    """A thin, testable wrapper over a local ``pygit2`` repository."""

    def __init__(self, repository: pygit2.Repository, repository_url: str) -> None:
        self.repository = repository
        self.repository_url = canonical_repository_url(repository_url)
        self.ids = IdFactory(self.repository_url)

    # ------------------------------------------------------------------ opening
    @classmethod
    def open(cls, path: str, *, repository_url: str) -> "GitHistory":
        return cls(pygit2.Repository(path), repository_url)

    @classmethod
    def clone(
        cls,
        url: str,
        path: str,
        *,
        repository_url: str | None = None,
        depth: int = 64,
        bare: bool = True,
    ) -> "GitHistory":
        repository = pygit2.clone_repository(url, path, bare=bare, depth=depth)
        return cls(repository, repository_url or url)

    def close(self) -> None:
        """Release the underlying libgit2 handle explicitly."""

        free = getattr(self.repository, "free", None)
        if callable(free):
            free()
        self.repository = None  # type: ignore[assignment]

    # ------------------------------------------------------------------ fetching
    def has_commit(self, revision: str) -> bool:
        try:
            self.repository.revparse_single(revision).peel(pygit2.Commit)
        except (KeyError, ValueError):
            return False
        return True

    def read_blob(self, oid: str) -> bytes | None:
        """Return raw blob bytes, or ``None`` when the object is unavailable."""

        try:
            blob = self.repository[oid]
        except KeyError:
            return None
        if not isinstance(blob, pygit2.Blob):
            return None
        return blob.data

    def fetch(self, *, depth: int, remote: str = "origin") -> None:
        """Fetch all branches to a bounded depth using libgit2 only."""

        if depth < 1:
            raise ValueError("fetch depth must be positive")
        handle = self.repository.remotes[remote]
        handle.fetch(
            ["+refs/heads/*:refs/remotes/" + remote + "/*"],
            depth=depth,
        )

    def ensure_anchor(
        self,
        anchor: str,
        *,
        initial_depth: int = 64,
        max_depth: int = 512,
        remote: str = "origin",
    ) -> bool:
        """Deepen a bounded fetch until the anchor exists or the cap is reached."""

        if self.has_commit(anchor):
            return True
        depth = initial_depth
        while depth <= max_depth:
            self.fetch(depth=depth, remote=remote)
            if self.has_commit(anchor):
                return True
            depth *= 2
        return False

    # ------------------------------------------------------------------ selection
    def select(
        self,
        anchor: str,
        window: TimeWindow,
        *,
        max_depth: int = 256,
        skew_tolerance_seconds: int = 0,
    ) -> GitSelection:
        if max_depth < 1:
            raise ValueError("max_depth must be positive")

        try:
            anchor_commit = self.repository.revparse_single(anchor).peel(pygit2.Commit)
        except (KeyError, ValueError) as error:
            gap = CoverageGap(
                code="git-anchor-missing",
                status=CoverageStatus.UNAVAILABLE,
                message=f"anchor commit is not available in the local object database: {error}",
                source="git",
                source_ids=(anchor,),
                retryable=True,
            )
            return GitSelection(
                anchor=anchor, window=window, gaps=(gap,), depth_limit=max_depth
            )

        anchor_sha = str(anchor_commit.id)
        visited: dict[str, CommitRecord] = {}
        missing: set[str] = set()
        truncated = False
        stack: list[tuple[str, int]] = [(anchor_sha, 0)]

        while stack:
            sha, depth = stack.pop()
            if sha in visited or sha in missing:
                continue
            try:
                commit = self.repository[sha].peel(pygit2.Commit)
            except KeyError:
                missing.add(sha)
                continue
            record = self._commit_record(commit)
            visited[sha] = record
            if depth >= max_depth:
                if commit.parent_ids:
                    truncated = True
                continue
            for parent in reversed(commit.parent_ids):
                stack.append((str(parent), depth + 1))

        selected = tuple(
            sorted(
                (record for record in visited.values() if window.contains(record.committed_at)),
                key=lambda record: (record.committed_at, record.sha),
            )
        )
        selected_shas = {record.sha for record in selected}
        boundary_shas = sorted(
            {
                parent
                for record in selected
                for parent in record.parents
                if parent not in selected_shas
            }
        )
        boundary = tuple(visited[sha] for sha in boundary_shas if sha in visited)
        unresolved_boundary = [sha for sha in boundary_shas if sha not in visited]

        gaps: list[CoverageGap] = []
        if truncated:
            gaps.append(
                CoverageGap(
                    code="git-history-truncated",
                    status=CoverageStatus.PARTIAL,
                    message=(
                        f"ancestry walk stopped at the configured depth {max_depth}; "
                        "older selected commits may be missing"
                    ),
                    source="git",
                    source_ids=(anchor_sha,),
                    retryable=True,
                )
            )
        for sha in sorted(missing):
            gaps.append(
                CoverageGap(
                    code="git-missing-object",
                    status=CoverageStatus.UNAVAILABLE,
                    message="commit object is absent from the local shallow history",
                    source="git",
                    source_ids=(sha,),
                    retryable=True,
                )
            )
        for sha in unresolved_boundary:
            if sha not in missing:
                gaps.append(
                    CoverageGap(
                        code="git-missing-boundary",
                        status=CoverageStatus.UNAVAILABLE,
                        message="boundary parent was not resolved during the bounded walk",
                        source="git",
                        source_ids=(sha,),
                        retryable=True,
                    )
                )
        if not selected:
            gaps.append(
                CoverageGap(
                    code="git-selection-empty",
                    status=CoverageStatus.UNAVAILABLE,
                    message="no commit reachable from the anchor falls inside the time window",
                    source="git",
                    source_ids=(anchor_sha,),
                )
            )

        clock_skews: list[ClockSkew] = []
        tolerance = timedelta(seconds=skew_tolerance_seconds)
        for record in sorted(visited.values(), key=lambda item: (item.committed_at, item.sha)):
            for parent_sha in record.parents:
                parent = visited.get(parent_sha)
                if parent is None:
                    continue
                delta = parent.committed_at - record.committed_at
                if delta > tolerance:
                    clock_skews.append(
                        ClockSkew(
                            commit_sha=record.sha,
                            parent_sha=parent_sha,
                            delta_seconds=int(delta.total_seconds()),
                        )
                    )
        for skew in clock_skews:
            gaps.append(
                CoverageGap(
                    code="git-clock-skew",
                    status=CoverageStatus.PARTIAL,
                    message=(
                        f"parent committer time is {skew.delta_seconds}s newer than "
                        "its child; replay must respect causal order"
                    ),
                    source="git",
                    source_ids=(skew.commit_sha, skew.parent_sha),
                )
            )

        return GitSelection(
            anchor=anchor_sha,
            window=window,
            selected_commits=selected,
            boundary_commits=boundary,
            clock_skews=tuple(clock_skews),
            gaps=tuple(gaps),
            visited_commits=len(visited),
            depth_limit=max_depth,
            truncated=truncated,
        )

    # ------------------------------------------------------------------ trees
    def snapshots(
        self,
        commit_sha: str,
        *,
        baseline: bool = False,
        max_blob_bytes: int = 2 * 1024 * 1024,
        blobs: dict[str, BlobRecord] | None = None,
    ) -> tuple[TreeEntryRecord, ...]:
        commit = self.repository[commit_sha].peel(pygit2.Commit)
        blob_index = blobs if blobs is not None else {}
        records: list[TreeEntryRecord] = []
        for path, entry in self._walk_tree(commit.tree):
            mode = int(entry.filemode)
            if mode == int(pygit2.enums.FileMode.COMMIT):
                kind = "gitlink"
            elif mode == int(pygit2.enums.FileMode.LINK):
                kind = "symlink"
            elif mode in (
                int(pygit2.enums.FileMode.BLOB),
                int(pygit2.enums.FileMode.BLOB_EXECUTABLE),
            ):
                kind = "blob"
            else:
                kind = "unknown"

            size: int | None = None
            is_binary: bool | None = None
            is_lfs: bool | None = None
            content_status: ContentStatus = "gitlink"
            if kind in {"blob", "symlink"}:
                try:
                    blob = self.repository[entry.id]
                    size = blob.size
                    is_binary, is_lfs, content_status = self._inspect_blob(
                        blob, size, max_blob_bytes
                    )
                    blob_index.setdefault(
                        str(entry.id),
                        BlobRecord(
                            blob_id=self.ids.blob(str(entry.id)),
                            oid=str(entry.id),
                            size=size,
                            is_binary=is_binary,
                            is_lfs_pointer=is_lfs,
                            content_status=content_status,
                        ),
                    )
                except KeyError:
                    content_status = "unavailable"
            elif kind == "unknown":
                content_status = "unavailable"

            records.append(
                TreeEntryRecord(
                    snapshot_id=self.ids.snapshot(commit_sha, path),
                    commit_sha=commit_sha,
                    path=path,
                    object_id=str(entry.id),
                    mode=mode,
                    kind=kind,
                    size=size,
                    is_binary=is_binary,
                    is_lfs_pointer=is_lfs,
                    content_status=content_status,
                    is_baseline=baseline,
                    source_url=self.source_url(commit_sha, path),
                )
            )
        return tuple(records)

    def changes(
        self,
        commit_sha: str,
        *,
        parent_sha: str | None = None,
        rename_threshold: int = 50,
        max_blob_bytes: int = 2 * 1024 * 1024,
    ) -> tuple[FileChangeRecord, ...]:
        commit = self.repository[commit_sha].peel(pygit2.Commit)
        if parent_sha is None and commit.parent_ids:
            parent_sha = str(commit.parent_ids[0])
        if parent_sha is None:
            old_tree = self.repository[self._empty_tree_oid()]
            comparison = "root-empty-tree"
        else:
            try:
                old_tree = self.repository[parent_sha].peel(pygit2.Commit).tree
            except (KeyError, pygit2.GitError) as error:
                raise MissingParentObject(parent_sha) from error
            comparison = "first-parent"

        diff = old_tree.diff_to_tree(
            commit.tree, flags=pygit2.enums.DiffOption.INCLUDE_TYPECHANGE
        )
        diff.find_similar(
            flags=pygit2.enums.DiffFind.FIND_RENAMES
            | pygit2.enums.DiffFind.FIND_COPIES,
            rename_threshold=rename_threshold,
        )

        records: list[FileChangeRecord] = []
        for patch in diff:
            delta = patch.delta
            status = _STATUS_BY_CHAR.get(delta.status_char(), "unknown")
            old_path = delta.old_file.path
            new_path = delta.new_file.path
            path = new_path if status != "deleted" else old_path
            old_oid = None if status == "added" else str(delta.old_file.id)
            new_oid = None if status == "deleted" else str(delta.new_file.id)
            old_mode = None if status == "added" else int(delta.old_file.mode)
            new_mode = None if status == "deleted" else int(delta.new_file.mode)
            similarity = (
                int(delta.similarity) if status in {"renamed", "copied"} else None
            )
            is_binary = self._change_binary(
                new_oid or old_oid, max_blob_bytes
            )
            records.append(
                FileChangeRecord(
                    change_id=self.ids.change(commit_sha, parent_sha, path),
                    commit_sha=commit_sha,
                    parent_sha=parent_sha,
                    comparison=comparison,
                    path=path,
                    old_path=old_path if status in {"renamed", "copied"} else None,
                    status=status,
                    old_oid=old_oid,
                    new_oid=new_oid,
                    old_mode=old_mode,
                    new_mode=new_mode,
                    similarity=similarity,
                    is_binary=is_binary,
                    source_url=self.source_url(commit_sha, path),
                )
            )
        return tuple(sorted(records, key=lambda record: record.path))

    # ------------------------------------------------------------------ refs
    def branches(self, anchor: str) -> tuple[BranchObservation, ...]:
        anchor_commit = self.repository.revparse_single(anchor).peel(pygit2.Commit)
        observations: list[BranchObservation] = []
        for ref in self.repository.references:
            if ref.startswith("refs/heads/"):
                source: Literal["local", "remote"] = "local"
                name = ref.removeprefix("refs/heads/")
            elif ref.startswith("refs/remotes/"):
                source = "remote"
                name = ref.removeprefix("refs/remotes/")
                if name.endswith("/HEAD"):
                    continue
            else:
                continue
            try:
                tip = self.repository.references[ref].peel(pygit2.Commit)
            except (KeyError, pygit2.GitError):
                continue
            contains = tip.id == anchor_commit.id or self.repository.descendant_of(
                tip.id, anchor_commit.id
            )
            observations.append(
                BranchObservation(
                    name=name,
                    ref=ref,
                    tip_sha=str(tip.id),
                    contains_anchor=contains,
                    source=source,
                )
            )
        return tuple(sorted(observations, key=lambda item: (item.source, item.name)))

    def worktrees(self) -> tuple[WorktreeObservation, ...]:
        observations: list[WorktreeObservation] = []
        if not self.repository.is_bare and self.repository.workdir:
            head_sha, branch, detached = self._head(self.repository)
            observations.append(
                WorktreeObservation(
                    identifier="main",
                    path=str(self.repository.workdir),
                    head_sha=head_sha,
                    branch=branch,
                    detached=detached,
                    bare=False,
                    is_main=True,
                )
            )
        try:
            linked = self.repository.list_worktrees()
        except (AttributeError, pygit2.GitError):
            linked = []
        for worktree_name in linked:
            try:
                worktree = self.repository.lookup_worktree(worktree_name)
                path = str(worktree.path)
            except (AttributeError, KeyError, pygit2.GitError):
                continue
            try:
                handle = pygit2.Repository(path)
                head_sha, branch, detached = self._head(handle)
                bare = handle.is_bare
            except (KeyError, pygit2.GitError):
                head_sha, branch, detached, bare = None, None, False, False
            observations.append(
                WorktreeObservation(
                    identifier=worktree_name,
                    path=path,
                    head_sha=head_sha,
                    branch=branch,
                    detached=detached,
                    bare=bare,
                    is_main=False,
                )
            )
        return tuple(observations)

    # ------------------------------------------------------------------ assembly
    def index(
        self,
        selection: GitSelection,
        *,
        settings: GitSettings | None = None,
        ledger: BudgetLedger | None = None,
    ) -> GitIndex:
        settings = settings or GitSettings()
        if ledger is not None:
            ledger.reserve_commit(len(selection.selected_commits))
            ledger.reserve_commit(len(selection.boundary_commits))

        blobs: dict[str, BlobRecord] = {}
        snapshots: list[TreeEntryRecord] = []
        for commit in selection.selected_commits:
            snapshots.extend(
                self.snapshots(
                    commit.sha, max_blob_bytes=settings.max_blob_bytes, blobs=blobs
                )
            )
        for commit in selection.boundary_commits:
            snapshots.extend(
                self.snapshots(
                    commit.sha,
                    baseline=True,
                    max_blob_bytes=settings.max_blob_bytes,
                    blobs=blobs,
                )
            )

        changes: list[FileChangeRecord] = []
        change_gaps: list[CoverageGap] = []
        for commit in selection.selected_commits:
            try:
                changes.extend(
                    self.changes(
                        commit.sha,
                        rename_threshold=settings.rename_threshold,
                        max_blob_bytes=settings.max_blob_bytes,
                    )
                )
            except MissingParentObject as error:
                change_gaps.append(
                    CoverageGap(
                        code="git-parent-missing",
                        status=CoverageStatus.PARTIAL,
                        message=(
                            "parent tree is absent from the bounded history; "
                            "file changes for this commit are not reconstructable"
                        ),
                        source="git",
                        source_ids=(commit.sha, error.parent_sha),
                        retryable=True,
                    )
                )

        gaps = [*selection.gaps, *change_gaps]
        unavailable = sorted(
            {
                snapshot.path
                for snapshot in snapshots
                if snapshot.content_status == "unavailable"
            }
        )
        if unavailable:
            gaps.append(
                CoverageGap(
                    code="git-blob-unavailable",
                    status=CoverageStatus.UNAVAILABLE,
                    message="some tree entries could not be read from the local object database",
                    source="git",
                    source_ids=tuple(unavailable),
                    retryable=True,
                )
            )

        branches = self.branches(selection.anchor)
        worktrees = self.worktrees()
        counters = {
            "selected_commits": len(selection.selected_commits),
            "boundary_commits": len(selection.boundary_commits),
            "snapshots": len(snapshots),
            "blobs": len(blobs),
            "changes": len(changes),
            "branches": len(branches),
            "worktrees": len(worktrees),
        }
        return GitIndex(
            selection=selection,
            snapshots=tuple(snapshots),
            blobs=tuple(sorted(blobs.values(), key=lambda blob: blob.oid)),
            changes=tuple(changes),
            branches=branches,
            worktrees=worktrees,
            gaps=tuple(gaps),
            counters=counters,
        )

    # ------------------------------------------------------------------ helpers
    def source_url(self, commit_sha: str, path: str | None = None) -> str:
        if path is None:
            return f"{self.repository_url}/commit/{commit_sha}"
        return f"{self.repository_url}/blob/{commit_sha}/{path}"

    def _empty_tree_oid(self) -> pygit2.Oid:
        return self.repository.TreeBuilder().write()

    def _commit_record(self, commit: pygit2.Commit) -> CommitRecord:
        sha = str(commit.id)
        return CommitRecord(
            sha=sha,
            parents=tuple(str(parent) for parent in commit.parent_ids),
            tree_sha=str(commit.tree_id),
            message=commit.message or "",
            author_name=commit.author.name,
            author_email=commit.author.email,
            authored_at=_git_time(commit.author.time),
            committer_name=commit.committer.name,
            committer_email=commit.committer.email,
            committed_at=_git_time(commit.committer.time),
            source_url=self.source_url(sha),
            is_merge=len(commit.parent_ids) > 1,
        )

    def _walk_tree(
        self, tree: pygit2.Tree, prefix: str = ""
    ) -> Iterator[tuple[str, pygit2.Blob]]:
        for entry in tree:
            path = f"{prefix}{entry.name}"
            if int(entry.filemode) == int(pygit2.enums.FileMode.TREE):
                yield from self._walk_tree(self.repository[entry.id], path + "/")
            else:
                yield path, entry

    @staticmethod
    def _inspect_blob(
        blob: pygit2.Blob, size: int, max_blob_bytes: int
    ) -> tuple[bool | None, bool | None, ContentStatus]:
        if size > max_blob_bytes:
            return None, None, "oversized"
        data = blob.data
        return (
            b"\x00" in data[:BINARY_SNIFF_BYTES],
            data.startswith(LFS_POINTER_PREFIX),
            "inspected",
        )

    def _change_binary(self, oid: str | None, max_blob_bytes: int) -> bool | None:
        if oid is None:
            return None
        try:
            blob = self.repository[oid]
        except KeyError:
            return None
        if blob.size > max_blob_bytes:
            return None
        return b"\x00" in blob.data[:BINARY_SNIFF_BYTES]

    @staticmethod
    def _head(
        repository: pygit2.Repository,
    ) -> tuple[str | None, str | None, bool]:
        try:
            head = repository.head
            head_sha = str(head.target)
        except (KeyError, pygit2.GitError):
            return None, None, repository.head_is_detached
        branch = head.shorthand if not repository.head_is_detached else None
        return head_sha, branch, repository.head_is_detached


def summarize(index: GitIndex) -> dict[str, object]:
    """Small deterministic summary suitable for manifests and reports."""

    return {
        "anchor": index.selection.anchor,
        "selected_commits": list(index.selection.selected_shas),
        "boundary_commits": list(index.selection.boundary_shas),
        "counters": index.counters,
        "gaps": [gap.code for gap in index.gaps],
        "truncated": index.selection.truncated,
        "clock_skews": [
            skew.model_dump(mode="json") for skew in index.selection.clock_skews
        ],
    }


def iter_selected(index: GitIndex) -> Iterable[CommitRecord]:
    return index.selection.selected_commits
