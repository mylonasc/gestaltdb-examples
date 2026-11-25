"""Bounded Git parsing against deterministic, synthetic repositories.

Every fixture is created with ``pygit2`` tree builders so timestamps, modes,
merge parents, renames, and missing objects are exact and reproducible. No
network access or Git CLI is involved.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pygit2
import pytest

from repo_history.budgets import BudgetExceeded, BudgetLedger
from repo_history.git import (
    GitHistory,
    GitIndex,
    GitSelection,
    MissingParentObject,
    summarize,
)
from repo_history.ids import IdFactory
from repo_history.models import Budgets, GitSettings, TimeWindow

UTC = timezone.utc
WINDOW = TimeWindow(
    start=datetime(2026, 9, 13, tzinfo=UTC),
    end=datetime(2026, 9, 15, tzinfo=UTC),
)
OUTSIDE = datetime(2026, 9, 1, tzinfo=UTC)
DAY_ONE = datetime(2026, 9, 14, 10, 0, tzinfo=UTC)
DAY_TWO = datetime(2026, 9, 14, 11, 0, tzinfo=UTC)
DAY_THREE = datetime(2026, 9, 14, 12, 0, tzinfo=UTC)
REPOSITORY_URL = "https://github.com/mylonasc/tf_gnns"


def _seconds(value: datetime) -> int:
    return int(value.timestamp())


class GitFixture:
    """Build commits without touching the working tree or a real clock."""

    def __init__(self, path) -> None:
        self.repo = pygit2.init_repository(str(path), bare=False)
        self.history = GitHistory(self.repo, REPOSITORY_URL)
        self.ids = IdFactory(REPOSITORY_URL)
        self.commits: dict[str, str] = {}

    # -- object construction ------------------------------------------------
    def _tree(self, spec: dict[str, object]) -> pygit2.Oid:
        builder = self.repo.TreeBuilder()
        for name, value in spec.items():
            if isinstance(value, dict):
                builder.insert(
                    name, self._tree(value), int(pygit2.enums.FileMode.TREE)
                )
                continue
            content, mode = value  # type: ignore[misc]
            if int(mode) == int(pygit2.enums.FileMode.COMMIT):
                oid = pygit2.Oid(hex=str(content))
            else:
                data = content.encode() if isinstance(content, str) else content
                oid = self.repo.create_blob(data)
            builder.insert(name, oid, int(mode))
        return builder.write()

    def commit(
        self,
        label: str,
        spec: dict[str, object],
        *,
        parents: tuple[str, ...] = (),
        at: datetime = DAY_ONE,
        author_at: datetime | None = None,
        ref: str = "refs/heads/main",
        author_offset: int = 0,
    ) -> str:
        tree = self._tree(spec)
        author = pygit2.Signature(
            "Fixture Author",
            "author@example.test",
            _seconds(author_at or at),
            author_offset,
        )
        committer = pygit2.Signature(
            "Fixture Committer", "committer@example.test", _seconds(at), 0
        )
        oid = self.repo.create_commit(
            ref,
            author,
            committer,
            label,
            tree,
            [pygit2.Oid(hex=parent) for parent in parents],
        )
        sha = str(oid)
        self.commits[label] = sha
        return sha

    def branch(self, name: str, sha: str) -> None:
        self.repo.create_reference(f"refs/heads/{name}", pygit2.Oid(hex=sha))

    def select(
        self,
        anchor: str,
        *,
        window: TimeWindow = WINDOW,
        max_depth: int = 256,
        skew_tolerance_seconds: int = 0,
    ) -> GitSelection:
        return self.history.select(
            anchor,
            window,
            max_depth=max_depth,
            skew_tolerance_seconds=skew_tolerance_seconds,
        )


@pytest.fixture()
def fixture(tmp_path) -> GitFixture:
    return GitFixture(tmp_path / "repository")


def test_root_commit_is_diffed_against_the_empty_tree(fixture: GitFixture):
    root = fixture.commit(
        "root",
        {
            "README.md": ("# demo\n", 0o100644),
            "docs": {"guide.md": ("guide\n", 0o100644)},
        },
        at=DAY_ONE,
    )

    selection = fixture.select(root)
    assert selection.selected_shas == (root,)
    assert selection.boundary_shas == ()
    assert not selection.truncated

    changes = fixture.history.changes(root)
    assert {change.path for change in changes} == {"README.md", "docs/guide.md"}
    assert {change.status for change in changes} == {"added"}
    assert {change.comparison for change in changes} == {"root-empty-tree"}
    assert all(change.parent_sha is None for change in changes)
    assert all(change.old_oid is None for change in changes)
    assert all(
        change.source_url == f"{REPOSITORY_URL}/blob/{root}/{change.path}"
        for change in changes
    )


def test_selection_includes_side_branches_and_merge_commits(fixture: GitFixture):
    base = fixture.commit(
        "base",
        {"base.txt": ("base\n", 0o100644)},
        at=OUTSIDE,
    )
    main = fixture.commit(
        "main",
        {"base.txt": ("base\n", 0o100644), "main.txt": ("main\n", 0o100644)},
        parents=(base,),
        at=DAY_ONE,
    )
    topic = fixture.commit(
        "topic",
        {"base.txt": ("base\n", 0o100644), "topic.txt": ("topic\n", 0o100644)},
        parents=(base,),
        at=DAY_TWO,
        ref="refs/heads/topic",
    )
    merge = fixture.commit(
        "merge",
        {
            "base.txt": ("base\n", 0o100644),
            "main.txt": ("main\n", 0o100644),
            "topic.txt": ("topic\n", 0o100644),
        },
        parents=(main, topic),
        at=DAY_THREE,
    )

    selection = fixture.select(merge)
    assert selection.selected_shas == (main, topic, merge)
    assert selection.boundary_shas == (base,)
    assert selection.selected_commits[-1].is_merge
    assert selection.selected_commits[-1].parents == (main, topic)

    index = fixture.history.index(selection, settings=GitSettings())
    assert all(
        any(snapshot.commit_sha == commit.sha for snapshot in index.snapshots)
        for commit in selection.selected_commits
    )
    assert {
        snapshot.path for snapshot in index.snapshots if snapshot.is_baseline
    } == {"base.txt"}
    merge_changes = {change.path: change for change in index.changes if change.commit_sha == merge}
    assert set(merge_changes) == {"topic.txt"}
    assert merge_changes["topic.txt"].comparison == "first-parent"
    assert merge_changes["topic.txt"].parent_sha == main
    assert index.counters["selected_commits"] == 3
    assert index.counters["boundary_commits"] == 1


def test_selection_uses_committer_time(fixture: GitFixture):
    commit = fixture.commit(
        "authored-later",
        {"file.txt": ("x\n", 0o100644)},
        at=DAY_ONE,
        author_at=datetime(2026, 9, 20, tzinfo=UTC),
    )
    selection = fixture.select(commit)
    assert selection.selected_shas == (commit,)
    assert selection.selected_commits[0].authored_at == datetime(
        2026, 9, 20, tzinfo=UTC
    )
    assert selection.selected_commits[0].committed_at == DAY_ONE


def test_window_is_half_open_and_boundary_parents_are_outside(fixture: GitFixture):
    before = fixture.commit(
        "before",
        {"f.txt": ("0\n", 0o100644)},
        at=WINDOW.start - timedelta(seconds=1),
    )
    inside = fixture.commit(
        "inside",
        {"f.txt": ("1\n", 0o100644)},
        parents=(before,),
        at=WINDOW.start,
    )
    latest = fixture.commit(
        "latest",
        {"f.txt": ("2\n", 0o100644)},
        parents=(inside,),
        at=WINDOW.end,
    )

    selection = fixture.select(latest)
    assert selection.selected_shas == (inside,)
    assert selection.boundary_shas == (before,)
    assert latest not in selection.selected_shas


def test_delete_and_rename_are_recorded(fixture: GitFixture):
    first = fixture.commit(
        "first",
        {
            "old.txt": ("identical content\n", 0o100644),
            "gone.txt": ("remove me\n", 0o100644),
            "keep.txt": ("keep\n", 0o100644),
        },
        at=DAY_ONE,
    )
    second = fixture.commit(
        "second",
        {
            "new.txt": ("identical content\n", 0o100644),
            "keep.txt": ("keep\n", 0o100644),
        },
        parents=(first,),
        at=DAY_TWO,
    )

    changes = {change.path: change for change in fixture.history.changes(second)}
    assert set(changes) == {"new.txt", "gone.txt"}

    renamed = changes["new.txt"]
    assert renamed.status == "renamed"
    assert renamed.old_path == "old.txt"
    assert renamed.similarity is not None and renamed.similarity >= 50

    deleted = changes["gone.txt"]
    assert deleted.status == "deleted"
    assert deleted.new_oid is None


def test_copy_detection_keeps_rename_and_copy_distinct(fixture: GitFixture):
    first = fixture.commit(
        "first",
        {"source.txt": ("shared body\n", 0o100644)},
        at=DAY_ONE,
    )
    second = fixture.commit(
        "second",
        {
            "source.txt": ("shared body changed\n", 0o100644),
            "copy.txt": ("shared body\n", 0o100644),
        },
        parents=(first,),
        at=DAY_TWO,
    )
    changes = {change.path: change for change in fixture.history.changes(second)}
    assert changes["copy.txt"].status == "copied"
    assert changes["copy.txt"].old_path == "source.txt"


def test_tree_modes_and_nested_paths_are_preserved(fixture: GitFixture):
    commit = fixture.commit(
        "modes",
        {
            "run.sh": ("#!/bin/sh\n", 0o100755),
            "README.md": ("# readme\n", 0o100644),
            "link": ("README.md", 0o120000),
            "vendor": ("1" * 40, 0o160000),
            "docs": {"nested": {"guide.md": ("guide\n", 0o100644)}},
        },
        at=DAY_ONE,
    )
    snapshots = {item.path: item for item in fixture.history.snapshots(commit)}
    assert snapshots["run.sh"].kind == "blob"
    assert snapshots["run.sh"].mode == 0o100755
    assert snapshots["README.md"].mode == 0o100644
    assert snapshots["link"].kind == "symlink"
    assert snapshots["link"].size == len("README.md")
    assert snapshots["vendor"].kind == "gitlink"
    assert snapshots["vendor"].content_status == "gitlink"
    assert snapshots["docs/nested/guide.md"].size == len(b"guide\n")
    assert snapshots["README.md"].snapshot_id == fixture.ids.snapshot(commit, "README.md")


def test_binary_lfs_and_oversized_content_are_flagged(fixture: GitFixture):
    lfs_pointer = (
        b"version https://git-lfs.github.com/spec/v1\n"
        b"oid sha256:deadbeef\nsize 1234\n"
    )
    commit = fixture.commit(
        "content",
        {
            "binary.dat": (b"abc\x00def", 0o100644),
            "pointer.bin": (lfs_pointer, 0o100644),
            "large.bin": (b"x" * 2048, 0o100644),
        },
        at=DAY_ONE,
    )
    snapshots = fixture.history.snapshots(commit, max_blob_bytes=1024)
    by_path = {item.path: item for item in snapshots}
    assert by_path["binary.dat"].is_binary is True
    assert by_path["pointer.bin"].is_lfs_pointer is True
    assert by_path["large.bin"].content_status == "oversized"
    assert by_path["large.bin"].is_binary is None
    assert all(item.content_status != "unavailable" for item in snapshots)

    blobs = {}
    fixture.history.snapshots(commit, max_blob_bytes=1024, blobs=blobs)
    assert {blob.content_status for blob in blobs.values()} == {
        "inspected",
        "oversized",
    }


def test_clock_skew_is_reported_and_tolerance_applies(fixture: GitFixture):
    parent = fixture.commit(
        "parent",
        {"f.txt": ("1\n", 0o100644)},
        at=DAY_TWO,
    )
    child = fixture.commit(
        "child",
        {"f.txt": ("2\n", 0o100644)},
        parents=(parent,),
        at=DAY_ONE,
    )

    strict = fixture.select(child)
    assert [(skew.commit_sha, skew.parent_sha, skew.delta_seconds) for skew in strict.clock_skews] == [
        (child, parent, 3600)
    ]
    assert "git-clock-skew" in {gap.code for gap in strict.gaps}

    tolerant = fixture.select(child, skew_tolerance_seconds=3600)
    assert tolerant.clock_skews == ()
    assert "git-clock-skew" not in {gap.code for gap in tolerant.gaps}


def test_depth_limit_is_explicitly_truncated(fixture: GitFixture):
    first = fixture.commit("first", {"f.txt": ("1\n", 0o100644)}, at=DAY_ONE)
    second = fixture.commit(
        "second", {"f.txt": ("2\n", 0o100644)}, parents=(first,), at=DAY_TWO
    )
    third = fixture.commit(
        "third", {"f.txt": ("3\n", 0o100644)}, parents=(second,), at=DAY_THREE
    )

    selection = fixture.select(third, max_depth=1)
    assert selection.selected_shas == (second, third)
    assert selection.truncated is True
    codes = {gap.code for gap in selection.gaps}
    assert "git-history-truncated" in codes
    assert "git-missing-boundary" in codes

    full = fixture.select(third, max_depth=16)
    assert full.selected_shas == (first, second, third)
    assert full.truncated is False


def test_missing_parent_object_is_a_coverage_gap(fixture: GitFixture):
    parent = fixture.commit("parent", {"f.txt": ("1\n", 0o100644)}, at=DAY_ONE)
    child = fixture.commit(
        "child", {"f.txt": ("2\n", 0o100644)}, parents=(parent,), at=DAY_TWO
    )

    object_path = os.path.join(
        fixture.repo.path, "objects", parent[:2], parent[2:]
    )
    os.remove(object_path)

    reopened = GitHistory.open(
        str(fixture.repo.path), repository_url=REPOSITORY_URL
    )
    selection = reopened.select(child, WINDOW)
    assert selection.selected_shas == (child,)
    assert selection.boundary_shas == ()

    index = reopened.index(selection)
    codes = {gap.code for gap in index.gaps}
    assert "git-missing-object" in codes
    assert "git-parent-missing" in codes
    assert index.changes == ()
    assert index.snapshots


def test_missing_anchor_is_reported_without_raising(fixture: GitFixture):
    fixture.commit("only", {"f.txt": ("1\n", 0o100644)}, at=DAY_ONE)
    selection = fixture.select("f" * 40)
    assert selection.selected_commits == ()
    assert {gap.code for gap in selection.gaps} == {"git-anchor-missing"}


def test_empty_window_is_reported_without_raising(fixture: GitFixture):
    commit = fixture.commit("outside", {"f.txt": ("1\n", 0o100644)}, at=OUTSIDE)
    empty = TimeWindow(
        start=datetime(2026, 9, 20, tzinfo=UTC),
        end=datetime(2026, 9, 21, tzinfo=UTC),
    )
    selection = fixture.select(commit, window=empty)
    assert selection.selected_commits == ()
    assert "git-selection-empty" in {gap.code for gap in selection.gaps}


def test_branch_containment_is_reported(fixture: GitFixture):
    base = fixture.commit("base", {"f.txt": ("1\n", 0o100644)}, at=DAY_ONE)
    topic = fixture.commit(
        "topic",
        {"f.txt": ("2\n", 0o100644)},
        parents=(base,),
        at=DAY_TWO,
        ref="refs/heads/topic",
    )
    anchor = fixture.commit(
        "anchor",
        {"f.txt": ("3\n", 0o100644)},
        parents=(topic,),
        at=DAY_THREE,
        ref="refs/heads/topic",
    )
    # ``main`` fast-forwards onto the same linear history as ``topic``.
    fixture.repo.create_reference(
        "refs/heads/main", pygit2.Oid(hex=anchor), force=True
    )
    fixture.branch("release", base)

    observations = {
        item.name: item for item in fixture.history.branches(anchor)
    }
    assert observations["main"].contains_anchor is True
    assert observations["topic"].contains_anchor is True
    assert observations["release"].contains_anchor is False
    assert observations["release"].tip_sha == base


def test_worktrees_include_main_and_linked_checkout(fixture: GitFixture, tmp_path):
    commit = fixture.commit("root", {"f.txt": ("1\n", 0o100644)}, at=DAY_ONE)
    fixture.repo.add_worktree(
        "secondary",
        str(tmp_path / "linked"),
        fixture.repo.references["refs/heads/main"],
    )

    observations = {
        item.identifier: item for item in fixture.history.worktrees()
    }
    assert observations["main"].is_main is True
    assert observations["main"].bare is False
    assert observations["secondary"].is_main is False
    assert observations["secondary"].branch == "main"
    assert observations["secondary"].head_sha == commit
    assert observations["secondary"].path.endswith("linked")


def test_index_charges_budgets_for_selected_and_boundary_commits(fixture: GitFixture):
    base = fixture.commit("base", {"f.txt": ("1\n", 0o100644)}, at=OUTSIDE)
    tip = fixture.commit(
        "tip", {"f.txt": ("2\n", 0o100644)}, parents=(base,), at=DAY_ONE
    )
    selection = fixture.select(tip)

    ledger = BudgetLedger(limits=Budgets(max_commits=3))
    fixture.history.index(selection, ledger=ledger)
    assert ledger.commits == 2

    tight = BudgetLedger(limits=Budgets(max_commits=1))
    with pytest.raises(BudgetExceeded):
        fixture.history.index(selection, ledger=tight)


def test_summary_is_deterministic(fixture: GitFixture):
    base = fixture.commit("base", {"f.txt": ("1\n", 0o100644)}, at=OUTSIDE)
    tip = fixture.commit(
        "tip", {"f.txt": ("2\n", 0o100644)}, parents=(base,), at=DAY_ONE
    )
    settings = GitSettings(rename_threshold=60)
    first = summarize(fixture.history.index(fixture.select(tip), settings=settings))
    second = summarize(fixture.history.index(fixture.select(tip), settings=settings))
    assert first == second
    assert first["selected_commits"] == [tip]
    assert first["boundary_commits"] == [base]
    assert first["truncated"] is False


def test_ensure_anchor_deepens_then_stops(fixture: GitFixture, monkeypatch):
    calls: list[int] = []
    answers = iter([False, False, True])
    monkeypatch.setattr(fixture.history, "has_commit", lambda revision: next(answers))
    monkeypatch.setattr(
        fixture.history, "fetch", lambda *, depth, remote="origin": calls.append(depth)
    )
    assert fixture.history.ensure_anchor(
        "a" * 40, initial_depth=8, max_depth=32
    ) is True
    assert calls == [8, 16]

    monkeypatch.setattr(fixture.history, "has_commit", lambda revision: False)
    calls.clear()
    assert fixture.history.ensure_anchor(
        "a" * 40, initial_depth=8, max_depth=32
    ) is False
    assert calls == [8, 16, 32]


def test_fetch_rejects_non_positive_depth(fixture: GitFixture):
    with pytest.raises(ValueError):
        fixture.history.fetch(depth=0)
    with pytest.raises(ValueError):
        fixture.select(fixture.commit("x", {}, at=DAY_ONE), max_depth=0)


def test_rename_threshold_is_configurable(fixture: GitFixture):
    first = fixture.commit(
        "first",
        {"a.txt": ("alpha\nbeta\ngamma\ndelta\n", 0o100644)},
        at=DAY_ONE,
    )
    second = fixture.commit(
        "second",
        {"b.txt": ("alpha\nbeta\ngamma\nepsilon\n", 0o100644)},
        parents=(first,),
        at=DAY_TWO,
    )
    strict = fixture.history.changes(second, rename_threshold=100)
    assert [change.status for change in strict] == ["deleted", "added"]
    loose = fixture.history.changes(second, rename_threshold=50)
    assert [change.status for change in loose] == ["renamed"]


def test_index_returns_git_index_type(fixture: GitFixture):
    commit = fixture.commit("root", {"f.txt": ("1\n", 0o100644)}, at=DAY_ONE)
    index = fixture.history.index(fixture.select(commit))
    assert isinstance(index, GitIndex)
    assert index.selection.anchor == commit
    assert set(index.counters) == {
        "selected_commits",
        "boundary_commits",
        "snapshots",
        "blobs",
        "changes",
        "branches",
        "worktrees",
    }


def test_missing_parent_exception_names_the_parent(fixture: GitFixture):
    with pytest.raises(MissingParentObject) as error:
        raise MissingParentObject("a" * 40)
    assert error.value.parent_sha == "a" * 40
    assert "a" * 40 in str(error.value)
