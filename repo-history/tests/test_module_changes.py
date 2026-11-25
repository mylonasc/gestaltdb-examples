"""Explicit module-change relationships, separate from symbol-body changes."""

from __future__ import annotations

import pytest

from repo_history.content import ContentExtractor
from repo_history.graph import map_code_components, map_diffs
from repo_history.ids import IdFactory
from repo_history.models import ContentSettings
from test_git import REPOSITORY_URL
from test_python_index import project
from test_viz_d3 import NOW, fixture_discovery
from repo_history.viz_d3 import build_viewer_data


def module_edges(mapped):
    return [r for r in mapped.relationships if r.edge_type == "AFFECTS" and r.properties["basis"] == "file-change"]


@pytest.mark.parametrize("old_header,new_header", [
    ("import os\n", "import sys\nimport json\n"),
    ("LIMIT = 1\n", "LIMIT = 2\n"),
    ("# old comment\n", "# new comment\n"),
])
def test_module_level_edits_affect_module_but_not_unchanged_function(tmp_path, old_header, new_header):
    body = "\ndef stable():\n    return 1\n"
    before = {"pkg": {"__init__.py": ("", 0o100644), "mod.py": (old_header + body, 0o100644)}}
    after = {"pkg": {"__init__.py": ("", 0o100644), "mod.py": (new_header + body, 0o100644)}}
    repo, index, content = project(tmp_path, before, after)
    ids = IdFactory(REPOSITORY_URL)
    mapped = map_diffs(content, index, ids, run_id="test")
    affects = [r for r in mapped.relationships if r.edge_type == "AFFECTS"]
    assert len(affects) == 1
    edge = affects[0]
    assert edge.source == mapped.entities[0].logical_id
    assert edge.target == ids.module("pkg/mod.py")
    assert edge.properties == {
        "status": "modified", "basis": "file-change", "path": "pkg/mod.py",
        "change_sides": ["before", "after"],
    }
    assert edge.valid_from == index.selection.selected_commits[0].committed_at
    assert edge.evidence_ids
    # Do not add transitive package-module changes or touches to carried files.
    assert not any(r.target == ids.module("pkg/__init__.py") for r in affects)
    repo.history.close()


@pytest.mark.parametrize("status,before,after,expected", [
    ("added", {"keep.txt": ("keep", 0o100644)},
     {"keep.txt": ("keep", 0o100644), "mod.py": ("import os\n", 0o100644)},
     {"mod.py": ["after"]}),
    ("deleted", {"mod.py": ("import os\n", 0o100644), "keep.txt": ("keep", 0o100644)},
     {"keep.txt": ("keep", 0o100644)}, {"mod.py": ["before"]}),
    ("renamed", {"old.py": ("import os\n", 0o100644)},
     {"new.py": ("import os\n", 0o100644)}, {"old.py": ["before"], "new.py": ["after"]}),
    ("renamed", {"old.py": ("import os\n", 0o100644)},
     {"new.txt": ("import os\n", 0o100644)}, {"old.py": ["before"]}),
    ("renamed", {"old.txt": ("import os\n", 0o100644)},
     {"new.py": ("import os\n", 0o100644)}, {"new.py": ["after"]}),
])
def test_module_add_delete_and_rename_sides(tmp_path, status, before, after, expected):
    repo, index, content = project(tmp_path, before, after)
    assert [c.status for c in index.changes] == [status]
    ids = IdFactory(REPOSITORY_URL)
    mapped = map_diffs(content, index, ids, run_id="test")
    edges = module_edges(mapped)
    assert {r.target: r.properties["change_sides"] for r in edges} == {
        ids.module(path): sides for path, sides in expected.items()
    }
    assert all(r.properties["status"] == status for r in edges)
    assert len({r.edge_id for r in edges}) == len(edges)
    known_modules = {e.logical_id for e in map_code_components(content, index, ids, run_id="test").entities if e.labels == ("Module",)}
    assert all(r.target in known_modules for r in edges)
    repo.history.close()


def test_copy_affects_only_destination_module(tmp_path):
    before = {"source.py": ("import os\n", 0o100644)}
    after = {**before, "copy.py": ("import os\n", 0o100644)}
    repo, index, content = project(tmp_path, before, after)
    # libgit2's default copy policy need not discover unchanged copy sources;
    # supply the observed copy record to check the mapper's copy semantics.
    change = index.changes[0]
    index = index.model_copy(update={"changes": (change.model_copy(update={
        "status": "copied", "old_path": "source.py", "old_oid": change.new_oid,
    }),)})
    ids = IdFactory(REPOSITORY_URL)
    edges = module_edges(map_diffs(content, index, ids, run_id="test"))
    assert len(edges) == 1
    assert edges[0].target == ids.module("copy.py")
    assert edges[0].properties["status"] == "copied"
    assert edges[0].properties["change_sides"] == ["after"]
    repo.history.close()


@pytest.mark.parametrize("path,before_text,after_text", [
    ("mod.pyi", "from typing import Any\n", "from typing import Optional\n"),
    ("scripts/run", "#!/usr/bin/env python3\nimport os\n", "#!/usr/bin/env python3\nimport sys\n"),
    ("broken.py", "def broken(:\n", "def still_broken(:\n"),
])
def test_stubs_scripts_and_invalid_python_still_have_module_change_links(tmp_path, path, before_text, after_text):
    def tree(text):
        if "/" in path:
            folder, name = path.split("/", 1)
            return {folder: {name: (text, 0o100644)}}
        return {path: (text, 0o100644)}

    repo, index, content = project(tmp_path, tree(before_text), tree(after_text))
    ids = IdFactory(REPOSITORY_URL)
    edges = module_edges(map_diffs(content, index, ids, run_id="test"))
    assert len(edges) == 1 and edges[0].target == ids.module(path)
    repo.history.close()


def test_oversized_source_retains_module_links_without_inventing_symbol_edits(tmp_path):
    repo, index, _ = project(
        tmp_path, {"mod.py": ("x = '" + "a" * 2000 + "'\n", 0o100644)},
        {"mod.py": ("x = '" + "b" * 2000 + "'\n", 0o100644)},
    )
    content = ContentExtractor.from_history(repo.history, settings=ContentSettings(max_file_bytes=1024)).extract(index)
    mapped = map_diffs(content, index, IdFactory(REPOSITORY_URL), run_id="test")
    assert len(module_edges(mapped)) == 1
    assert not [r for r in mapped.relationships if r.edge_type == "AFFECTS" and r.properties["basis"] == "before-after-source-comparison"]
    assert mapped.entities[0].properties["patch_status"] == "content-unavailable"
    repo.history.close()


def test_no_module_changes_for_unchanged_python_or_nonpython_edits(tmp_path):
    before = {"mod.py": ("import os\n", 0o100644), "notes.txt": ("old\n", 0o100644)}
    after = {**before, "notes.txt": ("new\n", 0o100644)}
    repo, index, content = project(tmp_path, before, after)
    mapped = map_diffs(content, index, IdFactory(REPOSITORY_URL), run_id="test")
    assert not module_edges(mapped)
    repo.history.close()


def test_viewer_preserves_explicit_commit_diff_module_chain_once(tmp_path):
    repo, index, content = project(
        tmp_path, {"mod.py": ("import os\n", 0o100644)},
        {"mod.py": ("import sys\n", 0o100644)},
    )
    ids = IdFactory(REPOSITORY_URL)
    components = map_code_components(content, index, ids, run_id="test")
    diffs = map_diffs(content, index, ids, run_id="test")
    data = build_viewer_data(
        index=index, content=content, discovery=fixture_discovery(), links=[], assignments=[],
        info_horizon=NOW, run_id="test", repository_url=REPOSITORY_URL, max_nodes=100,
        extra_nodes=[*components.entities, *diffs.entities],
        extra_edges=[*components.relationships, *diffs.relationships],
    )
    module = ids.module("mod.py")
    affects = [e for e in data["edges"] if e["relation"] == "AFFECTS" and e["target"] == module]
    assert len(affects) == 1
    assert affects[0]["props"]["basis"] == "file-change"
    assert any(e["source"] == ids.commit(index.selection.anchor) and e["target"] == affects[0]["source"] and e["relation"] == "TOUCHES" for e in data["edges"])
    assert not any(e["source"] == ids.commit(index.selection.anchor) and e["target"] == module for e in data["edges"])
    repo.history.close()
