"""True tree-sitter tree reuse, equivalent results, and Git-aware lineage."""

from __future__ import annotations

import pytest

from repo_history.content import ContentExtractor
from repo_history.models import ContentSettings
from repo_history.python_parser import PythonParser
from test_git import DAY_ONE, DAY_TWO, DAY_THREE, OUTSIDE, GitFixture


class RecordingParser(PythonParser):
    def __init__(self):
        super().__init__()
        self.parents = {}

    def parse_incremental(self, text, *, content_id, settings, previous_content_id):
        self.parents[content_id] = previous_content_id
        return super().parse_incremental(
            text, content_id=content_id, settings=settings,
            previous_content_id=previous_content_id,
        )


@pytest.mark.parametrize("before,after", [
    ("import os\nx = 1\n", "import sys\nx = 1\n"),
    ("import os\n", "# inserted\nimport os\nfrom pkg import Thing as T\n"),
    ("import os\nimport sys\n", "import sys\n"),
    ("π = 'café'\nimport os\n", "π = 'cafè'\nimport sys\n"),
    ("import os\r\nx = 1\r\n", "import os\r\nx = 2\r\nimport sys\r\n"),
    ("import os\n", ""),
    ("", "from . import mod\n"),
    ("import os\ndef broken(:\n", "import sys\ndef fixed(): pass\n"),
    ("import os\ndef f(): pass", "import sys\n# middle\ndef g(): return 1\n"),
])
def test_incremental_calls_tree_sitter_with_edited_tree_and_matches_full_parse(before, after):
    parser = PythonParser()
    native = parser._parser
    calls = []

    class ObserveTreeSitter:
        def parse(self, source, old_tree=None):
            calls.append(old_tree)
            return native.parse(source) if old_tree is None else native.parse(source, old_tree)

    parser._parser = ObserveTreeSitter()
    settings = ContentSettings()
    parser.parse(before, content_id="parent", settings=settings)
    parent_tree = parser._trees["parent"].tree
    parent_shape = str(parent_tree.root_node)
    result = parser.parse_incremental(
        after, content_id="child", previous_content_id="parent", settings=settings,
    )
    expected = PythonParser().parse(after, content_id="child", settings=settings)
    assert result == expected
    assert calls[0] is None and calls[1] is not None
    assert calls[1] is not parent_tree
    assert parser.counters == {"full_parses": 1, "incremental_parses": 1, "tree_cache_hits": 0}
    assert str(parent_tree.root_node) == parent_shape
    assert parent_tree.root_node.text == before.encode("utf-8")


def test_forked_children_do_not_mutate_the_shared_parent_tree():
    parser, settings = PythonParser(), ContentSettings()
    parser.parse("import os\ndef f(): return 1\n", content_id="base", settings=settings)
    branches = [
        ("main", "import numpy as np\ndef f(): return 2\n"),
        ("topic", "from . import local\ndef f(): return 3\n"),
    ]
    for content_id, text in branches:
        actual = parser.parse_incremental(
            text, content_id=content_id, settings=settings, previous_content_id="base",
        )
        assert actual == PythonParser().parse(text, content_id=content_id, settings=settings)
    assert parser.counters["incremental_parses"] == 2
    assert parser._trees["base"].tree.root_node.text == b"import os\ndef f(): return 1\n"


def test_missing_parent_tree_falls_back_to_full_parse_and_cached_trees_are_reused():
    parser, settings = PythonParser(), ContentSettings()
    text = "import os\ndef f(): pass\n"
    first = parser.parse_incremental(
        text, content_id="child", settings=settings, previous_content_id="unavailable",
    )
    assert first == parser.parse(text, content_id="child", settings=settings)
    assert parser.counters == {"full_parses": 1, "incremental_parses": 0, "tree_cache_hits": 1}


def test_extraction_uses_actual_first_parent_across_branches_and_merges(tmp_path):
    fixture = GitFixture(tmp_path / "repo")
    base = fixture.commit("base", {"mod.py": ("import os\nx = 0\n", 0o100644)}, at=OUTSIDE)
    main = fixture.commit("main", {"mod.py": ("import numpy\nx = 1\n", 0o100644)}, parents=(base,), at=DAY_ONE)
    topic = fixture.commit(
        "topic", {"mod.py": ("import jax\nx = 2\n", 0o100644)},
        parents=(base,), at=DAY_TWO, ref="refs/heads/topic",
    )
    merge = fixture.commit(
        "merge", {"mod.py": ("import numpy\nimport jax\nx = 3\n", 0o100644)},
        parents=(main, topic), at=DAY_THREE,
    )
    index = fixture.history.index(fixture.select(merge))
    # Intentionally destroy any accidental useful input order.
    index = index.model_copy(update={"snapshots": tuple(reversed(index.snapshots))})
    parser = RecordingParser()
    extractor = ContentExtractor.from_history(fixture.history, code_parsers={"python": parser})
    content = extractor.extract(index)
    cid = lambda sha: content.content_at(sha, "mod.py").content_id
    assert parser.parents[cid(main)] == cid(base)
    assert parser.parents[cid(topic)] == cid(base)
    assert parser.parents[cid(merge)] == cid(main)
    assert parser.counters["full_parses"] == 1
    assert parser.counters["incremental_parses"] == 3
    for parsed in content.contents:
        assert parsed.code == PythonParser().parse(parsed.text, content_id=parsed.content_id, settings=extractor.settings)
    fixture.history.close()


def test_parent_first_order_is_causal_not_timestamp_order(tmp_path):
    fixture = GitFixture(tmp_path / "repo")
    base = fixture.commit("base", {"mod.py": ("import os\n", 0o100644)}, at=DAY_TWO)
    tip = fixture.commit("tip", {"mod.py": ("import sys\n", 0o100644)}, parents=(base,), at=DAY_ONE)
    parser = RecordingParser()
    content = ContentExtractor.from_history(fixture.history, code_parsers={"python": parser}).extract(
        fixture.history.index(fixture.select(tip))
    )
    assert parser.parents[content.content_at(tip, "mod.py").content_id] == content.content_at(base, "mod.py").content_id
    assert parser.counters["incremental_parses"] == 1
    fixture.history.close()


def test_rename_with_an_edit_reuses_the_previous_path_tree(tmp_path):
    fixture = GitFixture(tmp_path / "repo")
    before = "import os\n\ndef f():\n    return 1\n\ndef stable():\n    return 3\n"
    after = before.replace("return 1", "return 2")
    base = fixture.commit("base", {"old.py": (before, 0o100644)}, at=OUTSIDE)
    tip = fixture.commit("tip", {"new.py": (after, 0o100644)}, parents=(base,), at=DAY_ONE)
    index = fixture.history.index(fixture.select(tip))
    assert [c.status for c in index.changes] == ["renamed"]
    parser = RecordingParser()
    content = ContentExtractor.from_history(fixture.history, code_parsers={"python": parser}).extract(index)
    assert parser.parents[content.content_at(tip, "new.py").content_id] == content.content_at(base, "old.py").content_id
    assert parser.counters["incremental_parses"] == 1
    fixture.history.close()


def test_repeated_blobs_and_absent_parent_snapshots_need_no_unsafe_lineage(tmp_path):
    fixture = GitFixture(tmp_path / "repo")
    base = fixture.commit("base", {"mod.py": ("import os\n", 0o100644)}, at=OUTSIDE)
    same = fixture.commit("same", {"mod.py": ("import os\n", 0o100644)}, parents=(base,), at=DAY_ONE)
    tip = fixture.commit("tip", {"mod.py": ("import sys\n", 0o100644)}, parents=(same,), at=DAY_TWO)
    index = fixture.history.index(fixture.select(tip))
    parser = RecordingParser()
    content = ContentExtractor.from_history(fixture.history, code_parsers={"python": parser}).extract(index)
    assert content.content_at(base, "mod.py").content_id == content.content_at(same, "mod.py").content_id
    assert parser.counters == {"full_parses": 1, "incremental_parses": 1, "tree_cache_hits": 0}
    partial = index.model_copy(update={"snapshots": tuple(s for s in index.snapshots if s.commit_sha == tip)})
    fallback = RecordingParser()
    ContentExtractor.from_history(fixture.history, code_parsers={"python": fallback}).extract(partial)
    assert fallback.counters == {"full_parses": 1, "incremental_parses": 0, "tree_cache_hits": 0}
    fixture.history.close()


def test_copy_with_an_edit_reuses_the_source_parent_tree(tmp_path):
    fixture = GitFixture(tmp_path / "repo")
    before = "import os\ndef f(): return 1\n"
    after = before.replace("return 1", "return 2")
    base = fixture.commit("base", {"source.py": (before, 0o100644)}, at=OUTSIDE)
    tip = fixture.commit(
        "copy", {"source.py": (before, 0o100644), "copy.py": (after, 0o100644)},
        parents=(base,), at=DAY_ONE,
    )
    index = fixture.history.index(fixture.select(tip))
    source_oid = next(s.object_id for s in index.snapshots if s.commit_sha == base)
    index = index.model_copy(update={"changes": tuple(
        change.model_copy(update={"status": "copied", "old_path": "source.py", "old_oid": source_oid})
        if change.path == "copy.py" else change for change in index.changes
    )})
    parser = RecordingParser()
    content = ContentExtractor.from_history(fixture.history, code_parsers={"python": parser}).extract(index)
    assert parser.parents[content.content_at(tip, "copy.py").content_id] == content.content_at(base, "source.py").content_id
    assert parser.counters == {"full_parses": 1, "incremental_parses": 1, "tree_cache_hits": 0}
    fixture.history.close()


def test_nonincremental_adapter_remains_supported_for_changed_files(tmp_path):
    from repo_history.code_models import CodeDocument

    class PlainAdapter:
        name, version = "plain-python", "test"

        def parse(self, text, *, content_id, settings):
            return CodeDocument(language="python", docstring=text)

    fixture = GitFixture(tmp_path / "repo")
    base = fixture.commit("base", {"mod.py": ("x = 1\n", 0o100644)}, at=OUTSIDE)
    tip = fixture.commit("tip", {"mod.py": ("x = 2\n", 0o100644)}, parents=(base,), at=DAY_ONE)
    content = ContentExtractor.from_history(fixture.history, code_parsers={"python": PlainAdapter()}).extract(
        fixture.history.index(fixture.select(tip))
    )
    assert content.content_at(tip, "mod.py").code.docstring == "x = 2\n"
    fixture.history.close()
