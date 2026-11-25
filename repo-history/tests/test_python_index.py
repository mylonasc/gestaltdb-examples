"""Parser and repository-resolution regressions, without running project code."""

from __future__ import annotations

from repo_history.content import ContentExtractor
from repo_history.graph import (
    GraphSink, map_code_components, map_diffs, map_git_index, map_imports,
)
from repo_history.ids import IdFactory
from repo_history.models import ContentSettings, GraphSettings
from repo_history.viz_d3 import build_viewer_data
from test_git import DAY_ONE, OUTSIDE, REPOSITORY_URL, GitFixture
from test_viz_d3 import fixture_discovery, NOW


def parsed(text, *, settings=None):
    extractor = ContentExtractor(lambda _: text.encode("utf-8"), IdFactory(REPOSITORY_URL), settings=settings)
    return extractor.parse("oid", path="pkg/mod.py")


def project(tmp_path, before, after=None):
    repo = GitFixture(tmp_path / "repo")
    base = repo.commit("base", before, at=OUTSIDE)
    tip = repo.commit("tip", after or before, parents=(base,), at=DAY_ONE)
    index = repo.history.index(repo.select(tip))
    content = ContentExtractor.from_history(repo.history).extract(index)
    return repo, index, content


def test_tree_sitter_imports_aliases_multiline_relative_and_syntax_recovery():
    code = parsed(
        "from __future__ import annotations\n"
        "import numpy as np, os.path\n"
        "from ..thing import (Thing as T, helper)\n"
        "from . import other\n"
        "from pkg import *\n"
        "def broken(:\n"
    ).code
    assert code.syntax_error
    assert [(r.module, r.names, r.aliases, r.level) for r in code.imports] == [
        ("__future__", ("annotations",), (None,), 0),
        (None, ("numpy", "os.path"), ("np", None), 0),
        ("thing", ("Thing", "helper"), ("T", None), 2),
        (None, ("other",), (None,), 1),
        ("pkg", ("*",), (None,), 0),
    ]


def test_lexical_ownership_fields_control_flow_async_and_decorators():
    code = parsed(
        "if True:\n"
        "    def conditional(): pass\n"
        "class Model:\n"
        "    size: int = 1\n"
        "    left, right = 1, 2\n"
        "    def __init__(this):\n"
        "        this.value: str = 'v'\n"
        "        local = 2\n"
        "    @staticmethod\n"
        "    def helper(self):\n"
        "        self.not_a_field = 1\n"
        "    async def run(self):\n"
        "        if True:\n"
        "            def nested(): pass\n"
    ).code
    syms = {s.qualname: s for s in code.symbols}
    assert syms["conditional"].kind == "function"
    assert syms["Model.size"].field_scope == "class"
    assert syms["Model.size"].annotation == "int"
    assert syms["Model.value"].field_scope == "instance"
    assert syms["Model.value"].parent == "Model"
    assert syms["Model.run"].kind == "async-method"
    assert syms["Model.run.nested"].kind == "function"
    assert syms["Model.run.nested"].parent == "Model.run"
    assert syms["Model.helper"].start_line == 9
    assert "Model.not_a_field" not in syms
    assert not any(s.name == "local" for s in code.symbols)


def test_exact_symbol_limit_does_not_report_false_truncation():
    result = parsed("def f(): pass\nx = 1\n", settings=ContentSettings(max_symbols=1))
    assert len(result.code.symbols) == 1
    assert not result.code.symbols_truncated
    assert parsed("def f(): pass\ndef g(): pass\n", settings=ContentSettings(max_symbols=1)).code.symbols_truncated


def test_cache_is_language_specific_and_python_encoding_cookies_work():
    ids = IdFactory(REPOSITORY_URL)
    extractor = ContentExtractor(lambda _: b"x = 1\n", ids)
    js = extractor.parse("oid", path="x.js")
    go = extractor.parse("oid", path="x.go")
    assert js.content_id != go.content_id
    assert (js.language, go.language) == ("javascript", "go")
    latin = ContentExtractor(lambda _: b"# coding: latin-1\nname = 'caf\xe9'\n", ids)
    result = latin.parse("latin", path="x.py")
    assert result.status == "parsed"
    assert "café" in result.text
    assert result.code.syntax_error is None
    assert ContentExtractor(lambda _: b"hello\x00there", ids).parse("bin", path="x").status == "unsupported"
    smaller = ContentExtractor(lambda _: b"x = 1\n", ids, settings=ContentSettings(max_symbols=1))
    assert smaller.parse("oid", path="x.js").content_id != js.content_id


def test_injectable_language_adapter_and_cached_lookup_round_trip():
    class Adapter:
        name, version = "javascript", "test-v1"

        def parse(self, text, *, content_id, settings):
            from repo_history.code_models import CodeDocument
            return CodeDocument(language="javascript", docstring="custom")

    extractor = ContentExtractor(lambda _: b"x", IdFactory(REPOSITORY_URL), code_parsers={"javascript": Adapter()})
    assert extractor.parse("oid", path="x.js").code.docstring == "custom"


def test_components_and_external_libraries_have_correct_edges_and_provenance(tmp_path):
    files = {"src": {"pkg": {
        "__init__.py": ("", 0o100644),
        "model.py": ("class Model:\n    size: int = 1\n    def run(self): pass\ndef helper(): pass\n", 0o100644),
        "use.py": (
            "from .model import Model as M, helper\n"
            "import pkg.model as model\nimport numpy as np\n"
            "from numpy.linalg import norm\nimport os.path\n", 0o100644),
    }}}
    repo, index, content = project(tmp_path, files)
    ids = IdFactory(REPOSITORY_URL)
    components = map_code_components(content, index, ids, run_id="test")
    imports = map_imports(content, index, ids, run_id="test")
    sha = index.selection.anchor
    entities = {e.logical_id: e for e in (*components.entities, *imports.entities)}
    assert {"Module", "Class", "Method", "Function", "Field", "ExternalLibrary"} <= {e.labels[0] for e in entities.values()}
    model_id = ids.symbol("src/pkg/model.py", "class", "Model")
    imports_at_tip = {(r.source, r.target) for r in imports.relationships if r.origin_id.startswith(sha)}
    assert (ids.file("src/pkg/use.py"), model_id) in imports_at_tip
    assert (ids.file("src/pkg/use.py"), ids.symbol("src/pkg/model.py", "function", "helper")) in imports_at_tip
    assert (ids.file("src/pkg/use.py"), ids.module("src/pkg/model.py")) in imports_at_tip
    assert (ids.module("src/pkg/use.py"), ids.library("numpy")) in imports_at_tip
    assert (ids.external_module("numpy.linalg"), ids.library("numpy")) in imports_at_tip
    assert entities[ids.library("numpy")].properties["distribution"] is None
    assert entities[ids.library("os")].properties["dependency_kind"] == "stdlib"
    owns = {(r.source, r.target) for r in components.relationships if r.edge_type == "BELONGS_TO"}
    assert (ids.symbol("src/pkg/model.py", "method", "Model.run"), model_id) in owns
    assert all(r.properties.get("hierarchical") for r in components.relationships)
    assert len({e.event_id for e in components.entities}) == len(components.entities)
    repo.history.close()


def test_imports_resolve_against_one_commit_not_future_files(tmp_path):
    before = {"pkg": {"use.py": ("import pkg.future\n", 0o100644)}}
    after = {"pkg": {"use.py": ("import pkg.future\n", 0o100644), "future.py": ("x = 1\n", 0o100644)}}
    repo, index, content = project(tmp_path, before, after)
    ids = IdFactory(REPOSITORY_URL)
    mapped = map_imports(content, index, ids, run_id="test")
    base = index.selection.boundary_commits[0].sha
    assert not [r for r in mapped.relationships if r.origin_id.startswith(base) and r.edge_type == "IMPORTS"]
    assert any("unresolved-local-import:pkg.future" in gap for gap in mapped.skipped)
    assert not [e for e in mapped.entities if e.labels == ("ExternalLibrary",)]
    repo.history.close()


def test_relative_module_join_beyond_package_and_namespace_packages(tmp_path):
    files = {"src": {"pkg": {
        "sub": {"use.py": ("from ..model import Model\nfrom ... import invalid\n", 0o100644)},
        "model.py": ("class Model: pass\n", 0o100644),
    }}}
    repo, index, content = project(tmp_path, files)
    ids = IdFactory(REPOSITORY_URL)
    mapped = map_imports(content, index, ids, run_id="test")
    assert any(r.target == ids.symbol("src/pkg/model.py", "class", "Model") for r in mapped.relationships)
    assert any("relative-import-beyond-package" in gap for gap in mapped.skipped)
    assert not [e for e in mapped.entities if e.labels == ("ExternalLibrary",)]
    repo.history.close()


def test_explicit_reexports_resolve_but_cycles_and_conditionals_are_not_guessed(tmp_path):
    files = {"pkg": {
        "__init__.py": ("from .model import Model as PublicModel\n", 0o100644),
        "model.py": ("class Model: pass\n", 0o100644),
        "use.py": ("from pkg import PublicModel as M\nfrom .cycle_a import Missing\nfrom .conditional import Maybe\n", 0o100644),
        "cycle_a.py": ("from .cycle_b import Missing\n", 0o100644),
        "cycle_b.py": ("from .cycle_a import Missing\n", 0o100644),
        "conditional.py": ("if True:\n    from .model import Model as Maybe\n", 0o100644),
    }}
    repo, index, content = project(tmp_path, files)
    ids = IdFactory(REPOSITORY_URL)
    imports = map_imports(content, index, ids, run_id="test")
    use = [r for r in imports.relationships if r.source == ids.file("pkg/use.py")]
    assert any(r.target == ids.symbol("pkg/model.py", "class", "Model") for r in use)
    assert any("unresolved-member:pkg.cycle_a.Missing" in gap for gap in imports.skipped)
    assert any("unresolved-member:pkg.conditional.Maybe" in gap for gap in imports.skipped)
    conditional = content.content_at(index.selection.anchor, "pkg/conditional.py")
    assert conditional.code.imports[0].conditional
    repo.history.close()


def test_duplicate_import_sites_and_unknown_members_remain_explicit(tmp_path):
    files = {"pkg": {
        "model.py": ("class Model: pass\n", 0o100644),
        "use.py": ("import numpy as np\ndef run():\n    import numpy\nfrom .model import Missing\n", 0o100644),
    }}
    repo, index, content = project(tmp_path, files)
    ids = IdFactory(REPOSITORY_URL)
    mapped = map_imports(content, index, ids, run_id="test")
    site = next(r for r in mapped.relationships if r.source == ids.file("pkg/use.py") and r.target == ids.external_module("numpy"))
    assert {s["line"] for s in site.properties["sites"]} == {1, 3}
    assert {s["scope"] for s in site.properties["sites"]} == {None, "run"}
    assert any("unresolved-member:pkg.model.Missing" in gap for gap in mapped.skipped)
    assert not any(e.properties.get("name") == "Missing" for e in mapped.entities)
    repo.history.close()


def test_module_with_stub_and_implementation_prefers_implementation(tmp_path):
    files = {
        "pkg.py": ("class Model: pass\n", 0o100644),
        "pkg.pyi": ("class Model: ...\n", 0o100644),
        "use.py": ("from pkg import Model\n", 0o100644),
    }
    repo, index, content = project(tmp_path, files)
    ids = IdFactory(REPOSITORY_URL)
    mapped = map_imports(content, index, ids, run_id="test")
    assert any(r.target == ids.symbol("pkg.py", "class", "Model") for r in mapped.relationships)
    assert not any(r.target == ids.symbol("pkg.pyi", "class", "Model") for r in mapped.relationships)
    repo.history.close()


def test_extensionless_python_scripts_keep_symbols_and_module_provenance(tmp_path):
    files = {"scripts": {"run": ("#!/usr/bin/env python3\ndef main(): pass\n", 0o100755)}}
    repo, index, content = project(tmp_path, files)
    ids = IdFactory(REPOSITORY_URL)
    mapped = map_code_components(content, index, ids, run_id="test")
    assert any(e.logical_id == ids.symbol("scripts/run", "function", "main") for e in mapped.entities)
    module = next(e for e in mapped.entities if e.logical_id == ids.module("scripts/run"))
    assert module.properties["importable"] is False
    repo.history.close()


def test_rename_preserves_paths_and_binary_or_missing_content_is_explicit(tmp_path):
    before = {"old.py": ("def f(): pass\n", 0o100644)}
    after = {"new.py": ("def f(): pass\n", 0o100644)}
    repo, index, content = project(tmp_path, before, after)
    ids = IdFactory(REPOSITORY_URL)
    diffs = map_diffs(content, index, ids, run_id="test")
    assert len(diffs.entities) == 1
    assert diffs.entities[0].properties["old_path"] == "old.py"
    assert diffs.entities[0].properties["path"] == "new.py"
    assert diffs.entities[0].properties["status"] == "renamed"
    assert diffs.entities[0].properties["patch_status"] == "available"
    missing = map_diffs(type(content)(), index, ids, run_id="test")
    assert missing.entities[0].properties["patch_status"] == "content-unavailable"
    assert not [r for r in missing.relationships if r.edge_type == "AFFECTS"]
    binary_index = index.model_copy(update={"changes": tuple(c.model_copy(update={"is_binary": True}) for c in index.changes)})
    binary = map_diffs(content, binary_index, ids, run_id="test")
    assert binary.entities[0].properties["patch_status"] == "binary"
    repo.history.close()


def test_patches_and_symbol_changes_do_not_mark_unchanged_neighbors(tmp_path):
    before = {"model.py": ("def changed():\n    return 1\ndef stable():\n    return 3\n", 0o100644)}
    after = {"model.py": ("def changed():\n    return 2\ndef stable():\n    return 3\n", 0o100644)}
    repo, index, content = project(tmp_path, before, after)
    ids = IdFactory(REPOSITORY_URL)
    mapped = map_diffs(content, index, ids, run_id="test")
    diff = mapped.entities[0]
    assert diff.properties["patch_status"] == "available"
    assert "-    return 1" in diff.properties["text"]
    assert "+    return 2" in diff.properties["text"]
    affected = [r for r in mapped.relationships if r.edge_type == "AFFECTS"]
    assert {r.target for r in affected} == {
        ids.module("model.py"), ids.symbol("model.py", "function", "changed"),
    }
    assert affected[0].properties["status"] == "modified"
    # Lazy indexes must not leak into serialized checkpoint data.
    restored = type(content).model_validate_json(content.model_dump_json())
    assert restored.content_at(index.selection.anchor, "model.py").text == after["model.py"][0]
    repo.history.close()


def test_deletions_and_missing_newline_patch(tmp_path):
    before = {"old.py": ("def gone(): pass", 0o100644), "keep.txt": ("ok", 0o100644)}
    after = {"keep.txt": ("ok\n", 0o100644)}
    repo, index, content = project(tmp_path, before, after)
    ids = IdFactory(REPOSITORY_URL)
    mapped = map_diffs(content, index, ids, run_id="test")
    deleted = next(e for e in mapped.entities if e.properties["path"] == "old.py")
    assert deleted.properties["status"] == "deleted"
    assert "+++ /dev/null" in deleted.properties["text"]
    assert "\\ No newline at end of file" in deleted.properties["text"]
    assert any(r.target == ids.symbol("old.py", "function", "gone") for r in mapped.relationships)
    newline = next(e for e in mapped.entities if e.properties["path"] == "keep.txt")
    assert newline.properties["text"]
    repo.history.close()


def test_persistence_and_viewer_use_the_same_nonduplicated_diff_graph(tmp_path):
    before = {"model.py": ("import numpy\ndef f(): return 1\n", 0o100644)}
    after = {"model.py": ("import numpy\ndef f(): return 2\n", 0o100644)}
    repo, index, content = project(tmp_path, before, after)
    ids = IdFactory(REPOSITORY_URL)
    maps = [mapper(content, index, ids, run_id="test") for mapper in (map_code_components, map_imports, map_diffs)]
    git = map_git_index(index, ids, run_id="test")
    with GraphSink.create(GraphSettings(path=str(tmp_path / "graph")), run_id="test", overwrite=True) as sink:
        entities = [*git.entities, *(e for m in maps for e in m.entities)]
        edges = [*git.relationships, *(r for m in maps for r in m.relationships)]
        known_ids = {e.logical_id for e in entities}
        assert all(r.source in known_ids and r.target in known_ids for m in maps for r in m.relationships)
        sink.write_entities(entities, analysis_run_id="test", defer_indexes=True)
        sink.write_relationships(edges, analysis_run_id="test", defer_indexes=True)
        duplicate = sink.write_entities(entities, analysis_run_id="test", defer_indexes=True)
        assert duplicate.entities_duplicate == len(entities)
        duplicate_edges = sink.write_relationships(edges, analysis_run_id="test", defer_indexes=True)
        assert duplicate_edges.relationships_duplicate == len(edges)
        assert sink.label_counts()["ExternalLibrary"] == 1
        assert sink.edge_counts()["TOUCHES"] == len(index.changes)
    data = build_viewer_data(
        index=index, content=content, discovery=fixture_discovery(), links=[], assignments=[],
        info_horizon=NOW, run_id="test", repository_url=REPOSITORY_URL, max_nodes=500,
        # Deliberately pass diffs too: the export must not double them.
        extra_nodes=[e for m in maps for e in m.entities],
        extra_edges=[r for m in maps for r in m.relationships],
    )
    triples = [(e["source"], e["target"], e["relation"]) for e in data["edges"]]
    assert len(triples) == len(set(triples))
    assert sum(e["relation"] == "TOUCHES" for e in data["edges"]) == len(index.changes)
    assert {"Module", "Function", "ExternalLibrary"} <= {n["kind"] for n in data["nodes"]}
    assert all(n["time"] for n in data["nodes"] if n["kind"] == "Diff")
    latest = next(n for n in data["nodes"] if n["id"] == ids.symbol("model.py", "function", "f"))
    assert latest["props"]["commit_sha"] == index.selection.anchor
    assert "return 2" in latest["props"]["text"]
    repo.history.close()


def test_export_rejects_transitive_parents_and_flattened_touches():
    from test_viz_d3 import fixture_inputs, ids, writes_edge, ROOT, MID, TIP

    data = build_viewer_data(**fixture_inputs(extra_edges=[
        writes_edge(ids().commit(TIP), ids().commit(MID), "PARENT_COMMIT"),
        writes_edge(ids().commit(TIP), ids().commit(ROOT), "PARENT_COMMIT"),
        writes_edge(ids().commit(TIP), ids().file("docs/guide.md"), "TOUCHES"),
    ]))
    assert [(e["source"], e["target"]) for e in data["edges"] if e["relation"] == "PARENT_COMMIT"] == [(ids().commit(TIP), ids().commit(MID))]
    assert not any(e["relation"] == "TOUCHES" for e in data["edges"])
