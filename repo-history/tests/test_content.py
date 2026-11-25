"""Bounded Markdown and code-content extraction.

The fixtures are synthetic ``pygit2`` repositories. Markdown is parsed with
``markdown-it-py`` and Python structure with :mod:`ast`; nothing is imported or
executed.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from repo_history.budgets import BudgetExceeded, BudgetLedger
from repo_history.content import (
    ContentExtractor,
    PARSER_VERSION,
    detect_language,
    media_type_for,
    module_name,
    summarize_content,
)
from repo_history.git import GitIndex
from repo_history.models import Budgets, ContentSettings, GitSettings
from test_git import DAY_ONE, DAY_TWO, OUTSIDE, REPOSITORY_URL, WINDOW, GitFixture

UTC = timezone.utc

MARKDOWN = (
    "# Title\n"
    "\n"
    "Intro [docs](https://example.test/docs) and ![logo](images/logo.png).\n"
    "\n"
    "## Usage\n"
    "\n"
    "```python\n"
    'print("hi")\n'
    "```\n"
    "\n"
    "## Usage\n"
    "\n"
    "Tail.\n"
)

PYTHON = (
    '"""Module doc."""\n'
    "import os\n"
    "from typing import Any as A, Optional\n"
    "\n"
    "@decorator\n"
    "class Thing(Base):\n"
    '    """Thing doc."""\n'
    "\n"
    "    @classmethod\n"
    '    def make(cls) -> "Thing":\n'
    "        ...\n"
    "\n"
    "    async def run(self):\n"
    "        def inner():\n"
    "            ...\n"
)


@pytest.fixture()
def fixture(tmp_path) -> GitFixture:
    return GitFixture(tmp_path / "repository")


def _index(fixture: GitFixture, anchor: str, *, git: GitSettings | None = None) -> GitIndex:
    selection = fixture.select(anchor)
    return fixture.history.index(selection, settings=git or GitSettings())


def _extract(
    fixture: GitFixture,
    index: GitIndex,
    *,
    settings: ContentSettings | None = None,
    ledger: BudgetLedger | None = None,
):
    extractor = ContentExtractor.from_history(fixture.history, settings=settings)
    return extractor, extractor.extract(index, ledger=ledger)


def _snapshot(index, path: str, *, baseline: bool = False):
    return next(
        item
        for item in index.snapshots
        if item.path == path and item.is_baseline is baseline
    )


def test_markdown_headings_links_code_and_sections_are_source_addressable(
    fixture: GitFixture,
):
    commit = fixture.commit(
        "docs",
        {"README.md": (MARKDOWN, 0o100644)},
        at=DAY_ONE,
    )
    _, content_index = _extract(fixture, _index(fixture, commit))
    snapshot = _snapshot(content_index, "README.md")
    content = content_index.content_for(snapshot.snapshot_id)

    assert content is not None
    assert content.status == "parsed"
    assert content.media_type == "markdown"
    assert content.language == "markdown"
    assert content.parser_version == PARSER_VERSION
    assert content.line_count == 13

    markdown = content.markdown
    assert markdown is not None
    assert [heading.level for heading in markdown.headings] == [1, 2, 2]
    assert [heading.text for heading in markdown.headings] == [
        "Title",
        "Usage",
        "Usage",
    ]
    assert [heading.slug for heading in markdown.headings] == [
        "title",
        "usage",
        "usage-1",
    ]
    assert [heading.line for heading in markdown.headings] == [1, 5, 11]

    for heading in markdown.headings:
        source = MARKDOWN[heading.offset : heading.end_offset]
        assert source.startswith("#" * heading.level)
        assert heading.text in source

    assert [(link.kind, link.text, link.target) for link in markdown.links] == [
        ("link", "docs", "https://example.test/docs"),
        ("image", "logo", "images/logo.png"),
    ]
    for link in markdown.links:
        marker = "![" if link.kind == "image" else "["
        assert MARKDOWN[link.offset : link.end_offset].startswith(
            f"{marker}{link.text}]"
        )

    assert len(markdown.code_blocks) == 1
    block = markdown.code_blocks[0]
    assert block.language == "python"
    assert block.content == 'print("hi")\n'
    assert (block.line, block.end_line) == (7, 9)
    assert MARKDOWN[block.offset : block.end_offset].startswith("```python")

    assert [(section.ordinal, section.heading) for section in markdown.sections] == [
        (0, "Title"),
        (1, "Usage"),
        (2, "Usage"),
    ]
    assert [(section.start_line, section.end_line) for section in markdown.sections] == [
        (1, 4),
        (5, 10),
        (11, 13),
    ]
    first = markdown.sections[0]
    assert MARKDOWN[first.start_offset : first.end_offset].startswith("# Title")
    assert "Intro" in MARKDOWN[first.start_offset : first.end_offset]

    assert [section.source_url for section in content_index.sections] == [
        f"{REPOSITORY_URL}/blob/{commit}/README.md#L1-L4",
        f"{REPOSITORY_URL}/blob/{commit}/README.md#L5-L10",
        f"{REPOSITORY_URL}/blob/{commit}/README.md#L11-L13",
    ]
    assert all(section.commit_sha == commit for section in content_index.sections)


def test_markdown_without_headings_keeps_one_source_addressable_section(
    fixture: GitFixture,
):
    body = "Just prose.\n\nNo headings here.\n"
    commit = fixture.commit(
        "prose",
        {"NOTES.md": (body, 0o100644)},
        at=DAY_ONE,
    )
    _, content_index = _extract(fixture, _index(fixture, commit))
    snapshot = _snapshot(content_index, "NOTES.md")
    content = content_index.content_for(snapshot.snapshot_id)

    assert content is not None and content.markdown is not None
    assert content.markdown.headings == ()
    assert len(content.markdown.sections) == 1
    section = content.markdown.sections[0]
    assert section.ordinal == 0
    assert section.heading is None
    assert (section.start_offset, section.end_offset) == (0, len(body))
    assert content_index.sections[0].source_url == (
        f"{REPOSITORY_URL}/blob/{commit}/NOTES.md#L1-L3"
    )


def test_python_ast_symbols_imports_and_module_features(fixture: GitFixture):
    commit = fixture.commit(
        "code",
        {"src": {"pkg": {"mod.py": (PYTHON, 0o100644)}}},
        at=DAY_ONE,
    )
    _, content_index = _extract(fixture, _index(fixture, commit))
    snapshot = _snapshot(content_index, "src/pkg/mod.py")
    content = content_index.content_for(snapshot.snapshot_id)

    assert snapshot.module == "pkg.mod"
    assert snapshot.language == "python"
    assert content is not None
    assert content.media_type == "python"
    assert content.code is not None

    code = content.code
    assert code.language == "python"
    assert code.docstring == "Module doc."
    assert code.syntax_error is None

    assert [
        (symbol.kind, symbol.qualname, symbol.line) for symbol in code.symbols
    ] == [
        ("class", "Thing", 6),
        ("method", "Thing.make", 10),
        ("async-method", "Thing.run", 13),
        ("function", "Thing.run.inner", 14),
    ]
    thing = code.symbols[0]
    assert thing.name == "Thing"
    assert thing.decorators == ("decorator",)
    assert thing.bases == ("Base",)
    assert thing.docstring == "Thing doc."
    make = code.symbols[1]
    assert make.decorators == ("classmethod",)
    assert make.docstring is None
    assert all(symbol.symbol_id.startswith("symbol:") for symbol in code.symbols)
    assert len({symbol.symbol_id for symbol in code.symbols}) == 4

    assert [
        (record.kind, record.module, record.names, record.level)
        for record in code.imports
    ] == [
        ("import", None, ("os",), 0),
        ("from", "typing", ("Any as A", "Optional"), 0),
    ]


def test_python_syntax_errors_are_reported_without_execution(fixture: GitFixture):
    broken = "def broken(:\n    pass\n"
    commit = fixture.commit(
        "broken",
        {"broken.py": (broken, 0o100644)},
        at=DAY_ONE,
    )
    _, content_index = _extract(fixture, _index(fixture, commit))
    snapshot = _snapshot(content_index, "broken.py")
    content = content_index.content_for(snapshot.snapshot_id)

    assert content is not None and content.code is not None
    assert content.status == "parsed"
    assert content.code.syntax_error is not None
    assert content.code.symbols == ()
    assert any(gap.code == "content-parse-error" for gap in content_index.gaps)


def test_identical_blobs_are_parsed_once_but_keep_snapshot_provenance(
    fixture: GitFixture,
):
    body = "# Shared\n\nsame bytes\n"
    commit = fixture.commit(
        "shared",
        {
            "a.md": (body, 0o100644),
            "b.md": (body, 0o100644),
            "c.py": ("x = 1\n", 0o100644),
            "d.txt": ("x = 1\n", 0o100644),
        },
        at=DAY_ONE,
    )
    _, content_index = _extract(fixture, _index(fixture, commit))

    first = _snapshot(content_index, "a.md")
    second = _snapshot(content_index, "b.md")
    assert first.oid == second.oid
    assert first.content_id == second.content_id
    assert first.snapshot_id != second.snapshot_id

    # Same bytes under a different media type parse independently.
    python = _snapshot(content_index, "c.py")
    text = _snapshot(content_index, "d.txt")
    assert python.oid == text.oid
    assert python.content_id != text.content_id
    assert python.media_type == "python"
    assert text.media_type == "text"

    assert len(content_index.snapshots) == 4
    assert content_index.counters["unique_contents"] == 3
    assert content_index.counters["markdown_documents"] == 1
    assert content_index.counters["python_documents"] == 1


def test_binary_oversized_undecodable_and_unavailable_statuses(fixture: GitFixture):
    big = "# Big\n\n" + ("x" * 2_000) + "\n"
    commit = fixture.commit(
        "statuses",
        {
            "logo.png": (b"\x89PNG\r\n\x1a\n\x00\x00", 0o100644),
            "big.md": (big, 0o100644),
            "bytes.txt": (b"\xff\xfe\x80\x81", 0o100644),
        },
        at=DAY_ONE,
    )
    index = _index(fixture, commit)
    _, content_index = _extract(
        fixture, index, settings=ContentSettings(max_file_bytes=1_024)
    )

    binary = _snapshot(content_index, "logo.png")
    assert binary.status == "unsupported"
    assert binary.media_type == "binary"
    binary_content = content_index.content_for(binary.snapshot_id)
    assert binary_content is not None
    assert binary_content.status == "unsupported"

    oversized = _snapshot(content_index, "big.md")
    assert oversized.status == "oversized"
    oversized_content = content_index.content_for(oversized.snapshot_id)
    assert oversized_content is not None
    assert oversized_content.byte_size == len(big)
    assert oversized_content.markdown is None
    assert "1024" in (oversized_content.error or "")

    undecodable = _snapshot(content_index, "bytes.txt")
    assert undecodable.status == "undecodable"
    undecodable_content = content_index.content_for(undecodable.snapshot_id)
    assert undecodable_content is not None
    assert undecodable_content.text_bytes == 0

    gap_codes = {gap.code for gap in content_index.gaps}
    assert {
        "content-unsupported",
        "content-oversized",
        "content-undecodable",
    } <= gap_codes
    assert content_index.counters["oversized"] == 1
    assert content_index.counters["undecodable"] == 1
    assert content_index.counters["unsupported"] == 1

    # Force an entry to look unavailable without deleting the underlying object.
    patched = index.model_copy(
        update={
            "snapshots": tuple(
                entry.model_copy(update={"content_status": "unavailable"})
                if entry.path == "big.md"
                else entry
                for entry in index.snapshots
            )
        }
    )
    _, unavailable_index = _extract(
        fixture, patched, settings=ContentSettings(max_file_bytes=1_024)
    )
    unavailable = _snapshot(unavailable_index, "big.md")
    assert unavailable.status == "unavailable"
    assert unavailable.content_id is not None
    assert any(gap.code == "content-unavailable" for gap in unavailable_index.gaps)
    assert any(
        gap.retryable
        for gap in unavailable_index.gaps
        if gap.code == "content-unavailable"
    )


def test_lfs_pointers_are_explicitly_unsupported(fixture: GitFixture):
    pointer = (
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\n"
        "size 12345\n"
    )
    commit = fixture.commit(
        "lfs",
        {"weights.bin": (pointer, 0o100644)},
        at=DAY_ONE,
    )
    _, content_index = _extract(fixture, _index(fixture, commit))
    snapshot = _snapshot(content_index, "weights.bin")
    content = content_index.content_for(snapshot.snapshot_id)

    assert snapshot.status == "unsupported"
    assert content is not None
    assert content.status == "unsupported"
    assert "LFS pointer" in (content.error or "")
    assert any(gap.code == "content-unsupported" for gap in content_index.gaps)


def test_gitlink_entries_have_no_content_and_are_marked(fixture: GitFixture):
    submodule = "1" * 40
    commit = fixture.commit(
        "submodule",
        {"sub": (submodule, 0o160000)},
        at=DAY_ONE,
    )
    _, content_index = _extract(fixture, _index(fixture, commit))
    snapshot = _snapshot(content_index, "sub")

    assert snapshot.status == "gitlink"
    assert snapshot.oid is None
    assert snapshot.content_id is None
    assert content_index.contents == ()
    assert content_index.counters["gitlinks"] == 1


def test_baseline_sections_come_from_the_historical_commit_not_head(
    fixture: GitFixture,
):
    before = fixture.commit(
        "before",
        {"README.md": ("# Old\n\nold body\n", 0o100644)},
        at=OUTSIDE,
    )
    inside = fixture.commit(
        "inside",
        {"README.md": ("# New\n\nnew body\n", 0o100644)},
        parents=(before,),
        at=DAY_ONE,
    )
    _, content_index = _extract(fixture, _index(fixture, inside))

    baseline = _snapshot(content_index, "README.md", baseline=True)
    selected = _snapshot(content_index, "README.md", baseline=False)
    assert baseline.commit_sha == before
    assert selected.commit_sha == inside
    assert baseline.content_id != selected.content_id

    baseline_sections = [
        section
        for section in content_index.sections
        if section.commit_sha == before
    ]
    selected_sections = [
        section
        for section in content_index.sections
        if section.commit_sha == inside
    ]
    assert [section.heading for section in baseline_sections] == ["Old"]
    assert [section.heading for section in selected_sections] == ["New"]
    assert baseline_sections[0].is_baseline is True
    assert selected_sections[0].is_baseline is False
    assert f"/blob/{before}/README.md#L1-L3" in baseline_sections[0].source_url
    assert f"/blob/{inside}/README.md#L1-L3" in selected_sections[0].source_url


def test_section_symbol_and_import_limits_are_observable(fixture: GitFixture):
    markdown = "# One\n\na\n\n# Two\n\nb\n\n# Three\n\nc\n"
    python = (
        "import a\n"
        "import b\n"
        "import c\n"
        "\n"
        "def first():\n"
        "    ...\n"
        "\n"
        "def second():\n"
        "    ...\n"
        "\n"
        "def third():\n"
        "    ...\n"
    )
    commit = fixture.commit(
        "limits",
        {"many.md": (markdown, 0o100644), "many.py": (python, 0o100644)},
        at=DAY_ONE,
    )
    settings = ContentSettings(max_sections=2, max_symbols=2, max_imports=2)
    _, content_index = _extract(fixture, _index(fixture, commit), settings=settings)

    markdown_content = content_index.content_for(
        _snapshot(content_index, "many.md").snapshot_id
    )
    assert markdown_content is not None and markdown_content.markdown is not None
    assert len(markdown_content.markdown.headings) == 2
    assert len(markdown_content.markdown.sections) == 2
    assert [
        (section.heading, section.start_line, section.end_line)
        for section in markdown_content.markdown.sections
    ] == [("One", 1, 4), ("Two", 5, 8)]
    assert markdown_content.markdown.sections_truncated is True

    python_content = content_index.content_for(
        _snapshot(content_index, "many.py").snapshot_id
    )
    assert python_content is not None and python_content.code is not None
    assert len(python_content.code.symbols) == 2
    assert python_content.code.symbols_truncated is True
    assert len(python_content.code.imports) == 2
    assert python_content.code.imports_truncated is True

    codes = {gap.code for gap in content_index.gaps}
    assert {
        "content-sections-truncated",
        "content-symbols-truncated",
        "content-imports-truncated",
    } <= codes


def test_extraction_is_deterministic(fixture: GitFixture):
    commit = fixture.commit(
        "docs",
        {"README.md": (MARKDOWN, 0o100644), "mod.py": (PYTHON, 0o100644)},
        at=DAY_ONE,
    )
    index = _index(fixture, commit)
    _, first = _extract(fixture, index)
    _, second = _extract(fixture, index)

    assert first.model_dump() == second.model_dump()
    assert summarize_content(first) == summarize_content(second)
    assert summarize_content(first)["contents"] == len(first.contents)


def test_budget_charges_each_unique_blob_once(fixture: GitFixture):
    shared = "x" * 600
    commit = fixture.commit(
        "shared",
        {
            "a.md": (shared, 0o100644),
            "b.md": (shared, 0o100644),
            "c.txt": ("y" * 300, 0o100644),
        },
        at=DAY_ONE,
    )
    index = _index(fixture, commit)
    ledger = BudgetLedger(Budgets(max_text_bytes=1_024))
    _extract(fixture, index, ledger=ledger)

    # 600 + 600 + 300 would exceed the budget; deduplication charges 900.
    assert ledger.text_bytes == 900
    assert ledger.counters()["text_bytes"] == 900


def test_budget_exhaustion_is_fail_closed(fixture: GitFixture):
    commit = fixture.commit(
        "large",
        {"large.md": ("x" * 2_000, 0o100644)},
        at=DAY_ONE,
    )
    ledger = BudgetLedger(Budgets(max_text_bytes=1_024))
    with pytest.raises(BudgetExceeded) as error:
        _extract(fixture, _index(fixture, commit), ledger=ledger)

    assert error.value.dimension == "text_bytes"
    assert "text_bytes" in ledger.exhausted


def test_empty_and_missing_blobs(fixture: GitFixture):
    commit = fixture.commit(
        "empty",
        {"empty.md": ("", 0o100644)},
        at=DAY_ONE,
    )
    extractor, content_index = _extract(fixture, _index(fixture, commit))
    snapshot = _snapshot(content_index, "empty.md")
    content = content_index.content_for(snapshot.snapshot_id)
    assert content is not None
    assert content.status == "empty"
    assert content.markdown is None

    missing = extractor.parse("0" * 40, path="missing.md")
    assert missing.status == "unavailable"
    assert missing.content_id.startswith("content:")
    assert missing.error is not None


@pytest.mark.parametrize(
    "path,expected",
    [
        ("README.md", "markdown"),
        ("docs/guide.markdown", "markdown"),
        ("src/mod.py", "python"),
        ("config.toml", "toml"),
        ("Makefile", "make"),
        ("Dockerfile", "dockerfile"),
        ("image.png", None),
        ("LICENSE", "text"),
    ],
)
def test_language_detection(path: str, expected: str | None):
    assert detect_language(path) == expected


def test_shebang_language_detection_and_media_type():
    text = "#!/usr/bin/env python3\nprint('x')\n"
    assert detect_language("script", text) == "python"
    assert media_type_for("script", text) == "python"
    assert media_type_for("image.png") == "binary"
    assert media_type_for("notes") == "unknown"
    assert media_type_for("notes", "plain text\n") == "text"


@pytest.mark.parametrize(
    "path,expected",
    [
        ("src/pkg/mod.py", "pkg.mod"),
        ("src/pkg/__init__.py", "pkg"),
        ("pkg/mod.pyi", "pkg.mod"),
        ("README.md", None),
        ("src/__init__.py", None),
    ],
)
def test_module_name_derivation(path: str, expected: str | None):
    assert module_name(path) == expected


def test_time_window_is_not_consulted_by_content(fixture: GitFixture):
    """Content extraction only consumes an existing Git index."""

    commit = fixture.commit(
        "docs",
        {"README.md": (MARKDOWN, 0o100644)},
        at=datetime(2026, 9, 14, tzinfo=UTC),
    )
    index = _index(fixture, commit)
    assert index.selection.window == WINDOW
    _, content_index = _extract(fixture, index)
    assert content_index.counters["snapshot_entries"] == len(index.snapshots)
    assert content_index.counters["baseline_entries"] == 0
