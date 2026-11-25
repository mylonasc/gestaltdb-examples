"""Bounded Markdown and source-code extraction without executing content.

Parsing is keyed by blob ID, media type, language, adapter version, and policy
so identical content is read and parsed once, while snapshots keep provenance.
Markdown sections, headings, links, and fenced code blocks carry character
offsets into the original text. Language adapters inspect source statically;
no project module is imported or executed.
"""

from __future__ import annotations

import hashlib
import io
import re
import tokenize
from bisect import bisect_right
from functools import cached_property
from graphlib import TopologicalSorter
from pathlib import PurePosixPath
from typing import Callable, Literal, Mapping

from markdown_it import MarkdownIt
from pydantic import Field

from .budgets import BudgetLedger
from .code_models import (
    CodeDocument, CodeSymbol, ImportRecord, IncrementalSourceCodeParser, SourceCodeParser,
)
from .git import GitHistory, GitIndex, TreeEntryRecord
from .ids import IdFactory, stable_id
from .models import (
    ContentSettings,
    CoverageGap,
    CoverageStatus,
    FrozenModel,
)
from .python_parser import PythonParser
from .python_names import python_module_name

ParseStatus = Literal[
    "parsed",
    "empty",
    "unsupported",
    "oversized",
    "undecodable",
    "gitlink",
    "unavailable",
]
MediaType = Literal["markdown", "python", "code", "text", "binary", "unknown"]
LinkKind = Literal["link", "image"]

PARSER_VERSION = "content-v3"

EXTENSION_LANGUAGES: dict[str, str] = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".mdown": "markdown",
    ".rst": "rst",
    ".py": "python",
    ".pyi": "python",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".jsonl": "json",
    ".ipynb": "json",
    ".ini": "ini",
    ".cfg": "ini",
    ".txt": "text",
    ".csv": "csv",
    ".tsv": "csv",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".fish": "shell",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".kt": "kotlin",
    ".rb": "ruby",
    ".php": "php",
    ".sql": "sql",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "scss",
    ".xml": "xml",
    ".svg": "xml",
    ".proto": "protobuf",
    ".tf": "hcl",
    ".hcl": "hcl",
    ".nix": "nix",
    ".lua": "lua",
    ".r": "r",
    ".jl": "julia",
}

FILENAME_LANGUAGES: dict[str, str] = {
    "Dockerfile": "dockerfile",
    "Makefile": "make",
    "Justfile": "just",
    "CMakeLists.txt": "cmake",
    "uv.lock": "toml",
    "Cargo.lock": "toml",
    "LICENSE": "text",
    "README": "text",
}

BINARY_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".webp",
    ".bmp",
    ".tiff",
    ".pdf",
    ".zip",
    ".gz",
    ".bz2",
    ".xz",
    ".tar",
    ".7z",
    ".woff",
    ".woff2",
    ".ttf",
    ".otf",
    ".eot",
    ".mp3",
    ".mp4",
    ".mov",
    ".avi",
    ".webm",
    ".so",
    ".dylib",
    ".dll",
    ".exe",
    ".bin",
    ".pyc",
    ".pyo",
    ".pkl",
    ".pickle",
    ".parquet",
    ".feather",
    ".arrow",
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".onnx",
    ".h5",
    ".hdf5",
    ".sqlite",
    ".db",
    ".safetensors",
}

SHEBANG_LANGUAGES: tuple[tuple[str, str], ...] = (
    ("python", "python"),
    ("bash", "shell"),
    ("zsh", "shell"),
    ("sh", "shell"),
    ("node", "javascript"),
    ("ruby", "ruby"),
    ("perl", "perl"),
)

_MAX_SYMBOL_TEXT_CHARS = 200


class MarkdownHeading(FrozenModel):
    level: int
    text: str
    slug: str
    line: int
    end_line: int
    offset: int
    end_offset: int


class MarkdownLink(FrozenModel):
    kind: LinkKind
    text: str
    target: str
    line: int
    offset: int
    end_offset: int
    title: str | None = None


class MarkdownCodeBlock(FrozenModel):
    language: str | None
    info: str
    content: str
    line: int
    end_line: int
    offset: int
    end_offset: int
    content_sha256: str


class MarkdownSection(FrozenModel):
    ordinal: int
    level: int
    heading: str | None
    slug: str | None
    start_line: int
    end_line: int
    start_offset: int
    end_offset: int


class MarkdownDocument(FrozenModel):
    headings: tuple[MarkdownHeading, ...] = ()
    links: tuple[MarkdownLink, ...] = ()
    code_blocks: tuple[MarkdownCodeBlock, ...] = ()
    sections: tuple[MarkdownSection, ...] = ()
    sections_truncated: bool = False


class ParsedContent(FrozenModel):
    content_id: str
    blob_id: str
    oid: str
    parser_version: str = PARSER_VERSION
    media_type: MediaType
    status: ParseStatus
    language: str | None = None
    byte_size: int = 0
    text_bytes: int = 0
    line_count: int = 0
    sha256: str | None = None
    markdown: MarkdownDocument | None = None
    code: CodeDocument | None = None
    error: str | None = None
    text: str | None = None
    """Full decoded text for parsed blobs (bounded by ``max_file_bytes``).

    Stored so downstream stages (diff text, file previews) can quote source
    without re-reading blobs; ``None`` for binary, oversized, undecodable,
    or otherwise unparsed content.
    """


class SnapshotContent(FrozenModel):
    snapshot_id: str
    commit_sha: str
    path: str
    oid: str | None
    content_id: str | None
    status: ParseStatus
    media_type: MediaType
    language: str | None
    module: str | None
    is_baseline: bool
    source_url: str


class SectionRecord(FrozenModel):
    section_id: str
    snapshot_id: str
    content_id: str
    commit_sha: str
    path: str
    ordinal: int
    level: int
    heading: str | None
    slug: str | None
    start_line: int
    end_line: int
    start_offset: int
    end_offset: int
    is_baseline: bool
    source_url: str

    @property
    def fragment(self) -> str:
        if self.start_line == self.end_line:
            return f"L{self.start_line}"
        return f"L{self.start_line}-L{self.end_line}"


class ContentIndex(FrozenModel):
    contents: tuple[ParsedContent, ...] = ()
    snapshots: tuple[SnapshotContent, ...] = ()
    sections: tuple[SectionRecord, ...] = ()
    gaps: tuple[CoverageGap, ...] = ()
    counters: dict[str, int] = Field(default_factory=dict)

    @cached_property
    def contents_by_id(self) -> dict[str, ParsedContent]:
        return {item.content_id: item for item in self.contents}

    @cached_property
    def snapshots_by_id(self) -> dict[str, SnapshotContent]:
        return {item.snapshot_id: item for item in self.snapshots}

    @cached_property
    def snapshots_by_path(self) -> dict[tuple[str, str], SnapshotContent]:
        return {(item.commit_sha, item.path): item for item in self.snapshots}

    @cached_property
    def contents_by_oid(self) -> dict[str, ParsedContent]:
        # The same bytes may occur as both source and plain text. Prefer
        # structural metadata for fallback lookups, without reparsing blobs.
        return {
            item.oid: item for item in sorted(
                self.contents, key=lambda item: (item.text is not None, item.code is not None)
            )
        }

    def content_for(self, snapshot_id: str) -> ParsedContent | None:
        snapshot = self.snapshots_by_id.get(snapshot_id)
        return self.contents_by_id.get(snapshot.content_id) if snapshot else None

    def content_at(self, commit_sha: str | None, path: str) -> ParsedContent | None:
        snapshot = self.snapshots_by_path.get((commit_sha, path))
        return self.contents_by_id.get(snapshot.content_id) if snapshot else None


def detect_language(path: str, text: str | None = None) -> str | None:
    """Detect a language from the path, then from an optional shebang line."""

    pure = PurePosixPath(path)
    language = FILENAME_LANGUAGES.get(pure.name)
    if language is None:
        language = EXTENSION_LANGUAGES.get(pure.suffix.lower())
    if text is not None:
        shebang = _shebang(text)
        if shebang is not None and language in {None, "text"}:
            lowered = shebang.lower()
            for marker, detected in SHEBANG_LANGUAGES:
                if marker in lowered:
                    language = detected
                    break
    return language


def module_name(path: str) -> str | None:
    """Derive a dotted Python module name without importing anything."""
    return python_module_name(path)


def media_type_for(path: str, text: str | None = None) -> MediaType:
    pure = PurePosixPath(path)
    suffix = pure.suffix.lower()
    if suffix in BINARY_EXTENSIONS:
        return "binary"
    language = detect_language(path, text)
    if language == "markdown":
        return "markdown"
    if language == "python":
        return "python"
    if language == "text":
        return "text"
    if language is not None:
        return "code"
    if text is None:
        return "unknown"
    return "text"


class ContentExtractor:
    """Read blobs once, then dispatch source to independently testable adapters."""

    def __init__(
        self,
        read_blob: Callable[[str], bytes | None],
        ids: IdFactory,
        *,
        settings: ContentSettings | None = None,
        code_parsers: Mapping[str, SourceCodeParser] | None = None,
    ) -> None:
        self._read_blob = read_blob
        self.ids = ids
        self.settings = settings or ContentSettings()
        self._policy_key = hashlib.sha256(self.settings.model_dump_json().encode()).hexdigest()[:12]
        self.code_parsers = dict(code_parsers) if code_parsers is not None else {"python": PythonParser()}
        self._markdown = MarkdownIt("commonmark")
        self._raw: dict[str, bytes | None] = {}
        self._parsed: dict[tuple[str, str], ParsedContent] = {}

    @classmethod
    def from_history(
        cls,
        history: GitHistory,
        *,
        settings: ContentSettings | None = None,
        code_parsers: Mapping[str, SourceCodeParser] | None = None,
    ) -> "ContentExtractor":
        return cls(history.read_blob, history.ids, settings=settings, code_parsers=code_parsers)

    # ------------------------------------------------------------------ public
    def parse(self, oid: str, *, path: str = "") -> ParsedContent:
        """Parse one blob; cache by object ID, language, adapter, and policy."""

        data = self._blob(oid)
        if data is None:
            return self._unavailable(oid, path)
        text = _decode(data, python=detect_language(path) == "python")
        media_type = media_type_for(path, text if text is not None else None)
        return self._parse_bytes(oid, data, text, media_type, path)

    def extract(
        self,
        index: GitIndex,
        *,
        ledger: BudgetLedger | None = None,
    ) -> ContentIndex:
        commits = {
            record.sha: record
            for record in (*index.selection.selected_commits, *index.selection.boundary_commits)
        }
        # Git timestamps can be skewed and SHAs are not ordered by ancestry.
        # Parse parents first; serialized output is still sorted by SHA/path.
        ancestry = {
            sha: tuple(parent for parent in commits[sha].parents if parent in commits)
            for sha in sorted(commits)
        }
        ranks = {sha: rank for rank, sha in enumerate(TopologicalSorter(ancestry).static_order())}
        entries = sorted(index.snapshots, key=lambda item: (
            ranks.get(item.commit_sha, len(ranks)), item.commit_sha, item.path,
        ))
        changes = {(change.commit_sha, change.path): change for change in index.changes}
        parsed_snapshots: dict[tuple[str, str], SnapshotContent] = {}
        contents: dict[str, ParsedContent] = {}
        snapshots: list[SnapshotContent] = []
        sections: list[SectionRecord] = []
        gap_keys: set[tuple[str, str]] = set()
        gaps: list[CoverageGap] = []

        def add_gap(
            code: str,
            status: CoverageStatus,
            message: str,
            content_id: str,
            source_ids: tuple[str, ...],
            retryable: bool = False,
        ) -> None:
            key = (code, content_id)
            if key in gap_keys:
                return
            gap_keys.add(key)
            gaps.append(
                CoverageGap(
                    code=code,
                    status=status,
                    message=message,
                    source="content",
                    source_ids=source_ids,
                    retryable=retryable,
                )
            )

        for entry in entries:
            media_type = media_type_for(entry.path)
            module = module_name(entry.path)
            content_id: str | None = None
            status: ParseStatus

            if entry.kind == "gitlink" or entry.content_status == "gitlink":
                status = "gitlink"
            elif entry.kind == "unknown" or entry.content_status == "unavailable":
                status = "unavailable"
                parsed = self._unavailable(entry.object_id, entry.path)
                content_id = parsed.content_id
                contents.setdefault(content_id, parsed)
            elif entry.content_status == "oversized" or (
                entry.size is not None
                and entry.size > self.settings.max_file_bytes
            ):
                status = "oversized"
                content_id = self._content_id(entry.object_id, media_type)
                contents.setdefault(
                    content_id,
                    ParsedContent(
                        content_id=content_id,
                        blob_id=self.ids.blob(entry.object_id),
                        oid=entry.object_id,
                        media_type=media_type,
                        status="oversized",
                        language=detect_language(entry.path),
                        byte_size=entry.size or 0,
                        error=(
                            f"blob exceeds the {self.settings.max_file_bytes}-byte "
                            "content limit"
                        ),
                    ),
                )
            elif entry.is_binary is True:
                status = "unsupported"
                content_id = self._content_id(entry.object_id, "binary")
                contents.setdefault(
                    content_id,
                    ParsedContent(
                        content_id=content_id,
                        blob_id=self.ids.blob(entry.object_id),
                        oid=entry.object_id,
                        media_type="binary",
                        status="unsupported",
                        language=None,
                        byte_size=entry.size or 0,
                        error="binary content is not parsed",
                    ),
                )
            elif entry.is_lfs_pointer is True:
                status = "unsupported"
                content_id = self._content_id(entry.object_id, "lfs")
                contents.setdefault(
                    content_id,
                    ParsedContent(
                        content_id=content_id,
                        blob_id=self.ids.blob(entry.object_id),
                        oid=entry.object_id,
                        media_type="binary",
                        status="unsupported",
                        language=None,
                        byte_size=entry.size or 0,
                        error=(
                            "Git LFS pointer content is not available in this "
                            "repository"
                        ),
                    ),
                )
            else:
                already_loaded = entry.object_id in self._raw
                data = self._blob(entry.object_id)
                if data is None:
                    status = "unavailable"
                    parsed = self._unavailable(entry.object_id, entry.path)
                    content_id = parsed.content_id
                    contents.setdefault(content_id, parsed)
                else:
                    if ledger is not None and not already_loaded:
                        ledger.reserve_text(len(data))
                    text = _decode(data, python=detect_language(entry.path) == "python")
                    media_type = media_type_for(entry.path, text)
                    commit = commits.get(entry.commit_sha)
                    parent_sha = commit.parents[0] if commit and commit.parents else None
                    change = changes.get((entry.commit_sha, entry.path))
                    parent_path = (
                        change.old_path
                        if change and change.status in {"renamed", "copied"} and change.old_path
                        else entry.path
                    )
                    previous = parsed_snapshots.get((parent_sha, parent_path)) if parent_sha is not None else None
                    parsed = self._parse_bytes(
                        entry.object_id, data, text, media_type, entry.path,
                        previous_content_id=previous.content_id if previous else None,
                    )
                    content_id = parsed.content_id
                    status = parsed.status
                    contents.setdefault(content_id, parsed)

            content = contents.get(content_id) if content_id is not None else None
            snapshot = SnapshotContent(
                snapshot_id=entry.snapshot_id,
                commit_sha=entry.commit_sha,
                path=entry.path,
                oid=None if status == "gitlink" else entry.object_id,
                content_id=content_id,
                status=status,
                media_type=content.media_type if content is not None else media_type,
                language=content.language if content is not None else detect_language(entry.path),
                module=module,
                is_baseline=entry.is_baseline,
                source_url=entry.source_url,
            )
            snapshots.append(snapshot)
            parsed_snapshots[entry.commit_sha, entry.path] = snapshot

            if content is not None and content.markdown is not None:
                for section in content.markdown.sections:
                    section_url = entry.source_url
                    if section.start_line == section.end_line:
                        section_url = f"{section_url}#L{section.start_line}"
                    else:
                        section_url = (
                            f"{section_url}#L{section.start_line}-L{section.end_line}"
                        )
                    sections.append(
                        SectionRecord(
                            section_id=self.ids.section(
                                entry.commit_sha, entry.path, section.ordinal
                            ),
                            snapshot_id=entry.snapshot_id,
                            content_id=content.content_id,
                            commit_sha=entry.commit_sha,
                            path=entry.path,
                            ordinal=section.ordinal,
                            level=section.level,
                            heading=section.heading,
                            slug=section.slug,
                            start_line=section.start_line,
                            end_line=section.end_line,
                            start_offset=section.start_offset,
                            end_offset=section.end_offset,
                            is_baseline=entry.is_baseline,
                            source_url=section_url,
                        )
                    )

            if content is not None:
                self._content_gaps(content, add_gap)

        ordered_contents = tuple(
            sorted(contents.values(), key=lambda item: item.content_id)
        )
        ordered_snapshots = tuple(
            sorted(snapshots, key=lambda item: (item.commit_sha, item.path))
        )
        ordered_sections = tuple(
            sorted(sections, key=lambda item: (item.commit_sha, item.path, item.ordinal))
        )
        counters = self._counters(
            entries, ordered_contents, ordered_snapshots, ordered_sections
        )
        return ContentIndex(
            contents=ordered_contents,
            snapshots=ordered_snapshots,
            sections=ordered_sections,
            gaps=tuple(sorted(gaps, key=lambda gap: (gap.code, gap.source_ids))),
            counters=counters,
        )

    # ------------------------------------------------------------------ content
    def _blob(self, oid: str) -> bytes | None:
        if oid not in self._raw:
            self._raw[oid] = self._read_blob(oid)
        return self._raw[oid]

    def _content_id(self, oid: str, kind: str) -> str:
        return self.ids.content(oid, f"{PARSER_VERSION}:{self._policy_key}:{kind}")

    def _unavailable(self, oid: str, path: str) -> ParsedContent:
        media_type = media_type_for(path)
        language = detect_language(path)
        kind = f"unavailable:{media_type}:{language}"
        content_id = self._content_id(oid, kind)
        cached = self._parsed.get((oid, kind))
        if cached is not None:
            return cached
        parsed = ParsedContent(
            content_id=content_id,
            blob_id=self.ids.blob(oid),
            oid=oid,
            media_type=media_type,
            status="unavailable",
            language=language,
            error="blob object is not available in the local object database",
        )
        self._parsed[(oid, kind)] = parsed
        return parsed

    def _parse_bytes(
        self,
        oid: str,
        data: bytes,
        text: str | None,
        media_type: MediaType,
        path: str,
        *,
        previous_content_id: str | None = None,
    ) -> ParsedContent:
        language = detect_language(path, text)
        parser = self.code_parsers.get(language)
        parser_key = f"{parser.name}:{parser.version}" if parser else (language or media_type)
        key = (oid, f"{media_type}:{parser_key}")
        cached = self._parsed.get(key)
        if cached is not None:
            return cached

        base = ParsedContent(
            content_id=self._content_id(oid, key[1]),
            blob_id=self.ids.blob(oid),
            oid=oid,
            media_type=media_type,
            status="parsed",
            language=language,
            byte_size=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
        )

        if len(data) > self.settings.max_file_bytes:
            parsed = base.model_copy(
                update={
                    "status": "oversized",
                    "error": (
                        f"blob exceeds the {self.settings.max_file_bytes}-byte "
                        "content limit"
                    ),
                }
            )
        elif media_type == "binary" or b"\x00" in data[:8000]:
            parsed = base.model_copy(
                update={
                    "status": "unsupported",
                    "media_type": "binary",
                    "language": None,
                    "error": "binary content is not parsed",
                }
            )
        elif text is None:
            parsed = base.model_copy(
                update={
                    "status": "undecodable",
                    "error": "content cannot be decoded using its declared encoding"
                    if media_type == "python" else "content is not valid UTF-8 text",
                }
            )
        elif text == "":
            parsed = base.model_copy(update={"status": "empty", "text": ""})
        elif media_type == "markdown":
            parsed = base.model_copy(
                update={
                    "text_bytes": len(data),
                    "line_count": _line_count(text),
                    "markdown": self._parse_markdown(text),
                    "text": text,
                }
            )
        elif media_type in {"python", "code"}:
            code = self._parse_code(
                base.content_id, text, language or "text",
                previous_content_id=previous_content_id,
            )
            parsed = base.model_copy(
                update={
                    "text_bytes": len(data),
                    "line_count": _line_count(text),
                    "code": code,
                    "text": text,
                }
            )
        else:
            parsed = base.model_copy(
                update={
                    "text_bytes": len(data),
                    "line_count": _line_count(text),
                    "text": text,
                }
            )

        self._parsed[key] = parsed
        return parsed

    # ------------------------------------------------------------------ markdown
    def _parse_markdown(self, text: str) -> MarkdownDocument:
        line_starts = _line_starts(text)
        tokens = self._markdown.parse(text)
        headings: list[MarkdownHeading] = []
        links: list[MarkdownLink] = []
        code_blocks: list[MarkdownCodeBlock] = []
        seen_slugs: dict[str, int] = {}

        for position, token in enumerate(tokens):
            if token.type == "heading_open" and token.map is not None:
                inline = tokens[position + 1] if position + 1 < len(tokens) else None
                heading_text = _plain_inline(inline)
                start_line, end_line = token.map
                start_offset = line_starts[start_line]
                end_offset = _line_end_offset(line_starts, text, end_line)
                slug = _unique_slug(heading_text, seen_slugs)
                headings.append(
                    MarkdownHeading(
                        level=int(token.tag[1:]),
                        text=heading_text,
                        slug=slug,
                        line=start_line + 1,
                        end_line=max(start_line + 1, end_line),
                        offset=start_offset,
                        end_offset=end_offset,
                    )
                )
            elif token.type == "fence" and token.map is not None:
                start_line, end_line = token.map
                start_offset = line_starts[start_line]
                end_offset = _line_end_offset(line_starts, text, end_line)
                info = (token.info or "").strip()
                language = info.split()[0] if info else None
                code_blocks.append(
                    MarkdownCodeBlock(
                        language=language,
                        info=info,
                        content=token.content,
                        line=start_line + 1,
                        end_line=max(start_line + 1, end_line),
                        offset=start_offset,
                        end_offset=end_offset,
                        content_sha256=hashlib.sha256(
                            token.content.encode("utf-8")
                        ).hexdigest(),
                    )
                )
            elif token.type == "code_block" and token.map is not None:
                start_line, end_line = token.map
                start_offset = line_starts[start_line]
                end_offset = _line_end_offset(line_starts, text, end_line)
                code_blocks.append(
                    MarkdownCodeBlock(
                        language=None,
                        info="",
                        content=token.content,
                        line=start_line + 1,
                        end_line=max(start_line + 1, end_line),
                        offset=start_offset,
                        end_offset=end_offset,
                        content_sha256=hashlib.sha256(
                            token.content.encode("utf-8")
                        ).hexdigest(),
                    )
                )
            elif token.type == "inline" and token.map is not None:
                links.extend(
                    self._inline_links(
                        token,
                        text,
                        line_starts,
                        block_start=line_starts[token.map[0]],
                        block_end=_line_end_offset(line_starts, text, token.map[1]),
                    )
                )

        kept_headings = headings[: self.settings.max_sections]
        # Build every section first so a truncated tail still ends where the
        # next heading begins rather than at the end of the document.
        sections = _markdown_sections(text, line_starts, headings)
        return MarkdownDocument(
            headings=tuple(kept_headings),
            links=tuple(links),
            code_blocks=tuple(code_blocks),
            sections=tuple(sections[: self.settings.max_sections]),
            sections_truncated=(
                len(headings) > self.settings.max_sections
                or len(sections) > self.settings.max_sections
            ),
        )

    def _inline_links(
        self,
        token: object,
        text: str,
        line_starts: list[int],
        *,
        block_start: int,
        block_end: int,
    ) -> list[MarkdownLink]:
        children = getattr(token, "children", None) or []
        block_text = text[block_start:block_end]
        cursor = 0
        links: list[MarkdownLink] = []
        index = 0
        while index < len(children):
            child = children[index]
            if child.type == "link_open":
                label_parts: list[str] = []
                close = index + 1
                while close < len(children) and children[close].type != "link_close":
                    if children[close].type in {"text", "code_inline"}:
                        label_parts.append(children[close].content)
                    elif children[close].type == "image":
                        label_parts.append(children[close].content)
                    close += 1
                label = "".join(label_parts).strip()
                target = (child.attrs or {}).get("href", "")
                title = (child.attrs or {}).get("title")
                link, cursor = self._locate_link(
                    "link",
                    label,
                    target,
                    title,
                    block_text,
                    block_start,
                    line_starts,
                    cursor,
                )
                links.append(link)
                index = close + 1
                continue
            if child.type == "image":
                label = (child.content or "").strip()
                target = (child.attrs or {}).get("src", "")
                title = (child.attrs or {}).get("title")
                link, cursor = self._locate_link(
                    "image",
                    label,
                    target,
                    title,
                    block_text,
                    block_start,
                    line_starts,
                    cursor,
                )
                links.append(link)
            index += 1
        return links

    @staticmethod
    def _locate_link(
        kind: LinkKind,
        label: str,
        target: str,
        title: str | None,
        block_text: str,
        block_start: int,
        line_starts: list[int],
        cursor: int,
    ) -> tuple[MarkdownLink, int]:
        candidates: list[str] = []
        if label:
            prefix = "![" if kind == "image" else "["
            candidates.append(f"{prefix}{label}]")
        if target:
            candidates.append(target)
        if label:
            candidates.append(label)

        relative = -1
        needle = ""
        for candidate in candidates:
            found = block_text.find(candidate, cursor)
            if found >= 0:
                relative, needle = found, candidate
                break
        if relative < 0:
            relative = min(cursor, max(len(block_text) - 1, 0))
            needle = block_text[relative : relative + 1]
        offset = block_start + relative
        return (
            MarkdownLink(
                kind=kind,
                text=label,
                target=target,
                line=_line_of(line_starts, offset),
                offset=offset,
                end_offset=offset + len(needle),
                title=title,
            ),
            relative + max(len(needle), 1),
        )

    # ------------------------------------------------------------------ code
    def _parse_code(
        self, content_id: str, text: str, language: str,
        *, previous_content_id: str | None = None,
    ) -> CodeDocument:
        parser = self.code_parsers.get(language)
        if parser is None:
            return CodeDocument(language=language, shebang=_shebang(text))
        if previous_content_id is not None and isinstance(parser, IncrementalSourceCodeParser):
            return parser.parse_incremental(
                text, content_id=content_id, settings=self.settings,
                previous_content_id=previous_content_id,
            )
        return parser.parse(text, content_id=content_id, settings=self.settings)

    # ------------------------------------------------------------------ reporting
    def _content_gaps(self, content: ParsedContent, add_gap) -> None:
        if content.status == "oversized":
            add_gap(
                "content-oversized",
                CoverageStatus.PARTIAL,
                content.error or "content exceeds the configured size limit",
                content.content_id,
                (content.oid,),
            )
        elif content.status == "undecodable":
            add_gap(
                "content-undecodable",
                CoverageStatus.UNAVAILABLE,
                content.error or "content is not valid UTF-8",
                content.content_id,
                (content.oid,),
            )
        elif content.status == "unsupported":
            add_gap(
                "content-unsupported",
                CoverageStatus.UNSUPPORTED,
                content.error or "content type is not parsed",
                content.content_id,
                (content.oid,),
            )
        elif content.status == "unavailable":
            add_gap(
                "content-unavailable",
                CoverageStatus.UNAVAILABLE,
                content.error or "content is unavailable",
                content.content_id,
                (content.oid,),
                retryable=True,
            )

        if content.markdown is not None and content.markdown.sections_truncated:
            add_gap(
                "content-sections-truncated",
                CoverageStatus.PARTIAL,
                (
                    f"markdown sections were capped at "
                    f"{self.settings.max_sections}"
                ),
                content.content_id,
                (content.oid,),
            )
        if content.code is not None:
            if content.code.symbols_truncated:
                add_gap(
                    "content-symbols-truncated",
                    CoverageStatus.PARTIAL,
                    f"symbols were capped at {self.settings.max_symbols}",
                    content.content_id,
                    (content.oid,),
                )
            if content.code.imports_truncated:
                add_gap(
                    "content-imports-truncated",
                    CoverageStatus.PARTIAL,
                    f"imports were capped at {self.settings.max_imports}",
                    content.content_id,
                    (content.oid,),
                )
            if content.code.syntax_error:
                add_gap(
                    "content-parse-error",
                    CoverageStatus.PARTIAL,
                    content.code.syntax_error,
                    content.content_id,
                    (content.oid,),
                )

    @staticmethod
    def _counters(
        entries: list[TreeEntryRecord],
        contents: tuple[ParsedContent, ...],
        snapshots: tuple[SnapshotContent, ...],
        sections: tuple[SectionRecord, ...],
    ) -> dict[str, int]:
        statuses = [content.status for content in contents]
        return {
            "snapshot_entries": len(entries),
            "baseline_entries": sum(1 for entry in entries if entry.is_baseline),
            "unique_contents": len(contents),
            "parsed_contents": statuses.count("parsed"),
            "empty_contents": statuses.count("empty"),
            "markdown_documents": sum(
                1 for content in contents if content.markdown is not None
            ),
            "code_documents": sum(
                1 for content in contents if content.code is not None
            ),
            "python_documents": sum(
                1
                for content in contents
                if content.code is not None and content.code.language == "python"
            ),
            "headings": sum(
                len(content.markdown.headings)
                for content in contents
                if content.markdown is not None
            ),
            "links": sum(
                len(content.markdown.links)
                for content in contents
                if content.markdown is not None
            ),
            "code_blocks": sum(
                len(content.markdown.code_blocks)
                for content in contents
                if content.markdown is not None
            ),
            "sections": len(sections),
            "symbols": sum(
                len(content.code.symbols)
                for content in contents
                if content.code is not None
            ),
            "imports": sum(
                len(content.code.imports)
                for content in contents
                if content.code is not None
            ),
            "oversized": statuses.count("oversized"),
            "undecodable": statuses.count("undecodable"),
            "unsupported": statuses.count("unsupported"),
            "unavailable": statuses.count("unavailable"),
            "gitlinks": sum(
                1 for snapshot in snapshots if snapshot.status == "gitlink"
            ),
            "text_bytes": sum(
                content.text_bytes
                for content in contents
                if content.status in {"parsed", "empty"}
            ),
        }


def summarize_content(index: ContentIndex) -> dict[str, object]:
    """Small deterministic summary suitable for manifests and reports."""

    return {
        "counters": index.counters,
        "gaps": sorted({gap.code for gap in index.gaps}),
        "contents": len(index.contents),
        "snapshots": len(index.snapshots),
        "sections": len(index.sections),
    }


# ---------------------------------------------------------------------- helpers
def _line_starts(text: str) -> list[int]:
    starts = [0]
    for match in re.finditer("\n", text):
        starts.append(match.end())
    return starts


def _line_of(line_starts: list[int], offset: int) -> int:
    return max(1, bisect_right(line_starts, offset))


def _line_end_offset(line_starts: list[int], text: str, end_line: int) -> int:
    if end_line < len(line_starts):
        return line_starts[end_line]
    return len(text)


def _line_count(text: str) -> int:
    if not text:
        return 0
    return text.count("\n") + (0 if text.endswith("\n") else 1)


def _decode(data: bytes, *, python: bool = False) -> str | None:
    try:
        encoding = tokenize.detect_encoding(io.BytesIO(data).readline)[0] if python else "utf-8-sig"
        return data.decode(encoding)
    except (UnicodeDecodeError, SyntaxError, LookupError):
        return None


def _shebang(text: str) -> str | None:
    if not text.startswith("#!"):
        return None
    first_line = text.split("\n", 1)[0].strip()
    return _truncate(first_line, _MAX_SYMBOL_TEXT_CHARS)


def _plain_inline(token: object | None) -> str:
    if token is None:
        return ""
    children = getattr(token, "children", None) or []
    parts: list[str] = []
    for child in children:
        if child.type in {"text", "code_inline"}:
            parts.append(child.content)
        elif child.type == "image":
            parts.append(child.content)
        elif child.type in {"softbreak", "hardbreak"}:
            parts.append(" ")
    value = "".join(parts).strip()
    if not value:
        value = (getattr(token, "content", "") or "").strip()
    return value


def _unique_slug(text: str, seen: dict[str, int]) -> str:
    base = re.sub(r"[^\w\s-]", "", text.strip().lower(), flags=re.UNICODE)
    base = re.sub(r"[\s_]+", "-", base).strip("-") or "section"
    count = seen.get(base, 0)
    seen[base] = count + 1
    return base if count == 0 else f"{base}-{count}"


def _markdown_sections(
    text: str, line_starts: list[int], headings: list[MarkdownHeading]
) -> list[MarkdownSection]:
    sections: list[MarkdownSection] = []
    if not headings:
        if text.strip():
            sections.append(
                MarkdownSection(
                    ordinal=0,
                    level=0,
                    heading=None,
                    slug=None,
                    start_line=1,
                    end_line=_line_of(line_starts, max(len(text) - 1, 0)),
                    start_offset=0,
                    end_offset=len(text),
                )
            )
        return sections
    if headings[0].offset > 0 and text[: headings[0].offset].strip():
        end = headings[0].offset
        sections.append(
            MarkdownSection(
                ordinal=0,
                level=0,
                heading=None,
                slug=None,
                start_line=1,
                end_line=_line_of(line_starts, max(0, end - 1)),
                start_offset=0,
                end_offset=end,
            )
        )
    for position, heading in enumerate(headings):
        end = (
            headings[position + 1].offset
            if position + 1 < len(headings)
            else len(text)
        )
        sections.append(
            MarkdownSection(
                ordinal=len(sections),
                level=heading.level,
                heading=heading.text,
                slug=heading.slug,
                start_line=heading.line,
                end_line=_line_of(line_starts, max(heading.offset, end - 1)),
                start_offset=heading.offset,
                end_offset=end,
            )
        )
    return sections


def _truncate(value: str | None, limit: int) -> str | None:
    if value is None:
        return None
    if len(value) <= limit:
        return value
    return value[:limit]
