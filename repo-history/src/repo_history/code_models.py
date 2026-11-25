"""Language-neutral source structure, independent of Git and graph storage."""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from .models import ContentSettings, FrozenModel


class ImportRecord(FrozenModel):
    kind: Literal["import", "from"]
    module: str | None
    names: tuple[str, ...]
    # Parallel to names; names never contain the presentation string " as ".
    aliases: tuple[str | None, ...] = ()
    level: int = 0
    line: int
    scope: str | None = None
    conditional: bool = False


class CodeSymbol(FrozenModel):
    symbol_id: str
    kind: Literal[
        "class", "function", "method", "async-function", "async-method", "field"
    ]
    name: str
    qualname: str
    parent: str | None = None
    line: int
    column: int
    end_line: int
    start_line: int | None = None  # Includes decorators when present.
    decorators: tuple[str, ...] = ()
    bases: tuple[str, ...] = ()
    annotation: str | None = None
    field_scope: Literal["class", "instance"] | None = None
    docstring: str | None = None


class CodeDocument(FrozenModel):
    language: str
    shebang: str | None = None
    docstring: str | None = None
    imports: tuple[ImportRecord, ...] = ()
    symbols: tuple[CodeSymbol, ...] = ()
    symbols_truncated: bool = False
    imports_truncated: bool = False
    syntax_error: str | None = None


class SourceCodeParser(Protocol):
    """A parser consumes text, returns structure, and never resolves imports."""

    name: str
    version: str

    def parse(
        self, text: str, *, content_id: str, settings: ContentSettings
    ) -> CodeDocument: ...


@runtime_checkable
class IncrementalSourceCodeParser(SourceCodeParser, Protocol):
    """Optional optimization; results must equal a fresh parse of the text.

    A previous content identity is a hint, not a dependency. Adapters fall
    back to full parsing when their transient tree cache lacks that identity.
    """

    def parse_incremental(
        self, text: str, *, content_id: str, settings: ContentSettings,
        previous_content_id: str,
    ) -> CodeDocument: ...
