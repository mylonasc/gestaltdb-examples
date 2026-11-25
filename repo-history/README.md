# Repository-history indexer

Temporal repository-history example for
[GestaltDB](https://github.com/mylonasc/gestaltdb). The demonstrator indexes a
pinned, bounded slice of [`mylonasc/tf_gnns`](https://github.com/mylonasc/tf_gnns)
and keeps source evidence, normalized history, and derived analysis distinct.

The epic specification is
[`../EPIC_01_REPO_HISTORY_EXAMPLE.md`](../EPIC_01_REPO_HISTORY_EXAMPLE.md).

## Development

```bash
cd repo-history
uv sync --extra test
uv run pytest
```

Importing the package does not require provider credentials. Live GitHub and LLM
runs are opt-in. See the epic document for temporal semantics, budgets, safety,
and coverage limitations.

## Content extraction

`ContentExtractor` reads blobs through the Git index instead of the working
tree, so baseline snapshots describe the historical commit rather than `HEAD`:

```python
from repo_history.content import ContentExtractor

index = history.index(selection)
extractor = ContentExtractor.from_history(history)
content_index = extractor.extract(index)
```

Parsing is keyed by `(blob oid, media type)`, so identical bytes are parsed once
while every snapshot keeps its own `snapshot_id`, commit, path, and source URL.
Markdown headings, links, fenced code blocks, and sections carry character
offsets and line ranges into the original blob; Python files are inspected with
`ast` only and are never imported or executed.

Statuses are explicit: `parsed`, `empty`, `unsupported` (binary or Git LFS
pointer), `oversized`, `undecodable`, `gitlink`, and `unavailable`. Each
non-parsed status emits a `CoverageGap`, as do truncated section, symbol, and
import lists, so configured limits stay observable in the output.
