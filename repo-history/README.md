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

## GitHub work items and evidence

`GitHubDiscoverer` reads only the work items named by
`repository.allowed_work_items` and keeps every raw response it used:

```python
from repo_history.github import (
    EvidenceCache,
    GitHubCredentials,
    GitHubDiscoverer,
)

discovery = GitHubDiscoverer(
    config.repository,
    config.github,
    cache=EvidenceCache(config.cache_dir),
    credentials=GitHubCredentials.from_env(config.github),
).run()
```

Responses are stored with fetch metadata (URL, status, hashes, ETag,
Last-Modified, rate-limit headers, and the next pagination link), keyed by a hash
of the canonical request URL. Re-runs send conditional requests, and an offline
run replays saved evidence without opening a socket. Missing evidence, denied
optional endpoints, exhausted budgets, and pagination truncation become explicit
`CoverageGap`s or `AbsenceRecord`s instead of silently empty results.

Current issue bodies, labels, and comments are observations stamped at fetch
time; they are never backdated to the item's creation. Creation, comments, and
timeline events keep their source timestamps, while `RevisionAvailability`
records that GitHub REST does not expose earlier body/label revisions. Labels,
native parent/sub-issue edges, PR heads, bases, merge commits, and PR commit
lists are captured when present, and out-of-scope hierarchy edges are flagged
without fetching the external item. Credentials are read from the configured
environment variable and are never written to the cache.
