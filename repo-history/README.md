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
