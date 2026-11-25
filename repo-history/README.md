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

## Setup and privacy

Setup is local-first: the default test suite and the recorded-evidence
fixtures need no secrets and no network.  Two capabilities are explicitly
opt-in and never run by default:

- GitHub access reads the public `tf_gnns` repository through `httpx`
  (`GITHUB_TOKEN` from the configured `token_env`; unauthenticated runs also
  work within rate limits).  Raw responses are cached under `data/cache`
  without credentials; the token never leaves the process or enters a file.
- LLM enrichment requires `llm.enabled = true` **and**
  `llm.privacy_consent = true`, plus the `openai` extra
  (`uv run --with langchain-openai ...`) and `OPENAI_API_KEY`.  Prompts carry
  only repository evidence already selected for indexing, and every call is
  budgeted, cached, and recorded with model/prompt/schema provenance.

Nothing in the pipeline mutates the target repository, opens sockets except
through those two gates, or invents history: unavailable source history is
reported as explicit coverage gaps.  `examples/evaluate.py` reproduces the
bounded live evaluation; pass `--llm` only with consent and a key.

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

A parent lookup that was a 404 online leaves no cache entry, so offline replay
treats the missing entry the same way: no parent edge, no gap. Online and
offline runs therefore agree on events and coverage.

Current issue bodies, labels, and comments are observations stamped at fetch
time; they are never backdated to the item's creation. Creation, comments, and
timeline events keep their source timestamps, while `RevisionAvailability`
records that GitHub REST does not expose earlier body/label revisions. Labels,
native parent/sub-issue edges, PR heads, bases, merge commits, and PR commit
lists are captured when present, and out-of-scope hierarchy edges are flagged
without fetching the external item. Credentials are read from the configured
environment variable and are never written to the cache.

## Temporal replay and resumable orchestration

`TemporalCoordinator` merges Git and GitHub event streams into one
deterministic history: stable ordering by `(event_time, source, source_id,
sequence)`, causal conflict resolution (later observations correct earlier
ones; exact ties break deterministically), and deferral of events whose
dependencies have not arrived yet. `git_selection_events` turns a bounded
`GitSelection` into `BASELINE` events for boundary commits and `COMMIT` events
for selected commits; immutable commits carry `info_horizon == event_time`,
while mutable remote observations carry `info_horizon == observed_at`.

```python
from repo_history.orchestration import (
    EventJournal,
    StageRunner,
    TemporalCoordinator,
    git_selection_events,
)

runner = StageRunner.resume_or_create(
    config.orchestration.journal_file,
    config.orchestration.checkpoint_file,
    manifest,
    config.budgets,
)
snapshot = runner.run_stage("git", lambda: (git_events, git_gaps))
result = TemporalCoordinator(config.repository).collect(snapshot.events)
usable = TemporalCoordinator.visible_at(result.effective, horizon)
```

`EventJournal` persists events and gaps as an append-only JSONL log with a
manifest header (run, repository, config fingerprint); byte-identical
duplicates are skipped and corrections are kept. `CheckpointStore` records
completed stages with the budget ledger counters and a digest binding the
checkpoint to the journal prefix, so retries and resumes continue spending the
same budgets and a diverged journal fails closed. `bootstrap` splits
per-boundary bootstrap state from in-window replay, and `visible_at` gates
enrichment by information horizon so future labels, docs, or async results
cannot seed past decisions. Unavailable history stays a retryable gap; nothing
is invented for it.

## GestaltDB temporal graph persistence

`GraphSink` is the single writer: it owns the database handle, stamps every
version with provenance (`prov_run_id`, origin event, information horizon,
evidence IDs), and records the run in each temporal commit's metadata:

```python
from repo_history.graph import GraphSink, map_discovery, map_git_index

mapped = map_git_index(index, ids, run_id=run_id)
sink = GraphSink.create(config.graph, run_id=run_id, overwrite=True)
sink.write_entities(mapped.entities, analysis_run_id=run_id, defer_indexes=True)
sink.write_relationships(mapped.relationships, analysis_run_id=run_id)
sink.close()
```

`map_git_index` emits `Commit`, `Snapshot`, `File` (with detected language
and media type), and `Branch` entities with `PARENT_COMMIT`, `MEMBER_OF`,
and `OF_FILE` relationships plus `TIP_OF` branch links. `map_diffs` turns
each changed file into a `Diff` entity carrying the change status and a
bounded text preview; commits reach diffs through `TOUCHES` edges and diffs
reach files through `CHANGES` edges, so ancestry is never invented and the
full text stays in parsed content. `map_imports` resolves intra-repository
Python imports (absolute, relative, and `__init__` packages; third-party
imports are reported, never invented) into `IMPORTS` file-to-file edges;
`map_classes` emits `Class` entities with `DEFINED_AT` edges to their files;
`map_folders` emits `Folder` entities, a single `Repository` root, and
immediate-parent `BELONGS_TO` file→folder and folder→folder edges (plus
`HAS_BRANCH` repo→branch edges via `map_repository_branches`). Containment
and ancestry edges (`PARENT_COMMIT`, `MEMBER_OF`, `OF_FILE`, `TOUCHES`,
`CHANGES`, `BELONGS_TO`, `HAS_BRANCH`, `HAS_FOLDER`, `TIP_OF`, `CHILD_OF`,
`DEFINED_AT`) carry `hierarchical: true` provenance for a stronger
configurable pull. `map_discovery` emits `WorkItem` and `ProviderLabel` entities
with `HAS_LABEL`, `CHILD_OF`, and `MERGED_AS` relationships; `map_registry`
emits `Topic` entities and `map_topic_assignments` emits `ABOUT` edges from
deterministic topic assignments (commits, files, and work items). Out-of-scope
parents and hierarchy edges are skipped, never invented.

Version IDs are deterministic UUIDs, so ingesting the same batch twice is
idempotent: the driver rejects the duplicate ID and the sink skips it after
verifying the stored payload. Corrections (`correct_entity`,
`correct_relationship`) append superseding versions; old evidence is retained
and readable via `node_history` and `node_as_of`. The mutable current
projection (`current_node`, `find_nodes`, `neighbors`) is rebuildable by
replaying the journal into a fresh database. Bulk loads use `defer_indexes`;
every read path rebuilds stale indexes first, since the driver refuses
temporal and indexed queries while they are stale. `GraphSettings` (`[graph]`
section) sets the path, backend, overwrite flag, and LMDB map size.

## Topic registry and deterministic labeling

`DeterministicLabeler` assigns topics without a model: deterministic
path/language/heading/reference features describe each subject, a classical
TF-IDF baseline retrieves candidate topics by cosine similarity, and the
outcome is reuse, a genuinely new proposal, an alias suggestion, or an
explicit abstention — each with matched terms as the rationale:

```python
from repo_history.labeling import (
    DeterministicLabeler,
    extract_snapshot_features,
    seed_registry,
)

labeler = DeterministicLabeler(config.labeling)
registry = seed_registry(at=now)
assignment = labeler.assign(
    features, registry, subject_id=subject_id,
    run_id=run_id, info_horizon=horizon,
)
```

Provider labels are preserved verbatim by `preserve_provider_labels` in a
separate assignment namespace; derived topics never absorb them silently.
Assignments are `AnalysisResult` envelopes, so each decision key already binds
subject, method, horizon, and evidence. Growing the registry (registrations,
aliases, merges) never rewrites old assignments: they reference the registry
version they were decided against, merges mark old topics as superseded
instead of deleting them, and re-running against a kept registry snapshot
reproduces the original decisions exactly.

Topic modeling favors prevalent aspects of the architecture over singletons.
`seed_structural_registry` seeds one topic per well-populated top-level
directory (shared vocabulary pulls its files into reuse instead of proposing
one-off names); `seed_commit_type_registry` seeds commit-type topics
(bug-fix, refactoring, feature, documentation, testing, build-ci,
version-release) decided deterministically from message keywords and touched
paths (`classify_commit`, `assign_commit`). `register_prevalent` only
registers a proposal shared by at least `min_subjects` subjects, so one-off
phrases never become topics, and `topic_membership` reports prevalence per
topic. `LabelingSettings` (`[labeling]` section) sets the reuse/alias
thresholds, minimum score, and retrieval limit.

## LangChain structured LLM enrichment

`Enricher` wraps an injected LangChain chat model (`BaseChatModel`) with
opt-in consent, budgets, caching, validation, and full provenance. Proposals
are typed (`LLMTopicProposal`, `LLMSummaryProposal`, `LLMLinkProposal`) and
grounded: reused topics and link targets must come from the supplied
candidates, and invalid output is retried, then abstained — never invented:

```python
from repo_history.enrichment import Enricher, EnrichmentCache, build_chat_model

settings.llm.enabled = True  # plus privacy_consent=True, or it refuses
model = build_chat_model(settings.llm)  # documented OpenAI configuration
enricher = Enricher(
    model, settings.llm, ledger=ledger,
    cache=EnrichmentCache(settings.llm.cache_file), run_id=run_id,
)
result = enricher.enrich_topic(
    subject_id=..., subject_text=..., candidates=[...],
    registry_version=..., info_horizon=...,
)
```

Every call reserves LLM requests/tokens from the shared `BudgetLedger`
before invoking (retries spend too), records estimated and actual usage in
`Provenance`, and caches the validated proposal under a key of model, prompt
version, schema version, method, and context hash — changing any of them
invalidates the entry. Tests use `GenericFakeChatModel` from
`langchain-core`; the real smoke runs `ChatOpenAI` (`gpt-4.1-mini`) via the
`openai` extra with explicit consent. Unavailable history becomes a
`llm-history-unavailable` gap instead of a prompt. `[llm]` settings cover
provider, model, temperature, privacy consent, retry attempts, prompt version,
and cache file.

## Separate worker processes

`WorkerPool` runs work-item analysis in spawned OS processes. Jobs are
durable, versioned envelopes with fully embedded inputs (feature snapshot,
candidates, registry snapshot, horizon), so any worker — now or after a
restart — replays the same job:

```python
from repo_history.worker import JobQueue, WorkerPool, collect_results, make_job_id

queue = JobQueue("data/work")
queue.submit(WorkJob(job_id=make_job_id(...), ...))  # resubmits dedupe
pool = WorkerPool(queue, max_workers=2, task_timeout_seconds=120.0)
pool.run_pending()  # crashed/timed-out jobs stay resumable
results = collect_results(queue)  # merged in (horizon, job_id) order
```

Required enrichment (deterministic labeling) must succeed for a job to
complete; optional enrichment (LLM proposals, rebuilt from embedded settings
and scripted replies) may abstain or fail while the job still completes, and
the two outcomes stay distinguishable in `WorkResult`. Results echo the job's
horizon and record the worker PID; merging sorts by horizon so out-of-order
completion cannot publish future context into an earlier event. Repeats are
byte-identical by `job_id`, and the coordinator charges the shared ledger only
for accepted optional usage.

## Evidence-backed cross-entity linking

`linking.py` resolves textual references (`#123`, `owner/repo#123`, issue/PR
URLs, commit SHAs, known file paths) and provider relationships (merged PRs,
head/base branches) against known entities only. Unresolved numbers, foreign
repositories, unknown or ambiguous SHAs, and deleted branches are reported as
references, never invented into links:

```python
from repo_history.linking import (
    infer_topical_links,
    resolve_provider_links,
    resolve_textual_links,
)

text_report = resolve_textual_links(texts, ids=ids, ...)
provider_report = resolve_provider_links(discovery, index, ids=ids, ...)
topic_report = infer_topical_links(item_topics, file_topics, ids=ids, ...)
```

Relations stay distinct: `MENTIONS` (textual reference), `INTENDS_TO_RESOLVE`
(fix/close/resolve intent, never inferred from `Merge pull request` lines),
`CLOSED_BY` (provider closure: merged PR to merge commit, with squash
evidence as `merge_style: non-head` and no invented ancestry), and
`RELATED_TO` (scored inference: shared derived topics, branch association).
Weak topical overlap abstains with its score instead of linking.
`LinkRecord.to_result()` wraps each link as an analysis envelope, and
`link_edge_writes()` maps accepted links to single-writer relationship
writes. `score_links()` measures precision/recall over annotated fixtures.
`LinkingSettings` (`[linking]` section) sets the inference score floor.

## Inspection queries and visualization

`Inspector` answers the six user-visible questions from adapter outputs —
no database, no network — with horizon and coverage attached to every
answer: commit-range changes, work-item traces with evidence, branch
containment, topic timelines (emergence then recurrence), files and sections
for a work item at a chosen time, and attribution for labels and links:

```python
from repo_history.inspection import Inspector

inspector = Inspector(
    index=index, discovery=discovery, content=content, links=links,
    assignments=assignments, info_horizon=horizon, run_id=run_id,
    repository_url=repo_url,
)
changes = inspector.changes_in_range(from_sha, to_sha)
traces = inspector.work_items_for_commits([sha, ...])
timeline = inspector.topic_timeline(topic_id)
```

`examples/runbook.py` executes the whole pipeline end to end on the live
slice and saves `examples/report.md` plus `examples/slice.svg`. The figure
is deterministic SVG (stdlib only): factual links are solid blue, inferred
topical links are dashed amber, nodes are capped at a fixed budget with
omissions reported, and the footer records run, horizon, counts, and gaps.
`viz.render_slice_svg()` renders any bounded slice the same way.

## Interactive D3 time-biased graph viewer

`examples/build_viewer.py` exports the live slice to `examples/graph.json`
and writes `examples/viewer.html`, a self-contained page (D3 loaded from
CDN, with an offline data-table fallback) with a force-directed layout whose
x-axis is time and whose y-axis follows the entity hierarchy
(Repository → Branch → WorkItem → Commit → Topic → Diff → Folder → File → Class). All edge
kinds are included — `PARENT_COMMIT`, `TIP_OF`, `TOUCHES` (commit→diff),
`CHANGES` (diff→file), `DEFINED_AT` (class→file), `IMPORTS` (intra-repo
Python imports), `CLOSED_BY`, `MENTIONS`, `INTENDS_TO_RESOLVE`,
`RELATED_TO` (provider and inferred), `BELONGS_TO`, `HAS_BRANCH`, and
`ABOUT` (topic assignments). Files connect only to the commits that actually
change them, via the Diff layer (commit→diff→file):

- every timestamped node is normalized to `t` in `[0, 1]` across the graph;
  an anchor force pulls every node toward the mid-time center while a second
  force pulls left/right with strength growing with `|t - 0.5|`, so the plot
  elongates along time as nodes appear.
- a rank force pulls each kind toward its hierarchy tier (configurable
  `hierarchy pull` slider); hierarchical edges (`PARENT_COMMIT`, `TOUCHES`,
  `TIP_OF`, `BELONGS_TO`, …) also pull harder than associative ones via a
  per-link strength multiplier (`hierarchical pull` slider).
- files are colored by detected type (source vs docs vs other, toggleable
  legend) instead of a single file color; folders group files by directory
  under a single repository root; classes appear as named nodes linked to
  their defining files. Selecting a file shows its text preview and
  per-commit change history; selecting any node shows its provenance and
  evidence.
- nodes without timestamps (deleted branches, unobserved tips, the repository
  root) stay centered with no time pull; snapshot occurrences aggregate to
  file nodes and remote tracking refs resolve to PR head branches.
- interactions: drag, zoom/pan, hover tooltips, click-to-focus neighborhoods
  with provenance and evidence, kind/edge-type filters, label and time-color
  toggles (files additionally colorable by detected type), a time-window
  dimmer, text search, pause/reset, and sliders for anchor pull, time spread,
  link distance and pull, hierarchical-link strength, and hierarchy pull.

`viz_d3.build_viewer_data()` bounds the graph (`max_nodes`, core entities
always kept, omissions reported) and `viz_d3.render_viewer_html()` embeds
the payload deterministically.