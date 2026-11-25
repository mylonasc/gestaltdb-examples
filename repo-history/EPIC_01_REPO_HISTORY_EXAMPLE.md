# EPIC-01: Temporal repository-history example

## Objective

Build a runnable, modular Python repository-indexing framework that represents
Git history, documentation, GitHub work items, topics, and their connections in
GestaltDB. Demonstrate it on a small, reproducible slice of
[`mylonasc/tf_gnns`](https://github.com/mylonasc/tf_gnns), not its entire history.

The first release is GitHub-first and includes working LLM enrichment through
LangChain interfaces. Deterministic parsing remains useful without model access.
This epic delivers the parser/indexer foundation and a small inspection example,
not an autonomous repository-management product.

## Tracking

Implementation work belongs in this examples repository, not in `tf_gnns`.
The epic and RH-indexed implementation issues use native GitHub parent/sub-issue
relationships. Shared example/type labels and area labels make them inspectable.

Parent issue: [EPIC-01 / #7](https://github.com/mylonasc/gestaltdb-examples/issues/7).

Filter all work with `example:repo-history`; distinguish `type:epic` and
`type:implementation`, then narrow by the `area:*` labels listed per slice.

| Slice | Deliverable | GitHub issue |
| --- | --- | --- |
| RH-01 | Contracts, configuration, and framework skeleton | [#8](https://github.com/mylonasc/gestaltdb-examples/issues/8) |
| RH-02 | Bounded Git history and file-change parser | [#9](https://github.com/mylonasc/gestaltdb-examples/issues/9) |
| RH-03 | Documentation and code-content extraction | [#10](https://github.com/mylonasc/gestaltdb-examples/issues/10) |
| RH-04 | GitHub work items, PRs, labels, and hierarchy | [#11](https://github.com/mylonasc/gestaltdb-examples/issues/11) |
| RH-05 | Temporal replay and resumable orchestration | [#12](https://github.com/mylonasc/gestaltdb-examples/issues/12) |
| RH-06 | GestaltDB temporal graph persistence | [#13](https://github.com/mylonasc/gestaltdb-examples/issues/13) |
| RH-07 | Topic registry and deterministic/NLP labeling | [#14](https://github.com/mylonasc/gestaltdb-examples/issues/14) |
| RH-08 | LangChain structured LLM enrichment | [#15](https://github.com/mylonasc/gestaltdb-examples/issues/15) |
| RH-09 | Separate issue-analysis worker process | [#16](https://github.com/mylonasc/gestaltdb-examples/issues/16) |
| RH-10 | Evidence-backed cross-entity linking | [#17](https://github.com/mylonasc/gestaltdb-examples/issues/17) |
| RH-11 | Inspection queries and graph visualization | [#18](https://github.com/mylonasc/gestaltdb-examples/issues/18) |
| RH-12 | Reproducible demonstrator and release evaluation | [#19](https://github.com/mylonasc/gestaltdb-examples/issues/19) |

## Demonstrator: a pinned two-day slice

The following selection was checked against GitHub on 2026-11-25:

| Setting | Initial value |
| --- | --- |
| Repository | `mylonasc/tf_gnns` |
| Time window | `[2026-09-13T00:00:00Z, 2026-09-15T00:00:00Z)` |
| Pinned history anchor | `9bfe6c19c02723dbbf101b9e311de684a4728bb1` |
| Selection | All commits reachable from that anchor whose committer time falls in the window; not first-parent-only |
| Expected selected commits | 16, including merge commits and their in-window side-branch commits |
| Initial PR allowlist | [#10](https://github.com/mylonasc/tf_gnns/pull/10), [#11](https://github.com/mylonasc/tf_gnns/pull/11), [#12](https://github.com/mylonasc/tf_gnns/pull/12) |
| Initial themes | Agent documentation, API contracts, benchmarking, and release/version metadata |

The slice includes the documentation/benchmarking changes and the three related
PR merges. Current `main` contains later commits; it is not the analysis anchor.
The two immediate out-of-window parent boundaries are
`bb1ea48ec019e4ea3a3e7c526744f8db7e14e4c1` and
`6c51068dde0196aa00bbb84be4998be1e2b691bf`. Their trees provide baseline context
for their respective histories; their earlier history is not replayed. Do not
assume that all participating branches share one starting snapshot.

No standalone issues, provider labels, or issue hierarchy were found among these
three PRs at planning time. Report these absences honestly. Exercise issue text,
labels, parent/child work items, clock skew, renames, and missing-history cases
with explicitly synthetic offline fixtures; never insert invented work items
into the live repository's history.

### Efficiency and reproducibility controls

- Fetch only required refs/objects using bounded-depth Git-library operations;
  do not default to an unlimited clone or a full-history traversal. Missing
  ancestry must be detectable, with capped deepening or an explicit stop.
- Index only the selected commits, required baseline trees, approved work items,
  and bounded directly related context. Exclude unrelated and post-window work.
- Start with configurable ceilings of 32 selected commits, 25 work items,
  100 GitHub requests, 2 MiB per analyzed text file, and 16 LLM requests with
  aggregate budgets of 64,000 input and 8,000 output tokens. Count retries against
  budgets and report estimated versus provider-reported token usage separately.
- Hitting a limit must stop the relevant work or emit an explicit incomplete
  result; never silently truncate and claim complete indexing.
- Record exact SHAs, selected paths, API response hashes/fetch times, parser/model
  configuration, coverage limitations, and budget usage in a run manifest.
- Cache Git objects, API evidence, parsed content, and model outputs. Replaying
  saved evidence must not require network access. A live GitHub observation is
  not a permanently reproducible historical API snapshot until captured.

## User-visible questions

The example must support these questions through a Python inspection API and
saved query/report examples:

1. What implementation and documentation changed across a commit range?
2. Which work items and PRs explain those changes, and what is the evidence?
3. Which observed branches contain the selected commits?
4. How did a topic emerge or recur during the selected period?
5. Which files and document sections relate to a work item at a chosen time?
6. Why was an entity labeled or connected to another entity?

Answers distinguish provider facts, deterministic extraction, and model/NLP
inference, and include commit-pinned source links where available.

## Domain graph and identity

Use stable, repository-qualified IDs; hash long composite identities rather than
exceeding GestaltDB index-key limits. Preserve original source identifiers in
properties. Keep these entities separate:

| Entity | Required representation |
| --- | --- |
| `Repository` | Provider identity, remotes, run coverage, and boundary metadata |
| `Commit` | SHA, ordered parents, tree, message, author/committer, authored/committed times |
| `Branch` | Namespaced ref, observed tips and reachability, source-backed ref events |
| `Worktree` | Observed local checkout, HEAD and optional branch; not a remote Git entity |
| `File` | Repository/path history and explicit lineage evidence |
| `FileSnapshot` | Immutable occurrence keyed by repository, commit SHA, and path |
| `Blob` | Deduplicated content/hash, size, language, and encoding metadata |
| `FileChange` | Before/after occurrences, status, diff metadata, parent-comparison policy |
| `DocumentSection` | Snapshot-scoped heading/content and source offsets |
| `WorkItem` / `Issue` | GitHub identity, revisions/observations, status, text, events, hierarchy |
| `PullRequest` | GitHub work-item identity plus head/base, included commits, merge information |
| `ProviderLabel` | Provider/repository-scoped original label identity and metadata |
| `Topic` | Stable semantic concept, aliases, description, lifecycle, and provenance |
| `AnalysisRun` | Inputs, information horizon, configuration, software/model versions, costs |

PRs are work items with additional PR semantics, not duplicate issue records.
Native node labels identify entity types. Semantic topics and provider labels
are graph entities; their assignments have independent evidence and history.

Representative relationships:

```text
Repository --HAS_BRANCH--> Branch --TIP_AT / REACHES--> Commit
Commit --PARENT {ordinal}--> Commit
Commit --CONTAINS_SNAPSHOT--> FileSnapshot --CONTENT--> Blob
Commit --HAS_CHANGE--> FileChange --BEFORE / AFTER--> FileSnapshot
File --HAS_SNAPSHOT--> FileSnapshot --HAS_SECTION--> DocumentSection
Worktree --CHECKED_OUT_AT--> Commit
WorkItem --PARENT_OF--> WorkItem
WorkItem --HAS_PROVIDER_LABEL--> ProviderLabel
PullRequest --INCLUDES_COMMIT--> Commit
PullRequest --TARGETS / HAS_HEAD--> Branch
Entity --ABOUT--> Topic
Entity --MENTIONS / INTENDS_TO_RESOLVE / RELATED_TO--> WorkItem
```

Branches and commits have many-to-many reachability, not exclusive ownership.
Keep complete commit trees conceptually distinct from changed-file lists. The
small demo may materialize explicit snapshot associations; deduplicate blobs and
parse identical content once. Tree-backed compression is a later optimization.
File mode belongs to the tree/snapshot occurrence, not solely to a shared blob.

Deletions have no after-snapshot. Root commits compare against the empty tree.
Merge commits preserve every parent; initial change summaries use an explicit
first-parent policy to avoid double counting. Rename/copy detection records its
algorithm and similarity score rather than asserting infallible file identity.
Handle symlinks, submodule gitlinks, binary data, and LFS pointers as typed
metadata; do not follow links or fetch external payloads by default.

Snapshot source URLs use immutable commit SHAs, never a moving branch name.

## Time, historical completeness, and replay

Maintain separate source `event_time`, ingestion `observed_at`, domain-valid
intervals, and GestaltDB system time. Preserve timestamp precision/time zones;
normalize comparison to UTC. A commit has one primary timeline timestamp
(`committed_at`) plus its original `authored_at`. Files and branches expose
multiple temporal events through their edits and relationships, not mutable
arrays of timestamps or dynamically generated node-type labels.

1. Establish the selected history and each required historical baseline tree.
2. Enumerate and parse baseline documentation/files without replaying older work.
3. Merge commit and available GitHub creation/update/timeline events into a
   stable timestamp stream with deterministic source/event-ID tie-breakers.
4. Respect Git causal dependencies when timestamps conflict; report clock skew.
   Structural placeholders may precede enrichment, but future content must not
   seed an earlier topic decision.
5. Enrich each event using only eligible prior context. Persist the applicable
   information horizon so asynchronous workers cannot leak later vocabulary.
6. Reconcile late evidence through new assertions/corrections, preserving the
   original evidence and prior results.

Never initialize old topics from today's README, labels, issue body, or branch
tips. A current-state inventory is allowed only as a separately identified view.
GitHub `updated_at` is not a complete edit history. Where old body versions or
labels are unavailable, store an observed snapshot with unknown earlier
validity; do not backdate its current contents to issue creation. Historical
events that remain retrievable can still be replayed independently.

Git alone does not establish branch creation/deletion times, original branch
ownership of a commit, or historical worktree existence. Current observations,
PR head/base evidence, and ref events have distinct provenance. Missing or
deleted branches do not invalidate preserved commit/PR evidence.

## Architecture and Python interfaces

```text
Git / GitHub adapters -> evidence cache -> normalized event journal
                                              |
                                     temporal coordinator
                                       /             \
                              repo extractors    issue worker process
                                       \             /
                                    structured analysis results
                                              |
                                      single graph writer
                                              |
                                          GestaltDB
                                              |
                                  inspection API / visualization
```

Provide small extensible interfaces: `RepositorySource`, `WorkItemProvider`,
`DocumentParser`, `CodeParser`, `Labeler`, `RelationshipResolver`, and `GraphSink`.
Plugins receive normalized data and return validated results; only the writer
owns database writes. Preserve raw evidence, normalized entities, and derived
analysis independently so models can be changed without reparsing/refetching.

Recommended components:

- `pygit2`/libgit2 for Git objects, refs, diffs, bounded fetching, and worktrees.
- `httpx` for GitHub REST/GraphQL, with explicit capability and coverage reports.
- Pydantic for plugin contracts and structured model output; `markdown-it-py`
  for Markdown; standard-library `ast` for initial Python code structure.
- `scikit-learn` for deterministic classical NLP baselines; additional languages
  and embeddings remain optional plugins.
- `langchain-core` chat-model interfaces and a documented real provider adapter
  such as `langchain-openai`; provider credentials are environment/config inputs.
- Python `multiprocessing` for a separate issue-analysis OS process, and SQLite
  for durable jobs/checkpoints only, not a second entity graph.
- GestaltDB for all indexed entities, temporal relationships, provenance, and
  queryable projections; validate/pin a version supporting the APIs used.

All framework Git/provider operations use Python libraries, not CLI wrappers or
`subprocess`. In particular, do not use GitPython's subprocess-based operations.
Tests create Git fixtures through the same library boundary.

GestaltDB temporal history and mutable current records are separate read layers.
Use native immutable versions for history, explicit property indexes, and a
documented rebuildable current projection if needed. Sourced claim APIs can
represent inferred assertions; claims are not automatically queryable graph
edges. Any claim-to-edge projection must preserve evidence and be rebuildable.

## Labels, LangChain enrichment, and issue processing

### Topic assignment

Preserve provider labels exactly, then add independently attributable semantic
assignments from paths, languages, headings, commit conventions, NLP, and models.
Retrieve candidate topics before deciding to reuse a topic or propose a new one.
Preserve topic aliases and merge decisions without rewriting earlier runs.
Snapshot labels are versioned; today's file topic must not overwrite old ones.

### Required first-release LLM path

Accept an injected LangChain `BaseChatModel`-compatible interface and use
schema-validated structured output (for example `with_structured_output`). Check
model capabilities explicitly. The release must exercise a real configured model
for grounded topic reuse/new-topic decisions, short summaries, and ambiguous
relationship proposals; merely defining an unused interface is insufficient.

Execution is opt-in even though the integration ships in the first release.
An offline fake-model path exercises the same interfaces and validation. Capture
model/provider identity, model parameters, prompt/schema versions, evidence IDs,
topic-registry state, usage, retries, and result hashes. Cache by all relevant
inputs, not text alone. Invalid output, unavailable providers, exhausted budgets,
and abstentions produce explicit statuses rather than fabricated conclusions.

Issue processing runs in a separate worker process with durable jobs, bounded
retries, independently replayable inputs, and validated output envelopes. Its
jobs receive an event-specific topic/context snapshot. Results return to the
coordinator for ordered publication; workers never concurrently mutate the graph.
Track required versus optional enrichment so a failed model cannot silently mark
a required analysis complete.

### Relationship evidence

Distinguish explicit references, provider-reported connections, intended fixes,
actual closure events, and semantic suggestions. `#123` is not proof of a fix;
branch-name similarity is not ownership. PR squash/rebase associations come from
provider evidence rather than assumed ancestry. Repository-qualified references
must not resolve to the wrong repository.

Each assignment/link records method and version, evidence IDs/spans, source URL,
analysis run, score, thresholds, information horizon, and decision status.
Do not call scores probabilities unless calibration has been measured. Prefer
abstention to unsupported links. Inferences must not mutate GitHub labels/issues.

### Safety

Repository text is untrusted data, including embedded agent instructions. Never
execute indexed code, hooks, builds, or document commands. Model workflows have
no tools that mutate repositories or remote work items. Sending content to a
hosted model requires explicit opt-in, documented filtering/redaction, and clear
privacy warnings. Never persist access tokens in manifests or test recordings.

## Indexed implementation slices

Each slice includes focused tests, failure/coverage reporting, and documentation.
Dependencies below are implementation ordering, not historical source events.
All child issues share `example:repo-history` and `type:implementation` labels.

<a id="rh-01"></a>
### RH-01 — Contracts, configuration, and framework skeleton

**Dependencies:** None. **Area:** `area:architecture`.

- Establish the installable example/package layout, plugin protocols, stable IDs,
  validated event/result schemas, run manifests, budgets, and configuration.
- Specify temporal/coverage semantics and GitHub-first provider capabilities;
  validate the required GestaltDB version/backend APIs in a small capability test.
- **Acceptance:** contract/ID tests pass; configuration can express the pinned
  slice and offline mode; unsupported capabilities fail explicitly; no mandatory
  model credentials are needed to import or test the framework.

<a id="rh-02"></a>
### RH-02 — Bounded Git history and file-change parser

**Dependencies:** RH-01. **Area:** `area:git`.

- Implement library-only bounded fetching, pinned selection, causal parents,
  historical baseline trees, refs/worktrees, snapshots/blobs, and change records.
- Preserve modes, root/merge behavior, rename evidence, binary/LFS/gitlink metadata,
  immutable source links, and missing-ancestry diagnostics.
- **Acceptance:** fixtures cover branching, merges, deletes, renames, skew, and
  incomplete history; the live manifest selects the expected 16 commits without
  indexing all `tf_gnns` history; every selected commit resolves its full tree.

<a id="rh-03"></a>
### RH-03 — Documentation and code-content extraction

**Dependencies:** RH-02. **Area:** `area:parsing`.

- Parse baseline and changed Markdown into source-addressable sections; extract
  language/path/module features and initial Python AST structure without execution.
- Deduplicate content parsing while retaining snapshot-specific provenance and
  explicit unsupported/oversized/undecodable-content statuses.
- **Acceptance:** headings, offsets, links, and code/doc features are reproducible;
  baseline context is historical, not HEAD; content limits are observable.

<a id="rh-04"></a>
### RH-04 — GitHub work items, PRs, labels, and hierarchy

**Dependencies:** RH-01. **Area:** `area:github`.

- Add an `httpx` adapter for scoped discovery, issues/comments/timeline evidence,
  original labels, native parent/sub-issues, PR head/base/commits/merge evidence,
  and saved raw responses with fetch metadata.
- Support pagination, conditional requests, rate limiting, transient failures,
  credentials, field permissions, and missing historical revisions explicitly.
- **Acceptance:** offline fixtures cover labels and hierarchy; live ingestion
  captures PRs #10-#12 and honestly reports absent standalone issues/labels.

<a id="rh-05"></a>
### RH-05 — Temporal replay and resumable orchestration

**Dependencies:** RH-02, RH-03, RH-04. **Area:** `area:temporal`.

- Implement the durable evidence/event journal, per-boundary bootstrap, stable
  source-event ordering, causal conflict policy, checkpoints, and late evidence.
- Gate enrichment by information horizon; distinguish unavailable historical
  content from reconstructable events; apply budgets across retries and resumes.
- **Acceptance:** interrupted and repeated replay yields identical logical output;
  skew and late-arrival tests pass; future docs, labels, and asynchronous results
  cannot seed past topic decisions.

<a id="rh-06"></a>
### RH-06 — GestaltDB temporal graph persistence

**Dependencies:** RH-01, RH-05. **Area:** `area:gestaltdb`.

- Implement the single-writer `GraphSink`, entity/relationship mappings, immutable
  temporal writes, evidence/analysis provenance, and explicit property indexes.
- Define historical reads and any rebuildable current or sourced-claim projection;
  preserve snapshot membership, issue hierarchy, and branch reachability semantics.
- **Acceptance:** persisted graphs reopen correctly; duplicate ingestion is
  idempotent; as-of/correction tests retain old evidence; deferred indexes are
  rebuilt before indexed reads; all database handles close safely.

<a id="rh-07"></a>
### RH-07 — Topic registry and deterministic/NLP labeling

**Dependencies:** RH-03, RH-04, RH-05. **Area:** `area:labeling`.

- Implement provider-label preservation, deterministic path/language/heading/
  reference features, a classical-NLP baseline, candidate-topic retrieval, and
  attributed topic reuse/proposal/alias decisions.
- Scope assignments to entity versions and replay horizons; keep uncertain
  mappings and later topic merges distinct from original provider metadata.
- **Acceptance:** fixture examples demonstrate reuse and genuinely new topics;
  assignments are explainable and repeatable without a model; no retroactive
  relabeling occurs when the vocabulary grows.

<a id="rh-08"></a>
### RH-08 — LangChain structured LLM enrichment

**Dependencies:** RH-07. **Area:** `area:llm`.

- Integrate injected LangChain chat models with typed topic/summary/link proposals,
  capability checks, grounded prompts, output validation, abstention, and caching.
- Ship at least one documented real provider configuration and an offline fake;
  enforce privacy opt-in, bounded retries, aggregate token/request budgets, and
  complete model/prompt/schema/context provenance.
- **Acceptance:** fake-model tests cover invalid output and budget failures; an
  explicitly enabled real-model smoke records grounded results and usage; changing
  model, prompt, or topic context invalidates the appropriate cache entries.

<a id="rh-09"></a>
### RH-09 — Separate issue-analysis worker process

**Dependencies:** RH-05, RH-06, RH-08. **Area:** `area:workers`.

- Run work-item labeling/topic analysis in a separate OS process using durable
  jobs, versioned envelopes, bounded retries, and independently replayable inputs.
- Deliver horizon-pinned context; return results to the coordinator/single writer;
  preserve required/optional enrichment and failure/completion distinctions.
- **Acceptance:** worker crash/restart loses no accepted work, repeats create no
  duplicate assertions, and out-of-order completion cannot publish future context
  into an earlier event. Process IDs demonstrate actual process separation.

<a id="rh-10"></a>
### RH-10 — Evidence-backed cross-entity linking

**Dependencies:** RH-02, RH-04, RH-06, RH-07, RH-09. **Area:** `area:linking`.

- Resolve qualified references and provider PR/issue/commit relationships; add
  scored topical/branch/document associations only with supporting evidence.
- Separate mentions, intended resolution, closure, and inference; handle deleted
  branches, multiple issues, and squash/rebase evidence without invented ancestry.
- **Acceptance:** an annotated fixture checks exact explicit links and measured
  inferred-link quality/abstention; every proposed link has inspectable evidence;
  live reports do not invent issue links for the PR-only slice.

<a id="rh-11"></a>
### RH-11 — Inspection queries and graph visualization

**Dependencies:** RH-06, RH-10. **Area:** `area:inspection`.

- Provide Python helpers and saved examples for commit-range/time-window changes,
  work-item traces, branch reachability, topic timelines, and assignment explanations.
- Add a small GestaltDB visualization/notebook using readable names, bounded graph
  sizes, provenance details, and commit-pinned source links.
- **Acceptance:** examples answer all six user-visible questions; historical views
  state their source coverage and information horizon; inferred and factual links
  remain visibly distinguishable. No full custom UI is required.

<a id="rh-12"></a>
### RH-12 — Reproducible demonstrator and release evaluation

**Dependencies:** RH-01 through RH-11. **Area:** `area:testing`.

- Package the pinned `tf_gnns` run configuration, synthetic/recorded fixtures,
  end-to-end tests, setup/privacy instructions, run manifests, and inspection report.
- Measure parse/ingest time, counts, storage, requests, model tokens, cache/resume
  behavior, and topic/link quality on a small annotated fixture; do not claim
  unmeasured accuracy or full-history coverage.
- **Acceptance:** offline tests need no secrets/network; the bounded live run
  indexes 16 commits and PRs #10-#12 with explicit gaps; an enabled LangChain model
  path is exercised; a rerun/resume adds no duplicates and respects all budgets.

## Release acceptance and scope boundaries

Release only when the indexed slices' acceptance checks pass, the demonstrator
answers the inspection questions, and every derived conclusion can be traced to
source evidence and an analysis configuration. Include actual live-run artifacts
or a reproducible report; synthetic cases must be identified as such.

Non-goals for this epic: GitLab implementation, distributed scheduling, complete
multi-language semantic compilation, full custom UI, unrestricted autonomous
agents, GitHub issue/label mutation, universal branch-history reconstruction,
and indexing the entire `tf_gnns` history. The interfaces should permit later
adapters and richer analysis without making them first-release requirements.
