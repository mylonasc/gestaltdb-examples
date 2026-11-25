# Parser/indexer review

## Scope and outcome

The original content extractor was safe in an important respect: it inspected
Git blobs without importing or running project code. It already retained Python
functions and methods, but its class-only graph projection hid them. The changes
in this review preserve that safety boundary and expose more source structure.

This is a static source index, **not a Python interpreter or type checker**.
Definition/dependency edges are evidence about source, not proof of runtime use.

## Correctness findings addressed

| Finding | Change |
| --- | --- |
| Aliases were embedded in names (`numpy as np`) and failed resolution. | Tree-sitter extracts names and aliases separately; import sites retain scope, conditionality, and line numbers. |
| Relative imports combined the package and imported name incorrectly; `src/` layouts and `.pyi` files were not resolved consistently. | Shared module naming and a dedicated Python module table handle relative imports, source roots, namespaces, and implementation/stub preference. |
| Import candidates came from the union of all historical trees. | Candidates are now scoped to the importing commit. Missing local targets remain diagnostics, not invented external dependencies. |
| Definitions inside `if`, `try`, and other control-flow blocks were missed. | Definition traversal preserves lexical scope while walking those blocks. Exact-limit truncation no longer reports a false gap. |
| Identical blobs under different non-Python languages could share an incorrect cached language. | Parse identities include language, parser version, extraction policy, and content-envelope version. Missing-content identities are language-specific too. |
| Every content lookup scanned the snapshot and content arrays. | Lazy, nonserialized lookup tables make repeated lookup constant-time. |
| Python encoding declarations were ignored. | Python's encoding-cookie detection is honored. Other content remains UTF-8; binary content is not interpreted as source. |
| Viewer and graph had separate diff implementations and exported each change edge twice. | Both use `map_diffs`; the viewer deduplicates logical edges and projects the newest node version. The live slice has **145**, not 290, distinct changes. |
| Diff previews contained file text rather than an actual patch. | Diffs now retain bounded before/after unified patches, comparison/parent provenance, explicit missing/binary status, and symbol-level `AFFECTS` edges. |
| The offline HTML fallback interpolated repository text without escaping it. | Both the fallback and selected-node panel escape untrusted text. |
| The evaluation example performed eager temporal-index maintenance during bulk ingestion. | It now uses the sink's existing deferred-index mode, with explicit fresh-state selection for changed schemas. |
| Duplicate verification after deferred writes queried stale temporal indexes. | The sink rebuilds before duplicate-payload reads; a deferred-ingestion regression checks both node and edge idempotence. |

## Component and dependency model

```text
Commit --TOUCHES--> Diff --CHANGES--> File
                      \--AFFECTS--> Module / Class / Function / Method / Field

Module / Class / Function / Method / Field --DEFINED_AT--> File
Class / Function / Method / Field --BELONGS_TO--> immediate lexical owner

File --IMPORTS--> Module / resolved Class / resolved Function
local Module --DEPENDS_ON--> ExternalLibrary
external Module --PROVIDED_BY--> ExternalLibrary
```

- Modules are separate from files: physical paths and Python import names are
  different identities. Namespace packages can have a module without one file.
  Diffs explicitly affect observed modules backed by changed Python files,
  even for import-only edits or incomplete source. These edges use
  `basis: file-change` and retain change status plus before/after sides.
  Renames link both path identities; copies link only the destination. No
  namespace-package aggregation or external-library changes are inferred.
- Functions include nested and async functions; methods include async methods,
  decorated methods, and properties as syntactic method definitions.
- Fields include class assignments/annotations and assignments to the apparent
  instance receiver (`self.x`, including another first-parameter name). Repeated
  assignments retain all declaration locations; ordinary local variables do not
  become fields. These are syntactic field observations, not inferred object
  layouts.
- Explicit unconditional re-exports can resolve to their original definitions;
  recursion is bounded and cyclic/conditional/ambiguous exports stay unresolved.
- Library identities are **observed import roots**, not guessed PyPI distribution
  names or versions (`PIL` must not silently become `Pillow`). Standard-library
  roots are classified separately using the indexing interpreter's inventory.
  Other roots are explicitly `third-party-or-unresolved`; script-path imports,
  missing vendored code, and typos can fall into this category.
- A symbol is affected only when its retained source differs before/after a
  change. Moving its line number alone does not count. Containing classes can
  also be affected when a method changes. Missing/truncated/invalid source
  suppresses symbol-change claims, while the file-level change remains indexed.
  Pure file moves preserve old/new paths but do not pretend unchanged symbol
  bodies were edited.

## Modularity

1. `content.py`: bounded blob loading, decoding, Markdown extraction, caching,
   snapshot provenance, and injectable language-adapter dispatch.
2. `code_models.py`: common source IR and typed `SourceCodeParser` contract.
3. `python_parser.py`: AST definitions and tree-sitter import syntax; no Git,
   repository resolution, graph writes, or project execution. It retains
   transient trees and supports the optional incremental parser contract.
4. `python_names.py` / `python_resolution.py`: Python module naming and static
   import resolution against a specific Git tree.
5. `code_graph.py`: component/dependency projections and source comparisons.
6. `graph.py`: persistence, compatibility mapping entry points, and diff writes.
7. `viz_d3.py`: presentation of those same mappings, not a competing indexer.

New languages can supply another text-to-IR adapter through `code_parsers`.
Repository import-resolution rules remain language-specific. Only Python has
semantic structure/resolution support today; other recognized languages retain
text and metadata unless an adapter is supplied.

### Incremental parsing

`ContentExtractor` processes the selected/boundary commit DAG parent-first and
passes each file's actual first-parent content identity to incremental-capable
adapters, following observed rename/copy paths. `PythonParser` copies and edits
that parent's tree using a UTF-8-safe replacement span, then calls tree-sitter
with the edited old tree. Unknown/missing parent trees trigger full parsing;
unchanged blobs retain the existing result-cache shortcut. Branches never edit
the shared cached parent tree in place. Counters distinguish full parses,
incremental parses, and direct tree-cache hits.

Only tree-sitter's syntax construction is incremental: AST extraction and
import-record traversal still operate on the full resulting document. Native
trees are not serialized, and no percentage speedup is claimed. Regression
tests compare incremental results with full parses and exercise Unicode/CRLF,
multi-region edits, syntax recovery, forks/merges, clock skew, renames/copies,
and missing-parent fallback.

## Recommended next improvements (not claimed as implemented)

1. **Temporal lifecycle reconciliation:** emit explicit end-of-presence events
   for removed symbols and imports. Current historical versions retain commit
   provenance, but a union/current projection is not a branch-specific active
   symbol table. Queries must not interpret every retained definition as active
   at every revision.
2. **Project configuration:** use `pyproject.toml`/packaging metadata to configure
   source roots and distribution names. The initial `src/`, `lib/`, and flat
   layout policy is not a general replacement for Python's runtime `sys.path`.
3. **Dependency manifests:** separate observed imports, declared dependencies,
   distribution names, environment markers, extras, and resolved lockfile
   versions. Do not derive these from the indexer's installed environment.
4. **Conservative binding analysis:** improve ambiguous redefinitions, property
   getter/setter grouping, custom descriptors, `__slots__`, generated dataclass
   fields, and import shadowing. Dynamic imports, monkey patching, metaclasses,
   wildcard exports, dispatch, and call graphs require a separate analysis layer.
5. **Target-Python validation:** AST validation currently follows the indexing
   interpreter. Tree-sitter can retain valid import evidence from partially
   invalid source, but does not certify compatibility with a project's Python
   version.
6. **Temporal-version lookup and bulk persistence:** the installed GestaltDB
   driver's `GraphDB._temporal_version_id_exists` scans prior temporal commit
   descriptors for every new version. With one commit per write, this produces
   quadratic lookup work as history grows, in addition to individual durable
   writes. An indexed version-ID lookup and batched temporal commits are higher
   priority than more parser micro-optimizations. Deferred secondary indexes
   alone do not remove that cost. Full live-history persistence benchmarking was
   interrupted rather than reported as successfully completed. Synthetic
   persistence and duplicate checks cover the new projections. The installed
   dependency was not modified.

Tests cover source decoding/caching, aliases and import recovery, lexical scope,
fields, async methods, namespaces, relative imports, re-exports/cycles, stubs,
commit-local resolution, external libraries, real patches, deletion/rename,
unchanged-symbol exclusion, idempotent persistence, and duplicate-free exports.

The rebuilt live viewer contains 349 functions, 151 methods, 319 field
observations, 56 local modules, and 59 external modules. The 51 library roots
comprise 35 stdlib roots and 16 third-party/unresolved roots. Its 145 distinct
diffs have available patches; 566 `AFFECTS` relationships identify source
definitions and 21 explicitly identify changed local modules (16 additions,
5 modifications). Extraction made 55 full and 4 incremental tree-sitter calls;
all 59 resulting Python documents matched fresh parsing in the live check.
These are bounded-slice observations, not a claim of complete Python runtime
analysis or a percentage performance gain.
