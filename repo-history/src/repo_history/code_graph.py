"""Source-structure and dependency projections, separate from persistence.

This module consumes the common code IR. Python's repository-aware import
policy lives in python_resolution, not in the graph sink or the parser.
"""

from __future__ import annotations

import difflib
import sys
from collections import defaultdict

from .content import ContentIndex
from .git import FileChangeRecord, GitIndex
from .graph import EntityWrite, MappedGraph, RelationshipWrite
from .ids import IdFactory, stable_id
from .python_resolution import PythonModuleIndex, python_module_name


class _Writes:
    def __init__(self, index, ids, run_id):
        self.ids, self.run_id = ids, run_id
        self.commits = {
            c.sha: c for c in (*index.selection.selected_commits, *index.selection.boundary_commits)
        }
        self.entities, self.edges, self.skipped = {}, {}, []

    def entity(self, sha, node_id, label, props):
        commit = self.commits[sha]
        self.entities[sha, node_id] = EntityWrite(
            logical_id=node_id, labels=(label,), properties=props,
            valid_from=commit.committed_at, origin="git", origin_id=f"{sha}:{node_id}",
            event_id=stable_id("graph-write", self.ids.repository_id, sha, node_id),
            run_id=self.run_id, info_horizon=commit.committed_at,
            evidence_ids=(self.ids.evidence("git", sha, commit.source_url),),
        )

    def edge(self, sha, relation, source, target, **props):
        commit = self.commits[sha]
        self.edges[sha, relation, source, target] = RelationshipWrite(
            edge_id=stable_id("edge", self.ids.repository_id, relation, source, target),
            edge_type=relation, source=source, target=target, properties=props,
            valid_from=commit.committed_at, origin="git",
            origin_id=f"{sha}:{source}:{relation}:{target}",
            event_id=stable_id("graph-write", self.ids.repository_id, sha, relation, source, target),
            run_id=self.run_id, info_horizon=commit.committed_at,
            evidence_ids=(self.ids.evidence("git", sha, commit.source_url),),
        )

    def result(self):
        order = lambda item: (item.valid_from, item.origin_id)
        return MappedGraph(
            entities=tuple(sorted(self.entities.values(), key=order)),
            relationships=tuple(sorted(self.edges.values(), key=order)),
            skipped=tuple(sorted(set(self.skipped))),
        )


def _symbols_by_name(parsed):
    groups = defaultdict(list)
    if parsed and parsed.code:
        for symbol in parsed.code.symbols:
            groups[symbol.kind.removeprefix("async-"), symbol.qualname].append(symbol)
    return groups


def map_code_components(content: ContentIndex, index: GitIndex, ids: IdFactory, *, run_id: str):
    """Map modules and all retained definitions with lexical ownership."""
    out = _Writes(index, ids, run_id)
    for snap in content.snapshots:
        if snap.commit_sha not in out.commits or snap.language != "python":
            continue
        parsed = content.content_for(snap.snapshot_id)
        module = python_module_name(snap.path)
        importable = module is not None
        module = module or snap.path.replace("/", ".")
        module_id = ids.module(snap.path)
        out.entity(snap.commit_sha, module_id, "Module", {
            "name": module, "path": snap.path, "language": "python",
            "is_external": False, "commit_sha": snap.commit_sha,
            "definition_url": snap.source_url, "parse_status": snap.status,
            "importable": importable,
        })
        out.edge(snap.commit_sha, "DEFINED_AT", module_id, ids.file(snap.path), hierarchical=True)
        groups = _symbols_by_name(parsed)
        owners = {qualname: ids.symbol(snap.path, kind, qualname) for kind, qualname in groups}
        for (kind, qualname), declarations in groups.items():
            symbol = declarations[0]
            node_id = ids.symbol(snap.path, kind, qualname)
            locations = [
                {"line": s.start_line or s.line, "end_line": s.end_line}
                for s in declarations
            ]
            props = {
                "name": symbol.name, "qualname": qualname, "path": snap.path,
                "language": "python", "kind": symbol.kind, "parent": symbol.parent,
                "commit_sha": snap.commit_sha, "line": symbol.line,
                "end_line": symbol.end_line, "locations": locations,
                "decorators": list(symbol.decorators), "bases": list(symbol.bases),
                "annotation": symbol.annotation, "field_scope": symbol.field_scope,
                "docstring": symbol.docstring,
                "definition_url": f"{snap.source_url}#L{symbol.line}-L{symbol.end_line}",
            }
            out.entity(snap.commit_sha, node_id, kind.title(), props)
            out.edge(snap.commit_sha, "DEFINED_AT", node_id, ids.file(snap.path), hierarchical=True)
            owner = owners.get(symbol.parent, module_id)
            out.edge(snap.commit_sha, "BELONGS_TO", node_id, owner, hierarchical=True)
    return out.result()


def map_imports(content: ContentIndex, index: GitIndex, ids: IdFactory, *, run_id: str):
    """File IMPORTS module/symbol; source module DEPENDS_ON external library.

    External Module --PROVIDED_BY--> ExternalLibrary preserves submodule
    detail while aggregating dependencies by observed import root. Stdlib is
    identified separately, using the indexing interpreter's stdlib inventory.
    """
    out = _Writes(index, ids, run_id)
    resolver = PythonModuleIndex(index, content)
    observations = defaultdict(list)
    for snap in content.snapshots:
        sha = snap.commit_sha
        if sha not in out.commits or snap.language != "python":
            continue
        parsed = content.content_for(snap.snapshot_id)
        if not parsed or not parsed.code:
            continue
        for record in parsed.code.imports:
            targets, gaps = resolver.resolve(sha, snap.path, record)
            out.skipped.extend(f"{sha}:{snap.path}:{record.line}:{gap}" for gap in gaps)
            for target in targets:
                if target.external:
                    root = target.module.split(".")[0]
                    target_id = ids.external_module(target.module)
                    library_id = ids.library(root)
                    out.entity(sha, library_id, "ExternalLibrary", {
                        "name": root, "language": "python", "import_root": root,
                        "dependency_kind": "stdlib" if root in sys.stdlib_module_names else "third-party-or-unresolved",
                        "identity_basis": "observed-import-name", "distribution": None,
                    })
                    out.entity(sha, target_id, "Module", {
                        "name": target.module, "language": "python", "is_external": True,
                        "resolution": target.resolution,
                    })
                    out.edge(sha, "PROVIDED_BY", target_id, library_id)
                    observations[sha, "DEPENDS_ON", ids.module(snap.path), library_id].append((record, target.resolution))
                elif target.namespace:
                    target_id = ids.module(f"namespace:{target.module}")
                    out.entity(sha, target_id, "Module", {
                        "name": target.module, "language": "python", "is_external": False,
                        "namespace": True,
                    })
                elif target.symbol:
                    target_id = ids.symbol(target.path, target.symbol.kind, target.symbol.qualname)
                else:
                    target_id = ids.module(target.path)
                observations[sha, "IMPORTS", ids.file(snap.path), target_id].append((record, target.resolution))
    for (sha, relation, source, target), records in observations.items():
        # All source sites survive, rather than dropping every import but the first.
        sites = sorted({
            (r.line, r.kind, r.module or "", r.names, r.aliases, r.scope, r.conditional, resolution)
            for r, resolution in records
        }, key=repr)
        out.edge(sha, relation, source, target, sites=[
            {"line": line, "kind": kind, "module": module,
             "names": list(names), "aliases": list(aliases), "scope": scope,
             "conditional": conditional, "resolution": resolution}
            for line, kind, module, names, aliases, scope, conditional, resolution in sites
        ])
    return out.result()


def _symbol_sources(parsed):
    if not parsed or parsed.text is None or not parsed.code:
        return {}
    lines = parsed.text.splitlines(keepends=True)
    return {
        key: "\n".join("".join(lines[(s.start_line or s.line) - 1:s.end_line]) for s in symbols)
        for key, symbols in _symbols_by_name(parsed).items()
    }


def module_change_targets(change: FileChangeRecord, content: ContentIndex) -> dict[str, list[str]]:
    """Observed local Python modules affected by a backing-file change.

    Unlike symbol edits, these links do not require a valid AST. Snapshot
    metadata is sufficient, even for import-only edits or unavailable text.
    A rename affects both path identities; a copy affects only its destination.
    Namespace packages and external modules are not inferred from file changes.
    """
    observations = []
    if change.status not in {"added", "copied"}:
        observations.append((change.parent_sha, change.old_path or change.path, "before"))
    if change.status != "deleted":
        observations.append((change.commit_sha, change.path, "after"))
    targets: dict[str, list[str]] = {}
    for sha, path, side in observations:
        snapshot = content.snapshots_by_path.get((sha, path))
        if snapshot is not None and snapshot.language == "python":
            targets.setdefault(path, []).append(side)
    return targets


def change_details(change, content: ContentIndex):
    """A real bounded text patch and conservative before/after symbol evidence."""
    by_oid = content.contents_by_oid
    old = None if change.status == "added" else (
        content.content_at(change.parent_sha, change.old_path or change.path) or by_oid.get(change.old_oid)
    )
    new = None if change.status == "deleted" else (
        content.content_at(change.commit_sha, change.path) or by_oid.get(change.new_oid)
    )
    old_text = "" if change.status == "added" else old.text if old else None
    new_text = "" if change.status == "deleted" else new.text if new else None
    props = {"parent_sha": change.parent_sha, "comparison": change.comparison}
    if change.is_binary:
        props.update(patch_status="binary", text=None)
    elif old_text is None or new_text is None:
        props.update(patch_status="content-unavailable", text=None)
    else:
        patch_lines = difflib.unified_diff(
            old_text.splitlines(keepends=True), new_text.splitlines(keepends=True),
            fromfile="/dev/null" if change.status == "added" else f"a/{change.old_path or change.path}",
            tofile="/dev/null" if change.status == "deleted" else f"b/{change.path}",
        )
        patch = "".join(
            line if line.endswith("\n") else line + "\n\\ No newline at end of file\n"
            for line in patch_lines
        )
        props.update(patch_status="available", text=patch[:200_000], patch_truncated=len(patch) > 200_000)
    complete = not change.is_binary and old_text is not None and new_text is not None
    for parsed in (old, new):
        if parsed and (not parsed.code or parsed.code.syntax_error or parsed.code.symbols_truncated):
            complete = False
    affected = []
    if complete:
        before, after = _symbol_sources(old), _symbol_sources(new)
        for key in sorted(before.keys() | after.keys()):
            if before.get(key) != after.get(key):
                status = "added" if key not in before else "deleted" if key not in after else "modified"
                affected.append((key, status))
    props["symbol_changes_status"] = "available" if complete else "content-unavailable-or-incomplete"
    return props, affected
