"""Repository-aware Python import resolution; no imports or filesystem probing.

Only a single commit's tree supplies resolution candidates. Missing local
members and relative imports stay unresolved rather than becoming libraries.
An external library identity is an observed import root, not a PyPI claim.
"""

from __future__ import annotations

from dataclasses import dataclass

from .code_models import CodeSymbol, ImportRecord
from .content import ContentIndex
from .git import GitIndex
from .python_names import python_module_name


@dataclass(frozen=True)
class ImportTarget:
    module: str
    path: str | None = None
    symbol: CodeSymbol | None = None
    external: bool = False
    namespace: bool = False
    resolution: str = "module"


class PythonModuleIndex:
    def __init__(self, index: GitIndex, content: ContentIndex, *, source_roots=("src", "lib")):
        self.content = content
        self.roots = source_roots
        self.modules: dict[str, dict[str, list[str]]] = {}
        self.local_roots: set[str] = set()
        for entry in index.snapshots:
            name = python_module_name(entry.path, source_roots)
            if name:
                self.modules.setdefault(entry.commit_sha, {}).setdefault(name, []).append(entry.path)
                self.local_roots.add(name.split(".")[0])

    def module(self, sha: str, name: str) -> ImportTarget | None:
        available = self.modules.get(sha, {})
        paths = sorted(set(available.get(name, ())))
        implementations = [path for path in paths if path.endswith(".py")]
        if len(implementations) == 1:
            paths = implementations
        if len(paths) == 1:
            return ImportTarget(name, path=paths[0])
        if not paths and any(key.startswith(name + ".") for key in available):
            return ImportTarget(name, namespace=True)
        return None

    def _member(self, sha: str, base: ImportTarget, name: str, visited) -> ImportTarget | None:
        key = (base.path, name)
        if key in visited or len(visited) >= 16:
            return None
        visited = {*visited, key}
        if base.path:
            parsed = self.content.content_at(sha, base.path)
            if parsed and parsed.code and not parsed.code.syntax_error:
                symbols = [s for s in parsed.code.symbols if s.qualname == name]
                if len(symbols) == 1:
                    return ImportTarget(base.module, path=base.path, symbol=symbols[0], resolution="symbol")
                # Follow only explicit, unconditional module-level re-exports.
                # Cycles, star imports, dynamic __all__, and conditional
                # bindings are intentionally not guessed.
                exports = []
                for record in parsed.code.imports:
                    if record.scope or record.conditional:
                        continue
                    for i, imported in enumerate(record.names):
                        alias = record.aliases[i] if i < len(record.aliases) else None
                        bound = alias or (imported.split(".")[0] if record.kind == "import" else imported)
                        if bound != name:
                            continue
                        if record.kind == "import" and not alias:
                            imported = imported.split(".")[0]
                        one = record.model_copy(update={"names": (imported,), "aliases": (alias,)})
                        targets, _ = self.resolve(sha, base.path, one, _visited=visited)
                        exports.extend(target for target in targets if "member-unresolved" not in target.resolution)
                if len(exports) == 1:
                    return exports[0]
        return self.module(sha, f"{base.module}.{name}")

    def resolve(self, sha: str, importing_path: str, record: ImportRecord, *, _visited=frozenset()):
        """Return (targets, diagnostic strings); ambiguity is never guessed."""
        own_name = python_module_name(importing_path, self.roots)
        module = record.module or ""
        if record.level:
            parts = (own_name or "").split(".") if own_name else []
            if not importing_path.endswith(("/__init__.py", "/__init__.pyi")):
                parts = parts[:-1]
            if record.level > len(parts):
                return [], ["relative-import-beyond-package"]
            parts = parts[:len(parts) - record.level + 1]
            module = ".".join([*parts, *([module] if module else [])])
        targets, gaps = [], []
        names = tuple(name.split(" as ", 1)[0] for name in record.names)
        queries = names if record.kind == "import" and not record.module else (module,)
        for query in queries:
            base = self.module(sha, query)
            if base is None:
                root = query.split(".")[0]
                if record.level or root in self.local_roots:
                    gaps.append(f"unresolved-local-import:{query}")
                elif root:
                    targets.append(ImportTarget(query, external=True, resolution="external-import-name"))
                continue
            if record.kind == "import" or not names:
                targets.append(base)
                continue
            for name in names:
                if name == "*":
                    targets.append(base)
                    gaps.append(f"wildcard-not-expanded:{query}")
                    continue
                member = self._member(sha, base, name, _visited)
                if member:
                    targets.append(member)
                else:
                    # A known module is still useful evidence, but the member
                    # could be a re-export or a dynamically defined attribute.
                    targets.append(ImportTarget(
                        base.module, path=base.path, namespace=base.namespace,
                        resolution="module-only; member-unresolved",
                    ))
                    gaps.append(f"unresolved-member:{query}.{name}")
        return targets, gaps
