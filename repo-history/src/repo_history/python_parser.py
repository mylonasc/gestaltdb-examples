"""Static Python adapter: AST definitions and tree-sitter import evidence.

No project code is imported. Resolution is deliberately a separate pass so
parsing stays blob-local and can be cached across paths and commits.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from itertools import islice

from tree_sitter import Language, Parser, Tree
import tree_sitter_python

from .code_models import CodeDocument, CodeSymbol, ImportRecord
from .ids import stable_id
from .models import ContentSettings


def _unparse(node: ast.AST | None) -> str | None:
    return ast.unparse(node)[:200] if node is not None else None


@dataclass(frozen=True)
class _TreeState:
    text: str
    tree: Tree


def _edit_tree(tree: Tree, before: str, after: str) -> Tree:
    """Edit a copy using one UTF-8-safe replacement spanning the changes.

    Character boundaries find the replacement; Tree.edit offsets and columns
    are measured in UTF-8 bytes, as required by tree-sitter. One span supports
    arbitrary multi-hunk edits without coupling parsing to a Git diff format.
    """
    prefix = 0
    common = min(len(before), len(after))
    while prefix < common and before[prefix] == after[prefix]:
        prefix += 1
    suffix = 0
    while suffix < common - prefix and before[-suffix - 1] == after[-suffix - 1]:
        suffix += 1
    start = before[:prefix].encode("utf-8")
    old_end = before[:len(before) - suffix].encode("utf-8")
    new_end = after[:len(after) - suffix].encode("utf-8")

    def point(data):
        return (data.count(b"\n"), len(data) - data.rfind(b"\n") - 1)

    edited = tree.copy()
    edited.edit(
        start_byte=len(start), old_end_byte=len(old_end), new_end_byte=len(new_end),
        start_point=point(start), old_end_point=point(old_end), new_end_point=point(new_end),
    )
    return edited


def _symbols(body, *, content_id, parent=None, scope="module", receiver=None):
    """Walk lexical scopes, including definitions inside control-flow blocks."""
    for node in body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            qualname = f"{parent}.{node.name}" if parent else node.name
            decorators = tuple(_unparse(item) for item in node.decorator_list)
            if isinstance(node, ast.ClassDef):
                kind = "class"
            else:
                kind = "method" if scope == "class" else "function"
                if isinstance(node, ast.AsyncFunctionDef):
                    kind = f"async-{kind}"
            yield CodeSymbol(
                symbol_id=stable_id("symbol", content_id, qualname, node.lineno),
                kind=kind, name=node.name, qualname=qualname, parent=parent,
                line=node.lineno, column=node.col_offset,
                end_line=node.end_lineno or node.lineno,
                start_line=min([node.lineno, *(d.lineno for d in node.decorator_list)]),
                decorators=decorators,
                bases=tuple(_unparse(item) for item in node.bases)
                if isinstance(node, ast.ClassDef) else (),
                docstring=(ast.get_docstring(node, clean=True) or "")[:1000] or None,
            )
            args = getattr(node, "args", None)
            positional = [*args.posonlyargs, *args.args] if args else []
            instance_receiver = None
            if scope == "class" and kind in {"method", "async-method"}:
                if not {"staticmethod", "classmethod"}.intersection(decorators) and positional:
                    instance_receiver = (positional[0].arg, parent)
            yield from _symbols(
                node.body, content_id=content_id, parent=qualname,
                scope="class" if kind == "class" else "function",
                receiver=instance_receiver,
            )
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            annotation = _unparse(node.annotation) if isinstance(node, ast.AnnAssign) else None

            def fields(target):
                if isinstance(target, (ast.Tuple, ast.List)):
                    for child in target.elts:
                        yield from fields(child)
                elif scope == "class" and isinstance(target, ast.Name):
                    yield target.id, parent, "class"
                elif (receiver and isinstance(target, ast.Attribute)
                      and isinstance(target.value, ast.Name)
                      and target.value.id == receiver[0]):
                    yield target.attr, receiver[1], "instance"

            for target in targets:
                for name, owner, field_scope in fields(target):
                    qualname = f"{owner}.{name}"
                    yield CodeSymbol(
                        symbol_id=stable_id("symbol", content_id, qualname, node.lineno),
                        kind="field", name=name, qualname=qualname, parent=owner,
                        line=node.lineno, column=node.col_offset,
                        end_line=node.end_lineno or node.lineno,
                        annotation=annotation, field_scope=field_scope,
                    )
        else:
            # if/try/with/for/match do not introduce a Python lexical scope.
            # Traverse handlers/cases too, without crossing a def/class twice.
            for _, value in ast.iter_fields(node):
                children = value if isinstance(value, list) else [value]
                for child in children:
                    if isinstance(child, (ast.stmt, ast.ExceptHandler, ast.match_case)):
                        yield from _symbols(
                            [child], content_id=content_id, parent=parent,
                            scope=scope, receiver=receiver,
                        )


class PythonParser:
    name = "python"
    version = "python-v3"

    def __init__(self):
        self._parser = Parser(Language(tree_sitter_python.language()))
        # Transient native trees, never serialized into the content IR. Cached
        # parents are immutable: each child/branch edits its own tree copy.
        self._trees: dict[str, _TreeState] = {}
        self._counts = {"full_parses": 0, "incremental_parses": 0, "tree_cache_hits": 0}

    @property
    def counters(self) -> dict[str, int]:
        """Count actual tree-sitter calls, separately from blob-result caching."""
        return dict(self._counts)

    def _tree(self, text: str, content_id: str, previous_content_id: str | None) -> Tree:
        cached = self._trees.get(content_id)
        previous = self._trees.get(previous_content_id) if previous_content_id is not None else None
        if cached is not None and cached.text == text:
            self._counts["tree_cache_hits"] += 1
            return cached.tree
        if previous is not None:
            edited = _edit_tree(previous.tree, previous.text, text)
            tree = self._parser.parse(text.encode("utf-8"), edited)
            self._counts["incremental_parses"] += 1
        else:
            tree = self._parser.parse(text.encode("utf-8"))
            self._counts["full_parses"] += 1
        self._trees[content_id] = _TreeState(text=text, tree=tree)
        return tree

    def _imports(self, tree: Tree, limit: int):
        root = tree.root_node
        records = []
        stack = [root]
        while stack:
            node = stack.pop()
            if node.type not in {"import_statement", "import_from_statement", "future_import_statement"}:
                stack.extend(reversed(node.named_children))
                continue
            # Never manufacture an import out of an error-recovery fragment.
            if node.has_error:
                continue
            module_node = node.child_by_field_name("module_name")
            raw_module = module_node.text.decode("utf-8") if module_node else ""
            if node.type == "future_import_statement":
                raw_module = "__future__"
            level = len(raw_module) - len(raw_module.lstrip("."))
            names, aliases = [], []
            for i, child in enumerate(node.children):
                if node.field_name_for_child(i) != "name":
                    if child.type == "wildcard_import":
                        names.append("*")
                        aliases.append(None)
                    continue
                name = child.child_by_field_name("name") if child.type == "aliased_import" else child
                alias = child.child_by_field_name("alias")
                names.append(name.text.decode("utf-8"))
                aliases.append(alias.text.decode("utf-8") if alias else None)
            records.append(ImportRecord(
                kind="from" if node.type != "import_statement" else "import",
                module=raw_module.lstrip(".") or None, level=level,
                names=tuple(names), aliases=tuple(aliases), line=node.start_point.row + 1,
                **self._import_context(node),
            ))
            if len(records) > limit:
                break
        return tuple(records[:limit]), len(records) > limit

    @staticmethod
    def _import_context(node):
        scopes, conditional = [], False
        ancestor = node.parent
        while ancestor:
            if ancestor.type in {"class_definition", "function_definition"}:
                name = ancestor.child_by_field_name("name")
                if name:
                    scopes.append(name.text.decode("utf-8"))
            elif ancestor.type in {"if_statement", "try_statement", "for_statement", "while_statement", "match_statement"}:
                conditional = True
            ancestor = ancestor.parent
        return {"scope": ".".join(reversed(scopes)) or None, "conditional": conditional}

    def parse(self, text, *, content_id, settings):
        return self._parse(text, content_id=content_id, settings=settings)

    def parse_incremental(self, text, *, content_id, settings, previous_content_id):
        return self._parse(
            text, content_id=content_id, settings=settings,
            previous_content_id=previous_content_id,
        )

    def _parse(self, text, *, content_id, settings, previous_content_id=None):
        syntax_tree = self._tree(text, content_id, previous_content_id)
        imports, imports_truncated = self._imports(syntax_tree, settings.max_imports)
        common = dict(
            language="python", imports=imports, imports_truncated=imports_truncated,
            shebang=text.split("\n", 1)[0][:200] if text.startswith("#!") else None,
        )
        try:
            tree = ast.parse(text)
            symbols = list(islice(
                _symbols(tree.body, content_id=content_id), settings.max_symbols + 1
            ))
        except (SyntaxError, ValueError, RecursionError) as error:
            return CodeDocument(
                **common, syntax_error=f"{error} (Python AST)"[:200]
            )
        return CodeDocument(
            **common,
            docstring=(ast.get_docstring(tree, clean=True) or "")[:1000] or None,
            symbols=tuple(symbols[:settings.max_symbols]),
            symbols_truncated=len(symbols) > settings.max_symbols,
        )
