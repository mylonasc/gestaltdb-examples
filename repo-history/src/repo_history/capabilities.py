"""Explicit dependency and API capability checks for a fail-closed startup."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any


@dataclass(frozen=True)
class Capability:
    name: str
    available: bool
    detail: str


class CapabilityError(RuntimeError):
    def __init__(self, failures: list[Capability]) -> None:
        message = "; ".join(f"{item.name}: {item.detail}" for item in failures)
        super().__init__(f"Required capabilities are unavailable: {message}")
        self.failures = tuple(failures)


def _installed_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "missing"


def check_capabilities(*, backend: str = "lmdb") -> tuple[Capability, ...]:
    checks: list[Capability] = []

    for import_name, distribution in (
        ("pygit2", "pygit2"),
        ("httpx", "httpx"),
        ("pydantic", "pydantic"),
        ("markdown_it", "markdown-it-py"),
        ("langchain_core", "langchain-core"),
        ("gestaltdb", "gestaltdb"),
    ):
        try:
            module = importlib.import_module(import_name)
            checks.append(
                Capability(import_name, True, f"{_installed_version(distribution)} at {module.__file__}")
            )
        except Exception as error:  # import failures are capability failures
            checks.append(Capability(import_name, False, str(error)))

    try:
        graphdb_module = importlib.import_module("gestaltdb.graphdb")
        graphdb: Any = graphdb_module.GraphDB
        required = (
            "create", "open", "put_node", "put_edge", "put_node_version",
            "put_edge_version", "get_node_as_of", "iter_node_versions",
            "create_node_property_index", "close",
        )
        missing = [name for name in required if not hasattr(graphdb, name)]
        backend_module = {
            "lmdb": "gestaltdb.kvstores",
            "leveldb": "gestaltdb.kvstores",
            "pyrex": "gestaltdb.kvstores",
        }.get(backend)
        if backend_module is None:
            checks.append(Capability("gestaltdb-backend", False, f"unknown backend {backend!r}"))
        else:
            module = importlib.import_module(backend_module)
            backend_class = {
                "lmdb": "LMDBStore", "leveldb": "LevelDBStore", "pyrex": "PyRexStore"
            }[backend]
            if not hasattr(module, backend_class):
                missing.append(f"kvstores.{backend_class}")
        checks.append(
            Capability(
                "gestaltdb-temporal-api",
                not missing,
                "available" if not missing else f"missing: {', '.join(missing)}",
            )
        )
    except Exception as error:
        checks.append(Capability("gestaltdb-temporal-api", False, str(error)))

    try:
        sampling = importlib.import_module("gestaltdb.sampling")
        checks.append(
            Capability("gestaltdb-sampling", hasattr(sampling, "SamplerSnapshot"), str(sampling.__file__))
        )
    except Exception as error:
        checks.append(Capability("gestaltdb-sampling", False, str(error)))

    return tuple(checks)


def require_capabilities(*, backend: str = "lmdb") -> tuple[Capability, ...]:
    checks = check_capabilities(backend=backend)
    failures = [item for item in checks if not item.available]
    if failures:
        raise CapabilityError(failures)
    return checks
