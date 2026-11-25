"""Capability checks fail explicitly instead of silently degrading."""

from __future__ import annotations

import importlib

import pytest

from repo_history import capabilities
from repo_history.capabilities import (
    Capability,
    CapabilityError,
    check_capabilities,
    require_capabilities,
)


def test_checks_are_reported_per_capability_without_raising():
    checks = check_capabilities()
    names = {item.name for item in checks}
    assert {"pygit2", "httpx", "pydantic", "markdown_it", "langchain_core", "gestaltdb"} <= names
    assert "gestaltdb-temporal-api" in names
    assert "gestaltdb-sampling" in names
    assert all(isinstance(item, Capability) for item in checks)
    assert all(item.detail for item in checks)


def test_missing_dependency_is_explicit(monkeypatch):
    real_import = importlib.import_module

    def fake_import(name: str, package: str | None = None):
        if name == "pygit2":
            raise ModuleNotFoundError("No module named 'pygit2'")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    checks = {item.name: item for item in check_capabilities()}
    assert checks["pygit2"].available is False
    assert "pygit2" in checks["pygit2"].detail


def test_require_capabilities_raises_with_failure_details(monkeypatch):
    monkeypatch.setattr(
        capabilities,
        "check_capabilities",
        lambda *, backend="lmdb": (
            Capability("pygit2", True, "1.0"),
            Capability("gestaltdb", False, "missing temporal API"),
        ),
    )
    with pytest.raises(CapabilityError) as error:
        require_capabilities()
    assert [item.name for item in error.value.failures] == ["gestaltdb"]
    assert "gestaltdb" in str(error.value)


def test_require_capabilities_returns_checks_when_all_pass(monkeypatch):
    monkeypatch.setattr(
        capabilities,
        "check_capabilities",
        lambda *, backend="lmdb": (Capability("pygit2", True, "1.0"),),
    )
    assert require_capabilities()[0].name == "pygit2"


def test_unknown_backend_is_reported(monkeypatch):
    checks = {item.name: item for item in check_capabilities(backend="does-not-exist")}
    backend = checks["gestaltdb-backend"]
    assert backend.available is False
    assert "does-not-exist" in backend.detail
