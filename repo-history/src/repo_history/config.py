"""TOML configuration loading and deterministic config fingerprints."""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any

from .models import (
    Budgets,
    GitSettings,
    LLMSettings,
    RepositorySpec,
    RuntimeConfig,
    TimeWindow,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "tf_gnns.toml"


def _parse_window(data: dict[str, Any]) -> TimeWindow:
    return TimeWindow(start=data["start"], end=data["end"])


def load_config(path: str | Path | None = None) -> RuntimeConfig:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    repository_data = dict(raw["repository"])
    window = repository_data.pop("window")
    repository = RepositorySpec(window=_parse_window(window), **repository_data)
    runtime = RuntimeConfig(
        repository=repository,
        budgets=Budgets(**raw.get("budgets", {})),
        git=GitSettings(**raw.get("git", {})),
        llm=LLMSettings(**raw.get("llm", {})),
        data_root=raw.get("runtime", {}).get("data_root", "data"),
        cache_dir=raw.get("runtime", {}).get("cache_dir", "data/cache"),
        offline=raw.get("runtime", {}).get("offline", False),
    )
    return runtime


def config_fingerprint(config: RuntimeConfig) -> str:
    payload = config.model_dump(mode="json")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
