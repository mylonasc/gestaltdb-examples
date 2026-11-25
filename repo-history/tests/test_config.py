"""Configuration loading and fingerprinting."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from repo_history.config import DEFAULT_CONFIG, config_fingerprint, load_config
from repo_history.models import GitSettings, RuntimeConfig


def test_sample_config_expresses_the_pinned_slice():
    config = load_config()
    assert DEFAULT_CONFIG.is_file()
    assert config.repository.full_name == "mylonasc/tf_gnns"
    assert config.repository.anchor == "9bfe6c19c02723dbbf101b9e311de684a4728bb1"
    assert config.repository.window.start == datetime(2026, 9, 13, tzinfo=timezone.utc)
    assert config.repository.window.end == datetime(2026, 9, 15, tzinfo=timezone.utc)
    assert config.repository.allowed_work_items == (10, 11, 12)
    assert config.llm.enabled is False
    assert config.llm.privacy_consent is False


def test_sample_config_is_offline_capable_without_credentials():
    config = load_config()
    assert config.budgets.max_commits == 32
    assert config.offline is False
    assert config.cache_dir == "data/cache"
    assert "openai" in config.llm.provider


def test_git_settings_are_loaded_from_toml():
    config = load_config()
    assert config.git.max_depth == 256
    assert config.git.initial_fetch_depth == 64
    assert config.git.max_fetch_depth == 512
    assert config.git.skew_tolerance_seconds == 0
    assert config.git.rename_threshold == 50
    assert config.git.max_blob_bytes == 2097152


def test_git_settings_reject_inverted_fetch_depths():
    with pytest.raises(ValidationError):
        GitSettings(initial_fetch_depth=32, max_fetch_depth=16)
    with pytest.raises(ValidationError):
        GitSettings(max_depth=0)
    with pytest.raises(ValidationError):
        GitSettings(rename_threshold=101)


def test_fingerprint_is_stable_and_sensitive(tmp_path):
    original = load_config()
    assert config_fingerprint(original) == config_fingerprint(load_config())
    assert len(config_fingerprint(original)) == 64

    changed = original.model_copy(
        update={"budgets": original.budgets.model_copy(update={"max_commits": 31})}
    )
    assert config_fingerprint(changed) != config_fingerprint(original)

    text = DEFAULT_CONFIG.read_text().replace("max_commits = 32", "max_commits = 31")
    path = tmp_path / "changed.toml"
    path.write_text(text)
    assert config_fingerprint(load_config(path)) == config_fingerprint(changed)


def test_typed_sections_reject_unknown_keys():
    config = load_config()
    with pytest.raises(ValidationError):
        RuntimeConfig.model_validate(
            {**config.model_dump(), "unexpected": "value"}
        )


def test_missing_required_config_key_fails(tmp_path):
    text = DEFAULT_CONFIG.read_text().replace(
        'anchor = "9bfe6c19c02723dbbf101b9e311de684a4728bb1"\n', ""
    )
    path = tmp_path / "missing-anchor.toml"
    path.write_text(text)
    with pytest.raises(ValidationError):
        load_config(path)
