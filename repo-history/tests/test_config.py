"""Configuration loading and fingerprinting."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from repo_history.config import DEFAULT_CONFIG, config_fingerprint, load_config
from repo_history.models import ContentSettings, GitHubSettings, GitSettings, RuntimeConfig


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


def test_content_settings_are_loaded_from_toml():
    config = load_config()
    assert config.content.max_file_bytes == 2097152
    assert config.content.max_sections == 500
    assert config.content.max_symbols == 1000
    assert config.content.max_imports == 500


def test_content_settings_reject_invalid_limits():
    with pytest.raises(ValidationError):
        ContentSettings(max_file_bytes=1)
    with pytest.raises(ValidationError):
        ContentSettings(max_sections=0)
    with pytest.raises(ValidationError):
        ContentSettings(max_symbols=10_001)
    with pytest.raises(ValidationError):
        ContentSettings(max_imports=0)


def test_github_settings_are_loaded_from_toml():
    config = load_config()
    assert config.github.api_url == "https://api.github.com"
    assert config.github.token_env == "GITHUB_TOKEN"
    assert config.github.per_page == 50
    assert config.github.max_pages == 20
    assert config.github.max_retries == 2
    assert config.github.max_response_bytes == 2097152
    assert config.github.required_scopes == ()
    assert "repo-history" in config.github.user_agent


def test_github_settings_reject_invalid_limits():
    with pytest.raises(ValidationError):
        GitHubSettings(per_page=101)
    with pytest.raises(ValidationError):
        GitHubSettings(max_pages=0)
    with pytest.raises(ValidationError):
        GitHubSettings(max_retries=6)
    with pytest.raises(ValidationError):
        GitHubSettings(api_url="api.github.com")
    with pytest.raises(ValidationError):
        GitHubSettings(token_env="  ")


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
