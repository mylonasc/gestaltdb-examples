"""Stable identity contracts."""

from __future__ import annotations

import pytest

from repo_history.ids import IdFactory, canonical_repository_url, stable_id


def test_canonical_repository_url_normalizes_supported_forms():
    expected = "https://github.com/mylonasc/tf_gnns"
    for value in (
        "https://github.com/mylonasc/tf_gnns",
        "https://github.com/mylonasc/tf_gnns/",
        "https://github.com/mylonasc/tf_gnns.git",
        "https://user:token@GitHub.com/mylonasc/tf_gnns.git",
        "github.com/mylonasc/tf_gnns",
    ):
        assert canonical_repository_url(value) == expected


@pytest.mark.parametrize("value", ["", "https://github.com", "not a url"])
def test_canonical_repository_url_rejects_incomplete_values(value: str):
    with pytest.raises(ValueError):
        canonical_repository_url(value)


def test_stable_id_is_deterministic_and_bounded():
    first = stable_id("commit", "repo", "abc")
    assert first == stable_id("commit", "repo", "abc")
    kind, digest = first.split(":", 1)
    assert kind == "commit"
    assert len(digest) == 24
    assert first.isascii()


def test_stable_id_rejects_bad_kind_and_size():
    with pytest.raises(ValueError):
        stable_id("bad:kind", "x")
    with pytest.raises(ValueError):
        stable_id("commit", "x", size=4)
    with pytest.raises(ValueError):
        stable_id("commit", "x", size=80)


def test_stable_id_avoids_ambiguous_delimiters():
    assert stable_id("x", "a", "b") != stable_id("x", "a\x1fb")


def test_ids_are_repository_qualified():
    first = IdFactory("https://github.com/mylonasc/tf_gnns.git")
    second = IdFactory("https://github.com/mylonasc/other")
    sha = "A" * 40
    assert first.repository_id != second.repository_id
    assert first.commit(sha) != second.commit(sha)
    assert first.commit(sha) == first.commit(sha.lower())
    assert first.commit(sha) != first.commit("B" * 40)


def test_entity_id_namespaces_do_not_collide():
    factory = IdFactory("https://github.com/mylonasc/tf_gnns")
    identifiers = {
        factory.commit("a" * 40),
        factory.branch("main"),
        factory.file("README.md"),
        factory.snapshot("a" * 40, "README.md"),
        factory.blob("b" * 40),
        factory.content("b" * 40, "markdown"),
        factory.work_item("github", 10),
        factory.topic("Benchmarking"),
    }
    assert len(identifiers) == 8


def test_content_ids_are_qualified_by_oid_and_kind():
    factory = IdFactory("https://github.com/mylonasc/tf_gnns")
    other = IdFactory("https://github.com/mylonasc/other")
    oid = "A" * 40
    markdown = factory.content(oid, "markdown")
    assert markdown == factory.content(oid.lower(), "markdown")
    assert markdown != factory.content(oid, "python")
    assert markdown != other.content(oid, "markdown")
    assert markdown.startswith("content:")


def test_topic_identity_ignores_case_and_whitespace():
    factory = IdFactory("https://github.com/mylonasc/tf_gnns")
    assert factory.topic("  Benchmarking   Agents ") == factory.topic("benchmarking agents")
    with pytest.raises(ValueError):
        factory.topic("   ")
