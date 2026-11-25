"""Real GestaltDB capability smoke test against the pinned backend.

This exercises the temporal APIs RH-06 depends on, using a temporary LMDB graph.
It requires no network access or model credentials.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("gestaltdb")

from gestaltdb.graphdb import GraphDB, Node  # noqa: E402

from repo_history.capabilities import require_capabilities  # noqa: E402

UTC = timezone.utc
VALID_FROM = datetime(2026, 9, 14, 20, 15, 31, tzinfo=UTC)


def test_required_capabilities_include_gestaltdb_temporal_api():
    checks = {item.name: item for item in require_capabilities(backend="lmdb")}
    assert checks["gestaltdb"].available
    assert checks["gestaltdb-temporal-api"].available


def test_temporal_write_read_reopen_and_index(tmp_path):
    path = tmp_path / "graph"
    database = GraphDB.create(str(path), backend="lmdb", overwrite=True)
    try:
        node = Node(
            node_id="commit:9bfe6c1",
            labels=("Commit",),
            properties={"sha": "9bfe6c1", "message": "trust fetched contracts"},
        )
        version = database.put_node_version(node, valid=(VALID_FROM, None))
        assert version.logical_id == "commit:9bfe6c1"

        before = database.get_node_as_of(
            "commit:9bfe6c1", valid_time=VALID_FROM - timedelta(seconds=1)
        )
        assert before is None

        observed = database.get_node_as_of(
            "commit:9bfe6c1", valid_time=VALID_FROM + timedelta(hours=1)
        )
        assert observed is not None
        assert observed.node.properties["sha"] == "9bfe6c1"
        assert observed.node.labels == ("Commit",)

        database.create_node_property_index("sha")
        assert "sha" in database.index_statistics()["indexed_node_properties"]
        assert len(list(database.iter_node_versions("commit:9bfe6c1"))) == 1
    finally:
        database.close()

    reopened = GraphDB.open(str(path))
    try:
        observed = reopened.get_node_as_of(
            "commit:9bfe6c1", valid_time=VALID_FROM + timedelta(hours=1)
        )
        assert observed is not None
        assert observed.node.properties["message"] == "trust fetched contracts"
    finally:
        reopened.close()
