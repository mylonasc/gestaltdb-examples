"""Protocols are runtime-checkable and credential-free to import."""

from __future__ import annotations

import inspect

from repo_history import (
    Budgets,
    CoverageGap,
    CoverageStatus,
    EventKind,
    IngestEvent,
    RepositorySpec,
    RuntimeConfig,
    TimeWindow,
    load_config,
)
from repo_history.protocols import (
    CodeParser,
    DocumentParser,
    GraphSink,
    Labeler,
    RelationshipResolver,
    RepositorySource,
    WorkItemProvider,
)


def test_public_api_imports_without_credentials():
    assert load_config().repository.name == "tf_gnns"
    assert all(
        value is not None
        for value in (
            Budgets, CoverageGap, CoverageStatus, EventKind,
            IngestEvent, RepositorySpec, RuntimeConfig, TimeWindow,
        )
    )


def test_protocols_are_runtime_checkable_and_small():
    for protocol in (
        RepositorySource, WorkItemProvider, DocumentParser, CodeParser,
        Labeler, RelationshipResolver, GraphSink,
    ):
        assert hasattr(protocol, "_is_runtime_protocol")
    assert list(inspect.signature(GraphSink.write_entities).parameters) == [
        "self", "entities", "analysis_run_id"
    ]


def test_a_structural_fake_satisfies_the_parser_protocol():
    class FakeDocumentParser:
        name = "fake-markdown"

        def parse(self, content: bytes, *, path: str):
            return {"path": path, "length": len(content)}

    assert isinstance(FakeDocumentParser(), DocumentParser)
    assert FakeDocumentParser().parse(b"# hi", path="README.md")["length"] == 4
