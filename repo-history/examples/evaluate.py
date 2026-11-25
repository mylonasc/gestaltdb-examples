"""RH-12 live evaluation: timed bounded run, budgets, resume, quality notes.

Default is offline (recorded GitHub evidence, read-only Git clone, no model
calls).  Flags opt in to more:

- ``--online``: refresh GitHub evidence from the API first (bounded).
- ``--llm``: exercise one enabled LangChain model call (needs OPENAI_API_KEY).

The run is idempotent across invocations: stages checkpoint in
``/tmp/opencode/rh12-eval`` and a rerun skips completed stages, writes no
duplicates, and spends no additional budget.  Results go to
``examples/evaluation.md`` and ``examples/run-manifest.json``.
"""

import argparse
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from repo_history.budgets import BudgetLedger
from repo_history.config import config_fingerprint, load_config
from repo_history.content import ContentExtractor
from repo_history.enrichment import Enricher, EnrichmentCache, build_chat_model
from repo_history.evaluate import (
    EvaluationReport,
    Timer,
    storage_bytes,
    summarize_ledger,
)
from repo_history.git import GitHistory
from repo_history.github import EvidenceCache, GitHubCredentials, GitHubDiscoverer
from repo_history.graph import (
    GraphSink,
    map_classes,
    map_discovery,
    map_folders,
    map_git_index,
    map_imports,
    map_registry,
    map_repository_branches,
    map_topic_assignments,
)
from repo_history.ids import IdFactory
from repo_history.labeling import (
    DeterministicLabeler,
    assign_commit,
    extract_snapshot_features,
    extract_work_item_features,
    register_prevalent,
    seed_commit_type_registry,
    seed_structural_registry,
)
from repo_history.models import LLMSettings
from repo_history.orchestration import (
    JournalManifest,
    StageRunner,
    TemporalCoordinator,
    git_selection_events,
    software_version,
)

EVAL_ROOT = Path("/tmp/opencode/rh12-eval")
RUN_ID = "live-eval"


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded live evaluation")
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--llm", action="store_true")
    args = parser.parse_args()

    overall_start = time.perf_counter()
    started_at = datetime.now(timezone.utc)
    config = load_config()
    ids = IdFactory(config.repository.url)
    timings: dict[str, float] = {}
    notes: list[str] = []

    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = JournalManifest(
        run_id=RUN_ID,
        repository_id=ids.repository_id,
        repository_url=ids.repository_url,
        anchor=config.repository.anchor,
        window_start=config.repository.window.start,
        window_end=config.repository.window.end,
        config_fingerprint=config_fingerprint(config),
        software_version=software_version(),
        created_at=started_at,
    )
    runner = StageRunner.resume_or_create(
        EVAL_ROOT / "events.jsonl",
        EVAL_ROOT / "checkpoint.json",
        manifest,
        config.budgets,
    )
    state: dict = {}

    snapshot_before = runner.journal.read()
    ran: set[str] = set()

    def git_stage():
        ran.add("git")
        with Timer() as timer:
            history = GitHistory.open(
                "/tmp/opencode/tf_gnns-rh03", repository_url=config.repository.url
            )
            try:
                selection = history.select(
                    config.repository.anchor,
                    config.repository.window,
                    max_depth=config.git.max_depth,
                )
                index = history.index(selection, ledger=runner.ledger)
            finally:
                history.close()
        timings["git-select-index"] = timer.seconds
        state["selection"] = selection
        state["index"] = index
        print(
            f"git: {len(selection.selected_commits)} selected, "
            f"{len(selection.boundary_commits)} boundary, "
            f"{len(index.snapshots)} snapshots",
            flush=True,
        )
        return (
            git_selection_events(selection, ids, observed_at=datetime.now(timezone.utc)),
            selection.gaps,
        )

    def github_stage():
        ran.add("github")
        with Timer() as timer:
            cache = EvidenceCache(
                Path("/tmp/opencode/rh04-live-cache"), offline=not args.online
            )
            discoverer = GitHubDiscoverer(
                config.repository, config.github, ledger=runner.ledger, cache=cache,
                credentials=GitHubCredentials.from_env(config.github),
            )
            discovery = discoverer.run()
        timings["github"] = timer.seconds
        state["discovery"] = discovery
        print(
            f"github: {[i.number for i in discovery.work_items]} "
            f"(online={args.online}, gaps={[g.code for g in discovery.gaps]})",
            flush=True,
        )
        return (
            discovery.to_events(ids),
            discovery.gaps,
        )

    runner.run_stage("git", git_stage)
    runner.run_stage("github", github_stage)
    if "git" not in ran or "github" not in ran:
        # Resumed run: re-derive read-only state without spending budget.
        history = GitHistory.open(
            "/tmp/opencode/tf_gnns-rh03", repository_url=config.repository.url
        )
        try:
            state.setdefault("selection", history.select(
                config.repository.anchor,
                config.repository.window,
                max_depth=config.git.max_depth,
            ))
            state.setdefault(
                "index",
                history.index(state["selection"]),
            )
        finally:
            history.close()
        if "discovery" not in state:
            cache = EvidenceCache(
                Path("/tmp/opencode/rh04-live-cache"), offline=True
            )
            state["discovery"] = GitHubDiscoverer(
                config.repository, config.github, cache=cache,
                credentials=GitHubCredentials.from_env(config.github),
            ).run()
        notes.append("Resumed: completed stages skipped, no new budget spent.")
    snapshot_after = runner.journal.read()
    new_events = len(snapshot_after.events) - len(snapshot_before.events)
    print(f"journal: +{new_events} events ({runner.completed_stages} complete)", flush=True)

    selection, index, discovery = state["selection"], state["index"], state["discovery"]
    coordinator = TemporalCoordinator(config.repository)
    result = coordinator.collect(snapshot_after.events)

    with Timer() as timer:
        history = GitHistory.open(
            "/tmp/opencode/tf_gnns-rh03", repository_url=config.repository.url
        )
        try:
            content = ContentExtractor.from_history(history).extract(index)
        finally:
            history.close()
    timings["content"] = timer.seconds
    print(f"content: {content.counters.get('parsed_contents', 0)} parsed", flush=True)

    def graph_stage():
        ran.add("graph")
        with Timer() as timer:
            labeler = DeterministicLabeler(config.labeling)
            all_paths = sorted({snapshot.path for snapshot in index.snapshots})
            registry = seed_structural_registry(
                all_paths, at=started_at, created_by=RUN_ID, ids=ids
            )
            registry = seed_commit_type_registry(
                at=started_at, created_by=RUN_ID, ids=ids, base=registry
            )
            file_features = [
                extract_snapshot_features(
                    snap, content.content_for(snap.snapshot_id), subject_id=snap.snapshot_id
                )
                for snap in content.snapshots
            ]
            proposals: dict[str, list[str]] = {}
            for features in file_features:
                got = labeler.assign(
                    features, registry, subject_id=features.subject_id,
                    run_id=RUN_ID, info_horizon=started_at,
                )
                name = got.result.value.get("proposed_name")
                if got.action == "propose" and name and registry.lookup(name) is None:
                    proposals.setdefault(name, []).append(features.text())
            registry, _ = register_prevalent(
                registry, proposals, min_subjects=2, at=started_at,
                created_by=RUN_ID, ids=ids,
            )
            file_assignments = [
                labeler.assign(
                    features, registry, subject_id=features.subject_id,
                    run_id=RUN_ID, info_horizon=started_at,
                )
                for features in file_features
            ]
            touched: dict[str, list[str]] = {}
            for entry in index.snapshots:
                touched.setdefault(entry.commit_sha, []).append(entry.path)
            records = (*selection.selected_commits, *selection.boundary_commits)
            commit_assignments = [
                assign_commit(
                    record.message, touched.get(record.sha, []), registry,
                    subject_id=ids.commit(record.sha), run_id=RUN_ID,
                    info_horizon=started_at, ids=ids,
                )
                for record in records
            ]
            graph_path = str(EVAL_ROOT / "graph")
            graph_settings = config.graph.model_copy(update={"path": graph_path})
            sink = (
                GraphSink.open(graph_settings, run_id=RUN_ID)
                if Path(graph_path).exists()
                else GraphSink.create(graph_settings, run_id=RUN_ID, overwrite=True)
            )
            try:
                git_mapped = map_git_index(index, ids, run_id=RUN_ID)
                gh_mapped = map_discovery(discovery, ids, run_id=RUN_ID)
                folders_mapped = map_folders(
                    [snapshot.path for snapshot in content.snapshots],
                    ids, run_id=RUN_ID, window_start=config.repository.window.start,
                )
                branches_mapped = map_repository_branches(
                    [branch.name for branch in index.branches],
                    ids, run_id=RUN_ID, window_start=config.repository.window.start,
                )
                topics_mapped = map_registry(registry, ids, run_id=RUN_ID)
                about_mapped = map_topic_assignments(
                    [*file_assignments, *commit_assignments], ids, run_id=RUN_ID
                )
                classes_mapped = map_classes(content, index, ids, run_id=RUN_ID)
                imports_mapped = map_imports(content, index, ids, run_id=RUN_ID)
                all_entities = [
                    *git_mapped.entities, *gh_mapped.entities,
                    *folders_mapped.entities, *topics_mapped.entities,
                    *classes_mapped.entities,
                ]
                all_relationships = [
                    *git_mapped.relationships, *gh_mapped.relationships,
                    *folders_mapped.relationships, *branches_mapped.relationships,
                    *about_mapped.relationships, *classes_mapped.relationships,
                    *imports_mapped.relationships,
                ]
                entities = sink.write_entities(all_entities, analysis_run_id=RUN_ID)
                relationships = sink.write_relationships(all_relationships, analysis_run_id=RUN_ID)
                # Same-run duplicate ingestion must be fully skipped.
                dup_entities = sink.write_entities(all_entities, analysis_run_id=RUN_ID)
                dup_relationships = sink.write_relationships(all_relationships, analysis_run_id=RUN_ID)
                state["graph_counts"] = (
                    entities.entities_written,
                    relationships.relationships_written,
                    dup_entities.entities_duplicate,
                    dup_relationships.relationships_duplicate,
                    sink.label_counts(),
                )
            finally:
                sink.close()
        timings["graph"] = timer.seconds
        written, rels, dup_e, dup_r, _ = state["graph_counts"]
        print(
            f"graph: +{written}/+{rels} new, {dup_e}/{dup_r} duplicates on re-ingest",
            flush=True,
        )
        return (), ()

    runner.run_stage("graph", graph_stage)
    if "graph" not in ran:
        state["graph_counts"] = (0, 0, 0, 0, {})
        print("graph: stage skipped (resume), 0 writes", flush=True)

    llm_usage = {"requests": 0, "input": 0, "output": 0}
    if args.llm:
        import os

        if not os.environ.get("OPENAI_API_KEY"):
            notes.append("LLM path skipped: OPENAI_API_KEY is not set.")
        else:
            def llm_stage():
                ran.add("llm")
                with Timer() as timer:
                    settings = LLMSettings(
                        enabled=True, privacy_consent=True, provider="openai",
                        model="gpt-4.1-mini", temperature=0.0, max_attempts=2,
                        prompt_version="v1",
                        cache_file=str(EVAL_ROOT / "llm-cache.json"),
                    )
                    model = build_chat_model(settings)
                    enricher = Enricher(
                        model, settings, ledger=runner.ledger,
                        cache=EnrichmentCache(settings.cache_file), run_id=RUN_ID,
                    )
                    item = discovery.work_items[0]
                    outcome = enricher.enrich_summary(
                        subject_id=f"workitem:github:{item.number}",
                        title=item.title, text=item.body or "",
                        info_horizon=datetime.now(timezone.utc),
                    )
                timings["llm"] = timer.seconds
                state["llm_usage"] = {
                    "requests": 1,
                    "input": outcome.provenance.actual_input_tokens,
                    "output": outcome.provenance.actual_output_tokens,
                }
                state["llm_status"] = outcome.status.value
                return (), ()

            runner.run_stage("llm", llm_stage)
            if "llm" in ran:
                llm_usage = state["llm_usage"]
                notes.append(
                    f"LLM path exercised: summary {state['llm_status']} "
                    f"(actual {llm_usage['input']}/{llm_usage['output']} tokens)."
                )
                print(f"llm: {state['llm_status']} {llm_usage}", flush=True)
            else:
                notes.append("LLM stage resumed: cached proposal reused, no new spend.")
                print("llm: stage skipped (resume), 0 spend", flush=True)
    else:
        notes.append("LLM path not requested; pass --llm with OPENAI_API_KEY to exercise it.")

    finished_at = datetime.now(timezone.utc)
    gaps = [*selection.gaps, *index.gaps, *discovery.gaps, *result.gaps]
    manifest_out = coordinator.build_manifest(
        run_id=RUN_ID, started_at=started_at, finished_at=finished_at,
        selection=selection,
        selected_work_items=tuple(item.number for item in discovery.work_items),
        result=result, ledger=runner.ledger, gaps=gaps,
        config_fingerprint=config_fingerprint(config),
    )
    here = Path(__file__).resolve().parent
    (here / "run-manifest.json").write_text(
        manifest_out.model_dump_json(indent=1), encoding="utf-8"
    )

    written, rels, dup_e, dup_r, _ = state["graph_counts"]
    report = EvaluationReport(
        run_id=RUN_ID,
        repository=config.repository.full_name,
        synthetic=False,
        seconds={**timings, "total": time.perf_counter() - overall_start},
        counts={
            "commits": len(selection.selected_commits) + len(selection.boundary_commits),
            "snapshots": len(index.snapshots),
            "work_items": len(discovery.work_items),
            "events": len(result.ordered),
            "effective_events": len(result.effective),
            "graph_duplicates": dup_e,
        },
        budgets=summarize_ledger(runner.ledger),
        storage={
            "graph": storage_bytes(EVAL_ROOT / "graph"),
            "journal": storage_bytes(EVAL_ROOT / "events.jsonl"),
            "llm_cache": storage_bytes(EVAL_ROOT / "llm-cache.json"),
        },
        quality={},
        gaps=tuple(gaps),
        notes=(
            *notes,
            "Quality is unmeasured on the live slice: no accuracy is claimed.",
            f"Rerun resumes {runner.completed_stages}; completed stages spend nothing.",
        ),
    )
    (here / "evaluation.md").write_text(report.to_markdown(), encoding="utf-8")
    print(f"evaluation: {here / 'evaluation.md'}", flush=True)
    print(f"manifest: {here / 'run-manifest.json'}", flush=True)
    print(
        f"budgets: {summarize_ledger(runner.ledger)} "
        f"limits: {config.budgets.model_dump()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
