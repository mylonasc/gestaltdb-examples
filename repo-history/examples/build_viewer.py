"""Build the interactive D3 viewer for the live tf_gnns slice.

Reads the read-only Git clone plus cached GitHub evidence (offline by
default), runs deterministic labeling and linking, and writes
``examples/graph.json`` (embedded data) plus ``examples/viewer.html``
(self-contained page, D3 loaded from CDN with an offline table fallback).
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from repo_history.config import load_config
from repo_history.content import ContentExtractor
from repo_history.git import GitHistory
from repo_history.github import EvidenceCache, GitHubCredentials, GitHubDiscoverer
from repo_history.graph import (
    CHILD_OF,
    PARENT_COMMIT,
    TIP_OF,
    TOUCHES,
    map_discovery,
    map_folders,
    map_git_index,
    map_imports,
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
    topic_membership,
)
from repo_history.linking import (
    infer_topical_links,
    resolve_provider_links,
    resolve_textual_links,
)
from repo_history.viz_d3 import build_viewer_data, render_viewer_html

HERE = Path(__file__).resolve().parent
RUN_ID = "viewer-live"


def main() -> None:
    now = datetime.now(timezone.utc)
    config = load_config()
    repo_url = config.repository.url
    ids = IdFactory(repo_url)

    history = GitHistory.open("/tmp/opencode/tf_gnns-rh03", repository_url=repo_url)
    try:
        selection = history.select(
            config.repository.anchor, config.repository.window,
            max_depth=config.git.max_depth,
        )
        index = history.index(selection)
    finally:
        history.close()

    cache = EvidenceCache(Path("/tmp/opencode/rh04-live-cache"), offline=True)
    discovery = GitHubDiscoverer(
        config.repository, config.github, cache=cache,
        credentials=GitHubCredentials.from_env(config.github),
    ).run()

    history = GitHistory.open("/tmp/opencode/tf_gnns-rh03", repository_url=repo_url)
    try:
        anchor_ids = {
            s.snapshot_id for s in index.snapshots
            if s.commit_sha == config.repository.anchor
        }
        subset = index.model_copy(update={"snapshots": tuple(
            s for s in index.snapshots if s.snapshot_id in anchor_ids)})
        content = ContentExtractor.from_history(history).extract(subset)
    finally:
        history.close()

    labeler = DeterministicLabeler(config.labeling)
    # Seed prevalent aspects first: repository layout plus commit types.
    # Per-subject proposals below can only register when shared.
    all_paths = sorted({snapshot.path for snapshot in content.snapshots})
    registry = seed_structural_registry(
        all_paths, at=now, created_by=RUN_ID, ids=ids
    )
    registry = seed_commit_type_registry(
        at=now, created_by=RUN_ID, ids=ids, base=registry
    )
    known_items = {
        item.number: ids.work_item("github", item.number) for item in discovery.work_items
    }
    file_features = [
        extract_snapshot_features(
            snap, content.content_for(snap.snapshot_id), subject_id=snap.snapshot_id
        )
        for snap in content.snapshots
    ]
    # Pass 1: collect proposals without registering singletons.
    proposals: dict[str, list[str]] = {}
    for features in file_features:
        got = labeler.assign(
            features, registry, subject_id=features.subject_id,
            run_id=RUN_ID, info_horizon=now,
        )
        name = got.result.value.get("proposed_name")
        if got.action == "propose" and name and registry.lookup(name) is None:
            proposals.setdefault(name, []).append(features.text())
    registry, _ = register_prevalent(
        registry, proposals, min_subjects=2, at=now, created_by=RUN_ID, ids=ids
    )
    singletons = sum(1 for texts in proposals.values() if len(set(texts)) < 2)
    # Pass 2: decide against structural, commit-type, and prevalent topics.
    item_assignments, file_assignments = {}, {}
    for item in discovery.work_items:
        workflow = extract_work_item_features(item, subject_id=f"workitem:github:{item.number}")
        item_assignments[workflow.subject_id] = [
            labeler.assign(workflow, registry, subject_id=workflow.subject_id,
                           run_id=RUN_ID, info_horizon=now)
        ]
    for features in file_features:
        file_assignments[features.subject_id] = [
            labeler.assign(features, registry, subject_id=features.subject_id,
                           run_id=RUN_ID, info_horizon=now)
        ]
    touched: dict[str, list[str]] = {}
    for entry in index.snapshots:
        touched.setdefault(entry.commit_sha, []).append(entry.path)
    commit_assignments = []
    for record in (*selection.selected_commits, *selection.boundary_commits):
        commit_assignments.append(
            assign_commit(
                record.message, touched.get(record.sha, []), registry,
                subject_id=ids.commit(record.sha), run_id=RUN_ID, info_horizon=now,
                ids=ids,
            )
        )
    # Work items get the same deterministic commit-type topics as commits
    # (title plus body as the message); TF-IDF assignments above still drive
    # item-file topical links.
    item_type_assignments = []
    for item in discovery.work_items:
        item_type_assignments.append(
            assign_commit(
                f"{item.title}\n{item.body or ''}", [], registry,
                subject_id=known_items[item.number], run_id=RUN_ID,
                info_horizon=now, ids=ids,
            )
        )
    assignments = [
        assignment
        for mapping in (item_assignments, file_assignments)
        for group in mapping.values()
        for assignment in group
    ] + commit_assignments + item_type_assignments
    membership = topic_membership(assignments)
    prevalent = {t: s for t, s in membership.items() if len(s) >= 2}
    print(
        f"topics: {len(registry.topics)} registered, "
        f"{len(prevalent)} prevalent (>{1} subjects), "
        f"{singletons} singleton proposals left unregistered",
        flush=True,
    )

    known_commits = {
        record.sha for record in (*selection.selected_commits, *selection.boundary_commits)
    }
    for pull in discovery.pull_requests:
        known_commits.update(
            [pull.merge_commit_sha, pull.head_sha, pull.base_sha,
             *(commit.sha for commit in pull.commits)]
        )
    texts = []
    for item in discovery.work_items:
        texts.append((
            known_items[item.number], item.title, f"issue:{item.number}:title", item.url,
        ))
        if item.body:
            texts.append((
                known_items[item.number], item.body, f"issue:{item.number}:body", item.url,
            ))
    for record in (*selection.selected_commits, *selection.boundary_commits):
        texts.append((
            ids.commit(record.sha), record.message,
            f"commit:{record.sha}", record.source_url,
        ))
    text_report = resolve_textual_links(
        texts, ids=ids, items=known_items, commits=known_commits,
        files={snap.path for snap in content.snapshots},
        repository=config.repository.full_name, run_id=RUN_ID, horizon=now,
    )
    provider_report = resolve_provider_links(
        discovery, index, ids=ids, run_id=RUN_ID, horizon=now
    )
    topic_report = infer_topical_links(
        item_assignments, file_assignments, ids=ids, run_id=RUN_ID, horizon=now
    )
    links = [*text_report.links, *provider_report.links, *topic_report.links]

    folders = map_folders(
        [snap.path for snap in content.snapshots], ids, run_id=RUN_ID,
        window_start=config.repository.window.start,
    )
    has_branches = map_repository_branches(
        [branch.name for branch in index.branches], ids, run_id=RUN_ID,
        window_start=config.repository.window.start,
    )
    data = build_viewer_data(
        index=index, discovery=discovery, content=content,
        links=[*text_report.links, *provider_report.links, *topic_report.links],
        assignments=assignments, info_horizon=now, run_id=RUN_ID,
        repository_url=repo_url,
        max_nodes=400,
        branch_names={
            ids.branch(pull.head_ref): pull.head_ref
            for pull in discovery.pull_requests if pull.head_ref
        } | {
            ids.branch(pull.base_ref): pull.base_ref
            for pull in discovery.pull_requests if pull.base_ref
        },
        extra_edges=[
            *(r for r in map_git_index(index, ids, run_id=RUN_ID).relationships
              if r.edge_type in (PARENT_COMMIT, TIP_OF, TOUCHES)),
            *(r for r in map_discovery(discovery, ids, run_id=RUN_ID).relationships
              if r.edge_type == CHILD_OF),
            *map_imports(content, index, ids, run_id=RUN_ID).relationships,
            *map_topic_assignments(assignments, ids, run_id=RUN_ID).relationships,
            *folders.relationships,
            *has_branches.relationships,
        ],
        extra_nodes=[*folders.entities],
    )
    (HERE / "graph.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    (HERE / "viewer.html").write_text(
        render_viewer_html(data, title=f"tf_gnns slice @ {config.repository.anchor[:8]}"),
        encoding="utf-8",
    )
    print(f"nodes={len(data['nodes'])} edges={len(data['edges'])} "
          f"omitted={data['meta']['counts'].get('omitted', 0)}")
    print(f"wrote {HERE / 'graph.json'} and {HERE / 'viewer.html'}")


if __name__ == "__main__":
    main()
