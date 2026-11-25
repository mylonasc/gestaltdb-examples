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
    PARENT_COMMIT,
    TIP_OF,
    TOUCHES,
    map_git_index,
    map_topic_assignments,
)
from repo_history.ids import IdFactory
from repo_history.labeling import (
    DeterministicLabeler,
    extract_snapshot_features,
    extract_work_item_features,
    seed_registry,
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
    registry = seed_registry(at=now)
    file_features = [
        extract_snapshot_features(
            snap, content.content_for(snap.snapshot_id), subject_id=snap.snapshot_id
        )
        for snap in content.snapshots
    ]
    for features in file_features:
        got = labeler.assign(
            features, registry, subject_id=features.subject_id,
            run_id=RUN_ID, info_horizon=now,
        )
        proposed = got.result.value["proposed_name"]
        if got.action == "propose" and registry.lookup(proposed) is None:
            registry, _ = registry.register(
                proposed, at=now, created_by=RUN_ID, ids=ids,
                description=features.text(),
            )
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
    assignments = [
        assignment
        for mapping in (item_assignments, file_assignments)
        for group in mapping.values()
        for assignment in group
    ]

    known_items = {
        item.number: ids.work_item("github", item.number) for item in discovery.work_items
    }
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
            *map_topic_assignments(assignments, ids, run_id=RUN_ID).relationships,
        ],
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
