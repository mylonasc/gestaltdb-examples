"""RH-11 runbook: answer the six user-visible questions on the live slice.

Builds the full pipeline from read-only evidence (local Git clone, cached
GitHub responses — no network, no model calls), writes
``examples/report.md`` and ``examples/slice.svg``, and prints a summary.
Re-running reproduces the same answers for the same evidence.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from repo_history.config import load_config
from repo_history.content import ContentExtractor
from repo_history.git import GitHistory
from repo_history.github import EvidenceCache, GitHubCredentials, GitHubDiscoverer
from repo_history.ids import IdFactory
from repo_history.inspection import Inspector
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
from repo_history.viz import MAX_NODES, render_slice_svg

HERE = Path(__file__).resolve().parent
RUN_ID = "runbook-live"


def main() -> None:
    now = datetime.now(timezone.utc)
    config = load_config()
    repo_url = config.repository.url
    ids = IdFactory(repo_url)

    history = GitHistory.open("/tmp/opencode/tf_gnns-rh03", repository_url=repo_url)
    selection = history.select(
        config.repository.anchor, config.repository.window,
        max_depth=config.git.max_depth,
    )
    index = history.index(selection)
    anchor_ids = {
        s.snapshot_id for s in index.snapshots if s.commit_sha == config.repository.anchor
    }
    subset = index.model_copy(update={"snapshots": tuple(
        s for s in index.snapshots if s.snapshot_id in anchor_ids)})
    content = ContentExtractor.from_history(history).extract(subset)

    cache = EvidenceCache(Path("/tmp/opencode/rh04-live-cache"), offline=True)
    discovery = GitHubDiscoverer(
        config.repository, config.github, cache=cache,
        credentials=GitHubCredentials.from_env(config.github),
    ).run()

    known_items = {
        item.number: ids.work_item("github", item.number) for item in discovery.work_items
    }
    known_commits = {
        c.sha for c in (*selection.selected_commits, *selection.boundary_commits)
    }
    for pull in discovery.pull_requests:
        known_commits.update(
            [pull.merge_commit_sha, pull.head_sha, pull.base_sha,
             *(c.sha for c in pull.commits)]
        )
    known_files = {s.path for s in content.snapshots}

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
        texts, ids=ids, items=known_items, commits=known_commits, files=known_files,
        repository=config.repository.full_name, run_id=RUN_ID, horizon=now,
    )
    provider_report = resolve_provider_links(
        discovery, index, ids=ids, run_id=RUN_ID, horizon=now
    )

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
                got.result.value["proposed_name"], at=now, created_by=RUN_ID,
                ids=ids, description=features.text(),
            )
    item_assignments, file_assignments = {}, {}
    for item in discovery.work_items:
        workflow = extract_work_item_features(item, subject_id=f"workitem:github:{item.number}")
        item_assignments[f"workitem:github:{item.number}"] = [
            labeler.assign(workflow, registry, subject_id=workflow.subject_id,
                           run_id=RUN_ID, info_horizon=now)
        ]
    for features in file_features:
        file_assignments[features.subject_id] = [
            labeler.assign(features, registry, subject_id=features.subject_id,
                           run_id=RUN_ID, info_horizon=now)
        ]
    topic_report = infer_topical_links(
        item_assignments, file_assignments, ids=ids, run_id=RUN_ID, horizon=now
    )
    links = [
        *text_report.links, *provider_report.links, *topic_report.links,
    ]
    assignments = [
        assignment
        for mapping in (item_assignments, file_assignments)
        for group in mapping.values()
        for assignment in group
    ]

    inspector = Inspector(
        index=index, discovery=discovery, content=content, links=links,
        assignments=assignments, info_horizon=now, run_id=RUN_ID,
        repository_url=repo_url,
    )

    # -- the six questions -------------------------------------------------
    ordered = sorted(selection.selected_commits, key=lambda c: c.committed_at)
    q1 = inspector.changes_in_range(ordered[0].sha, config.repository.anchor)
    merge_shas = [pull.merge_commit_sha for pull in discovery.pull_requests
                  if pull.merge_commit_sha]
    q2 = inspector.work_items_for_commits(
        [config.repository.anchor, *merge_shas]
    )
    q3_anchor = inspector.branches_containing(config.repository.anchor)
    # A positive control inside the bounded selection: the merged branch tip.
    q3_tip = inspector.branches_containing("0f05ec009fcc7e244b044273103ccc74546ee883")
    topic_counts: dict[str, int] = {}
    for assignment in assignments:
        if assignment.kind == "derived" and assignment.topic_id:
            topic_counts[assignment.topic_id] = topic_counts.get(assignment.topic_id, 0) + 1
    top_topic = max(topic_counts, key=lambda key: (topic_counts[key], key))
    q4 = inspector.topic_timeline(top_topic)
    q5 = inspector.files_for_work_item(11, now)
    explain_file = file_features[0].subject_id if file_features else ""
    q6_file = inspector.explain(explain_file)
    q6_item = inspector.explain(known_items[11])
    coverage = inspector.coverage()

    lines = [
        "# Repository-history inspection report",
        "",
        f"Run `{RUN_ID}` · horizon {now.isoformat()} · repository "
        f"`{config.repository.full_name}` @ `{config.repository.anchor[:8]}`.",
        "",
        f"Coverage: {coverage.counts} · "
        f"{len(coverage.gaps)} gaps "
        f"({', '.join(sorted({gap.code for gap in coverage.gaps})) or 'none'}).",
        "",
        "Factual links (provider, mentions, structural) are solid blue in the "
        "figure; inferred topical links are dashed amber.",
        "",
        "## Q1 — What changed across the commit range?",
        "",
        f"Range `({q1.from_sha[:8]}, {q1.to_sha[:8]}]`: "
        f"{len(q1.commits)} commits, {len(q1.changes)} file changes"
        + (" (truncated)" if q1.truncated else ""),
        "",
    ]
    for commit_summary in q1.commits[-8:]:
        lines.append(
            f"- `{commit_summary.sha[:8]}` {commit_summary.message.splitlines()[0][:80]} "
            f"([source]({commit_summary.source_url}))"
        )
    lines += ["", "## Q2 — Which work items explain those changes?", ""]
    for trace in q2:
        lines.append(
            f"- PR/issue #{trace.work_item_number} ({trace.work_item_kind}) via "
            f"**{trace.relation}**: {trace.title[:60]} — {trace.rationale[:100]}"
        )
    lines += ["", "## Q3 — Which observed branches contain these commits?", ""]
    anchor_branches = ", ".join(q3_anchor.branches)
    if not anchor_branches:
        anchor_branches = (
            "none: all observed tips postdate the anchor, and the bounded "
            "selection cannot see past its boundary parents"
        )
    lines.append(f"Anchor reachable from tips: {anchor_branches}.")
    lines.append(
        f"Merged tip `0f05ec00` reachable from: {', '.join(q3_tip.branches) or 'none'}."
    )
    lines += ["", "## Q4 — How did the top topic emerge or recur?", ""]
    lines.append(f"Top topic `{top_topic}` ({topic_counts[top_topic]} assignments):")
    for event in q4[:10]:
        lines.append(
            f"- {event.time.isoformat()} · `{event.subject_id[-24:]}` · {event.action} "
            f"conf={event.confidence:.2f}"
        )
    lines += ["", "## Q5 — Which files relate to work item #11 now?", ""]
    for related in q5[:12]:
        sections = ", ".join(related.sections[:4]) or "no headings"
        lines.append(
            f"- `{related.path}` via {related.relation} conf={related.confidence:.2f} "
            f"· sections: {sections}"
        )
    lines += ["", "## Q6 — Why was an entity labeled or connected?", ""]
    lines.append(f"Explanations for `{explain_file[-32:]}`:")
    for explanation in q6_file[:6]:
        lines.append(
            f"- **{explanation.kind}** {explanation.summary[:60]} — "
            f"{explanation.rationale[:120]} (method `{explanation.method}`)"
        )
    lines.append(f"Explanations for work item #11:")
    for explanation in q6_item[:6]:
        lines.append(
            f"- **{explanation.kind}** {explanation.summary[:60]} — "
            f"{explanation.rationale[:120]} (method `{explanation.method}`)"
        )
    report = "\n".join(lines) + "\n"
    (HERE / "report.md").write_text(report, encoding="utf-8")

    # -- figure --------------------------------------------------------------
    file_link_counts: dict[str, int] = {}
    for record in topic_report.links:
        file_link_counts[record.target_id] = file_link_counts.get(record.target_id, 0) + 1
    top_files = sorted(file_link_counts, key=lambda key: (-file_link_counts[key], key))[:12]
    path_of = {ids.file(snap.path): snap.path for snap in content.snapshots}
    topic_names = {
        assignment.topic_id: assignment.topic_name or assignment.topic_id
        for group in list(item_assignments.values()) + list(file_assignments.values())
        for assignment in group
        if assignment.topic_id
    }
    svg_links = []
    for record in links:
        if record.relation.value in ("MEMBER_OF", "OF_FILE", "HAS_LABEL", "CHILD_OF", "TIP_OF"):
            continue
        svg_links.append({
            "source": record.source_id, "target": record.target_id,
            "relation": record.relation.value,
            "inferred": record.method == "link-inference",
        })
    shown_files = [fid for fid in top_files if any(
        link["source"] == fid or link["target"] == fid for link in svg_links
    )]
    figure = render_slice_svg(
        commits=[
            {"id": ids.commit(c.sha), "label": f"{c.sha[:8]} {c.message.splitlines()[0][:18]}",
             "sort": c.committed_at.isoformat(), "title": c.message}
            for c in ordered
        ],
        work_items=[
            {"id": known_items[item.number], "label": f"#{item.number} {item.title[:18]}",
             "title": item.title}
            for item in discovery.work_items
        ],
        topics=[
            {"id": topic_id, "label": topic_names.get(topic_id, topic_id)[-24:]}
            for topic_id in sorted(topic_names)[:10]
        ],
        files=[
            {"id": fid, "label": path_of.get(fid, fid)[-24:]} for fid in shown_files
        ],
        links=svg_links,
        meta={
            "title": f"tf_gnns slice @ {config.repository.anchor[:8]}",
            "run_id": RUN_ID, "horizon": now.isoformat(),
            "gaps": len(coverage.gaps),
        },
    )
    (HERE / "slice.svg").write_text(figure, encoding="utf-8")

    print(f"report: {HERE / 'report.md'}")
    print(f"figure: {HERE / 'slice.svg'}")
    print(f"commits={len(ordered)} items={len(discovery.work_items)} "
          f"topics={len(topic_names)} links={len(links)} "
          f"gaps={len(coverage.gaps)}")


if __name__ == "__main__":
    main()
