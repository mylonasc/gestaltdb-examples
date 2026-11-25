"""Interactive D3 force-directed viewer data and page.

:func:`build_viewer_data` turns adapter outputs into a bounded, time-annotated
node/edge document: every node with a known timestamp gets ``t`` in ``[0, 1]``
normalized across the exported nodes, and nodes without timestamps get
``t = None``.  :func:`render_viewer_html` embeds that document in a
self-contained page: a force-directed graph whose x-axis is time-biased (an
anchor pulls every node toward the mid-time center while a second force pulls
nodes left/right with a strength that grows with their distance from
mid-time), with kind/edge filters, search, labels, and a details panel.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Sequence

from .content import ContentIndex, detect_language
from .git import GitIndex
from .github import GitHubDiscovery
from .graph import HIERARCHICAL_EDGE_TYPES, EntityWrite, RelationshipWrite
from .ids import IdFactory
from .labeling import TopicAssignment
from .linking import LinkRecord, LinkRelation

__all__ = [
    "D3_CDN_URL",
    "MAX_VIEWER_NODES",
    "build_viewer_data",
    "render_viewer_html",
]

D3_CDN_URL = "https://cdn.jsdelivr.net/npm/d3@7"
MAX_VIEWER_NODES = 120


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def build_viewer_data(
    *,
    index: GitIndex,
    discovery: GitHubDiscovery,
    content: ContentIndex,
    links: Sequence[LinkRecord],
    assignments: Sequence[TopicAssignment],
    info_horizon: datetime,
    run_id: str,
    repository_url: str,
    max_nodes: int = MAX_VIEWER_NODES,
    branch_names: dict[str, str] | None = None,
    extra_edges: Sequence[RelationshipWrite] = (),
    extra_nodes: Sequence[EntityWrite] = (),
) -> dict[str, Any]:
    """Build a bounded, time-normalized node/edge document for the viewer.

    ``branch_names`` maps branch node IDs to ref names for branches that are
    referenced by links but absent from the index (deleted or out of slice);
    they appear as unobserved placeholder nodes instead of dropping the edge.
    ``extra_edges`` carries indexer writes (structural and ``ABOUT`` edges)
    that are not ``LinkRecord`` reports; endpoints are canonicalized the same
    way and edges to dropped nodes are skipped.  ``ABOUT`` edges land on
    files: snapshot occurrences aggregate to their file for the file-level
    view (noted in ``meta``).  Structural ``extra_nodes`` (folders, the
    repository root) are added verbatim; folder timestamps follow their
    contained files while the repository stays timeless.
    """

    ids = IdFactory(repository_url)
    nodes: dict[str, dict[str, Any]] = {}
    times: dict[str, datetime] = {}

    def add(node_id: str, kind: str, label: str, **extra: Any) -> None:
        if node_id not in nodes:
            nodes[node_id] = {"id": node_id, "kind": kind, "label": label, **extra}

    commit_times = {
        record.sha: record.committed_at
        for record in (*index.selection.selected_commits, *index.selection.boundary_commits)
    }
    for record in (*index.selection.selected_commits, *index.selection.boundary_commits):
        node_id = ids.commit(record.sha)
        times[node_id] = record.committed_at
        add(
            node_id, "Commit", f"{record.sha[:8]} {record.message.splitlines()[0][:30]}",
            title=record.message, time=_iso(record.committed_at),
            props={"sha": record.sha, "message": record.message.splitlines()[0][:120],
                   "committed_at": _iso(record.committed_at), "is_merge": record.is_merge,
                   "url": record.source_url},
            prov={"method": "git-history", "run_id": run_id,
                  "horizon": info_horizon.isoformat()},
        )

    for item in discovery.work_items:
        node_id = ids.work_item("github", item.number)
        times[node_id] = item.created_at
        add(
            node_id, "WorkItem", f"#{item.number} {item.title[:34]}",
            title=item.title, time=_iso(item.created_at),
            props={"number": item.number, "kind": item.kind, "state": item.state,
                   "url": item.url},
            prov={"method": "github-discovery", "run_id": run_id,
                  "horizon": info_horizon.isoformat()},
        )

    for branch in index.branches:
        node_id = ids.branch(branch.name)
        tip_time = commit_times.get(branch.tip_sha)
        if tip_time is not None:
            times[node_id] = tip_time
        add(
            node_id, "Branch", branch.name, title=f"{branch.ref} @ {branch.tip_sha[:8]}",
            time=_iso(tip_time),
            props={"name": branch.name, "ref": branch.ref, "tip_sha": branch.tip_sha,
                   "source": branch.source},
            prov={"method": "git-history", "run_id": run_id,
                  "horizon": info_horizon.isoformat()},
        )

    topic_names: dict[str, str] = {}
    topic_moments: dict[str, list[datetime]] = {}
    subject_time: dict[str, datetime] = {
        snapshot.snapshot_id: commit_times[snapshot.commit_sha]
        for snapshot in content.snapshots
        if snapshot.commit_sha in commit_times
    }
    for item in discovery.work_items:
        subject_time[f"workitem:github:{item.number}"] = item.created_at
        subject_time[ids.work_item("github", item.number)] = item.created_at
    for sha, committed_at in commit_times.items():
        subject_time[sha] = committed_at
        subject_time[ids.commit(sha)] = committed_at
    for assignment in assignments:
        if assignment.kind != "derived" or assignment.topic_id is None:
            continue
        topic_names.setdefault(
            assignment.topic_id, assignment.topic_name or assignment.topic_id
        )
        moment = subject_time.get(assignment.subject_id)
        topic_moments.setdefault(assignment.topic_id, []).append(
            moment if moment is not None else assignment.result.provenance.info_horizon
        )
    for topic_id, name in topic_names.items():
        first_seen = min(topic_moments[topic_id])
        times[topic_id] = first_seen
        add(
            topic_id, "Topic", name[-30:], title=name, time=_iso(first_seen),
            props={"topic_id": topic_id},
            prov={"method": "deterministic-tfidf-labeler", "run_id": run_id,
                  "horizon": info_horizon.isoformat()},
        )

    # Links and indexer writes reference subjects by the IDs their own layer
    # uses (snapshot occurrences, literal ``workitem:github:N`` subjects);
    # canonicalize both to the graph node IDs so edges join kept nodes.
    canonical_id = {
        snapshot.snapshot_id: ids.file(snapshot.path) for snapshot in content.snapshots
    }
    canonical_id.update({
        f"workitem:github:{item.number}": ids.work_item("github", item.number)
        for item in discovery.work_items
    })

    def canonical_endpoint(endpoint: str) -> str:
        return canonical_id.get(endpoint, endpoint)

    aggregated: list[LinkRecord] = []
    for link in links:
        source = canonical_endpoint(link.source_id)
        target = canonical_endpoint(link.target_id)
        if (source, target) == (link.source_id, link.target_id):
            aggregated.append(link)
        else:
            aggregated.append(
                link.model_copy(update={"source_id": source, "target_id": target})
            )
    links = aggregated
    extra: list[dict[str, Any]] = []
    for write in extra_edges:
        source = canonical_endpoint(write.source)
        target = canonical_endpoint(write.target)
        hierarchical = bool(write.properties.get("hierarchical", False))
        if not hierarchical and write.edge_type in HIERARCHICAL_EDGE_TYPES:
            hierarchical = True
        extra.append({
            "source": source,
            "target": target,
            "relation": write.edge_type,
            "hierarchical": hierarchical,
            "inferred": write.edge_type == "RELATED_TO" and write.origin == "linking",
            "rationale": str(write.properties.get("rationale", write.edge_type)),
            "evidence": len(write.evidence_ids),
        })
    # Commit-to-file touches stay aggregated at file level: the index keeps
    # per-commit diffs, but the viewer draws one edge per touched file.
    for snapshot in index.snapshots:
        if snapshot.commit_sha not in commit_times:
            continue
        extra.append({
            "source": canonical_endpoint(ids.commit(snapshot.commit_sha)),
            "target": canonical_endpoint(ids.file(snapshot.path)),
            "relation": "TOUCHES",
            "hierarchical": True,
            "inferred": False,
            "rationale": "commit touches file (diffs folded)",
            "evidence": 0,
        })

    path_of_id = {
        ids.file(snapshot.path): snapshot.path for snapshot in content.snapshots
    }
    file_targets = {
        link.target_id
        for link in links
        if link.relation in (LinkRelation.MENTIONS, LinkRelation.RELATED_TO)
        and link.target_id in path_of_id
    }
    for edge in extra:
        if edge["target"] in path_of_id:
            file_targets.add(edge["target"])
        if edge["source"] in path_of_id:
            file_targets.add(edge["source"])
    file_times: dict[str, list[datetime]] = {}
    for snapshot in content.snapshots:
        moment = commit_times.get(snapshot.commit_sha)
        if moment is not None:
            file_times.setdefault(ids.file(snapshot.path), []).append(moment)
    for file_id in sorted(file_targets):
        path = path_of_id.get(file_id)
        if path is None:
            continue
        moments = file_times.get(file_id, [])
        first_seen = min(moments) if moments else None
        if first_seen is not None:
            times[file_id] = first_seen
        language = detect_language(path)
        history = sorted(
            {
                (commit_times[entry.commit_sha], entry.commit_sha, entry.content_status)
                for entry in index.snapshots
                if entry.path == path and entry.commit_sha in commit_times
            }
        )
        preview = ""
        candidates = [
            entry for entry in index.snapshots if entry.path == path
        ] or [snap for snap in content.snapshots if snap.path == path]
        latest = max(
            candidates,
            key=lambda entry: commit_times.get(entry.commit_sha, datetime.min.replace(tzinfo=timezone.utc)),
            default=None,
        )
        definition_url = latest.source_url if latest is not None else None
        if latest is not None:
            snapshot_id = getattr(latest, "snapshot_id", None)
            parsed = content.content_for(snapshot_id) if snapshot_id else None
            if parsed is not None and parsed.text:
                preview = "\n".join(parsed.text.splitlines()[:30])[:1500]
        basename = path.rsplit("/", 1)[-1]
        add(
            file_id, "File", basename[-34:], title=path, time=_iso(first_seen),
            language=language,
            props={
                "path": path,
                "language": language,
                "definition_url": definition_url,
                "history": [
                    {"at": moment.isoformat(), "commit": sha, "status": status}
                    for moment, sha, status in history
                ],
                "preview": preview,
            },
            prov={"method": "linking", "run_id": run_id,
                  "horizon": info_horizon.isoformat()},
        )

    # Referenced but unobserved branches become placeholders, never drops.
    # The caller-supplied name map is the explicit guard: only mapped IDs
    # get placeholder nodes.
    known_branches = dict(branch_names or {})
    for link in links:
        if (
            link.relation is LinkRelation.RELATED_TO
            and link.target_id not in nodes
            and link.target_id in known_branches
        ):
            ref = known_branches[link.target_id]
            add(
                link.target_id, "Branch", f"{ref} ⚠", title=f"{ref} (unobserved tip)",
                time=None,
                props={
                    "name": ref, "source": "unobserved",
                    "status": "deleted or outside slice",
                },
                prov={"method": "link-provider", "run_id": run_id,
                      "horizon": info_horizon.isoformat()},
            )
    # Structural nodes from the indexer (folders, repository root).
    folder_moments: dict[str, list[datetime]] = {}
    for snapshot in content.snapshots:
        moment = commit_times.get(snapshot.commit_sha)
        if moment is None:
            continue
        parts = snapshot.path.split("/")[:-1]
        for depth in range(1, len(parts) + 1):
            folder_moments.setdefault("/".join(parts[:depth]), []).append(moment)
    for write in extra_nodes:
        if write.logical_id in nodes:
            continue
        kind = write.labels[0] if write.labels else "Unknown"
        moment = None
        if kind == "Folder":
            path = str(write.properties.get("path", ""))
            moments = folder_moments.get(path, [])
            if moments:
                moment = min(moments)
        elif kind != "Repository" and isinstance(write.valid_from, datetime):
            moment = write.valid_from
        if moment is not None:
            times[write.logical_id] = moment
        if kind == "Class":
            label = str(write.properties.get("name", write.logical_id))
            title = str(write.properties.get("qualname", label))
        else:
            label = str(write.properties.get("path", write.properties.get("name", write.logical_id)))
            title = label
        add(
            write.logical_id, kind, label[-34:], title=title, time=_iso(moment),
            props=dict(write.properties),
            prov={"method": write.origin, "run_id": run_id,
                  "horizon": info_horizon.isoformat()},
        )
    known = sorted(times.values())
    if len(known) >= 2 and known[0] != known[-1]:
        span = (known[-1] - known[0]).total_seconds()
        normalized = {
            nid: (moment - known[0]).total_seconds() / span
            for nid, moment in times.items()
        }
    else:
        normalized = {nid: 0.5 for nid in times}
    for node_id, node in nodes.items():
        node["t"] = normalized.get(node_id)

    degree: dict[str, int] = {}
    for link in links:
        if link.source_id in nodes and link.target_id in nodes:
            degree[link.source_id] = degree.get(link.source_id, 0) + 1
            degree[link.target_id] = degree.get(link.target_id, 0) + 1
    for edge in extra:
        if edge["source"] in nodes and edge["target"] in nodes:
            degree[edge["source"]] = degree.get(edge["source"], 0) + 1
            degree[edge["target"]] = degree.get(edge["target"], 0) + 1

    core_kinds = {"Commit", "WorkItem", "Branch"}
    core = [nid for nid, node in nodes.items() if node["kind"] in core_kinds]
    rest = sorted(
        (nid for nid, node in nodes.items() if node["kind"] not in core_kinds),
        key=lambda nid: (-degree.get(nid, 0), nid),
    )
    kept = set(core)
    omitted = 0
    for nid in rest:
        if len(kept) < max_nodes:
            kept.add(nid)
        else:
            omitted += 1
    # If even the core exceeds the budget, trim deterministically by id.
    while len(kept) > max_nodes:
        kept.discard(sorted(kept)[-1])
        omitted += 1

    edges = [
        {
            "source": link.source_id,
            "target": link.target_id,
            "relation": link.relation.value,
            "hierarchical": False,
            "inferred": link.relation is LinkRelation.RELATED_TO
            and link.method == "link-inference",
            "rationale": link.rationale[:200],
            "evidence": len(link.evidence),
        }
        for link in links
        if link.source_id in kept and link.target_id in kept
    ]
    edges.extend(
        edge for edge in extra if edge["source"] in kept and edge["target"] in kept
    )
    repository_bits = repository_url.rstrip("/").split("/")
    repository_name = (
        "/".join(repository_bits[-2:]) if len(repository_bits) >= 2 else repository_url
    )
    if repository_name.endswith(".git"):
        repository_name = repository_name[:-4]
    kept_nodes = [nodes[nid] for nid in sorted(kept)]
    counts = {"links": len(edges), "omitted": omitted}
    for node in kept_nodes:
        counts[node["kind"]] = counts.get(node["kind"], 0) + 1
    window = index.selection.window
    return {
        "nodes": kept_nodes,
        "edges": sorted(edges, key=lambda e: (e["source"], e["target"], e["relation"])),
        "meta": {
            "title": "Repository slice",
            "run_id": run_id,
            "horizon": info_horizon.isoformat(),
            "window": {"start": window.start.isoformat(), "end": window.end.isoformat()},
            "counts": counts,
            "max_nodes": max_nodes,
            "repository": {"name": repository_name, "url": repository_url},
            "notes": (
                "Snapshot occurrences aggregate to file nodes; "
                "MEMBER_OF/OF_FILE snapshot granularity is folded into TOUCHES. "
                "Folder nodes group files by directory; the repository node roots branches."
            ),
        },
    }


_VIEWER_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ — temporal repository graph</title>
<style>
body { font-family: sans-serif; margin: 0; display: flex; height: 100vh; }
#main { flex: 1; position: relative; min-width: 0; }
#panel { width: 340px; border-left: 1px solid #ccc; padding: 12px; overflow-y: auto; font-size: 13px; }
#panel h2 { font-size: 15px; margin: 0 0 6px; }
#panel h3 { font-size: 13px; margin: 12px 0 4px; }
.row { margin: 4px 0; }
label.kind { display: inline-block; margin-right: 10px; }
#tip { position: absolute; pointer-events: none; background: #fff; border: 1px solid #999; padding: 6px 8px; font-size: 12px; display: none; max-width: 320px; }
#fallback { padding: 12px; font-size: 13px; }
table { border-collapse: collapse; } td, th { border: 1px solid #ccc; padding: 2px 6px; font-size: 12px; }
input[type=text] { width: 180px; }
button { margin: 2px; }
.dim { opacity: 0.08; }
</style>
</head>
<body>
<div id="main"><svg id="graph" width="100%" height="100%"></svg><div id="tip"></div></div>
<div id="panel">
<h2>Temporal repository graph</h2>
<div class="row" id="stats"></div>
<h3>Node kinds</h3><div class="row" id="kindFilters"></div>
<h3>Edge types</h3><div class="row" id="edgeFilters"></div>
<h3>Time window</h3><div class="row"><input type="range" id="tmin" min="0" max="100" value="0"> <input type="range" id="tmax" min="0" max="100" value="100"> <span id="trange"></span></div>
<h3>Forces</h3>
<div class="row">anchor pull <input type="range" id="anchor" min="0" max="30" value="6"></div>
<div class="row">time spread <input type="range" id="spread" min="0" max="40" value="14"></div>
<div class="row">link distance <input type="range" id="ldist" min="20" max="240" value="90"></div>
<div class="row">link pull <input type="range" id="lpull" min="5" max="100" value="25"></div>
<div class="row">hierarchical pull ×<input type="range" id="hier" min="100" max="300" value="160"></div>
<div class="row">repulsion <input type="range" id="charge" min="0" max="400" value="140"></div>
<div class="row">hierarchy pull <input type="range" id="rank" min="0" max="40" value="12"></div>
<div class="row"><label><input type="checkbox" id="labels" checked> labels</label>
<label><input type="checkbox" id="bytime"> color by time</label>
<label><input type="checkbox" id="filecolors" checked> color files by type</label></div>
<h3>Search</h3><div class="row"><input type="text" id="search" placeholder="label substring"><button id="find">Find</button></div>
<div class="row"><button id="pause">Pause</button><button id="reset">Reset view</button></div>
<h3>Details</h3><div class="row" id="repolink"></div><div id="details">Click a node to inspect its relations.</div>
<h3>Legend</h3><div class="row" id="legend"></div>
</div>
<div id="fallback" style="display:none"></div>
<script src="__D3_URL__"></script>
<script>
const DATA = __DATA__;
(function () {
  if (!window.d3) {
    document.getElementById('graph').style.display = 'none';
    const fb = document.getElementById('fallback');
    fb.style.display = 'block';
    let html = '<p>D3 failed to load (offline?). Raw embedded data:</p><h3>Nodes</h3><table><tr><th>id</th><th>kind</th><th>label</th><th>t</th></tr>';
    for (const n of DATA.nodes.slice(0, 200)) html += '<tr><td>' + n.id + '</td><td>' + n.kind + '</td><td>' + n.label + '</td><td>' + n.t + '</td></tr>';
    html += '</table><h3>Edges</h3><table><tr><th>source</th><th>relation</th><th>target</th></tr>';
    for (const e of DATA.edges.slice(0, 200)) html += '<tr><td>' + e.source + '</td><td>' + e.relation + '</td><td>' + e.target + '</td></tr>';
    html += '</table>';
    fb.innerHTML = html;
    return;
  }
  const W = 1200, H = 640, CX = W / 2, CY = H / 2;
  const kinds = [...new Set(DATA.nodes.map(n => n.kind))];
  const relations = [...new Set(DATA.edges.map(e => e.relation))];
  const kindColor = d3.scaleOrdinal(d3.schemeTableau10).domain(kinds);
  const timeColor = d3.scaleSequential(d3.interpolateTurbo).domain([0, 1]);
  const svg = d3.select('#graph').attr('viewBox', `0 0 ${W} ${H}`);
  const g = svg.append('g');
  svg.call(d3.zoom().on('zoom', (ev) => g.attr('transform', ev.transform)));
  document.getElementById('stats').textContent =
    `${DATA.nodes.length} nodes, ${DATA.edges.length} edges shown (+${DATA.meta.counts.omitted || 0} omitted by budget ${DATA.meta.max_nodes}). Horizon ${DATA.meta.horizon}.`;

  // --- temporal forces: anchor pulls every x to mid-time center; ---
  // --- spread pulls left/right with strength growing from mid-time. ---
  const P = { anchor: 0.06, base: 0.015, far: 0.14, gamma: 1.0, link: 0.25, hierMult: 1.6, rank: 0.12 };
  const TIERS = { Repository: 0, Branch: 1, WorkItem: 2, Commit: 3, Topic: 4, Folder: 5, File: 6, Class: 7 };
  function temporalForce() {
    let nodes = [];
    function force(alpha) {
      for (const n of nodes) {
        if (n.t == null) continue;
        const d = n.t - 0.5;
        const target = CX + Math.sign(d) * Math.pow(Math.abs(d), P.gamma) * (W / 2 - 90);
        n.vx += (target - n.x) * (P.base + P.far * Math.abs(d)) * alpha;
      }
    }
    force.initialize = (_) => { nodes = _; };
    return force;
  }
  const tForce = temporalForce();
  const maxTier = Math.max(...Object.values(TIERS));
  function rankY(d) { const t = TIERS[d.kind] === undefined ? maxTier : TIERS[d.kind]; return 70 + t * ((H - 140) / maxTier); }


  const sim = d3.forceSimulation(DATA.nodes)
    .force('link', d3.forceLink(DATA.edges).id(d => d.id).distance(90)
      .strength(e => e.hierarchical ? P.link * P.hierMult : P.link))
    .force('charge', d3.forceManyBody().strength(-140))
    .force('collide', d3.forceCollide().radius(22))
    .force('anchor', d3.forceX(CX).strength(P.anchor))
    .force('spread', tForce)
    .force('rank', d3.forceY(d => rankY(d)).strength(P.rank))
    .force('yc', d3.forceY(CY).strength(0.06));
  DATA.nodes.forEach((n, i) => {
    n.x = n.t == null ? CX + ((i % 9) - 4) * 8 : CX + (n.t - 0.5) * W * 0.7;
    n.y = CY + ((i % 7) - 3) * 26;
  });

  const link = g.append('g').selectAll('line').data(DATA.edges).join('line')
    .attr('stroke', e => e.inferred ? '#9e6a03' : (['PARENT_COMMIT', 'MEMBER_OF', 'OF_FILE', 'TIP_OF', 'TOUCHES', 'BELONGS_TO', 'HAS_BRANCH', 'HAS_FOLDER', 'DEFINED_AT'].includes(e.relation) ? '#8b949e' : '#1f6feb'))
    .attr('stroke-width', e => ['PARENT_COMMIT', 'TIP_OF', 'TOUCHES', 'BELONGS_TO'].includes(e.relation) ? 1 : 2)
    .attr('stroke-dasharray', e => e.inferred ? '5,3' : null);
  link.append('title').text(e => e.relation + (e.rationale ? ': ' + e.rationale : ''));
  const node = g.append('g').selectAll('g').data(DATA.nodes).join('g')
    .attr('class', 'node').call(d3.drag()
      .on('start', (ev, d) => { if (!ev.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
      .on('drag', (ev, d) => { d.fx = ev.x; d.fy = ev.y; })
      .on('end', (ev, d) => { if (!ev.active) sim.alphaTarget(0); d.fx = null; d.fy = null; }))
    .on('click', (ev, d) => { ev.stopPropagation(); select(d); })
    .on('mouseover', (ev, d) => {
      const tip = document.getElementById('tip');
      tip.style.display = 'block'; tip.style.left = (ev.offsetX + 14) + 'px'; tip.style.top = (ev.offsetY + 10) + 'px';
      tip.textContent = `${d.kind} · ${d.label}` + (d.t == null ? ' · no timestamp' : ` · t=${d.t.toFixed(2)}`);
    })
    .on('mouseout', () => { document.getElementById('tip').style.display = 'none'; });
  node.append('circle').attr('r', 9).attr('fill', d => kindColor(d.kind)).attr('stroke', '#333');
  node.append('text').attr('dx', 12).attr('dy', 4).attr('font-size', 11).text(d => d.label);
  svg.on('click', () => select(null));

  sim.on('tick', () => {
    link.attr('x1', e => e.source.x).attr('y1', e => e.source.y)
        .attr('x2', e => e.target.x).attr('y2', e => e.target.y);
    node.attr('transform', d => `translate(${d.x},${d.y})`);
  });

  const fileLangs = [...new Set(DATA.nodes.filter(n => n.kind === 'File').map(n => n.language || 'unknown'))];
  const langColor = d3.scaleOrdinal(d3.schemeSet2).domain(fileLangs);

  function paint() {
    node.select('circle').attr('fill', d => {
      if (colorByTime) return d.t == null ? '#999' : timeColor(d.t);
      if (fileTypeColors && d.kind === 'File') return langColor(d.language || 'unknown');
      return kindColor(d.kind);
    });
    node.select('text').style('display', showLabels ? null : 'none');
  }
  let showLabels = true, colorByTime = false, fileTypeColors = true;

  function visible() {
    const onKinds = new Set([...document.querySelectorAll('#kindFilters input:checked')].map(i => i.value));
    const onRels = new Set([...document.querySelectorAll('#edgeFilters input:checked')].map(i => i.value));
    const lo = +document.getElementById('tmin').value / 100, hi = +document.getElementById('tmax').value / 100;
    document.getElementById('trange').textContent = `${lo.toFixed(2)}–${hi.toFixed(2)}`;
    const keepN = new Set(DATA.nodes.filter(n => onKinds.has(n.kind)).map(n => n));
    const keepE = DATA.edges.filter(e => onRels.has(e.relation) && keepN.has(e.source) && keepN.has(e.target));
    node.style('display', d => keepN.has(d) ? null : 'none');
    link.style('display', e => keepE.includes(e) ? null : 'none');
    node.classed('dim', d => !(d.t == null || (d.t >= lo && d.t <= hi)));
    sim.nodes([...keepN]);
    sim.force('link').links(keepE);
    sim.alpha(0.5).restart();
  }

  function select(d) {
    if (!d) {
      node.style('opacity', 1); link.style('opacity', 1);
      document.getElementById('details').textContent = 'Click a node to inspect its relations.';
      return;
    }
    const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const idOf = x => (x && x.id) || x;
    const near = new Set([d.id]);
    DATA.edges.forEach(e => {
      if (idOf(e.source) === d.id) near.add(idOf(e.target));
      if (idOf(e.target) === d.id) near.add(idOf(e.source));
    });
    node.style('opacity', n => near.has(n.id) ? 1 : 0.12);
    link.style('opacity', e => (idOf(e.source) === d.id || idOf(e.target) === d.id) ? 1 : 0.08);
    const nbrs = DATA.edges.filter(e => idOf(e.source) === d.id || idOf(e.target) === d.id)
      .map(e => {
        const other = idOf(e.source) === d.id ? e.target : e.source;
        const arrow = idOf(e.source) === d.id ? '→' : '←';
        return `<div>${arrow} <b>${esc(e.relation)}</b> ${esc(idOf(other))}${e.inferred ? ' (inferred)' : ''}</div>`;
      }).join('');
    let extra = '';
    if (d.props) {
      if (d.props.path) extra += `path: <code>${esc(d.props.path)}</code><br>`;
      if (d.props.definition_url) extra += `definition: <a href="${esc(d.props.definition_url)}" target="_blank" rel="noopener">${esc((DATA.meta.repository && DATA.meta.repository.name) || 'repo')}:${esc(d.props.path || '')}</a><br>`;
      if (d.props.language) extra += `language: ${esc(d.props.language)}<br>`;
      if (d.props.qualname) extra += `qualname: <code>${esc(d.props.qualname)}</code><br>`;
      if (d.props.line) extra += `line: ${d.props.line}<br>`;
      if (d.props.relation) extra += `relation: ${esc(d.props.relation)}<br>`;
      if (d.props.status) extra += `status: ${esc(d.props.status)}<br>`;
      if (d.props.preview) extra += `<br><b>preview</b><pre style="white-space:pre-wrap;font-size:11px;max-height:220px;overflow:auto">${esc(d.props.preview)}</pre>`;
      if (Array.isArray(d.props.history) && d.props.history.length) {
        extra += '<br><b>change history</b><br>' + d.props.history.slice(0, 20).map(h =>
          `<div>${esc(h.at.slice(0, 10))} · <code>${esc(h.commit.slice(0, 8))}</code> · ${esc(h.status)}</div>`
        ).join('') + (d.props.history.length > 20 ? `<div>… +${d.props.history.length - 20} more</div>` : '');
      }
    }
    document.getElementById('details').innerHTML =
      `<b>${d.kind}</b> · ${esc(d.label)}<br>id: <code>${esc(d.id)}</code><br>` +
      (d.time ? `time: ${d.time} (t=${d.t.toFixed(2)})<br>` : 'time: n/a (centered, no time pull)<br>') +
      extra +
      `<br>${nbrs || 'No relations in the current filter.'}`;
  }

  // controls
  if (DATA.meta && DATA.meta.repository && DATA.meta.repository.url) {
    document.getElementById('repolink').innerHTML =
      `repository: <a href="${esc(DATA.meta.repository.url)}" target="_blank" rel="noopener">${esc(DATA.meta.repository.name)}</a>`;
  }
  const kf = document.getElementById('kindFilters');
  kinds.forEach(k => { kf.innerHTML += `<label class="kind"><input type="checkbox" value="${k}" checked> ${k}</label>`; });
  const ef = document.getElementById('edgeFilters');
  relations.forEach(r => { ef.innerHTML += `<label class="kind"><input type="checkbox" value="${r}" checked> ${r}</label>`; });
  document.querySelectorAll('#kindFilters input, #edgeFilters input').forEach(i => i.onchange = visible);
  document.getElementById('tmin').oninput = document.getElementById('tmax').oninput = visible;
  document.getElementById('anchor').oninput = ev => { P.anchor = +ev.target.value / 100; sim.force('anchor').strength(P.anchor); sim.alpha(0.4).restart(); };
  document.getElementById('spread').oninput = ev => { P.far = +ev.target.value / 100; sim.alpha(0.4).restart(); };
  document.getElementById('ldist').oninput = ev => { sim.force('link').distance(+ev.target.value); sim.alpha(0.4).restart(); };
  document.getElementById('lpull').oninput = ev => { P.link = +ev.target.value / 100; sim.alpha(0.4).restart(); };
  document.getElementById('hier').oninput = ev => { P.hierMult = +ev.target.value / 100; sim.alpha(0.4).restart(); };
  document.getElementById('charge').oninput = ev => { sim.force('charge').strength(-ev.target.value); sim.alpha(0.4).restart(); };
  document.getElementById('rank').oninput = ev => { P.rank = +ev.target.value / 100; sim.force('rank').strength(P.rank); sim.alpha(0.4).restart(); };
  document.getElementById('labels').onchange = ev => { showLabels = ev.target.checked; paint(); };
  document.getElementById('bytime').onchange = ev => { colorByTime = ev.target.checked; paint(); };
  document.getElementById('filecolors').onchange = ev => { fileTypeColors = ev.target.checked; paint(); };
  document.getElementById('find').onclick = () => {
    const q = document.getElementById('search').value.toLowerCase();
    const hit = DATA.nodes.find(n => n.label.toLowerCase().includes(q));
    if (hit) select(hit);
  };
  document.getElementById('pause').onclick = ev => {
    if (sim.alpha() > 0) { sim.stop(); ev.target.textContent = 'Resume'; }
    else { sim.restart(); ev.target.textContent = 'Pause'; }
  };
  document.getElementById('reset').onclick = () => { svg.transition().call(d3.zoom().transform, d3.zoomIdentity); select(null); };
  const lg = document.getElementById('legend');
  kinds.forEach(k => { lg.innerHTML += `<div><span style="display:inline-block;width:10px;height:10px;background:${kindColor(k)}"></span> ${k}</div>`; });
  if (fileLangs.length) lg.innerHTML += '<div>files by type:</div>';
  fileLangs.forEach(l => { lg.innerHTML += `<div>&nbsp;&nbsp;<span style="display:inline-block;width:10px;height:10px;background:${langColor(l)}"></span> ${esc(l)}</div>`; });
  lg.innerHTML += '<div><span style="color:#1f6feb">━━</span> factual link</div><div><span style="color:#9e6a03">┅</span> inferred link</div>';
  paint();
})();
</script>
</body>
</html>
"""


def render_viewer_html(data: dict[str, Any], *, title: str = "Repository graph", d3_url: str = D3_CDN_URL) -> str:
    """Embed the viewer document in a self-contained HTML page."""

    payload = json.dumps(data, sort_keys=True, separators=(",", ":")).replace("</", "<\\/")
    # NOTE: str.replace is literal (unlike JS); no $-pattern hazard here.
    return (
        _VIEWER_TEMPLATE.replace("__TITLE__", title)
        .replace("__D3_URL__", d3_url)
        .replace("__DATA__", payload)
    )
