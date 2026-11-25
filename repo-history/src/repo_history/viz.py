"""Deterministic SVG rendering of a bounded repository slice.

The standard library only: no plotting backend, no random layout.  Node
positions derive from sorted labels and time order, so the same inputs always
produce the same bytes.  Factual links (structural, provider, mentions) are
solid blue; inferred links (topical RELATED_TO) are dashed amber, keeping the
two visibly distinguishable.  The footer records run, horizon, counts, and
coverage, and over-budget nodes are reported as omitted rather than drawn.
"""

from __future__ import annotations

from typing import Any, Sequence
from xml.sax.saxutils import escape

__all__ = ["MAX_NODES", "render_slice_svg"]

MAX_NODES = 60
FACTUAL = "#1f6feb"
INFERRED = "#9e6a03"
STRUCTURAL = "#8b949e"


def _short(label: str, limit: int = 26) -> str:
    return label if len(label) <= limit else label[: limit - 1] + "…"


def _row(
    parts: list[str],
    nodes: Sequence[dict[str, Any]],
    y: int,
    x0: int,
    x1: int,
    fill: str,
) -> dict[str, tuple[int, int]]:
    positions: dict[str, tuple[int, int]] = {}
    count = len(nodes)
    for index, node in enumerate(nodes):
        x = x0 if count == 1 else round(x0 + (x1 - x0) * index / max(count - 1, 1))
        positions[node["id"]] = (x, y)
        label = escape(_short(str(node["label"])))
        title = escape(str(node.get("title", node["label"])))
        parts.append(
            f'<g class="node"><title>{title}</title>'
            f'<rect x="{x - 62}" y="{y - 16}" width="124" height="32" rx="6" '
            f'fill="{fill}" stroke="#30363d"/>'
            f'<text x="{x}" y="{y + 5}" text-anchor="middle" font-size="12" '
            f'font-family="sans-serif" fill="#0b0e14">{label}</text></g>'
        )
    return positions


def render_slice_svg(
    *,
    commits: Sequence[dict[str, Any]],
    work_items: Sequence[dict[str, Any]],
    topics: Sequence[dict[str, Any]],
    files: Sequence[dict[str, Any]],
    links: Sequence[dict[str, Any]],
    meta: dict[str, Any],
    max_nodes: int = MAX_NODES,
) -> str:
    """Render the slice as a self-contained SVG document."""

    ordered_commits = sorted(
        commits, key=lambda node: (str(node.get("sort", "")), str(node["id"]))
    )
    ordered_items = sorted(work_items, key=lambda node: str(node["id"]))
    ordered_topics = sorted(topics, key=lambda node: str(node["id"]))
    ordered_files = sorted(files, key=lambda node: str(node["id"]))

    kept_topics = list(ordered_topics)
    kept_files = list(ordered_files)
    omitted = 0
    over_budget = (
        len(ordered_commits) + len(ordered_items) + len(kept_topics) + len(kept_files)
    )
    while over_budget > max_nodes and (kept_topics or kept_files):
        if len(kept_files) >= len(kept_topics) and kept_files:
            kept_files.pop()
            omitted += 1
        elif kept_topics:
            kept_topics.pop()
            omitted += 1
        over_budget = (
            len(ordered_commits) + len(ordered_items)
            + len(kept_topics) + len(kept_files)
        )

    parts: list[str] = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="560" role="img">',
        "<defs>"
        '<marker id="arrow-f" markerWidth="8" markerHeight="8" '
        'refX="7" refY="4" orient="auto">'
        f'<path d="M0,0 L8,4 L0,8" fill="none" stroke="{FACTUAL}" '
        'stroke-width="1.5"/></marker>'
        '<marker id="arrow-i" markerWidth="8" markerHeight="8" '
        'refX="7" refY="4" orient="auto">'
        f'<path d="M0,0 L8,4 L0,8" fill="none" stroke="{INFERRED}" '
        'stroke-width="1.5"/></marker>'
        '<marker id="arrow-s" markerWidth="8" markerHeight="8" '
        'refX="7" refY="4" orient="auto">'
        f'<path d="M0,0 L8,4 L0,8" fill="none" stroke="{STRUCTURAL}" '
        'stroke-width="1"/></marker>'
        "</defs>",
        '<text x="20" y="28" font-size="16" font-family="sans-serif" fill="#0b0e14">'
        f"{escape(str(meta.get('title', 'Repository slice')))}</text>",
        '<g class="legend" font-size="12" font-family="sans-serif">'
        f'<line x1="20" y1="48" x2="60" y2="48" stroke="{STRUCTURAL}" stroke-width="2"/>'
        '<text x="66" y="52" fill="#0b0e14">structural</text>'
        f'<line x1="170" y1="48" x2="210" y2="48" stroke="{FACTUAL}" stroke-width="2"/>'
        '<text x="216" y="52" fill="#0b0e14">factual link</text>'
        f'<line x1="330" y1="48" x2="370" y2="48" stroke="{INFERRED}" '
        'stroke-width="2" stroke-dasharray="5,3"/>'
        '<text x="376" y="52" fill="#0b0e14">inferred link</text></g>',
    ]
    positions: dict[str, tuple[int, int]] = {}
    positions.update(_row(parts, ordered_commits, 110, 90, 1110, "#ddf4ff"))
    positions.update(_row(parts, ordered_items, 250, 90, 1110, "#dafbe1"))
    positions.update(_row(parts, kept_topics, 390, 90, 1110, "#fff8c5"))
    positions.update(_row(parts, kept_files, 480, 90, 1110, "#fbefff"))

    drawn = 0
    for link in sorted(links, key=lambda edge: (str(edge["source"]), str(edge["target"]))):
        start = positions.get(str(link["source"]))
        end = positions.get(str(link["target"]))
        if start is None or end is None:
            continue
        relation = str(link.get("relation", ""))
        if relation in ("PARENT_COMMIT", "MEMBER_OF", "OF_FILE", "TIP_OF"):
            color, marker, dash = STRUCTURAL, "arrow-s", ""
            width = "1"
        elif relation == "RELATED_TO" and link.get("inferred", True):
            color, marker, dash = INFERRED, "arrow-i", ' stroke-dasharray="5,3"'
            width = "2"
        else:
            color, marker, dash = FACTUAL, "arrow-f", ""
            width = "2"
        parts.append(
            f'<line x1="{start[0]}" y1="{start[1] + 16}" x2="{end[0]}" y2="{end[1] - 16}" '
            f'stroke="{color}" stroke-width="{width}"{dash} marker-end="url(#{marker})">'
            f"<title>{escape(relation)}</title></line>"
        )
        drawn += 1

    footer = (
        f"run {meta.get('run_id', '?')} · horizon {meta.get('horizon', '?')} · "
        f"{len(ordered_commits)} commits, {len(ordered_items)} work items, "
        f"{len(kept_topics)} topics, {len(kept_files)} files, {drawn} edges drawn"
    )
    if omitted:
        footer += f" · +{omitted} nodes omitted (budget {max_nodes})"
    gaps = int(meta.get("gaps", 0) or 0)
    footer += f" · {gaps} coverage gaps"
    parts.append(
        f'<text x="20" y="540" font-size="12" font-family="sans-serif" fill="#57606a">'
        f"{escape(footer)}</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts)
