"""Serve demo: snapshot lookup + relation-conditioned KGE retrieval.

Avoids Cypher planning; uses the memory-mapped sampler snapshot
(compact IDs in the hot path, external IDs only at the boundary).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yago.config import PathsConfig, make_config
from yago.scoring import resolve_score_function, score_objects


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", type=str, default=None)
    ap.add_argument("--data-root", type=str, default=None)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--relation", type=str, default=None,
                    help="Relation for (entity, relation, ?) retrieval; defaults to first outgoing edge")
    ap.add_argument("--cosine-diagnostic", action="store_true",
                    help="Also print raw cosine neighbors (not the KGE objective)")
    ap.add_argument("--score", choices=["complex", "distmult"], default=None,
                    help="Required only for legacy artifacts without metadata")
    args = ap.parse_args()
    cfg = make_config()
    if args.data_root:
        cfg.paths = PathsConfig(data_root=Path(args.data_root))

    from gestaltdb.sampling import SamplerSnapshot
    sn = SamplerSnapshot.load(cfg.paths.sampler_snapshot_path, mmap=True)
    eid = args.entity or str(sn.external_node_ids[0])
    try:
        i = int(sn.node_int(eid))
    except KeyError:
        i = 0
        eid = str(sn.external_node_ids[0])
        print(f"(entity not in map, using {eid})")
    def show(adj, arr, other, direction):
        r = adj.edge_range(i)
        eids = adj.edge_indices[r][:10]
        for e in list(eids):
            e = int(e)
            print(f"  {direction} {sn.external_relation_ids[int(sn.rel_int[e])]}"
                  f" {'->' if direction == 'out' else '<-'} {sn.external_node_ids[int(other[e])]}")
    print(f"--- edges of {eid} (snapshot CSR) ---")
    show(sn.out, sn.out.edge_indices, sn.dst_int, "out")
    show(sn.in_, sn.in_.edge_indices, sn.src_int, "in")

    data = np.load(cfg.paths.embeddings_path)
    meta_path = cfg.paths.embeddings_path.parent / "artifact_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    E = data[meta.get("entity_key", "entity")]
    R = data[meta.get("relation_key", "relation")]
    ids = [str(x) for x in data["node_ids"].tolist()]
    relation_ids = [str(x) for x in data["relation_ids"].tolist()]
    j = ids.index(eid) if eid in ids else 0
    if args.relation:
        relation = args.relation
    else:
        edge_range = sn.out.edge_range(i)
        edge_ids = sn.out.edge_indices[edge_range]
        if not len(edge_ids):
            raise ValueError("entity has no outgoing relation; pass --relation")
        relation = str(sn.external_relation_ids[int(sn.rel_int[int(edge_ids[0])])])
    r = relation_ids.index(relation)
    try:
        scorer = resolve_score_function(args.score, meta.get("score_function"))
    except ValueError as exc:
        ap.error(str(exc))
    scores = score_objects(E, R, j, r, np.arange(len(E)), scorer)
    top = np.argpartition(-scores, min(args.k, len(scores) - 1))[:args.k]
    top = top[np.argsort(-scores[top])]
    print(f"--- relation-conditioned retrieval ({ids[j]}, {relation}, ?) [{scorer}] ---")
    for t in top:
        print(f"  {ids[t]}  {scores[t]:.4f}")
    if args.cosine_diagnostic:
        En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        sims = En @ En[j]
        sims[j] = -1
        top = np.argpartition(-sims, min(args.k, len(sims) - 1))[:args.k]
        top = top[np.argsort(-sims[top])]
        print(f"--- raw cosine diagnostic for {ids[j]} (not trained KGE query) ---")
        for t in top:
            print(f"  {ids[t]}  {sims[t]:.4f}")


if __name__ == "__main__":
    main()
