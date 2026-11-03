"""Serve demo: snapshot CSR graph lookup + embedding nearest neighbors.

Avoids Cypher planning; uses the memory-mapped sampler snapshot
(compact IDs in the hot path, external IDs only at the boundary).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yago.config import PathsConfig, make_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", type=str, default=None)
    ap.add_argument("--data-root", type=str, default=None)
    ap.add_argument("--k", type=int, default=5)
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
    E = data["entity"]
    ids = [str(x) for x in data["node_ids"].tolist()]
    j = ids.index(eid) if eid in ids else 0
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    sims = En @ En[j]
    sims[j] = -1
    top = np.argsort(-sims)[: args.k]
    print(f"--- embedding NNs of {ids[j]} ---")
    for t in top:
        print(f"  {ids[t]}  {sims[t]:.4f}")


if __name__ == "__main__":
    main()
