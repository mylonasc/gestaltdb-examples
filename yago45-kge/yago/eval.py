"""Filtered link-prediction eval: MRR / Hits@k with random negatives."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yago.config import make_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--triples", type=int, default=2000)
    ap.add_argument("--negatives", type=int, default=50)
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--data-root", type=str, default=None)
    args = ap.parse_args()
    cfg = make_config(smoke_test=args.smoke_test)
    if args.data_root:
        from pathlib import Path as _Path
        from yago.config import PathsConfig as _PC
        cfg.paths = _PC(data_root=_Path(args.data_root))
    data = np.load(cfg.paths.embeddings_path)
    E, R = data["entity"].astype(np.float32), data["relation"].astype(np.float32)
    arrays = np.load(cfg.paths.edge_arrays_path)
    src, dst, rel = arrays["src"], arrays["dst"], arrays["rel"]
    rng = np.random.default_rng(0)
    n = min(args.triples, len(src))
    sel = rng.choice(len(src), size=n, replace=False)
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    ranks = []
    for s, r, o in zip(src[sel], rel[sel], dst[sel]):
        cand_o = rng.integers(0, len(E), size=args.negatives)
        cands = np.concatenate([[o], cand_o])
        scores = np.sum(En[s] * R[r] * En[cands], axis=1)
        rank = int(1 + np.sum(scores[1:] >= scores[0]))
        ranks.append(rank)
    ranks = np.array(ranks)
    mrr = float(np.mean(1.0 / ranks))
    print(f"n={n} MRR={mrr:.4f} Hits@1={np.mean(ranks<=1):.4f} Hits@10={np.mean(ranks<=10):.4f}")


if __name__ == "__main__":
    main()
