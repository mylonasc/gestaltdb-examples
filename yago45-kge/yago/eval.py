"""Link-prediction eval on the HELD-OUT TEST split (filtered MRR / Hits@k).

Test triples were excluded from the sampler snapshot at prepare time, so this
measures generalization. Filtered: train/valid positives sharing (s, r) are
removed from the candidate list before ranking (standard filtered protocol).
"""
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
    ap.add_argument("--negatives", type=int, default=200)
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--data-root", type=str, default=None)
    ap.add_argument("--split", type=str, default="test", choices=["test", "valid"])
    ap.add_argument("--score", type=str, default="complex", choices=["distmult", "complex"])
    args = ap.parse_args()
    cfg = make_config(smoke_test=args.smoke_test)
    if args.data_root:
        from yago.config import PathsConfig as _PC
        cfg.paths = _PC(data_root=Path(args.data_root))
    import polars as pl
    data = np.load(cfg.paths.embeddings_path)
    E, R = data["entity"].astype(np.float32), data["relation"].astype(np.float32)
    node_map = pl.read_parquet(cfg.paths.node_map_path).sort("node_int")
    rel_map = pl.read_parquet(cfg.paths.relation_map_path).sort("rel_int")
    id2int = dict(zip(node_map["node_id"].to_list(), node_map["node_int"].to_list()))
    rel2int = dict(zip(rel_map["edge_type"].to_list(), rel_map["rel_int"].to_list()))

    def load_int(path):
        rows = []
        for r in pl.read_parquet(path).iter_rows(named=True):
            if r["source"] in id2int and r["target"] in id2int and r["edge_type"] in rel2int:
                rows.append((id2int[r["source"]], rel2int[r["edge_type"]], id2int[r["target"]]))
        return np.array(rows, dtype=np.int64)

    split_path = cfg.paths.triples_test_path if args.split == "test" else cfg.paths.triples_valid_path
    tst = load_int(split_path)
    train = load_int(cfg.paths.triples_parquet)
    known = set(map(tuple, train.tolist())) | set(map(tuple, tst.tolist()))
    try:
        valid = load_int(cfg.paths.triples_valid_path)
        known |= set(map(tuple, valid.tolist()))
    except Exception:
        pass
    # (s, r) -> {known o} for filtered ranking.
    from collections import defaultdict
    known_o = defaultdict(set)
    for s, r, o in known:
        known_o[(s, r)].add(o)

    rng = np.random.default_rng(0)
    n = min(args.triples, len(tst))
    sel = rng.choice(len(tst), size=n, replace=False)
    # Standard scores (raw, NOT cosine: norms carry signal). ComplEx =
    # Re(<s, r, conj(o)>) over 200-d = 100 complex dims; matches training.
    def batch_scores(s, r, cands):
        if args.score == "complex":
            d = E.shape[1] // 2
            s_re, s_im = E[s, :d], E[s, d:]
            r_re, r_im = R[r, :d], R[r, d:]
            o = E[cands]
            o_re, o_im = o[:, :d], o[:, d:]
            return np.sum(
                r_re * (s_re * o_re + s_im * o_im)
                + r_im * (s_re * o_im - s_im * o_re), axis=1)
        return np.sum(E[s] * R[r] * E[cands], axis=1)
    ranks = []
    for s, r, o in tst[sel]:
        cand_o = rng.integers(0, len(E), size=args.negatives).tolist()
        cands = [o] + [c for c in cand_o if (s, r, c) not in known_o or c == o]
        cands = cands[: args.negatives + 1]
        scores = batch_scores(s, r, cands)
        ranks.append(int(1 + np.sum(scores[1:] >= scores[0])))
    ranks = np.array(ranks)
    print(f"split={args.split} score={args.score} n={n} MRR={float(np.mean(1.0/ranks)):.4f} "
          f"Hits@1={np.mean(ranks<=1):.4f} Hits@10={np.mean(ranks<=10):.4f} Hits@50={np.mean(ranks<=50):.4f}")
    # Unfiltered random baseline for reference: E[1/rank] over K+1 candidates.
    K = args.negatives
    print(f"(random baseline MRR ~ {float(np.mean(1.0/np.arange(1, K+2))):.4f})")


if __name__ == "__main__":
    main()
