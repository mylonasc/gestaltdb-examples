"""Link-prediction eval on the held-out split (sampled-filtered MRR / Hits@k).

Test triples were excluded from the sampler snapshot at prepare time, so this
measures generalization. Filtered: train/valid positives sharing (s, r) are
removed from each sampled candidate list before ranking.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yago.config import make_config
from yago.scoring import filtered_object_candidates, resolve_score_function, score_objects


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--triples", type=int, default=2000)
    ap.add_argument("--negatives", type=int, default=200)
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--data-root", type=str, default=None)
    ap.add_argument("--split", type=str, default="test", choices=["test", "valid"])
    ap.add_argument("--score", type=str, default=None, choices=["distmult", "complex"],
                    help="Override artifact scorer (an inconsistent override is rejected)")
    ap.add_argument("--embeddings", type=str, default=None)
    ap.add_argument("--candidate-policy", choices=["uniform", "relation-range"],
                    default="uniform",
                    help="relation-range samples objects observed with that relation (harder)")
    args = ap.parse_args()
    cfg = make_config(smoke_test=args.smoke_test)
    if args.data_root:
        from yago.config import PathsConfig as _PC
        cfg.paths = _PC(data_root=Path(args.data_root))
    import polars as pl
    embeddings_path = Path(args.embeddings) if args.embeddings else cfg.paths.embeddings_path
    metadata_path = embeddings_path.parent / "artifact_metadata.json"
    artifact_meta = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    trained_score = artifact_meta.get("score_function")
    try:
        score_function = resolve_score_function(args.score, trained_score)
    except ValueError as exc:
        ap.error(str(exc))
    data = np.load(embeddings_path)
    entity_key = artifact_meta.get("entity_key", "entity")
    relation_key = artifact_meta.get("relation_key", "relation")
    E, R = data[entity_key].astype(np.float32), data[relation_key].astype(np.float32)
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
    relation_objects = {}
    if args.candidate_policy == "relation-range":
        for relation in np.unique(tst[sel, 1]):
            relation_objects[int(relation)] = np.unique(train[train[:, 1] == relation, 2])
    ranks = []
    for s, r, o in tst[sel]:
        if args.candidate_policy == "relation-range":
            pool = relation_objects[int(r)]
            cand_o = rng.choice(pool, size=args.negatives, replace=len(pool) < args.negatives)
        else:
            cand_o = rng.integers(0, len(E), size=args.negatives)
        cands = filtered_object_candidates(s, r, o, cand_o, known_o)[: args.negatives + 1]
        scores = score_objects(E, R, s, r, cands, score_function)
        ranks.append(int(1 + np.sum(scores[1:] >= scores[0])))
    ranks = np.array(ranks)
    print(f"protocol=sampled-filtered policy={args.candidate_policy} split={args.split} "
          f"score={score_function} candidates<={args.negatives + 1} "
          f"n={n} MRR={float(np.mean(1.0/ranks)):.4f} "
          f"Hits@1={np.mean(ranks<=1):.4f} Hits@10={np.mean(ranks<=10):.4f} Hits@50={np.mean(ranks<=50):.4f}")
    # Unfiltered random baseline for reference: E[1/rank] over K+1 candidates.
    K = args.negatives
    print(f"(random baseline MRR ~ {float(np.mean(1.0/np.arange(1, K+2))):.4f})")


if __name__ == "__main__":
    main()
