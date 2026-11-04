"""Export 200-d entity/relation embeddings + sanity report."""
from __future__ import annotations

import json

import numpy as np
import polars as pl

from .config import AppConfig
from .scoring import score_objects


def export_embeddings(cfg: AppConfig, model, *, output_dir=None, metadata: dict | None = None) -> dict:
    output_dir = cfg.paths.artifacts_dir if output_dir is None else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    entity_mat = model.entity_emb.get_weights()[0].astype(np.float32)
    rel_mat = model.rel_emb.get_weights()[0].astype(np.float32)
    assert entity_mat.shape[1] == 200, entity_mat.shape
    assert rel_mat.shape[1] == 200, rel_mat.shape
    node_map = pl.read_parquet(cfg.paths.node_map_path).sort("node_int")
    rel_map = pl.read_parquet(cfg.paths.relation_map_path).sort("rel_int")
    assert len(node_map) == entity_mat.shape[0]
    assert len(rel_map) == rel_mat.shape[0]
    np.savez_compressed(
        output_dir / "yago45_embeddings_200d.npz",
        entity_base_200d=entity_mat,
        relation_base_200d=rel_mat,
        node_ids=np.array(node_map["node_id"].to_list()),
        relation_ids=np.array(rel_map["edge_type"].to_list()),
    )
    artifact_meta = {
        "representation": "base KGE tables (not contextual GNN states)",
        "score_function": cfg.model.score_function,
        "entity_key": "entity_base_200d",
        "relation_key": "relation_base_200d",
        "entity_dim": int(entity_mat.shape[1]),
        "relation_dim": int(rel_mat.shape[1]),
        "generic_cosine_is_training_objective": cfg.model.semantic_weight > 0,
        "semantic_relations": list(cfg.model.semantic_relations) if cfg.model.semantic_weight > 0 else [],
    }
    if metadata:
        artifact_meta["training_run"] = metadata
    (output_dir / "artifact_metadata.json").write_text(json.dumps(artifact_meta, indent=2))

    # Raw cosine is retained only as an explicitly labelled diagnostic.
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(entity_mat), size=min(5, len(entity_mat)))
    cosmic = entity_mat / (np.linalg.norm(entity_mat, axis=1, keepdims=True) + 1e-9)
    report = []
    ids = node_map["node_id"].to_list()
    for i in list(idx):
        sims = cosmic @ cosmic[i]
        sims[i] = -1
        top = np.argpartition(-sims, 5)[:5]
        top = top[np.argsort(-sims[top])]
        report.append({"entity": ids[i], "neighbors": [(ids[j], round(float(sims[j]), 4)) for j in top]})
    diagnostic = {
        "warning": "Raw cosine similarity was not optimized by the KGE objective; use relation-conditioned retrieval.",
        "queries": report,
    }
    (output_dir / "cosine_diagnostic.json").write_text(json.dumps(diagnostic, indent=2))
    del cosmic
    # Primary retrieval example: relation-conditioned sampled object ranking.
    heldout = pl.read_parquet(cfg.paths.triples_valid_path).head(5)
    src_map = node_map.select(
        pl.col("node_id").alias("source"), pl.col("node_int").alias("source_int"))
    dst_map = node_map.select(
        pl.col("node_id").alias("target"), pl.col("node_int").alias("target_int"))
    rmap = rel_map.select(
        pl.col("edge_type"), pl.col("rel_int"))
    queries = heldout.join(src_map, on="source").join(dst_map, on="target").join(rmap, on="edge_type")
    retrieval = []
    edge_arrays = np.load(cfg.paths.edge_arrays_path, mmap_mode="r")
    train_dst, train_rel = edge_arrays["dst"], edge_arrays["rel"]
    for row in queries.iter_rows(named=True):
        relation = int(row["rel_int"])
        pool = np.unique(train_dst[train_rel == relation])
        candidate_count = min(50_000, len(pool))
        sampled = rng.choice(pool, size=candidate_count, replace=False)
        candidates = np.unique(np.r_[int(row["target_int"]), sampled])
        scores = score_objects(entity_mat, rel_mat, int(row["source_int"]),
                               int(row["rel_int"]), candidates, cfg.model.score_function)
        target_at = int(np.flatnonzero(candidates == int(row["target_int"]))[0])
        order = np.argsort(-scores)[:5]
        retrieval.append({
            "subject": row["source"], "relation": row["edge_type"],
            "true_object": row["target"],
            "true_object_sampled_rank": int(1 + np.sum(scores > scores[target_at])),
            "candidate_count": int(len(candidates)),
            "top_objects": [(ids[int(candidates[j])], round(float(scores[j]), 5)) for j in order],
        })
    relation_report = {
        "protocol": "relation-conditioned relation-range retrieval: (subject, relation, ?)",
        "score_function": cfg.model.score_function,
        "queries": retrieval,
    }
    (output_dir / "relation_retrieval_report.json").write_text(json.dumps(relation_report, indent=2))
    print(f"wrote {output_dir / 'yago45_embeddings_200d.npz'} entity={entity_mat.shape} relation={rel_mat.shape}")
    print(json.dumps(diagnostic, indent=2))
    return {"entity_shape": entity_mat.shape, "relation_shape": rel_mat.shape,
            "relation_report": retrieval, "cosine_diagnostic": report}
