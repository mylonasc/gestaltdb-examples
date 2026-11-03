"""Export 200-d entity/relation embeddings + sanity report."""
from __future__ import annotations

import json

import numpy as np
import polars as pl

from .config import AppConfig


def export_embeddings(cfg: AppConfig, model) -> dict:
    cfg.paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
    entity_mat = model.entity_emb.get_weights()[0].astype(np.float32)
    rel_mat = model.rel_emb.get_weights()[0].astype(np.float32)
    assert entity_mat.shape[1] == 200, entity_mat.shape
    assert rel_mat.shape[1] == 200, rel_mat.shape
    node_map = pl.read_parquet(cfg.paths.node_map_path).sort("node_int")
    rel_map = pl.read_parquet(cfg.paths.relation_map_path).sort("rel_int")
    assert len(node_map) == entity_mat.shape[0]
    assert len(rel_map) == rel_mat.shape[0]
    np.savez_compressed(
        cfg.paths.embeddings_path,
        entity=entity_mat,
        relation=rel_mat,
        node_ids=np.array(node_map["node_id"].to_list()),
        relation_ids=np.array(rel_map["edge_type"].to_list()),
    )
    # Cosine-NN sanity on 5 random entities.
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(entity_mat), size=min(5, len(entity_mat)))
    cosmic = entity_mat / (np.linalg.norm(entity_mat, axis=1, keepdims=True) + 1e-9)
    report = []
    ids = node_map["node_id"].to_list()
    for i in list(idx):
        sims = cosmic @ cosmic[i]
        sims[i] = -1
        top = np.argsort(-sims)[:5]
        report.append({"entity": ids[i], "neighbors": [(ids[j], round(float(sims[j]), 4)) for j in top]})
    (cfg.paths.artifacts_dir / "nn_report.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {cfg.paths.embeddings_path} entity={entity_mat.shape} relation={rel_mat.shape}")
    print(json.dumps(report, indent=2))
    return {"entity_shape": entity_mat.shape, "relation_shape": rel_mat.shape, "report": report}
