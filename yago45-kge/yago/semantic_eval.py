"""Quantify semantic cosine geometry for selected YAGO attributes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yago.config import PathsConfig, make_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--per-class", type=int, default=400)
    ap.add_argument("--relations", nargs="+", default=["schema:gender", "schema:nationality", "rdf:type"])
    args = ap.parse_args()
    cfg = make_config(); cfg.paths = PathsConfig(data_root=Path(args.data_root))
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    meta_path = Path(args.embeddings).parent / "artifact_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    data = np.load(args.embeddings)
    E = data[meta.get("entity_key", "entity")].astype(np.float32)
    node_ids = pl.read_parquet(cfg.paths.node_map_path).sort("node_int")["node_id"].to_list()
    rel_map = pl.read_parquet(cfg.paths.relation_map_path)
    rel_to_int = dict(zip(rel_map["edge_type"].to_list(), rel_map["rel_int"].to_list()))
    arrays = np.load(cfg.paths.edge_arrays_path, mmap_mode="r")
    rng = np.random.default_rng(13)
    results = []
    for relation_name in args.relations:
        if relation_name not in rel_to_int:
            continue
        relation = int(rel_to_int[relation_name])
        mask = arrays["rel"] == relation
        src, label = arrays["src"][mask], arrays["dst"][mask]
        labels, counts = np.unique(label, return_counts=True)
        # Keep a small, interpretable set with enough examples. Gender uses two
        # major classes; nationality five; type ten.
        n_classes = 2 if "gender" in relation_name else (5 if "nationality" in relation_name else 10)
        keep = labels[np.argsort(-counts)[:n_classes]]
        rows, y = [], []
        for class_index, value in enumerate(keep):
            members = np.unique(src[label == value])
            take = rng.choice(members, size=min(args.per_class, len(members)), replace=False)
            rows.extend(take.tolist()); y.extend([class_index] * len(take))
        rows, y = np.asarray(rows, dtype=np.int64), np.asarray(y, dtype=np.int64)
        X = E[rows]
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
        sims = X @ X.T
        np.fill_diagonal(sims, -np.inf)
        top10 = np.argpartition(-sims, min(10, len(rows) - 1), axis=1)[:, :10]
        one_nn = float(np.mean(y[np.argmax(sims, axis=1)] == y))
        precision10 = float(np.mean(y[top10] == y[:, None]))
        same = y[:, None] == y[None, :]
        eye = np.eye(len(y), dtype=bool)
        within = float(np.mean((X @ X.T)[same & ~eye]))
        between = float(np.mean((X @ X.T)[~same]))
        result = {
            "relation": relation_name, "classes": int(len(keep)), "entities": int(len(rows)),
            "class_values": [node_ids[int(value)] for value in keep],
            "random_accuracy": 1.0 / len(keep),
            "cosine_1nn_accuracy": one_nn, "cosine_precision_at_10": precision10,
            "mean_within_cosine": within, "mean_between_cosine": between,
            "cosine_margin": within - between,
        }
        results.append(result)
        np.savez_compressed(out / f"{relation_name.split(':')[-1]}_sample.npz", rows=rows, labels=y)
        z = TSNE(n_components=2, perplexity=30, max_iter=1_000, metric="cosine",
                 init="random", learning_rate="auto", random_state=13).fit_transform(X)
        plt.figure(figsize=(9, 6))
        for class_index, value in enumerate(keep):
            selected = y == class_index
            name = node_ids[int(value)].split(":")[-1][:28]
            plt.scatter(z[selected, 0], z[selected, 1], s=8, alpha=0.7,
                        label=f"{name} (n={selected.sum()})")
        plt.legend(markerscale=2, fontsize=8)
        plt.title(f"Cosine t-SNE by {relation_name} (1-NN={one_nn:.3f}, P@10={precision10:.3f})")
        plt.tight_layout()
        plt.savefig(out / f"tsne_{relation_name.split(':')[-1]}.png", dpi=130)
        plt.close()
        print(json.dumps(result))
    report = {"embeddings": str(args.embeddings), "per_class": args.per_class, "results": results}
    (out / "semantic_metrics.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
