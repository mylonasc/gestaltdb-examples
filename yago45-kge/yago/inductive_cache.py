"""Cache the trained GNN's entity vectors, independently of shallow ID tables."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import signal
import time

import numpy as np
import tensorflow as tf

from .inductive_data import GraphKnownObjects, NeighborhoodGraph, file_digest
from .inductive_eval import evaluate


def scrub_queries(edges, queries, num_nodes, num_relations, return_removed=False):
    """Remove all held-out queries and their reverse copies for static caching."""
    if num_nodes * num_nodes * num_relations >= np.iinfo(np.int64).max:
        raise ValueError("triple key exceeds int64; use a smaller local graph namespace")
    blocked = np.concatenate([part for part in queries if len(part)], axis=0)
    reversed_rows = blocked[:, [2, 1, 0]]
    blocked = np.concatenate([blocked, reversed_rows])
    keys = np.unique((blocked[:, 0] * num_relations + blocked[:, 1]) * num_nodes + blocked[:, 2])
    keep = np.ones(len(edges), dtype=bool)
    for start in range(0, len(edges), 1000000):
        part = edges[start:start + 1000000]
        encoded = (part[:, 0] * num_relations + part[:, 1]) * num_nodes + part[:, 2]
        keep[start:start + len(part)] = ~np.isin(encoded, keys)
    return (edges[keep], edges[~keep]) if return_removed else edges[keep]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New cache directory")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-nodes", type=int, default=None, help="Benchmark prefix only; default is EVERY mapped node")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-triples", type=int, default=1000)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch size must be positive")
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    spec = json.loads((args.run_dir / "run.json").read_text())
    if file_digest(spec["dataset"]) != spec["dataset_digest"]:
        parser.error("dataset content differs from the trained run")
    data = np.load(spec["dataset"], allow_pickle=False)
    num_nodes, num_relations = len(data["node_ids"]), len(data["relation_ids"])
    node_count = min(num_nodes, args.max_nodes) if args.max_nodes is not None else num_nodes
    dim = spec["model"]["dim"]
    saved_model = args.run_dir / "gnn_saved_model"
    if not (saved_model / "saved_model.pb").exists():
        parser.error("no standalone GNN export; finish training/export first")
    if node_count <= 0:
        parser.error("node count must be positive")
    digest = hashlib.sha256()
    for path in sorted(saved_model.rglob("*")):
        if path.is_file():
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    identity = dict(model_digest=digest.hexdigest(), dataset_digest=spec["dataset_digest"],
                    nodes=node_count, dim=dim, dtype="float16", batch_size=args.batch_size,
                    fanouts=spec["fanouts"], sampling_version=spec.get("sampling_version", 1),
                    feature_mode=spec.get("feature_mode", "sampled"),
                    protocol="train + support, ALL heldout queries/reverses globally scrubbed")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "cache.json"
    vectors_path = args.output / "gnn_embeddings.npy"
    rng = np.random.default_rng(spec["seed"])
    completed = 0
    if args.resume:
        manifest = json.loads(manifest_path.read_text())
        if manifest["identity"] != identity:
            parser.error("cache identity/recipe changed; choose a new output")
        completed = manifest["completed_nodes"]
        rng.bit_generator.state = manifest["rng_state"]
        vectors = np.load(vectors_path, mmap_mode="r+")
    else:
        if any(args.output.iterdir()):
            parser.error("cache directory is not empty; use --resume or a new output")
        if shutil.disk_usage(args.output).free < node_count * dim * 2 + 1024**3:
            parser.error("insufficient disk for requested float16 cache and 1 GiB safety margin")
        vectors = np.lib.format.open_memmap(vectors_path, mode="w+", dtype=np.float16, shape=(node_count, dim))
    def save_progress():
        vectors.flush()
        manifest = dict(identity=identity, completed_nodes=completed, complete=completed == node_count,
                        rng_state=rng.bit_generator.state, node_mapping_path=str(data["node_mapping_path"])
                        if "node_mapping_path" in data else None)
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2))
        temporary.replace(manifest_path)

    if not args.resume:
        save_progress()
    queries = [data[key] for key in ("valid", "test", "temporal_valid", "temporal_test") if key in data]
    edges, removed = scrub_queries(np.concatenate([data["train"], data["support"]]), queries,
                                    num_nodes, num_relations, return_removed=True)
    print(f"cache context: {len(edges):,} edges; encoding {node_count:,} nodes", flush=True)
    context = NeighborhoodGraph(edges, num_nodes, num_relations,
        snapshot_path=args.run_dir / "snapshots" / "cache", compact=True,
        feature_mode=spec.get("feature_mode", "sampled"), sampling_version=spec.get("sampling_version", 1))
    model = tf.saved_model.load(str(saved_model))
    stop_requested = False

    def stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    previous_handlers = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
    started, initial = time.perf_counter(), completed
    try:
        for start in range(completed, node_count, args.batch_size):
            roots = np.arange(start, min(start + args.batch_size, node_count), dtype=np.int64)
            ids, graph = context.sample(roots, spec["fanouts"], rng)
            encoded = model.encode({key: tf.convert_to_tensor(value) for key, value in graph.items()}).numpy()
            vectors[roots] = encoded[np.searchsorted(ids, roots)]
            completed = start + len(roots)
            if (start // args.batch_size) % 50 == 0 or completed == node_count or stop_requested:
                save_progress()
                rate = (completed - initial) / max(time.perf_counter() - started, 1e-6)
                print(f"cached={completed:,}/{node_count:,} nodes_per_second={rate:.1f}", flush=True)
            if stop_requested:
                return
    finally:
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
    save_progress()
    if args.eval_triples and completed == num_nodes:
        known = GraphKnownObjects(context, *queries, removed)
        pool = np.unique(edges[:, [0, 2]])
        results = {policy: evaluate(model, context, data["test"], known, pool,
                                   fanouts=spec["fanouts"], negatives=50, max_queries=args.eval_triples,
                                   candidate_policy=policy, cached_vectors=vectors)
                   for policy in ("uniform", "relation-range")}
        (args.output / "cached_link_prediction.json").write_text(json.dumps(results, indent=2))
        print(json.dumps({policy: result["mrr"] for policy, result in results.items()}), flush=True)


if __name__ == "__main__":
    main()
