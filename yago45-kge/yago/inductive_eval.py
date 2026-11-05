"""Restore a trained inductive GNN and rank held-out links using its decoder."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from .inductive_data import GraphKnownObjects, NeighborhoodGraph, mapping_digest, sampled_candidates, zero_relation_degree_candidates
from .inductive_model import InductiveAutoencoder


def evaluate(model, context, queries, all_known, candidate_pool, *, fanouts,
             negatives=50, max_queries=200, seed=12345, candidate_policy="uniform",
             include_details=False, message_passing=True, cached_vectors=None):
    """Sampled-filtered tail ranking; encode true and negative candidates equally."""
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(queries), min(max_queries, len(queries)), replace=False)
    ranks, counts, details = [], [], []
    ranges = {}
    range_counts = {}
    matched_pools = {}
    frequencies = {}
    for relation in np.unique(queries[selected, 1]):
        objects, frequency = np.unique(context.edges[context.edges[:, 1] == relation, 2], return_counts=True)
        ranges[int(relation)] = objects
        range_counts[int(relation)] = frequency
        frequencies[int(relation)] = dict(zip(objects.tolist(), frequency.tolist()))
    for query in queries[selected]:
        s, r, _ = map(int, query)
        pool = ranges[r] if candidate_policy == "relation-range" else candidate_pool
        degree = frequencies[r].get(int(query[2]), 0)
        if candidate_policy == "degree-matched" and degree == 0:
            candidates = zero_relation_degree_candidates(query, candidate_pool, ranges[r], negatives, all_known, rng)
        else:
            if candidate_policy == "degree-matched":
                lower = 2 ** (int(degree).bit_length() - 1)
                key = (r, lower)
                if key not in matched_pools:
                    matched_pools[key] = ranges[r][(range_counts[r] >= lower) & (range_counts[r] < 2 * lower)]
                pool = matched_pools[key]
            candidates = sampled_candidates(query, pool, negatives, all_known, rng)
        if cached_vectors is not None:
            subjects = np.repeat(np.asarray(cached_vectors[s:s + 1], dtype=np.float32), len(candidates), axis=0)
            objects = np.asarray(cached_vectors[candidates], dtype=np.float32)
            scores = model.score_embeddings(subjects, np.full(len(candidates), r, dtype=np.int32), objects).numpy()
        else:
            ids, graph = context.sample(np.r_[s, candidates], fanouts, rng, masked=[query])
            if not message_passing:
                for key in ("senders", "receivers", "relations"):
                    graph[key] = np.empty(0, dtype=np.int32)
            local = np.stack([np.full(len(candidates), np.searchsorted(ids, s)),
                              np.full(len(candidates), r), np.searchsorted(ids, candidates)], axis=1)
            scores = model.predict_graph({key: tf.convert_to_tensor(value) for key, value in graph.items()},
                                         tf.convert_to_tensor(local, tf.int32)).numpy()
        ranks.append(1 + np.sum(scores[1:] >= scores[0]))
        counts.append(len(candidates))
        freq = np.asarray([frequencies[r].get(int(o), 0) for o in candidates])
        # Expected reciprocal rank under random tie-breaking for the frequency
        # baseline, rather than optimistically awarding all tied tails rank one.
        higher, tied = np.sum(freq > freq[0]), np.sum(freq == freq[0])
        frequency_rr = float(np.mean(1 / np.arange(higher + 1, higher + tied + 1)))
        details.append(dict(subject=s, relation=r, object=int(query[2]), rank=int(ranks[-1]),
                            candidates=len(candidates), frequency_rr=frequency_rr,
                            frequency_pessimistic_rr=float(1 / np.sum(freq >= freq[0])),
                            random_rr=float(np.mean(1 / np.arange(1, len(candidates) + 1)))))
    if not ranks:
        raise ValueError("no queries available for evaluation")
    result = metrics_from_details(details, candidate_policy, seed)
    if include_details:
        result["details"] = details
    if cached_vectors is not None:
        result["protocol"] = "cached GNN embeddings + contextual decoder, sampled-filtered tails"
    if candidate_policy == "degree-matched":
        result["matching"] = "observed incoming-predicate-degree bins: zero, 1, 2-3, 4-7, ..."
    return result


def metrics_from_details(details, candidate_policy="uniform", seed=12345):
    """Reconstruct exact metrics from persisted rankings after an interruption."""
    if not details:
        raise ValueError("no persisted rankings")
    ranks = np.asarray([row["rank"] for row in details])
    counts = np.asarray([row["candidates"] for row in details])
    rr = 1 / ranks
    bootstrap_rng = np.random.default_rng(seed + 1)
    bootstrap = np.mean(bootstrap_rng.choice(rr, size=(1000, len(rr)), replace=True), axis=1)
    per_relation = {}
    for relation in sorted(set(row["relation"] for row in details)):
        group = [row for row in details if row["relation"] == relation]
        per_relation[str(relation)] = dict(queries=len(group), mrr=float(np.mean([1 / row["rank"] for row in group])))
    result = dict(protocol="GNN sampled-filtered tail prediction", candidate_policy=candidate_policy,
                tie_policy="pessimistic (other scores >= true score)", queries=len(ranks),
                 mean_candidates=float(np.mean(counts)), mrr=float(np.mean(1 / ranks)),
                hits1=float(np.mean(ranks <= 1)), hits10=float(np.mean(ranks <= 10)),
                mrr_ci95=np.quantile(bootstrap, [0.025, 0.975]).tolist(),
                random_mrr=float(np.mean([row["random_rr"] for row in details])),
                frequency_mrr=float(np.mean([row["frequency_rr"] for row in details])),
                frequency_pessimistic_mrr=float(np.mean([row["frequency_pessimistic_rr"] for row in details])),
                macro_relation_mrr=float(np.mean([row["mrr"] for row in per_relation.values()])),
                per_relation=per_relation)
    nontrivial = np.asarray(counts) > 1
    result["single_candidate_queries"] = int(np.sum(~nontrivial))
    result["mrr_nontrivial_queries"] = float(np.mean(rr[nontrivial])) if np.any(nontrivial) else None
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--dataset", type=Path, help="Defaults to the training dataset; mappings must match")
    parser.add_argument("--split", choices=["valid", "test", "temporal_valid", "temporal_test"], default="test")
    parser.add_argument("--triples", type=int, default=200)
    parser.add_argument("--negatives", type=int, default=50)
    parser.add_argument("--checkpoint", choices=["best", "latest"], default="best")
    parser.add_argument("--candidate-policy", choices=["uniform", "relation-range", "degree-matched"], default="uniform")
    parser.add_argument("--saved-model", action="store_true", help="Use the lightweight exported GNN, without loading ID tables")
    args = parser.parse_args()
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    if args.triples <= 0 or args.negatives <= 0:
        parser.error("--triples and --negatives must be positive")
    spec = json.loads((args.run_dir / "run.json").read_text())
    dataset = args.dataset or Path(spec["dataset"])
    data = np.load(dataset, allow_pickle=False)
    if args.saved_model:
        if args.checkpoint != "best":
            parser.error("the standalone SavedModel is validation-selected; use --checkpoint best")
        latest = str(args.run_dir / "gnn_saved_model")
        model = tf.saved_model.load(latest)
    else:
        model = InductiveAutoencoder(**spec["model"])
        checkpoint_dir = args.run_dir / ("best" if args.checkpoint == "best" else "checkpoints")
        latest = tf.train.latest_checkpoint(str(checkpoint_dir))
        if latest is None:
            parser.error("run has no checkpoint")
        # Full training checkpoint includes optimizer and step, unused in inference.
        status = tf.train.Checkpoint(model=model).restore(latest)
        status.assert_existing_objects_matched()
        status.expect_partial()
    if mapping_digest(data) != spec["mapping_digest"]:
        parser.error("dataset mappings differ from the trained run")
    train = data["train"]
    if args.split == "test":
        edges = np.concatenate([train, data["support"]])
        pool = np.unique(edges[:, [0, 2]])
    else:
        edges = train
        pool = np.arange(int(data["num_train_nodes"]))
    context = NeighborhoodGraph(edges, len(data["node_ids"]), len(data["relation_ids"]),
                                snapshot_path=args.run_dir / "snapshots" / args.split
                                if spec.get("sampler", "").startswith("gestaltdb") else None,
                                feature_mode=spec.get("feature_mode", "sampled"), compact=spec.get("sampler") == "gestaltdb-csr",
                                sampling_version=spec.get("sampling_version", 1))
    known = GraphKnownObjects(context, *(data[key] for key in ("valid", "test", "temporal_valid", "temporal_test") if key in data))
    result = evaluate(model, context, data[args.split], known, pool,
                      fanouts=spec["fanouts"], negatives=args.negatives, max_queries=args.triples,
                      candidate_policy=args.candidate_policy, include_details=True)
    result.update(split=args.split, checkpoint=latest,
                  context="train + inference support" if args.split == "test" else "train only")
    if args.split == "test":
        first_unseen = int(data["num_train_nodes"])
        result["unseen_entity_count"] = len(data["node_ids"]) - first_unseen
        result["all_test_queries_touch_unseen_entity"] = all(
            int(s) >= first_unseen or int(o) >= first_unseen for s, _, o in data["test"])
    output = args.run_dir / (f"gnn_{args.split}_evaluation.json" if args.candidate_policy == "uniform"
                             else f"gnn_{args.split}_{args.candidate_policy}_evaluation.json")
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != "details"}, indent=2))


if __name__ == "__main__":
    main()
