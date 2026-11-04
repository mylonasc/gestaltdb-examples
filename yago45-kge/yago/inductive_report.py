"""Checkpoint-backed quantitative results, learning curves and diagnostic t-SNE."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
import tensorflow as tf

from .inductive_data import GraphKnownObjects, NeighborhoodGraph, mapping_digest, sampled_candidates
from .inductive_eval import evaluate
from .inductive_model import InductiveAutoencoder


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize_groups(result, data, context):
    unseen = set(map(int, data["unseen_nodes"]))
    degree = np.bincount(context.edges[:, [0, 2]].ravel(), minlength=context.num_nodes)
    groups = {}
    for row in result["details"]:
        s, o = row["subject"], row["object"]
        labels = ["unseen_tail" if o in unseen else "seen_tail",
                  "unseen_subject" if s in unseen else "seen_subject"]
        incident = context.incident(o)
        reverse_copies = np.sum((context.rel[incident] == row["relation"]) & (context.dst[incident] == s))
        unseen_degree = min([max(0, int(degree[x]) - int(reverse_copies)) for x in (s, o) if x in unseen] or [0])
        labels.append("support_degree_0" if unseen_degree == 0 else "support_degree_1" if unseen_degree == 1 else
                      "support_degree_2_to_5" if unseen_degree <= 5 else "support_degree_6_plus")
        for label in labels:
            groups.setdefault(label, []).append(row["rank"])
    return {name: dict(queries=len(ranks), mrr=float(np.mean(1 / np.asarray(ranks))),
                       hits10=float(np.mean(np.asarray(ranks) <= 10))) for name, ranks in groups.items()}


def shallow_evaluate(model, data, negatives, max_queries):
    rng = np.random.default_rng(12345)
    context = NeighborhoodGraph(data["train"], len(data["node_ids"]), len(data["relation_ids"]))
    known = GraphKnownObjects(context, data["valid"])
    pool = np.arange(int(data["num_train_nodes"]))
    valid = data["valid"]
    selected = rng.choice(len(valid), min(max_queries, len(valid)), replace=False)
    ranks = []
    for query in valid[selected]:
        s, r, _ = map(int, query)
        candidates = sampled_candidates(query, pool, negatives, known, rng)
        triples = np.stack([np.full(len(candidates), s), np.full(len(candidates), r), candidates], axis=1)
        scores = model.shallow_scores(tf.convert_to_tensor(triples, tf.int32)).numpy()
        ranks.append(1 + np.sum(scores[1:] >= scores[0]))
    ranks = np.asarray(ranks)
    return dict(protocol="shallow known-entity validation, uniform sampled-filtered tails",
                queries=len(ranks), mrr=float(np.mean(1 / ranks)), hits1=float(np.mean(ranks == 1)),
                hits10=float(np.mean(ranks <= 10)))


def plot_curves(run_dir, output):
    curves = read_jsonl(run_dir / "curves.jsonl")
    losses = read_jsonl(run_dir / "train_losses.jsonl")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for key, label in (("loss", "joint"), ("gnn_loss", "GNN InfoNCE"), ("shallow_loss", "shallow InfoNCE")):
        axes[0].plot([row["step"] for row in losses], [row[key] for row in losses], label=label, alpha=.8)
    axes[0].set(xlabel="Completed optimizer steps", ylabel="Training loss", title="Masked autoencoding objectives")
    axes[0].legend()
    axes[1].plot([row["step"] for row in curves], [row["mrr"] for row in curves], "o-", label="GNN validation MRR")
    axes[1].plot([row["step"] for row in curves], [row["hits10"] for row in curves], "o-", label="GNN validation Hits@10")
    axes[1].set(xlabel="Completed optimizer steps", ylabel="Sampled-filtered metric",
                title=f"Validation: {curves[-1].get('candidate_policy', 'uniform')} candidates")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=170)
    plt.close(fig)
    return curves, losses


def plot_embeddings(model, data, context, spec, output, max_nodes):
    rng = np.random.default_rng(13)
    # Balanced diagnostic by the most common observed outgoing relation.
    # Labels are visualization annotations, not held-out semantic accuracy.
    src, rel, _ = context.edges.T
    pairs, counts = np.unique(np.stack([src, rel], axis=1), axis=0, return_counts=True)
    order = np.lexsort((pairs[:, 1], -counts, pairs[:, 0]))
    pairs = pairs[order]
    _, first = np.unique(pairs[:, 0], return_index=True)
    pairs = pairs[first]
    labels, frequencies = np.unique(pairs[:, 1], return_counts=True)
    top = labels[np.argsort(frequencies)[-8:]]
    chosen = []
    for label in top:
        group = pairs[pairs[:, 1] == label, 0]
        chosen.extend(rng.choice(group, min(len(group), max_nodes // len(top)), replace=False).tolist())
    chosen = np.asarray(chosen, dtype=np.int64)
    label_map = dict(pairs.tolist())
    annotation = np.asarray([label_map[int(node)] for node in chosen])

    @tf.function(reduce_retracing=True)
    def encode(graph):
        return model.encode(graph)

    vectors = []
    for start in range(0, len(chosen), 64):
        roots = chosen[start:start + 64]
        ids, graph = context.sample(roots, spec["fanouts"], rng)
        states = encode({key: tf.convert_to_tensor(value) for key, value in graph.items()}).numpy()
        vectors.append(states[np.searchsorted(ids, roots)])
    vectors = np.concatenate(vectors)
    components = min(50, vectors.shape[1], len(vectors) - 1)
    projected = PCA(n_components=components, random_state=13).fit_transform(vectors)
    coordinates = TSNE(n_components=2, perplexity=min(30, (len(vectors) - 1) / 3),
                       init="pca", learning_rate="auto", random_state=13).fit_transform(projected)
    unseen = np.isin(chosen, data["unseen_nodes"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for i, label in enumerate(top):
        mask = annotation == label
        axes[0].scatter(*coordinates[mask].T, s=7, alpha=.6, color=plt.get_cmap("tab10")(i),
                        label=str(data["relation_ids"][label]))
    axes[0].legend(fontsize=7, markerscale=2)
    axes[0].set_title("GNN embeddings: observed dominant outgoing relation")
    for mask, label, color in ((~unseen, "Other context entities", "#9b9b9b"),
                               (unseen, "Held-out unseen entities", "#d33f49")):
        axes[1].scatter(*coordinates[mask].T, s=7, alpha=.6, color=color, label=label)
    axes[1].legend(fontsize=8, markerscale=2)
    axes[1].set_title("Same t-SNE: unseen entities encoded without ID rows")
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle("Diagnostic only: 2D clusters do not measure link-prediction quality", fontsize=10)
    fig.tight_layout()
    fig.savefig(output / "gnn_tsne.png", dpi=180)
    plt.close(fig)
    np.savez_compressed(output / "gnn_tsne_data.npz", node_ids=data["node_ids"][chosen],
                        vectors=vectors, coordinates=coordinates, relation=annotation, unseen=unseen)
    # Also show the separately useful shallow embeddings, on known entities only.
    known = chosen < int(data["num_train_nodes"])
    base = model.entity_table(chosen[known]).numpy()
    base_coords = TSNE(n_components=2, perplexity=min(30, (len(base) - 1) / 3),
                       init="pca", learning_rate="auto", random_state=13).fit_transform(base)
    fig, axis = plt.subplots(figsize=(7, 5))
    for i, label in enumerate(top):
        mask = annotation[known] == label
        axis.scatter(*base_coords[mask].T, s=7, alpha=.6, color=plt.get_cmap("tab10")(i),
                     label=str(data["relation_ids"][label]))
    axis.set(title="Shallow ComplEx embeddings (known entities; cosine not trained)", xticks=[], yticks=[])
    axis.legend(fontsize=7, markerscale=2)
    fig.tight_layout()
    fig.savefig(output / "shallow_tsne.png", dpi=170)
    plt.close(fig)
    normalized = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)
    similarities = normalized @ normalized.T
    np.fill_diagonal(similarities, -np.inf)
    nearest = np.argmax(similarities, axis=1)
    nn_accuracy = np.mean(annotation[nearest] == annotation)
    return dict(nodes=len(chosen), unseen_nodes=int(unseen.sum()), seed=13,
                annotations="dominant relation in observed train+support context; diagnostic only",
                relation_1nn_agreement=float(nn_accuracy),
                tsne_trustworthiness=float(trustworthiness(vectors, coordinates, n_neighbors=10)),
                mean_vector_standard_deviation=float(np.std(vectors, axis=0).mean()))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--triples", type=int, default=1000)
    parser.add_argument("--negatives", type=int, default=50)
    parser.add_argument("--tsne-nodes", type=int, default=2000)
    args = parser.parse_args()
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    args.output.mkdir(parents=True, exist_ok=True)
    spec = json.loads((args.run_dir / "run.json").read_text())
    data = np.load(spec["dataset"], allow_pickle=False)
    if mapping_digest(data) != spec["mapping_digest"]:
        raise ValueError("dataset mapping mismatch")
    curves, losses = plot_curves(args.run_dir, args.output)
    edges = np.concatenate([data["train"], data["support"]])
    context = NeighborhoodGraph(edges, len(data["node_ids"]), len(data["relation_ids"]),
                                snapshot_path=args.run_dir / "snapshots" / "test"
                                if spec.get("sampler") == "gestaltdb" else None,
                                feature_mode=spec.get("feature_mode", "sampled"))
    pool = np.unique(edges[:, [0, 2]])
    known = GraphKnownObjects(context, *(data[key] for key in ("valid", "test", "temporal_valid", "temporal_test") if key in data))
    tf.keras.utils.set_random_seed(spec["seed"])
    model = InductiveAutoencoder(**spec["model"])
    results = {}
    # Evaluate the untrained control with exactly the same candidate/context RNG.
    for policy in ("uniform", "relation-range"):
        results[f"untrained_{policy}"] = evaluate(model, context, data["test"], known, pool,
            fanouts=spec["fanouts"], negatives=args.negatives, max_queries=args.triples,
            candidate_policy=policy)
    checkpoint_path = tf.train.latest_checkpoint(str(args.run_dir / "best"))
    if checkpoint_path is None:
        raise ValueError("no validation-selected checkpoint")
    status = tf.train.Checkpoint(model=model).restore(checkpoint_path)
    status.assert_existing_objects_matched()
    status.expect_partial()
    for policy in ("uniform", "relation-range"):
        result = evaluate(model, context, data["test"], known, pool, fanouts=spec["fanouts"],
                          negatives=args.negatives, max_queries=args.triples,
                          candidate_policy=policy, include_details=True)
        result["inductive_groups"] = summarize_groups(result, data, context)
        (args.output / f"test_{policy}_details.json").write_text(json.dumps(result, indent=2))
        result.pop("details")
        results[f"trained_{policy}"] = result
        print(f"trained {policy}: MRR={result['mrr']:.4f}", flush=True)
    results["no_message_relation_range"] = evaluate(model, context, data["test"], known, pool,
        fanouts=spec["fanouts"], negatives=args.negatives, max_queries=args.triples,
        candidate_policy="relation-range", message_passing=False)
    if "temporal_test" in data and len(data["temporal_test"]):
        temporal_context = NeighborhoodGraph(data["train"], len(data["node_ids"]), len(data["relation_ids"]),
            snapshot_path=args.run_dir / "snapshots" / "train" if spec.get("sampler") == "gestaltdb" else None,
            feature_mode=spec.get("feature_mode", "sampled"))
        temporal_known = GraphKnownObjects(temporal_context, *(data[key] for key in
            ("valid", "test", "temporal_valid", "temporal_test")))
        for policy in ("uniform", "relation-range"):
            results[f"temporal_test_{policy}"] = evaluate(model, temporal_context, data["temporal_test"],
                temporal_known, np.arange(int(data["num_train_nodes"])), fanouts=spec["fanouts"],
                negatives=args.negatives, max_queries=args.triples, candidate_policy=policy)
        del temporal_context, temporal_known
    visualization = plot_embeddings(model, data, context, spec, args.output, args.tsne_nodes)
    latest = tf.train.latest_checkpoint(str(args.run_dir / "checkpoints"))
    status = tf.train.Checkpoint(model=model).restore(latest)
    status.assert_existing_objects_matched()
    status.expect_partial()
    results["shallow_final_known_valid"] = shallow_evaluate(model, data, args.negatives, args.triples)
    summary = dict(run=spec, checkpoint=checkpoint_path, shallow_checkpoint=latest,
                   best=json.loads((args.run_dir / "best.json").read_text()),
                   environment=json.loads((args.run_dir / "environment.json").read_text()),
                   splits={key: len(data[key]) for key in ("train", "valid", "support", "test", "temporal_valid", "temporal_test") if key in data},
                   num_nodes=len(data["node_ids"]), num_train_nodes=int(data["num_train_nodes"]),
                   num_unseen_nodes=len(data["unseen_nodes"]), num_relations=len(data["relation_ids"]),
                   validation_initial=curves[0], validation_final=curves[-1],
                   training_initial=losses[0], training_final=losses[-1],
                   visualization=visualization, results=results)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    table = ["# Inductive YAGO graph-autoencoder results", "",
             f"Validation-selected checkpoint: `{checkpoint_path}`.", "",
             "All test metrics below are sampled-filtered tail prediction, not full ranking.", "",
             "| Test model / candidate policy | MRR | Hits@1 | Hits@10 |",
             "|---|---:|---:|---:|"]
    for name in ("untrained_uniform", "trained_uniform", "untrained_relation-range", "trained_relation-range", "no_message_relation_range"):
        result = results[name]
        table.append(f"| {name} | {result['mrr']:.4f} | {result['hits1']:.4f} | {result['hits10']:.4f} |")
    table += ["", "![Learning curves](learning_curves.png)", "", "![Contextual GNN t-SNE](gnn_tsne.png)",
              "", "![Shallow embeddings t-SNE](shallow_tsne.png)", "",
              "t-SNE labels describe observed neighborhoods and are diagnostic, not a semantic benchmark.",
              "See summary.json for confidence intervals, frequency/random controls, per-relation and support-degree groups."]
    (args.output / "README.md").write_text("\n".join(table) + "\n")
    print(f"report written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
