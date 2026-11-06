"""Checkpoint-backed quantitative results, learning curves and diagnostic t-SNE."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
import tensorflow as tf

from .inductive_data import GraphKnownObjects, NeighborhoodGraph, file_digest, sampled_candidates
from .inductive_eval import evaluate, metrics_from_details
from .inductive_model import InductiveAutoencoder


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize_groups(result, data, context):
    # Some unselected vertices also lose every training edge when their only
    # neighbors are held out. They too have no ID row and are genuinely unseen.
    unseen = range(int(data["num_train_nodes"]), context.num_nodes)
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


def shallow_evaluate(model, data, negatives, max_queries, context=None):
    rng = np.random.default_rng(12345)
    if context is None:
        context = NeighborhoodGraph(data["train"], len(data["node_ids"]), len(data["relation_ids"]))
    known = GraphKnownObjects(context, *(data[key] for key in ("valid", "test", "temporal_valid", "temporal_test") if key in data))
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
    prefix = "Mean-policy" if curves[-1].get("candidate_policy") == "both" else "GNN validation"
    axes[1].plot([row["step"] for row in curves], [row["mrr"] for row in curves], "o-", label=f"{prefix} MRR")
    axes[1].plot([row["step"] for row in curves], [row["hits10"] for row in curves], "o-", label=f"{prefix} Hits@10")
    if curves[-1].get("candidate_policy") == "both":
        for policy in ("uniform", "relation-range"):
            axes[1].plot([row["step"] for row in curves],
                         [row["per_policy"][policy]["mrr"] for row in curves], "--", label=f"{policy} MRR")
    axes[1].set(xlabel="Completed optimizer steps", ylabel="Sampled-filtered metric",
                title=f"{curves[-1].get('validation_split', 'valid')}: {curves[-1].get('candidate_policy', 'uniform')} candidates")
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
    # Do not build a Python dict for tens of millions of full-graph vertices.
    annotation = pairs[np.searchsorted(pairs[:, 0], chosen), 1]

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
    unseen = chosen >= int(data["num_train_nodes"])
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
    base = model.base_vectors(chosen[known]).numpy()
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
    return visualization_metrics(vectors, coordinates, annotation, unseen)


def visualization_metrics(vectors, coordinates, annotation, unseen):
    normalized = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)
    similarities = normalized @ normalized.T
    np.fill_diagonal(similarities, -np.inf)
    nearest = np.argmax(similarities, axis=1)
    nn_accuracy = np.mean(annotation[nearest] == annotation)
    return dict(nodes=len(vectors), unseen_nodes=int(unseen.sum()), seed=13,
                annotations="dominant relation in observed train+support context; diagnostic only",
                relation_1nn_agreement=float(nn_accuracy),
                tsne_trustworthiness=float(trustworthiness(vectors, coordinates,
                    n_neighbors=min(10, max(1, (len(vectors) - 1) // 2)))),
                mean_vector_standard_deviation=float(np.std(vectors, axis=0).mean()))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--triples", type=int, default=1000)
    parser.add_argument("--negatives", type=int, default=50)
    parser.add_argument("--tsne-nodes", type=int, default=2000)
    parser.add_argument("--checkpoint", choices=["best", "latest"], default=None,
                        help="Defaults to the run's export policy")
    parser.add_argument("--publish-gnn", action="store_true", help="Copy the small standalone GNN export into the report")
    parser.add_argument("--resume-report", action="store_true", help="Reuse completed ranking files, plots and per-stage metrics")
    parser.add_argument("--refresh-summary", action="store_true", help="Refresh Markdown and curves only; do not load graphs or models")
    args = parser.parse_args()
    if args.refresh_summary:
        summary = json.loads((args.output / "summary.json").read_text())
        plot_curves(args.run_dir, args.output)
        render_summary(summary, args.output)
        return
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    args.output.mkdir(parents=True, exist_ok=True)
    spec = json.loads((args.run_dir / "run.json").read_text())
    if file_digest(spec["dataset"]) != spec["dataset_digest"]:
        raise ValueError("dataset content differs from the trained run")
    data = np.load(spec["dataset"], allow_pickle=False)
    checkpoint_policy = args.checkpoint or spec.get("export_checkpoint", "best")
    checkpoint_path = tf.train.latest_checkpoint(str(args.run_dir / ("best" if checkpoint_policy == "best" else "checkpoints")))
    if checkpoint_path is None:
        raise ValueError("no validation-selected checkpoint")
    identity = dict(dataset_digest=spec["dataset_digest"], checkpoint=str(Path(checkpoint_path).resolve()),
                    checkpoint_index_digest=file_digest(checkpoint_path + ".index"),
                    triples=args.triples, negatives=args.negatives, tsne_nodes=args.tsne_nodes)
    state_path = args.output / "report_state.json"
    state = dict(identity=identity, results={})
    if state_path.exists():
        if not args.resume_report:
            raise ValueError("report state exists; use --resume-report or a new output directory")
        state = json.loads(state_path.read_text())
        if state["identity"] != identity:
            raise ValueError("report inputs/checkpoint changed; choose a new output directory")
    results = state["results"]

    def save_state():
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(state_path)

    def stage(key, compute):
        if key not in results:
            results[key] = compute()
            save_state()
        return results[key]

    # Explicit legacy resume adopts ranking files written before per-stage
    # metrics were introduced. Their GNN export must match the selected model.
    if args.resume_report:
        published = args.output / "gnn_saved_model" / "metadata.json"
        if published.exists() and Path(json.loads(published.read_text())["source_checkpoint"]).name != Path(checkpoint_path).name:
            raise ValueError("existing report belongs to a different checkpoint")
        for policy in ("uniform", "relation-range"):
            path = args.output / f"test_{policy}_details.json"
            if path.exists() and f"trained_{policy}" not in results:
                try:
                    result = json.loads(path.read_text())
                except json.JSONDecodeError:
                    continue  # interrupted rank-file write; recompute this stage
                if result["queries"] != min(args.triples, len(data["test"])) or result["candidate_policy"] != policy:
                    raise ValueError("existing ranking protocol differs from requested report")
                result.pop("details")
                results[f"trained_{policy}"] = result
        path = args.output / "no_message_details.json"
        if path.exists() and "no_message_relation_range" not in results:
            try:
                details = json.loads(path.read_text())
            except json.JSONDecodeError:
                details = None
            if details is not None:
                if len(details) != min(args.triples, len(data["test"])):
                    raise ValueError("existing ablation query count differs")
                results["no_message_relation_range"] = metrics_from_details(details, "relation-range")
    save_state()
    curves, losses = plot_curves(args.run_dir, args.output)
    edges = np.concatenate([data["train"], data["support"]])
    context = NeighborhoodGraph(edges, len(data["node_ids"]), len(data["relation_ids"]),
                                snapshot_path=args.run_dir / "snapshots" / "test"
                                if spec.get("sampler", "").startswith("gestaltdb") else None,
                                feature_mode=spec.get("feature_mode", "sampled"), compact=spec.get("sampler") == "gestaltdb-csr",
                                sampling_version=spec.get("sampling_version", 1))
    pool = np.unique(edges[:, [0, 2]])
    known = GraphKnownObjects(context, *(data[key] for key in ("valid", "test", "temporal_valid", "temporal_test") if key in data))
    tf.keras.utils.set_random_seed(spec["seed"])
    model = InductiveAutoencoder(**spec["model"])
    # Evaluate the untrained control with exactly the same candidate/context RNG.
    for policy in ("uniform", "relation-range"):
        stage(f"untrained_{policy}", lambda: evaluate(model, context, data["test"], known, pool,
            fanouts=spec["fanouts"], negatives=args.negatives, max_queries=args.triples,
            candidate_policy=policy))
    if args.publish_gnn:
        source = args.run_dir / "gnn_saved_model"
        exported = json.loads((source / "metadata.json").read_text())
        if Path(exported["source_checkpoint"]).name != Path(checkpoint_path).name:
            raise ValueError("standalone GNN export is stale; rerun the trainer's export with --resume")
        shutil.copytree(source, args.output / "gnn_saved_model", dirs_exist_ok=True)
    status = tf.train.Checkpoint(model=model).restore(checkpoint_path)
    status.assert_existing_objects_matched()
    status.expect_partial()
    for policy in ("uniform", "relation-range"):
        if f"trained_{policy}" in results:
            print(f"reusing trained {policy}: MRR={results[f'trained_{policy}']['mrr']:.4f}", flush=True)
            continue
        result = evaluate(model, context, data["test"], known, pool, fanouts=spec["fanouts"],
                          negatives=args.negatives, max_queries=args.triples,
                          candidate_policy=policy, include_details=True)
        result["inductive_groups"] = summarize_groups(result, data, context)
        (args.output / f"test_{policy}_details.json").write_text(json.dumps(result, indent=2))
        result.pop("details")
        results[f"trained_{policy}"] = result
        save_state()
        print(f"trained {policy}: MRR={result['mrr']:.4f}", flush=True)
    if "no_message_relation_range" not in results:
        result = evaluate(model, context, data["test"], known, pool,
            fanouts=spec["fanouts"], negatives=args.negatives, max_queries=args.triples,
            candidate_policy="relation-range", message_passing=False, include_details=True)
        no_message_details = result.pop("details")
        (args.output / "no_message_details.json").write_text(json.dumps(no_message_details))
        results["no_message_relation_range"] = result
        save_state()
    trained_details = json.loads((args.output / "test_relation-range_details.json").read_text())["details"]
    no_message_details = json.loads((args.output / "no_message_details.json").read_text())
    differences = np.asarray([1 / trained["rank"] - 1 / ablated["rank"]
                              for trained, ablated in zip(trained_details, no_message_details)])
    bootstrap = np.random.default_rng(13).choice(differences, (1000, len(differences)), replace=True).mean(axis=1)
    results["message_passing_gain"] = dict(mean_delta_mrr=float(differences.mean()),
                                           paired_bootstrap_ci95=np.quantile(bootstrap, [.025, .975]).tolist())
    save_state()
    if ("temporal_test" in data and len(data["temporal_test"])
            and any(f"temporal_test_{policy}" not in results for policy in ("uniform", "relation-range"))):
        temporal_context = NeighborhoodGraph(data["train"], len(data["node_ids"]), len(data["relation_ids"]),
            snapshot_path=args.run_dir / "snapshots" / "train" if spec.get("sampler", "").startswith("gestaltdb") else None,
            feature_mode=spec.get("feature_mode", "sampled"), compact=spec.get("sampler") == "gestaltdb-csr",
            sampling_version=spec.get("sampling_version", 1))
        temporal_known = GraphKnownObjects(temporal_context, *(data[key] for key in
            ("valid", "test", "temporal_valid", "temporal_test")))
        for policy in ("uniform", "relation-range"):
            stage(f"temporal_test_{policy}", lambda: evaluate(model, temporal_context, data["temporal_test"],
                temporal_known, np.arange(int(data["num_train_nodes"])), fanouts=spec["fanouts"],
                negatives=args.negatives, max_queries=args.triples, candidate_policy=policy))
        del temporal_context, temporal_known
    if "visualization" not in state:
        cached = args.output / "gnn_tsne_data.npz"
        if args.resume_report and cached.exists() and all((args.output / name).exists() for name in ("gnn_tsne.png", "shallow_tsne.png")):
            points = np.load(cached, allow_pickle=False)
            state["visualization"] = visualization_metrics(points["vectors"], points["coordinates"], points["relation"], points["unseen"])
        else:
            state["visualization"] = plot_embeddings(model, data, context, spec, args.output, args.tsne_nodes)
        save_state()
    visualization = state["visualization"]
    del context, known, edges, pool
    latest = tf.train.latest_checkpoint(str(args.run_dir / "checkpoints"))
    status = tf.train.Checkpoint(model=model).restore(latest)
    status.assert_existing_objects_matched()
    status.expect_partial()
    if "shallow_final_known_valid" not in results:
        shallow_context = NeighborhoodGraph(data["train"], len(data["node_ids"]), len(data["relation_ids"]),
            snapshot_path=args.run_dir / "snapshots" / "train" if spec.get("sampler", "").startswith("gestaltdb") else None,
            compact=spec.get("sampler") == "gestaltdb-csr")
        stage("shallow_final_known_valid", lambda: shallow_evaluate(model, data, args.negatives, args.triples, context=shallow_context))
        del shallow_context
    completed_steps = int(tf.train.load_checkpoint(latest).get_tensor("step/.ATTRIBUTES/VARIABLE_VALUE"))
    coverage = None
    selected_coverage = None
    if spec.get("entity_passes", 0):
        from .inductive_schedule import EntityCoverageSchedule
        schedule = EntityCoverageSchedule(int(data["num_train_nodes"]), spec["batch_size"],
                                          spec["entity_passes"], spec["seed"], phases=spec.get("coverage_phases"))
        coverage = schedule.progress(completed_steps)
        selected_steps = completed_steps if checkpoint_policy == "latest" else int(
            json.loads((args.run_dir / "best.json").read_text())["step"])
        selected_coverage = schedule.progress(selected_steps)
    seed_draws = coverage["supervised_anchor_draws"] if coverage else completed_steps * spec["batch_size"]
    summary = dict(run=spec, checkpoint=checkpoint_path, shallow_checkpoint=latest,
                   selected_checkpoint_policy=checkpoint_policy, coverage=coverage,
                   selected_model_coverage=selected_coverage,
                   best=json.loads((args.run_dir / "best.json").read_text()),
                   environment=json.loads((args.run_dir / "environment.json").read_text()),
                   splits={key: len(data[key]) for key in ("train", "valid", "support", "test", "temporal_valid", "temporal_test") if key in data},
                   num_nodes=len(data["node_ids"]), num_train_nodes=int(data["num_train_nodes"]),
                   num_unseen_nodes=len(data["node_ids"]) - int(data["num_train_nodes"]),
                   num_sampled_heldout_nodes=len(data["heldout_nodes"] if "heldout_nodes" in data else data["unseen_nodes"]),
                   num_relations=len(data["relation_ids"]),
                   completed_steps=completed_steps, positive_seed_draws=seed_draws,
                   seed_draw_equivalent_epochs=seed_draws / len(data["train"]),
                   validation_initial=curves[0], validation_final=curves[-1],
                   training_initial=losses[0], training_final=losses[-1],
                   visualization=visualization, results=results)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    render_summary(summary, args.output)
    print(f"report written to {args.output}", flush=True)


def render_summary(summary, output):
    results = summary["results"]
    checkpoint_path = summary["checkpoint"]
    completed_steps = summary["completed_steps"]
    table = ["# Inductive YAGO graph-autoencoder results", "",
             f"{'Validation-selected' if summary.get('selected_checkpoint_policy', 'best') == 'best' else 'Latest training'} checkpoint: `{checkpoint_path}`.", "",
             "All test metrics below are sampled-filtered tail prediction, not full ranking.", "",
             "| Test model / candidate policy | MRR | Hits@1 | Hits@10 |",
             "|---|---:|---:|---:|"]
    for name in ("untrained_uniform", "trained_uniform", "untrained_relation-range", "trained_relation-range", "no_message_relation_range"):
        result = results[name]
        table.append(f"| {name} | {result['mrr']:.4f} | {result['hits1']:.4f} | {result['hits10']:.4f} |")
    if "temporal_test_uniform" in results:
        table += ["", "## Chronological valid-time completion", "",
                  "Known-entity queries starting in 2019–2024; training-dated facts through 2010, validation 2011–2018.",
                  "Undated static background is allowed: this is not causal knowledge-acquisition forecasting.", "",
                  "| Temporal test policy | MRR | Hits@1 | Hits@10 | Frequency MRR (expected ties) |",
                  "|---|---:|---:|---:|---:|"]
        for policy in ("uniform", "relation-range"):
            result = results[f"temporal_test_{policy}"]
            table.append(f"| {policy} | {result['mrr']:.4f} | {result['hits1']:.4f} | "
                         f"{result['hits10']:.4f} | {result['frequency_mrr']:.4f} |")
    table += ["", "## Training budget and representations", "",
              f"{completed_steps:,} updates; {summary['splits']['train']:,} training facts; "
              f"{summary['seed_draw_equivalent_epochs']:.4f} seed-draw-equivalent epochs. "
              "The entire training graph is available as context; this is not an exhaustive supervised pass.",
              f"Final shallow-table known-entity validation MRR: {results['shallow_final_known_valid']['mrr']:.4f}.",
              "The transferable GNN and contextual decoder are distinct from the shallow ID tables."]
    if summary.get("selected_model_coverage"):
        progress = summary["selected_model_coverage"]
        table += ["", f"Selected GNN coverage: {progress['completed_entity_passes']} complete entity passes; "
                  f"{progress['supervised_anchor_draws']:,} supervised anchor draws; "
                  f"minimum {progress['guaranteed_minimum_supervised_appearances_per_entity']} positive endpoint appearances per training entity.",
                  "Coverage is not a convergence guarantee or a claim of two exhaustive edge epochs."]
    table += ["", "![Learning curves](learning_curves.png)", "", "![Contextual GNN t-SNE](gnn_tsne.png)",
              "", "![Shallow embeddings t-SNE](shallow_tsne.png)", "",
              "t-SNE labels describe observed neighborhoods and are diagnostic, not a semantic benchmark.",
              "See summary.json for confidence intervals, frequency/random controls, per-relation and support-degree groups."]
    table += ["", "## Candidate calibration and controls", "",
              "| Policy | GNN MRR | Frequency MRR (pessimistic ties) | Frequency MRR (expected ties) | Random expected MRR |",
              "|---|---:|---:|---:|---:|"]
    for policy in ("uniform", "relation-range"):
        result = results[f"trained_{policy}"]
        table.append(f"| {policy} | {result['mrr']:.4f} | {result['frequency_pessimistic_mrr']:.4f} | "
                     f"{result['frequency_mrr']:.4f} | {result['random_mrr']:.4f} |")
    if results["trained_uniform"]["mrr"] < results["trained_uniform"]["frequency_pessimistic_mrr"]:
        table += ["", "**Limitation:** this checkpoint underperforms the frequency control on global uniform candidates. "
                  "Strong relation-range results must not be presented as good candidate-independent completion. "
                  "Training-negative policy and checkpoint-selection policy matter."]
    cache_metrics = output / "cached_link_prediction.json"
    if cache_metrics.exists():
        cached = json.loads(cache_metrics.read_text())
        cache = json.loads((output / "cache_metadata.json").read_text())
        matrix = cache['matrix']
        table += ["", "## Complete all-node static GNN cache", "",
                  f"{matrix['shape'][0]:,} × {matrix['shape'][1]} {matrix['dtype']} vectors "
                  f"({matrix['file_bytes'] / 1e9:.1f} GB), stored outside Git at `{matrix['local_path']}`.",
                  f"All values finite: {matrix['all_finite']}. See cache_metadata.json for completion and model/dataset identity.",
                  "Held-out queries/reverses are globally scrubbed before encoding; these batched-context results are separate from per-query inference.", "",
                  "| Cache candidate policy | MRR | Hits@1 | Hits@10 | Mean candidates | Frequency MRR |",
                  "|---|---:|---:|---:|---:|---:|"]
        for policy, result in cached.items():
            table.append(f"| {policy} | {result['mrr']:.4f} | {result['hits1']:.4f} | "
                         f"{result['hits10']:.4f} | {result['mean_candidates']:.3f} | {result['frequency_mrr']:.4f} |")
        table += ["", "Degree-matched candidates control incoming-predicate degree; sparse bins reduce candidate counts, so compare against that policy's own controls."]
    (output / "README.md").write_text("\n".join(table) + "\n")


if __name__ == "__main__":
    main()
