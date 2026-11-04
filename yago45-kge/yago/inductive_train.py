"""Train a masked inductive graph autoencoder and a useful shallow KGE branch."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from .inductive_data import GraphKnownObjects, NeighborhoodGraph, mapping_digest
from .inductive_eval import evaluate
from .inductive_model import InductiveAutoencoder, infonce


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=5000, help="Total target steps, including restored steps")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--negatives", type=int, default=16)
    parser.add_argument("--fanouts", type=int, nargs="+", default=[12, 8])
    parser.add_argument("--dim", type=int, default=200)
    parser.add_argument("--base-dim", type=int, default=None,
                        help="Independent shallow-table width; defaults to --dim")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--shallow-weight", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-triples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--encoder", choices=["relational", "tf-gnns"], default="tf-gnns")
    parser.add_argument("--sampler", choices=["numpy", "gestaltdb", "gestaltdb-csr"], default="gestaltdb")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--negative-policy", choices=["uniform", "mixed", "relation-range"], default="uniform")
    parser.add_argument("--validation-policy", choices=["uniform", "relation-range", "both"], default="uniform")
    parser.add_argument("--feature-mode", choices=["sampled", "full"], default="full")
    parser.add_argument("--embedding-device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--shallow-optimizer", choices=["adam", "sgd", "sparse-adagrad"], default="adam")
    parser.add_argument("--shallow-learning-rate", type=float, default=None)
    parser.add_argument("--validation-split", choices=["valid", "temporal_valid"], default="valid")
    parser.add_argument("--no-export", action="store_true")
    args = parser.parse_args()
    gpus = tf.config.list_physical_devices("GPU")
    if args.require_gpu and not gpus:
        parser.error("GPU required but TensorFlow cannot see one; run via scripts/inductive.sh")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"TensorFlow devices: {gpus}", flush=True)
    if (min(args.steps, args.batch_size, args.negatives, args.checkpoint_every, args.eval_every,
            args.eval_triples, args.keep_checkpoints, args.dim, args.hidden_dim, *args.fanouts) <= 0
            or args.temperature <= 0 or args.learning_rate <= 0 or args.shallow_weight <= 0):
        parser.error("steps, dimensions, fanouts, intervals, weights and temperatures must be positive")
    data = np.load(args.dataset, allow_pickle=False)
    train = data["train"]
    if args.validation_split not in data or not len(data[args.validation_split]):
        parser.error(f"dataset has no nonempty {args.validation_split} split")
    num_nodes, num_relations = len(data["node_ids"]), len(data["relation_ids"])
    num_train_nodes = int(data["num_train_nodes"])
    base_dim = args.dim if args.base_dim is None else args.base_dim
    if base_dim <= 0 or base_dim % 2:
        parser.error("--base-dim must be a positive even integer")
    if args.shallow_learning_rate is not None and args.shallow_learning_rate <= 0:
        parser.error("--shallow-learning-rate must be positive")
    table_factor = {"sgd": 1, "adam": 3, "sparse-adagrad": 2}[args.shallow_optimizer]
    print(f"shallow table + optimizer slots: approximately {num_train_nodes * base_dim * 4 * table_factor / 2**30:.2f} GiB", flush=True)
    # Hash the exact dataset, not just its shape, to prevent silent split drift.
    digest = hashlib.sha256()
    with args.dataset.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    spec = dict(dataset=str(args.dataset.resolve()), dataset_digest=digest.hexdigest(),
                mapping_digest=mapping_digest(data),
                model=dict(num_train_nodes=num_train_nodes, num_relations=num_relations,
                           dim=args.dim, base_dim=base_dim, embedding_device=args.embedding_device,
                           hidden_dim=args.hidden_dim, layers=len(args.fanouts), encoder=args.encoder),
                sampler=args.sampler,
                shallow_optimizer=args.shallow_optimizer, shallow_learning_rate=args.shallow_learning_rate,
                validation_split=args.validation_split,
                validation_queries=args.eval_triples,
                feature_mode=args.feature_mode,
                negative_policy=args.negative_policy, validation_policy=args.validation_policy,
                fanouts=args.fanouts, batch_size=args.batch_size, negatives=args.negatives,
                learning_rate=args.learning_rate, temperature=args.temperature,
                shallow_weight=args.shallow_weight, seed=args.seed,
                features=f"log1p masked {args.feature_mode} directional relation counts + generic Entity constant",
                encoder=f"ID-independent {args.encoder} mean message passing",
                decoder="ComplEx", representation="normalized contextual GNN states + separate shallow tables")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.run_dir / "run.json"
    if args.resume:
        if not config_path.exists():
            parser.error("resume requires run.json and a checkpoint")
        saved = json.loads(config_path.read_text())
        saved.setdefault("negative_policy", "uniform")
        saved.setdefault("validation_policy", "uniform")
        saved.setdefault("feature_mode", "sampled")
        saved["model"].setdefault("base_dim", saved["model"]["dim"])
        saved["model"].setdefault("embedding_device", "gpu")
        saved.setdefault("shallow_optimizer", "adam")
        saved.setdefault("shallow_learning_rate", None)
        saved.setdefault("validation_split", "valid")
        saved.setdefault("validation_queries", args.eval_triples)
        if saved != spec:
            parser.error("resume configuration/dataset differs; reuse the original training arguments")
    elif config_path.exists() or any(args.run_dir.iterdir()):
        parser.error("run directory is not empty; use --resume or a new directory")
    else:
        config_path.write_text(json.dumps(spec, indent=2))
    versions = {name: importlib.metadata.version(name) for name in ("tensorflow", "keras", "gestaltdb", "tf_gnns", "numpy")}
    (args.run_dir / "environment.json").write_text(json.dumps(dict(versions=versions, gpus=[str(g) for g in gpus]), indent=2))

    tf.keras.utils.set_random_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = InductiveAutoencoder(**spec["model"])
    optimizer = tf.keras.optimizers.Adam(args.learning_rate)
    shallow_variables = model.entity_table.trainable_variables + model.base_relations.trainable_variables
    shallow_ids = {id(variable) for variable in shallow_variables}
    gnn_variables = [variable for variable in model.trainable_variables if id(variable) not in shallow_ids]
    separate_shallow = args.shallow_optimizer != "adam" or args.shallow_learning_rate is not None
    shallow_optimizer = None
    if separate_shallow:
        optimizer.build(gnn_variables)
        rate = args.shallow_learning_rate if args.shallow_learning_rate is not None else args.learning_rate
        if args.shallow_optimizer == "sparse-adagrad":
            from .sparse_optimizer import SparseAdagrad
            shallow_optimizer = SparseAdagrad(rate)
        else:
            shallow_optimizer = (tf.keras.optimizers.SGD(rate) if args.shallow_optimizer == "sgd"
                                 else tf.keras.optimizers.Adam(rate))
        shallow_optimizer.build(shallow_variables)
    else:
        optimizer.build(model.trainable_variables)
    step = tf.Variable(0, dtype=tf.int64, trainable=False)
    rng_state = tf.Variable("", dtype=tf.string, trainable=False)
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer, step=step, rng_state=rng_state)
    if shallow_optimizer is not None:
        checkpoint.shallow_optimizer = shallow_optimizer
    manager = tf.train.CheckpointManager(checkpoint, str(args.run_dir / "checkpoints"), max_to_keep=args.keep_checkpoints)
    best_manager = tf.train.CheckpointManager(tf.train.Checkpoint(model=model), str(args.run_dir / "best"), max_to_keep=1)
    best_path = args.run_dir / "best.json"
    best_mrr = json.loads(best_path.read_text())["mrr"] if args.resume and best_path.exists() else -1.0
    if args.resume:
        if manager.latest_checkpoint is None:
            parser.error("no checkpoint to resume")
        checkpoint.restore(manager.latest_checkpoint).assert_consumed()
        rng.bit_generator.state = json.loads(rng_state.numpy().decode())
        print(f"resumed {manager.latest_checkpoint} at step {int(step.numpy())}", flush=True)
    start = int(step.numpy())
    if start > args.steps:
        parser.error("--steps is less than the checkpoint's completed step count")
    curves_path = args.run_dir / "curves.jsonl"
    curves = []
    if args.resume and curves_path.exists():
        for line in curves_path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate an interrupted final append
            if row["step"] <= start:
                curves.append(row)
    curves_path.write_text("".join(json.dumps(row) + "\n" for row in curves))

    context = NeighborhoodGraph(train, num_nodes, num_relations,
                                snapshot_path=args.run_dir / "snapshots" / "train" if args.sampler.startswith("gestaltdb") else None,
                                feature_mode=args.feature_mode, compact=args.sampler == "gestaltdb-csr")
    # Training negatives are filtered against TRAIN ONLY; no test/support access.
    training_known = GraphKnownObjects(context)
    validation_known = GraphKnownObjects(context, data["valid"], data[args.validation_split])
    relation_pools = {int(r): np.unique(train[train[:, 1] == r, 2]) for r in np.unique(train[:, 1])}

    graph_signature = dict(features=tf.TensorSpec((None, 2 * num_relations + 1), tf.float32),
                           senders=tf.TensorSpec((None,), tf.int32),
                           receivers=tf.TensorSpec((None,), tf.int32), relations=tf.TensorSpec((None,), tf.int32))

    @tf.function(input_signature=[graph_signature, tf.TensorSpec((None, 3), tf.int32),
                                tf.TensorSpec((None, args.negatives, 3), tf.int32),
                                tf.TensorSpec((None, 3), tf.int32),
                                tf.TensorSpec((None, args.negatives, 3), tf.int32)])
    def update(graph, positive_local, negative_local, positive_global, negative_global):
        with tf.GradientTape() as tape:
            encoded = model.encode(graph)
            gnn_loss = infonce(model.contextual_scores(encoded, positive_local),
                               model.contextual_scores(encoded, negative_local), args.temperature)
            base_loss = infonce(model.shallow_scores(positive_global),
                                model.shallow_scores(negative_global), args.temperature)
            loss = gnn_loss + args.shallow_weight * base_loss
        gradients = tape.gradient(loss, model.trainable_variables)
        # tf_gnns also computes decoded edge outputs. Their final output-only
        # layers are not used by this node-decoding loss, so have no gradients.
        pairs = [(g, v) for g, v in zip(gradients, model.trainable_variables) if g is not None]
        if shallow_optimizer is not None:
            optimizer.apply_gradients((g, v) for g, v in pairs if id(v) not in shallow_ids)
            shallow_optimizer.apply_gradients((g, v) for g, v in pairs if id(v) in shallow_ids)
        else:
            optimizer.apply_gradients(pairs)
        return loss, gnn_loss, base_loss

    def save():
        rng_state.assign(json.dumps(rng.bit_generator.state))
        path = manager.save(checkpoint_number=int(step.numpy()))
        print(f"saved {path}", flush=True)

    def validate(current, losses=None):
        nonlocal best_mrr
        policies = ("uniform", "relation-range") if args.validation_policy == "both" else (args.validation_policy,)
        by_policy = {policy: evaluate(model, context, data[args.validation_split], validation_known,
                                     np.arange(num_train_nodes), fanouts=args.fanouts,
                                     max_queries=args.eval_triples, candidate_policy=policy) for policy in policies}
        if args.validation_policy == "both":
            metrics = dict(protocol="balanced uniform/range sampled-filtered tail prediction",
                           candidate_policy="both", per_policy=by_policy,
                           **{key: float(np.mean([result[key] for result in by_policy.values()]))
                              for key in ("mrr", "hits1", "hits10")})
        else:
            metrics = by_policy[policies[0]]
        metrics.update(step=current, loss=float(losses[0]) if losses is not None else None,
                       validation_split=args.validation_split,
                       gnn_loss=float(losses[1]) if losses is not None else None,
                       shallow_loss=float(losses[2]) if losses is not None else None)
        with curves_path.open("a") as handle:
            handle.write(json.dumps(metrics) + "\n")
        print(json.dumps(metrics), flush=True)
        if metrics["mrr"] > best_mrr:
            best_mrr = metrics["mrr"]
            best_checkpoint = best_manager.save(checkpoint_number=current)
            best_path.write_text(json.dumps(dict(step=current, mrr=best_mrr, checkpoint=best_checkpoint), indent=2))
        return metrics

    if not args.resume:
        validate(0)

    train_log = args.run_dir / "train_losses.jsonl"
    if args.resume and train_log.exists():
        records = []
        for line in train_log.read_text().splitlines():
            try:
                row = json.loads(line)
                if row["step"] <= start:
                    records.append(row)
            except json.JSONDecodeError:
                continue
        train_log.write_text("".join(json.dumps(row) + "\n" for row in records))
    t0 = time.perf_counter()

    try:
        for current in range(start + 1, args.steps + 1):
            positive = train[rng.integers(len(train), size=args.batch_size)]
            negative = np.repeat(positive[:, None, :], args.negatives, axis=1)
            # Filtered tail corruptions; bounded retries fail loudly rather than
            # quietly labeling a known fact as false in dense relation ranges.
            for i, (s, r, _) in enumerate(positive):
                forbidden = training_known[(int(s), int(r))]
                if len(forbidden) >= num_train_nodes:
                    raise ValueError("query has no valid negative tails")
                for j in range(args.negatives):
                    use_range = (args.negative_policy == "relation-range"
                                 or (args.negative_policy == "mixed" and rng.random() < 0.5))
                    pool = relation_pools[int(r)]
                    # A singleton/saturated range has no false tail. Fall back
                    # to a global corruption instead of labeling true facts false.
                    if use_range and all(int(x) in forbidden for x in pool):
                        use_range = False
                    for _ in range(10000):
                        candidate = int(rng.choice(pool)) if use_range else int(rng.integers(num_train_nodes))
                        if candidate not in forbidden:
                            negative[i, j, 2] = candidate
                            break
                    else:
                        raise ValueError("negative rejection limit reached")
            roots = np.unique(np.r_[positive[:, [0, 2]].ravel(), negative[..., [0, 2]].ravel()])
            ids, graph = context.sample(roots, args.fanouts, rng, masked=positive)

            def localize(triples):
                local = triples.copy()
                local[..., 0] = np.searchsorted(ids, triples[..., 0])
                local[..., 2] = np.searchsorted(ids, triples[..., 2])
                return local.astype(np.int32)

            losses = update({k: tf.convert_to_tensor(v) for k, v in graph.items()},
                            localize(positive), localize(negative), positive.astype(np.int32), negative.astype(np.int32))
            step.assign(current)
            if current % 50 == 0 or current == start + 1:
                elapsed = time.perf_counter() - t0
                row = dict(step=current, loss=float(losses[0]), gnn_loss=float(losses[1]),
                           shallow_loss=float(losses[2]), elapsed_seconds=elapsed,
                           steps_per_second=(current - start) / max(elapsed, 1e-9))
                with train_log.open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                print(f"step={current} loss={float(losses[0]):.4f} "
                      f"gnn={float(losses[1]):.4f} shallow={float(losses[2]):.4f} "
                      f"steps/s={row['steps_per_second']:.2f}", flush=True)
            if current % args.checkpoint_every == 0 or current == args.steps:
                save()  # checkpoint BEFORE potentially expensive evaluation/export
            if current % args.eval_every == 0 or current == args.steps:
                validate(current, losses)
    except KeyboardInterrupt:
        # Only completed updates count. SIGKILL/power loss uses periodic saves.
        save()
        raise

    if not args.no_export:
        np.savez_compressed(args.run_dir / "shallow_embeddings.npz",
                            entity=model.entity_table.get_weights()[0],
                            relation=model.base_relations.get_weights()[0],
                            node_ids=data["node_ids"][:num_train_nodes], relation_ids=data["relation_ids"])
        metadata = dict(score_function="complex", entity_key="entity",
                        relation_key="relation", representation="shallow KGE tables",
                        entity_dim=base_dim, relation_dim=base_dim,
                        gnn_checkpoint=manager.latest_checkpoint)
        if "node_mapping_path" in data:
            metadata["node_mapping_path"] = str(data["node_mapping_path"])
        (args.run_dir / "artifact_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"completed step {int(step.numpy())}; contextual inference: python -m yago.inductive_eval "
          f"--run-dir {args.run_dir}", flush=True)


if __name__ == "__main__":
    main()
