"""Train a masked inductive graph autoencoder and a useful shallow KGE branch."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import signal
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from .inductive_data import GraphKnownObjects, NeighborhoodGraph, file_digest, mapping_digest
from .inductive_eval import evaluate
from .inductive_model import InductiveAutoencoder, infonce
from .inductive_batch import prepare_batch
from .inductive_parallel import BATCH_RNG


def reconcile_completed_step(completed, *optimizer_steps):
    """Handle legacy SIGINT between a completed update and step.assign()."""
    if all(value == completed for value in optimizer_steps):
        return completed
    if all(value == completed + 1 for value in optimizer_steps):
        return completed + 1
    raise ValueError(f"inconsistent checkpoint: completed={completed}, optimizer steps={optimizer_steps}")


def latest_checkpoint_step(directory: Path):
    """Read the latest checkpoint number without initializing TensorFlow/CUDA."""
    state = directory / "checkpoint"
    if state.exists():
        for line in state.read_text().splitlines():
            if line.startswith("model_checkpoint_path:"):
                name = line.split('"')[1]
                return int(name.rsplit("-", 1)[1])
    indices = list(directory.glob("ckpt-*.index"))
    return max((int(path.name.rsplit("-", 1)[1].split(".")[0]) for path in indices), default=None)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=5000, help="Total target steps, including restored steps")
    parser.add_argument("--entity-passes", type=int, default=0,
                        help="Cover every training entity this many times; derives --steps instead of random fact sampling")
    parser.add_argument("--export-checkpoint", choices=["best", "latest"], default=None,
                        help="Default: latest for entity coverage, best for random training")
    parser.add_argument("--residual", action=argparse.BooleanOptionalAction, default=None,
                        help="Per-message-step residual connections (tf-gnns already defaults to residual)")
    parser.add_argument("--resume-batch-size", type=int, default=None,
                        help="Explicit coverage-preserving batch-size migration at the latest checkpoint")
    parser.add_argument("--upgrade-sampler", action="store_true",
                        help="Explicitly allow a checkpointed v2-to-v3 sampler upgrade; records provenance")
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
    parser.add_argument("--sampling-version", choices=[1, 2, 3], type=int, default=None)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--negative-policy", choices=["uniform", "mixed", "relation-range"], default="uniform")
    parser.add_argument("--validation-policy", choices=["uniform", "relation-range", "both"], default="uniform")
    parser.add_argument("--feature-mode", choices=["sampled", "full"], default="full")
    parser.add_argument("--embedding-device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--shallow-optimizer", choices=["adam", "sgd", "sparse-adagrad"], default="adam")
    parser.add_argument("--shallow-learning-rate", type=float, default=None)
    parser.add_argument("--validation-split", choices=["valid", "temporal_valid"], default="valid")
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--profile-batches", type=int, nargs="+", default=None,
                        help="Resume-only ephemeral batch benchmarks; never save updates or advance coverage")
    parser.add_argument("--profile-updates", type=int, default=20)
    parser.add_argument("--profile-warmup", type=int, default=3)
    parser.add_argument("--profile-output", type=Path, default=None)
    parser.add_argument("--profile-sampling-version", choices=[2, 3], type=int, default=None,
                        help="Benchmark a sampler without changing the run recipe")
    parser.add_argument("--prefetch-workers", type=int, default=0,
                        help="Forked shared-memory batch producers (0 disables)")
    parser.add_argument("--prefetch-slots", type=int, default=None,
                        help="Shared-memory batch slots; defaults to twice the workers")
    parser.add_argument("--prefetch-max-nodes", type=int, default=1_048_576,
                        help="Capacity guard per shared-memory slot (pages are lazy)")
    parser.add_argument("--prefetch-max-edges", type=int, default=2_097_152,
                        help="Capacity guard per shared-memory slot (pages are lazy)")
    args = parser.parse_args()
    if args.profile_batches is not None:
        if (not args.resume or args.profile_output is None or min(args.profile_batches) < 1
                or args.profile_updates < 1 or args.profile_warmup < 1):
            parser.error("profiling requires --resume, --profile-output and positive batch sizes/update counts")
    elif args.profile_sampling_version is not None:
        parser.error("--profile-sampling-version requires --profile-batches")
    if (args.resume_batch_size is not None or args.upgrade_sampler) and (not args.resume or args.profile_batches):
        parser.error("performance migration requires --resume and cannot be combined with profiling")
    if args.prefetch_workers < 0:
        parser.error("--prefetch-workers cannot be negative")
    if args.prefetch_workers and args.profile_batches is not None:
        parser.error("--prefetch-workers cannot be combined with profiling")
    if args.prefetch_slots is not None and args.prefetch_slots < max(2, 2 * args.prefetch_workers):
        parser.error("--prefetch-slots must be at least twice --prefetch-workers")
    if (args.prefetch_max_nodes < 1 or args.prefetch_max_edges < 1):
        parser.error("prefetch buffer capacities must be positive")
    phases = None
    migration_step = None
    if args.resume and (args.run_dir / "run.json").exists():
        prior = json.loads((args.run_dir / "run.json").read_text())
        phases = prior.get("coverage_phases")
        if args.resume_batch_size is not None:
            if args.resume_batch_size <= 0 or not args.entity_passes or not prior.get("entity_passes"):
                parser.error("batch migration requires positive batch size and an entity-coverage run")
            if args.resume_batch_size != prior["batch_size"]:
                from .inductive_schedule import EntityCoverageSchedule
                source = latest_checkpoint_step(args.run_dir / "checkpoints")
                if source is None:
                    parser.error("batch migration needs a complete checkpoint")
                migration_step = int(source)
                old_schedule = EntityCoverageSchedule(prior["model"]["num_train_nodes"], prior["batch_size"],
                    prior["entity_passes"], prior["seed"], phases=phases)
                draws = old_schedule.progress(migration_step)["supervised_anchor_draws"]
                phases = [phase for phase in old_schedule.phases if phase["start_step"] < migration_step] + [dict(start_step=migration_step,
                    start_anchor_draws=draws, batch_size=args.resume_batch_size)]
                args.batch_size = args.resume_batch_size
    if args.entity_passes < 0:
        parser.error("--entity-passes cannot be negative")
    export_checkpoint = args.export_checkpoint or ("latest" if args.entity_passes else "best")
    sampling_version = args.sampling_version
    if sampling_version is None and args.resume and (args.run_dir / "run.json").exists():
        sampling_version = json.loads((args.run_dir / "run.json").read_text()).get("sampling_version", 1)
    sampling_version = 2 if sampling_version is None else sampling_version
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
    schedule = None
    if args.entity_passes:
        from .inductive_schedule import EntityCoverageSchedule
        schedule = EntityCoverageSchedule(num_train_nodes, args.batch_size, args.entity_passes, args.seed, phases=phases)
        args.steps = schedule.total_steps
        print(f"entity coverage: {args.entity_passes} passes × {num_train_nodes:,} entities; "
              f"{args.steps:,} updates (overrides --steps); export={export_checkpoint}", flush=True)
    base_dim = args.dim if args.base_dim is None else args.base_dim
    if base_dim <= 0 or base_dim % 2:
        parser.error("--base-dim must be a positive even integer")
    if args.shallow_learning_rate is not None and args.shallow_learning_rate <= 0:
        parser.error("--shallow-learning-rate must be positive")
    table_factor = {"sgd": 1, "adam": 3, "sparse-adagrad": 2}[args.shallow_optimizer]
    print(f"shallow table + optimizer slots: approximately {num_train_nodes * base_dim * 4 * table_factor / 2**30:.2f} GiB", flush=True)
    # Hash the exact dataset, not just its shape, to prevent silent split drift.
    spec = dict(dataset=str(args.dataset.resolve()), dataset_digest=file_digest(args.dataset),
                mapping_digest=mapping_digest(data),
                model=dict(num_train_nodes=num_train_nodes, num_relations=num_relations,
                           dim=args.dim, base_dim=base_dim, embedding_device=args.embedding_device,
                           hidden_dim=args.hidden_dim, layers=len(args.fanouts), encoder=args.encoder),
                sampler=args.sampler,
                sampling_version=sampling_version,
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
    if args.residual is not None:
        spec["model"]["residual"] = args.residual
    if schedule is not None or args.export_checkpoint is not None:
        spec.update(entity_passes=args.entity_passes, export_checkpoint=export_checkpoint,
                    positive_schedule="entity-affine-v1" if schedule else "random-facts")
    if phases is not None:
        spec["coverage_phases"] = phases
    if args.prefetch_workers:
        spec.update(prefetch_workers=args.prefetch_workers,
                    prefetch_slots=args.prefetch_slots or 2 * args.prefetch_workers,
                    prefetch_max_nodes=args.prefetch_max_nodes,
                    prefetch_max_edges=args.prefetch_max_edges,
                    batch_rng=BATCH_RNG)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.run_dir / "run.json"
    migration = None
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
        saved.setdefault("sampling_version", 1)
        if saved != spec:
            allowed = dict(saved)
            if args.resume_batch_size is not None:
                allowed["batch_size"] = args.batch_size
                if phases is not None:
                    allowed["coverage_phases"] = phases
            if args.upgrade_sampler and saved["sampling_version"] == 2 and sampling_version == 3:
                allowed["sampling_version"] = 3
            if args.prefetch_workers:
                allowed["prefetch_workers"] = args.prefetch_workers
                allowed["prefetch_slots"] = args.prefetch_slots or 2 * args.prefetch_workers
                allowed["prefetch_max_nodes"] = args.prefetch_max_nodes
                allowed["prefetch_max_edges"] = args.prefetch_max_edges
                allowed["batch_rng"] = BATCH_RNG
            if allowed != spec:
                parser.error("resume configuration/dataset differs; reuse the original training arguments")
            migration = dict(previous_recipe=saved, new_recipe=spec,
                             reason="explicit throughput migration; preserve optimizer weights and all scheduled entity anchors",
                             batch_rng=BATCH_RNG if args.prefetch_workers else saved.get("batch_rng"))
    elif config_path.exists() or any(args.run_dir.iterdir()):
        parser.error("run directory is not empty; use --resume or a new directory")
    else:
        config_path.write_text(json.dumps(spec, indent=2))

    # Build the read-only training graph before TensorFlow initializes CUDA.
    # Forked producers inherit these pages copy-on-write; see inductive_parallel.
    context = NeighborhoodGraph(train, num_nodes, num_relations,
                                snapshot_path=args.run_dir / "snapshots" / "train" if args.sampler.startswith("gestaltdb") else None,
                                feature_mode=args.feature_mode, compact=args.sampler == "gestaltdb-csr",
                                sampling_version=sampling_version)
    if args.profile_sampling_version is not None:
        context.sampling_version = args.profile_sampling_version
    if schedule is not None:
        schedule.validate_graph(context)
    # Training negatives are filtered against TRAIN ONLY; no test/support access.
    training_known = GraphKnownObjects(context)
    validation_known = GraphKnownObjects(context, data["valid"], data[args.validation_split])
    relation_pools = {int(r): np.unique(train[train[:, 1] == r, 2]) for r in np.unique(train[:, 1])}

    producer_pool = None
    if args.prefetch_workers and args.profile_batches is None:
        from .inductive_parallel import BatchProducerPool
        producer_pool = BatchProducerPool(
            context=context, training_known=training_known, relation_pools=relation_pools,
            num_train_nodes=num_train_nodes, schedule=schedule,
            batch_size=args.batch_size, negatives=args.negatives,
            negative_policy=args.negative_policy, fanouts=args.fanouts, seed=args.seed,
            end_step=args.steps, workers=args.prefetch_workers,
            slots=args.prefetch_slots, max_nodes=args.prefetch_max_nodes,
            max_edges=args.prefetch_max_edges,
        )
        producer_pool.start()
        print(f"batch producers: {args.prefetch_workers} fork workers, "
              f"{producer_pool.slots} shared slots, {BATCH_RNG}", flush=True)

    gpus = tf.config.list_physical_devices("GPU")
    if args.require_gpu and not gpus:
        if producer_pool is not None:
            producer_pool.close()
        parser.error("GPU required but TensorFlow cannot see one; run via scripts/inductive.sh")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"TensorFlow devices: {gpus}", flush=True)

    versions = {name: importlib.metadata.version(name) for name in ("tensorflow", "keras", "gestaltdb", "tf_gnns", "numpy")}
    if args.profile_batches is None:
        (args.run_dir / "environment.json").write_text(json.dumps(dict(versions=versions, gpus=[str(g) for g in gpus]), indent=2))

    tf.keras.utils.set_random_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = InductiveAutoencoder(**spec["model"])
    shallow_parameter_ids = {id(v) for layer in (model.entity_table, model.base_relations) for v in layer.variables}
    print(f"GNN/decoder parameters: {sum(int(np.prod(v.shape)) for v in model.trainable_variables if id(v) not in shallow_parameter_ids):,}", flush=True)
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
        optimizer_steps = [int(optimizer.iterations.numpy())]
        if shallow_optimizer is not None:
            optimizer_steps.append(int(shallow_optimizer.iterations.numpy()))
        restored_steps = reconcile_completed_step(int(step.numpy()), *optimizer_steps)
        if restored_steps != int(step.numpy()):
            print(f"reconciled legacy interrupted step counter to completed optimizer step {restored_steps}", flush=True)
            step.assign(restored_steps)
        if producer_pool is None:
            rng.bit_generator.state = json.loads(rng_state.numpy().decode())
        else:
            print(f"batch RNG: {BATCH_RNG}; legacy sequential state is not reused", flush=True)
        if migration is not None:
            if migration_step is not None and int(step.numpy()) != migration_step:
                parser.error("migration step differs from completed optimizer step")
            reason = "explicit throughput migration; preserve optimizer weights and all scheduled entity anchors"
            if args.prefetch_workers:
                reason += (f"; batch RNG {BATCH_RNG} with {args.prefetch_workers} "
                           "forked shared-memory producers")
            migration.update(checkpoint=manager.latest_checkpoint, completed_step=int(step.numpy()),
                             reason=reason)
            (args.run_dir / f"performance_migration_{int(step.numpy())}.json").write_text(json.dumps(migration, indent=2))
            temporary = config_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(spec, indent=2))
            temporary.replace(config_path)
            print(f"recorded performance migration at complete step {int(step.numpy())}", flush=True)
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
    if args.profile_batches is None:
        curves_path.write_text("".join(json.dumps(row) + "\n" for row in curves))

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

    if args.profile_batches is not None:
        from .inductive_profile import benchmark_batches
        benchmark_batches(args, spec, context, training_known, relation_pools, update,
                          checkpoint, manager.latest_checkpoint, start)
        return

    def save_coverage(path):
        if schedule is not None:
            progress = schedule.progress(int(step.numpy()))
            progress["checkpoint"] = path
            temporary = args.run_dir / "coverage.tmp"
            temporary.write_text(json.dumps(progress, indent=2))
            temporary.replace(args.run_dir / "coverage.json")

    if migration is not None:
        save_coverage(manager.latest_checkpoint)

    def save():
        rng_state.assign(json.dumps(rng.bit_generator.state) if producer_pool is None
                         else json.dumps(dict(mode=BATCH_RNG)))
        path = manager.save(checkpoint_number=int(step.numpy()))
        save_coverage(path)
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
        save()  # baseline/evaluation/first compilation can also be interrupted
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
    if producer_pool is not None:
        producer_pool.begin(start)
    t0 = time.perf_counter()
    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    # Do not interrupt an optimizer call or save the RNG halfway through a
    # sampled batch. Both signals request a checkpoint at an update boundary.
    previous_handlers = {signum: signal.signal(signum, request_stop)
                         for signum in (signal.SIGINT, signal.SIGTERM)}

    try:
        for current in range(start + 1, args.steps + 1):
            if stop_requested:
                save()
                break
            produced = None
            values = None
            if producer_pool is None:
                arrays, batch_timings = prepare_batch(context, training_known, relation_pools, num_train_nodes,
                    rng=rng, batch_size=args.batch_size, negatives=args.negatives,
                    negative_policy=args.negative_policy, fanouts=args.fanouts,
                    schedule=schedule, completed_steps=current - 1)
                graph, positive_local, negative_local, positive, negative = arrays
                losses = update({k: tf.convert_to_tensor(v) for k, v in graph.items()},
                                positive_local, negative_local, positive, negative)
            else:
                produced = producer_pool.get(current - 1)
                try:
                    completed, graph, positive_local, negative_local, positive, negative = producer_pool.read(produced)
                    if completed != current - 1:
                        raise RuntimeError(f"producer returned step {completed}, expected {current - 1}")
                    losses = update({k: tf.convert_to_tensor(v) for k, v in graph.items()},
                                    positive_local, negative_local, positive, negative)
                    # The shared slot cannot be recycled until the device copy
                    # and optimizer update have completed.
                    values = tuple(float(loss) for loss in losses)
                finally:
                    producer_pool.release(produced)
                    produced = None
            step.assign(current)
            if current % 50 == 0 or current == start + 1:
                if values is None:
                    values = tuple(float(loss) for loss in losses)
                elapsed = time.perf_counter() - t0
                row = dict(step=current, loss=values[0], gnn_loss=values[1],
                           shallow_loss=values[2], elapsed_seconds=elapsed,
                            steps_per_second=(current - start) / max(elapsed, 1e-9))
                if schedule is not None:
                    row["coverage"] = schedule.progress(current)
                with train_log.open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                print(f"step={current} loss={values[0]:.4f} "
                      f"gnn={values[1]:.4f} shallow={values[2]:.4f} "
                      f"steps/s={row['steps_per_second']:.2f}", flush=True)
            if current % args.checkpoint_every == 0 or current == args.steps or stop_requested:
                save()  # checkpoint BEFORE potentially expensive evaluation/export
            if stop_requested:
                break
            if current % args.eval_every == 0 or current == args.steps:
                validate(current, losses)
    except KeyboardInterrupt:
        # Only completed updates count. SIGKILL/power loss uses periodic saves.
        save()
        raise
    finally:
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
        if producer_pool is not None:
            producer_pool.close()

    if stop_requested:
        print(f"stopped after checkpointing complete step {int(step.numpy())}; resume with --resume", flush=True)
        return

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
        # The standalone GNN must not require a multi-gigabyte ID table at
        # inference. Coverage runs default to latest, not an earlier best model.
        best_checkpoint = best_manager.latest_checkpoint if export_checkpoint == "best" else manager.latest_checkpoint
        if best_checkpoint:
            restored = tf.train.Checkpoint(model=model).restore(best_checkpoint)
            restored.assert_existing_objects_matched()
            restored.expect_partial()
        from .inductive_export import export_encoder
        export_encoder(model, args.run_dir / "gnn_saved_model")
        (args.run_dir / "gnn_saved_model" / "metadata.json").write_text(json.dumps(dict(
            source_checkpoint=best_checkpoint, feature_recipe=spec["features"],
            checkpoint_policy=export_checkpoint,
            coverage=schedule.progress(int(step.numpy())) if schedule is not None and export_checkpoint == "latest" else None,
            relation_ids=data["relation_ids"].tolist(), output_dim=args.dim,
            entity_ids_required=False), indent=2))
    print(f"completed step {int(step.numpy())}; contextual inference: python -m yago.inductive_eval "
          f"--run-dir {args.run_dir}", flush=True)


if __name__ == "__main__":
    main()
