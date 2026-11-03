"""Train YAGO KGE: compiled tf_gnns GNN + gestaltdb sampler feed."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yago.config import make_config
from yago.losses import contrastive_infonce_loss
from yago.model import YagoKGEModel
from yago.sampler import YagoSampler, output_signature, sampled_batch_to_tensor_dict


def validate_gpu(allow_cpu: bool) -> None:
    gpus = tf.config.list_physical_devices("GPU")
    print("GPUs:", gpus)
    if not gpus:
        if allow_cpu:
            print("WARNING: no GPU visible, continuing on CPU")
            return
        raise RuntimeError("TensorFlow cannot see a GPU. Run via scripts/train.sh (sets LD_LIBRARY_PATH).")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass


def dataset_from_sampler(sampler: YagoSampler, prefetch: int):
    def gen():
        while True:
            yield sampled_batch_to_tensor_dict(sampler.sample_batch())

    return tf.data.Dataset.from_generator(
        gen,
        output_signature=output_signature(sampler.cfg.sampler.negatives_per_positive),
    ).prefetch(prefetch)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--xla", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--hop1-fanout", type=int, default=None)
    parser.add_argument("--hop2-fanout", type=int, default=None)
    parser.add_argument("--negatives", type=int, default=None)
    args = parser.parse_args()
    cfg = make_config(smoke_test=args.smoke_test)
    if args.hop1_fanout is not None:
        cfg.sampler.hop1_fanout = args.hop1_fanout
    if args.hop2_fanout is not None:
        cfg.sampler.hop2_fanout = args.hop2_fanout
    if args.negatives is not None:
        cfg.sampler.negatives_per_positive = args.negatives
    if args.data_root:
        from yago.config import PathsConfig
        cfg.paths = PathsConfig(data_root=Path(args.data_root))
    if args.steps is not None:
        cfg.training.steps = args.steps
    if args.batch_size is not None:
        cfg.sampler.batch_size = args.batch_size
    if args.xla:
        cfg.training.xla = True
    if args.learning_rate is not None:
        cfg.training.learning_rate = args.learning_rate

    validate_gpu(args.allow_cpu)
    sampler = YagoSampler(cfg)
    print(f"graph: nodes={sampler.num_nodes:,} relations={sampler.num_relations} "
          f"node_types={sampler.num_node_types} edges={sampler.snapshot.num_edges:,}")
    ds = dataset_from_sampler(sampler, cfg.training.prefetch)
    model = YagoKGEModel(
        cfg.model,
        num_nodes=sampler.num_nodes,
        num_node_types=max(2, sampler.num_node_types),
        num_relations=sampler.num_relations,
    )
    opt = tf.keras.optimizers.Adam(learning_rate=cfg.training.learning_rate)
    sig = output_signature(cfg.sampler.negatives_per_positive)

    @tf.function(input_signature=sig, reduce_retracing=True, jit_compile=cfg.training.xla)
    def train_step(graph, labels):
        with tf.GradientTape() as tape:
            pos, neg = model(graph, labels, training=True)
            loss = contrastive_infonce_loss(pos, neg, cfg.model.temperature)
        grads = tape.gradient(loss, model.trainable_variables)
        grads_and_vars = [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]
        opt.apply_gradients(grads_and_vars)
        return loss, tf.shape(graph["nodes"])[0], tf.shape(graph["edges"])[0]

    # Warmup compile on one batch before timing.
    for g0, l0 in ds.take(1):
        _ = train_step(g0, l0)
    print(f"compiled train_step (xla={cfg.training.xla})")

    ckpt_dir = cfg.paths.checkpoints_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = tf.train.Checkpoint(model=model, opt=opt)
    mgr = tf.train.CheckpointManager(ckpt, str(ckpt_dir), max_to_keep=2)

    t0 = time.perf_counter()
    for step, (graph, labels) in enumerate(ds.take(cfg.training.steps), start=1):
        loss, n_nodes, n_edges = train_step(graph, labels)
        if step == 1 or step % cfg.training.log_every == 0:
            dt = max(time.perf_counter() - t0, 1e-6)
            print(f"step={step} loss={float(loss):.4f} "
                  f"sg_nodes={int(n_nodes)} sg_edges={int(n_edges)} steps_per_s={step/dt:.2f}", flush=True)
        if step % cfg.training.eval_every == 0:
            mgr.save(checkpoint_number=step)
    mgr.save(checkpoint_number=cfg.training.steps)

    # Export 200-d canonical tables.
    from yago.export import export_embeddings
    export_embeddings(cfg, model)
    print("done.")


if __name__ == "__main__":
    main()
