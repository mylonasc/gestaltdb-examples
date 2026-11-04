"""Feed-vs-compute benchmark: is the sampler starving the GPU?

Measures four rates with the *same* config the training run uses:
  1. feeder      - YagoSampler.sample_batch() + tensor conversion (CPU only)
  2. delivered   - batches/s through tf.data with the training prefetch depth
  3. compute     - compiled train_step on ONE cached batch (pure GPU ceiling)
  4. end_to_end  - compiled train_step on fresh batches (what training logs)

Verdict logic:
  - feeder/delivered >> end_to_end ~= compute  -> GPU-bound, feeder keeps up
  - end_to_end << compute                    -> feeder is starving the GPU
  - end_to_end(no-prefetch) << end_to_end     -> prefetch is load-bearing

Usage:
  python -m yago.bench_feed --data-root data/yago45_facts --batches 30
  python -m yago.bench_feed --data-root data/yago45_facts --mode feeder --workers 4
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yago.config import PathsConfig, make_config
from yago.losses import contrastive_infonce_loss, weighted_infonce_loss
from yago.model import YagoKGEModel
from yago.sampler import ParallelYagoSampler, YagoSampler, output_signature, sampled_batch_to_tensor_dict
from yago.train import dataset_from_sampler


def _timed(n: int, fn, warmup: int = 2) -> list[float]:
    for _ in range(warmup):
        fn()
    dts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        dts.append(time.perf_counter() - t0)
    return dts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--batches", type=int, default=30)
    ap.add_argument("--mode", choices=["all", "feeder", "delivered", "compute", "end_to_end"],
                    default="all")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--negatives", type=int, default=None)
    ap.add_argument("--negative-sampler", choices=["engine", "hybrid"], default=None)
    args = ap.parse_args()

    cfg = make_config()
    if args.data_root:
        cfg.paths = PathsConfig(data_root=Path(args.data_root))
    if args.batch_size:
        cfg.sampler.batch_size = args.batch_size
    if args.negatives:
        cfg.sampler.negatives_per_positive = args.negatives
    if args.negative_sampler:
        cfg.sampler.negative_sampler = args.negative_sampler
    if args.workers is not None:
        cfg.sampler.num_workers = args.workers

    use_parallel = cfg.sampler.num_workers > 0
    sampler = (ParallelYagoSampler(cfg, num_workers=cfg.sampler.num_workers)
               if use_parallel else YagoSampler(cfg))
    n, k = args.batches, cfg.sampler.negatives_per_positive
    print(f"sampler={'parallel x%d' % cfg.sampler.num_workers if use_parallel else 'single'} "
          f"policy={cfg.sampler.negative_sampler} batch={cfg.sampler.batch_size} neg={k}", flush=True)

    results: dict[str, float] = {}
    if args.mode in ("all", "feeder"):
        batch_holder: dict = {}
        def produce():
            b = sampler.sample_batch()
            batch_holder["t"] = sampled_batch_to_tensor_dict(b)
        dts = _timed(n, produce)
        results["feeder"] = len(dts) / sum(dts)
        print(f"feeder:        {results['feeder']:6.1f} batches/s "
              f"({1000 * np.median(dts):6.1f} ms/batch median)", flush=True)

    if args.mode in ("all", "delivered", "end_to_end", "compute"):
        ds = dataset_from_sampler(sampler, cfg.training.prefetch)
        it = iter(ds)
        if args.mode in ("all", "delivered"):
            dts = _timed(n, lambda: next(it))
            results["delivered"] = len(dts) / sum(dts)
            print(f"delivered(prefetch={cfg.training.prefetch}): "
                  f"{results['delivered']:6.1f} batches/s "
                  f"({1000 * np.median(dts):6.1f} ms/batch median)", flush=True)

    if args.mode in ("all", "compute", "end_to_end"):
        gpus = tf.config.list_physical_devices("GPU")
        print("GPUs:", gpus, flush=True)
        model = YagoKGEModel(cfg.model, num_nodes=sampler.num_nodes,
                             num_node_types=max(2, sampler.num_node_types),
                             num_relations=sampler.num_relations)
        opt = tf.keras.optimizers.Adam(learning_rate=cfg.training.learning_rate)
        use_adv = cfg.model.adv_temperature > 0
        aux_w = cfg.model.aux_shallow_weight

        def _loss_fn(p, ng):
            if use_adv:
                return weighted_infonce_loss(p, ng, cfg.model.temperature, cfg.model.adv_temperature)
            return contrastive_infonce_loss(p, ng, cfg.model.temperature)

        sig = output_signature(k, cfg.model.semantic_negatives)

        @tf.function(input_signature=sig, reduce_retracing=True)
        def train_step(graph, labels):
            with tf.GradientTape() as tape:
                if cfg.model.use_gnn:
                    pos, neg = model(graph, labels, training=True)
                else:
                    gids = tf.cast(graph["nodes"][:, 0], tf.int64)
                    base = model.entity_emb.weights[0]
                    pos = YagoKGEModel._score_impl(
                        base, tf.gather(gids, labels["positive_src"]), labels["positive_rel"],
                        tf.gather(gids, labels["positive_dst"]),
                        cfg.model.score_function, model.rel_emb)
                    neg = tf.reshape(YagoKGEModel._score_impl(
                        base, tf.reshape(labels["negative_src_global"], [-1]),
                        tf.reshape(labels["negative_rel"], [-1]),
                        tf.reshape(labels["negative_dst_global"], [-1]),
                        cfg.model.score_function, model.rel_emb), tf.shape(labels["negative_src"]))
                loss = _loss_fn(pos, neg)
                if aux_w > 0 and cfg.model.use_gnn:
                    gids = tf.cast(graph["nodes"][:, 0], tf.int64)
                    base = model.entity_emb.weights[0]
                    apos = YagoKGEModel._score_impl(
                        base, tf.gather(gids, labels["positive_src"]), labels["positive_rel"],
                        tf.gather(gids, labels["positive_dst"]),
                        cfg.model.score_function, model.rel_emb)
                    aneg = tf.reshape(YagoKGEModel._score_impl(
                        base, tf.reshape(labels["negative_src_global"], [-1]),
                        tf.reshape(labels["negative_rel"], [-1]),
                        tf.reshape(labels["negative_dst_global"], [-1]),
                        cfg.model.score_function, model.rel_emb), tf.shape(labels["negative_src"]))
                    loss = loss + aux_w * _loss_fn(apos, aneg)
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients([(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None])
            return loss

        for g0, l0 in ds.take(1):  # trace/compile (untimed warmup, also runs one opt step)
            _ = train_step(g0, l0)
        print("compiled train_step", flush=True)
        if args.mode in ("all", "compute"):
            g_fixed, l_fixed = next(iter(ds))
            dts = _timed(n, lambda: train_step(g_fixed, l_fixed))
            results["compute"] = len(dts) / sum(dts)
            print(f"compute-only:  {results['compute']:6.1f} steps/s "
                  f"({1000 * np.median(dts):6.1f} ms/step median)", flush=True)
        if args.mode in ("all", "end_to_end"):
            it2 = iter(ds)
            dts = _timed(n, lambda: train_step(*next(it2)))
            results["end_to_end"] = len(dts) / sum(dts)
            print(f"end_to_end:    {results['end_to_end']:6.1f} steps/s "
                  f"({1000 * np.median(dts):6.1f} ms/step median)", flush=True)

    if set(results) >= {"feeder", "compute", "end_to_end"}:
        headroom = results["feeder"] / results["end_to_end"]
        gpu_bound = results["end_to_end"] > 0.9 * results["compute"]
        print(f"\nheadroom(feeder/end_to_end) = {headroom:.2f}x; "
              f"end_to_end/compute = {results['end_to_end'] / results['compute']:.2f}", flush=True)
        if headroom >= 2.0 and gpu_bound:
            print("VERDICT: GPU-bound. The feeder produces batches "
                  f"{headroom:.1f}x faster than training consumes them; "
                  "the GPU is not starved.", flush=True)
        elif not gpu_bound:
            print("VERDICT: FEEDER-BOUND. End-to-end is well below the pure-GPU "
                  "ceiling: raise prefetch/workers, shrink fanouts, or use hybrid negatives.", flush=True)
        else:
            print("VERDICT: marginal. Feeder barely keeps up; watch nvidia-smi "
                  "utilization during long runs.", flush=True)
    if hasattr(sampler, "close"):
        sampler.close()


if __name__ == "__main__":
    main()
