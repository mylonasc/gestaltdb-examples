"""Train YAGO KGE: compiled tf_gnns GNN + gestaltdb sampler feed.

Logs train loss and held-out validation MRR/Hits to a JSONL curves file and
renders train/validation curves at the end. Validation triples come from the
link-prediction split saved at prepare time (never in the sampler snapshot).
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yago.config import make_config
from yago.losses import cosine_infonce_loss, contrastive_infonce_loss, weighted_infonce_loss
from yago.model import YagoKGEModel
from yago.sampler import (
    ParallelYagoSampler,
    YagoSampler,
    output_signature,
    sampled_batch_to_tensor_dict,
)


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


def dataset_from_sampler(sampler, prefetch: int):
    def gen():
        while True:
            yield sampled_batch_to_tensor_dict(sampler.sample_batch())

    return tf.data.Dataset.from_generator(
        gen,
        output_signature=output_signature(
            sampler.cfg.sampler.negatives_per_positive,
            sampler.cfg.model.semantic_negatives),
    ).prefetch(prefetch)


def load_valid_int(cfg, max_triples: int):
    """Map held-out valid triples to compact ints; drop unknowns."""
    import polars as pl
    node_map = pl.read_parquet(cfg.paths.node_map_path).sort("node_int")
    rel_map = pl.read_parquet(cfg.paths.relation_map_path).sort("rel_int")
    id2int = dict(zip(node_map["node_id"].to_list(), node_map["node_int"].to_list()))
    rel2int = dict(zip(rel_map["edge_type"].to_list(), rel_map["rel_int"].to_list()))
    df = pl.read_parquet(cfg.paths.triples_valid_path)
    if len(df) > max_triples:
        df = df.sample(n=max_triples, seed=cfg.subset.seed)
    rows, dropped = [], 0
    for r in df.iter_rows(named=True):
        if r["source"] in id2int and r["target"] in id2int and r["edge_type"] in rel2int:
            rows.append((id2int[r["source"]], rel2int[r["edge_type"]], id2int[r["target"]]))
        else:
            dropped += 1
    arr = np.array(rows, dtype=np.int64)
    print(f"valid: kept={len(arr):,} dropped_unknown={dropped}")
    return arr


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
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--fast-negatives", action="store_true",
                        help="Deprecated alias for --negative-sampler hybrid")
    parser.add_argument("--negative-sampler", choices=["engine", "hybrid"], default=None)
    parser.add_argument("--score", type=str, default=None, choices=["distmult", "complex"])
    parser.add_argument("--no-gnn", action="store_true")
    parser.add_argument("--adv-temp", type=float, default=None)
    parser.add_argument("--aux-weight", type=float, default=None)
    parser.add_argument("--semantic-weight", type=float, default=None)
    parser.add_argument("--semantic-temperature", type=float, default=None)
    parser.add_argument("--semantic-batch-size", type=int, default=None)
    parser.add_argument("--semantic-negatives", type=int, default=None)
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Write curves/checkpoints/export under this directory")
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--no-checkpoints", action="store_true")
    parser.add_argument("--init-embeddings", type=str, default=None,
                        help="Initialize base entity/relation tables from an exported NPZ")
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
    if args.workers is not None:
        cfg.sampler.num_workers = args.workers
    if args.seed is not None:
        cfg.sampler.seed = args.seed
    if args.fast_negatives:
        cfg.sampler.negative_sampler = "hybrid"
    if args.negative_sampler is not None:
        cfg.sampler.negative_sampler = args.negative_sampler
    if args.score is not None:
        cfg.model.score_function = args.score
    if args.no_gnn:
        cfg.model.use_gnn = False
    if args.adv_temp is not None:
        cfg.model.adv_temperature = args.adv_temp
    if args.aux_weight is not None:
        cfg.model.aux_shallow_weight = args.aux_weight
    if args.semantic_weight is not None:
        cfg.model.semantic_weight = args.semantic_weight
    if args.semantic_temperature is not None:
        cfg.model.semantic_temperature = args.semantic_temperature
    if args.semantic_batch_size is not None:
        cfg.model.semantic_batch_size = args.semantic_batch_size
    if args.semantic_negatives is not None:
        cfg.model.semantic_negatives = args.semantic_negatives
    if args.eval_every is not None:
        cfg.training.eval_every = args.eval_every

    validate_gpu(args.allow_cpu)
    tf.keras.utils.set_random_seed(cfg.sampler.seed)
    print(f"cfg: steps={cfg.training.steps} batch={cfg.sampler.batch_size} "
          f"neg={cfg.sampler.negatives_per_positive} negative_sampler={cfg.sampler.negative_sampler} "
          f"fanouts=[{cfg.sampler.hop1_fanout},{cfg.sampler.hop2_fanout}] "
          f"lr={cfg.training.learning_rate} eval_every={cfg.training.eval_every} "
          f"workers={cfg.sampler.num_workers} xla={cfg.training.xla}", flush=True)
    if cfg.sampler.num_workers > 0:
        sampler = ParallelYagoSampler(cfg, num_workers=cfg.sampler.num_workers)
        print(f"parallel feeder: {cfg.sampler.num_workers} workers")
    else:
        sampler = YagoSampler(cfg)
    n_edges = sampler.snapshot.num_edges if hasattr(sampler, "snapshot") else -1
    print(f"graph: nodes={sampler.num_nodes:,} relations={sampler.num_relations} "
          f"node_types={sampler.num_node_types} edges={n_edges:,}")
    ds = dataset_from_sampler(sampler, cfg.training.prefetch)
    model = YagoKGEModel(
        cfg.model,
        num_nodes=sampler.num_nodes,
        num_node_types=max(2, sampler.num_node_types),
        num_relations=sampler.num_relations,
    )
    if args.init_embeddings:
        init_path = Path(args.init_embeddings)
        init_data = np.load(init_path)
        meta_path = init_path.parent / "artifact_metadata.json"
        init_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        entity_key = init_meta.get("entity_key", "entity")
        relation_key = init_meta.get("relation_key", "relation")
        entity_init, relation_init = init_data[entity_key], init_data[relation_key]
        if tuple(entity_init.shape) != tuple(model.entity_emb.weights[0].shape):
            raise ValueError(f"entity init shape {entity_init.shape} != {model.entity_emb.weights[0].shape}")
        if tuple(relation_init.shape) != tuple(model.rel_emb.weights[0].shape):
            raise ValueError(f"relation init shape {relation_init.shape} != {model.rel_emb.weights[0].shape}")
        model.entity_emb.weights[0].assign(entity_init)
        model.rel_emb.weights[0].assign(relation_init)
        print(f"initialized base tables from {init_path}")
    opt = tf.keras.optimizers.Adam(learning_rate=cfg.training.learning_rate)
    sig = output_signature(cfg.sampler.negatives_per_positive, cfg.model.semantic_negatives)

    use_adv = cfg.model.adv_temperature is not None and cfg.model.adv_temperature > 0
    aux_w = cfg.model.aux_shallow_weight

    def _loss_fn(p, n):
        if use_adv:
            return weighted_infonce_loss(p, n, cfg.model.temperature, cfg.model.adv_temperature)
        return contrastive_infonce_loss(p, n, cfg.model.temperature)

    @tf.function(input_signature=sig, reduce_retracing=True, jit_compile=cfg.training.xla)
    def train_step(graph, labels):
        with tf.GradientTape() as tape:
            if cfg.model.use_gnn:
                pos, neg = model(graph, labels, training=True)
            else:
                # True shallow ablation: score exported tables with global IDs.
                gids = tf.cast(graph["nodes"][:, 0], tf.int64)
                ps = tf.gather(gids, labels["positive_src"])
                pd = tf.gather(gids, labels["positive_dst"])
                pr = labels["positive_rel"]
                ns = labels["negative_src_global"]
                nd = labels["negative_dst_global"]
                nr = labels["negative_rel"]
                pos = YagoKGEModel._score_impl(model.entity_emb.weights[0], ps, pr, pd,
                                               cfg.model.score_function, model.rel_emb)
                neg_flat = YagoKGEModel._score_impl(
                    model.entity_emb.weights[0], tf.reshape(ns, [-1]),
                    tf.reshape(nr, [-1]), tf.reshape(nd, [-1]),
                    cfg.model.score_function, model.rel_emb)
                neg = tf.reshape(neg_flat, tf.shape(ns))
            loss = _loss_fn(pos, neg)
            if aux_w > 0 and cfg.model.use_gnn:
                # Shallow auxiliary loss on the EXPORTED base tables (global ids),
                # so link-prediction structure lands in the artifact, not just GNN states.
                gids = tf.cast(graph["nodes"][:, 0], tf.int64)
                ps = tf.gather(gids, labels["positive_src"])
                pd = tf.gather(gids, labels["positive_dst"])
                pr = labels["positive_rel"]
                # These are actual global entity IDs. In hybrid mode they need
                # not occur in the sampled message-passing subgraph.
                ns = labels["negative_src_global"]
                nd = labels["negative_dst_global"]
                nr = labels["negative_rel"]
                base_nodes = model.entity_emb.weights[0]
                apos = YagoKGEModel._score_impl(base_nodes, ps, pr, pd, cfg.model.score_function, model.rel_emb)
                aneg_flat = YagoKGEModel._score_impl(
                    base_nodes,
                    tf.reshape(ns, [-1]), tf.reshape(nr, [-1]), tf.reshape(nd, [-1]),
                    cfg.model.score_function, model.rel_emb)
                aneg = tf.reshape(aneg_flat, tf.shape(ns))
                loss = loss + aux_w * _loss_fn(apos, aneg)
            if cfg.model.semantic_weight > 0:
                table = model.entity_emb.weights[0]
                sem_a = tf.gather(table, labels["semantic_anchor_global"])
                sem_p = tf.gather(table, labels["semantic_positive_global"])
                sem_n = tf.gather(table, labels["semantic_negative_global"])
                semantic_loss = cosine_infonce_loss(
                    sem_a, sem_p, sem_n, cfg.model.semantic_temperature)
                loss = loss + cfg.model.semantic_weight * semantic_loss
        grads = tape.gradient(loss, model.trainable_variables)
        grads_and_vars = [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]
        opt.apply_gradients(grads_and_vars)
        return loss, tf.shape(graph["nodes"])[0], tf.shape(graph["edges"])[0]

    # Validation scorer over the exported base tables (ComplEx or DistMult, no GNN):
    # compiled gather+matmul, negatives drawn fresh each eval (fixed seed).
    is_complex = cfg.model.score_function == "complex"

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None,), dtype=tf.int64),
        tf.TensorSpec(shape=(None,), dtype=tf.int64),
        tf.TensorSpec(shape=(None,), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ])
    def valid_scores(s, r, o_pos, o_neg):
        E = model.entity_emb.weights[0]
        R = model.rel_emb.weights[0]
        if is_complex:
            d = int(E.shape[-1]) // 2
            assert int(E.shape[-1]) % 2 == 0
            def cx(x):
                return tf.split(x, 2, axis=-1)
            s_re, s_im = cx(tf.gather(E, s))
            o_re, o_im = cx(tf.gather(E, o_pos))
            r_re, r_im = cx(tf.gather(R, r))
            pos = tf.reduce_sum(
                r_re * (s_re * o_re + s_im * o_im)
                + r_im * (s_re * o_im - s_im * o_re), axis=-1)
            n_re, n_im = cx(tf.gather(E, o_neg))
            s_re_b = tf.reshape(s_re, (-1, 1, d))
            s_im_b = tf.reshape(s_im, (-1, 1, d))
            r_re_b = tf.reshape(r_re, (-1, 1, d))
            r_im_b = tf.reshape(r_im, (-1, 1, d))
            neg = tf.reduce_sum(
                r_re_b * (s_re_b * n_re + s_im_b * n_im)
                + r_im_b * (s_re_b * n_im - s_im_b * n_re), axis=-1)
        else:
            d_full = int(E.shape[-1])
            pos = tf.reduce_sum(tf.gather(E, s) * tf.gather(R, r) * tf.gather(E, o_pos), axis=-1)
            neg = tf.reduce_sum(
                tf.gather(E, tf.reshape(s, (-1, 1))) * tf.reshape(tf.gather(R, r), (-1, 1, d_full))
                * tf.gather(E, o_neg), axis=-1)
        return pos, neg

    # Trace without consuming a batch or applying an unreported optimizer step.
    _ = train_step.get_concrete_function()
    print(f"compiled train_step (xla={cfg.training.xla})")

    valid_arr = load_valid_int(cfg, cfg.training.eval_triples)
    vrng = np.random.default_rng(12345)

    semantic_index = getattr(sampler, "semantic_index", None)
    if semantic_index is None and hasattr(sampler, "workers"):
        semantic_index = sampler.workers[0].semantic_index

    def run_semantic_validation():
        if semantic_index is None:
            return None, None
        srng = np.random.default_rng(2026)
        a, p, n = semantic_index.sample(srng, 2_000, cfg.model.semantic_negatives)
        ids = np.unique(np.concatenate([a, p, n.reshape(-1)]))
        vectors = tf.gather(model.entity_emb.weights[0], ids).numpy()
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
        pos_map = np.searchsorted(ids, p)
        anchor_map = np.searchsorted(ids, a)
        neg_map = np.searchsorted(ids, n)
        pos_score = np.sum(vectors[anchor_map] * vectors[pos_map], axis=1)
        neg_score = np.einsum("bd,bkd->bk", vectors[anchor_map], vectors[neg_map])
        top1 = float(np.mean(pos_score > np.max(neg_score, axis=1)))
        margin = float(np.mean(pos_score - np.max(neg_score, axis=1)))
        return top1, margin

    def run_validation():
        K = cfg.training.eval_negatives
        s = valid_arr[:, 0]
        r = valid_arr[:, 1]
        o = valid_arr[:, 2]
        neg_o = vrng.integers(0, sampler.num_nodes, size=(len(s), K), dtype=np.int64)
        pos, neg = valid_scores(
            tf.convert_to_tensor(s), tf.convert_to_tensor(r),
            tf.convert_to_tensor(o), tf.convert_to_tensor(neg_o))
        pos, neg = pos.numpy(), neg.numpy()
        ranks = 1 + np.sum(neg >= pos[:, None], axis=1)
        return float(np.mean(1.0 / ranks)), float(np.mean(ranks <= 10)), float(np.mean(ranks <= 1))

    # Baseline before any update.
    run_dir = Path(args.run_dir) if args.run_dir else cfg.paths.artifacts_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    curves_path = run_dir / "curves.jsonl"
    curves_path.write_text("")
    curves = []
    v_mrr, v_h10, v_h1 = run_validation()
    sem_top1, sem_margin = run_semantic_validation()
    curves.append({"step": 0, "train_loss": float("nan"),
                   "valid_mrr": v_mrr, "valid_hits10": v_h10, "valid_hits1": v_h1,
                   "semantic_top1": sem_top1, "semantic_margin": sem_margin})
    print(f"step=0 valid_mrr={v_mrr:.4f} valid_hits10={v_h10:.4f} valid_hits1={v_h1:.4f}", flush=True)

    ckpt_dir = run_dir / "checkpoints"
    if not args.no_checkpoints:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = tf.train.Checkpoint(model=model, opt=opt)
    mgr = None if args.no_checkpoints else tf.train.CheckpointManager(ckpt, str(ckpt_dir), max_to_keep=2)

    t0 = time.perf_counter()
    for step, (graph, labels) in enumerate(ds.take(cfg.training.steps), start=1):
        loss, n_nodes, n_edges = train_step(graph, labels)
        if step == 1 or step % cfg.training.log_every == 0:
            dt = max(time.perf_counter() - t0, 1e-6)
            print(f"step={step} loss={float(loss):.4f} "
                  f"sg_nodes={int(n_nodes)} sg_edges={int(n_edges)} steps_per_s={step/dt:.2f}", flush=True)
        if step % cfg.training.eval_every == 0 or step == cfg.training.steps:
            v_mrr, v_h10, v_h1 = run_validation()
            sem_top1, sem_margin = run_semantic_validation()
            curves.append({"step": step, "train_loss": float(loss),
                           "valid_mrr": v_mrr, "valid_hits10": v_h10, "valid_hits1": v_h1,
                           "semantic_top1": sem_top1, "semantic_margin": sem_margin})
            with open(curves_path, "a") as fh:
                fh.write(json.dumps(curves[-1]) + "\n")
            print(f"step={step} valid_mrr={v_mrr:.4f} valid_hits10={v_h10:.4f} "
                  f"valid_hits1={v_h1:.4f}" +
                  (f" semantic_top1={sem_top1:.4f} semantic_margin={sem_margin:.4f}"
                   if sem_top1 is not None else ""), flush=True)
            if mgr is not None:
                mgr.save(checkpoint_number=step)
    if mgr is not None:
        mgr.save(checkpoint_number=cfg.training.steps)

    # Render train/validation curves.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    steps = [c["step"] for c in curves[1:]]
    losses = [c["train_loss"] for c in curves[1:]]
    base = [c for c in curves]
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot([c["step"] for c in base], [c["train_loss"] for c in base], "o-", ms=3, label="train loss")
    ax1.set_xlabel("step"); ax1.set_ylabel("InfoNCE loss"); ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.plot([c["step"] for c in base], [c["valid_mrr"] for c in base], "s-", ms=3, color="C1", label="valid MRR")
    ax2.plot([c["step"] for c in base], [c["valid_hits10"] for c in base], "^-", ms=3, color="C2", label="valid Hits@10")
    ax2.set_ylabel("MRR / Hits@10"); ax2.legend(loc="upper right")
    plt.title("Train loss vs validation link-prediction (held-out triples)")
    plt.tight_layout()
    fig_path = run_dir / "curves.png"
    plt.savefig(fig_path, dpi=120)
    print(f"wrote {curves_path} ({len(curves)} points) + {fig_path}")

    best = max(curves, key=lambda c: c["valid_mrr"])
    summary = {
        "best_valid_mrr": best["valid_mrr"],
        "best_step": best["step"],
        "final_valid_mrr": curves[-1]["valid_mrr"],
        "final_valid_hits10": curves[-1]["valid_hits10"],
        "final_semantic_top1": curves[-1]["semantic_top1"],
        "final_semantic_margin": curves[-1]["semantic_margin"],
        "steps": cfg.training.steps,
        "elapsed_seconds": time.perf_counter() - t0,
        "model": asdict(cfg.model),
        "sampler": asdict(cfg.sampler),
        "training": asdict(cfg.training),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if not args.no_export:
        from yago.export import export_embeddings
        export_embeddings(cfg, model, output_dir=run_dir, metadata=summary)
    if hasattr(sampler, "close"):
        sampler.close()
    print("done.")


if __name__ == "__main__":
    main()
