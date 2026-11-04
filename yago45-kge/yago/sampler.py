"""Sampler + tf.data batch conversion for YAGO."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
import tensorflow as tf

from .config import AppConfig


@dataclass(slots=True)
class SampledBatch:
    node_ids: np.ndarray
    node_global_ids: np.ndarray
    node_type_ids: np.ndarray
    senders: np.ndarray
    receivers: np.ndarray
    edge_rel_ids: np.ndarray
    positive_src: np.ndarray
    positive_dst: np.ndarray
    positive_rel: np.ndarray
    negative_src: np.ndarray
    negative_dst: np.ndarray
    negative_rel: np.ndarray
    # Global IDs used only by the shallow exported-table objective. Contextual
    # GNN negatives above remain local IDs into this sampled subgraph.
    negative_src_global: np.ndarray
    negative_dst_global: np.ndarray
    semantic_anchor_global: np.ndarray
    semantic_positive_global: np.ndarray
    semantic_negative_global: np.ndarray


class SemanticPairIndex:
    """Balanced attribute groups for cosine-supervised entity sampling."""

    def __init__(self, cfg: AppConfig):
        rel_map = pl.read_parquet(cfg.paths.relation_map_path)
        rel_to_int = dict(zip(rel_map["edge_type"].to_list(), rel_map["rel_int"].to_list()))
        arrays = np.load(cfg.paths.edge_arrays_path, mmap_mode="r")
        src_all, dst_all, rel_all = arrays["src"], arrays["dst"], arrays["rel"]
        self.relations = []
        for name in cfg.model.semantic_relations:
            if name not in rel_to_int:
                continue
            relation = int(rel_to_int[name])
            mask = rel_all == relation
            src, dst = src_all[mask], dst_all[mask]
            order = np.argsort(dst, kind="stable")
            src, dst = src[order], dst[order]
            _, starts, counts = np.unique(dst, return_index=True, return_counts=True)
            valid = counts >= 2
            starts, counts = starts[valid], counts[valid]
            if len(starts) >= 2:
                # sqrt weighting prevents huge classes from dominating without
                # making extremely rare classes overwhelmingly likely.
                weights = np.sqrt(counts.astype(np.float64))
                weights /= weights.sum()
                self.relations.append((name, src, starts, counts, weights))
        if not self.relations:
            raise ValueError(f"none of semantic_relations found: {cfg.model.semantic_relations}")

    def sample(self, rng: np.random.Generator, batch_size: int, negatives: int):
        anchor = np.empty(batch_size, dtype=np.int64)
        positive = np.empty(batch_size, dtype=np.int64)
        negative = np.empty((batch_size, negatives), dtype=np.int64)
        relation_choice = rng.integers(0, len(self.relations), size=batch_size)
        for rel_idx in range(len(self.relations)):
            rows = np.flatnonzero(relation_choice == rel_idx)
            if not len(rows):
                continue
            _, src, starts, counts, weights = self.relations[rel_idx]
            groups = rng.choice(len(starts), size=len(rows), p=weights)
            group_counts = counts[groups]
            first = (rng.random(len(rows)) * group_counts).astype(np.int64)
            second = (rng.random(len(rows)) * (group_counts - 1)).astype(np.int64)
            second += second >= first
            anchor[rows] = src[starts[groups] + first]
            positive[rows] = src[starts[groups] + second]
            neg_groups = rng.integers(0, len(starts) - 1, size=(len(rows), negatives))
            neg_groups += neg_groups >= groups[:, None]
            offsets = (rng.random((len(rows), negatives)) * counts[neg_groups]).astype(np.int64)
            negative[rows] = src[starts[neg_groups] + offsets]
        return anchor, positive, negative


def mask_supervised_message_edges(
    senders: np.ndarray,
    receivers: np.ndarray,
    edge_rel_ids: np.ndarray,
    positive_src: np.ndarray,
    positive_dst: np.ndarray,
    positive_rel: np.ndarray,
    *,
    mask_reverse: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove labelled triples from the GNN context, including reverse copies."""
    supervised = set(zip(positive_src.tolist(), positive_rel.tolist(), positive_dst.tolist()))
    keep = np.fromiter(
        (
            (int(s), int(r), int(o)) not in supervised
            and (not mask_reverse or (int(o), int(r), int(s)) not in supervised)
            for s, r, o in zip(senders, edge_rel_ids, receivers)
        ),
        dtype=bool,
        count=len(senders),
    )
    return senders[keep], receivers[keep], edge_rel_ids[keep]


class YagoSampler:
    def __init__(self, cfg: AppConfig, *, mode: str = "ram", snapshot=None, seed: int | None = None):
        from .prepare import build_snapshot
        from gestaltdb.sampling import HardNegativeConfig, SamplerEngine
        self.cfg = cfg
        self._HardNegativeConfig = HardNegativeConfig
        self.snapshot = snapshot if snapshot is not None else build_snapshot(cfg, mode=mode)
        self.engine = SamplerEngine(self.snapshot, seed=cfg.sampler.seed if seed is None else seed)
        self.rng = np.random.default_rng(cfg.sampler.seed if seed is None else seed)
        self.semantic_index = SemanticPairIndex(cfg) if cfg.model.semantic_weight > 0 else None

    def _semantic_batch(self):
        if self.semantic_index is None:
            k = self.cfg.model.semantic_negatives
            return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
                    np.empty((0, k), dtype=np.int64))
        return self.semantic_index.sample(
            self.rng, self.cfg.model.semantic_batch_size, self.cfg.model.semantic_negatives)

    @property
    def num_nodes(self) -> int:
        return self.snapshot.num_nodes

    @property
    def num_relations(self) -> int:
        return self.snapshot.num_relations

    @property
    def num_node_types(self) -> int:
        try:
            return max(1, int(self.snapshot.metadata.get("node_type_count", 1)))
        except Exception:
            return 1

    def sample_batch(self) -> SampledBatch:
        n_edges = self.snapshot.num_edges
        edge_indices = self.rng.integers(n_edges, size=self.cfg.sampler.batch_size, dtype=np.int64)
        if self.cfg.sampler.negative_sampler == "hybrid":
            return self._sample_batch_fast(edge_indices)
        batch = self.engine.sample_subgraph(
            edge_indices,
            fanouts=[self.cfg.sampler.hop1_fanout, self.cfg.sampler.hop2_fanout],
            direction=self.cfg.sampler.bfs_direction,
            negative_config=self._HardNegativeConfig(
                negatives_per_positive=self.cfg.sampler.negatives_per_positive,
                source=self.cfg.sampler.engine_negative_source,
                reject_known_positives=True,
                same_endpoint_type=self.cfg.sampler.same_endpoint_type_negatives,
                relation_endpoint_types=True,
                candidate_fanout=self.cfg.sampler.negative_candidate_fanout,
                head_probability=self.cfg.sampler.hard_negative_head_probability,
                max_retries=self.cfg.sampler.max_negative_retries,
            ),
        )
        senders, receivers, edge_rels = self._masked_context(batch)
        gids = batch.node_ids_global
        neg_src = batch.negatives[:, :, 0].astype(np.int64)
        neg_dst = batch.negatives[:, :, 2].astype(np.int64)
        sem_a, sem_p, sem_n = self._semantic_batch()
        return SampledBatch(
            node_ids=np.arange(batch.n_nodes, dtype=np.int64),
            node_global_ids=batch.node_ids_global,
            node_type_ids=batch.node_type_ids,
            senders=senders,
            receivers=receivers,
            edge_rel_ids=edge_rels,
            positive_src=batch.positives[:, 0].astype(np.int64),
            positive_dst=batch.positives[:, 2].astype(np.int64),
            positive_rel=batch.positives[:, 1].astype(np.int64),
            negative_src=neg_src,
            negative_dst=neg_dst,
            negative_rel=batch.negatives[:, :, 1].astype(np.int64),
            negative_src_global=gids[neg_src],
            negative_dst_global=gids[neg_dst],
            semantic_anchor_global=sem_a,
            semantic_positive_global=sem_p,
            semantic_negative_global=sem_n,
        )

    def _masked_context(self, batch):
        if not self.cfg.sampler.mask_supervised_edges:
            return batch.senders, batch.receivers, batch.edge_relation_ids
        pos = batch.positives.astype(np.int64)
        return mask_supervised_message_edges(
            batch.senders,
            batch.receivers,
            batch.edge_relation_ids,
            pos[:, 0],
            pos[:, 2],
            pos[:, 1],
            mask_reverse=self.cfg.sampler.mask_reverse_supervised_edges,
        )

    def _sample_batch_fast(self, edge_indices: np.ndarray) -> SampledBatch:
        """Context via engine BFS (~30ms); negatives via numpy corruptions.

        Contextual negatives are sampled locally because GNN states only exist
        for subgraph nodes. A second, genuinely global corruption is retained
        for the auxiliary loss that directly trains the exported base tables.
        This path does not reject accidental known positives.
        """
        batch = self.engine.sample_subgraph(
            edge_indices,
            fanouts=[self.cfg.sampler.hop1_fanout, self.cfg.sampler.hop2_fanout],
            direction=self.cfg.sampler.bfs_direction,
            negative_config=None,
        )
        k = self.cfg.sampler.negatives_per_positive
        pos = batch.positives.astype(np.int64)  # (B, 3) local (src, rel, dst)
        B = pos.shape[0]
        n_sub = max(int(batch.n_nodes), 1)
        # Local draws score contextual states. Global draws score base tables.
        sub_draws = self.rng.integers(0, n_sub, size=(B, k), dtype=np.int64)
        glob_draws = self.rng.integers(0, int(self.snapshot.num_nodes), size=(B, k), dtype=np.int64)
        head_mask = self.rng.random((B, k)) < self.cfg.sampler.hard_negative_head_probability
        neg = np.empty((B, k, 3), dtype=np.int64)
        neg[:, :, 1] = pos[:, 1:2]
        neg[:, :, 0] = np.where(head_mask, sub_draws, np.repeat(pos[:, 0:1], k, axis=1))
        neg[:, :, 2] = np.where(head_mask, np.repeat(pos[:, 2:3], k, axis=1), sub_draws)
        pos_src_global = batch.node_ids_global[pos[:, 0]][:, None]
        pos_dst_global = batch.node_ids_global[pos[:, 2]][:, None]
        neg_src_global = np.where(head_mask, glob_draws, pos_src_global)
        neg_dst_global = np.where(head_mask, pos_dst_global, glob_draws)
        senders, receivers, edge_rels = self._masked_context(batch)
        sem_a, sem_p, sem_n = self._semantic_batch()
        return SampledBatch(
            node_ids=np.arange(batch.n_nodes, dtype=np.int64),
            node_global_ids=batch.node_ids_global,
            node_type_ids=batch.node_type_ids,
            senders=senders,
            receivers=receivers,
            edge_rel_ids=edge_rels,
            positive_src=pos[:, 0],
            positive_dst=pos[:, 2],
            positive_rel=pos[:, 1],
            negative_src=neg[:, :, 0],
            negative_dst=neg[:, :, 2],
            negative_rel=neg[:, :, 1],
            negative_src_global=neg_src_global,
            negative_dst_global=neg_dst_global,
            semantic_anchor_global=sem_a,
            semantic_positive_global=sem_p,
            semantic_negative_global=sem_n,
        )


def sampled_batch_to_tensor_dict(batch: SampledBatch):
    graph = {
        "nodes": tf.convert_to_tensor(
            np.stack([batch.node_global_ids, batch.node_type_ids], axis=1), dtype=tf.float32),
        "edges": tf.convert_to_tensor(batch.edge_rel_ids[:, None], dtype=tf.float32),
        "senders": tf.convert_to_tensor(batch.senders, dtype=tf.int64),
        "receivers": tf.convert_to_tensor(batch.receivers, dtype=tf.int64),
        "n_nodes": tf.convert_to_tensor([len(batch.node_ids)], dtype=tf.int64),
        "n_edges": tf.convert_to_tensor([len(batch.edge_rel_ids)], dtype=tf.int64),
        "n_graphs": tf.convert_to_tensor(1, dtype=tf.int64),
    }
    labels = {
        "positive_src": tf.convert_to_tensor(batch.positive_src, dtype=tf.int64),
        "positive_dst": tf.convert_to_tensor(batch.positive_dst, dtype=tf.int64),
        "positive_rel": tf.convert_to_tensor(batch.positive_rel, dtype=tf.int64),
        "negative_src": tf.convert_to_tensor(batch.negative_src, dtype=tf.int64),
        "negative_dst": tf.convert_to_tensor(batch.negative_dst, dtype=tf.int64),
        "negative_rel": tf.convert_to_tensor(batch.negative_rel, dtype=tf.int64),
        "negative_src_global": tf.convert_to_tensor(batch.negative_src_global, dtype=tf.int64),
        "negative_dst_global": tf.convert_to_tensor(batch.negative_dst_global, dtype=tf.int64),
        "semantic_anchor_global": tf.convert_to_tensor(batch.semantic_anchor_global, dtype=tf.int64),
        "semantic_positive_global": tf.convert_to_tensor(batch.semantic_positive_global, dtype=tf.int64),
        "semantic_negative_global": tf.convert_to_tensor(batch.semantic_negative_global, dtype=tf.int64),
    }
    return graph, labels


def output_signature(negatives_per_positive: int, semantic_negatives: int = 16):
    graph_sig = {
        "nodes": tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
        "edges": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
        "senders": tf.TensorSpec(shape=(None,), dtype=tf.int64),
        "receivers": tf.TensorSpec(shape=(None,), dtype=tf.int64),
        "n_nodes": tf.TensorSpec(shape=(1,), dtype=tf.int64),
        "n_edges": tf.TensorSpec(shape=(1,), dtype=tf.int64),
        "n_graphs": tf.TensorSpec(shape=(), dtype=tf.int64),
    }
    labels_sig = {
        "positive_src": tf.TensorSpec(shape=(None,), dtype=tf.int64),
        "positive_dst": tf.TensorSpec(shape=(None,), dtype=tf.int64),
        "positive_rel": tf.TensorSpec(shape=(None,), dtype=tf.int64),
        "negative_src": tf.TensorSpec(shape=(None, negatives_per_positive), dtype=tf.int64),
        "negative_dst": tf.TensorSpec(shape=(None, negatives_per_positive), dtype=tf.int64),
        "negative_rel": tf.TensorSpec(shape=(None, negatives_per_positive), dtype=tf.int64),
        "negative_src_global": tf.TensorSpec(shape=(None, negatives_per_positive), dtype=tf.int64),
        "negative_dst_global": tf.TensorSpec(shape=(None, negatives_per_positive), dtype=tf.int64),
        "semantic_anchor_global": tf.TensorSpec(shape=(None,), dtype=tf.int64),
        "semantic_positive_global": tf.TensorSpec(shape=(None,), dtype=tf.int64),
        "semantic_negative_global": tf.TensorSpec(shape=(None, semantic_negatives), dtype=tf.int64),
    }
    return graph_sig, labels_sig


class ParallelYagoSampler:
    """Threaded feeder: N per-thread engines over one shared snapshot.

    Sampling is partly GIL-bound (Python-level rejection loops), so a pool of
    worker threads each owning a ``YagoSampler`` (own engine + rng, shared
    read-only snapshot) raises batch throughput ~linearly for small N.
    Batches are produced in submission order per worker; overall order across
    workers is interleaved but deterministic per worker seed.
    """

    def __init__(self, cfg: AppConfig, *, num_workers: int = 4, mode: str = "ram"):
        from concurrent.futures import ThreadPoolExecutor
        from .prepare import build_snapshot
        self.cfg = cfg
        self.snapshot = build_snapshot(cfg, mode=mode)
        self.workers = [
            YagoSampler(cfg, snapshot=self.snapshot, seed=cfg.sampler.seed + 1000 * (i + 1))
            for i in range(max(1, num_workers))
        ]
        self.pool = ThreadPoolExecutor(
            max_workers=len(self.workers),
            thread_name_prefix="yago-sample",
        )
        self._pending = [
            self.pool.submit(w.sample_batch) for w in self.workers
            for _ in range(cfg.sampler.worker_prefetch)
        ]

    @property
    def num_nodes(self) -> int:
        return self.snapshot.num_nodes

    @property
    def num_relations(self) -> int:
        return self.snapshot.num_relations

    @property
    def num_node_types(self) -> int:
        return self.workers[0].num_node_types

    def sample_batch(self) -> SampledBatch:
        fut = self._pending.pop(0)
        batch = fut.result()
        owner = self.workers[len(self._pending) % len(self.workers)]
        self._pending.append(self.pool.submit(owner.sample_batch))
        return batch

    def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=True)
