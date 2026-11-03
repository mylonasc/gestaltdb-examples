"""Sampler + tf.data batch conversion for YAGO."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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


class YagoSampler:
    def __init__(self, cfg: AppConfig, *, mode: str = "ram", snapshot=None, seed: int | None = None):
        from .prepare import build_snapshot
        from gestaltdb.sampling import HardNegativeConfig, SamplerEngine
        self.cfg = cfg
        self._HardNegativeConfig = HardNegativeConfig
        self.snapshot = snapshot if snapshot is not None else build_snapshot(cfg, mode=mode)
        self.engine = SamplerEngine(self.snapshot, seed=cfg.sampler.seed if seed is None else seed)
        self.rng = np.random.default_rng(cfg.sampler.seed if seed is None else seed)

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
        if self.cfg.sampler.fast_negatives:
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
        return SampledBatch(
            node_ids=np.arange(batch.n_nodes, dtype=np.int64),
            node_global_ids=batch.node_ids_global,
            node_type_ids=batch.node_type_ids,
            senders=batch.senders,
            receivers=batch.receivers,
            edge_rel_ids=batch.edge_relation_ids,
            positive_src=batch.positives[:, 0].astype(np.int64),
            positive_dst=batch.positives[:, 2].astype(np.int64),
            positive_rel=batch.positives[:, 1].astype(np.int64),
            negative_src=batch.negatives[:, :, 0].astype(np.int64),
            negative_dst=batch.negatives[:, :, 2].astype(np.int64),
            negative_rel=batch.negatives[:, :, 1].astype(np.int64),
        )

    def _sample_batch_fast(self, edge_indices: np.ndarray) -> SampledBatch:
        """Context via engine BFS (~30ms); negatives via numpy corruptions.

        No rejection sampling: a uniform corruption collides with a true train
        triple with probability ~1e-5, so filtering is skipped (standard
        practice for in-training negatives). Half the corruptions replace an
        endpoint with a node from the sampled subgraph (harder), half with a
        globally uniform node; relations are kept.
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
        # In-subgraph draws are local ids; global draws are converted to local
        # positions (exact when the node is in the subgraph, else a random
        # local node — ids always stay valid for the model gather).
        order = np.argsort(batch.node_ids_global)
        sorted_ids = batch.node_ids_global[order]
        sub_draws = self.rng.integers(0, n_sub, size=(B, k), dtype=np.int64)
        glob_draws = self.rng.integers(0, int(self.snapshot.num_nodes), size=(B, k), dtype=np.int64)
        uniq, inv = np.unique(glob_draws, return_inverse=True)
        loc = np.searchsorted(sorted_ids, uniq)
        hit = (loc < n_sub) & (sorted_ids[np.minimum(loc, n_sub - 1)] == uniq)
        back = np.empty_like(uniq)
        back[hit] = order[loc[hit]]
        back[~hit] = self.rng.integers(0, n_sub, size=int((~hit).sum()), dtype=np.int64)
        glob_local = back[inv].reshape(B, k)
        head_mask = self.rng.random((B, k)) < self.cfg.sampler.hard_negative_head_probability
        glob_mask = self.rng.random((B, k)) < self.cfg.sampler.fast_negative_global_fraction
        draws = np.where(glob_mask, glob_local, sub_draws)
        neg = np.empty((B, k, 3), dtype=np.int64)
        neg[:, :, 1] = pos[:, 1:2]
        neg[:, :, 0] = np.where(head_mask, draws, np.repeat(pos[:, 0:1], k, axis=1))
        neg[:, :, 2] = np.where(head_mask, np.repeat(pos[:, 2:3], k, axis=1), draws)
        return SampledBatch(
            node_ids=np.arange(batch.n_nodes, dtype=np.int64),
            node_global_ids=batch.node_ids_global,
            node_type_ids=batch.node_type_ids,
            senders=batch.senders,
            receivers=batch.receivers,
            edge_rel_ids=batch.edge_relation_ids,
            positive_src=pos[:, 0],
            positive_dst=pos[:, 2],
            positive_rel=pos[:, 1],
            negative_src=neg[:, :, 0],
            negative_dst=neg[:, :, 2],
            negative_rel=neg[:, :, 1],
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
    }
    return graph, labels


def output_signature(negatives_per_positive: int):
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
