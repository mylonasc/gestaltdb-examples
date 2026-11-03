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
    def __init__(self, cfg: AppConfig, *, mode: str = "ram"):
        from .prepare import build_snapshot
        from gestaltdb.sampling import HardNegativeConfig, SamplerEngine
        self.cfg = cfg
        self._HardNegativeConfig = HardNegativeConfig
        self.snapshot = build_snapshot(cfg, mode=mode)
        self.engine = SamplerEngine(self.snapshot, seed=cfg.sampler.seed)
        self.rng = np.random.default_rng(cfg.sampler.seed)

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
