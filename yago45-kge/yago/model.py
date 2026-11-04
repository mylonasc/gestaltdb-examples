"""GNN-autoencoder KGE model: 200-d entity/relation tables + tf_gnns MPNN encoder.

Canonical embeddings are the trainable `entity_emb` [N,200] and
`relation_emb` [R,200] tables. The GNN contextualizes node states for
scoring; the exported artifact is the 200-d base tables.
"""
from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import tensorflow as tf
from tf_gnns import GraphNetMPNN_MLP

from .config import ModelConfig


class YagoKGEModel(tf.keras.Model):
    def __init__(self, cfg: ModelConfig, *, num_nodes: int, num_node_types: int, num_relations: int):
        super().__init__()
        self.cfg = cfg
        self.entity_emb = tf.keras.layers.Embedding(num_nodes, cfg.entity_dim, name="entity_emb")
        self.node_type_emb = tf.keras.layers.Embedding(num_node_types, cfg.node_type_dim, name="node_type_emb")
        self.rel_emb = tf.keras.layers.Embedding(num_relations, cfg.relation_dim, name="relation_emb")
        self.node_proj = tf.keras.layers.Dense(cfg.hidden_dim, activation="relu", name="node_proj")
        self.edge_proj = tf.keras.layers.Dense(cfg.hidden_dim, activation="relu", name="edge_proj")
        self.gnn = GraphNetMPNN_MLP(
            units=cfg.hidden_dim,
            core_units=cfg.hidden_dim,
            core_size=cfg.hidden_dim,
            gi_units=cfg.hidden_dim,
            core_steps=cfg.mp_steps,
            node_output_size=cfg.hidden_dim,
            edge_output_size=cfg.hidden_dim,
            aggregation_function=cfg.aggregation_function,
        )
        self.dropout = tf.keras.layers.Dropout(cfg.dropout)
        self.score_rel = tf.keras.layers.Embedding(num_relations, cfg.hidden_dim, name="score_rel")
        # Build embedding tables eagerly: validation scores them directly and
        # shallow (no-GNN) mode never calls rel_emb inside encode().
        self.entity_emb.build((None,))
        self.node_type_emb.build((None,))
        self.rel_emb.build((None,))
        self.score_rel.build((None,))

    def encode(self, graph: dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        node_global = tf.cast(graph["nodes"][:, 0], tf.int64)
        node_type = tf.cast(graph["nodes"][:, 1], tf.int64)
        edge_rel = tf.cast(graph["edges"][:, 0], tf.int64)
        ent = self.entity_emb(node_global)
        typ = self.node_type_emb(tf.clip_by_value(node_type, 0, 10**9))
        nodes_in = self.node_proj(tf.concat([ent, typ], axis=-1))
        if not self.cfg.use_gnn:
            return self.dropout(nodes_in, training=training)
        graph_in = dict(graph)
        graph_in["nodes"] = nodes_in
        graph_in["edges"] = self.edge_proj(self.rel_emb(edge_rel))
        out = self.gnn(graph_in)
        return self.dropout(out["nodes"], training=training)

    @staticmethod
    def _split_complex(x: tf.Tensor):
        re, im = tf.split(x, 2, axis=-1)
        return re, im

    def _score(self, nodes: tf.Tensor, src: tf.Tensor, rel: tf.Tensor, dst: tf.Tensor) -> tf.Tensor:
        return self._score_impl(nodes, src, rel, dst, self.cfg.score_function, self.score_rel)

    @staticmethod
    def _score_impl(nodes, src, rel, dst, score_function, rel_table) -> tf.Tensor:
        if score_function == "complex":
            # ComplEx: Re(<h_s, r, conj(h_o)>); 200-d = 100 complex dims.
            hs_re, hs_im = YagoKGEModel._split_complex(tf.gather(nodes, src))
            ho_re, ho_im = YagoKGEModel._split_complex(tf.gather(nodes, dst))
            r_re, r_im = YagoKGEModel._split_complex(rel_table(rel))
            return tf.reduce_sum(
                r_re * (hs_re * ho_re + hs_im * ho_im)
                + r_im * (hs_re * ho_im - hs_im * ho_re),
                axis=-1,
            )
        src_h = tf.gather(nodes, src)
        dst_h = tf.gather(nodes, dst)
        rel_h = rel_table(rel)
        return tf.reduce_sum(src_h * rel_h * dst_h, axis=-1)

    def call(self, graph: dict[str, tf.Tensor], labels: dict[str, tf.Tensor], training: bool = False):
        nodes = self.encode(graph, training=training)
        pos = self._score(nodes, labels["positive_src"], labels["positive_rel"], labels["positive_dst"])
        neg_shape = tf.shape(labels["negative_src"])
        neg = self._score(
            nodes,
            tf.reshape(labels["negative_src"], [-1]),
            tf.reshape(labels["negative_rel"], [-1]),
            tf.reshape(labels["negative_dst"], [-1]),
        )
        return pos, tf.reshape(neg, neg_shape)
