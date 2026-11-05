"""Transferable relational graph autoencoder plus independent shallow KGE."""
from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
import tensorflow as tf


def complex_score(subject, relation, obj):
    sr, si = tf.split(subject, 2, axis=-1)
    rr, ri = tf.split(relation, 2, axis=-1)
    ore, oi = tf.split(obj, 2, axis=-1)
    return tf.reduce_sum(rr * (sr * ore + si * oi) + ri * (sr * oi - si * ore), axis=-1)


def infonce(positive, negative, temperature):
    logits = tf.concat([positive[:, None], negative], axis=1) / temperature
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tf.zeros(tf.shape(positive)[0], tf.int32), logits=logits))


class InductiveAutoencoder(tf.keras.Model):
    """No entity IDs enter encode(); unseen entities require no new parameters.

    Direction-specific relation embeddings condition each mean-aggregated
    message. Separate self transforms preserve features at isolated nodes.
    The independent shallow branch learns reusable embeddings for known IDs.
    """

    def __init__(self, num_train_nodes, num_relations, dim=200, hidden_dim=256, layers=2,
                 encoder="relational", base_dim=None, embedding_device="gpu"):
        super().__init__()
        base_dim = dim if base_dim is None else base_dim
        if dim % 2 or base_dim % 2 or layers < 1:
            raise ValueError("dim must be even and layers must be positive")
        self.input_projection = tf.keras.layers.Dense(hidden_dim, activation="relu")
        self.message_relations = tf.keras.layers.Embedding(2 * num_relations, hidden_dim)
        self.encoder_kind = encoder
        if encoder == "tf-gnns":
            from tf_gnns import GraphNetMPNN_MLP
            self.gnn = GraphNetMPNN_MLP(
                units=hidden_dim, core_units=hidden_dim, core_size=hidden_dim,
                gi_units=hidden_dim, core_steps=layers, node_output_size=hidden_dim,
                edge_output_size=hidden_dim, aggregation_function="mean")
        elif encoder != "relational":
            raise ValueError(f"unknown encoder: {encoder}")
        self.message_layers = ([tf.keras.layers.Dense(hidden_dim, activation="relu") for _ in range(layers)]
                               if encoder == "relational" else [])
        self.self_layers = ([tf.keras.layers.Dense(hidden_dim) for _ in range(layers)]
                            if encoder == "relational" else [])
        # Explicit attributes also make every layer reachable by TF checkpoints
        # on Keras versions whose tracked lists are not TF Trackable containers.
        for i, (message, self_transform) in enumerate(zip(self.message_layers, self.self_layers)):
            setattr(self, f"message_{i}", message)
            setattr(self, f"self_transform_{i}", self_transform)
        self.output_projection = tf.keras.layers.Dense(dim)
        self.context_relations = tf.keras.layers.Embedding(num_relations, dim)
        self.shallow_device = "/CPU:0" if embedding_device == "cpu" else "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
        self.cpu_embeddings = embedding_device == "cpu"
        with tf.device(self.shallow_device):
            self.entity_table = tf.keras.layers.Embedding(num_train_nodes, base_dim)
            self.base_relations = tf.keras.layers.Embedding(num_relations, base_dim)
            self.entity_table(tf.constant([0]))
            self.base_relations(tf.constant([0]))
        # Build every variable, including shallow tables, before restore.
        self.encode(dict(features=tf.zeros((1, 2 * num_relations + 1)),
                         senders=tf.zeros((0,), tf.int32), receivers=tf.zeros((0,), tf.int32),
                         relations=tf.zeros((0,), tf.int32)))
        if encoder == "tf-gnns":
            # GraphNet containers are plain Python objects. Register their Keras
            # submodels explicitly so TF checkpoints preserve *all* GNN weights.
            blocks = [self.gnn.g_enc_determ, *self.gnn.g_core_determ, self.gnn.g_dec_determ]
            for i, block in enumerate(blocks):
                for name in ("edge_function", "node_function"):
                    component = getattr(block, name, None)
                    if component is not None:
                        setattr(self, f"gnn_component_{i}_{name}", component)
        self.context_relations(tf.constant([0]))

    def encode(self, graph):
        h = self.input_projection(graph["features"])
        rel = self.message_relations(graph["relations"])
        if self.encoder_kind == "tf-gnns":
            encoded = self.gnn(dict(nodes=h, edges=rel, senders=graph["senders"],
                                    receivers=graph["receivers"],
                                    n_nodes=tf.reshape(tf.shape(h)[0], (1,)),
                                    n_edges=tf.reshape(tf.shape(rel)[0], (1,)),
                                    n_graphs=tf.constant(1, tf.int32)))
            h = h + encoded["nodes"]
        for message_layer, self_layer in zip(self.message_layers, self.self_layers):
            messages = message_layer(tf.concat([tf.gather(h, graph["senders"]), rel], axis=-1))
            aggregate = tf.math.unsorted_segment_mean(messages, graph["receivers"], tf.shape(h)[0])
            h = tf.nn.relu(self_layer(h) + aggregate)
        # Norm control makes scores comparable across different sampled contexts.
        return tf.math.l2_normalize(self.output_projection(h), axis=-1)

    def contextual_scores(self, encoded, triples):
        return complex_score(tf.gather(encoded, triples[..., 0]),
                             self.context_relations(triples[..., 1]),
                             tf.gather(encoded, triples[..., 2]))

    @tf.function(reduce_retracing=True)
    def predict_graph(self, graph, triples):
        return self.contextual_scores(self.encode(graph), triples)

    def shallow_scores(self, triples):
        # Variable placement alone is insufficient: an unplaced Keras gather
        # may execute on GPU and transfer the *entire* CPU table there first.
        with tf.device(self.shallow_device):
            return complex_score(self._base_lookup(self.entity_table, triples[..., 0]),
                                 self._base_lookup(self.base_relations, triples[..., 1]),
                                 self._base_lookup(self.entity_table, triples[..., 2]))

    def _base_lookup(self, layer, ids):
        if self.cpu_embeddings:
            # ResourceGather reads just the requested rows. ReadVariableOp +
            # GatherV2 can hold a full-table tensor alive during backprop and
            # trigger multi-GB copy-on-write before the sparse optimizer runs.
            return layer.embeddings.value.sparse_read(ids)
        return layer(ids)

    def base_vectors(self, ids):
        with tf.device(self.shallow_device):
            return self._base_lookup(self.entity_table, tf.convert_to_tensor(ids))
