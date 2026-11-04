"""InfoNCE contrastive losses for KGE."""
from __future__ import annotations

import tensorflow as tf


def contrastive_infonce_loss(positive_logits: tf.Tensor, negative_logits: tf.Tensor, temperature: float) -> tf.Tensor:
    logits = tf.concat([positive_logits[:, tf.newaxis], negative_logits], axis=1) / temperature
    labels = tf.zeros(tf.shape(positive_logits)[0], dtype=tf.int32)
    return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True))


def weighted_infonce_loss(
    positive_logits: tf.Tensor, negative_logits: tf.Tensor, temperature: float, adv_temperature: float
) -> tf.Tensor:
    """InfoNCE with self-adversarial negative weights (RotatE-style).

    Negatives with higher scores get larger weights: w = softmax(neg / adv_T).
    Weights are detached so gradients only scale the logit terms.
    """
    neg = negative_logits / temperature
    pos = positive_logits / temperature
    weights = tf.stop_gradient(tf.nn.softmax(neg / adv_temperature, axis=1))
    log_prob = tf.nn.log_softmax(tf.concat([pos[:, tf.newaxis], neg], axis=1), axis=1)
    return -tf.reduce_mean(log_prob[:, 0] + tf.reduce_sum(weights * log_prob[:, 1:], axis=1))
