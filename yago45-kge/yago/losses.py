"""InfoNCE contrastive loss for KGE."""
from __future__ import annotations

import tensorflow as tf


def contrastive_infonce_loss(positive_logits: tf.Tensor, negative_logits: tf.Tensor, temperature: float) -> tf.Tensor:
    logits = tf.concat([positive_logits[:, tf.newaxis], negative_logits], axis=1) / temperature
    labels = tf.zeros(tf.shape(positive_logits)[0], dtype=tf.int32)
    return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True))
