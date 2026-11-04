"""Row-sparse Adagrad for very large CPU embedding tables (TensorFlow)."""
import tensorflow as tf


class SparseAdagrad(tf.Module):
    """Native resource updates touch only gathered rows, with one moment table.

    In particular, this never constructs a dense [all_entities, dim] gradient
    or a full-size denominator tensor on each step. Duplicate gathers are
    coalesced before accumulation, matching dense Adagrad semantics.
    """

    def __init__(self, learning_rate, initial_accumulator=1e-8, epsilon=1e-7):
        super().__init__()
        self.learning_rate = float(learning_rate)
        self.initial_accumulator = float(initial_accumulator)
        self.epsilon = float(epsilon)
        self.iterations = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.accumulators = []
        self.indices = {}

    def build(self, variables):
        for i, variable in enumerate(variables):
            resource = variable.value if hasattr(variable, "value") and not isinstance(variable, tf.Variable) else variable
            with tf.device(resource.device):
                accumulator = tf.Variable(tf.fill(resource.shape, tf.cast(self.initial_accumulator, resource.dtype)),
                                          trainable=False, name=f"accumulator_{i}")
            setattr(self, f"accumulator_{i}", accumulator)
            self.accumulators.append(accumulator)
            self.indices[id(variable)] = i

    def apply_gradients(self, pairs):
        for gradient, variable in pairs:
            resource = variable.value if not isinstance(variable, tf.Variable) else variable
            accumulator = self.accumulators[self.indices[id(variable)]]
            kwargs = dict(var=resource.handle, accum=accumulator.handle,
                          lr=tf.cast(self.learning_rate, resource.dtype),
                          epsilon=tf.cast(self.epsilon, resource.dtype), update_slots=True)
            if isinstance(gradient, tf.IndexedSlices):
                indices, segment = tf.unique(gradient.indices)
                values = tf.math.unsorted_segment_sum(gradient.values, segment, tf.shape(indices)[0])
                tf.raw_ops.ResourceSparseApplyAdagradV2(grad=values, indices=indices, **kwargs)
            else:
                tf.raw_ops.ResourceApplyAdagradV2(grad=gradient, **kwargs)
        self.iterations.assign_add(1)
