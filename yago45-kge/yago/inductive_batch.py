"""Training-only minibatch preparation, shared by training and profiling."""
from time import perf_counter

import numpy as np


def prepare_batch(context, training_known, relation_pools, num_train_nodes, *,
                  rng, batch_size, negatives, negative_policy, fanouts,
                  schedule=None, completed_steps=0):
    started = perf_counter()
    if schedule is None:
        positive = context.edges[rng.integers(len(context.edges), size=batch_size)]
    else:
        anchors, positive = schedule.positives(context, completed_steps, rng)
        if not np.all((positive[:, 0] == anchors) | (positive[:, 2] == anchors)):
            raise AssertionError("supervised positives do not contain scheduled entities")
    positives_done = perf_counter()
    negative = np.repeat(positive[:, None, :], negatives, axis=1)
    for i, (source, relation, _) in enumerate(positive):
        forbidden = training_known[(int(source), int(relation))]
        if len(forbidden) >= num_train_nodes:
            raise ValueError("query has no valid negative tails")
        for j in range(negatives):
            use_range = (negative_policy == "relation-range" or
                         (negative_policy == "mixed" and rng.random() < .5))
            pool = relation_pools[int(relation)]
            if use_range and all(int(x) in forbidden for x in pool):
                use_range = False
            for _ in range(10000):
                candidate = int(rng.choice(pool)) if use_range else int(rng.integers(num_train_nodes))
                if candidate not in forbidden:
                    negative[i, j, 2] = candidate
                    break
            else:
                raise ValueError("negative rejection limit reached")
    negatives_done = perf_counter()
    roots = np.unique(np.r_[positive[:, [0, 2]].ravel(), negative[..., [0, 2]].ravel()])
    ids, graph = context.sample(roots, fanouts, rng, masked=positive)
    sampling_done = perf_counter()

    def localize(triples):
        local = triples.copy()
        local[..., 0] = np.searchsorted(ids, triples[..., 0])
        local[..., 2] = np.searchsorted(ids, triples[..., 2])
        return local.astype(np.int32)

    arrays = (graph, localize(positive), localize(negative),
              positive.astype(np.int32), negative.astype(np.int32))
    timings = dict(positive_seconds=positives_done - started,
                   negatives_seconds=negatives_done - positives_done,
                   neighborhood_seconds=sampling_done - negatives_done,
                   localization_seconds=perf_counter() - sampling_done)
    return arrays, timings
