"""Shared-memory producer tests: no TensorFlow or GPU required."""
import time

import numpy as np

from yago.inductive_batch import prepare_batch
from yago.inductive_data import GraphKnownObjects, NeighborhoodGraph
from yago.inductive_parallel import BatchProducerPool, SharedBatchBuffers, rng_for_step
from yago.inductive_schedule import EntityCoverageSchedule


def test_per_step_rng_is_deterministic_and_step_independent():
    first = rng_for_step(13, 7).integers(0, 1_000_000, size=8)
    np.testing.assert_array_equal(first, rng_for_step(13, 7).integers(0, 1_000_000, size=8))
    assert not np.array_equal(first, rng_for_step(13, 8).integers(0, 1_000_000, size=8))
    assert not np.array_equal(first, rng_for_step(14, 7).integers(0, 1_000_000, size=8))


def test_shared_buffers_roundtrip_preserves_arrays():
    buffers = SharedBatchBuffers(slots=2, max_nodes=16, max_edges=32, batch_size=3,
                                 negatives=2, feature_dim=5)
    try:
        graph = dict(features=np.arange(20, dtype=np.float32).reshape(4, 5),
                     senders=np.arange(8, dtype=np.int32),
                     receivers=np.arange(8, dtype=np.int32)[::-1].copy(),
                     relations=np.arange(8, dtype=np.int32) % 2)
        positive = np.arange(9, dtype=np.int32).reshape(3, 3)
        negative = np.arange(18, dtype=np.int32).reshape(3, 2, 3)
        buffers.write(1, 42, (graph, positive, negative, positive + 1, negative + 1))
        step, actual_graph, actual_positive, actual_negative, actual_global, actual_negative_global = buffers.read(1)
        assert step == 42
        for key in graph:
            np.testing.assert_array_equal(actual_graph[key], graph[key])
        np.testing.assert_array_equal(actual_positive, positive)
        np.testing.assert_array_equal(actual_negative, negative)
        np.testing.assert_array_equal(actual_global, positive + 1)
        np.testing.assert_array_equal(actual_negative_global, negative + 1)
    finally:
        buffers.close()


def test_producer_pool_covers_every_entity_and_consumes_in_order():
    rng = np.random.default_rng(5)
    edges = np.asarray([(i, i % 2, (i * 7 + 3) % 40) for i in range(40) if i != (i * 7 + 3) % 40])
    edges = np.unique(edges, axis=0)
    num_nodes = 40
    context = NeighborhoodGraph(edges, num_nodes, 2, sampling_version=3)
    schedule = EntityCoverageSchedule(num_nodes, 5, 2, 13)
    known = GraphKnownObjects(context)
    pools = {int(rel): np.unique(edges[edges[:, 1] == rel, 2]) for rel in np.unique(edges[:, 1])}
    pool = BatchProducerPool(
        context=context, training_known=known, relation_pools=pools,
        num_train_nodes=num_nodes, schedule=schedule, batch_size=5, negatives=2,
        negative_policy="mixed", fanouts=(2, 2), seed=13,
        end_step=schedule.total_steps, workers=2, slots=4,
        max_nodes=256, max_edges=256,
    )
    try:
        pool.start()
        pool.begin(0)
        seen = []
        for step in range(schedule.total_steps):
            slot = pool.get(step)
            try:
                actual_step, graph, positive_local, negative_local, positive, negative = pool.read(slot)
                assert actual_step == step
                assert np.isfinite(graph["features"]).all()
                assert len(graph["senders"]) == len(graph["receivers"]) == len(graph["relations"])
                assert len(positive_local) == len(positive) == len(negative_local) == len(negative)
                # Local triples index into the sampled subgraph; only the global
                # copy can be compared with the scheduled entity anchors.
                assert positive_local[..., [0, 2]].min() >= 0
                assert positive_local[..., [0, 2]].max() < len(graph["features"])
                anchors = schedule.anchors(step)
                assert set(anchors) <= set(positive[:, 0].tolist()) | set(positive[:, 2].tolist())
                seen.extend(anchors.tolist())
            finally:
                pool.release(slot)
        np.testing.assert_array_equal(np.asarray(seen, dtype=np.int64),
                                      np.concatenate([schedule.anchors(step) for step in range(schedule.total_steps)]))
        progress = schedule.progress(schedule.total_steps)
        assert progress["complete"]
        assert progress["guaranteed_minimum_supervised_appearances_per_entity"] == 2
    finally:
        pool.close()


def test_producer_batches_equal_sequential_preparation():
    """The fork pool must reproduce the exact per-step RNG data stream."""
    edges = np.asarray([(i, i % 2, (i * 7 + 3) % 40) for i in range(40) if i != (i * 7 + 3) % 40])
    edges = np.unique(edges, axis=0)
    num_nodes = 40
    context = NeighborhoodGraph(edges, num_nodes, 2, sampling_version=3)
    schedule = EntityCoverageSchedule(num_nodes, 5, 2, 13)
    known = GraphKnownObjects(context)
    pools = {int(rel): np.unique(edges[edges[:, 1] == rel, 2]) for rel in np.unique(edges[:, 1])}
    kwargs = dict(context=context, training_known=known, relation_pools=pools,
                  num_train_nodes=num_nodes, schedule=schedule, batch_size=5,
                  negatives=2, negative_policy="mixed", fanouts=(2, 2), seed=13,
                  max_nodes=256, max_edges=256)
    steps = list(range(6))
    pool = BatchProducerPool(end_step=len(steps), workers=2, slots=4, **kwargs)
    try:
        pool.start()
        pool.begin(0)
        produced = {}
        for step in steps:
            slot = pool.get(step)
            try:
                _, graph, *arrays = pool.read(slot)
                produced[step] = (step, {key: value.copy() for key, value in graph.items()},
                                  *(array.copy() for array in arrays))
            finally:
                pool.release(slot)
    finally:
        pool.close()
    for step in steps:
        arrays, _ = prepare_batch(context, known, pools, num_nodes,
                                  rng=rng_for_step(13, step), batch_size=5,
                                  negatives=2, negative_policy="mixed",
                                  fanouts=(2, 2), schedule=schedule,
                                  completed_steps=step)
        expected = arrays
        actual = produced[step][1:]
        for left, right in zip(actual, expected):
            if isinstance(left, dict):
                for key in left:
                    np.testing.assert_array_equal(left[key], right[key])
            else:
                np.testing.assert_array_equal(left, right)


def test_producer_pool_slow_consumer_receives_every_step():
    """Producers run ahead and exit while the trainer still drains the pipe.

    Regression test: liveness must only gate waiting, never discard a
    successfully received batch when producers exit at end-of-work.
    """
    edges = np.asarray([(s, (s + o) % 2, o) for s in range(16) for o in range(16) if s != o])
    num_nodes = 16
    context = NeighborhoodGraph(edges, num_nodes, 2, sampling_version=3)
    known = GraphKnownObjects(context)
    pools = {int(rel): np.unique(edges[edges[:, 1] == rel, 2]) for rel in np.unique(edges[:, 1])}
    pool = BatchProducerPool(
        context=context, training_known=known, relation_pools=pools,
        num_train_nodes=num_nodes, schedule=None, batch_size=2, negatives=2,
        negative_policy="mixed", fanouts=(2, 2), seed=13,
        end_step=4, workers=2, slots=4, max_nodes=256, max_edges=256,
        consume_timeout=30.0,
    )
    try:
        pool.start()
        pool.begin(0)
        for step in range(4):
            # Let producers run ahead, recycle freed slots and exit early.
            time.sleep(0.5)
            slot = pool.get(step)
            try:
                actual_step, graph, _, _, positive, _ = pool.read(slot)
                assert actual_step == step
                assert len(positive) == 2
                assert np.isfinite(graph["features"]).all()
            finally:
                pool.release(slot)
    finally:
        pool.close()
