"""Coverage guarantees apply to positive endpoints, not random sample averages."""
import json

import numpy as np
import pytest

from yago.inductive_data import NeighborhoodGraph
from yago.inductive_schedule import EntityCoverageSchedule


@pytest.mark.parametrize("nodes,batch", [(1, 4), (7, 3), (12, 4), (31, 8)])
def test_every_entity_is_anchored_exactly_twice_including_short_final_batch(nodes, batch):
    schedule = EntityCoverageSchedule(nodes, batch, 2, 13)
    for epoch in range(2):
        anchors = np.concatenate([schedule.anchors(epoch * schedule.steps_per_pass + i)
                                  for i in range(schedule.steps_per_pass)])
        np.testing.assert_array_equal(np.sort(anchors), np.arange(nodes))
    final = schedule.progress(schedule.total_steps)
    assert final["complete"]
    assert final["supervised_anchor_draws"] == nodes * 2
    assert final["guaranteed_minimum_supervised_appearances_per_entity"] == 2


@pytest.mark.parametrize("compact", [False, True])
def test_schedule_uses_original_training_facts_and_exact_rng_resume(tmp_path, compact):
    edges = np.asarray([[0, 0, 1], [1, 1, 2], [2, 0, 3], [3, 1, 4], [4, 0, 5], [5, 1, 6]])
    graph = NeighborhoodGraph(edges, 9, 2, snapshot_path=tmp_path / 'snapshot' if compact else None,
                              compact=compact)
    schedule = EntityCoverageSchedule(7, 3, 2, 13)
    schedule.validate_graph(graph)
    rng = np.random.default_rng(13)
    reference = []
    state = None
    for step in range(schedule.total_steps):
        anchors, positives = schedule.positives(graph, step, rng)
        assert np.all((positives[:, 0] == anchors) | (positives[:, 2] == anchors))
        assert all(tuple(row) in set(map(tuple, edges)) for row in positives)
        assert positives[:, [0, 2]].max() < 7
        reference.append(positives)
        if step == 1:
            state = json.loads(json.dumps(rng.bit_generator.state))
    restored_rng = np.random.default_rng()
    restored_rng.bit_generator.state = state
    restored = EntityCoverageSchedule(7, 3, 2, 13)
    for step in range(2, schedule.total_steps):
        _, positives = restored.positives(graph, step, restored_rng)
        np.testing.assert_array_equal(positives, reference[step])
    assert restored_rng.bit_generator.state == rng.bit_generator.state


def test_coverage_rejects_heldout_context_and_untrainable_ids():
    schedule = EntityCoverageSchedule(3, 2, 2, 13)
    with pytest.raises(ValueError, match="held-out"):
        schedule.validate_graph(NeighborhoodGraph(np.asarray([[0, 0, 3]]), 4, 1))
    with pytest.raises(ValueError, match="isolated"):
        schedule.validate_graph(NeighborhoodGraph(np.asarray([[0, 0, 1]]), 3, 1))


@pytest.mark.parametrize("new_batch", [1, 4, 16])
def test_batch_size_migration_preserves_exact_anchor_sequence_and_progress(new_batch):
    original = EntityCoverageSchedule(17, 3, 2, 13)
    step = 3
    draws = original.progress(step)["supervised_anchor_draws"]
    phases = [*original.phases, dict(start_step=step, start_anchor_draws=draws, batch_size=new_batch)]
    migrated = EntityCoverageSchedule(17, new_batch, 2, 13, phases=phases)
    before = np.concatenate([original.anchors(i) for i in range(step)])
    after = np.concatenate([migrated.anchors(i) for i in range(step, migrated.total_steps)])
    expected = np.concatenate([original.anchors(i) for i in range(original.total_steps)])
    np.testing.assert_array_equal(np.r_[before, after], expected)
    assert migrated.progress(migrated.total_steps)["guaranteed_minimum_supervised_appearances_per_entity"] == 2
    assert migrated.draws_at(step) == draws
    for i in range(step):
        assert migrated.draws_at(i) == original.draws_at(i)
    with pytest.raises(ValueError, match="skips or repeats"):
        EntityCoverageSchedule(17, new_batch, 2, 13,
            phases=[*original.phases, dict(start_step=step, start_anchor_draws=draws + 1, batch_size=new_batch)])
