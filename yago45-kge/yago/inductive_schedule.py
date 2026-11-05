"""Resumable, bounded-memory supervised entity coverage (training graph only)."""
from __future__ import annotations

from functools import lru_cache
import math

import numpy as np


class EntityCoverageSchedule:
    """Every training ID anchors one positive fact per complete pass.

    A seeded affine bijection of [0, N) avoids an N-row permutation allocation.
    This is a shuffled bijective traversal, not a uniformly random permutation.
    The batch position is derived from the *completed* optimizer-step counter;
    incident-fact choices use the trainer's checkpointed RNG. Tail-only training
    entities are included by sampling both incoming and outgoing facts, while
    retaining the fact's original direction. No support/test facts are read.
    """

    def __init__(self, num_entities, batch_size, passes, seed):
        if min(num_entities, batch_size, passes) <= 0:
            raise ValueError("entity count, batch size and passes must be positive")
        # Bound vectorized affine products to signed int64.
        if num_entities * num_entities >= np.iinfo(np.int64).max:
            raise ValueError("entity namespace exceeds affine int64 range")
        self.num_entities = int(num_entities)
        self.batch_size = int(batch_size)
        self.passes = int(passes)
        self.seed = int(seed)
        self.steps_per_pass = math.ceil(num_entities / batch_size)
        self.total_steps = self.steps_per_pass * passes

    @lru_cache(maxsize=4)
    def permutation(self, epoch):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, 417]))
        multiplier = int(rng.integers(1, max(2, self.num_entities)))
        while math.gcd(multiplier, self.num_entities) != 1:
            multiplier += 1
        return multiplier, int(rng.integers(self.num_entities))

    def anchors(self, completed_steps):
        if not 0 <= completed_steps < self.total_steps:
            raise ValueError("step lies outside the coverage schedule")
        epoch, batch = divmod(completed_steps, self.steps_per_pass)
        start = batch * self.batch_size
        positions = np.arange(start, min(start + self.batch_size, self.num_entities), dtype=np.int64)
        multiplier, offset = self.permutation(epoch)
        return (positions * multiplier + offset) % self.num_entities

    def positives(self, graph, completed_steps, rng):
        anchors = self.anchors(completed_steps)
        indices = []
        for node in anchors:
            sampled, degree = graph.draw_incident(int(node), 1, rng)
            if degree == 0:
                raise ValueError(f"training entity {node} has no training fact; cannot satisfy coverage")
            indices.append(int(sampled[0]))
        indices = np.asarray(indices, dtype=np.int64)
        source, target = graph.src[indices].copy(), graph.dst[indices].copy()
        relations = graph.rel[indices]
        reverse = relations >= graph.num_relations
        source[reverse], target[reverse] = target[reverse], source[reverse].copy()
        triples = np.stack([source, relations % graph.num_relations, target], axis=1)
        if np.any(triples[:, [0, 2]] >= self.num_entities):
            raise ValueError("unseen entity found in training positives")
        return anchors, triples

    def validate_graph(self, graph):
        """Fail before training if strict entity coverage cannot be fulfilled."""
        for start in range(0, len(graph.edges), 1000000):
            endpoints = graph.edges[start:start + 1000000, [0, 2]]
            if endpoints.min() < 0 or endpoints.max() >= self.num_entities:
                raise ValueError("training context contains a held-out/invalid entity")
        for start in range(0, self.num_entities, 1000000):
            stop = min(start + 1000000, self.num_entities)
            if graph.snapshot is None:
                degrees = np.diff(graph.offsets[start:stop + 1])
            else:
                degrees = (np.diff(graph.snapshot.out.indptr[start:stop + 1]) +
                           np.diff(graph.snapshot.in_.indptr[start:stop + 1]))
            if np.any(degrees == 0):
                raise ValueError("training namespace contains an isolated entity")

    def progress(self, completed_steps):
        if not 0 <= completed_steps <= self.total_steps:
            raise ValueError("step lies outside the coverage schedule")
        full_passes, batch = divmod(completed_steps, self.steps_per_pass)
        partial = min(batch * self.batch_size, self.num_entities)
        return dict(protocol="entity-anchored incident training facts; affine bijection v1",
                    training_entities=self.num_entities, requested_passes=self.passes,
                    completed_steps=completed_steps, target_steps=self.total_steps,
                    complete=completed_steps == self.total_steps,
                    completed_entity_passes=full_passes, entities_in_partial_pass=partial,
                    supervised_anchor_draws=full_passes * self.num_entities + partial,
                    guaranteed_minimum_supervised_appearances_per_entity=full_passes,
                    excludes_unseen_entities=True,
                    guarantee="positive endpoint participation, not nonzero-gradient or convergence guarantee")
