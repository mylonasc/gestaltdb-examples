# Correctness, Splitting, and Leakage

## Split before building context

Validation and test triples must be removed before constructing:

- the GestaltDB training graph;
- integer edge arrays used for sampling;
- the CSR sampler snapshot;
- known-positive structures used by negative sampling.

The preparation pipeline keeps held-out endpoints observable by requiring them
to occur at least three times before selecting a held-out triple. This is a
transductive edge split: entities remain known, but the held-out edge is absent.

For inductive evaluation, this is insufficient. Entire nodes—not just selected
edges—must be held out. See [inductive tasks](07-completion-and-inductive-tasks.md).

## Supervised-edge leakage

The original sampler retained every seed edge in the GNN message graph. A direct
diagnostic found `128/128` supervised positives in each tested batch. A GNN could
therefore observe the exact relation it was asked to predict.

The corrected pipeline removes:

- `(source, relation, target)` seed edges;
- same-relation reverse copies `(target, relation, source)`.

Regression tests assert zero overlap. This masking is required for defensible
GNN link-prediction results.

## Filtered ranking bug

The known-positive index is keyed by `(subject, relation)` and stores objects.
The old evaluation tested whether `(subject, relation, candidate)` was a key,
which never filtered alternatives correctly.

Correct filtering is:

```python
candidate not in known_objects[(subject, relation)]
```

with the target object always retained. Tests include multiple valid objects for
one `(subject, relation)` pair.

## Sampled ranking is not full ranking

Metrics against one positive plus 50 random candidates are **sampled-filtered**
MRR and Hits@k. They must not be called standard full-candidate filtered MRR.
With millions of entities, uniformly random candidates are often easy.

The example therefore also evaluates **relation-range negatives**: candidates
are objects previously observed with the predicate. This is harder and usually
more representative.

## Scorer metadata

Training once defaulted to DistMult while evaluation defaulted to ComplEx.
Artifacts now persist:

- score function;
- entity and relation keys;
- dimensions;
- representation description;
- training and sampler settings.

Evaluation rejects an explicit scorer that conflicts with artifact metadata.
Silent train/evaluation mismatch is a correctness failure, not a tuning issue.

## Historical metrics

The historical 150k run reported sampled test MRR `0.467`, but its GNN context
contained supervised edges. The corrected filtering code gave the same rounded
score because known-positive collisions among 50 uniform draws were rare; that
does not repair the context leakage. Historical and corrected runs should not be
compared as if their protocols were identical.
