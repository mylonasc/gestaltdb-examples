# Reproducibility and Operational Checklist

## Before training

- [ ] Record source archive and parser/filter settings.
- [ ] Fingerprint train/validation/test files.
- [ ] Split before building the DB and sampler snapshot.
- [ ] Verify held-out edges are absent from training arrays.
- [ ] Fix random seeds for splitting, sampling, TensorFlow, and evaluation.
- [ ] Choose the intended task: link prediction, semantic cosine, or inductive.

## Batch correctness

- [ ] Exact supervised-edge overlap with message edges is zero.
- [ ] Configured reverse-edge overlap is zero.
- [ ] Local GNN IDs and global table IDs are not conflated.
- [ ] “Global” negatives are genuinely global.
- [ ] Known-positive rejection is tested where claimed.

## Artifact correctness

- [ ] Persist score function and dimensions.
- [ ] Name representations explicitly (`entity_base_200d`, not ambiguous
  `entity`).
- [ ] Persist relation table paired with each scorer.
- [ ] Record split fingerprint, step, seed, optimizer, and negative policy.
- [ ] Validate finite values and map alignment.
- [ ] Retain full checkpoints when contextual GNN inference is required.

## Evaluation

- [ ] Distinguish sampled-filtered from full-candidate ranking.
- [ ] Include relation-range/type-compatible candidates.
- [ ] Unit-test multi-object filtering.
- [ ] Report per-relation and degree buckets.
- [ ] For semantic claims, report balanced 1-NN, P@10, and cosine margin.
- [ ] Treat t-SNE as visualization, not evidence by itself.
- [ ] For inductive claims, hold out complete entities.

## Performance

- [ ] Measure feeder-only batches/s.
- [ ] Measure compute-only steps/s on a cached batch.
- [ ] Measure end-to-end steps/s with fresh batches.
- [ ] Confirm end-to-end approximately equals compute-only.
- [ ] Re-check after changing batch size, workers, fanouts, or negatives.
- [ ] Monitor GPU utilization and peak memory, not memory allocation alone.

## Experiment management

- [ ] Store every trial in an isolated directory.
- [ ] Use persistent Optuna storage and export a compact trial table.
- [ ] Do not overwrite the canonical artifact during search.
- [ ] Keep machine-readable curves and summaries.
- [ ] Retain checkpoints for long or interruption-prone runs.

The checkpoint lesson is important: one local 30k continuation completed all
updates and wrote its metrics, but a server restart during final export lost the
weights because checkpoints had been disabled. Metrics without a recoverable
artifact are not a deployable result.
