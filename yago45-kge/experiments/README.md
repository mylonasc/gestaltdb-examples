# Experiment Records

This directory contains compact, machine-readable evidence from the YAGO 4.5
experiments. Multi-gigabyte embedding tables, optimizer state, and sampler data
are intentionally excluded from Git.

## Inductive graph autoencoder

- `inductive_optuna_1500.csv`: four equal-budget, validation-only trials on the
  tiny graph's entity-held-out dataset. Objective: contextual GNN relation-range
  sampled-filtered tail MRR, 300 validation queries and up to 50 negatives.
  Full masked relation-count features and relation-range training negatives
  selected LR `0.001`, temperature `0.1`, best validation MRR `0.5269`.
- `inductive_source_audit.json`: audit of the tiny release, which lacks actual
  triple-level temporal metadata. The full archive separately contains
  `yago-meta-facts.ntx`; see the temporal pipeline documentation.

The selected run is extended separately from the short-budget search. Full graph
experiments retain all parsed structural entity-to-entity facts without edge caps.
No inductive test metrics are used by the search.

`inductive_tiny_10k/` contains actual checkpoint-backed results, learning curves,
contextual/shallow t-SNE, raw sampled rankings, and the complete recipe for the
10k range-only extension. Its hard-candidate gain is real, but global candidate
calibration fails (also confirmed on validation). This limitation is retained
explicitly; a separate mixed-negative run addresses both candidate policies.

`inductive_tiny_mixed_10k/` contains the balanced-validation mixed-negative run:
inductive test MRR `0.7724` (uniform) / `0.5378` (relation-range). The no-message
ablation gives hard-candidate MRR `0.4471`; paired message-passing gain is
`+0.0907`, 95% bootstrap interval `[0.0695, 0.1121]`. The final shallow tables
reach known-entity validation MRR `0.3303`. All use up to 50 filtered negatives.

`inductive_sparse_cpu_control.json` verifies that the full-scale CPU optimizer
also learns: 32-dimensional shallow ComplEx with native sparse Adagrad reaches
known-entity validation MRR `0.2262` after 2,000 steps, versus random MRR about
`0.0886`. This is a small-graph optimizer control, not a full-graph result.

## Corrected masked KGE

- `masked_negative_control.csv`: equal-budget GestaltDB-engine versus hybrid
  negative sampling.
- `optuna_masked_hybrid_1k.csv`: four persistent Optuna trials.
- `masked_model_ablation_1k.csv`: shallow, GNN-only, and GNN-plus-auxiliary
  comparison.
- `masked_20k_metadata.json`, `masked_20k_curves.jsonl`, and
  `masked_20k_relation_retrieval.json`: retained corrected 20k run record.
- `masked_test_evaluation.csv`: uniform and relation-range sampled-filtered test
  protocols.

## Semantic cosine fine-tuning

- `semantic_20k_summary.json` and `semantic_20k_curves.jsonl`: complete
  warm-started 20k run.
- `semantic_20k_metadata.json`: scorer, representation, and recipe metadata.
- `semantic_20k_cosine_metrics.json`: balanced gender, nationality, and type
  cosine 1-NN/P@10 measurements.
- `semantic_30k_directional_summary.json` and
  `semantic_30k_directional_curves.jsonl`: a later continuation that completed
  optimization, but whose final weights were lost during server restart/export.

The 30k continuation is directional evidence only and is not a recoverable or
deployable artifact. Do not compare its in-training sampled validation metric to
full-candidate link prediction.

See `../docs/` for protocols, caveats, inductive-task guidance, and the
recommended interpretation of these records.
