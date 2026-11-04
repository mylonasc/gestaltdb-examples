# Experiment Records

This directory contains compact, machine-readable evidence from the YAGO 4.5
experiments. Multi-gigabyte embedding tables, optimizer state, and sampler data
are intentionally excluded from Git.

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
