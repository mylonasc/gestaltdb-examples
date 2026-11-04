# Large-KG Training Notes

This directory collects the practical lessons from training 200-dimensional
embeddings on the YAGO 4.5 facts graph with GestaltDB, TensorFlow, `tf_gnns`,
ComplEx, and an RTX 4090.

The central lesson is that “good embeddings” is not one objective:

1. **Link prediction** asks whether `(subject, relation, object)` receives a high
   relation-conditioned score.
2. **Generic semantic retrieval** asks whether related entities are close under
   ordinary cosine distance.
3. **Inductive inference** asks whether the system can represent entities that
   had no trainable row during training.

A model can perform well on one and poorly on the others. The files below keep
those goals and their evaluation protocols separate.

## Documents

1. [Problem formulation and data](01-problem-and-data.md)
2. [Correctness, splitting, and leakage](02-correctness-and-splits.md)
3. [Models, representations, and objectives](03-models-and-objectives.md)
4. [GestaltDB sampling and pipeline performance](04-sampling-and-performance.md)
5. [Evaluation, nearest neighbors, and visualization](05-evaluation-and-visualization.md)
6. [Experiments and recommended training recipe](06-experiments-and-recommendations.md)
7. [KG completion, updates, and inductive tasks](07-completion-and-inductive-tasks.md)
8. [Reproducibility and operational checklist](08-reproducibility-checklist.md)
9. [Inductive graph autoencoder: prepare, train, resume, and GNN evaluation](09-inductive-autoencoder.md)

## Scope and evidence

Published, reproducible records live in `../experiments/`. Large embedding
matrices are intentionally excluded from Git. Some later semantic follow-up
experiments were run locally; this documentation labels any result whose final
weights were not retained.

Versions used in the measured environment:

- GestaltDB `0.12.1`
- PyRex/RocksDB backend (`pyrex-rocksdb`) `0.7.1`
- `tf_gnns` `0.3.2`
- TensorFlow `2.21`-series environment with pip CUDA libraries
