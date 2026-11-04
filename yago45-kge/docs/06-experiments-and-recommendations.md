# Experiments and Recommended Training Recipe

## What the experiments established

1. **ComplEx beat the DistMult design assumption** for asymmetric YAGO facts.
2. **The shallow auxiliary loss is required** when exporting base tables.
3. **Supervised-edge masking is non-negotiable** for GNN link prediction.
4. **Hybrid negatives are the practical throughput choice** on this hardware.
5. **Uniform 50-negative MRR is optimistic** relative to relation-range ranking.
6. **KGE accuracy does not imply cosine semantic clustering.**
7. **Explicit cosine supervision helps**, but enough entity coverage is needed.
8. **Optuna and isolated run directories prevent undocumented recipe drift.**

## Corrected link-prediction recipe

The four-trial 1k Optuna study selected approximately:

```text
score                 ComplEx
batch size            128
learning rate         5.04e-4
negatives             16 hybrid
self-adversarial temp 1.414
shallow aux weight    1.003
fanouts               12, 8
seed-edge masking     enabled, including reverse
```

The 20k masked run reached best validation MRR `0.2743` at 19k steps. Published
records are in `../experiments/`.

## Throughput-oriented update

Measurements suggest batch 256 with four feeder workers can nearly double
triples/second while remaining GPU-bound. This configuration was benchmarked for
throughput but not subjected to a complete quality run. Re-run a fixed-seed
ablation before replacing the published recipe.

## Semantic-retrieval recipe

For generic cosine neighbors, add balanced supervised contrastive batches:

- positives share selected informative attributes;
- negatives have another value of the same attribute;
- sample attribute values with balancing (sqrt-frequency worked as a compromise);
- retain a ComplEx term to preserve KG completion;
- measure balanced 1-NN and P@10 at every evaluation.

Warm-starting from the validated KGE artifact preserved link prediction better
than training the semantic objective from scratch. A complete 20k fine-tune
improved uniform test MRR to `0.3076`, relation-range MRR to `0.2438`, and
semantic metrics modestly. Compact metrics and curves are published in
`../experiments/`; its large embedding artifact is not.

## Suggested next experiment matrix

Keep split, seed, update budget, candidate policy, and evaluation subsets fixed:

| Experiment | Purpose |
|---|---|
| Shallow ComplEx | Base-table reference |
| Masked GNN only | Context contribution |
| Masked GNN + shallow aux | Combined link prediction |
| Warm-start shallow + semantic cosine | Semantic retrieval tradeoff |
| Feature-based inductive GNN | Unseen-entity support |

For each report link MRR/Hits, semantic 1-NN/P@10, throughput, and peak memory.

## Stopping criteria

Do not stop because a t-SNE plot looks attractive. Require:

- stable or improving held-out link metrics;
- semantic metrics materially above balanced random baselines;
- positive within-minus-between cosine margin;
- no train/context leakage;
- reproducible metadata and retained weights;
- acceptable performance under harder candidate policies.
