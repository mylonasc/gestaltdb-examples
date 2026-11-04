# Evaluation, Nearest Neighbors, and Visualization

## Link-prediction evaluation

Always report:

- split and whether entities were known during training;
- score function loaded from artifact metadata;
- candidate policy;
- number of candidates;
- filtering policy;
- per-relation and degree-stratified metrics where possible.

The corrected 20k masked artifact produced:

| Candidate policy, 50 negatives | MRR | Hits@1 | Hits@10 |
|---|---:|---:|---:|
| Uniform sampled-filtered | 0.2770 | 0.2005 | 0.4080 |
| Relation-range sampled-filtered | 0.2198 | 0.1330 | 0.3765 |

The harder relation-range result is more informative. Full-candidate ranking or
large type-compatible pools are preferable when computationally feasible.

## Correct retrieval example

The primary example for a KGE model is:

```text
given (subject, relation, ?), rank plausible object candidates
```

Use relation-range/type-compatible candidates and remove other known positives.
Do not present arbitrary entity cosine neighbors as the model’s link-prediction
output.

## Quantifying semantic geometry

t-SNE is not an accuracy metric. For balanced attribute classes, report:

- cosine 1-nearest-neighbor accuracy;
- cosine precision@10;
- mean within-class and between-class cosine;
- cosine margin;
- random-class baseline.

Plots can then illustrate, rather than substitute for, those measurements.

## t-SNE cautions

t-SNE:

- preserves local neighborhoods, not global distances;
- can create apparent islands from continuous data;
- changes with seed, perplexity, metric, and sampling;
- should use the same balanced subset and settings when comparing models;
- should use cosine distance when evaluating cosine-trained embeddings.

The corrected KGE-only plots were heavily mixed. The 20k semantic fine-tune
improved balanced retrieval metrics but still did not create clean class islands.
That is a valid negative finding: relation-conditioned accuracy improved before
generic semantic geometry became visually compelling.

## Avoid cherry-picking

Retrieval examples should be deterministic first-held-out rows, random seeded
rows, or explicitly labeled best/median/worst cases. Aggregate metrics must be
shown beside examples. A few attractive queries are not evidence of general
quality.

## Relation and entity norms

Do not normalize ComplEx scores unless normalization was part of training; vector
norms can carry signal. Normalize only for the separate cosine-retrieval metric.
