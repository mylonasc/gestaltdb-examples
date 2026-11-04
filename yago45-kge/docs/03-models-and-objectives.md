# Models, Representations, and Objectives

## Base tables versus contextual states

The model contains multiple representations:

```text
entity ID -> entity table [N, 200]              (exported)
relation ID -> relation table [R, 200]          (exported)
entity + type -> projection -> GNN [*, 256]     (contextual)
relation ID -> contextual scorer table [R, 256]
```

The main GNN loss scores contextual 256-dimensional node states. Exporting the
200-dimensional input tables does not automatically export what the GNN scored.
This was the primary representation mismatch in the first implementation.

## Why the shallow auxiliary loss matters

The corrected GNN recipe adds a ComplEx loss directly on the exported entity and
relation tables. This ensures the artifact being validated and served receives
the intended relation-conditioned supervision.

At 1,000 equal-budget steps:

| Variant | Sampled valid MRR on exported tables |
|---|---:|
| Shallow ComplEx | 0.1025 |
| Masked GNN only | 0.0941 |
| Masked GNN + shallow auxiliary | 0.1041 |

The GNN-only objective did not organize the exported tables as effectively.

## ComplEx and cosine answer different questions

ComplEx scores:

\[
f(s,r,o)=\operatorname{Re}\sum_k e_{s,k} r_k \overline{e_{o,k}}
\]

This relation-conditioned bilinear score does not require `cos(e_s, e_o)` to be
large. Two people can share nationality while remaining distant under ordinary
cosine distance.

Therefore:

- use ComplEx/DistMult scores for KG completion;
- use raw cosine only as a diagnostic unless cosine was trained explicitly;
- add supervised contrastive or prototype losses for semantic retrieval.

## Semantic contrastive follow-up

A later experiment, now represented by compact records in `../experiments/`,
added balanced cosine InfoNCE batches for entities that share:

- `schema:gender`;
- `schema:nationality`;
- `rdf:type`.

Negatives came from different values of the same attribute. This improved
balanced cosine 1-NN over the corrected KGE baseline:

| Attribute | KGE baseline | Semantic fine-tune | Random |
|---|---:|---:|---:|
| Gender (2 classes) | 0.521 | 0.593 | 0.500 |
| Nationality (5 classes) | 0.246 | 0.300 | 0.200 |
| Type (10 classes) | 0.177 | 0.202 | 0.100 |

The gains were real but the 20k t-SNE plots remained mixed. The objective was
still improving with more coverage; a subsequent 30k continuation reached
in-training semantic top-1 `0.5345` against 16 hard negatives and positive
margin `+0.0726`, but a server restart occurred during final export, so those
weights were not retained. Treat that continuation as directional evidence, not
a deployable artifact.

## Shallow versus GNN deployment

For link prediction using exported tables, shallow ComplEx is simpler and often
competitive. A GNN is justified when contextual neighborhood information adds
measured value, especially for sparse entities or inductive inference. If the
deployed artifact is the base table, always report base-table metrics separately
from contextual GNN metrics.
