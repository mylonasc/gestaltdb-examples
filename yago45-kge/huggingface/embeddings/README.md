---
license: cc-by-sa-3.0
language:
- en
tags:
- knowledge-graph
- embeddings
- graph-neural-network
- yago
size_categories:
- 10M<n<100M
---

# YAGO 4.5 inductive GNN embeddings

A complete static cache of **49,281,921 × 200 float16 GNN entity vectors**
generated using an independently trained inductive YAGO graph autoencoder.

**Repository status:** private release preparation. Documentation and licensing
notices only; the embedding matrix and entity mapping are not uploaded yet.
This is not an official YAGO release and is not endorsed by the YAGO team.

## Planned contents and usage

- Memory-mappable entity-vector matrix: 19,712,768,528 bytes including NPY header
  (about 19.7 GB); all values verified finite.
- Entity-to-row mapping, relation mapping, cache metadata, and checksums.
- Separate evaluation results and reproduction instructions.

These are **200-dimensional GNN vectors**, not the independent 32-dimensional
shallow ComplEx embeddings. Row order must be interpreted using the supplied
mapping, not assumed to match external YAGO IDs or sorted entity names.
Use the matching model's `score_embeddings` interface, casting vectors to
float32, for relation-conditioned scoring. Generic semantic cosine retrieval
has not been established. New entities require fresh neighborhood encoding;
they cannot be looked up in this fixed cache.

Matching model: https://huggingface.co/mylonasc/yago45-inductive-gnn

## Source and transformations

Derived from YAGO 4.5 archive `yago-4.5.0.2.zip`, SHA256
`b226ba9ce82e3c13187c501bca39c766cb243d0ed02c6b5137146370fadb8a8a`.
All Turtle members were streamed without edge/line caps; literals/text excluded.
119,045,262 structural facts were retained before temporal exclusion,
deduplication and independent entity/edge splits. Prepared context contains
96,428,602 training and 21,783,785 inference-support facts, over 63 relations.

The GNN uses masked directional relation-count features, a three-step `tf_gnns`
MPNN (hidden width 128), and a contextual ComplEx decoder. It was trained for
185,111 updates under strict two-pass entity coverage (86,927,656 supervised
anchor draws). The exported checkpoint is the coverage-complete final step.

Before caching, **all held-out queries and same-relation reverse copies were
globally removed** from context and degree features. Vectors were generated
with fanouts 8/4/2 in batches of 2,048 and stored as float16. The cache covers
every mapped entity, including entities with no trained shallow ID row.

## Separate cached-vector evaluation

Sampled-filtered tail prediction, 2,000 inductive test queries, up to 50
negatives, pessimistic model ties. Not full-ranking metrics.

| Candidates | MRR | Hits@1 | Hits@10 | Mean candidates | Frequency MRR (expected ties) |
|---|---:|---:|---:|---:|---:|
| Uniform | 0.977372 | 0.9660 | 0.9980 | 51.000 | 0.967879 |
| Relation-range | 0.887115 | 0.8270 | 0.9845 | 49.528 | 0.779899 |
| Degree-matched | 0.572603 | 0.4225 | 0.8895 | 30.712 | 0.311573 |

Static batched contexts, float16 storage and the globally scrubbed graph differ
from per-query float32 GNN evaluation. These results are separate measurements,
not exact reproduction of the model card's rankings. Degree-matched negatives
control incoming-predicate degree using zero/logarithmic bins; sparse bins
reduce candidate counts, so compare each policy with its own controls.

Strong structural shortcuts limit interpretation of high sampled MRR.
t-SNE plots in the linked notebook are diagnostic, not semantic benchmarks.
The source contains real-world person/entity relations; learned vectors may
encode source errors, societal biases and sensitive associations. They should
not be treated as verified personal attributes or used for consequential
decisions about individuals without appropriate independent review.

## License and attribution

Embeddings and redistributed mappings are released under **Creative Commons
Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0)**, conservatively treating
this release as an adaptation of YAGO 4.5. See [LICENSE](LICENSE) and
[NOTICE.md](NOTICE.md). We do not assert that every learned vector necessarily
constitutes a derivative work; this release preserves upstream conditions.

YAGO 4.5 is licensed under CC BY-SA 3.0 by the **YAGO team of Télécom Paris**.
Source and licensing: https://yago-knowledge.org/downloads/yago-4-5
Archive: https://yago-knowledge.org/data/yago4.5/yago-4.5.0.2.zip

Changes include structural filtering, temporal/entity splits, training,
global query/reverse scrubbing, sampled-context encoding and float16 conversion.
Retain attribution/license notices, identify subsequent modifications, and
distribute adaptations under the same or a permitted compatible ShareAlike
license. No warranties or YAGO endorsement are provided. Independent code and
third-party libraries retain their own licenses.

### Source citation

Fabian M. Suchanek, Mehwish Alam, Thomas Bonald, Lihu Chen, Pierre-Henri Paris,
and Jules Soria. **YAGO 4.5: A Large and Clean Knowledge Base with a Rich
Taxonomy.** SIGIR, 2024. https://suchanek.name/work/publications/sigir-2024.pdf

## Reproduction and results

Code, source manifest, metadata, raw evaluation evidence and an executed
notebook with embedded t-SNE and loss figures:
https://github.com/mylonasc/gestaltdb-examples/tree/main/yago45-kge

Results milestone: Git commit bfc5ad9. Validity dates used in training do not
establish acquisition time; undated static background remains available. No
causal forecasting claim is made.
