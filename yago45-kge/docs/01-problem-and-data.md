# Problem Formulation and Data

## What was trained

The facts-scale experiment uses YAGO 4.5 entity-to-entity facts:

- 11,987,137 training triples (reported as 12.0M)
- 4,181,515 entities
- 25 relations
- 10,000 validation triples
- 10,000 test triples
- 200-dimensional entity and relation tables

The source parser streams Turtle directly from the YAGO zip. Literals, labels,
descriptions, dates, and other non-entity targets are omitted. Relations are
filtered and capped to keep the experiment tractable and diverse.

That choice matters. Without text and literals, the model can only infer
semantics represented by graph structure. Similar names, biographies, dates, or
descriptions cannot help unless they appear as retained entities and relations.

## Three distinct tasks

### 1. Transductive link prediction

All candidate entities and relations are known during training. The model ranks:

```text
(subject, relation, ?)
(?, relation, object)
(subject, ?, object)
```

ComplEx is appropriate because YAGO relations are often asymmetric.

### 2. Semantic entity retrieval

This task asks for high cosine similarity between entities sharing a useful
attribute such as type, nationality, occupation, or genre. ComplEx does not
imply this geometry; a separate cosine/contrastive objective is required.

### 3. Inductive KG inference

This task introduces entities absent from the embedding table used in training.
The current ID-table architecture is transductive. Supporting unseen entities
requires feature-based encoding or few-shot inference of a new embedding row.

## Data artifacts

Preparation creates:

- train/validation/test Parquet files;
- compact node and relation maps;
- integer edge arrays;
- a GestaltDB RocksDB database for ingest and serving;
- a reusable CSR `SamplerSnapshot` for training.

The training hot path reads the CSR snapshot, not RocksDB. The database backend
therefore does not determine training batch latency after snapshot creation.

## Sparsity implications

Twelve million triples over 4.18 million entities is sparse. Many entities have
few distinguishing edges. This limits both link prediction and generic semantic
retrieval, especially after labels and text are removed. More optimization steps
cannot create information that is absent from the retained graph.
