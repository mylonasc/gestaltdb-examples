# KG Completion, Updates, and Inductive Tasks

The limitations below describe the historical ID-input model. A separate,
implemented ID-independent pipeline is now available; see
[Inductive graph autoencoder](09-inductive-autoencoder.md) for training and
checkpoint-backed GNN link prediction on unseen entities.

## KG completion with existing entities

The exported ComplEx tables support link prediction immediately:

```text
(subject, relation, ?)  rank objects
(?, relation, object)   rank subjects
(subject, ?, object)    rank known relations
```

Rank relation-range or type-compatible candidates, remove other known positives,
and return top-k scores. Accepted predictions can be written to GestaltDB.

Adding an edge between existing entities does not require creating a new table
row. However, the embedding artifact and CSR snapshot do not automatically
change. Periodically rebuild the snapshot and fine-tune so new evidence affects
future predictions.

## Why new entities do not work automatically

The current entity encoder begins with a trainable row indexed by global entity
ID. A new entity has no row. All retained nodes also use the same generic entity
type, so the GNN has almost no feature signal independent of identity.

Consequently, the current model is transductive. Message passing alone does not
make it inductive.

## Option 1: few-shot new-entity inference

When a new entity has several observed links:

1. append a new 200-dimensional row;
2. initialize it from neighboring embeddings or a small random vector;
3. freeze existing entity and relation tables;
4. optimize only the new row against observed triples and negatives;
5. regularize toward the initialization;
6. evaluate with 1, 3, 5, and 10 observed links.

This is the simplest practical online extension. A neighbor-mean initialization
is usually better than random. Relation-transformed neighbor estimates can be
better still.

## Option 2: periodic warm-start retraining

For batches of graph updates:

1. ingest new nodes and edges;
2. regenerate maps/edge arrays and the CSR snapshot;
3. copy old embedding rows into an expanded table;
4. initialize only new rows;
5. train on a mixture of old and new facts to avoid forgetting;
6. export a new versioned artifact.

This remains transductive but is operationally straightforward.

## Option 3: genuinely inductive GNN

Provide features available for unseen entities:

- label/description text embeddings;
- node types and structured attributes;
- relation histograms and degrees;
- numeric/date features;
- sampled neighborhoods.

Then train:

```text
features + neighborhood -> relation-aware GNN -> 200-d embedding
```

Known entities may retain an optional ID embedding, but the model must learn to
work without it (ID dropout or a learned unknown-entity vector). Persist the
complete feature encoder, GNN, relation scorer, and preprocessing metadata—not
only the base tables.

## Correct inductive evaluation

An edge split does not test inductivity. Construct node-disjoint or temporal
splits:

- remove all training edges for held-out entities;
- reveal a controlled number of support edges at inference;
- evaluate remaining query edges;
- separate zero-shot and few-shot results;
- ensure text/features do not leak target relations.

Report performance by support-edge count and entity degree.

## New relations

A new relation also lacks a table row. With examples, initialize and optimize a
new relation embedding while freezing the rest. For zero-shot relations, encode
relation names/descriptions or use a compositional relation model. Sampling
alternatives cannot solve a missing relation representation by themselves.

## Role of GestaltDB

GestaltDB can ingest updates and sample neighborhoods for new nodes. For
inductive serving, the application should:

1. ingest the new node and support edges;
2. expose them in a refreshed/current sampler snapshot;
3. sample the new neighborhood;
4. run the feature encoder and GNN;
5. rank relation-compatible candidates;
6. optionally fine-tune a new row.
