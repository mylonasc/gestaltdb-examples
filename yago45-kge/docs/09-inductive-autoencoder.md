# Inductive graph autoencoder

The `yago.inductive_*` pipeline is separate from the historical ID-input GNN.
It learns **two useful representations**:

1. A transferable relational GNN and contextual ComplEx decoder for GNN link
   prediction, including entities that had no training edges or ID-table rows.
2. Shallow entity/relation ComplEx embeddings for fast known-entity retrieval.

## Prepare, train, resume, evaluate

Run from `yago45-kge/`, using the project's Python environment:

```bash
# First prepare YAGO normally, if not already done.
# bash scripts/prepare.sh ...

# Create a NEW dataset, without changing existing splits/artifacts.
python -m yago.inductive_prepare \
  --data-root ../data/yago45_facts \
  --output ../data/yago45_facts/inductive.npz \
  --unseen-fraction 0.1 --valid-size 10000 --test-size 10000

python -m yago.inductive_train \
  --dataset ../data/yago45_facts/inductive.npz \
  --run-dir ../data/yago45_facts/inductive_run --steps 20000

# Use the same recipe; --steps means total completed steps, not additional steps.
python -m yago.inductive_train \
  --dataset ../data/yago45_facts/inductive.npz \
  --run-dir ../data/yago45_facts/inductive_run --steps 20000 --resume

# Restore and actually execute the GNN, not the shallow embedding tables.
python -m yago.inductive_eval \
  --run-dir ../data/yago45_facts/inductive_run \
  --split test --triples 2000 --negatives 50
```

Use the environment's Python executable (for example `./.venv/bin/python`) in
place of `python` if necessary. These modules require TensorFlow, NumPy, and
Polars for preparation. Training now defaults to the latest published
`tf_gnns` MPNN encoder and GestaltDB CSR snapshots. No live database connection
is needed after snapshot creation. `--encoder relational --sampler numpy` keeps
the small reference implementation available. CPU execution is supported.
Use `bash scripts/inductive.sh train ... --require-gpu` for the shared CUDA
library environment and an explicit check that TensorFlow can see the GPU.

## Encoder and training objective

The encoder has **no entity-ID input**. Its features are `log1p` directional
relation counts from the masked observed context plus a shared constant for
the generic `Entity` type. Existing YAGO preparation assigns every node that
generic type; this pipeline does not claim to have richer type/text features.

The default `tf_gnns.GraphNetMPNN_MLP` receives projected structural features and
direction-specific relation embeddings, with residual mean-aggregated message
passing. The reference encoder concatenates sender and relation vectors with
separate self transforms. The default two-step encoder produces normalized
200-dimensional vectors; the contextual ComplEx relation table scores links.
Explicit inverse relation channels distinguish incoming and outgoing facts.

For every training batch, seed facts and their same-relation reverse copies
are removed **before neighborhood expansion and feature computation**. This is
a deterministic masked graph autoencoder, not a variational autoencoder.

```text
L = InfoNCE(contextual GNN ComplEx scores)
    + shallow_weight * InfoNCE(shallow ID-table ComplEx scores)
```

Both objectives use filtered tail corruptions among training entities.
`--negative-policy` supports global uniform, relation-range and 50/50 mixed
corruptions. Dense/singleton ranges fall back to global negatives when needed.
`--feature-mode full` (default) uses sparse, full-context relation histograms,
subtracting supervised facts and reverse copies before computing features.
This preserves degree information beyond the sampled fanout. `sampled` retains
the original sampled-context feature ablation.
The shallow branch is deliberately independent: ID vectors cannot become a
shortcut in the inductive encoder. The encoder/decoder train jointly, while
the shallow tables receive their own direct link-prediction supervision.

## Split and inference protocol

- By default, input facts come only from the original prepared **training**
  split. The historical validation/test facts are not recycled into support.
  `--from-shards` instead creates a new independent, uncapped split directly
  from all parsed source facts, including a compact external-ID Parquet sidecar.
- A seeded entity subset is held out. **Every** incident fact is excluded from
  training positives, training negatives and training GNN context.
- Among the remaining entities, validation queries are withheld from context.
  These transductive validation metrics select/monitor a recipe without using
  the unseen-entity test graph.
- Held-out incident facts become inference-only support and inductive test
  queries. Query endpoints retain support edges. Unknown training relations
  are excluded: this supports unseen entities, not unseen relations.
- At test time the encoder observes `train + support`, never validation/test
  query edges. Exact query/reverse copies are additionally masked while encoding.
  Candidates, including negatives, receive the same feature/context treatment.
- Test queries each touch an unseen entity. Support may connect unseen entities
  to seen ones: this is **entity-held-out few-shot inference**, not a benchmark
  with two entirely disconnected, node-disjoint graphs. Support counts vary;
  this is not a controlled fixed-K support experiment.
- Metrics are **sampled-filtered tail-prediction** MRR/Hits, not full-candidate
  rankings. Candidate pools contain entities present in inference context; other
  known true tails are filtered. No head-prediction results are claimed.

Without support, nodes have only the shared constant and are indistinguishable.
Structural features cannot distinguish graph-automorphic entities. Add genuine
text/type/attribute features for stronger zero-shot inference; do not insert ID
vectors into this encoder to paper over that limitation.

## Artifacts and recovery

- `run.json`: architecture, feature recipe, dataset/mapping digests, and settings.
- `checkpoints/`: complete GNN, both ComplEx decoders, shallow entity tables,
  Adam slots, completed step and NumPy sampler RNG state; latest two retained.
- `curves.jsonl`: contextual GNN validation metrics and both training losses.
- `train_losses.jsonl`: frequent loss/throughput records for learning curves.
- `best/`, `best.json`: validation-selected model checkpoint and selection metric.
- `shallow_embeddings.npz`: useful ID embeddings for training entities only,
  matching relation embeddings and external ID maps.
- `artifact_metadata.json`: shallow representation/scorer metadata.
- `gnn_test_evaluation.json`: checkpoint-backed contextual GNN evaluation.

Checkpoints are saved every `--checkpoint-every` steps (default 1000), at final
training completion **before** evaluation/export, and on `KeyboardInterrupt`.
A hard kill resumes from the latest completed periodic checkpoint. Resume checks
the dataset and training recipe, restores optimizer/RNG/step, and trims metric
records beyond the restored step. An occupied run directory cannot accidentally
be overwritten. An export interrupted after final checkpointing can be rerun
with `--resume` and the same total `--steps` without retraining.

`inductive_eval` restores the best-validation model by default; pass
`--checkpoint latest` to evaluate final weights. `--dataset` may point to a compatible
copy with the same mappings. For application-level unseen graphs, instantiate
`NeighborhoodGraph` using new local node numbering and the **trained relation
vocabulary**, sample neighborhoods, call `model.encode(graph)`, then
`model.contextual_scores(encoded, local_triples)`. No entity-table expansion or
gradient update is needed. Load the trained model as in `inductive_eval.py`.
Contextual vectors can be computed/cached for a chosen support snapshot; unlike
the shallow table, they depend on the observed graph and sampled context.

## Large graph, valid-time completion, and resource controls

The full archive contains `yago-facts.ttl`, `yago-beyond-wikipedia.ttl`, and
`yago-taxonomy.ttl`; parsing all Turtle members is the default. No edge cap is
applied by the inductive preparation path. Labels/literals are still excluded
according to the structural entity-to-entity task definition.

```bash
# Stream the archive; never extract its ~151 GB of Turtle onto disk.
bash scripts/inductive.sh prepare --data-root ../data/yago45_full \
  --output ../data/yago45_full/inductive.npz --from-shards \
  --ttl ../data/yago45/yago-4.5.0.2.zip --parse-only

# Unlike the tiny release, the full archive has triple-level validity metadata.
bash scripts/inductive.sh temporal --source ../data/yago45/yago-4.5.0.2.zip \
  --output ../data/yago45_full/temporal_start_dates.parquet

bash scripts/inductive.sh prepare --data-root ../data/yago45_full \
  --output ../data/yago45_full/inductive.npz --from-shards \
  --temporal-dates ../data/yago45_full/temporal_start_dates.parquet \
  --temporal-train-through 2010 --temporal-valid-through 2018
```

Triple-level `schema:startDate` comes from RDF-star `yago-meta-facts.ntx`, not
from an entity's birth date. The earliest annotated start determines the cohort;
out-of-range/sentinel years and years beyond the 2024 source release are excluded.
All dated post-2010 facts **and their same-relation reverses** are removed before
building training and inductive support. Known-training-entity queries dated
2011–2018 form `temporal_valid`; 2019–2024 form `temporal_test`.
`--validation-split temporal_valid` selects checkpoints using this chronological
completion task; `inductive_eval --split temporal_test` evaluates its held-out
period. Random edge validation and entity-held-out tests remain separate.

**This is valid-time completion with undated static background, not causal
knowledge-acquisition forecasting.** The snapshot does not establish when its
undated facts became available; end-date/current-validity prediction is not
claimed. Fact validity must not be confused with data acquisition time.

Full relation histograms are sparse CSR and positive filtering is lazy CSR-backed,
avoiding per-edge Python sets at full scale. Huge external entity names live in
`*.nodes.parquet`; numeric `node_ids` map through that sidecar. Independent
`--base-dim`, `--embedding-device cpu`, and `--shallow-optimizer sgd
--shallow-learning-rate 0.1` can keep the shallow branch within memory limits
without changing the GNN output width. The default shallow optimizer remains
Adam; sparse SGD avoids its two full-size moment tables.

For very large CPU tables, `--shallow-optimizer sparse-adagrad` uses native
row-sparse resource updates with one accumulator table, coalescing repeated
gathers without allocating full-size per-step gradients. `--sampler gestaltdb-csr`
uses GestaltDB's directional CSR primitives in a compact application cache,
instead of persisting the many external-ID/membership/relation-policy indexes
of a complete `SamplerSnapshot`. This is explicitly a thin CSR adapter, not the
complete GestaltDB snapshot format. `--keep-checkpoints 1` bounds full-state
checkpoint retention when disk is constrained.

## Tuning and reports

```bash
bash scripts/inductive.sh tune --dataset ../data/yago45/inductive.npz \
  --study-dir ../data/yago45/inductive_optuna --trials 4
bash scripts/inductive.sh report --run-dir /path/to/run \
  --output experiments/inductive_run --triples 1000 --negatives 50
```

Optuna stores its study in SQLite and tunes only validation metrics. Reports
restore actual checkpoints and include uniform and harder relation-range test
MRR/Hits, bootstrap intervals, per-relation/support-degree groups, random and
relation-frequency controls, a no-message ablation, both training losses,
learning curves and reproducible contextual/shallow t-SNE plots. t-SNE coloring
describes observed neighborhoods; it is not evidence of held-out semantic
retrieval quality. All ranking metrics are sampled-filtered, not full ranking.

Range-only negatives can lose calibration against global uniform candidates.
Use `--negative-policy mixed --validation-policy both` to train on both and
select checkpoints by the mean of the two validation MRRs. Both component
metrics remain in the curves; do not report the aggregate as a single-policy
ranking metric. Original range-only results and their calibration failure are
retained under `experiments/inductive_tiny_10k/`.

## Tests

```bash
python -m pytest -q tests/test_inductive_data.py tests/test_inductive_pipeline.py
```

Tests cover entity exclusion, masked feature leakage, isolated unseen nodes,
filtered candidates, identity independence, GNN checkpoint restoration, the
preparation CLI, restart equivalence to uninterrupted training, and contextual
inductive CLI evaluation.
