# YAGO 4.5 Knowledge-Graph Embeddings (gestaltdb + tf_gnns)

High-performance TensorFlow training pipeline that ingests the YAGO 4.5 graph into
[GestaltDB](https://github.com/mylonasc/gestaltdb), serves micro-batches from a
reusable CSR sampler snapshot, encodes them with a compiled
[tf_gnns](https://github.com/mylonasc/tf_gnns) message-passing network, and trains
200-dimensional entity (and relation) embeddings with an in-batch contrastive
(GNN-autoencoder style) loss.

## Result

- `data/yago45/artifacts/yago45_embeddings_200d.npz`:
  - `entity`: `[num_entities, 200]` float32 — one 200-d vector per entity
  - `relation`: `[num_relations, 200]` float32 — one 200-d vector per relation
  - `node_ids`, `relation_ids`: external ID maps (row-aligned with the matrices)
- `data/yago45/artifacts/nn_report.json`: cosine nearest-neighbor sanity check
- `data/yago45/artifacts/checkpoints/`: training checkpoints

This run: official `yago-4.5.0.2-tiny` distribution (23.2M Turtle lines) filtered to
**351,934 entity–entity triples / 299,739 entities / 57 relations**. The pipeline
streams from the `.zip` without extracting it, so the same commands scale to the
full `yago-4.5.0.2.zip` via `--ttl` (see "Scaling to full YAGO").

Two trained artifacts are produced in this repo:
- `data/yago45/artifacts/yago45_embeddings_200d.npz` — tiny-graph run
  (300k entities x 200, 57 relations x 200; 5k steps; loss 2.20 -> ~1.0;
  MRR 0.089 / Hits@10 0.20 @ 50 negatives).
- `data/yago45_facts/artifacts/yago45_embeddings_200d.npz` — facts-scale run on
  `yago-facts.ttl` (all YAGO 4.5 entities with an English Wikipedia page):
  **8.0M triples / 3.64M entities / 27 relations** (stratified per-relation cap
  500k), 8k steps (loss 2.20 -> ~1.2; MRR 0.088 / Hits@10 0.19 @ 50 negatives).
  Entity matrix ~2.9GB.
- `data/yago45_facts/nn_report` note: NN similarities (~0.3-0.36) reflect a
  short 8k-step run (~6% triple coverage); extend `--steps` for tighter clusters.

## Method

- **Model (GNN-autoencoder, DistMult-style decoder):** trainable entity table
  `[N, 200]` + relation table `[R, 200]` (+ small node-type table); linear
  projections to 256-d; 2-step `tf_gnns.GraphNetMPNN_MLP` encoder; triple score
  `sum(h_s * r * h_o)`. The exported 200-d artifacts are the canonical entity /
  relation tables trained end-to-end through the GNN.
- **Loss:** InfoNCE over 1 positive + 8 hard negatives (`temperature=0.1`).
- **Negatives:** GestaltDB `SamplerEngine.sample_subgraph` with
  `HardNegativeConfig(context_relation_neighbors, reject_known_positives,
  same_endpoint_type)` — hard, type-consistent, filtered negatives.
- **Compilation:** application-level `@tf.function(input_signature,
  reduce_retracing=True)` over the whole `train_step` (GNN + loss + optimizer);
  optional `--xla` for `jit_compile=True`. Library internals stay eager/backend-portable.
- **Feeding:** array-backed CSR snapshot (`SamplerSnapshot.from_edge_arrays`,
  `mode="ram"`) — no KV reads in the hot path; `tf.data.Dataset.from_generator`
  with explicit output signature + `.prefetch(4)`.

## Setup

```bash
cd yago45-kge && python3 -m venv .venv
./.venv/bin/pip install "tensorflow[and-cuda]>=2.17" tf_keras tensorflow_probability \
  keras polars pyarrow numpy tqdm "pyrex-rocksdb>=0.4.1" lmdb plyvel msgpack
pip install -r requirements.txt  # or: pip install -e <tf_gnns checkout> -e <gestaltdb checkout>
```

`scripts/*.sh` set `KERAS_BACKEND=tensorflow` and extend `LD_LIBRARY_PATH` with the
pip `nvidia` CUDA libs so TensorFlow sees the GPU (same trick as
`gestaltdb/example/gnn_kg/run_training.sh`).

## Pipeline

```bash
# 1. Parse YAGO Turtle -> parquet shards -> maps/arrays -> GestaltDB ingest -> CSR snapshot
bash yago45-kge/scripts/prepare.sh --reset-db
# 2. Smoke test (10 steps, tiny model)
bash yago45-kge/scripts/train.sh --smoke-test
# 3. Full training (default: 5000 steps, batch 64, 8 hard negatives)
bash yago45-kge/scripts/train.sh --steps 5000 --batch-size 64
# 4. Link-prediction sanity (MRR / Hits@k, DistMult on exported tables)
cd yago45-kge && ./.venv/bin/python -m yago.eval --triples 2000 --negatives 50
# 5. Serve demo: snapshot CSR neighborhood + embedding nearest neighbors
cd yago45-kge && ./.venv/bin/python -m yago.serve --k 5
```

Useful flags: `prepare.sh --ttl <file.zip|file.ttl> --limit-lines N --min-relation-edges K
--target-edges M --skip-ingest --skip-parse`; `train.sh --xla --learning-rate 3e-4
--allow-cpu`.

## Performance (RTX 4090, this repo)

| Stage | Throughput |
|---|---|
| Turtle parse (23.2M lines, streamed from zip) | ~350k entity–entity triples kept |
| GestaltDB RocksDB ingest (300k nodes + 352k edges) | ~1.5 s (~430k records/s) |
| Sampler (batch 64, 2-hop 12/8, 8 hard negs) | ~2.3 batches/s, ~1.6k-node subgraphs |
| End-to-end training (compiled `train_step`) | ~2.2 steps/s (~140 triples/s positive throughput) |
| Smoke (batch 8, 1 MP step) | ~60 steps/s |

Feed-side is sampler-bound (expected: 2-hop expansion + filtered hard negatives
per batch). Knobs: smaller fanouts, fewer negatives, larger prefetch, and (code-ready)
thread-pooled prefetch around `sample_batch`.

## Layout

- `yago/config.py` — all hyper/config (embedding dims: entity 200 / relation 200)
- `yago/parse.py` — streaming Turtle -> entity-entity parquet shards
- `yago/prepare.py` + `yago/prepare_cli.py` — maps, arrays, GestaltDB ingest, snapshot
- `yago/sampler.py` — `YagoSampler` (CSR + hard negatives) -> `tf.data` batch dicts
- `yago/model.py` — `YagoKGEModel` (tf_gnns MPNN + DistMult scorer)
- `yago/losses.py`, `yago/train.py`, `yago/eval.py`, `yago/export.py`
- `scripts/prepare.sh`, `scripts/train.sh` — GPU-env entry points
- `data/yago45/` — downloads, shards, artifacts, DB, snapshot
- `PLAN.md` — build log

## Scaling to full YAGO 4.5

```bash
curl -L -o data/yago45/yago-4.5.0.2.zip https://yago-knowledge.org/data/yago4.5/yago-4.5.0.2.zip
# facts = entities with an EN Wikipedia page (23GB Turtle, streamed from zip)
bash yago45-kge/scripts/prepare.sh --ttl data/yago45/yago-4.5.0.2.zip --ttl-member yago-facts.ttl \
  --data-root data/yago45_facts --reset-db \
  --min-relation-edges 50 --max-edges-per-relation 500000 --target-edges 8000000
bash yago45-kge/scripts/train.sh --data-root data/yago45_facts --steps 8000 --batch-size 64 \
  --hop1-fanout 8 --hop2-fanout 4
```

The parser streams from the zip (no extraction). For 10s of millions of triples,
raise `--min-relation-edges`, keep `--target-edges` unset, and consider a machine
with >64GB RAM for the CSR snapshot + a multi-GB embeddings artifact
(49M entities x 200 x 4B ≈ 39GB uncompressed at full scale).
