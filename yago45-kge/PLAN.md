# YAGO 4.5 KGE Pipeline Plan (gestalt-example)

## Goal
Train a knowledge-graph embedding model for YAGO 4.5 with:
- `gestaltdb` for ingestion + serving (RocksDB/PyRex + SamplerSnapshot/SamplerEngine)
- `tf_gnns` for GNN encoder, compiled with `tf.function` (+XLA option)
- Output: 200-dim entity embeddings (+ relation embeddings) for all entities in dataset

## Constraints
- All work inside `/home/charilaos/Workspace/gestalt-example` (no /tmp).
- GPU: RTX 4090 24GB, RAM 62GB, disk free ~143GB.
- YAGO 4.5 full = 49M entities, 109M facts, 12.3GB zip. Full entity-entity subset still 10s of millions.
  Strategy: streaming Turtle parser, keep entity-entity triples only, support full ingest but default to tractable subset if needed. Tiny zip (200MB) for smoke.

## Architecture
1. `yago/` library package:
   - `config.py`: paths, subset, ingestion, sampler, model (200-d), training configs
   - `download.py`: fetch yago-4.5.0.2.zip / tiny, verify, unzip list
   - `parse.py`: streaming Turtle parser -> parquet shards (src, dst, rel entity-entity only)
   - `prepare.py`: select relations/entities, build node_map/relation_map/edge_arrays.npz, ingest gestaltdb, build snapshot
   - `sampler.py`: SamplerEngine wrapper -> SampledBatch (2-hop + hard negatives)
   - `batch.py`: sampled_batch_to_tensor_dict + output_signature (tf.data)
   - `model.py`: entity/relation 200-d tables, GraphNetMPNN_MLP encoder, ComplEx scorer
   - `losses.py`: InfoNCE contrastive
   - `train.py`: tf.data.from_generator + prefetch + compiled train_step, export embeddings.npz
   - `export.py`: named base-table export + scorer metadata + cosine diagnostic
   - `eval.py`: sampled-filtered MRR/Hits@k on held-out triples
2. Scripts: `scripts/train.sh`, `scripts/prepare.sh`, persistent Optuna workflow
3. Data layout: `data/yago45/` (downloads, parquet, artifacts, gestaltdb_rocksdb, sampler_snapshot)
4. Venv: `./.venv` (TF 2.21 + CUDA via pip nvidia, gestaltdb[fast-ingest], tf_gnns from local path)

## Model choice
- GNN-autoencoder: shallow 200-d tables -> projection -> masked 2-step GraphNetMPNN_MLP -> ComplEx scorer.
- Loss: contextual InfoNCE plus shallow ComplEx InfoNCE directly on exported tables.
- Data: SamplerEngine.sample_subgraph 2-hop neighborhoods + hard negatives (reject known positives, same-type).
- Compile: `@tf.function(input_signature, jit_compile=FLAG, reduce_retracing=True)` at app level (never inside tf_gnns).
- Feed: tf.data.Dataset.from_generator(pre-fetch) with explicit output_signature; sampler is array-backed CSR (no KV reads in hot path).

## Status (2026-11-04)
- [x] Inspect gestaltdb/tf_gnns reference (example/gnn_kg)
- [x] Create venv + install deps (TF 2.21 CUDA, gestaltdb, tf_gnns editable)
- [x] Download tiny + validate Turtle parser (23.2M lines -> 352k entity-entity)
- [x] Implement yago/ library + scripts
- [x] Smoke test (10 steps, ~60 steps/s, 200-d export verified)
- [x] GestaltDB ingest (300k nodes + 352k edges in 1.5s) + CSR snapshot
- [x] Full training tiny (5000 steps, batch 64, ~2.2 steps/s, loss 2.20 -> ~1.0)
- [x] Eval tiny: MRR 0.089 / Hits@10 0.20 (50 random negatives, DistMult on tables)
- [x] Full YAGO zip downloaded (12.4GB); facts-scale parse (yago-facts.ttl, 23GB) — running in data/yago45_facts
- [x] Facts prepare: 8.0M triples / 3.64M entities / 27 relations (per-relation cap 500k), gestaltdb ingest 33s, snapshot built
- [x] Facts training (8000 steps, batch 64, fanouts 8/4, ~1 step/s, loss 2.20 -> ~1.2) — done
- [x] Eval facts: MRR 0.088 / Hits@10 0.19; serve demo works on both graphs
- [x] Verified: 200-d float32 entity+relation matrices, finite, id-aligned; gestaltdb serves Cypher + CSR sampling
- [x] README + scripts + yago/ library complete
- [x] Pushed as yago45-kge to mylonasc/gestaltdb-examples (scripts + docs)
- [x] Added notes/visualization notebook (t-SNE subset, relation heatmap, NN checks)
- [x] Facts v2: 12.0M train / 10k valid / 10k test, 25 rels, 4.18M nodes; snapshot OK
- [x] Fast numpy negatives (51x: 1642ms -> 32ms/batch); threads dropped (GIL-bound)
- [x] LR 3e-4 + 16 negs (sparse base-table updates); train/val curves JSONL+PNG
- [x] v4 (complex+adv+aux1.0, bs128) trained to 150k steps
- [x] v4 completed at 150k: historical sampled valid MRR 0.473, test MRR 0.467 (50 negatives)
- [x] Audit fixes: mask supervised/reverse message edges; repair known-positive filtering; align scorer defaults and persist metadata
- [x] Replace primary generic cosine example with relation-conditioned ComplEx retrieval; retain cosine as diagnostic
- [x] Split hybrid negatives into local contextual corruptions and true-global shallow-table corruptions
- [x] Add regression tests and Optuna SQLite/run-directory workflow
- [x] Controlled 50-step masked comparison: engine MRR 0.0816 in 7m; hybrid MRR 0.0839 in 51s overall (9.3s timed loop)
- [x] Four-trial hybrid Optuna tuning @1k: best masked valid MRR 0.1025 (LR 5.04e-4, bs128, 16 neg, adv 1.414, aux 1.003)
- [x] Fixed-budget @1k: shallow 0.1025, masked GNN-only 0.0941, masked GNN+aux 0.1041 MRR
- [x] 20k best-recipe masked run: best valid MRR 0.2743; test uniform 0.2770; relation-range 0.2198
- [x] Corrected notebook executed (28 cells, 10 figures, no errors); final code/results ready to publish

## Notes
- TF GPU needs LD_LIBRARY_PATH from pip nvidia libs (see gestaltdb example run_training.sh).
- Turtle parsing must stream (full facts file >> RAM). Use regex per line, skip literals/labels.
- Keep compact int64 IDs in hot path; map to external only at boundaries.
