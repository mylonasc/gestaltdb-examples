# GestaltDB Sampling and Pipeline Performance

## Backend and hot path

Preparation uses GestaltDB `0.12.1` with PyRex/RocksDB
(`pyrex-rocksdb 0.7.1`). RocksDB is used for ingest and serving. Training reads a
RAM/memory-mapped CSR `SamplerSnapshot`; it performs no key-value reads in the
hot loop.

## Context sampling used in the example

The GNN uses edge-seeded two-hop subgraphs:

```text
sample_subgraph(seed_edges, fanouts=[12, 8], direction="any")
```

Exact supervised edges and reverse copies are removed before message passing.

## Negative sampling

### GestaltDB engine negatives

`HardNegativeConfig` supports:

- sources: `random`, `non_visited_relation_neighbors`, and
  `context_relation_neighbors`;
- known-positive rejection;
- same endpoint type and relation endpoint type constraints;
- head/tail corruption probability;
- candidate fanout and retry limits;
- temporal positive policies.

These negatives are high quality but expensive in this workload. In a fixed
batch-128, 16-negative control, 50 engine steps required 379.6 seconds.

### Hybrid negatives

The optimized path separates two ID spaces:

- local subgraph corruptions for contextual GNN states;
- genuinely global entity IDs for the shallow exported-table objective.

The earlier “global” implementation mapped missing global IDs back to random
local nodes, so almost all global draws were actually local. The corrected batch
schema carries global negative IDs separately.

The same 50-step control required 9.3 timed seconds with hybrid negatives and
had no short-run quality disadvantage. Long runs therefore use hybrid negatives.

## Other GestaltDB sampling APIs

GestaltDB also exposes:

- `sample_neighbors`: one-hop fanout, direction, relation filters, replacement,
  and strategies;
- `sample_layers`: GraphSAGE-style layer specifications;
- `sample_multihop`: node-seeded multi-hop expansion;
- `sample_random_walks`: direction, relation filters, metapaths, restart
  probability, and dead-end behavior;
- `sample_nodes` and `sample_edges`: optional weighted sampling;
- `sample_hard_negatives` and `is_positive`;
- temporal windows and causal policies;
- `AsyncBatchFeeder` and external-ID variants.

Sampling choice changes context and negative difficulty; it does not by itself
make an ID-table model inductive.

## Is the GPU starved?

The correct diagnostic compares:

1. feeder-only batches/second;
2. `tf.data` delivered batches/second;
3. compute-only steps/second on one cached batch;
4. end-to-end steps/second on fresh batches.

If end-to-end approximately equals compute-only, the GPU is not waiting for
input. Measured at batch 128, 16 hybrid negatives:

| Measurement | Rate |
|---|---:|
| Single-thread feeder | 13.3 batches/s |
| Delivered with prefetch 4 | 9.6 batches/s |
| Compute-only GPU ceiling | 7.5 steps/s |
| End-to-end | 7.5 steps/s |

Thus the run was compute-bound, with 1.77x feeder headroom. Four feeder workers
raised production to 23.1 batches/s.

At batch 256, GPU compute remained about 7.3 steps/s while one feeder produced
only 6.5 batches/s. Four workers raised it to 13.1 batches/s. The practical
recommendation is **batch 256 plus four workers**, followed by another
compute-versus-end-to-end check. Batch 512 remains unmeasured.

GPU memory use can look surprising because the dominant allocations are the
4.18M x 200 table and Adam optimizer state; activations are comparatively small.
Unused memory alone is not proof that a larger batch will improve throughput.
