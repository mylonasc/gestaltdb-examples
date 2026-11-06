# Full-graph sampling and batch-size investigation

## Finding

The initial live run averaged about 5.21 updates/sec at batch 128 (~667
supervised entity anchors/sec). A GPU snapshot showed 18% utilization and
3,354 MiB whole-device memory use; the process averaged roughly one CPU core.
Real-graph stage timings confirmed a CPU input bottleneck: neighborhood
preparation consumed **172 ms of a 194 ms batch**, versus **15 ms** for the
synchronized model/optimizer update. The shared GNN has only 877,568 trainable
encoder/decoder parameters; allocating more VRAM alone does not fix this.

Training was stopped at a complete optimizer boundary, **step 20,439**, with
**2,616,192 supervised anchor draws**. No partial batch was counted as complete.
All benchmarks use that checkpoint and the same width-128, three-step residual
GNN, training graph, fanouts 8/4/2, eight negatives and CPU shallow optimizer.
Benchmark updates are ephemeral and do not overwrite the training checkpoint.

## Measurements

| Sampler / implementation | Batch | Neighborhood ms | Update ms | Total batch ms | Anchors/sec |
|---|---:|---:|---:|---:|---:|
| Original v2, mmap subclasses | 128 | 172.3 | 14.7 | 193.5 | 661 |
| Original v2, mmap subclasses | 256 | 341.8 | 22.5 | 380.0 | 674 |
| Original v2, isolated process | 512 | 578.5 | 41.2 | 643.6 | 796 |
| v2, zero-copy ndarray CSR views | 128 | 129.3 | 13.2 | 148.6 | 862 |
| v2, zero-copy ndarray CSR views | 512 | 445.6 | 41.5 | 509.0 | 1,006 |
| v2, zero-copy ndarray CSR views | 1,024 | 835.7 | 77.7 | 957.6 | 1,069 |
| v3, vectorized CSR frontiers | 512 | 187.4 | 41.2 | 249.6 | 2,051 |
| v3, vectorized CSR frontiers | 1,024 | 377.6 | 77.6 | 498.5 | 2,054 |
| **Final v3 + packed edge deduplication** | **512** | **113.5** | **46.6** | **180.3** | **2,839** |

This is about **4.3× the initial batch-128 throughput**, measured in anchors/sec,
not updates/sec. Increasing the batch alone helps only modestly. The 1,024
sweep had essentially no advantage over 512 before the final deduplication
optimization, while TensorFlow peak GPU allocation rose from ~3.0 GB to ~5.5 GB.
Batch **512** is the selected conservative size; we do not claim it is a global
optimum or that final-code batch 1,024 was benchmarked.

The final isolated batch-512 measurement uses five warm-up and 50 measured
batches. Means include positive selection, filtered negatives, masked
neighborhoods, localization, tensor conversion and synchronized loss/optimizer
updates. Replaying the same prepared tensors takes about 45 ms/update, confirming
that CPU preparation still limits utilization. Approximate remaining compute
time at this short benchmark rate is 8.3 hours, **before** validation,
checkpointing, exports, interruptions and throughput variation. This is an
estimate, not a completion guarantee.

After resuming the migrated run, a 15-second `nvidia-smi dmon` window averaged
about **28% SM utilization**, with 0–53% bursts. Whole-device framebuffer
allocation was about **12.6 GB**, higher than post-warmup TensorFlow peak
allocation because restore/runtime reservations and other processes are not
represented by that reset allocator counter. We do not claim the GPU is
saturated: measured CPU preparation remains longer than the model update.

## Changes and correctness

1. **Zero-copy ndarray views over mmap CSR arrays.** CPU profiling found repeated
   `np.memmap.__getitem__` calls in the per-node loop. The view retains the mmap
   backing; no multi-gigabyte eager copy is made. Tests verify identical samples
   and RNG state against the original mmap-subclass path.
2. **Vectorized frontier sampling (version 3).** Batched CSR position draws avoid
   the per-node adjacency/allocation loop. Draws are without replacement,
   supervised facts/reverses are rejected before expansion, isolated roots are
   retained, and full masked degree features remain unchanged. Small masked
   adjacencies are randomly trimmed after rejection rather than biased toward
   the first CSR entries. Statistical, masking and duplicate-edge tests cover
   these cases. Version 3 changes the RNG stream and is explicitly versioned.
3. **Packed edge deduplication.** Integer keys replace the structured-row
   `np.unique(axis=0)` sort while preserving lexicographic directional-message
   ordering and guarding against int64 overflow.

The main experiment resumes with batch 512 and sampler v3. A provenance record
stores both recipes, source checkpoint and migration step. The coverage schedule
records the original batch-128 phase and exact next anchor; it does **not**
reinterpret 20,439 old steps as 20,439 batches of 512. Tests concatenate pre/post
migration anchor sequences and verify exact equality to the original sequence.
Optimizer states and learning rates are retained. Larger averaged batches do
change the optimization trajectory and reduce the number of subsequent
optimizer updates; this is not bitwise-equivalent small-batch training.

The target remains **86,927,656 anchor draws**, covering every training entity
at least twice, with held-out entities excluded. The migrated run ends at
**185,111 total updates**, instead of the original batch-128 budget of 679,124.

## Batch-generation parallelism and GestaltDB features

The live run already uses GestaltDB's compact `CSRAdjacency`/`build_csr`
snapshot. We measured whether a fuller GestaltDB path or parallel producers
would create batches faster. These tests used the live snapshot at batch 512,
with the same positives, filtered negatives, masked 8/4/2 neighborhoods and
features. They are **generation-only** unless stated otherwise.

| Variant | ms/batch | batches/s |
|---|---:|---:|
| mmap CSR, sequential | 127.7 | 7.83 |
| RAM-resident CSR, sequential | 126.2 | 7.93 |
| Sequential + synthetic 46 ms GPU update | 173.6 | 5.76 |
| Thread prefetch=2 + 46 ms consumer | 137.2 | 7.29 |
| Thread prefetch=4 + 46 ms consumer | 131.3 | 7.62 |
| 2 forked producer processes | 67.0 | 14.93 |
| 4 forked producer processes | 39.6 | 25.28 |
| 8 forked producer processes | 26.6 | 37.65 |

**Loading the CSR into RAM is not the answer.** Resident and mmap arrays are
within noise because the 2.2 GB snapshot is already in the page cache. The cost
is CPU frontier construction, deduplication, masking and feature assembly, not
disk access.

**GestaltDB's own `SamplerEngine.sample_subgraph`/`sample_layers` is not faster
for this recipe.** It iterates rows in Python and materializes sets of global
node/edge IDs; it also does not implement the custom supervised-positive masking
used here. The useful primitives are the array-native snapshot, `AsyncBatchFeeder`,
`worker_engine` and deterministic `split_seed`.

**What does help is overlap and process parallelism.** `AsyncBatchFeeder` is a
bounded background producer thread: it reduces a 173.6 ms sequential
prepare+update cycle to 137.2 ms (1.27×). Forked producer processes scale
generation to 14.9/25.3/37.7 batches/s with 2/4/8 workers. Since the synchronized
model update is 46 ms, batch 512 has a **21.7 updates/s ceiling**; four workers
already exceed it, and eight overproduce. A prepared batch pickles to 31.4 MiB
(11.1 ms dump, 1.8 ms load), so a real multi-process trainer needs shared
memory or parent-side feature assembly. With that overhead controlled, the
projected end-to-end rate is roughly **15–18 updates/s**, not the 25/s
generation rate.

## Applied to the running experiment (step 44,631)

The live run was stopped at complete step **44,631** (clean SIGTERM save) and
resumed with 4 forked shared-memory producers / 8 slots
(`performance_migration_44631.json`). Coverage phases and the anchor sequence
are byte-identical to the pre-migration schedule; only the incident-fact /
negative RNG stream changed from sequential PCG to deterministic per-step
`SeedSequence([seed, 0x9E3779B1, step])`, recorded as `batch_rng:
per-step-seed-v1` in `run.json`. Coverage at the boundary is continuous:
15,002,496 supervised anchor draws, matching the old schedule exactly.

Measured end-to-end rate after resume ramped **5.2 → 18.0 updates/s**
(cumulative since resume, still climbing toward steady state), matching the
15–18 projection. Throughput flags live in
`scripts/train_inductive_entity2.sh`; profiling commands must explicitly pass
`--prefetch-workers 0` since prefetch and `--profile-batches` are mutually
exclusive.

GestaltDB exposes no CUDA/GPU-resident sampler; its Torch/PyG/DGL support is
output conversion only. Keeping the GPU busier therefore means feeding its CPU
preparation from multiple workers, not moving graph sampling onto the GPU.

Evidence: [scaling benchmark](inductive_entity2_scaling/benchmark.json),
[raw output](inductive_entity2_scaling/output.log) and
[script](inductive_entity2_scaling/bench_batch_scaling.py).

## Evidence and benchmark caveats

- [Initial 128/256 timings](inductive_entity2_profile_baseline/benchmark.json)
- [Isolated original batch 512](inductive_entity2_profile_baseline512/benchmark.json)
- [Original CPU profile](inductive_entity2_profile_baseline512/sampling_profile.txt)
- [Zero-copy CSR sweep](inductive_entity2_profile_csr_views/benchmark.json)
- [Vectorized sweep](inductive_entity2_profile_vectorized/benchmark.json)
- [Final isolated batch-512 benchmark](inductive_entity2_profile_optimized512/benchmark.json)
- [Final CPU profile](inductive_entity2_profile_optimized512/sampling_profile.txt)

The first sweep was killed by the host OOM killer during repeated restoration
of the giant CPU tables, after persisting the 128/256 measurements. This was a
benchmark harness failure, not a training-checkpoint loss or GPU OOM. Profiling
now restores once per process. The isolated 512 run and later complete sweeps
succeeded. Multi-size sweeps retain ephemeral in-memory optimizer updates
between sizes; their losses are **not** model-quality comparisons. Hardware
allocator peaks exclude display/other-process GPU memory. No held-out test
metrics were used to select throughput settings. The scaling test ran while the
real trainer was active, so its wall-clock rates include normal CPU contention;
fork-pool numbers exclude IPC and TensorFlow tensor conversion.

Resume the migrated experiment with:

```bash
bash scripts/train_inductive_entity2.sh --resume
```

Profile its current checkpoint without advancing training (pause the real
trainer first to avoid resource contention):

```bash
bash scripts/train_inductive_entity2.sh --resume \
  --profile-batches 512 --profile-warmup 5 --profile-updates 50 \
  --profile-output /path/to/new-profile
```
