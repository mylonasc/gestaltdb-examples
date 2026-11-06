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
metrics were used to select throughput settings.

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
