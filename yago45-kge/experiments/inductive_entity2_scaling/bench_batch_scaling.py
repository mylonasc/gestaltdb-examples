"""Measure batch-generation options against the live YAGO training snapshot.

Read-only: no checkpoints, coverage or run metadata are modified.
"""
import json
import multiprocessing as mp
import pickle
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path("/home/charilaos/Workspace/gestaltdb-examples/yago45-kge")
sys.path.insert(0, str(ROOT))
from yago.inductive_batch import prepare_batch
from yago.inductive_data import GraphKnownObjects, NeighborhoodGraph
from yago.inductive_schedule import EntityCoverageSchedule

RUN = ROOT.parent / "data/yago45_full/inductive_full_entity2_h128_mp3"
DATASET = ROOT.parent / "data/yago45_full/inductive.npz"
BATCH = 512
NEGATIVES = 8
START_STEP = 23_000
MEASURED = 20
WARMUP = 2

print("loading dataset", flush=True)
data = np.load(DATASET, allow_pickle=False)
train = data["train"]
num_nodes = len(data["node_ids"])
num_relations = len(data["relation_ids"])
num_train = int(data["num_train_nodes"])
print(f"train={train.shape} nodes={num_nodes} rel={num_relations}", flush=True)

t0 = time.perf_counter()
context = NeighborhoodGraph(
    train, num_nodes, num_relations,
    snapshot_path=RUN / "snapshots" / "train", feature_mode="full",
    compact=True, sampling_version=3,
)
known = GraphKnownObjects(context)
rel_ids, inverse = np.unique(train[:, 1], return_inverse=True)
pools = {int(r): np.unique(train[inverse == i, 2]) for i, r in enumerate(rel_ids)}
schedule = EntityCoverageSchedule(
    num_train, BATCH, 2, 13,
    phases=json.loads((RUN / "run.json").read_text())["coverage_phases"],
)
print(f"context ready in {time.perf_counter() - t0:.1f}s, "
      f"threads-independent batches at step {START_STEP}", flush=True)


def batch(step):
    rng = np.random.default_rng([13, int(step)])
    arrays, timings = prepare_batch(
        context, known, pools, num_train, rng=rng, batch_size=BATCH,
        negatives=NEGATIVES, negative_policy="mixed", fanouts=(8, 4, 2),
        schedule=schedule, completed_steps=int(step),
    )
    return arrays, timings


def timed(label, fn, count):
    for i in range(WARMUP):
        fn(START_STEP + i)
    totals = []
    started = time.perf_counter()
    for i in range(count):
        tick = time.perf_counter()
        value = fn(START_STEP + WARMUP + i)
        totals.append(time.perf_counter() - tick)
    wall = time.perf_counter() - started
    print(f"{label}: {wall / count * 1000:.1f} ms/batch, "
          f"{count / wall:.2f} batches/s, p50={np.median(totals)*1000:.1f} ms", flush=True)
    return value


timed("mmap CSR, sequential", batch, MEASURED)

# Same arrays, but resident (not mmap subclasses). 2.2 GB on this graph.
mmap_adjacency = {name: getattr(context.snapshot, name) for name in ("out", "in_")}
for name, adj in mmap_adjacency.items():
    setattr(context.snapshot, name, replace(adj, indptr=np.array(adj.indptr),
                                            edge_indices=np.array(adj.edge_indices)))
timed("RAM CSR, sequential", batch, MEASURED)
for name, adj in mmap_adjacency.items():
    setattr(context.snapshot, name, adj)

print("pickle/IPC size of one prepared batch:", flush=True)
value = batch(START_STEP)
tick = time.perf_counter()
payload = pickle.dumps(value[0])
dump_s = time.perf_counter() - tick
tick = time.perf_counter()
pickle.loads(payload)
load_s = time.perf_counter() - tick
print(f"pickle {len(payload)/2**20:.1f} MiB, dump {dump_s*1000:.1f} ms, load {load_s*1000:.1f} ms", flush=True)

# Threaded producer, consumer sleeps like the 46 ms GPU update. GIL-bound
# Python sections cannot overlap, but vectorized NumPy can.
from gestaltdb.sampling import AsyncBatchFeeder


def sequential_with_consumer(count, consumer_seconds):
    started = time.perf_counter()
    for i in range(count):
        batch(START_STEP + WARMUP + i)
        time.sleep(consumer_seconds)
    return time.perf_counter() - started


def threaded_with_consumer(count, consumer_seconds, prefetch):
    counter = {"i": 0}

    def producer():
        step = START_STEP + WARMUP + counter["i"]
        counter["i"] += 1
        return batch(step)

    started = time.perf_counter()
    with AsyncBatchFeeder(producer, max_prefetch=prefetch) as feeder:
        for _ in range(count):
            feeder.get()
            time.sleep(consumer_seconds)
    return time.perf_counter() - started


sleep_s = 0.046
for label, elapsed in (
    ("sequential + 46ms consumer", sequential_with_consumer(MEASURED, sleep_s)),
    ("thread prefetch=2 + 46ms consumer", threaded_with_consumer(MEASURED, sleep_s, 2)),
    ("thread prefetch=4 + 46ms consumer", threaded_with_consumer(MEASURED, sleep_s, 4)),
):
    print(f"{label}: {elapsed/MEASURED*1000:.1f} ms/batch, {MEASURED/elapsed:.2f} batches/s", flush=True)

# Multiprocess generation. fork shares the read-only mmap/CSR arrays.
GLOBAL = dict(context=context, known=known, pools=pools, schedule=schedule, num_train=num_train)


def worker(step):
    tick = time.perf_counter()
    arrays, _ = prepare_batch(
        GLOBAL["context"], GLOBAL["known"], GLOBAL["pools"], GLOBAL["num_train"],
        rng=np.random.default_rng([13, int(step)]), batch_size=BATCH,
        negatives=NEGATIVES, negative_policy="mixed", fanouts=(8, 4, 2),
        schedule=GLOBAL["schedule"], completed_steps=int(step),
    )
    return time.perf_counter() - tick, len(arrays[0]["features"])


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    steps = list(range(START_STEP + 2 * MEASURED, START_STEP + 3 * MEASURED))
    for workers in (1, 2, 4, 8):
        with mp.Pool(workers) as pool:
            for step in steps[:WARMUP]:
                pool.apply(worker, (step,))
            started = time.perf_counter()
            results = list(pool.imap_unordered(worker, steps, chunksize=1))
            wall = time.perf_counter() - started
        print(f"fork pool workers={workers}: {wall/len(steps)*1000:.1f} ms/batch, "
              f"{len(steps)/wall:.2f} batches/s, mean nodes={np.mean([r[1] for r in results]):.0f}",
              flush=True)
