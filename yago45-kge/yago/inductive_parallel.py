"""Forked batch producers writing into a shared-memory ring.

The producer workers inherit the read-only training graph and the coverage
schedule, so they share those pages copy-on-write instead of loading a second
copy of the dataset. Each worker prepares complete mini-batches with a
per-step deterministic RNG and writes arrays into one anonymous shared mmap
slot; the trainer consumes slots in step order.

This module deliberately imports no TensorFlow. Create the pool before TensorFlow
initializes CUDA, then start producing after the checkpoint/step counter has
been restored.
"""
from __future__ import annotations

import json
import mmap
import multiprocessing as mp
import queue
import signal
import time
import traceback
from pathlib import Path

import numpy as np

from .inductive_batch import prepare_batch


BATCH_RNG = "per-step-seed-v1"
_RNG_TAG = 0x9E3779B1


def rng_for_step(seed: int, step: int) -> np.random.Generator:
    """Independent, worker-order-independent RNG for one scheduled batch."""
    return np.random.default_rng(np.random.SeedSequence([int(seed), _RNG_TAG, int(step)]))


def _align(value: int) -> int:
    return (value + 63) & ~63


class SharedBatchBuffers:
    """One anonymous shared mapping holding every batch slot."""

    def __init__(self, *, slots, max_nodes, max_edges, batch_size, negatives, feature_dim):
        self.slots = int(slots)
        self.max_nodes = int(max_nodes)
        self.max_edges = int(max_edges)
        self.batch_size = int(batch_size)
        self.negatives = int(negatives)
        self.feature_dim = int(feature_dim)
        regions = [
            ("features", (self.slots, self.max_nodes, self.feature_dim), np.float32),
            ("senders", (self.slots, self.max_edges), np.int32),
            ("receivers", (self.slots, self.max_edges), np.int32),
            ("relations", (self.slots, self.max_edges), np.int32),
            ("positive_local", (self.slots, self.batch_size, 3), np.int32),
            ("negative_local", (self.slots, self.batch_size, self.negatives, 3), np.int32),
            ("positive_global", (self.slots, self.batch_size, 3), np.int32),
            ("negative_global", (self.slots, self.batch_size, self.negatives, 3), np.int32),
        ]
        offset = 0
        self._specs = {}
        for name, shape, dtype in regions:
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            self._specs[name] = (shape, dtype, offset)
            offset = _align(offset + nbytes)
        self.byte_size = max(offset, 1)
        self.mapping = mmap.mmap(-1, self.byte_size)
        for name, (shape, dtype, start) in self._specs.items():
            setattr(self, name, np.ndarray(shape, dtype=dtype, buffer=self.mapping, offset=start))
        # step, nodes, edges, positives, error flag, reserved; queue operations
        # provide the happens-before ordering for these non-atomic arrays.
        self.meta_raw = mp.Array("q", self.slots * 6, lock=False)
        self.meta = np.frombuffer(self.meta_raw, dtype=np.int64)

    def write(self, slot, step, arrays):
        graph, positive_local, negative_local, positive_global, negative_global = arrays
        nodes = len(graph["features"])
        edges = len(graph["senders"])
        positives = len(positive_local)
        if nodes > self.max_nodes or edges > self.max_edges or positives > self.batch_size:
            raise ValueError(
                f"batch exceeds shared buffers: nodes={nodes}/{self.max_nodes}, "
                f"edges={edges}/{self.max_edges}, positives={positives}/{self.batch_size}"
            )
        self.features[slot, :nodes] = graph["features"]
        self.senders[slot, :edges] = graph["senders"]
        self.receivers[slot, :edges] = graph["receivers"]
        self.relations[slot, :edges] = graph["relations"]
        self.positive_local[slot, :positives] = positive_local
        self.negative_local[slot, :positives] = negative_local
        self.positive_global[slot, :positives] = positive_global
        self.negative_global[slot, :positives] = negative_global
        base = slot * 6
        for index, value in enumerate((step, nodes, edges, positives, 0, 0)):
            self.meta[base + index] = value

    def read(self, slot):
        base = slot * 6
        step, nodes, edges, positives, error, _ = (int(value) for value in self.meta[base:base + 6])
        if error:
            raise RuntimeError("producer marked a shared slot as failed")
        graph = dict(features=self.features[slot, :nodes],
                     senders=self.senders[slot, :edges],
                     receivers=self.receivers[slot, :edges],
                     relations=self.relations[slot, :edges])
        return (step, graph,
                self.positive_local[slot, :positives],
                self.negative_local[slot, :positives],
                self.positive_global[slot, :positives],
                self.negative_global[slot, :positives])

    def close(self):
        self.mapping.close()


def _send(channel, lock, message):
    # Small messages are atomically appended to the pipe; the shared lock
    # serializes concurrent writers so framed error tracebacks cannot
    # interleave. Unlike Queue, there is no background feeder thread whose
    # lifetime is tied to an exiting producer.
    with lock:
        channel.send(message)


def _receive(channel, lock, timeout):
    # Poll and receive must be atomic across consumer processes: two workers
    # polling the same pipe can both see readability, but only one may
    # consume the bytes. The lock is never held while waiting, so producers
    # (which need the same lock to send) cannot deadlock against consumers.
    deadline = time.monotonic() + float(timeout)
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise queue.Empty
        if channel.poll(min(remaining, 0.05)):
            if lock.acquire(block=False):
                try:
                    if channel.poll(0):
                        return channel.recv()
                finally:
                    lock.release()
                # Another consumer won the race; retry until the deadline.


def _producer_loop(*, buffers, free_reader, free_writer, ready_writer,
                   send_lock, stop, start_event, step_value, step_lock,
                   end_step, seed, context, training_known, relation_pools,
                   num_train_nodes, batch_size, negatives, negative_policy, fanouts,
                   schedule):
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    start_event.wait()
    while not stop.is_set():
        try:
            slot = _receive(free_reader, send_lock, 0.5)
        except queue.Empty:
            continue
        with step_lock:
            step = int(step_value.value)
            if step >= end_step:
                _send(free_writer, send_lock, slot)
                break
            step_value.value = step + 1
        try:
            arrays, _ = prepare_batch(
                context, training_known, relation_pools, num_train_nodes,
                rng=rng_for_step(seed, step), batch_size=batch_size,
                negatives=negatives, negative_policy=negative_policy,
                fanouts=fanouts, schedule=schedule, completed_steps=step,
            )
            buffers.write(slot, step, arrays)
            _send(ready_writer, send_lock, (slot, step, None))
        except BaseException:
            _send(free_writer, send_lock, slot)
            _send(ready_writer, send_lock, (slot, step, traceback.format_exc()))
            stop.set()
            break


class BatchProducerPool:
    """Four- (or N-) process producer ring with ordered consumption."""

    def __init__(self, *, context, training_known, relation_pools, num_train_nodes,
                 schedule, batch_size, negatives, negative_policy, fanouts, seed,
                 end_step, workers=4, slots=None, max_nodes=1_048_576, max_edges=2_097_152,
                 consume_timeout=180.0):
        if workers < 1:
            raise ValueError("workers must be positive")
        self.workers = int(workers)
        self.slots = int(slots or max(4, self.workers * 2))
        self.consume_timeout = float(consume_timeout)
        self.batch_size = int(batch_size)
        self.negatives = int(negatives)
        self.negative_policy = negative_policy
        self.fanouts = tuple(fanouts)
        self.seed = int(seed)
        self.context = context
        self.training_known = training_known
        self.relation_pools = relation_pools
        self.num_train_nodes = int(num_train_nodes)
        self.schedule = schedule
        self.end_step = int(end_step)
        self.feature_dim = 2 * int(context.num_relations) + 1
        self.buffers = SharedBatchBuffers(
            slots=self.slots, max_nodes=max_nodes, max_edges=max_edges,
            batch_size=self.batch_size, negatives=self.negatives,
            feature_dim=self.feature_dim,
        )
        self.ctx = mp.get_context("fork")
        # Direct pipes, not Queues: Queue's background feeder thread and its
        # shared buffer do not survive the fork/fire-and-exit producer pattern
        # (prepared batches were silently lost once producers ran ahead and
        # exited). Pipe bytes are kernel-buffered and survive producer exit.
        self.free_reader, self.free_writer = self.ctx.Pipe(duplex=False)
        self.ready_reader, self.ready_writer = self.ctx.Pipe(duplex=False)
        self.send_lock = self.ctx.Lock()
        self.stop = self.ctx.Event()
        self.start_event = self.ctx.Event()
        self.step_value = self.ctx.Value("q", -1)
        self.step_lock = self.ctx.Lock()
        self.pending = {}
        self.processes = []
        self.started = False

    def start(self):
        if self.started:
            return self
        for index in range(self.workers):
            process = self.ctx.Process(
                target=_producer_loop, daemon=True, name=f"batch-producer-{index}",
                kwargs=dict(
                    buffers=self.buffers,
                    free_reader=self.free_reader, free_writer=self.free_writer,
                    ready_writer=self.ready_writer, send_lock=self.send_lock,
                    stop=self.stop,
                    start_event=self.start_event, step_value=self.step_value,
                    step_lock=self.step_lock, end_step=self.end_step, seed=self.seed,
                    context=self.context, training_known=self.training_known,
                    relation_pools=self.relation_pools, num_train_nodes=self.num_train_nodes,
                    batch_size=self.batch_size, negatives=self.negatives,
                    negative_policy=self.negative_policy, fanouts=self.fanouts,
                    schedule=self.schedule,
                ),
            )
            process.start()
            self.processes.append(process)
        # Publish free slots only after the fork, so children inherit exactly
        # the read ends they need and no pipe state is duplicated mid-stream.
        for slot in range(self.slots):
            _send(self.free_writer, self.send_lock, slot)
        self.started = True
        return self

    def begin(self, start_step, end_step=None):
        """Release workers at a restored/known optimizer boundary."""
        if not self.started:
            raise RuntimeError("start the producer pool before begin")
        if end_step is not None and int(end_step) != self.end_step:
            raise ValueError("producer end step cannot change after workers start")
        if not 0 <= start_step <= self.end_step:
            raise ValueError("invalid producer step range")
        with self.step_lock:
            self.step_value.value = int(start_step)
        self.start_event.set()

    def _fail_if_dead(self):
        if self.processes and not any(process.is_alive() for process in self.processes):
            codes = [process.exitcode for process in self.processes]
            raise RuntimeError(f"all batch producers exited unexpectedly: {codes}")

    def get(self, expected_step):
        """Return the slot for ``expected_step``, buffering out-of-order completions."""
        while expected_step not in self.pending:
            try:
                slot, step, error = _receive(self.ready_reader, self.send_lock, self.consume_timeout)
            except queue.Empty:
                # Raise promptly on a hard worker crash; otherwise report the
                # stall. Liveness is only meaningful while still waiting:
                # producers exit normally once every step is claimed, while
                # prepared batches may still sit in the pipe or pending buffer.
                self._fail_if_dead()
                raise TimeoutError(
                    f"no prepared batch after {self.consume_timeout:.0f}s "
                    f"(waiting for step {expected_step}, buffered {sorted(self.pending)})"
                )
            if error is not None:
                self.stop.set()
                raise RuntimeError(f"batch producer failed at completed step {step}:\n{error}")
            if step < expected_step:
                # Only possible after a restart or a stale queued batch; discard.
                _send(self.free_writer, self.send_lock, slot)
                continue
            self.pending[step] = slot
        return self.pending.pop(expected_step)

    def read(self, slot):
        return self.buffers.read(slot)

    def release(self, slot):
        _send(self.free_writer, self.send_lock, slot)

    def close(self):
        self.stop.set()
        # Release workers blocked in begin(); harmless if already set.
        try:
            self.start_event.set()
        except (OSError, ValueError):
            pass
        for process in self.processes:
            process.join(timeout=5)
        for process in self.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        for connection in (self.free_reader, self.free_writer,
                           self.ready_reader, self.ready_writer):
            try:
                connection.close()
            except (OSError, ValueError):
                pass
        self.buffers.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def metadata(self):
        return dict(backend="fork-shared-memory", workers=self.workers, slots=self.slots,
                    max_nodes=self.buffers.max_nodes, max_edges=self.buffers.max_edges,
                    batch_rng=BATCH_RNG, consume_timeout_seconds=self.consume_timeout)
