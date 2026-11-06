"""Non-persisting real-graph CPU/GPU batch benchmarks using the trainer's update."""
import cProfile
import io
import json
import pstats
from time import perf_counter

import numpy as np
import tensorflow as tf

from .inductive_batch import prepare_batch
from .inductive_schedule import EntityCoverageSchedule


def benchmark_batches(args, spec, context, known, pools, update, checkpoint, checkpoint_path, start):
    """Never save updates. Use one size per process for isolated comparisons."""
    destination = args.profile_output
    destination.mkdir(parents=True, exist_ok=True)
    device = "GPU:0" if tf.config.list_physical_devices("GPU") else None
    report = dict(run_directory=str(args.run_dir), checkpoint=checkpoint_path,
                  model=spec["model"], fanouts=args.fanouts, negatives=args.negatives,
                  original_batch_size=args.batch_size, completed_training_steps=start,
                  sampling_version=context.sampling_version,
                  csr_indptr_type=type(context.snapshot.out.indptr).__name__ if context.snapshot else None,
                  warmup=args.profile_warmup, measured_batches=args.profile_updates,
                  note="Ephemeral optimizer updates; checkpoint restored once by trainer; use one size per process for isolated comparisons; run checkpoint and coverage untouched.",
                  batches=[])
    num_entities = spec["model"]["num_train_nodes"]

    # Persist CPU evidence first, even if a later GPU benchmark exhausts RAM.
    size = args.profile_batches[0]
    rng = np.random.default_rng(args.seed + 882)
    schedule = EntityCoverageSchedule(num_entities, size, args.entity_passes, args.seed) if args.entity_passes else None
    profiler = cProfile.Profile()
    with profiler:
        for i in range(3):
            prepare_batch(context, known, pools, num_entities, rng=rng, batch_size=size,
                          negatives=args.negatives, negative_policy=args.negative_policy,
                          fanouts=args.fanouts, schedule=schedule, completed_steps=i)
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(50)
    (destination / "sampling_profile.txt").write_text(stream.getvalue())
    profiler.dump_stats(str(destination / "sampling_profile.pstats"))

    def tensors(arrays):
        graph, *other = arrays
        return ({k: tf.convert_to_tensor(v) for k, v in graph.items()},
                *(tf.convert_to_tensor(value) for value in other))

    for size in args.profile_batches:
        # Independent benchmark RNG avoids moving the persisted training RNG.
        rng = np.random.default_rng(args.seed + 882)
        schedule = EntityCoverageSchedule(num_entities, size, args.entity_passes, args.seed) if args.entity_passes else None
        # Choose approximately the current traversal position, for all sizes.
        trained_schedule = EntityCoverageSchedule(num_entities, args.batch_size, args.entity_passes,
            args.seed, phases=spec.get("coverage_phases")) if schedule else None
        draw = trained_schedule.draws_at(start) if trained_schedule else start * args.batch_size
        epoch, within = divmod(draw, num_entities)
        first = epoch * schedule.steps_per_pass + within // size if schedule else 0
        rows = []
        known.cache.clear()
        if device:
            tf.config.experimental.reset_memory_stats(device)
        for i in range(args.profile_warmup + args.profile_updates):
            tick = perf_counter()
            arrays, row = prepare_batch(context, known, pools, num_entities, rng=rng,
                batch_size=size, negatives=args.negatives, negative_policy=args.negative_policy,
                fanouts=args.fanouts, schedule=schedule,
                completed_steps=(first + i) % schedule.total_steps if schedule else 0)
            convert_started = perf_counter()
            inputs = tensors(arrays)
            row["tensor_conversion_seconds"] = perf_counter() - convert_started
            update_started = perf_counter()
            losses = update(*inputs)
            # Host synchronization is mandatory for honest GPU timings.
            loss = float(losses[0].numpy())
            row["update_seconds"] = perf_counter() - update_started
            row["total_seconds"] = perf_counter() - tick
            row.update(entities=len(arrays[3]), nodes=len(arrays[0]["features"]),
                       edges=len(arrays[0]["senders"]), loss=loss)
            if i >= args.profile_warmup:
                rows.append(row)
        totals = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
        totals.update(batch_size=size,
                      entities_per_second=sum(row["entities"] for row in rows) / sum(row["total_seconds"] for row in rows),
                      gpu_memory=tf.config.experimental.get_memory_info(device) if device else None)
        # Replay the same tensors to isolate model/optimizer from preparation.
        tick = perf_counter()
        for _ in range(args.profile_updates):
            float(update(*inputs)[0].numpy())
        totals["replay_update_seconds"] = (perf_counter() - tick) / args.profile_updates
        report["batches"].append(totals)
        (destination / "benchmark.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(totals), flush=True)

    print(f"profile written to {destination}; no training checkpoint modified", flush=True)
