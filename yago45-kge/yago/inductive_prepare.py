"""Create an independent inductive dataset from existing prepared YAGO facts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from .config import PathsConfig
from .inductive_data import split_entities


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--unseen-fraction", type=float, default=0.1)
    parser.add_argument("--valid-size", type=int, default=10000)
    parser.add_argument("--test-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--from-shards", action="store_true",
                        help="Use all parsed entity-entity shards without a relation/edge cap")
    parser.add_argument("--ttl", type=Path, help="Parse a full .zip/.ttl source first (use with --from-shards)")
    parser.add_argument("--ttl-member", help="Optional explicit archive member; default parses all Turtle members")
    parser.add_argument("--parse-only", action="store_true", help="Stop after writing uncapped Parquet shards")
    parser.add_argument("--temporal-dates", type=Path, help="Triple start-year Parquet from inductive_temporal")
    parser.add_argument("--temporal-train-through", type=int, default=2010)
    parser.add_argument("--temporal-valid-through", type=int, default=2018)
    args = parser.parse_args()
    if args.output.suffix != ".npz":
        parser.error("--output must end in .npz")
    if args.valid_size <= 0 or args.test_size <= 0:
        parser.error("--valid-size and --test-size must be positive")
    if args.output.exists():
        parser.error("output already exists; choose a new dataset path")
    paths = PathsConfig(data_root=args.data_root)
    if args.ttl:
        if not args.from_shards:
            parser.error("--ttl requires --from-shards")
        from .config import make_config
        from .parse import parse_yago_ttl_to_shards
        cfg = make_config()
        cfg.paths = paths
        parse_yago_ttl_to_shards(cfg, args.ttl, member=args.ttl_member)
        if args.parse_only:
            return
    if args.from_shards:
        shards = sorted(paths.parquet_dir.glob("shard_*.parquet"))
        if not shards:
            parser.error("no parsed shards; parse the archive first")
        triples = pl.scan_parquet(shards).select("source", "edge_type", "target")
        nodes_frame = pl.concat([triples.select(pl.col("source").alias("node_id")),
                                 triples.select(pl.col("target").alias("node_id"))]).unique().sort("node_id").collect(
                                     engine="streaming").with_row_index("node_int")
        print(f"entity map: {len(nodes_frame):,}", flush=True)
        relations_frame = triples.select("edge_type").unique().sort("edge_type").collect(
            engine="streaming").with_row_index("rel_int")
        mapped = triples.join(nodes_frame.rename({"node_id": "source", "node_int": "s"}).lazy(), on="source").join(
            nodes_frame.rename({"node_id": "target", "node_int": "o"}).lazy(), on="target").join(
            relations_frame.lazy(), on="edge_type").select("s", "rel_int", "o")
        temporal = {}
        if args.temporal_dates:
            if args.temporal_train_through >= args.temporal_valid_through:
                parser.error("temporal training cutoff must precede validation cutoff")
            dated = pl.read_parquet(args.temporal_dates).join(
                nodes_frame.rename({"node_id": "source", "node_int": "s"}), on="source").join(
                nodes_frame.rename({"node_id": "target", "node_int": "o"}), on="target").join(
                relations_frame, on="edge_type").select("s", "rel_int", "o", "start_year")
            future = mapped.join(dated.lazy().filter(pl.col("start_year") > args.temporal_train_through),
                                 on=["s", "rel_int", "o"], how="inner").unique().collect(engine="streaming")
            excluded = future.select("s", "rel_int", "o")
            excluded = pl.concat([excluded, excluded.select(pl.col("o").alias("s"), "rel_int", pl.col("s").alias("o"))]).unique()
            mapped = mapped.join(excluded.lazy(), on=["s", "rel_int", "o"], how="anti")
            for name, part in (("temporal_valid", future.filter(pl.col("start_year") <= args.temporal_valid_through)),
                               ("temporal_test", future.filter(pl.col("start_year") > args.temporal_valid_through))):
                temporal[name] = (part.select("s", "rel_int", "o").to_numpy().astype(np.int64),
                                  part["start_year"].to_numpy())
            print(f"dated future queries excluded from all training/support context: {len(future):,}", flush=True)
            del dated, future, excluded
        # Stream integerization to disk. Keeping all full-graph strings AND
        # all joined copies resident can exceed workstation RAM.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        numeric_path = args.output.with_suffix(".edges.parquet")
        mapped.sink_parquet(numeric_path, engine="streaming")
        print(f"integer facts written to {numeric_path}", flush=True)
        edges = pl.read_parquet(numeric_path).to_numpy().astype(np.int64)
        del mapped, triples
        print(f"uncapped source: {len(edges):,} facts, {len(nodes_frame):,} entities, {len(relations_frame):,} relations", flush=True)
        result = split_entities(edges, len(nodes_frame), args.unseen_fraction, args.valid_size, args.test_size, args.seed)
        del edges
        order = result.pop("node_order")
        if temporal:
            inverse = np.empty(len(order), dtype=np.int64)
            inverse[order] = np.arange(len(order))
            known_relations = np.unique(result["train"][:, 1])
            for name, (queries, years) in temporal.items():
                queries[:, 0] = inverse[queries[:, 0]]
                queries[:, 2] = inverse[queries[:, 2]]
                known = ((queries[:, 0] < result["num_train_nodes"])
                         & (queries[:, 2] < result["num_train_nodes"])
                         & np.isin(queries[:, 1], known_relations))
                result[name] = queries[known]
                result[name + "_years"] = years[known]
                print(f"{name}: known-entity queries={known.sum():,}", flush=True)
            result["temporal_cutoffs"] = np.asarray([args.temporal_train_through, args.temporal_valid_through])
        # Avoid fixed-width Unicode padding to the longest entity name across
        # millions of rows. External names live in a compact Parquet sidecar.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        sidecar = args.output.with_suffix(".nodes.parquet")
        nodes_frame[order].drop("node_int").with_row_index("node_int").write_parquet(sidecar)
        np.savez_compressed(args.output, **result, node_ids=np.arange(len(nodes_frame), dtype=np.int64),
                            relation_ids=np.asarray(relations_frame["edge_type"].to_list(), dtype=str), seed=args.seed,
                            node_mapping_path=str(sidecar.resolve()), source_mode="uncapped parsed shards")
        print(json.dumps({key: len(result[key]) for key in ("train", "valid", "support", "test")}, indent=2))
        return
    nodes = pl.read_parquet(paths.node_map_path).sort("node_int")["node_id"].to_list()
    relations = pl.read_parquet(paths.relation_map_path).sort("rel_int")["edge_type"].to_list()
    node_index = {name: i for i, name in enumerate(nodes)}
    relation_index = {name: i for i, name in enumerate(relations)}
    rows = []
    # Start from TRAIN ONLY: original valid/test facts are not repurposed as support.
    for row in pl.read_parquet(paths.triples_train_path).iter_rows(named=True):
        rows.append((node_index[row["source"]], relation_index[row["edge_type"]], node_index[row["target"]]))
    result = split_entities(np.asarray(rows), len(nodes), args.unseen_fraction,
                            args.valid_size, args.test_size, args.seed)
    order = result.pop("node_order")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result, node_ids=np.asarray(nodes)[order],
                        relation_ids=np.asarray(relations), seed=args.seed)
    print(json.dumps({key: len(result[key]) for key in ("train", "valid", "support", "test")}, indent=2))
    print(f"saved {args.output}; training entities={result['num_train_nodes']:,}")


if __name__ == "__main__":
    main()
