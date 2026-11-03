"""Build maps/arrays, ingest GestaltDB, build sampler snapshot for YAGO."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import polars as pl

from .config import AppConfig


def collect_triples(cfg: AppConfig) -> pl.LazyFrame:
    shards = sorted(cfg.paths.parquet_dir.glob("shard_*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no shards in {cfg.paths.parquet_dir}; run parse first")
    if cfg.paths.triples_parquet.exists():
        return pl.scan_parquet(cfg.paths.triples_parquet)
    lf = pl.scan_parquet(list(shards))
    # Filter rare relations for tractability/noise.
    if cfg.subset.min_relation_edges > 1:
        counts = lf.group_by("edge_type").len().collect()
        keep = set(counts.filter(pl.col("len") >= cfg.subset.min_relation_edges)["edge_type"].to_list())
        if cfg.subset.max_relations is not None:
            top = counts.sort("len", descending=True).head(cfg.subset.max_relations)
            keep &= set(top["edge_type"].to_list())
        lf = lf.filter(pl.col("edge_type").is_in(list(keep)))
    if cfg.subset.max_edges_per_relation is not None:
        cap = cfg.subset.max_edges_per_relation
        df = lf.collect()
        parts = []
        for _, g in df.group_by("edge_type", maintain_order=True):
            if len(g) > cap:
                parts.append(g.sample(n=cap, seed=cfg.subset.seed))
            else:
                parts.append(g)
        df = pl.concat(parts)
        lf = df.lazy()
    if cfg.subset.target_edges is not None:
        counts = lf.group_by("edge_type").len().collect().sort("len", descending=True)
        picked: list[str] = []
        total = 0
        for row in counts.iter_rows(named=True):
            picked.append(row["edge_type"])
            total += row["len"]
            if total >= cfg.subset.target_edges:
                break
        lf = lf.filter(pl.col("edge_type").is_in(picked))
    cfg.paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
    lf.sink_parquet(cfg.paths.triples_parquet)
    return pl.scan_parquet(cfg.paths.triples_parquet)


def materialize_artifacts(cfg: AppConfig) -> dict:
    triples = collect_triples(cfg).collect()
    print(f"triples={len(triples):,}")
    nodes = pl.concat([
        triples.select(pl.col("source").alias("node_id")),
        triples.select(pl.col("target").alias("node_id")),
    ]).unique()
    print(f"nodes={len(nodes):,}")
    node_map = nodes.with_row_index("node_int").with_columns(pl.lit("Entity").alias("kind"))
    rel_map = (
        triples.group_by("edge_type").len().sort("edge_type")
        .with_row_index("rel_int")
    )
    node_map.write_parquet(cfg.paths.node_map_path)
    rel_map.write_parquet(cfg.paths.relation_map_path)
    id2int = dict(zip(node_map["node_id"].to_list(), node_map["node_int"].to_list()))
    rel2int = dict(zip(rel_map["edge_type"].to_list(), rel_map["rel_int"].to_list()))
    # Map in chunks to bound RAM.
    src = triples["source"].to_list()
    dst = triples["target"].to_list()
    rel = triples["edge_type"].to_list()
    import array as _arr
    src_int = np.fromiter((id2int[s] for s in src), dtype=np.int64, count=len(src))
    dst_int = np.fromiter((id2int[d] for d in dst), dtype=np.int64, count=len(dst))
    rel_int = np.fromiter((rel2int[r] for r in rel), dtype=np.int64, count=len(rel))
    edge_ids = np.array([f"e{i}".encode() if False else f"e{i}" for i in range(len(src))])
    node_type = np.array(["Entity"] * len(src))
    np.savez_compressed(
        cfg.paths.edge_arrays_path,
        src=src_int, dst=dst_int, rel=rel_int,
        edge_id=edge_ids,
        node1_type=node_type, node2_type=node_type,
    )
    meta = {
        "num_nodes": len(node_map),
        "num_edges": len(triples),
        "num_relations": len(rel_map),
        "relations": rel_map.select(["rel_int", "edge_type", "len"]).to_dicts(),
    }
    cfg.paths.metadata_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "relations"}, indent=2))
    return meta


def ingest_gestaltdb(cfg: AppConfig) -> None:
    import sys
    from gestaltdb import IndexMaintenanceMode
    from gestaltdb.graphdb import GraphDB

    if cfg.ingestion.reset_db and cfg.paths.db_path.exists():
        shutil.rmtree(cfg.paths.db_path)
    graphdb = GraphDB.create(
        cfg.paths.db_path,
        backend="pyrex",
        serializer="json",
        backend_options={
            "disable_wal": True,
            "write_buffer_size": cfg.ingestion.write_buffer_size,
            "parallelism": cfg.ingestion.parallelism,
            "max_background_jobs": cfg.ingestion.max_background_jobs,
        },
    )
    try:
        nodes = pl.scan_parquet(cfg.paths.node_map_path).select(
            pl.col("node_id"), pl.lit(["Entity"]).alias("labels") if False else pl.concat_list(pl.lit("Entity")).alias("labels"),
            pl.col("kind"),
        )
        edges = pl.scan_parquet(cfg.paths.triples_parquet).with_row_index("edge_id")
        edges = edges.with_columns(pl.col("edge_id").cast(pl.Utf8).alias("edge_id"))
        t0 = time.perf_counter()
        n_nodes = graphdb.ingest_nodes_polars_entities(
            nodes, node_id="node_id", labels="labels",
            property_columns=["kind"],
            append_only=True, index_mode=IndexMaintenanceMode.DEFER,
            chunk_size=cfg.ingestion.chunk_size, progress=cfg.ingestion.progress,
        )
        n_edges = graphdb.ingest_edges_polars_entities(
            edges, edge_id="edge_id", source="source", target="target",
            edge_type="edge_type", property_columns=[],
            append_only=True, index_mode=IndexMaintenanceMode.DEFER,
            chunk_size=cfg.ingestion.chunk_size, progress=cfg.ingestion.progress,
        )
        print(f"ingested {n_nodes:,} nodes {n_edges:,} edges in {time.perf_counter()-t0:.1f}s")
    finally:
        graphdb.close()


def build_snapshot(cfg: AppConfig, *, mode: str = "ram"):
    from gestaltdb.sampling import SamplerSnapshot
    path = cfg.paths.sampler_snapshot_path
    if not (path / "metadata.json").exists():
        import shutil as _shutil
        if path.exists():
            _shutil.rmtree(path)
        arrays = np.load(cfg.paths.edge_arrays_path, allow_pickle=True)
        node_map = pl.read_parquet(cfg.paths.node_map_path).sort("node_int")
        rel_map = pl.read_parquet(cfg.paths.relation_map_path).sort("rel_int")
        SamplerSnapshot.from_edge_arrays(
            path,
            external_node_ids=node_map["node_id"].to_list(),
            external_edge_ids=arrays["edge_id"].astype(str),
            external_relation_ids=rel_map["edge_type"].to_list(),
            src_int=arrays["src"].astype(np.int64),
            dst_int=arrays["dst"].astype(np.int64),
            rel_int=arrays["rel"].astype(np.int64),
            node_type_values=node_map["kind"].to_list(),
            edge_src_type_values=arrays["node1_type"].astype(str),
            edge_dst_type_values=arrays["node2_type"].astype(str),
            source_db={"path": cfg.paths.db_path, "path_type": "relative_to_snapshot"},
            source_artifacts={
                "edge_arrays": cfg.paths.edge_arrays_path,
                "node_map": cfg.paths.node_map_path,
            },
            metadata={"source": "yago45 entity-entity triples"},
        )
    return SamplerSnapshot.load(path, mmap=(mode == "memmap"))
