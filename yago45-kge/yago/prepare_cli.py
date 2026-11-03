"""Prepare CLI: parse TTL -> artifacts -> gestaltdb ingest -> snapshot."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yago.config import make_config
from yago.parse import parse_yago_ttl_to_shards
from yago.prepare import build_snapshot, collect_triples, ingest_gestaltdb, materialize_artifacts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ttl", type=str, default=None, help="path to .ttl or .zip (default: tiny zip)")
    ap.add_argument("--ttl-member", type=str, default=None,
                    help="zip member to stream (e.g. yago-facts.ttl for the full archive)")
    ap.add_argument("--limit-lines", type=int, default=None)
    ap.add_argument("--shard-rows", type=int, default=1_000_000)
    ap.add_argument("--skip-parse", action="store_true")
    ap.add_argument("--skip-ingest", action="store_true")
    ap.add_argument("--reset-db", action="store_true")
    ap.add_argument("--min-relation-edges", type=int, default=None)
    ap.add_argument("--target-edges", type=int, default=None)
    ap.add_argument("--max-edges-per-relation", type=int, default=None)
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--data-root", type=str, default=None,
                    help="override data dir (default: data/yago45)")
    args = ap.parse_args()
    cfg = make_config(smoke_test=args.smoke_test)
    if args.data_root:
        from yago.config import PathsConfig
        cfg.paths = PathsConfig(data_root=Path(args.data_root))
    if args.reset_db:
        cfg.ingestion.reset_db = True
    if args.min_relation_edges is not None:
        cfg.subset.min_relation_edges = args.min_relation_edges
    if args.target_edges is not None:
        cfg.subset.target_edges = args.target_edges
    if args.max_edges_per_relation is not None:
        cfg.subset.max_edges_per_relation = args.max_edges_per_relation
    ttl = Path(args.ttl) if args.ttl else cfg.paths.tiny_zip
    if not args.skip_parse:
        parse_yago_ttl_to_shards(cfg, ttl, shard_rows=args.shard_rows,
                                 limit_lines=args.limit_lines, member=args.ttl_member)
    collect_triples(cfg)
    materialize_artifacts(cfg)
    if not args.skip_ingest:
        ingest_gestaltdb(cfg)
    snap = build_snapshot(cfg)
    print(f"snapshot nodes={snap.num_nodes:,} edges={snap.num_edges:,} relations={snap.num_relations}")


if __name__ == "__main__":
    main()
