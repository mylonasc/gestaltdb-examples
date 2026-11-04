"""Extract actual triple-level validity dates from YAGO RDF-star metadata."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import re
import zipfile

import polars as pl

from .parse import _is_entity_object, _is_entity_subject, _shorten


_START = re.compile(r'^<<\s+(\S+)\s+(\S+)\s+(\S+)\s+>>\s+schema:startDate\s+"([+-]?\d+)-')


def extract_start_years(source, output, snapshot_year=2024):
    rows = []
    shards = []
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / (output.stem + "_shards")
    staging.mkdir(exist_ok=True)

    def flush():
        if rows:
            path = staging / f"part_{len(shards):05d}.parquet"
            pl.DataFrame(rows, schema=["source", "edge_type", "target", "start_year"], orient="row").write_parquet(path)
            shards.append(path)
            rows.clear()

    with zipfile.ZipFile(source) as archive:
        with archive.open("yago-meta-facts.ntx") as stream:
            for raw in io.BufferedReader(stream, 1 << 20):
                match = _START.match(raw.decode("utf-8"))
                if match:
                    s, r, o, year = match.groups()
                    if _is_entity_subject(s) and _is_entity_object(o):
                        rows.append((_shorten(s), r, _shorten(o), int(year)))
                        if len(rows) >= 500000:
                            flush()
    flush()
    if not shards:
        raise ValueError("no entity-to-entity triple start dates found")
    # Several validity intervals can annotate one triple. Its earliest start
    # defines this completion cohort; it is not a fact acquisition timestamp.
    frame = pl.scan_parquet(shards).group_by("source", "edge_type", "target").agg(
        pl.col("start_year").min()).collect(engine="streaming")
    raw_count = len(frame)
    # Wikidata/YAGO contains enormous sentinel/geological years and planned
    # future dates. Do not turn these into spurious "recent" test examples.
    frame = frame.filter(pl.col("start_year").is_between(-10000, snapshot_year))
    frame.write_parquet(output)
    years = frame.group_by("start_year").len().sort("start_year")
    audit = dict(source=str(source.resolve()), metadata_member="yago-meta-facts.ntx",
                 entity_triples_with_start_year=len(frame), earliest_year=int(frame["start_year"].min()),
                 latest_year=int(frame["start_year"].max()), yearly_counts=years.to_dicts(),
                 excluded_out_of_range_years=raw_count - len(frame), snapshot_year=snapshot_year,
                 semantics="valid time (earliest annotated start), not knowledge availability",
                 limitation="Undated static background has unknown availability; not causal forecasting.")
    output.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps({key: value for key, value in audit.items() if key != "yearly_counts"}, indent=2))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--snapshot-year", type=int, default=2024)
    args = parser.parse_args()
    extract_start_years(args.source, args.output, args.snapshot_year)


if __name__ == "__main__":
    main()
