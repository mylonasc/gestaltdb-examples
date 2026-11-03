"""Streaming Turtle -> entity-entity triple extractor for YAGO 4.5.

Only stdlib + polars/pyarrow. Streams .ttl (or zipped .ttl) line by line,
keeps triples whose subject and object are both entities (yago:/wd: IRIs
or prefixed names) and whose predicate is not in the skip lists.
Writes sharded parquet files: source, target, edge_type.
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

import polars as pl

from .config import AppConfig

_TRIPLE_RE = re.compile(
    r"^(?P<sub>\S+)\s+(?P<pred>\S+)\s+(?P<obj>.+?)\s+\.\s*$"
)
_PREFIX_RE = re.compile(r"^@prefix\s+(\w+):\s+<([^>]+)>")
_QNAME_RE = re.compile(r"^([A-Za-z][\w\-]*):(.*)$")

# Full IRIs that denote entities (vs literal-typed objects).
_ENTITY_IRI_PREFIXES = (
    "http://yago-knowledge.org/resource/",
    "http://www.wikidata.org/entity/",
)


def _strip_qname(token: str) -> tuple[str, str] | None:
    m = _QNAME_RE.match(token)
    if not m:
        return None
    return m.group(1), m.group(2)


def _is_entity_object(obj: str) -> bool:
    obj = obj.strip()
    if not obj:
        return False
    if obj.startswith('"') or obj.startswith("'"):
        return False
    # Typed literals / language tags.
    if obj.startswith("+") or obj.startswith("-"):
        # Could be a bare number literal.
        rest = obj[1:2]
        if rest.isdigit() or rest in ('"', "'"):
            return False
    if "@" in obj and ('"' in obj or "'" in obj):
        return False
    if "^^" in obj:
        return False
    if obj.startswith("<"):
        iri = obj[1:].split(">", 1)[0]
        return iri.startswith(_ENTITY_IRI_PREFIXES)
    q = _strip_qname(obj)
    if q is None:
        return False
    prefix, _ = q
    return prefix in ("yago", "wd")


def _is_entity_subject(sub: str) -> bool:
    sub = sub.strip()
    if sub.startswith("<"):
        iri = sub[1:].split(">", 1)[0]
        return iri.startswith(_ENTITY_IRI_PREFIXES)
    q = _strip_qname(sub)
    if q is None:
        return False
    return q[0] in ("yago", "wd")


def _shorten(token: str) -> str:
    token = token.strip()
    if token.startswith("<"):
        iri = token[1:].split(">", 1)[0]
        for pre, full in (("yago", "http://yago-knowledge.org/resource/"),
                          ("wd", "http://www.wikidata.org/entity/")):
            if iri.startswith(full):
                return f"{pre}:{iri[len(full):]}"
        return iri
    return token


def iter_ttl_lines(path: Path, member: str | None = None):
    """Yield text lines from a .ttl file or a .zip containing one.

    Args:
        path: .ttl file or .zip archive.
        member: when ``path`` is a zip, only stream this member name
            (e.g. ``"yago-facts.ttl"``). Default streams all ``*.ttl`` members.
    """
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if n.endswith(".ttl")]
            if member is not None:
                if member not in names:
                    raise ValueError(f"member {member!r} not in {path}; have {names}")
                names = [member]
            if not names:
                raise ValueError(f"no .ttl in {path}")
            for name in names:
                with zf.open(name) as fh:
                    import io
                    buf = io.BufferedReader(fh, buffer_size=1 << 20)
                    for raw in buf:
                        yield raw.decode("utf-8", errors="replace")
    else:
        with open(path, encoding="utf-8", errors="replace") as fh:
            yield from fh


def parse_yago_ttl_to_shards(
    cfg: AppConfig,
    ttl_path: Path,
    *,
    shard_rows: int = 1_000_000,
    limit_lines: int | None = None,
    member: str | None = None,
) -> list[Path]:
    """Stream a YAGO .ttl/.zip into parquet shards of entity-entity triples."""
    skip_pred = set(cfg.subset.skip_predicates)
    skip_pref = tuple(cfg.subset.skip_prefixes)
    out_dir = cfg.paths.parquet_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    # Clean previous shards for this source.
    for old in sorted(out_dir.glob("shard_*.parquet")):
        old.unlink()

    srcs: list[str] = []
    dsts: list[str] = []
    rels: list[str] = []
    shard_id = 0
    written: list[Path] = []
    kept = 0
    scanned = 0

    def flush():
        nonlocal shard_id, srcs, dsts, rels
        if not srcs:
            return
        p = out_dir / f"shard_{shard_id:05d}.parquet"
        pl.DataFrame({"source": srcs, "target": dsts, "edge_type": rels}).write_parquet(p)
        written.append(p)
        shard_id += 1
        srcs, dsts, rels = [], [], []

    for line in iter_ttl_lines(ttl_path, member=member):
        scanned += 1
        if limit_lines is not None and scanned > limit_lines:
            break
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("@"):
            continue
        m = _TRIPLE_RE.match(s)
        if not m:
            continue
        pred = m.group("pred")
        if pred in skip_pred:
            continue
        if pred.startswith(skip_pref):
            continue
        # Skip predicates that are clearly literal-valued schema props.
        sub, obj = m.group("sub"), m.group("obj").strip()
        if not _is_entity_subject(sub):
            continue
        # Remove trailing datatype/lang from obj check already handled.
        if not _is_entity_object(obj.split("^^")[0]):
            continue
        # Also skip literal-valued predicates by name as a safety net.
        plow = pred.lower()
        if any(k in plow for k in ("label", "comment", "altname", "sameas",
                                   "mainentity", "image", "url", "geo",
                                   "birthdate", "deathdate", "datecreated",
                                   "postalcode", "population", "area", "elevation",
                                   "demonym", "slogan", "unemployment", "humandevelopment")):
            continue
        srcs.append(_shorten(sub))
        dsts.append(_shorten(obj))
        rels.append(pred)
        kept += 1
        if len(srcs) >= shard_rows:
            flush()
    flush()
    print(f"scanned_lines={scanned:,} kept_entity_entity={kept:,} shards={len(written)}")
    return written
