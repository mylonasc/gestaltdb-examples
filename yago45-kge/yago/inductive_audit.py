"""Audit source date predicates without confusing event dates with fact availability."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from .parse import _TRIPLE_RE, _is_entity_subject, iter_ttl_lines


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    counts = Counter()
    temporal = Counter()
    reified_lines = 0
    lines = 0
    pattern = re.compile(r"Date|Time|valid|^prov:|^pq:|^ps:|^p:|^schema:date")
    for line in iter_ttl_lines(args.source):
        lines += 1
        if line.lstrip().startswith(("<<", "s:", "<http://www.wikidata.org/entity/statement/")):
            reified_lines += 1
        match = _TRIPLE_RE.match(line.strip())
        if match and _is_entity_subject(match.group("sub")):
            predicate = match.group("pred")
            counts[predicate] += 1
            if pattern.search(predicate):
                temporal[predicate] += 1
    availability = {key: value for key, value in temporal.items()
                    if key in ("prov:generatedAtTime", "prov:invalidatedAtTime", "schema:dateModified")}
    digest = hashlib.sha256()
    with args.source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    result = dict(source=str(args.source.resolve()), sha256=digest.hexdigest(), lines=lines,
                  temporal_predicates=dict(temporal), reified_statement_lines=reified_lines,
                  potential_availability_predicates=availability,
                  temporal_split_used=False,
                  reason="Static YAGO snapshot; entity/event dates are not timestamps of fact availability. "
                         "No chronological ordering is inferred from Turtle line order or birth/death dates.",
                  most_common_predicates=counts.most_common(30))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
