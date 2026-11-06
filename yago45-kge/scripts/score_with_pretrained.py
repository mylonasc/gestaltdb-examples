"""Score triples with the pretrained inductive YAGO model, no training data needed.

Two independent modes:

1. Static cache mode (--cache-dir): memory-map the released
   gnn_embeddings.npy, resolve entity names through node_mapping.parquet,
   and rank (subject, relation, object) triples with the SavedModel
   `score_embeddings` signature. No graph or GPU required.

2. Fresh-neighborhood mode (--edges): encode entities never seen in
   training from a small user-supplied edge list, using the SavedModel
   `predict_graph` signature over a masked sampled neighborhood. This is
   the inductive path: entity names need an observed neighborhood.

Pretrained artifacts:
  hf download mylonasc/yago45-inductive-gnn            # SavedModel
  hf download mylonasc/yago45-inductive-embeddings --repo-type dataset

Run from yago45-kge, e.g.:
  .venv/bin/python scripts/score_with_pretrained.py --model-dir <saved_model> \\
      --cache-dir <embeddings_checkout> --triples queries.tsv
  .venv/bin/python scripts/score_with_pretrained.py --model-dir <saved_model> \\
      --edges my_graph.tsv --query 0 --corruptions 4
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


def resolve_model(model_dir: str) -> Path:
    path = Path(model_dir).expanduser()
    for candidate in (path, path / "gnn_saved_model"):
        if (candidate / "saved_model.pb").exists():
            return candidate
    if "/" in model_dir:
        try:
            out = subprocess.run(["hf", "download", model_dir], text=True,
                                 capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        else:
            roots = [Path(line.strip()) for line in out.stdout.splitlines() + out.stderr.splitlines()]
            for root in roots:
                for candidate in (root, root / "gnn_saved_model"):
                    if candidate.is_dir() and (candidate / "saved_model.pb").exists():
                        print(f"downloaded model to {candidate}", flush=True)
                        return candidate
    raise SystemExit(f"SavedModel not found at {model_dir}; "
                     "run `hf download mylonasc/yago45-inductive-gnn` first")


def load_pretrained(model_dir: Path):
    import tensorflow as tf
    model = tf.saved_model.load(str(model_dir))
    graph_spec = model.encode.input_signature[0]
    feature_dim = int(graph_spec["features"].shape[-1])
    if (feature_dim - 1) % 2:
        raise SystemExit(f"unexpected encoder feature width {feature_dim}")
    slots = (feature_dim - 1) // 2
    metadata = model_dir / "metadata.json"
    vocabulary = None
    if metadata.exists():
        vocabulary = json.loads(metadata.read_text()).get("relation_ids")
        if vocabulary and len(vocabulary) != slots:
            raise SystemExit("export metadata disagrees with encoder feature width")
    return model, slots, vocabulary


def cache_mode(model, args):
    import polars as pl
    cache = Path(args.cache_dir)
    vectors = np.load(cache / "gnn_embeddings.npy", mmap_mode="r").astype(np.float32)
    mapping = pl.read_parquet(cache / "node_mapping.parquet").sort("node_int")
    row_of = dict(zip(mapping["node_id"].to_list(), mapping["node_int"].to_list()))
    relations = json.loads((cache / "relation_ids.json").read_text())
    rel_of = {name: index for index, name in enumerate(relations)}
    triples, names = [], []
    for line in Path(args.triples).read_text().splitlines():
        if line.strip():
            s, r, o = line.split("\t")
            triples.append((row_of[s], rel_of[r], row_of[o]))
            names.append((s, r, o))
    triples = np.asarray(triples, dtype=np.int32)
    scores = model.score_embeddings(
        tf_subjects(vectors, triples[:, 0]), triples[:, 1].astype(np.int32),
        tf_subjects(vectors, triples[:, 2])).numpy()
    for (s, r, o), score in zip(names, scores):
        print(f"{score:+.4f}\t{s}\t{r}\t{o}")


def tf_subjects(vectors, rows):
    import tensorflow as tf
    return tf.convert_to_tensor(np.asarray(vectors[np.asarray(rows)], dtype=np.float32))


def edges_mode(model, slots, vocabulary, args):
    import tensorflow as tf
    from yago.inductive_data import NeighborhoodGraph
    edges, nodes = [], {}
    if vocabulary is not None:
        rel_of = {name: index for index, name in enumerate(vocabulary)}
    else:
        rel_of = {}
    for line in Path(args.edges).read_text().splitlines():
        if line.strip():
            s, r, o = line.split("\t")
            if r not in rel_of:
                if vocabulary is not None:
                    raise SystemExit(f"relation {r!r} is not one of the trained relation IDs")
                rel_of[r] = len(rel_of)
            edges.append((nodes.setdefault(s, len(nodes)), rel_of[r], nodes.setdefault(o, len(nodes))))
    if len(rel_of) > slots:
        raise SystemExit(f"demo graph uses {len(rel_of)} relations but the model has {slots} slots")
    edges = np.asarray(edges, dtype=np.int64)
    query = edges[int(args.query)]
    context = NeighborhoodGraph(edges, len(nodes), slots, sampling_version=3)
    rng = np.random.default_rng(13)
    roots = sorted({int(s) for s, _, _ in edges} | {int(o) for _, _, o in edges})
    ids, graph = context.sample(roots, (8, 4, 2), rng, masked=[tuple(int(v) for v in query)])
    local = {node: index for index, node in enumerate(ids.tolist())}
    names = {index: name for name, index in nodes.items()}
    inv_rels = {index: name for name, index in rel_of.items()}
    candidates = [tuple(int(v) for v in query)]
    objects = [o for _, r, o in edges if r == query[1] and o != query[2]][:args.corruptions]
    candidates += [(int(query[0]), int(query[1]), int(o)) for o in objects]
    local_triples = np.asarray([[local[s], r, local[o]] for s, r, o in candidates], dtype=np.int32)
    tensors = {key: tf.convert_to_tensor(value) for key, value in graph.items()}
    scores = model.predict_graph(tensors, local_triples).numpy()
    order = np.argsort(-scores)
    for rank, index in enumerate(order, 1):
        s, r, o = candidates[index]
        mark = "true" if index == 0 else "corrupted"
        print(f"rank={rank} score={scores[index]:+.4f} {mark} "
              f"{names[s]} {inv_rels[r]} {names[o]}")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--model-dir", default="mylonasc/yago45-inductive-gnn")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cache-dir", help="Embeddings checkout (static vectors)")
    group.add_argument("--edges", help="TSV edge list for fresh-neighborhood inference")
    parser.add_argument("--triples", help="TSV (subject, relation, object) names; cache mode")
    parser.add_argument("--query", type=int, default=0, help="Edge row to score; edges mode")
    parser.add_argument("--corruptions", type=int, default=4)
    args = parser.parse_args()
    if args.cache_dir and not args.triples:
        parser.error("--cache-dir requires --triples")
    model, slots, vocabulary = load_pretrained(resolve_model(args.model_dir))
    if args.cache_dir:
        cache_mode(model, args)
    else:
        edges_mode(model, slots, vocabulary, args)


if __name__ == "__main__":
    main()
