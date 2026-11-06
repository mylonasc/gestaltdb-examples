"""Smoke tests for the pretrained quickstart script (no dataset, no GPU)."""
import json
import subprocess
import sys

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")


def _toy_model(path):
    from yago.inductive_export import export_encoder
    from yago.inductive_model import InductiveAutoencoder
    model = InductiveAutoencoder(6, 2, dim=8, hidden_dim=12, layers=1, encoder="relational")
    context = __import__("yago.inductive_data", fromlist=["NeighborhoodGraph"]).NeighborhoodGraph(
        np.asarray([[0, 0, 1], [1, 1, 2], [2, 0, 3]]), 6, 2)
    _, graph = context.sample([0, 1], [2, 2], np.random.default_rng(1))
    tensors = {key: tf.convert_to_tensor(value) for key, value in graph.items()}
    model.encode(tensors)
    export_encoder(model, path / "gnn")


def test_edges_mode_ranks_masked_true_fact_first(tmp_path):
    _toy_model(tmp_path)
    edges = tmp_path / "edges.tsv"
    edges.write_text("a\tr0\tb\nb\tr1\tc\nc\tr0\td\na\tr1\td\n")
    result = subprocess.run(
        [sys.executable, "scripts/score_with_pretrained.py",
         "--model-dir", str(tmp_path / "gnn"), "--edges", str(edges),
         "--query", "0", "--corruptions", "2"],
        text=True, capture_output=True, timeout=300, cwd=".")
    assert result.returncode == 0, result.stdout + result.stderr
    lines = result.stdout.strip().splitlines()
    assert len(lines) == 2 and lines[0].startswith("rank=1")
    assert sum("true" in line for line in lines) == 1


def test_cache_mode_scores_named_triples(tmp_path):
    import polars as pl
    _toy_model(tmp_path)
    rng = np.random.default_rng(0)
    np.save(tmp_path / "gnn_embeddings.npy", rng.normal(size=(4, 8)).astype(np.float16))
    pl.DataFrame({"node_id": ["n0", "n1", "n2", "n3"],
                  "node_int": [0, 1, 2, 3]}).write_parquet(tmp_path / "node_mapping.parquet")
    (tmp_path / "relation_ids.json").write_text(json.dumps(["r0", "r1"]))
    queries = tmp_path / "queries.tsv"
    queries.write_text("n0\tr0\tn1\nn2\tr1\tn3\n")
    result = subprocess.run(
        [sys.executable, "scripts/score_with_pretrained.py",
         "--model-dir", str(tmp_path / "gnn"), "--cache-dir", str(tmp_path),
         "--triples", str(queries)],
        text=True, capture_output=True, timeout=300, cwd=".")
    assert result.returncode == 0, result.stdout + result.stderr
    lines = result.stdout.strip().splitlines()
    assert len(lines) == 2 and all("\t" in line for line in lines)
