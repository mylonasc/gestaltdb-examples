"""CPU smoke tests including fresh-process restore and CLI inference."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from yago.inductive_data import NeighborhoodGraph, split_entities
from yago.inductive_model import InductiveAutoencoder


@pytest.mark.parametrize("encoder", ["relational", "tf-gnns"])
def test_encoder_is_independent_of_entity_table_and_handles_unseen_nodes(tmp_path, encoder):
    model = InductiveAutoencoder(2, 2, dim=8, hidden_dim=12, encoder=encoder)
    context = NeighborhoodGraph(np.asarray([[0, 0, 1], [1, 1, 2]]), 3, 2)
    _, graph = context.sample([2], [10, 10], np.random.default_rng(1))
    tensors = {key: tf.convert_to_tensor(value) for key, value in graph.items()}
    before = model.encode(tensors).numpy()
    model.entity_table.embeddings.assign(tf.ones_like(model.entity_table.embeddings) * 100)
    np.testing.assert_array_equal(before, model.encode(tensors).numpy())
    path = tf.train.Checkpoint(model=model).save(str(tmp_path / "ckpt"))
    restored = InductiveAutoencoder(2, 2, dim=8, hidden_dim=12, encoder=encoder)
    tf.train.Checkpoint(model=restored).restore(path).assert_consumed()
    np.testing.assert_allclose(before, restored.encode(tensors).numpy(), atol=1e-7)


@pytest.mark.parametrize("shallow_recipe", ["adam", "cpu-sgd", "cpu-adagrad"])
def test_training_resume_and_inductive_cli(tmp_path, shallow_recipe):
    edges = np.asarray([(s, (s + o) % 2, o) for s in range(16) for o in range(16) if s != o])
    data = split_entities(edges, 16, 0.25, 8, 8, 13)
    order = data.pop("node_order")
    dataset = tmp_path / "dataset.npz"
    np.savez_compressed(dataset, **data, node_ids=np.asarray([f"n{i}" for i in order]),
                        relation_ids=np.asarray(["r0", "r1"]))
    run = tmp_path / "run"
    common = [sys.executable, "-m", "yago.inductive_train", "--dataset", str(dataset),
              "--run-dir", str(run), "--batch-size", "2", "--negatives", "2",
              "--dim", "8", "--hidden-dim", "12", "--fanouts", "2", "2",
              "--checkpoint-every", "1", "--eval-every", "1", "--eval-triples", "2"]
    if shallow_recipe != "adam":
        common += ["--embedding-device", "cpu", "--shallow-optimizer", "sgd" if shallow_recipe == "cpu-sgd" else "sparse-adagrad",
                   "--shallow-learning-rate", "0.1", "--base-dim", "4", "--validation-policy", "both"]
    if shallow_recipe == "cpu-adagrad":
        common += ["--sampler", "gestaltdb-csr", "--keep-checkpoints", "1"]
    env = dict(os.environ, TF_NUM_INTEROP_THREADS="1", TF_NUM_INTRAOP_THREADS="1", TF_CPP_MIN_LOG_LEVEL="3")

    def invoke(command):
        result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        return result

    invoke(common + ["--steps", "1"])
    artifact = np.load(run / "shallow_embeddings.npz")
    assert len(artifact["node_ids"]) == data["num_train_nodes"]
    invoke(common + ["--steps", "2", "--resume"])
    curves = [json.loads(line) for line in (run / "curves.jsonl").read_text().splitlines()]
    assert [row["step"] for row in curves] == [0, 1, 2]
    checkpoint = tf.train.load_checkpoint(tf.train.latest_checkpoint(str(run / "checkpoints")))
    assert checkpoint.get_tensor("step/.ATTRIBUTES/VARIABLE_VALUE") == 2
    invoke([sys.executable, "-m", "yago.inductive_eval", "--run-dir", str(run),
            "--triples", "3", "--negatives", "4"])
    result = json.loads((run / "gnn_test_evaluation.json").read_text())
    assert result["queries"] == 3
    assert result["context"] == "train + inference support"
    assert 0 < result["mrr"] <= 1
    # Resume must be equivalent to uninterrupted training (same NumPy sampler
    # RNG, optimizer slots and weights; encoder contains no stochastic layers).
    full_run = tmp_path / "full_run"
    uninterrupted = common.copy()
    uninterrupted[uninterrupted.index(str(run))] = str(full_run)
    invoke(uninterrupted + ["--steps", "2"])
    full = tf.train.load_checkpoint(tf.train.latest_checkpoint(str(full_run / "checkpoints")))
    for name, _ in tf.train.list_variables(tf.train.latest_checkpoint(str(run / "checkpoints"))):
        if name.startswith(("model/", "optimizer/", "shallow_optimizer/")):
            np.testing.assert_allclose(checkpoint.get_tensor(name), full.get_tensor(name), atol=1e-6)


def test_native_sparse_adagrad_coalesces_duplicates_and_leaves_other_rows_untouched():
    from yago.sparse_optimizer import SparseAdagrad
    table = tf.keras.layers.Embedding(100, 4)
    table(tf.constant([0]))
    table.embeddings.assign(tf.ones_like(table.embeddings))
    optimizer = SparseAdagrad(.01)
    optimizer.build([table.embeddings])
    gradient = tf.IndexedSlices(tf.ones((3, 4)), tf.constant([2, 2, 5]), tf.constant([100, 4]))
    optimizer.apply_gradients([(gradient, table.embeddings)])
    np.testing.assert_array_equal(table.embeddings.numpy()[0], np.ones(4))
    np.testing.assert_allclose(table.embeddings.numpy()[2], np.full(4, .99), atol=1e-6)
    np.testing.assert_allclose(optimizer.accumulators[0].numpy()[2], np.full(4, 4), atol=1e-6)


def test_prepare_cli_uses_only_original_training_facts(tmp_path):
    pl = pytest.importorskip("polars")
    artifacts = tmp_path / "source" / "artifacts"
    artifacts.mkdir(parents=True)
    pl.DataFrame(dict(node_int=list(range(20)), node_id=[f"n{i}" for i in range(20)])).write_parquet(
        artifacts / "node_map.parquet")
    pl.DataFrame(dict(rel_int=[0, 1], edge_type=["r0", "r1"])).write_parquet(
        artifacts / "relation_map.parquet")
    rows = [(f"n{s}", f"r{(s + o) % 2}", f"n{o}") for s in range(20) for o in range(20) if s != o]
    pl.DataFrame(rows, schema=["source", "edge_type", "target"], orient="row").write_parquet(
        artifacts / "triples_train.parquet")
    dataset = tmp_path / "prepared.npz"
    result = subprocess.run([sys.executable, "-m", "yago.inductive_prepare", "--data-root",
                             str(artifacts.parent), "--output", str(dataset), "--valid-size", "8",
                             "--test-size", "8"], text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    data = np.load(dataset)
    assert len(data["test"]) == 8
    assert len(data["valid"]) == 8
    # The large-graph path must retain all parsed facts and preserve external
    # identities in its compact sidecar, rather than padding a huge Unicode array.
    shards = artifacts.parent / "parquet_shards"
    shards.mkdir()
    pl.DataFrame(rows, schema=["source", "edge_type", "target"], orient="row").write_parquet(
        shards / "shard_00000.parquet")
    full_dataset = tmp_path / "uncapped.npz"
    result = subprocess.run([sys.executable, "-m", "yago.inductive_prepare", "--data-root",
                             str(artifacts.parent), "--output", str(full_dataset), "--valid-size", "8",
                             "--test-size", "8", "--from-shards"], text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    full = np.load(full_dataset)
    names = pl.read_parquet(str(full["node_mapping_path"])).sort("node_int")["node_id"].to_list()
    recovered = set()
    for part in ("train", "valid", "support", "test"):
        for s, r, o in full[part]:
            recovered.add((names[s], str(full["relation_ids"][r]), names[o]))
    assert recovered == set(rows)
    dates = tmp_path / "dates.parquet"
    pl.DataFrame(dict(source=["n0", "n2", "n4"], edge_type=["r1"] * 3,
                      target=["n1", "n3", "n5"], start_year=[2014, 2020, 1999])).write_parquet(dates)
    temporal_dataset = tmp_path / "temporal.npz"
    result = subprocess.run([sys.executable, "-m", "yago.inductive_prepare", "--data-root",
                             str(artifacts.parent), "--output", str(temporal_dataset), "--valid-size", "8",
                             "--test-size", "8", "--from-shards", "--temporal-dates", str(dates)],
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    temporal = np.load(temporal_dataset)
    assert temporal["temporal_valid_years"].tolist() == [2014]
    assert temporal["temporal_test_years"].tolist() == [2020]
    context = set(map(tuple, np.concatenate([temporal["train"], temporal["support"]])))
    for key in ("temporal_valid", "temporal_test"):
        for s, r, o in temporal[key]:
            assert (s, r, o) not in context
            assert (o, r, s) not in context


def test_actual_triple_temporal_metadata_not_entity_dates(tmp_path):
    import zipfile
    from yago.inductive_temporal import extract_start_years
    archive = tmp_path / "yago.zip"
    lines = '\n'.join([
        '<< yago:A schema:memberOf yago:B >> schema:startDate "2000-01-01T00:00:00Z"^^xsd:dateTime',
        '<< yago:A schema:memberOf yago:B >> schema:startDate "2005-01-01T00:00:00Z"^^xsd:dateTime',
        '<< yago:C schema:memberOf yago:D >> schema:startDate "5000000000-01-01T00:00:00Z"^^xsd:dateTime',
        'yago:E schema:birthDate "2000-01-01"^^xsd:date .',
        '<< yago:A schema:birthDate "2000-01-01"^^xsd:date >> schema:startDate "2000-01-01"^^xsd:date',
    ])
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("yago-meta-facts.ntx", lines)
    output = tmp_path / "dates.parquet"
    extract_start_years(archive, output)
    import polars as pl
    frame = pl.read_parquet(output)
    assert len(frame) == 1
    assert frame["start_year"].to_list() == [2000]
