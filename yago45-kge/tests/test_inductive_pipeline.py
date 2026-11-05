"""CPU smoke tests including fresh-process restore and CLI inference."""
import json
import os
import signal
import subprocess
import sys

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

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
    if shallow_recipe == "adam":
        cache = tmp_path / "cache"
        command = [sys.executable, "-m", "yago.inductive_cache", "--run-dir", str(run),
                   "--output", str(cache), "--batch-size", "4", "--eval-triples", "2"]
        invoke(command)
        manifest = json.loads((cache / "cache.json").read_text())
        assert manifest["complete"]
        assert manifest["completed_nodes"] == 16
        vectors = np.load(cache / "gnn_embeddings.npy", mmap_mode="r")
        assert vectors.shape == (16, 8)
        assert vectors.dtype == np.float16
        assert np.all(np.isfinite(vectors))
        assert (cache / "cached_link_prediction.json").exists()
        invoke(command + ["--resume"])
        report = tmp_path / "report"
        command = [sys.executable, "-m", "yago.inductive_report", "--run-dir", str(run),
                   "--output", str(report), "--triples", "2", "--tsne-nodes", "16", "--publish-gnn"]
        invoke(command)
        ranking = report / "test_uniform_details.json"
        original_mtime = ranking.stat().st_mtime_ns
        invoke(command + ["--resume-report"])
        assert ranking.stat().st_mtime_ns == original_mtime
        assert (report / "summary.json").exists()
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
    if shallow_recipe == "cpu-adagrad":
        stopping_run = tmp_path / "stopping_run"
        stopping = common.copy()
        stopping[stopping.index(str(run))] = str(stopping_run)
        process = subprocess.Popen(stopping + ["--steps", "100000", "--no-export"], env=env,
                                   text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = []
        for line in process.stdout:
            output.append(line)
            if line.startswith("step=1 loss="):
                process.send_signal(signal.SIGTERM)
                break
        tail, _ = process.communicate(timeout=120)
        assert process.returncode == 0, "".join(output) + tail
        assert "stopped after checkpointing complete step" in tail
        stopped_checkpoint = tf.train.latest_checkpoint(str(stopping_run / "checkpoints"))
        stopped_step = int(tf.train.load_checkpoint(stopped_checkpoint).get_tensor("step/.ATTRIBUTES/VARIABLE_VALUE"))
        assert 1 <= stopped_step < 100000
        invoke(stopping + ["--steps", str(stopped_step + 1), "--resume", "--no-export"])


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


def test_standalone_export_has_no_entity_id_table_and_reproduces_gnn_scores(tmp_path):
    from yago.inductive_export import export_encoder
    model = InductiveAutoencoder(1000, 2, dim=8, base_dim=4, hidden_dim=12, encoder="tf-gnns")
    context = NeighborhoodGraph(np.asarray([[0, 0, 1], [1, 1, 2]]), 3, 2)
    _, graph = context.sample([0, 2], [10, 10], np.random.default_rng(1))
    tensors = {key: tf.convert_to_tensor(value) for key, value in graph.items()}
    triples = tf.constant([[0, 0, 2]], tf.int32)
    expected = model.predict_graph(tensors, triples).numpy()
    export_encoder(model, tmp_path / "gnn")
    restored = tf.saved_model.load(str(tmp_path / "gnn"))
    np.testing.assert_allclose(expected, restored.predict_graph(tensors, triples).numpy(), atol=1e-7)
    states = restored.encode(tensors)
    cached_score = restored.score_embeddings(tf.gather(states, [0]), tf.constant([0], tf.int32), tf.gather(states, [2]))
    np.testing.assert_allclose(expected, cached_score.numpy(), atol=1e-7)
    assert all(tuple(variable.shape) != (1000, 4) for variable in restored.encoder_variables)


def test_cpu_table_gather_operations_are_explicitly_cpu_placed():
    model = InductiveAutoencoder(100, 2, dim=8, base_dim=4, hidden_dim=12, embedding_device="cpu")

    @tf.function(input_signature=[tf.TensorSpec((None, 3), tf.int32)])
    def scores(triples):
        return model.shallow_scores(triples)

    graph = scores.get_concrete_function().graph
    gathers = [operation for operation in graph.get_operations() if operation.type in ("GatherV2", "ResourceGather")]
    assert gathers
    assert all("CPU:0" in operation.device for operation in gathers)
    assert all(operation.type == "ResourceGather" for operation in gathers)


def test_legacy_interrupted_step_counter_requires_optimizer_agreement():
    from yago.inductive_train import reconcile_completed_step
    assert reconcile_completed_step(5, 5, 5) == 5
    assert reconcile_completed_step(5, 6, 6) == 6
    with pytest.raises(ValueError, match="inconsistent checkpoint"):
        reconcile_completed_step(5, 6, 5)


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
