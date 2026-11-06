import numpy as np
import pytest

from yago.inductive_data import GraphKnownObjects, NeighborhoodGraph, known_objects, sampled_candidates, split_entities, zero_relation_degree_candidates


def example_split():
    edges = np.asarray([(s, (s + o) % 2, o) for s in range(20) for o in range(20) if s != o])
    return split_entities(edges, 20, 0.25, 12, 15, 13)


def test_unseen_entities_are_absent_from_every_training_endpoint():
    data = example_split()
    unseen = set(data["unseen_nodes"])
    for key in ("train", "valid"):
        assert not unseen.intersection(data[key][:, [0, 2]].ravel())
    assert all(s in unseen or o in unseen for s, _, o in data["test"])
    assert data["train"][:, [0, 2]].max() < data["num_train_nodes"]
    train = set(map(tuple, data["train"]))
    assert not train.intersection(map(tuple, data["support"]))
    assert not train.intersection(map(tuple, data["test"]))
    support_nodes = set(data["support"][:, [0, 2]].ravel())
    assert set(data["test"][:, [0, 2]].ravel()).issubset(support_nodes)


@pytest.mark.parametrize("feature_mode", ["sampled", "full"])
def test_masking_happens_before_features_and_neighborhood_expansion(feature_mode):
    edges = np.asarray([[0, 0, 1], [1, 0, 0], [0, 1, 2]])
    graph = NeighborhoodGraph(edges, 3, 2, feature_mode=feature_mode)
    ids, batch = graph.sample([0], [10, 10], np.random.default_rng(1), masked=[[0, 0, 1]])
    assert ids.tolist() == [0, 2]
    assert not np.any(batch["relations"] % 2 == 0)
    assert not np.any(batch["features"][:, [0, 2]])


def test_isolated_unseen_node_has_valid_identity_free_features():
    graph = NeighborhoodGraph(np.asarray([[0, 0, 1]]), 3, 1)
    ids, batch = graph.sample([2], [2, 2], np.random.default_rng(1))
    assert ids.tolist() == [2]
    np.testing.assert_array_equal(batch["features"], [[0, 0, 1]])
    assert batch["senders"].size == 0


def test_candidate_filter_keeps_target_and_removes_other_known_facts():
    query = np.asarray([0, 1, 3])
    known = known_objects(np.asarray([[0, 1, 2], query]))
    candidates = sampled_candidates(query, np.arange(5), 5, known, np.random.default_rng(1))
    assert candidates[0] == 3
    assert 2 not in candidates
    assert len(np.unique(candidates)) == len(candidates)


@pytest.mark.parametrize("compact", [False, True])
def test_gestaltdb_csr_uses_edge_ids_not_positions(tmp_path, compact):
    pytest.importorskip("gestaltdb")
    edges = np.asarray([[5, 0, 4], [1, 0, 3], [0, 1, 2], [3, 1, 4]])
    graph = NeighborhoodGraph(edges, 6, 2, snapshot_path=tmp_path / "snapshot", compact=compact)
    ids, batch = graph.sample([0], [10], np.random.default_rng(1))
    assert ids.tolist() == [0, 2]
    assert batch["senders"].max() < len(ids)
    assert batch["receivers"].max() < len(ids)


def test_csr_positive_index_matches_materialized_sets():
    edges = np.asarray([[0, 0, 2], [0, 0, 3], [2, 1, 0], [1, 0, 3]])
    extra = np.asarray([[0, 0, 1]])
    graph = NeighborhoodGraph(edges, 4, 2)
    index = GraphKnownObjects(graph, extra)
    expected = known_objects(edges, extra)
    for s in range(4):
        for r in range(2):
            assert index.get((s, r)) == expected.get((s, r), set())


def test_full_statistics_mask_a_query_not_present_in_context_without_subtracting():
    edges = np.asarray([[0, 0, 1], [0, 0, 2]])
    graph = NeighborhoodGraph(edges, 4, 1, feature_mode="full")
    ids, batch = graph.sample([0, 3], [1], np.random.default_rng(1), masked=[[0, 0, 3]])
    assert batch["features"][np.searchsorted(ids, 0), 0] == np.float32(np.log1p(2))
    assert batch["features"][np.searchsorted(ids, 3), 1] == 0


def test_hub_sampling_masks_before_expansion_and_remains_uniform():
    edges = np.asarray([[0, 0, other] for other in range(1, 1001)])
    graph = NeighborhoodGraph(edges, 1001, 1, sampling_version=2)
    rng = np.random.default_rng(13)
    draws = []
    for _ in range(300):
        ids, _ = graph.sample([0], [4], rng, masked=[[0, 0, 1]])
        assert len(ids) == 5
        assert 1 not in ids
        draws.extend(ids[ids != 0])
    assert abs(np.mean(draws) - 501) < 40


def test_direct_csr_position_draw_preserves_original_rng_stream(tmp_path):
    pytest.importorskip("gestaltdb")
    edges = np.asarray([[0, 0, i] for i in range(1, 101)] + [[i, 1, 0] for i in range(1, 51)])
    graph = NeighborhoodGraph(edges, 102, 2, snapshot_path=tmp_path / "csr", compact=True)
    old_rng = np.random.default_rng(13)
    new_rng = np.random.default_rng(13)
    for node in (0, 1, 33, 101):
        all_edges = graph.incident(node)
        expected = old_rng.choice(all_edges, 7, replace=False) if len(all_edges) > 7 else all_edges
        actual, total = graph.draw_incident(node, 7, new_rng)
        np.testing.assert_array_equal(actual, expected)
        assert total == len(all_edges)
    assert old_rng.bit_generator.state == new_rng.bit_generator.state


def test_static_cache_scrubs_queries_and_reverse_edges():
    from yago.inductive_cache import scrub_queries
    edges = np.asarray([[0, 0, 1], [1, 0, 0], [1, 1, 2]])
    got = scrub_queries(edges, [np.asarray([[0, 0, 1]])], 3, 2)
    np.testing.assert_array_equal(got, [[1, 1, 2]])


def test_zero_degree_controls_do_not_inject_the_only_zero_degree_object():
    query = np.asarray([7, 0, 5])
    candidates = zero_relation_degree_candidates(query, np.arange(8), np.asarray([1, 2, 3]),
        3, {(7, 0): {0, 5}}, np.random.default_rng(13))
    assert candidates[0] == 5
    assert set(candidates[1:]) == {4, 6, 7}


def test_plain_csr_views_preserve_memmap_backing_and_exact_rng_stream(tmp_path):
    from types import SimpleNamespace
    from gestaltdb.sampling.adjacency import CSRAdjacency
    edges = np.asarray([[0, 0, i] for i in range(1, 81)] + [[i, 1, 0] for i in range(1, 40)])
    graph = NeighborhoodGraph(edges, 82, 2, snapshot_path=tmp_path / "csr", compact=True, feature_mode="full")
    assert type(graph.snapshot.out.indptr) is np.ndarray
    assert isinstance(graph.snapshot.out.indptr.base, np.memmap)
    path = next((tmp_path / "csr").iterdir())
    legacy = NeighborhoodGraph(edges, 82, 2, snapshot_path=tmp_path / "csr", compact=True, feature_mode="full")
    legacy.snapshot = SimpleNamespace(**{name: CSRAdjacency(
        np.load(path / f"{direction}_indptr.npy", mmap_mode="r"),
        np.load(path / f"{direction}_edge_indices.npy", mmap_mode="r"))
        for name, direction in (("out", "out"), ("in_", "in"))})
    old_rng, new_rng = np.random.default_rng(13), np.random.default_rng(13)
    for _ in range(10):
        expected_ids, expected_graph = legacy.sample([0, 1, 81], [8, 4, 2], old_rng, masked=[[0, 0, 1]])
        ids, actual = graph.sample([0, 1, 81], [8, 4, 2], new_rng, masked=[[0, 0, 1]])
        np.testing.assert_array_equal(ids, expected_ids)
        for key in actual:
            np.testing.assert_array_equal(actual[key], expected_graph[key])
    assert old_rng.bit_generator.state == new_rng.bit_generator.state


@pytest.mark.parametrize("compact", [False, True])
def test_vectorized_sampler_masks_before_expansion_and_is_uniform(tmp_path, compact):
    edges = np.asarray([[0, 0, i] for i in range(1, 1001)])
    graph = NeighborhoodGraph(edges, 1001, 1, sampling_version=3,
                              snapshot_path=tmp_path / "csr" if compact else None, compact=compact)
    rng = np.random.default_rng(13)
    draws = []
    for _ in range(300):
        ids, batch = graph.sample([0], [4], rng, masked=[[0, 0, 1]])
        assert len(ids) == 5 and 1 not in ids
        assert len(batch["senders"]) == 8
        draws.extend(ids[ids != 0])
    assert abs(np.mean(draws) - 501) < 40


def test_vectorized_sampler_matches_full_fanout_and_handles_duplicate_masked_edges(tmp_path):
    edges = np.asarray([[0, 0, 1]] * 12 + [[0, 0, i] for i in range(2, 7)] + [[6, 1, 7]])
    graph = NeighborhoodGraph(edges, 9, 2, sampling_version=3, feature_mode="full",
                              snapshot_path=tmp_path / "csr", compact=True)
    for fans in ([4], [100, 100, 100]):
        ids, batch = graph.sample([0, 8], fans, np.random.default_rng(1), masked=[[0, 0, 1]])
        assert 1 not in ids and 8 in ids
        assert np.all(np.isfinite(batch['features']))
        assert len(ids) >= 6
    graph.sampling_version = 2
    expected_ids, expected = graph.sample([0, 8], [100, 100, 100], np.random.default_rng(1), masked=[[0, 0, 1]])
    graph.sampling_version = 3
    ids, actual = graph.sample([0, 8], [100, 100, 100], np.random.default_rng(1), masked=[[0, 0, 1]])
    np.testing.assert_array_equal(ids, expected_ids)
    for key in actual:
        np.testing.assert_array_equal(actual[key], expected[key])


def test_vectorized_sampler_does_not_bias_small_masked_adjacencies():
    graph = NeighborhoodGraph(np.asarray([[0, 0, i] for i in range(1, 6)]), 6, 1, sampling_version=3)
    rng = np.random.default_rng(13)
    counts = np.zeros(6, dtype=np.int64)
    for _ in range(300):
        ids, _ = graph.sample([0], [3], rng, masked=[[0, 0, 1]])
        counts[ids] += 1
    assert counts[1] == 0
    assert all(185 < count < 265 for count in counts[2:])
