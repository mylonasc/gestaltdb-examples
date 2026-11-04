import numpy as np

from yago.sampler import SemanticPairIndex, mask_supervised_message_edges
from yago.scoring import filtered_object_candidates, resolve_score_function, score_objects
from yago.config import make_config
from yago.train import checkpoint_number, load_curves_through_step, write_curves


def test_quality_defaults_are_scorer_aligned():
    cfg = make_config()
    assert cfg.model.score_function == "complex"
    assert cfg.model.aux_shallow_weight > 0
    assert cfg.sampler.negative_sampler == "hybrid"
    assert cfg.sampler.mask_supervised_edges


def test_masks_supervised_and_reverse_message_edges():
    senders = np.array([0, 1, 0, 2, 3])
    receivers = np.array([1, 0, 2, 0, 4])
    relations = np.array([7, 7, 8, 9, 7])
    s, o, r = mask_supervised_message_edges(
        senders, receivers, relations,
        np.array([0]), np.array([1]), np.array([7]), mask_reverse=True)
    assert list(zip(s, r, o)) == [(0, 8, 2), (2, 9, 0), (3, 7, 4)]


def test_filtered_candidates_remove_all_other_known_objects():
    known = {(2, 3): {5, 7, 11}}
    got = filtered_object_candidates(2, 3, 5, np.array([7, 9, 11, 9, 5, 13]), known)
    assert got.tolist() == [5, 9, 13]


def test_distmult_scores_candidates():
    E = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    R = np.array([[2, 1]], dtype=np.float32)
    got = score_objects(E, R, 0, 0, np.array([1, 2]), "distmult")
    np.testing.assert_allclose(got, [14, 22])


def test_scorer_metadata_cannot_silently_disagree():
    assert resolve_score_function(None, "complex") == "complex"
    assert resolve_score_function("complex", "complex") == "complex"
    try:
        resolve_score_function("distmult", "complex")
    except ValueError as exc:
        assert "disagrees" in str(exc)
    else:
        raise AssertionError("mismatched scorer was accepted")


def test_semantic_pairs_share_value_and_negatives_do_not():
    index = SemanticPairIndex.__new__(SemanticPairIndex)
    # Three contiguous attribute groups: {10,11,12}, {20,21}, {30,31,32,33}.
    src = np.array([10, 11, 12, 20, 21, 30, 31, 32, 33])
    starts = np.array([0, 3, 5]); counts = np.array([3, 2, 4])
    index.relations = [("attribute", src, starts, counts, np.ones(3) / 3)]
    anchor, positive, negative = index.sample(np.random.default_rng(3), 100, 5)
    group = {v: i for i, values in enumerate(([10, 11, 12], [20, 21], [30, 31, 32, 33])) for v in values}
    assert all(group[int(a)] == group[int(p)] and a != p for a, p in zip(anchor, positive))
    assert all(group[int(a)] != group[int(n)] for a, row in zip(anchor, negative) for n in row)


def test_resume_curves_are_trimmed_to_checkpoint(tmp_path):
    path = tmp_path / "curves.jsonl"
    write_curves(path, [{"step": 0}, {"step": 100}, {"step": 200}])
    assert load_curves_through_step(path, 100) == [{"step": 0}, {"step": 100}]


def test_checkpoint_number_uses_tensorflow_suffix():
    assert checkpoint_number("/run/checkpoints/ckpt-12000") == 12_000
