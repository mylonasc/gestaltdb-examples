"""NumPy scoring and candidate filtering shared by export, eval and notebooks."""
from __future__ import annotations

import numpy as np


def resolve_score_function(requested: str | None, artifact_score: str | None) -> str:
    """Resolve scorer metadata without permitting a silent train/eval mismatch."""
    if requested and artifact_score and requested != artifact_score:
        raise ValueError(
            f"requested score_function={requested} disagrees with artifact={artifact_score}")
    score = requested or artifact_score
    if not score:
        raise ValueError("artifact has no scorer metadata; pass a score function explicitly")
    return score


def score_objects(E: np.ndarray, R: np.ndarray, subject: int, relation: int,
                  candidates: np.ndarray, score_function: str) -> np.ndarray:
    """Score ``(subject, relation, candidate)`` triples."""
    candidates = np.asarray(candidates, dtype=np.int64)
    if score_function == "complex":
        if E.shape[1] % 2:
            raise ValueError("ComplEx requires an even embedding dimension")
        d = E.shape[1] // 2
        s_re, s_im = E[subject, :d], E[subject, d:]
        r_re, r_im = R[relation, :d], R[relation, d:]
        o_re, o_im = E[candidates, :d], E[candidates, d:]
        return np.sum(
            r_re * (s_re * o_re + s_im * o_im)
            + r_im * (s_re * o_im - s_im * o_re), axis=1)
    if score_function == "distmult":
        return np.sum(E[subject] * R[relation] * E[candidates], axis=1)
    raise ValueError(f"unknown score function: {score_function}")


def filtered_object_candidates(subject: int, relation: int, true_object: int,
                               sampled: np.ndarray,
                               known_objects: dict[tuple[int, int], set[int]]) -> np.ndarray:
    """Put the target first and discard other known-positive objects."""
    known = known_objects.get((int(subject), int(relation)), set())
    kept = [int(true_object)]
    seen = {int(true_object)}
    for candidate in sampled:
        candidate = int(candidate)
        if candidate in seen or candidate in known:
            continue
        kept.append(candidate)
        seen.add(candidate)
    return np.asarray(kept, dtype=np.int64)
