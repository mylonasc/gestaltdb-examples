"""Entity-held-out splits and ID-independent, masked neighborhood sampling."""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mapping_digest(data):
    digest = hashlib.sha256()
    for key in ("node_ids", "relation_ids"):
        for name in data[key]:
            digest.update(str(name).encode())
            digest.update(b"\0")
    return digest.hexdigest()


def hold_out_edges(edges, count, rng):
    """Hold out unique facts while retaining support at both endpoints."""
    degree = np.bincount(edges[:, [0, 2]].ravel(), minlength=int(edges[:, [0, 2]].max()) + 1)
    keep = np.ones(len(edges), dtype=bool)
    selected = []
    for i in rng.permutation(len(edges)):
        if len(selected) >= count:
            break
        s, _, o = edges[i]
        needed = 2 if s == o else 1
        if degree[s] > needed and degree[o] > needed:
            degree[s] -= 1
            degree[o] -= 1
            keep[i] = False
            selected.append(i)
    return edges[keep], edges[np.asarray(selected, dtype=np.int64)]


def split_entities(edges, num_nodes, fraction, valid_size, test_size, seed):
    """Exclude every edge incident to an unseen entity from optimization."""
    if not 0 < fraction < 1:
        raise ValueError("unseen fraction must be between zero and one")
    edges = np.unique(np.asarray(edges, dtype=np.int64), axis=0)
    rng = np.random.default_rng(seed)
    unseen = rng.choice(num_nodes, max(1, int(num_nodes * fraction)), replace=False)
    held = np.zeros(num_nodes, dtype=bool)
    held[unseen] = True
    incident = held[edges[:, 0]] | held[edges[:, 2]]
    train_pool = edges[~incident]
    if not len(train_pool):
        raise ValueError("entity split left no training edges; reduce --unseen-fraction")
    # Only relations actually observed in training can be inferred at test time.
    relations = np.unique(train_pool[:, 1])
    inductive_pool = edges[incident & np.isin(edges[:, 1], relations)]
    if not len(inductive_pool):
        raise ValueError("no inductive facts with training-observed relations")
    support, test = hold_out_edges(inductive_pool, test_size, rng)
    train, valid = hold_out_edges(train_pool, valid_size, rng)
    # Avoid withholding the last training example of a relation.
    missing = ~np.isin(valid[:, 1], np.unique(train[:, 1]))
    train = np.concatenate([train, valid[missing]])
    valid = valid[~missing]
    train_nodes = np.unique(train[:, [0, 2]])
    # Reindex seen nodes first: the shallow table contains *only* training nodes.
    order = np.r_[train_nodes, np.setdiff1d(np.arange(num_nodes), train_nodes)]
    inverse = np.empty(num_nodes, dtype=np.int64)
    inverse[order] = np.arange(num_nodes)

    def remap(rows):
        rows = rows.copy()
        rows[:, 0] = inverse[rows[:, 0]]
        rows[:, 2] = inverse[rows[:, 2]]
        return rows

    if not len(valid) or not len(test):
        raise ValueError("insufficient supported edges for nonempty validation and inductive test splits")
    return dict(train=remap(train), valid=remap(valid), support=remap(support),
                test=remap(test), node_order=order, num_train_nodes=len(train_nodes),
                unseen_nodes=np.arange(len(train_nodes), num_nodes, dtype=np.int64),
                heldout_nodes=inverse[unseen])


class NeighborhoodGraph:
    """Compact bidirectional CSR with explicit inverse relation channels.

    Features are directional relation counts computed AFTER supervised-edge
    masking, from the sampled context only. No ID, text or target fact is used.
    All YAGO nodes currently have the same generic Entity type.
    """

    def __init__(self, edges, num_nodes, num_relations, snapshot_path=None, feature_mode="sampled", compact=False,
                 sampling_version=2):
        self.edges = np.asarray(edges, dtype=np.int64)
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.feature_mode = feature_mode
        self.sampling_version = sampling_version
        s, r, o = self.edges.T
        if feature_mode == "full":
            from scipy.sparse import coo_matrix
            # O(observed node/relation pairs), not O(nodes * relations): the
            # full graph has millions of nodes but very sparse histograms.
            self.counts = coo_matrix((np.ones(2 * len(edges), dtype=np.float32),
                                      (np.r_[s, o], np.r_[r, r + num_relations])),
                                     shape=(num_nodes, 2 * num_relations)).tocsr()
        elif feature_mode != "sampled":
            raise ValueError(f"unknown feature mode {feature_mode}")
        self.src = np.r_[s, o]
        self.dst = np.r_[o, s]
        self.rel = np.r_[r, r + num_relations]
        self.snapshot = None
        if snapshot_path is not None:
            from pathlib import Path
            digest = hashlib.sha256(str((num_nodes, num_relations)).encode())
            for start in range(0, len(edges), 1000000):
                digest.update(np.ascontiguousarray(self.edges[start:start + 1000000]).tobytes())
            fingerprint = digest.hexdigest()[:16]
            path = Path(snapshot_path) / fingerprint
            if compact:
                # Only directional CSR is used by pre-masked sampling. The full
                # SamplerSnapshot also writes external-ID lookups, membership,
                # relation indexes and weights, which are unnecessary here and
                # can exceed available disk on hundred-million-edge graphs.
                from gestaltdb.sampling.adjacency import CSRAdjacency, build_csr
                path.mkdir(parents=True, exist_ok=True)
                if not (path / "metadata.json").exists():
                    edge_ids = np.arange(len(edges), dtype=np.int64)
                    for direction, keys in (("out", s), ("in", o)):
                        csr = build_csr(keys, edge_ids, num_nodes)
                        np.save(path / f"{direction}_indptr.npy", csr.indptr, allow_pickle=False)
                        np.save(path / f"{direction}_edge_indices.npy", csr.edge_indices, allow_pickle=False)
                        del csr
                    (path / "metadata.json").write_text(json.dumps(dict(
                        backend="gestaltdb.sampling.adjacency", digest=fingerprint,
                        nodes=num_nodes, edges=len(edges), complete=True)))
                adjacencies = {}
                for direction in ("out", "in"):
                    adjacencies[direction] = CSRAdjacency(
                        np.asarray(np.load(path / f"{direction}_indptr.npy", mmap_mode="r")),
                        np.asarray(np.load(path / f"{direction}_edge_indices.npy", mmap_mode="r")))
                self.snapshot = SimpleNamespace(out=adjacencies["out"], in_=adjacencies["in"])
                return
            from gestaltdb.sampling import SamplerSnapshot
            if not (path / "metadata.json").exists():
                SamplerSnapshot.from_edge_arrays(
                    path, external_node_ids=np.arange(num_nodes).astype(str),
                    external_edge_ids=np.arange(len(edges)).astype(str),
                    external_relation_ids=np.arange(num_relations).astype(str),
                    src_int=s, dst_int=o, rel_int=r,
                    node_type_ids=np.zeros(num_nodes, dtype=np.int64),
                    metadata={"source": "inductive YAGO masked graph AE", "edge_digest": fingerprint})
            self.snapshot = SamplerSnapshot.load(path, mmap=True)
            return
        order = np.argsort(self.src, kind="stable")
        self.src, self.dst, self.rel = self.src[order], self.dst[order], self.rel[order]
        self.offsets = np.r_[0, np.cumsum(np.bincount(self.src, minlength=num_nodes))]

    def incident(self, node):
        if self.snapshot is None:
            return np.arange(self.offsets[node], self.offsets[node + 1])
        return np.r_[self.snapshot.out.edge_range(node), self.snapshot.in_.edge_range(node) + len(self.edges)]

    def outgoing(self, node):
        if self.snapshot is not None:
            return self.snapshot.out.edge_range(node)
        candidates = self.incident(node)
        return candidates[self.rel[candidates] < self.num_relations]

    def draw_incident(self, node, count, rng):
        """Sample CSR positions without copying a hub's entire adjacency."""
        if self.snapshot is None:
            candidates = self.incident(node)
            total = len(candidates)
            return (rng.choice(candidates, count, replace=False) if total > count else candidates), total
        out = self.snapshot.out.edge_range(node)
        incoming = self.snapshot.in_.edge_range(node)
        total = len(out) + len(incoming)
        positions = rng.choice(total, count, replace=False) if total > count else np.arange(total)
        selected = np.empty(len(positions), dtype=np.int64)
        forward = positions < len(out)
        selected[forward] = out[positions[forward]]
        selected[~forward] = incoming[positions[~forward] - len(out)] + len(self.edges)
        return selected, total

    def sample(self, roots, fanouts, rng, masked=()):
        blocked = set()
        for s, r, o in masked:
            # Remove exact and same-relation reverse copies, before expansion.
            blocked.add((int(s), int(r), int(o)))
            blocked.add((int(o), int(r), int(s)))
        blocked_by_node = {}
        for s, r, o in blocked:
            blocked_by_node.setdefault(s, set()).add((r, o))
        nodes = set(map(int, roots))
        frontier = set(nodes)
        chosen = set()
        if self.sampling_version == 3:
            nodes, chosen = self.vectorized_neighborhood(roots, fanouts, rng, blocked_by_node)
        for fanout in (() if self.sampling_version == 3 else fanouts):
            following = set()
            for node in sorted(frontier):
                if self.sampling_version == 2:
                    bad = blocked_by_node.get(node, set())
                    # Sample a uniform superset before rejecting labels. This
                    # avoids a Python scan of every edge of million-degree hubs.
                    # Rejection still precedes expansion and feature creation.
                    candidates, total = self.draw_incident(node, fanout + 2 * len(bad), rng)
                    draw_count = len(candidates)
                    if bad:
                        candidates = np.asarray([i for i in candidates
                            if (int(self.rel[i] % self.num_relations), int(self.dst[i])) not in bad], dtype=np.int64)
                        if len(candidates) < fanout and total > draw_count:
                            # Fallback for a non-deduplicated caller graph.
                            candidates = np.asarray([i for i in self.incident(node)
                                if (int(self.rel[i] % self.num_relations), int(self.dst[i])) not in bad], dtype=np.int64)
                            if len(candidates) > fanout:
                                candidates = rng.choice(candidates, fanout, replace=False)
                    candidates = candidates[:fanout]
                else:
                    candidates = self.incident(node)
                    if blocked:
                        candidates = np.asarray([
                            i for i in candidates
                            if (node, int(self.rel[i] % self.num_relations), int(self.dst[i])) not in blocked
                        ], dtype=np.int64)
                if self.sampling_version == 1 and len(candidates) > fanout:
                    candidates = rng.choice(candidates, fanout, replace=False)
                for i in candidates:
                    chosen.add(int(i))
                    following.add(int(self.dst[i]))
            frontier = following - nodes
            nodes.update(following)
        ids = np.asarray(nodes if self.sampling_version == 3 else sorted(nodes), dtype=np.int64)
        indices = np.asarray(chosen if self.sampling_version == 3 else sorted(chosen), dtype=np.int64)
        s = np.searchsorted(ids, self.src[indices])
        o = np.searchsorted(ids, self.dst[indices])
        r = self.rel[indices]
        # Always include inverse messages, preserving direction in the channel.
        senders = np.r_[s, o]
        receivers = np.r_[o, s]
        relations = np.r_[r, (r + self.num_relations) % (2 * self.num_relations)]
        # A packed int64 key preserves exact lexicographic edge order without
        # NumPy's expensive axis=0 structured-array row sort.
        if len(ids) ** 2 * (2 * self.num_relations) >= np.iinfo(np.int64).max:
            raise ValueError("sampled edge key exceeds int64 range")
        keys = np.unique((senders * (2 * self.num_relations) + relations) * len(ids) + receivers)
        receivers = keys % len(ids)
        directional = keys // len(ids)
        relations = directional % (2 * self.num_relations)
        senders = directional // (2 * self.num_relations)
        features = np.zeros((len(ids), 2 * self.num_relations + 1), dtype=np.float32)
        features[:, -1] = 1.0  # shared generic Entity type / isolated-node feature
        if self.feature_mode == "full":
            features[:, :-1] = self.counts[ids].toarray()
            # Mask statistics as well as message edges: labels must not be
            # recoverable from full-degree features of their endpoints.
            for s, r, o in blocked:
                incident = self.outgoing(s)
                multiplicity = np.sum((self.rel[incident] == r) & (self.dst[incident] == o))
                if not multiplicity:
                    continue
                for node, channel in ((s, r), (o, r + self.num_relations)):
                    index = np.searchsorted(ids, node)
                    if index < len(ids) and ids[index] == node:
                        features[index, channel] -= multiplicity
        else:
            np.add.at(features, (senders, relations), 1)
        features[:, :-1] = np.log1p(features[:, :-1])
        return ids, dict(features=features, senders=senders.astype(np.int32),
                         receivers=receivers.astype(np.int32), relations=relations.astype(np.int32))

    def vectorized_neighborhood(self, roots, fanouts, rng, blocked_by_node):
        """Uniform CSR draws without replacement, vectorized across each frontier.

        Version 3 preserves the distribution/masking contract, not the v2 RNG
        stream. No per-node adjacency copies or Python loop for unmasked draws.
        """
        nodes = np.unique(roots)
        frontier = nodes
        chosen = []
        for fanout in fanouts:
            if not len(frontier):
                break
            if self.snapshot is None:
                out_start = self.offsets[frontier]
                out_degree = self.offsets[frontier + 1] - out_start
                in_start = np.zeros(len(frontier), dtype=np.int64)
                in_degree = np.zeros(len(frontier), dtype=np.int64)
            else:
                out_start = self.snapshot.out.indptr[frontier]
                out_degree = self.snapshot.out.indptr[frontier + 1] - out_start
                in_start = self.snapshot.in_.indptr[frontier]
                in_degree = self.snapshot.in_.indptr[frontier + 1] - in_start
            degree = out_degree + in_degree
            requested = np.full(len(frontier), fanout, dtype=np.int64)
            masked_rows = []
            for node, bad in blocked_by_node.items():
                row = np.searchsorted(frontier, node)
                if row < len(frontier) and frontier[row] == node:
                    requested[row] += 2 * len(bad)
                    masked_rows.append(int(row))
            take = np.minimum(degree, requested)
            positions = np.full((len(frontier), int(take.max(initial=0))), -1, dtype=np.int64)
            random_rows = degree > requested
            for slot in range(positions.shape[1]):
                rows = np.flatnonzero(take > slot)
                positions[rows, slot] = slot  # enumerate low-degree rows, v2 semantics
                random = rows[random_rows[rows]]
                if len(random):
                    draws = rng.integers(degree[random])
                    if slot:
                        duplicates = np.any(positions[random, :slot] == draws[:, None], axis=1)
                        while np.any(duplicates):
                            draws[duplicates] = rng.integers(degree[random[duplicates]])
                            duplicates = np.any(positions[random, :slot] == draws[:, None], axis=1)
                    positions[random, slot] = draws
            rows, slots = np.nonzero(positions >= 0)
            draws = positions[rows, slots]
            if self.snapshot is None:
                candidates = out_start[rows] + draws
            else:
                forward = draws < out_degree[rows]
                candidates = np.empty(len(draws), dtype=np.int64)
                candidates[forward] = self.snapshot.out.edge_indices[out_start[rows[forward]] + draws[forward]]
                candidates[~forward] = self.snapshot.in_.edge_indices[
                    in_start[rows[~forward]] + draws[~forward] - out_degree[rows[~forward]]] + len(self.edges)
            # Only masked rows need Python membership checks (typically roots),
            # rather than every edge/node in each multi-hop frontier.
            allowed = np.ones(len(candidates), dtype=bool)
            for row in masked_rows:
                indices = np.flatnonzero(rows == row)
                bad = blocked_by_node[int(frontier[row])]
                allowed[indices] = [(int(self.rel[i] % self.num_relations), int(self.dst[i])) not in bad
                                    for i in candidates[indices]]
            rows, candidates = rows[allowed], candidates[allowed]
            counts = np.bincount(rows, minlength=len(frontier))
            within_row = np.arange(len(rows)) - np.repeat(np.cumsum(counts) - counts, counts)
            selected = candidates[within_row < fanout]
            # If the oversample enumerated a small adjacency, masking can leave
            # more than fanout valid edges. Randomly trim those rows rather than
            # taking their first CSR positions (which would bias the sample).
            for row in np.flatnonzero((degree <= requested) & (counts > fanout)):
                alternatives = candidates[rows == row]
                node = int(frontier[row])
                selected = np.r_[selected[self.src[selected] != node], rng.choice(alternatives, fanout, replace=False)]
            # Rare duplicate-edge fallback, matching the v2 rejection contract.
            for row in masked_rows:
                if counts[row] < fanout and degree[row] > take[row]:
                    node = int(frontier[row])
                    bad = blocked_by_node[node]
                    original = self.incident(node)
                    alternatives = np.asarray([i for i in original
                        if (int(self.rel[i] % self.num_relations), int(self.dst[i])) not in bad], dtype=np.int64)
                    if len(alternatives) > fanout:
                        alternatives = rng.choice(alternatives, fanout, replace=False)
                    selected = np.r_[selected[self.src[selected] != node], alternatives]
            chosen.append(selected)
            following = np.setdiff1d(np.unique(self.dst[selected]), nodes, assume_unique=True)
            nodes = np.union1d(nodes, following)
            frontier = following
        return nodes, np.unique(np.concatenate(chosen)) if chosen else np.empty(0, dtype=np.int64)


def sampled_candidates(query, pool, negatives, known, rng):
    s, r, o = map(int, query)
    draws = rng.choice(pool, min(len(pool), negatives), replace=False)
    return np.asarray([o] + [int(x) for x in draws
                            if x != o and int(x) not in known.get((s, r), set())], dtype=np.int64)


def zero_relation_degree_candidates(query, pool, observed_objects, negatives, known, rng):
    """Match a zero-incoming-degree target without materializing a huge complement."""
    s, r, o = map(int, query)
    candidates, seen = [o], {o}
    forbidden = known.get((s, r), set())
    for _ in range(64):
        draws = rng.choice(pool, min(len(pool), max(32, 2 * negatives)), replace=False)
        positions = np.searchsorted(observed_objects, draws)
        present = positions < len(observed_objects)
        present[present] = observed_objects[positions[present]] == draws[present]
        for candidate in draws[~present]:
            candidate = int(candidate)
            if candidate not in seen and candidate not in forbidden:
                candidates.append(candidate)
                seen.add(candidate)
                if len(candidates) >= negatives + 1:
                    return np.asarray(candidates, dtype=np.int64)
    return np.asarray(candidates, dtype=np.int64)


def known_objects(*parts):
    known = {}
    for rows in parts:
        for s, r, o in rows:
            known.setdefault((int(s), int(r)), set()).add(int(o))
    return known


class GraphKnownObjects:
    """CSR-backed positive filtering without a Python set per full-graph edge."""

    def __init__(self, graph, *extra_parts):
        self.graph = graph
        self.extra = known_objects(*extra_parts)
        self.cache = OrderedDict()

    def get(self, key, default=None):
        key = (int(key[0]), int(key[1]))
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        source, relation = key
        edges = self.graph.outgoing(source)
        objects = set(self.graph.dst[edges[self.graph.rel[edges] == relation]].tolist())
        objects.update(self.extra.get(key, ()))
        if len(objects) <= 2048:
            self.cache[key] = objects
            if len(self.cache) > 2048:
                self.cache.popitem(last=False)
        return objects

    def __getitem__(self, key):
        return self.get(key)
