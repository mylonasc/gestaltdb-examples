"""Entity-held-out splits and ID-independent, masked neighborhood sampling."""
from __future__ import annotations

import hashlib
from collections import OrderedDict

import numpy as np


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
                unseen_nodes=inverse[unseen])


class NeighborhoodGraph:
    """Compact bidirectional CSR with explicit inverse relation channels.

    Features are directional relation counts computed AFTER supervised-edge
    masking, from the sampled context only. No ID, text or target fact is used.
    All YAGO nodes currently have the same generic Entity type.
    """

    def __init__(self, edges, num_nodes, num_relations, snapshot_path=None, feature_mode="sampled"):
        self.edges = np.asarray(edges, dtype=np.int64)
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.feature_mode = feature_mode
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
            from gestaltdb.sampling import SamplerSnapshot
            fingerprint = hashlib.sha256(self.edges.tobytes() + str((num_nodes, num_relations)).encode()).hexdigest()[:16]
            path = Path(snapshot_path) / fingerprint
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

    def sample(self, roots, fanouts, rng, masked=()):
        blocked = set()
        for s, r, o in masked:
            # Remove exact and same-relation reverse copies, before expansion.
            blocked.add((int(s), int(r), int(o)))
            blocked.add((int(o), int(r), int(s)))
        nodes = set(map(int, roots))
        frontier = set(nodes)
        chosen = set()
        for fanout in fanouts:
            following = set()
            for node in sorted(frontier):
                candidates = self.incident(node)
                if blocked:
                    candidates = np.asarray([
                        i for i in candidates
                        if (node, int(self.rel[i] % self.num_relations), int(self.dst[i])) not in blocked
                    ], dtype=np.int64)
                if len(candidates) > fanout:
                    candidates = rng.choice(candidates, fanout, replace=False)
                for i in candidates:
                    chosen.add(int(i))
                    following.add(int(self.dst[i]))
            frontier = following - nodes
            nodes.update(following)
        ids = np.asarray(sorted(nodes), dtype=np.int64)
        indices = np.asarray(sorted(chosen), dtype=np.int64)
        s = np.searchsorted(ids, self.src[indices])
        o = np.searchsorted(ids, self.dst[indices])
        r = self.rel[indices]
        # Always include inverse messages, preserving direction in the channel.
        senders = np.r_[s, o]
        receivers = np.r_[o, s]
        relations = np.r_[r, (r + self.num_relations) % (2 * self.num_relations)]
        pairs = np.unique(np.stack([senders, relations, receivers], axis=1), axis=0)
        senders, relations, receivers = pairs.T
        features = np.zeros((len(ids), 2 * self.num_relations + 1), dtype=np.float32)
        features[:, -1] = 1.0  # shared generic Entity type / isolated-node feature
        if self.feature_mode == "full":
            features[:, :-1] = self.counts[ids].toarray()
            # Mask statistics as well as message edges: labels must not be
            # recoverable from full-degree features of their endpoints.
            for s, r, o in blocked:
                incident = self.incident(s)
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


def sampled_candidates(query, pool, negatives, known, rng):
    s, r, o = map(int, query)
    draws = rng.choice(pool, min(len(pool), negatives), replace=False)
    return np.asarray([o] + [int(x) for x in draws
                            if x != o and int(x) not in known.get((s, r), set())], dtype=np.int64)


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
        edges = self.graph.incident(source)
        objects = set(self.graph.dst[edges[self.graph.rel[edges] == relation]].tolist())
        objects.update(self.extra.get(key, ()))
        if len(objects) <= 2048:
            self.cache[key] = objects
            if len(self.cache) > 2048:
                self.cache.popitem(last=False)
        return objects

    def __getitem__(self, key):
        return self.get(key)
