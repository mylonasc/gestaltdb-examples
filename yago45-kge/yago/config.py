"""YAGO 4.5 KGE config."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class PathsConfig:
    project_root: Path = Path(__file__).resolve().parents[2]
    data_root: Path | None = None
    tiny_zip: Path | None = None
    full_zip: Path | None = None
    ttl_cache: Path | None = None
    parquet_dir: Path | None = None
    artifacts_dir: Path | None = None
    triples_parquet: Path | None = None
    triples_train_path: Path | None = None
    triples_valid_path: Path | None = None
    triples_test_path: Path | None = None
    node_map_path: Path | None = None
    relation_map_path: Path | None = None
    edge_arrays_path: Path | None = None
    sampler_snapshot_path: Path | None = None
    db_path: Path | None = None
    metadata_path: Path | None = None
    embeddings_path: Path | None = None
    checkpoints_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.data_root is None:
            object.__setattr__(self, "data_root", self.project_root / "data" / "yago45")
        dr = self.data_root
        defaults: dict[str, Path] = {
            "tiny_zip": dr / "yago-4.5.0.2-tiny.zip",
            "full_zip": dr / "yago-4.5.0.2.zip",
            "ttl_cache": dr / "ttl",
            "parquet_dir": dr / "parquet_shards",
            "artifacts_dir": dr / "artifacts",
            "triples_parquet": dr / "artifacts" / "triples_train.parquet",
            "triples_train_path": dr / "artifacts" / "triples_train.parquet",
            "triples_valid_path": dr / "artifacts" / "triples_valid.parquet",
            "triples_test_path": dr / "artifacts" / "triples_test.parquet",
            "node_map_path": dr / "artifacts" / "node_map.parquet",
            "relation_map_path": dr / "artifacts" / "relation_map.parquet",
            "edge_arrays_path": dr / "artifacts" / "edge_arrays.npz",
            "sampler_snapshot_path": dr / "artifacts" / "sampler_snapshot",
            "db_path": dr / "gestaltdb_yago45_rocksdb",
            "metadata_path": dr / "artifacts" / "metadata.json",
            "embeddings_path": dr / "artifacts" / "yago45_embeddings_200d.npz",
            "checkpoints_dir": dr / "artifacts" / "checkpoints",
        }
        for k, v in defaults.items():
            if getattr(self, k) is None:
                object.__setattr__(self, k, v)


@dataclass(slots=True)
class SubsetConfig:
    # Entity-entity predicate blocklist (labels, literals, identifiers).
    skip_predicates: tuple[str, ...] = (
        "rdfs:label", "rdfs:comment", "schema:alternateName",
        "schema:mainEntityOfPage", "schema:image", "schema:url",
        "schema:sameAs", "owl:sameAs", "skos:prefLabel",
        "schema:geo", "schema:birthDate", "schema:deathDate",
        "schema:dateCreated", "schema:postalCode",
    )
    skip_prefixes: tuple[str, ...] = ("sh:", "ys:", "prov:", "cc:", "dct:")
    min_relation_edges: int = 5
    max_relations: int | None = None
    target_edges: int | None = None  # None = keep all entity-entity triples
    max_edges_per_relation: int | None = None  # stratified cap per relation
    valid_size: int = 10_000  # held-out validation triples (link-prediction split)
    test_size: int = 10_000  # held-out test triples
    seed: int = 13


@dataclass(slots=True)
class IngestionConfig:
    reset_db: bool = False
    chunk_size: int = 250_000
    write_buffer_size: int = 256 * 1024 * 1024
    parallelism: int = 8
    max_background_jobs: int = 8
    progress: bool = True


@dataclass(slots=True)
class SamplerConfig:
    batch_size: int = 64
    negatives_per_positive: int = 8
    hop1_fanout: int = 12
    hop2_fanout: int = 8
    negative_candidate_fanout: int = 24
    bfs_direction: str = "any"
    hard_negative_head_probability: float = 0.5
    same_endpoint_type_negatives: bool = True
    engine_negative_source: str = "context_relation_neighbors"
    max_negative_retries: int = 64
    seed: int = 13
    num_workers: int = 0  # 0 = single-threaded; >0 = ThreadPool feeder threads
    worker_prefetch: int = 2  # batches prefetched per worker
    fast_negatives: bool = False  # numpy random corruptions (no engine rejection)
    fast_negative_global_fraction: float = 0.5  # share drawn globally vs in-subgraph


@dataclass(slots=True)
class ModelConfig:
    entity_dim: int = 200  # REQUIRED output dim for entities
    relation_dim: int = 200  # relation embeddings also 200-d
    node_type_dim: int = 32
    hidden_dim: int = 256
    mp_steps: int = 2
    dropout: float = 0.0
    aggregation_function: str = "mean"
    temperature: float = 0.1


@dataclass(slots=True)
class TrainingConfig:
    steps: int = 5_000
    learning_rate: float = 1e-4
    log_every: int = 50
    prefetch: int = 4
    xla: bool = False
    eval_every: int = 1_000
    eval_triples: int = 2_000
    eval_negatives: int = 50
    smoke_test: bool = False


@dataclass(slots=True)
class AppConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    subset: SubsetConfig = field(default_factory=SubsetConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def make_config(smoke_test: bool = False) -> AppConfig:
    cfg = AppConfig()
    if smoke_test:
        cfg.sampler.batch_size = 8
        cfg.sampler.negatives_per_positive = 2
        cfg.sampler.hop1_fanout = 4
        cfg.sampler.hop2_fanout = 2
        cfg.model.hidden_dim = 64
        cfg.model.mp_steps = 1
        cfg.training.steps = 10
        cfg.training.log_every = 2
        cfg.training.prefetch = 1
        cfg.training.smoke_test = True
    return cfg
