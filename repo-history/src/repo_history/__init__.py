"""Temporal repository-history indexing framework.

The package separates source evidence, normalized events, derived analysis, and
GestaltDB persistence. Importing the package does not require model credentials
or an open database.
"""

from .config import RuntimeConfig, load_config
from .events import EventKind, IngestEvent
from .models import Budgets, CoverageGap, CoverageStatus, RepositorySpec, TimeWindow

__all__ = [
    "Budgets",
    "CoverageGap",
    "CoverageStatus",
    "EventKind",
    "IngestEvent",
    "RepositorySpec",
    "RuntimeConfig",
    "TimeWindow",
    "load_config",
]
