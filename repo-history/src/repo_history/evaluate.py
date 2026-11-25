"""Release evaluation: timed runs, budgets, storage, and quality measures.

Helpers are pure and deterministic; the numbers they report come from the
caller's pipeline run, never from claims.  Quality is measured only on
annotated fixtures (``score_links`` from :mod:`linking` plus topic-decision
accuracy here); unmeasured accuracy is never asserted.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

from .budgets import BudgetLedger
from .models import CoverageGap, FrozenModel

__all__ = [
    "EvaluationReport",
    "Timer",
    "score_topic_decisions",
    "storage_bytes",
    "summarize_ledger",
]


class Timer:
    """Wall-clock timer used as ``with Timer() as timer: ...``."""

    def __init__(self) -> None:
        self.seconds: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.seconds = time.perf_counter() - self._start


def storage_bytes(path: str | Path) -> int:
    """Recursively sum file sizes; missing paths report zero."""

    root = Path(path)
    if root.is_file():
        return root.stat().st_size
    if not root.is_dir():
        return 0
    return sum(item.stat().st_size for item in root.rglob("*") if item.is_file())


def summarize_ledger(ledger: BudgetLedger) -> dict[str, int]:
    return dict(ledger.counters())


def score_topic_decisions(
    predicted: Sequence[tuple[str, str, str | None]],
    expected: Sequence[tuple[str, str, str | None]],
) -> dict[str, float]:
    """Accuracy of ``(subject, action, topic-or-name)`` triples on a fixture."""

    predicted_set = set(predicted)
    expected_set = set(expected)
    hits = len(predicted_set & expected_set)
    return {
        "accuracy": hits / len(expected_set) if expected_set else 1.0,
        "predicted": float(len(predicted_set)),
        "expected": float(len(expected_set)),
        "true_positives": float(hits),
    }


class EvaluationReport(FrozenModel):
    run_id: str
    repository: str
    synthetic: bool
    seconds: dict[str, float] = {}
    counts: dict[str, int] = {}
    budgets: dict[str, int] = {}
    storage: dict[str, int] = {}
    quality: dict[str, dict[str, float]] = {}
    gaps: tuple[CoverageGap, ...] = ()
    notes: tuple[str, ...] = ()

    def to_markdown(self) -> str:
        lines = [
            f"# Evaluation `{self.run_id}` "
            f"({'synthetic fixture' if self.synthetic else 'live slice'})",
            "",
            f"Repository: `{self.repository}`",
            "",
            "## Time (seconds)",
            "",
        ]
        for stage, seconds in sorted(self.seconds.items()):
            lines.append(f"- {stage}: {seconds:.2f}")
        lines += ["", "## Counts", ""]
        for key, value in sorted(self.counts.items()):
            lines.append(f"- {key}: {value}")
        lines += ["", "## Budgets consumed", ""]
        for key, value in sorted(self.budgets.items()):
            lines.append(f"- {key}: {value}")
        lines += ["", "## Storage (bytes)", ""]
        for key, value in sorted(self.storage.items()):
            lines.append(f"- {key}: {value}")
        lines += ["", "## Quality (annotated fixtures only)", ""]
        if not self.quality:
            lines.append("- no quality claims: nothing annotated was measured")
        for name, scores in sorted(self.quality.items()):
            formatted = ", ".join(f"{key}={value:.3f}" for key, value in sorted(scores.items()))
            lines.append(f"- {name}: {formatted}")
        lines += ["", "## Coverage gaps", ""]
        if not self.gaps:
            lines.append("- none")
        for gap in self.gaps:
            lines.append(f"- `{gap.code}` ({gap.status.value}): {gap.message}")
        if self.notes:
            lines += ["", "## Notes", ""]
            lines.extend(f"- {note}" for note in self.notes)
        return "\n".join(lines) + "\n"
