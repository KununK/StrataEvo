"""Shared evaluation transaction for external evolution profiles."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .evaluation import BenchmarkEvaluator
from .types import EvaluationReport


@dataclass(slots=True)
class ProfileEvaluation:
    decision: str
    outcome_type: str
    reason: str
    candidate_report: EvaluationReport | None
    promotion_parent_report: EvaluationReport | None


def evaluate_profile(
    parent_report: EvaluationReport,
    evaluator: BenchmarkEvaluator,
    output_dir: Path,
    parent: dict[str, Any] | None,
    candidate: dict[str, Any],
    activate: Callable[[dict[str, Any] | None], None],
) -> ProfileEvaluation:
    """Screen a profile, compare it with a fresh parent, and restore on rejection."""
    activate(candidate)
    try:
        screening = evaluator.evaluate(output_dir / "screening" / "evaluation")
        if screening.task_score <= parent_report.task_score:
            activate(parent)
            return ProfileEvaluation(
                "rejected",
                "benchmark_rejected",
                (
                    f"candidate screening: pass@1 {screening.task_score:.6f} "
                    f"did not exceed parent {parent_report.task_score:.6f}"
                ),
                screening,
                None,
            )

        activate(parent)
        fresh_parent = evaluator.evaluate(output_dir / "promotion" / "parent" / "evaluation")
        activate(candidate)
        confirmed = evaluator.evaluate(output_dir / "promotion" / "candidate" / "evaluation")
    except BaseException:
        activate(parent)
        raise

    if confirmed.task_score <= fresh_parent.task_score:
        activate(parent)
        return ProfileEvaluation(
            "rejected",
            "benchmark_rejected",
            (
                f"fresh promotion comparison: pass@1 {confirmed.task_score:.6f} "
                f"did not exceed parent {fresh_parent.task_score:.6f}"
            ),
            confirmed,
            fresh_parent,
        )
    return ProfileEvaluation(
        "accepted",
        "accepted",
        "fresh promotion comparison: pass@1 strictly improved",
        confirmed,
        fresh_parent,
    )
