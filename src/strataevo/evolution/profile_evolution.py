"""Shared persistence and evaluation for externally activated candidates."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .evaluation import BenchmarkEvaluator
from .io import read_jsonl, write_json
from .types import EvaluationReport

ProfileGenerator = Callable[
    [list[dict[str, Any]]],
    tuple[dict[str, Any], dict[str, Any]],
]


@dataclass(slots=True)
class ProfileEvaluation:
    decision: str
    outcome_type: str
    reason: str
    candidate_report: EvaluationReport | None
    promotion_parent_report: EvaluationReport | None


@dataclass(slots=True)
class ProfileEvolutionResult(ProfileEvaluation):
    candidate: dict[str, Any]
    input_tokens: int = 0
    output_tokens: int = 0


def evolve_profile(
    *,
    generation: int,
    output_dir: Path,
    label: str,
    parent_report: EvaluationReport,
    evaluator: BenchmarkEvaluator,
    normalized_parent: dict[str, Any],
    active_parent: dict[str, Any] | None,
    create_candidate: ProfileGenerator,
    activate: Callable[[dict[str, Any] | None], None],
    max_evaluations: int,
) -> ProfileEvolutionResult:
    """Refine a profile with screening feedback, then confirm its first improvement."""
    if max_evaluations <= 0:
        raise ValueError("max_evaluations must be positive")
    write_json(output_dir / "parent.json", normalized_parent)
    feedback: list[dict[str, Any]] = []
    seen_candidates: dict[str, int] = {}
    best: tuple[dict[str, Any], dict[str, Any], EvaluationReport] | None = None
    input_tokens = 0
    output_tokens = 0
    candidate_path = output_dir / "candidate.json"
    try:
        for attempt in range(1, max_evaluations + 1):
            candidate, metadata = create_candidate(feedback)
            input_tokens += int(metadata["input_tokens"])
            output_tokens += int(metadata["output_tokens"])
            attempt_dir = output_dir / f"attempt-{attempt:04d}"
            attempt_path = attempt_dir / "candidate.json"
            candidate_record = {**candidate, "path": str(candidate_path)}
            candidate_artifact = {
                **candidate,
                "generation": generation,
                "refinement_attempt": attempt,
                **metadata,
            }
            write_json(attempt_path, candidate_artifact)
            write_json(candidate_path, candidate_artifact)

            if candidate == normalized_parent:
                observation = {
                    "attempt": attempt,
                    "outcome_type": "no_change",
                    "reason": f"{label} candidate is identical to its parent",
                    "candidate": candidate,
                }
                feedback.append(observation)
                write_json(attempt_dir / "result.json", observation)
                write_json(output_dir / "refinement.json", feedback)
                continue

            candidate_key = json.dumps(candidate, ensure_ascii=False, sort_keys=True)
            if candidate_key in seen_candidates:
                observation = {
                    "attempt": attempt,
                    "outcome_type": "duplicate_candidate",
                    "reason": (
                        f"{label} candidate repeats attempt "
                        f"{seen_candidates[candidate_key]}"
                    ),
                    "candidate": candidate,
                }
                feedback.append(observation)
                write_json(attempt_dir / "result.json", observation)
                write_json(output_dir / "refinement.json", feedback)
                continue
            seen_candidates[candidate_key] = attempt

            print(
                f"[evolution] {label} candidate screening {attempt}/{max_evaluations}",
                flush=True,
            )
            activate(candidate_record)
            screening = evaluator.evaluate(attempt_dir / "screening" / "evaluation")
            activate(active_parent)
            changes = task_changes(parent_report.output_dir, screening.output_dir)
            observation = {
                "attempt": attempt,
                "outcome_type": "evaluated",
                "task_score": screening.task_score,
                "score_delta": screening.task_score - parent_report.task_score,
                "task_changes": {
                    name: {"count": len(tasks), "examples": tasks[:12]}
                    for name, tasks in changes.items()
                },
                "candidate": candidate,
            }
            feedback.append(observation)
            write_json(attempt_dir / "result.json", observation)
            write_json(output_dir / "refinement.json", feedback)
            if best is None or screening.task_score > best[2].task_score:
                best = candidate_record, metadata, screening
            if screening.task_score <= parent_report.task_score:
                continue

            evaluation = confirm_profile(
                evaluator,
                output_dir,
                active_parent,
                candidate_record,
                activate,
                parent_report.task_score,
            )
            return ProfileEvolutionResult(
                evaluation.decision,
                evaluation.outcome_type,
                evaluation.reason,
                evaluation.candidate_report,
                evaluation.promotion_parent_report,
                candidate_record,
                input_tokens,
                output_tokens,
            )
    except BaseException:
        activate(active_parent)
        raise

    activate(active_parent)
    if best is None:
        return ProfileEvolutionResult(
            "rejected",
            "no_change",
            f"{label} refinement produced no changed candidate",
            None,
            None,
            normalized_parent,
            input_tokens,
            output_tokens,
        )
    candidate_record, metadata, screening = best
    write_json(
        candidate_path,
        {
            **candidate_record,
            "generation": generation,
            **metadata,
        },
    )
    return ProfileEvolutionResult(
        "rejected",
        "benchmark_rejected",
        (
            f"best candidate screening: pass@1 {screening.task_score:.6f} "
            f"did not exceed parent {parent_report.task_score:.6f}"
        ),
        screening,
        None,
        candidate_record,
        input_tokens,
        output_tokens,
    )


def confirm_profile(
    evaluator: BenchmarkEvaluator,
    output_dir: Path,
    parent: dict[str, Any] | None,
    candidate: dict[str, Any],
    activate: Callable[[dict[str, Any] | None], None],
    recorded_parent_score: float,
) -> ProfileEvaluation:
    """Compare one screened profile with a fresh parent and restore it on rejection."""
    try:
        activate(parent)
        fresh_parent = evaluator.evaluate(output_dir / "promotion" / "parent" / "evaluation")
        activate(candidate)
        confirmed = evaluator.evaluate(output_dir / "promotion" / "candidate" / "evaluation")
    except BaseException:
        activate(parent)
        raise

    if confirmed.task_score <= max(recorded_parent_score, fresh_parent.task_score):
        activate(parent)
        return ProfileEvaluation(
            "rejected",
            "benchmark_rejected",
            (
                f"fresh promotion comparison: pass@1 {confirmed.task_score:.6f} "
                f"must exceed stored parent {recorded_parent_score:.6f} "
                f"and fresh parent {fresh_parent.task_score:.6f}"
            ),
            confirmed,
            fresh_parent,
        )
    return ProfileEvaluation(
        "accepted",
        "accepted",
        "fresh promotion comparison: pass@1 strictly improved over stored and fresh parent",
        confirmed,
        fresh_parent,
    )


def evaluate_profile(
    parent_report: EvaluationReport,
    evaluator: BenchmarkEvaluator,
    output_dir: Path,
    parent: dict[str, Any] | None,
    candidate: dict[str, Any],
    activate: Callable[[dict[str, Any] | None], None],
) -> ProfileEvaluation:
    """Screen a profile, compare it with a fresh parent, and restore on rejection."""
    try:
        activate(candidate)
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

        return confirm_profile(
            evaluator,
            output_dir,
            parent,
            candidate,
            activate,
            parent_report.task_score,
        )
    except BaseException:
        activate(parent)
        raise


def task_changes(parent_dir: str | Path, candidate_dir: str | Path) -> dict[str, list[str]]:
    parent = _task_results(parent_dir)
    candidate = _task_results(candidate_dir)
    shared = sorted(parent.keys() & candidate.keys())
    return {
        "fixed_tasks": [task for task in shared if not parent[task] and candidate[task]],
        "regressed_tasks": [task for task in shared if parent[task] and not candidate[task]],
        "still_failed_tasks": [task for task in shared if not parent[task] and not candidate[task]],
    }


def _task_results(directory: str | Path) -> dict[str, bool]:
    return {
        str(row["task_id"]): bool(row.get("passed", row.get("status") == "pass"))
        for row in read_jsonl(Path(directory) / "results.jsonl", missing_ok=True)
    }
