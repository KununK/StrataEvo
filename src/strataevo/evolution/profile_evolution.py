"""Shared persistence and evaluation for externally activated candidates."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .evaluation import BenchmarkEvaluator, promotion_decision, promotion_observation
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
    """Refine a profile and keep the first candidate that clears promotion."""
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
                "promotion": promotion_observation(parent_report, screening),
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
            accepted, reason = promotion_decision(parent_report, screening)
            if not accepted:
                continue
            activate(candidate_record)
            return ProfileEvolutionResult(
                "accepted",
                "accepted",
                f"candidate screening: {reason}",
                screening,
                None,
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
    _, reason = promotion_decision(parent_report, screening)
    return ProfileEvolutionResult(
        "rejected",
        "benchmark_rejected",
        f"best candidate screening: {reason}",
        screening,
        None,
        candidate_record,
        input_tokens,
        output_tokens,
    )

def evaluate_profile(
    parent_report: EvaluationReport,
    evaluator: BenchmarkEvaluator,
    output_dir: Path,
    parent: dict[str, Any] | None,
    candidate: dict[str, Any],
    activate: Callable[[dict[str, Any] | None], None],
) -> ProfileEvaluation:
    """Evaluate a profile once against the recorded parent."""
    try:
        activate(candidate)
        screening = evaluator.evaluate(output_dir / "screening" / "evaluation")
        accepted, reason = promotion_decision(parent_report, screening)
        if not accepted:
            activate(parent)
            return ProfileEvaluation(
                "rejected",
                "benchmark_rejected",
                f"candidate screening: {reason}",
                screening,
                None,
            )
        return ProfileEvaluation(
            "accepted",
            "accepted",
            f"candidate screening: {reason}",
            screening,
            None,
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
