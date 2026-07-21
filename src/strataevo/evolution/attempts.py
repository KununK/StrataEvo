"""Bounded candidate evaluation and selection within one evolution generation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from .evaluation import run_commands
from .git import GitRepository
from .types import EvaluationReport


class Evaluator(Protocol):
    def evaluate(self, output_dir: Path) -> EvaluationReport: ...


@dataclass(slots=True)
class EvaluationAttempt:
    number: int
    outcome_type: str
    reason: str
    changed_paths: list[str]
    patch_path: str | None
    validation_log: str | None
    report: dict | None

    def to_dict(self) -> dict:
        return asdict(self)


class CandidateEvaluationSession:
    """Evaluate refinements, preserve their patches, and restore the best one."""

    def __init__(
        self,
        repo: Path,
        git: GitRepository,
        evaluator: Evaluator,
        parent_report: EvaluationReport,
        generation_dir: Path,
        validation_commands: list[list[str]],
        max_attempts: int,
    ) -> None:
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        self.repo = repo
        self.git = git
        self.evaluator = evaluator
        self.parent_report = parent_report
        self.generation_dir = generation_dir
        self.validation_commands = validation_commands
        self.max_attempts = max_attempts
        self.attempts: list[EvaluationAttempt] = []

    def evaluate(self) -> str:
        """Validate and benchmark the current candidate, returning feedback to the mutator."""
        if len(self.attempts) >= self.max_attempts:
            return json.dumps(
                {
                    "outcome_type": "limit_reached",
                    "reason": f"all {self.max_attempts} candidate evaluations have been used",
                    "attempts_remaining": 0,
                },
                ensure_ascii=False,
            )

        changed_paths = self.git.changed_paths()
        number = len(self.attempts) + 1
        attempt_dir = self.generation_dir / f"attempt-{number:04d}"
        attempt_dir.mkdir(parents=True, exist_ok=False)
        if not changed_paths:
            attempt = EvaluationAttempt(
                number,
                "no_change",
                "candidate has no source changes",
                [],
                None,
                None,
                None,
            )
            return self._finish_attempt(attempt)

        print(
            f"[evolution] candidate attempt {number}/{self.max_attempts}: validating",
            flush=True,
        )
        self.git.stage()
        patch_path = attempt_dir / "changes.patch"
        patch_path.write_text(self.git.staged_diff(), encoding="utf-8")
        validation_log = attempt_dir / "validation.log"
        gates_passed, output = run_commands(
            self.validation_commands,
            self.repo,
            validation_log,
            timeout=600,
        )
        if not gates_passed:
            attempt = EvaluationAttempt(
                number,
                "validation_failed",
                _last_output(output),
                changed_paths,
                str(patch_path),
                str(validation_log),
                None,
            )
            return self._finish_attempt(attempt)

        try:
            report = self.evaluator.evaluate(attempt_dir / "evaluation")
        except Exception as error:
            attempt = EvaluationAttempt(
                number,
                "evaluation_failed",
                f"{type(error).__name__}: {error}",
                changed_paths,
                str(patch_path),
                str(validation_log),
                None,
            )
            return self._finish_attempt(attempt)
        delta = report.task_score - self.parent_report.task_score
        attempt = EvaluationAttempt(
            number,
            "evaluated",
            f"pass@1 delta versus parent: {delta:+.6f}",
            changed_paths,
            str(patch_path),
            str(validation_log),
            report.to_dict(),
        )
        return self._finish_attempt(attempt)

    def ensure_evaluated(self) -> None:
        """Evaluate once when the mutator edited code but never requested feedback."""
        if len(self.attempts) >= self.max_attempts or not self.git.changed_paths():
            return
        self.git.stage()
        current_patch = self.git.staged_diff()
        last_patch = self.attempts[-1].patch_path if self.attempts else None
        if not last_patch or Path(last_patch).read_text(encoding="utf-8") != current_patch:
            self.evaluate()

    def restore_best(self) -> EvaluationAttempt | None:
        """Discard untested final edits and restore the highest-scoring evaluated patch."""
        evaluated = [attempt for attempt in self.attempts if attempt.report is not None]
        best = (
            max(evaluated, key=lambda item: float(item.report["task_score"]))
            if evaluated
            else None
        )
        if self.git.changed_paths():
            self.git.rollback()
        if best and best.patch_path:
            self.git.apply_patch(Path(best.patch_path))
        return best

    def to_dicts(self) -> list[dict]:
        return [attempt.to_dict() for attempt in self.attempts]

    def _finish_attempt(self, attempt: EvaluationAttempt) -> str:
        self.attempts.append(attempt)
        attempt_path = self.generation_dir / f"attempt-{attempt.number:04d}" / "attempt.json"
        attempt_path.write_text(
            json.dumps(attempt.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        report = attempt.report
        feedback = {
            "attempt": attempt.number,
            "outcome_type": attempt.outcome_type,
            "reason": attempt.reason,
            "attempts_remaining": self.max_attempts - len(self.attempts),
            "parent_task_score": self.parent_report.task_score,
            "candidate_task_score": report["task_score"] if report else None,
            "candidate_utility": report["utility"] if report else None,
            "evidence_path": report["metrics"].get("evidence_path") if report else None,
            "signal_counts": report["metrics"].get("evidence_signal_counts") if report else None,
        }
        if report and float(report["task_score"]) >= 1.0:
            feedback["instruction"] = "Maximum task score reached; stop editing."
        elif self.max_attempts == len(self.attempts):
            feedback["instruction"] = "Evaluation budget exhausted; stop editing."
        else:
            feedback["instruction"] = (
                "Inspect the evidence, then refine the candidate or stop if no useful "
                "correction remains."
            )
        return json.dumps(feedback, indent=2, ensure_ascii=False)


def _last_output(output: str, limit: int = 4000) -> str:
    text = output.strip()
    return text[-limit:] if text else "fixed validation commands failed"
