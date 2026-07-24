"""Bounded candidate evaluation and selection within one evolution generation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .contract import ChangeImpact
from .evaluation import Evaluator, run_commands
from .git import GitRepository
from .types import EvaluationReport


@dataclass(slots=True)
class EvaluationAttempt:
    number: int
    outcome_type: str
    reason: str
    changed_paths: list[str]
    patch_path: str | None
    validation_log: str | None
    report: dict | None
    change_impact: dict | None

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
        max_evaluations: int,
    ) -> None:
        if max_evaluations <= 0:
            raise ValueError("max_evaluations must be positive")
        self.repo = repo
        self.git = git
        self.evaluator = evaluator
        self.parent_report = parent_report
        self.generation_dir = generation_dir
        self.validation_commands = validation_commands
        self.max_evaluations = max_evaluations
        self.attempts: list[EvaluationAttempt] = []

    @property
    def evaluations_used(self) -> int:
        return sum(attempt.report is not None for attempt in self.attempts)

    def evaluate(self) -> str:
        """Validate and benchmark the current candidate, returning feedback to the mutator."""
        if self.evaluations_used >= self.max_evaluations:
            return json.dumps(
                {
                    "outcome_type": "limit_reached",
                    "reason": f"all {self.max_evaluations} benchmark evaluations have been used",
                    "evaluations_used": self.evaluations_used,
                    "evaluations_remaining": 0,
                },
                ensure_ascii=False,
            )

        changed_paths = self.git.changed_paths()
        if not changed_paths:
            number = len(self.attempts) + 1
            attempt_dir = self.generation_dir / f"attempt-{number:04d}"
            attempt_dir.mkdir(parents=True, exist_ok=False)
            attempt = EvaluationAttempt(
                number,
                "no_change",
                "candidate has no source changes",
                [],
                None,
                None,
                None,
                None,
            )
            return self._finish_attempt(attempt)

        self.git.stage()
        patch = self.git.staged_diff()
        duplicate = self._find_patch(patch)
        if duplicate is not None:
            if duplicate is not self._best_improving_attempt():
                self._restore(self._best_improving_attempt())
            return self._feedback(duplicate, cached=True)

        number = len(self.attempts) + 1
        attempt_dir = self.generation_dir / f"attempt-{number:04d}"
        attempt_dir.mkdir(parents=True, exist_ok=False)
        patch_path = attempt_dir / "changes.patch"
        patch_path.write_text(patch, encoding="utf-8")
        impact = self.evaluator.contract.classify(changed_paths)
        scope_error = _scope_error(impact)
        if scope_error:
            attempt = EvaluationAttempt(
                number,
                scope_error[0],
                scope_error[1],
                changed_paths,
                str(patch_path),
                None,
                None,
                impact.to_dict(),
            )
            feedback = self._finish_attempt(attempt)
            self._restore(self._best_improving_attempt())
            return feedback

        print(
            f"[evolution] candidate check {number}: validating",
            flush=True,
        )
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
                impact.to_dict(),
            )
            return self._finish_attempt(attempt)

        benchmark_number = self.evaluations_used + 1
        print(
            f"[evolution] candidate benchmark {benchmark_number}/{self.max_evaluations}",
            flush=True,
        )
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
                impact.to_dict(),
            )
            return self._finish_attempt(attempt)
        previous_best = self._best_attempt()
        best_score = max(
            self.parent_report.task_score,
            float(previous_best.report["task_score"]) if previous_best else float("-inf"),
        )
        regressed = report.task_score <= best_score
        restore_attempt = self._best_improving_attempt()
        reason = (
            "pass@1 delta versus parent: "
            f"{report.task_score - self.parent_report.task_score:+.6f}"
        )
        if regressed:
            restored = "best candidate" if restore_attempt else "parent"
            reason += f"; restored {restored} after non-improving result"
        attempt = EvaluationAttempt(
            number,
            "evaluated",
            reason,
            changed_paths,
            str(patch_path),
            str(validation_log),
            report.to_dict(),
            impact.to_dict(),
        )
        feedback = self._finish_attempt(attempt)
        if regressed:
            self._restore(restore_attempt)
        return feedback

    def ensure_evaluated(self) -> None:
        """Evaluate once when the mutator edited code but never requested feedback."""
        if self.evaluations_used >= self.max_evaluations or not self.git.changed_paths():
            return
        self.git.stage()
        current_patch = self.git.staged_diff()
        last_patch = self.attempts[-1].patch_path if self.attempts else None
        if not last_patch or Path(last_patch).read_text(encoding="utf-8") != current_patch:
            self.evaluate()

    def restore_best(self) -> EvaluationAttempt | None:
        """Discard untested final edits and restore the highest-scoring evaluated patch."""
        best = self._best_attempt()
        self._restore(best)
        return best

    def confirm(
        self, attempt: EvaluationAttempt
    ) -> tuple[EvaluationReport, EvaluationReport]:
        """Compare parent and selected candidate on fresh, unselected benchmark runs."""
        if not attempt.patch_path or attempt.report is None:
            raise ValueError("only an evaluated candidate can be confirmed")
        promotion_dir = self.generation_dir / "promotion"
        promotion_dir.mkdir(parents=True, exist_ok=False)
        print("[evolution] promotion check: re-evaluating parent", flush=True)
        self._restore(None)
        parent = self.evaluator.evaluate(promotion_dir / "parent" / "evaluation")
        print("[evolution] promotion check: confirming candidate", flush=True)
        self._restore(attempt)
        candidate = self.evaluator.evaluate(promotion_dir / "candidate" / "evaluation")
        (promotion_dir / "comparison.json").write_text(
            json.dumps(
                {
                    "selection_report": attempt.report,
                    "parent_report": parent.to_dict(),
                    "candidate_report": candidate.to_dict(),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return parent, candidate

    def to_dicts(self) -> list[dict]:
        return [attempt.to_dict() for attempt in self.attempts]

    def _finish_attempt(self, attempt: EvaluationAttempt) -> str:
        self.attempts.append(attempt)
        attempt_path = self.generation_dir / f"attempt-{attempt.number:04d}" / "attempt.json"
        attempt_path.write_text(
            json.dumps(attempt.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return self._feedback(attempt)

    def _feedback(self, attempt: EvaluationAttempt, *, cached: bool = False) -> str:
        report = attempt.report
        feedback = {
            "attempt": attempt.number,
            "cached": cached,
            "outcome_type": attempt.outcome_type,
            "reason": attempt.reason,
            "evaluations_used": self.evaluations_used,
            "evaluations_remaining": self.max_evaluations - self.evaluations_used,
            "parent_task_score": self.parent_report.task_score,
            "candidate_task_score": report["task_score"] if report else None,
            "candidate_utility": report["utility"] if report else None,
            "evidence_path": report["metrics"].get("evidence_path") if report else None,
            "signal_counts": report["metrics"].get("evidence_signal_counts") if report else None,
            "change_impact": attempt.change_impact,
        }
        if attempt.outcome_type in {
            "deferred_change",
            "mixed_change_scope",
            "unclassified_change",
        }:
            feedback["instruction"] = (
                "This patch cannot be attributed to the active benchmark. Make one focused change "
                "only under its direct_paths, or stop."
            )
        elif report and float(report["task_score"]) >= 1.0:
            feedback["instruction"] = "Maximum task score reached; stop editing."
        elif self.evaluations_used >= self.max_evaluations:
            feedback["instruction"] = "Evaluation budget exhausted; stop editing."
        else:
            feedback["instruction"] = (
                "Inspect the evidence, then refine the candidate or stop if no useful "
                "correction remains."
            )
        return json.dumps(feedback, indent=2, ensure_ascii=False)

    def _find_patch(self, patch: str) -> EvaluationAttempt | None:
        for attempt in reversed(self.attempts):
            if attempt.patch_path and Path(attempt.patch_path).read_text(encoding="utf-8") == patch:
                return attempt
        return None

    def _best_attempt(self) -> EvaluationAttempt | None:
        evaluated = [attempt for attempt in self.attempts if attempt.report is not None]
        return (
            max(evaluated, key=lambda item: float(item.report["task_score"]))
            if evaluated
            else None
        )

    def _best_improving_attempt(self) -> EvaluationAttempt | None:
        best = self._best_attempt()
        if best and float(best.report["task_score"]) > self.parent_report.task_score:
            return best
        return None

    def _restore(self, attempt: EvaluationAttempt | None) -> None:
        if self.git.changed_paths():
            self.git.rollback()
        if attempt and attempt.patch_path:
            self.git.apply_patch(Path(attempt.patch_path))


def _last_output(output: str, limit: int = 4000) -> str:
    text = output.strip()
    return text[-limit:] if text else "fixed validation commands failed"


def _scope_error(impact: ChangeImpact) -> tuple[str, str] | None:
    if impact.unclassified_paths:
        return (
            "unclassified_change",
            "the evaluation contract does not classify every changed path",
        )
    if impact.direct_paths and impact.deferred_paths:
        return (
            "mixed_change_scope",
            "patch mixes code used by this benchmark with code that only affects later generations",
        )
    if impact.deferred_paths:
        return (
            "deferred_change",
            "changed code is not loaded by this benchmark and cannot receive its task score",
        )
    return None
