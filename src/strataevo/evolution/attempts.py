"""Bounded candidate evaluation and selection within one evolution generation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .contract import ChangeImpact
from .evaluation import Evaluator, run_commands
from .git import GitRepository
from .io import write_json
from .semantics import classify_changes
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
    change_semantics: dict | None = None

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
        self._working_tree_state = "parent"

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
                    "candidate_retained": None,
                    "working_tree_state": self._working_tree_state,
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
            return self._finish_attempt(
                attempt,
                candidate_retained=False,
                working_tree_state=self._working_tree_state,
            )

        self._working_tree_state = "current_candidate"
        self.git.stage()
        patch = self.git.staged_diff()
        duplicate = self._find_patch(patch)
        if duplicate is not None:
            best = self._best_improving_attempt()
            if duplicate is best:
                self._working_tree_state = "best_candidate"
                return self._feedback(
                    duplicate,
                    cached=True,
                    candidate_retained=True,
                    working_tree_state="best_candidate",
                )
            self._restore(best)
            return self._feedback(
                duplicate,
                cached=True,
                candidate_retained=False,
                working_tree_state=self._working_tree_state,
            )

        number = len(self.attempts) + 1
        attempt_dir = self.generation_dir / f"attempt-{number:04d}"
        attempt_dir.mkdir(parents=True, exist_ok=False)
        patch_path = attempt_dir / "changes.patch"
        patch_path.write_text(patch, encoding="utf-8")
        impact = self.evaluator.contract.classify(changed_paths)
        semantics = classify_changes(self.repo, changed_paths, self.git.head_text)
        if semantics.classification == "semantic_noop":
            attempt = EvaluationAttempt(
                number,
                "semantic_noop",
                "candidate changes only comments, formatting, or equivalent structured data",
                changed_paths,
                str(patch_path),
                None,
                None,
                impact.to_dict(),
                semantics.to_dict(),
            )
            self._restore(self._best_improving_attempt())
            return self._finish_attempt(
                attempt,
                candidate_retained=False,
                working_tree_state=self._working_tree_state,
            )
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
                semantics.to_dict(),
            )
            self._restore(self._best_improving_attempt())
            return self._finish_attempt(
                attempt,
                candidate_retained=False,
                working_tree_state=self._working_tree_state,
            )

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
                semantics.to_dict(),
            )
            return self._finish_attempt(
                attempt,
                candidate_retained=True,
                working_tree_state="current_candidate",
            )

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
                semantics.to_dict(),
            )
            return self._finish_attempt(
                attempt,
                candidate_retained=True,
                working_tree_state="current_candidate",
            )
        previous_best = self._best_attempt()
        best_score = max(
            self.parent_report.task_score,
            float(previous_best.report["task_score"]) if previous_best else float("-inf"),
        )
        regressed = report.task_score <= best_score
        restore_attempt = self._best_improving_attempt()
        reason = (
            f"pass@1 delta versus parent: {report.task_score - self.parent_report.task_score:+.6f}"
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
            semantics.to_dict(),
        )
        if regressed:
            self._restore(restore_attempt)
            return self._finish_attempt(
                attempt,
                candidate_retained=False,
                working_tree_state=self._working_tree_state,
            )
        return self._finish_attempt(
            attempt,
            candidate_retained=True,
            working_tree_state="current_candidate",
        )

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

    def confirm(self, attempt: EvaluationAttempt) -> tuple[EvaluationReport, EvaluationReport]:
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
        write_json(
            promotion_dir / "comparison.json",
            {
                "selection_report": attempt.report,
                "parent_report": parent.to_dict(),
                "candidate_report": candidate.to_dict(),
            },
        )
        return parent, candidate

    def to_dicts(self) -> list[dict]:
        return [attempt.to_dict() for attempt in self.attempts]

    def _finish_attempt(
        self,
        attempt: EvaluationAttempt,
        *,
        candidate_retained: bool,
        working_tree_state: str,
    ) -> str:
        self.attempts.append(attempt)
        attempt_path = self.generation_dir / f"attempt-{attempt.number:04d}" / "attempt.json"
        write_json(attempt_path, attempt.to_dict())
        self._working_tree_state = working_tree_state
        return self._feedback(
            attempt,
            candidate_retained=candidate_retained,
            working_tree_state=working_tree_state,
        )

    def _feedback(
        self,
        attempt: EvaluationAttempt,
        *,
        cached: bool = False,
        candidate_retained: bool,
        working_tree_state: str,
    ) -> str:
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
            "evidence_path": report["metrics"].get("evidence_path") if report else None,
            "signal_counts": report["metrics"].get("evidence_signal_counts") if report else None,
            "change_impact": attempt.change_impact,
            "candidate_retained": candidate_retained,
            "working_tree_state": working_tree_state,
        }
        if attempt.outcome_type == "no_change":
            tree_instruction = (
                f"No submitted patch is active; the working tree contains "
                f"the {working_tree_state.replace('_', ' ')}."
            )
        elif attempt.outcome_type == "semantic_noop":
            tree_instruction = (
                "The submitted patch was discarded because it has no detectable semantic change; "
                f"the working tree now contains the {working_tree_state.replace('_', ' ')}."
            )
        elif candidate_retained:
            tree_instruction = "The submitted patch remains in the working tree."
        else:
            tree_instruction = (
                f"The submitted patch is not active; the working tree now contains "
                f"the {working_tree_state.replace('_', ' ')}."
            )
        if attempt.outcome_type in {
            "deferred_change",
            "mixed_change_scope",
            "unclassified_change",
        }:
            next_action = (
                "This patch cannot be attributed to the active benchmark. Make one focused change "
                "only under its direct_paths, or stop."
            )
        elif attempt.outcome_type == "semantic_noop":
            next_action = "Make a focused behavioral change before requesting another check."
        elif attempt.outcome_type == "validation_failed":
            next_action = (
                "Correct the reported validation errors before requesting another check. "
                "Use read_file as the source of truth; diagnostic gutters are annotations, "
                "not source characters."
            )
        elif attempt.outcome_type == "evaluation_failed":
            next_action = "Inspect the evaluation error and correct the retained candidate."
        elif report and float(report["task_score"]) >= 1.0:
            next_action = "Maximum task score reached; stop editing."
        elif self.evaluations_used >= self.max_evaluations:
            next_action = "Evaluation budget exhausted; stop editing."
        else:
            next_action = (
                "Inspect the evidence, then refine the candidate or stop if no useful "
                "correction remains."
            )
        feedback["instruction"] = f"{tree_instruction} {next_action}"
        return json.dumps(feedback, indent=2, ensure_ascii=False)

    def _find_patch(self, patch: str) -> EvaluationAttempt | None:
        for attempt in reversed(self.attempts):
            if attempt.patch_path and Path(attempt.patch_path).read_text(encoding="utf-8") == patch:
                return attempt
        return None

    def _best_attempt(self) -> EvaluationAttempt | None:
        evaluated = [attempt for attempt in self.attempts if attempt.report is not None]
        return (
            max(evaluated, key=lambda item: float(item.report["task_score"])) if evaluated else None
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
            self._working_tree_state = "best_candidate"
        else:
            self._working_tree_state = "parent"


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
