"""Transactional validation and selection of source candidates."""

from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ..runtime.evaluation import Evaluator, run_commands
from ..runtime.repository import GitRepository
from ..types import EvaluationReport
from ..utils.io import write_json


@dataclass(slots=True)
class EvaluationAttempt:
    number: int
    outcome: str
    reason: str
    changed_paths: list[str]
    patch_path: str | None = None
    validation_log: str | None = None
    report: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class CandidateEvaluationSession:
    """Keep only executable candidates and restore the best evaluated patch."""

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
        return sum(item.report is not None for item in self.attempts)

    def evaluate(self) -> str:
        if self.evaluations_used >= self.max_evaluations:
            return self._feedback(None, "evaluation budget exhausted")
        paths = self.git.changed_paths()
        if not paths:
            return self._store(
                EvaluationAttempt(
                    len(self.attempts) + 1, "no_change", "candidate has no source changes", []
                )
            )

        self.git.stage()
        patch = self.git.staged_diff()
        if not patch.strip():
            return self._store(
                EvaluationAttempt(
                    len(self.attempts) + 1,
                    "no_change",
                    "candidate has no staged source diff",
                    [],
                )
            )
        if not self._has_python_semantic_change(paths):
            return self._store(
                EvaluationAttempt(
                    len(self.attempts) + 1,
                    "no_change",
                    "candidate has no Python semantic change",
                    paths,
                )
            )
        duplicate = self._find_patch(patch)
        if duplicate:
            self._restore(self._best_improving_attempt())
            return self._feedback(duplicate, "unchanged candidate; reused prior result")

        number = len(self.attempts) + 1
        directory = self.generation_dir / f"attempt-{number:04d}"
        directory.mkdir(parents=True, exist_ok=False)
        patch_path = directory / "changes.patch"
        patch_path.write_text(patch, encoding="utf-8")
        print(f"[evolution] candidate check {number}: validating", flush=True)
        validation_log = directory / "validation.log"
        valid, output = run_commands(
            self.validation_commands, self.repo, validation_log, timeout=600
        )
        if not valid:
            return self._store(
                EvaluationAttempt(
                    number,
                    "validation_failed",
                    output.strip()[-4000:] or "validation failed",
                    paths,
                    str(patch_path),
                    str(validation_log),
                )
            )

        print(
            f"[evolution] candidate benchmark {self.evaluations_used + 1}/{self.max_evaluations}",
            flush=True,
        )
        try:
            report = self.evaluator.evaluate(directory / "evaluation")
        except Exception as error:
            return self._store(
                EvaluationAttempt(
                    number,
                    "evaluation_failed",
                    f"{type(error).__name__}: {error}",
                    paths,
                    str(patch_path),
                    str(validation_log),
                )
            )
        attempt = EvaluationAttempt(
            number,
            "evaluated",
            f"pass@1 delta {report.task_score - self.parent_report.task_score:+.6f}",
            paths,
            str(patch_path),
            str(validation_log),
            report.to_dict(),
        )
        result = self._store(attempt)
        if self._best_attempt() is not attempt:
            self._restore(self._best_improving_attempt())
        return result

    def ensure_evaluated(self) -> None:
        if self.git.changed_paths() and self.evaluations_used < self.max_evaluations:
            self.git.stage()
            patch = self.git.staged_diff()
            if not self._find_patch(patch):
                self.evaluate()

    def restore_best(self) -> EvaluationAttempt | None:
        best = self._best_attempt()
        self._restore(best)
        return best

    def _store(self, attempt: EvaluationAttempt) -> str:
        self.attempts.append(attempt)
        directory = self.generation_dir / f"attempt-{attempt.number:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / "attempt.json", attempt.to_dict())
        return self._feedback(attempt, attempt.reason)

    def _feedback(self, attempt: EvaluationAttempt | None, reason: str) -> str:
        report = attempt.report if attempt else None
        return json.dumps(
            {
                "outcome": attempt.outcome if attempt else "limit_reached",
                "reason": reason,
                "parent_task_score": self.parent_report.task_score,
                "candidate_task_score": report["task_score"] if report else None,
                "evidence_path": report["metrics"].get("evidence_path") if report else None,
                "evaluations_remaining": self.max_evaluations - self.evaluations_used,
            },
            ensure_ascii=False,
        )

    def _find_patch(self, patch: str) -> EvaluationAttempt | None:
        return next(
            (
                item
                for item in reversed(self.attempts)
                if item.patch_path and Path(item.patch_path).read_text(encoding="utf-8") == patch
            ),
            None,
        )

    def _has_python_semantic_change(self, paths: list[str]) -> bool:
        python_paths = [path for path in paths if path.endswith(".py")]
        if len(python_paths) != len(paths):
            return False
        for path in python_paths:
            before = self.git.head_text(path) or ""
            target = self.repo / path
            after = target.read_text(encoding="utf-8") if target.exists() else ""
            try:
                if ast.dump(ast.parse(before)) != ast.dump(ast.parse(after)):
                    return True
            except SyntaxError:
                return True
        return False

    def _best_attempt(self) -> EvaluationAttempt | None:
        evaluated = [item for item in self.attempts if item.report]
        return max(evaluated, key=lambda item: item.report["task_score"], default=None)

    def _best_improving_attempt(self) -> EvaluationAttempt | None:
        best = self._best_attempt()
        return best if best and best.report["task_score"] > self.parent_report.task_score else None

    def _restore(self, attempt: EvaluationAttempt | None) -> None:
        if self.git.changed_paths():
            self.git.rollback()
        if attempt and attempt.patch_path:
            patch_path = Path(attempt.patch_path)
            if patch_path.stat().st_size:
                self.git.apply_patch(patch_path)
