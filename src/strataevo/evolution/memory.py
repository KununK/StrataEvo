"""Cross-generation memory for completed self-evolution attempts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .types import GenerationRecord

if TYPE_CHECKING:
    from .diagnosis import DiagnosisReport
    from .plan import EvolutionPlanReport


DEFAULT_MEMORY_CONTEXT_ENTRIES = 8
PATCH_EXCERPT_CHARS = 3000
EVOLUTION_LAYERS = {"model", "context", "tools", "architecture"}


@dataclass(slots=True)
class MemoryDiagnosis:
    primary_layer: str
    related_layers: list[str]
    problem: str
    affected_tasks: list[str]
    proposed_direction: str
    confidence: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryDiagnosis:
        diagnosis = cls(
            primary_layer=str(data["primary_layer"]),
            related_layers=_string_list(data.get("related_layers", []), "related_layers"),
            problem=str(data["problem"]),
            affected_tasks=_string_list(data.get("affected_tasks", []), "affected_tasks"),
            proposed_direction=str(data["proposed_direction"]),
            confidence=float(data["confidence"]),
        )
        unknown_layers = {diagnosis.primary_layer, *diagnosis.related_layers} - EVOLUTION_LAYERS
        if unknown_layers:
            raise ValueError(f"invalid memory evolution layers: {sorted(unknown_layers)}")
        if not 0.0 <= diagnosis.confidence <= 1.0:
            raise ValueError("memory diagnosis confidence must be between 0 and 1")
        return diagnosis


@dataclass(slots=True)
class EvolutionMemoryEntry:
    generation: int
    parent_commit: str
    resulting_commit: str | None
    decision: str
    outcome_type: str
    reason: str
    diagnoses: list[MemoryDiagnosis]
    plan: dict[str, Any]
    outcome_observations: list[dict[str, Any]]
    changed_paths: list[str]
    patch_path: str | None
    patch_excerpt: str
    agent_output: str
    parent_task_score: float
    parent_utility: float
    candidate_task_score: float | None
    candidate_utility: float | None
    utility_delta: float | None
    evaluation_attempts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_context_dict(self) -> dict[str, Any]:
        """Return the compact facts useful to a later model call."""
        return {
            "generation": self.generation,
            "decision": self.decision,
            "outcome_type": self.outcome_type,
            "reason": self.reason,
            "diagnoses": [asdict(item) for item in self.diagnoses],
            "plan": self.plan,
            "outcome_observations": self.outcome_observations,
            "changed_paths": self.changed_paths,
            "patch_path": self.patch_path,
            "patch_excerpt": self.patch_excerpt[:PATCH_EXCERPT_CHARS],
            "agent_output": self.agent_output[:2000],
            "parent_task_score": self.parent_task_score,
            "parent_utility": self.parent_utility,
            "candidate_task_score": self.candidate_task_score,
            "candidate_utility": self.candidate_utility,
            "utility_delta": self.utility_delta,
            "evaluation_attempts": [_attempt_context(item) for item in self.evaluation_attempts],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvolutionMemoryEntry:
        values = dict(data)
        values["diagnoses"] = [
            MemoryDiagnosis.from_dict(item) for item in values.get("diagnoses", [])
        ]
        values["plan"] = _object(values.get("plan", {}), "plan")
        observations = values.get("outcome_observations", [])
        if not isinstance(observations, list) or not all(
            isinstance(item, dict) for item in observations
        ):
            raise ValueError("memory outcome_observations must be a list of objects")
        values["outcome_observations"] = observations
        values["changed_paths"] = _string_list(values.get("changed_paths", []), "changed_paths")
        values.setdefault("patch_path", None)
        values.setdefault("patch_excerpt", "")
        values.setdefault(
            "outcome_type", _legacy_outcome_type(values.get("decision"), values.get("reason"))
        )
        values.setdefault("evaluation_attempts", [])
        if not isinstance(values["evaluation_attempts"], list) or not all(
            isinstance(item, dict) for item in values["evaluation_attempts"]
        ):
            raise ValueError("memory evaluation_attempts must be a list of objects")
        entry = cls(**values)
        if entry.generation <= 0:
            raise ValueError("memory generation must be positive")
        if entry.decision not in {"accepted", "rejected"}:
            raise ValueError(f"invalid memory decision: {entry.decision}")
        if entry.outcome_type not in {
            "accepted",
            "no_change",
            "validation_failed",
            "evaluation_failed",
            "deferred_change",
            "mixed_change_scope",
            "unclassified_change",
            "benchmark_rejected",
        }:
            raise ValueError(f"invalid memory outcome type: {entry.outcome_type}")
        return entry

    @classmethod
    def from_generation(
        cls,
        record: GenerationRecord,
        diagnosis: DiagnosisReport,
        plan_report: EvolutionPlanReport,
        agent_output: str,
    ) -> EvolutionMemoryEntry:
        from .plan import observe_expected_outcomes

        parent = record.promotion_parent_report or record.parent_report
        candidate = record.candidate_report
        parent_utility = float(parent["utility"])
        candidate_utility = float(candidate["utility"]) if candidate else None
        patch_path = Path(record.patch_path) if record.patch_path else None
        patch_excerpt = ""
        if patch_path and patch_path.is_file():
            patch_excerpt = patch_path.read_text(encoding="utf-8")[:PATCH_EXCERPT_CHARS]
        return cls(
            generation=record.generation,
            parent_commit=record.parent_commit,
            resulting_commit=record.resulting_commit,
            decision=record.decision,
            outcome_type=record.outcome_type,
            reason=record.reason,
            diagnoses=[MemoryDiagnosis.from_dict(item.to_dict()) for item in diagnosis.diagnoses],
            plan=plan_report.plan.to_dict(),
            outcome_observations=observe_expected_outcomes(
                plan_report.plan, parent, record.candidate_report
            ),
            changed_paths=list(record.changed_paths),
            patch_path=record.patch_path,
            patch_excerpt=patch_excerpt,
            agent_output=agent_output.strip(),
            parent_task_score=float(parent["task_score"]),
            parent_utility=parent_utility,
            candidate_task_score=float(candidate["task_score"]) if candidate else None,
            candidate_utility=candidate_utility,
            utility_delta=(
                candidate_utility - parent_utility if candidate_utility is not None else None
            ),
            evaluation_attempts=list(record.evaluation_attempts),
        )


class EvolutionMemory:
    """Persist one auditable entry for every completed generation."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> list[EvolutionMemoryEntry]:
        if not self.path.exists():
            return []
        entries: list[EvolutionMemoryEntry] = []
        generations: set[int] = set()
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if not isinstance(data, dict):
                    raise ValueError("entry must be an object")
                entry = EvolutionMemoryEntry.from_dict(data)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid evolution memory line {line_number}: {error}") from error
            if entry.generation in generations:
                raise ValueError(f"duplicate evolution memory generation: {entry.generation}")
            generations.add(entry.generation)
            entries.append(entry)
        return sorted(entries, key=lambda item: item.generation)

    def append(self, entry: EvolutionMemoryEntry) -> None:
        entries = self.load()
        existing = next((item for item in entries if item.generation == entry.generation), None)
        if existing:
            if existing != entry:
                raise ValueError(f"conflicting evolution memory generation: {entry.generation}")
            return
        entries.append(entry)
        self._write(entries)

    def latest(self, limit: int = DEFAULT_MEMORY_CONTEXT_ENTRIES) -> list[EvolutionMemoryEntry]:
        _validate_limit(limit)
        return self.load()[-limit:]

    def relevant(
        self,
        layers: set[str],
        limit: int = DEFAULT_MEMORY_CONTEXT_ENTRIES,
    ) -> list[EvolutionMemoryEntry]:
        _validate_limit(limit)
        entries = self.load()
        matching = [
            entry
            for entry in entries
            if any(
                diagnosis.primary_layer in layers
                or bool(layers.intersection(diagnosis.related_layers))
                for diagnosis in entry.diagnoses
            )
        ]
        selected = matching[-limit:]
        if len(selected) < limit:
            selected_generations = {entry.generation for entry in selected}
            fallback = [entry for entry in entries if entry.generation not in selected_generations][
                -(limit - len(selected)) :
            ]
            selected = [*fallback, *selected]
        return sorted(selected, key=lambda item: item.generation)

    def _write(self, entries: list[EvolutionMemoryEntry]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        content = "".join(
            json.dumps(entry.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            for entry in sorted(entries, key=lambda item: item.generation)
        )
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(self.path)


def memory_context(entries: list[EvolutionMemoryEntry]) -> list[dict[str, Any]]:
    return [entry.to_context_dict() for entry in entries]


def _string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"memory {field_name} must be a list of strings")
    return list(dict.fromkeys(value))


def _object(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"memory {field_name} must be an object")
    return value


def _attempt_context(attempt: dict[str, Any]) -> dict[str, Any]:
    report = attempt.get("report")
    return {
        "number": attempt.get("number"),
        "outcome_type": attempt.get("outcome_type"),
        "reason": str(attempt.get("reason", ""))[:500],
        "changed_paths": attempt.get("changed_paths", []),
        "task_score": report.get("task_score") if isinstance(report, dict) else None,
        "utility": report.get("utility") if isinstance(report, dict) else None,
        "output_dir": report.get("output_dir") if isinstance(report, dict) else None,
    }


def _validate_limit(limit: int) -> None:
    if limit <= 0:
        raise ValueError("memory context limit must be positive")


def _legacy_outcome_type(decision: Any, reason: Any) -> str:
    if decision == "accepted":
        return "accepted"
    text = str(reason or "")
    if "no source changes" in text:
        return "no_change"
    if "validation" in text:
        return "validation_failed"
    return "benchmark_rejected"
