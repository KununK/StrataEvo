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
PATCH_CONTEXT_CHARS = 1200
TASK_CONTEXT_LIMIT = 12
TEXT_CONTEXT_CHARS = 500
EVOLUTION_LAYERS = {"model", "context", "tools", "architecture"}
OUTCOME_STATUSES = {"supported", "regressed", "no_measured_gain", "not_evaluated"}
HYPOTHESIS_VERDICTS = {"supported", "refuted", "untested"}


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
class MemoryOutcome:
    status: str
    score_delta: float | None
    fixed_tasks: list[str]
    regressed_tasks: list[str]
    remaining_failures: list[str]
    summary: str
    next_step: str
    hypothesis_verdict: str = "untested"
    counterevidence: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryOutcome:
        status = str(data["status"])
        verdict = str(data.get("hypothesis_verdict", _legacy_verdict(status)))
        raw_counterevidence = data.get("counterevidence")
        if raw_counterevidence is None:
            raw_counterevidence = [str(data["summary"])] if verdict == "refuted" else []
        outcome = cls(
            status=status,
            score_delta=(
                float(data["score_delta"]) if data.get("score_delta") is not None else None
            ),
            fixed_tasks=_string_list(data.get("fixed_tasks", []), "fixed_tasks"),
            regressed_tasks=_string_list(data.get("regressed_tasks", []), "regressed_tasks"),
            remaining_failures=_string_list(
                data.get("remaining_failures", []), "remaining_failures"
            ),
            summary=str(data["summary"]),
            next_step=str(data["next_step"]),
            hypothesis_verdict=verdict,
            counterevidence=_string_list(raw_counterevidence, "counterevidence"),
        )
        if outcome.status not in OUTCOME_STATUSES:
            raise ValueError(f"invalid memory outcome status: {outcome.status}")
        if outcome.hypothesis_verdict not in HYPOTHESIS_VERDICTS:
            raise ValueError(
                f"invalid memory hypothesis verdict: {outcome.hypothesis_verdict}"
            )
        return outcome

    def to_context_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "score_delta": self.score_delta,
            "fixed_tasks": _task_context(self.fixed_tasks),
            "regressed_tasks": _task_context(self.regressed_tasks),
            "remaining_failures": _task_context(self.remaining_failures),
            "summary": self.summary[:TEXT_CONTEXT_CHARS],
            "next_step": self.next_step[:TEXT_CONTEXT_CHARS],
            "hypothesis_verdict": self.hypothesis_verdict,
            "counterevidence": [
                item[:TEXT_CONTEXT_CHARS] for item in self.counterevidence[:4]
            ],
        }


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
    candidate_task_score: float | None
    evaluation_attempts: list[dict[str, Any]] = field(default_factory=list)
    model_candidate: dict[str, Any] | None = None
    context_candidate: dict[str, Any] | None = None
    tool_candidate: dict[str, Any] | None = None
    outcome: MemoryOutcome = field(
        default_factory=lambda: MemoryOutcome(
            status="not_evaluated",
            score_delta=None,
            fixed_tasks=[],
            regressed_tasks=[],
            remaining_failures=[],
            summary="No outcome summary was recorded.",
            next_step="Use the raw generation record before reusing this intervention.",
        )
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_context_dict(self) -> dict[str, Any]:
        """Return the causal facts useful to a later model call."""
        selected = _selected_diagnosis(self.diagnoses, self.plan)
        return {
            "generation": self.generation,
            "decision": self.decision,
            "outcome_type": self.outcome_type,
            "reason": self.reason[:TEXT_CONTEXT_CHARS],
            "selected_diagnosis": asdict(selected) if selected else None,
            "action": {
                "primary_layer": self.plan.get("primary_layer"),
                "hypothesis": self.plan.get("hypothesis"),
                "intervention": self.plan.get("intervention"),
                "expected_outcomes": self.plan.get("expected_outcomes", []),
                "changed_paths": self.changed_paths,
                "patch_excerpt": self.patch_excerpt[:PATCH_CONTEXT_CHARS],
                "model_candidate": self.model_candidate,
                "context_candidate": self.context_candidate,
                "tool_candidate": self.tool_candidate,
            },
            "outcome": self.outcome.to_context_dict(),
            "outcome_observations": self.outcome_observations,
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
        for legacy in ("parent_utility", "candidate_utility", "utility_delta"):
            values.pop(legacy, None)
        values.setdefault("evaluation_attempts", [])
        values.setdefault("model_candidate", None)
        values.setdefault("context_candidate", None)
        values.setdefault("tool_candidate", None)
        if not isinstance(values["evaluation_attempts"], list) or not all(
            isinstance(item, dict) for item in values["evaluation_attempts"]
        ):
            raise ValueError("memory evaluation_attempts must be a list of objects")
        raw_outcome = values.get("outcome")
        values["outcome"] = (
            MemoryOutcome.from_dict(raw_outcome)
            if isinstance(raw_outcome, dict)
            else _summarize_outcome(values)
        )
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
            "model_training_skipped",
            "semantic_noop",
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
        patch_path = Path(record.patch_path) if record.patch_path else None
        patch_excerpt = ""
        if patch_path and patch_path.is_file():
            patch_excerpt = patch_path.read_text(encoding="utf-8")[:PATCH_EXCERPT_CHARS]
        outcome = _summarize_generation(record, parent, candidate)
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
            candidate_task_score=float(candidate["task_score"]) if candidate else None,
            evaluation_attempts=list(record.evaluation_attempts),
            model_candidate=record.model_candidate,
            context_candidate=record.context_candidate,
            tool_candidate=record.tool_candidate,
            outcome=outcome,
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
            if bool(layers.intersection(_entry_layers(entry)))
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


def memory_context(
    entries: list[EvolutionMemoryEntry],
    max_chars: int | None = None,
) -> list[dict[str, Any]]:
    items = [entry.to_context_dict() for entry in entries]
    if max_chars is None:
        return items
    if max_chars <= 0:
        raise ValueError("memory context max_chars must be positive")
    selected: list[dict[str, Any]] = []
    size = 2
    for item in reversed(items):
        item_size = len(json.dumps(item, ensure_ascii=False)) + 1
        if size + item_size > max_chars:
            continue
        selected.append(item)
        size += item_size
    return list(reversed(selected))


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
        "reason": str(attempt.get("reason", ""))[:250],
        "changed_paths": attempt.get("changed_paths", []),
        "task_score": report.get("task_score") if isinstance(report, dict) else None,
    }


def _summarize_generation(
    record: GenerationRecord,
    parent: dict[str, Any],
    candidate: dict[str, Any] | None,
) -> MemoryOutcome:
    values = {
        "decision": record.decision,
        "outcome_type": record.outcome_type,
        "reason": record.reason,
        "parent_task_score": parent.get("task_score"),
        "candidate_task_score": candidate.get("task_score") if candidate else None,
    }
    outcome = _summarize_outcome(values)
    if candidate:
        transitions = _task_transitions(parent, candidate)
        outcome.fixed_tasks = transitions["fixed_tasks"]
        outcome.regressed_tasks = transitions["regressed_tasks"]
        outcome.remaining_failures = transitions["remaining_failures"]
        if outcome.hypothesis_verdict == "refuted":
            if outcome.regressed_tasks:
                outcome.counterevidence.append(
                    f"Regressed {len(outcome.regressed_tasks)} previously passing tasks."
                )
            if not outcome.fixed_tasks:
                outcome.counterevidence.append("No failing task was fixed in the comparison.")
    return outcome


def _summarize_outcome(values: dict[str, Any]) -> MemoryOutcome:
    parent_score = _optional_float(values.get("parent_task_score"))
    candidate_score = _optional_float(values.get("candidate_task_score"))
    delta = (
        candidate_score - parent_score
        if parent_score is not None and candidate_score is not None
        else None
    )
    outcome_type = str(values.get("outcome_type", ""))
    decision = str(values.get("decision", ""))
    reason = str(values.get("reason", "")).strip()
    if decision == "accepted":
        status = "supported"
        verdict = "supported"
        counterevidence = []
        summary = f"The intervention was accepted with task-score delta {_format_delta(delta)}."
        next_step = "Retain this change and build on the evidence that it improved the benchmark."
    elif outcome_type == "benchmark_rejected" and delta is not None and delta < 0:
        status = "regressed"
        verdict = "refuted"
        counterevidence = [f"Candidate task-score delta was {_format_delta(delta)}."]
        summary = f"The evaluated intervention regressed by {_format_delta(delta)}."
        next_step = "Do not repeat this intervention unchanged; revise its causal hypothesis."
    elif outcome_type == "benchmark_rejected" and delta is not None:
        status = "no_measured_gain"
        verdict = "refuted"
        counterevidence = [f"Candidate task-score delta was {_format_delta(delta)}."]
        summary = (
            f"The evaluated intervention produced no promotable gain ({_format_delta(delta)})."
        )
        next_step = (
            "Retry this direction only with new evidence or a materially different intervention."
        )
    else:
        status = "not_evaluated"
        verdict = "untested"
        counterevidence = []
        summary = f"The hypothesis was not tested successfully: {reason or outcome_type}."
        next_step = (
            "Resolve the execution failure before treating it as evidence about the hypothesis."
        )
    return MemoryOutcome(
        status,
        delta,
        [],
        [],
        [],
        summary,
        next_step,
        verdict,
        counterevidence,
    )


def _task_transitions(
    parent_report: dict[str, Any],
    candidate_report: dict[str, Any],
) -> dict[str, list[str]]:
    parent = _task_results(parent_report)
    candidate = _task_results(candidate_report)
    shared = sorted(parent.keys() & candidate.keys())
    return {
        "fixed_tasks": [task for task in shared if not parent[task] and candidate[task]],
        "regressed_tasks": [task for task in shared if parent[task] and not candidate[task]],
        "remaining_failures": [task for task in shared if not candidate[task]],
    }


def _task_results(report: dict[str, Any]) -> dict[str, bool]:
    output_dir = report.get("output_dir")
    if not isinstance(output_dir, str):
        return {}
    path = Path(output_dir) / "results.jsonl"
    if not path.is_file():
        return {}
    results: dict[str, bool] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            return {}
        task_id = item.get("task_id")
        if not isinstance(task_id, (str, int)):
            return {}
        results[str(task_id)] = bool(item.get("passed", item.get("status") == "pass"))
    return results


def _selected_diagnosis(
    diagnoses: list[MemoryDiagnosis],
    plan: dict[str, Any],
) -> MemoryDiagnosis | None:
    target = plan.get("target_diagnosis")
    if isinstance(target, int) and 0 <= target < len(diagnoses):
        return diagnoses[target]
    return diagnoses[0] if diagnoses else None


def _entry_layers(entry: EvolutionMemoryEntry) -> set[str]:
    layers = {
        layer
        for diagnosis in entry.diagnoses
        for layer in [diagnosis.primary_layer, *diagnosis.related_layers]
    }
    planned = entry.plan.get("primary_layer")
    if isinstance(planned, str):
        layers.add(planned)
    return layers


def _task_context(tasks: list[str]) -> dict[str, Any]:
    return {"count": len(tasks), "examples": tasks[:TASK_CONTEXT_LIMIT]}


def _optional_float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _format_delta(delta: float | None) -> str:
    return "unknown" if delta is None else f"{delta:+.6f}"


def _validate_limit(limit: int) -> None:
    if limit <= 0:
        raise ValueError("memory context limit must be positive")


def _legacy_verdict(status: str) -> str:
    if status == "supported":
        return "supported"
    if status in {"regressed", "no_measured_gain"}:
        return "refuted"
    return "untested"


def _legacy_outcome_type(decision: Any, reason: Any) -> str:
    if decision == "accepted":
        return "accepted"
    text = str(reason or "")
    if "no source changes" in text:
        return "no_change"
    if "validation" in text:
        return "validation_failed"
    return "benchmark_rejected"
