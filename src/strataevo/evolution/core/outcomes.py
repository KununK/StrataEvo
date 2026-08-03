"""Derive compact, causal outcomes from generation evaluations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..types import GenerationRecord

TASK_CONTEXT_LIMIT = 12
TEXT_CONTEXT_CHARS = 500
OUTCOME_STATUSES = {"supported", "regressed", "no_measured_gain", "not_evaluated"}
HYPOTHESIS_VERDICTS = {"supported", "refuted", "untested"}
INTERVENTION_VERDICTS = {"effective", "ineffective", "failed", "untested"}
FAILED_INTERVENTION_OUTCOMES = {
    "deferred_change",
    "mixed_change_scope",
    "model_training_skipped",
    "no_change",
    "semantic_noop",
    "unclassified_change",
    "validation_failed",
}


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
    intervention_verdict: str = "untested"
    intervention_evidence: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        outcome_type: str = "",
    ) -> MemoryOutcome:
        status = str(data["status"])
        verdict = str(data.get("hypothesis_verdict", _legacy_verdict(status)))
        raw_counterevidence = data.get("counterevidence")
        if raw_counterevidence is None:
            raw_counterevidence = [str(data["summary"])] if verdict == "refuted" else []
        intervention_verdict = str(
            data.get(
                "intervention_verdict",
                _legacy_intervention_verdict(status, outcome_type),
            )
        )
        intervention_evidence = data.get("intervention_evidence")
        if intervention_evidence is None:
            intervention_evidence = (
                [str(data["summary"])]
                if intervention_verdict in {"ineffective", "failed"}
                else []
            )
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
            intervention_verdict=intervention_verdict,
            intervention_evidence=_string_list(
                intervention_evidence, "intervention_evidence"
            ),
        )
        if outcome.status not in OUTCOME_STATUSES:
            raise ValueError(f"invalid memory outcome status: {outcome.status}")
        if outcome.hypothesis_verdict not in HYPOTHESIS_VERDICTS:
            raise ValueError(
                f"invalid memory hypothesis verdict: {outcome.hypothesis_verdict}"
            )
        if outcome.intervention_verdict not in INTERVENTION_VERDICTS:
            raise ValueError(
                f"invalid memory intervention verdict: {outcome.intervention_verdict}"
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
            "intervention_verdict": self.intervention_verdict,
            "intervention_evidence": [
                item[:TEXT_CONTEXT_CHARS] for item in self.intervention_evidence[:4]
            ],
        }


def summarize_generation(
    record: GenerationRecord,
    parent: dict[str, Any],
    candidate: dict[str, Any] | None,
) -> MemoryOutcome:
    outcome = summarize_outcome(
        {
            "decision": record.decision,
            "outcome_type": record.outcome_type,
            "reason": record.reason,
            "parent_task_score": parent.get("task_score"),
            "candidate_task_score": candidate.get("task_score") if candidate else None,
        }
    )
    if candidate:
        transitions = task_transitions(parent, candidate)
        outcome.fixed_tasks = transitions["fixed_tasks"]
        outcome.regressed_tasks = transitions["regressed_tasks"]
        outcome.remaining_failures = transitions["remaining_failures"]
        if outcome.intervention_verdict == "ineffective":
            if outcome.regressed_tasks:
                outcome.intervention_evidence.append(
                    f"Regressed {len(outcome.regressed_tasks)} previously passing tasks."
                )
            if not outcome.fixed_tasks:
                outcome.intervention_evidence.append(
                    "No failing task was fixed in the comparison."
                )
    return outcome


def summarize_outcome(values: dict[str, Any]) -> MemoryOutcome:
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
        return MemoryOutcome(
            "supported",
            delta,
            [],
            [],
            [],
            f"The intervention was accepted with task-score delta {_format_delta(delta)}.",
            "Retain this change and build on the evidence that it improved the benchmark.",
            "supported",
            [],
            "effective",
            [f"Accepted with candidate task-score delta {_format_delta(delta)}."],
        )
    if outcome_type == "benchmark_rejected" and delta is not None:
        regressed = delta < 0
        return MemoryOutcome(
            "regressed" if regressed else "no_measured_gain",
            delta,
            [],
            [],
            [],
            (
                f"The evaluated intervention regressed by {_format_delta(delta)}."
                if regressed
                else f"The evaluated intervention produced no promotable gain "
                f"({_format_delta(delta)})."
            ),
            (
                "Keep the hypothesis open, but do not repeat this intervention unchanged."
                if regressed
                else "Retry this direction only with new evidence or a materially different "
                "intervention."
            ),
            "untested",
            [],
            "ineffective",
            [f"Candidate task-score delta was {_format_delta(delta)}."],
        )
    failed = outcome_type in FAILED_INTERVENTION_OUTCOMES
    return MemoryOutcome(
        "not_evaluated",
        delta,
        [],
        [],
        [],
        f"The hypothesis was not tested successfully: {reason or outcome_type}.",
        (
            "Keep the causal hypothesis open, but do not repeat this failed intervention."
            if failed
            else "Resolve the evaluation failure before drawing a causal conclusion."
        ),
        "untested",
        [],
        "failed" if failed else "untested",
        [reason or outcome_type] if failed else [],
    )


def task_transitions(
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


def _string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"memory {field_name} must be a list of strings")
    return list(dict.fromkeys(value))


def _task_context(tasks: list[str]) -> dict[str, Any]:
    return {"count": len(tasks), "examples": tasks[:TASK_CONTEXT_LIMIT]}


def _optional_float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _format_delta(delta: float | None) -> str:
    return "unknown" if delta is None else f"{delta:+.6f}"


def _legacy_verdict(status: str) -> str:
    if status == "supported":
        return "supported"
    if status in {"regressed", "no_measured_gain"}:
        return "refuted"
    return "untested"


def _legacy_intervention_verdict(status: str, outcome_type: str) -> str:
    if status == "supported":
        return "effective"
    if status in {"regressed", "no_measured_gain"}:
        return "ineffective"
    if outcome_type in FAILED_INTERVENTION_OUTCOMES:
        return "failed"
    return "untested"
