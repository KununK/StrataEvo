"""Observable task outcomes supplied to the evolution decider."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .utils.io import read_json, read_jsonl, write_json


@dataclass(slots=True)
class ToolEvent:
    index: int
    name: str
    arguments: dict[str, Any]
    result: str | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolEvent:
        return cls(
            int(data["index"]),
            str(data["name"]),
            dict(data.get("arguments", {})),
            str(data["result"]) if data.get("result") is not None else None,
        )


@dataclass(slots=True)
class TaskEvidence:
    task_id: str
    status: str
    passed: bool
    stop_reason: str
    steps: int
    candidate_present: bool
    candidate_created: bool
    tool_events: list[ToolEvent]
    error: str
    signals: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskEvidence:
        return cls(
            task_id=str(data["task_id"]),
            status=str(data["status"]),
            passed=bool(data["passed"]),
            stop_reason=str(data.get("stop_reason", "")),
            steps=int(data.get("steps", 0)),
            candidate_present=bool(data.get("candidate_present", False)),
            candidate_created=bool(data.get("candidate_created", False)),
            tool_events=[ToolEvent.from_dict(item) for item in data.get("tool_events", [])],
            error=str(data.get("error", "")),
            signals=[str(item) for item in data.get("signals", [])],
        )


@dataclass(slots=True)
class EvidenceBundle:
    summary: dict[str, Any]
    signal_counts: dict[str, int]
    cases: list[TaskEvidence]

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "signal_counts": self.signal_counts,
            "cases": [case.to_dict() for case in self.cases],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceBundle:
        return cls(
            dict(data["summary"]),
            {str(key): int(value) for key, value in data.get("signal_counts", {}).items()},
            [TaskEvidence.from_dict(item) for item in data.get("cases", [])],
        )


class CodingAgentEvidenceCollector:
    def collect_and_write(self, output_dir: str | Path) -> EvidenceBundle:
        root = Path(output_dir).resolve()
        results = _rows_by_task(read_jsonl(root / "results.jsonl"))
        generations = _rows_by_task(read_jsonl(root / "generations.jsonl"))
        if results.keys() != generations.keys():
            raise ValueError("benchmark results and generations contain different tasks")
        cases = [
            _task_evidence(task_id, results[task_id], generations[task_id])
            for task_id in sorted(results)
        ]
        signals: dict[str, int] = {}
        for case in cases:
            for signal in case.signals:
                signals[signal] = signals.get(signal, 0) + 1
        bundle = EvidenceBundle(
            read_json(root / "summary.json"),
            signals,
            cases,
        )
        write_json(root / "evidence.json", bundle.to_dict())
        return bundle


def _task_evidence(
    task_id: str,
    result: dict[str, Any],
    generation: dict[str, Any],
) -> TaskEvidence:
    events = _tool_events(generation.get("messages") or [])
    created = any(
        event.name == "write_file"
        and Path(str(event.arguments.get("path", ""))).name == "solution.py"
        for event in events
    )
    candidate = result.get("candidate_path")
    present = bool(candidate and Path(candidate).is_file())
    artifact_expected = bool(result.get("artifact_expected", True))
    status = str(result.get("status", "unknown"))
    stop = str(generation.get("stop_reason", result.get("agent_stop_reason", "")))
    error = str(generation.get("agent_error") or result.get("stderr") or "")
    signals = [] if status == "pass" else [status]
    if stop == "max_steps":
        signals.append("max_steps")
    if artifact_expected and not present:
        signals.append("artifact_missing")
    if artifact_expected and created and not present:
        signals.append("artifact_created_then_missing")
    if error:
        signals.append("agent_error")
    return TaskEvidence(
        task_id,
        status,
        bool(result.get("passed")),
        stop,
        int(generation.get("steps", 0)),
        present,
        created,
        events,
        error,
        signals,
    )


def _tool_events(messages: list[dict[str, Any]]) -> list[ToolEvent]:
    events: list[ToolEvent] = []
    pending: dict[str, ToolEvent] = {}
    for message in messages:
        for call in message.get("tool_calls") or []:
            event = ToolEvent(
                len(events) + 1,
                str(call.get("name", "")),
                dict(call.get("arguments") or {}),
                None,
            )
            events.append(event)
            pending[str(call.get("id", ""))] = event
        if message.get("role") == "tool":
            event = pending.get(str(message.get("tool_call_id", "")))
            if event:
                event.result = str(message.get("content", ""))
    return events


def _rows_by_task(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row["task_id"])
        if task_id in result:
            raise ValueError(f"duplicate task: {task_id}")
        result[task_id] = row
    return result
