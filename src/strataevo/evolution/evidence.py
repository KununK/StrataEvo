"""Structured evidence extracted from coding-agent benchmark artifacts."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .io import read_json, read_jsonl, write_json


@dataclass(slots=True)
class TaskEvidence:
    """Observable evidence for one evaluated task, without layer attribution."""

    task_id: str
    entry_point: str
    status: str
    passed: bool
    stop_reason: str
    steps: int
    input_tokens: int
    output_tokens: int
    generation_seconds: float
    test_seconds: float
    candidate_path: str | None
    candidate_present: bool
    candidate_created: bool
    tool_sequence: list[str]
    shell_commands: list[str]
    error: str
    generation_path: str
    session_path: str | None
    signals: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskEvidence:
        values = dict(data)
        values.pop("artifact_delete_attempted", None)
        values.pop("candidate_deleted", None)
        values["signals"] = [
            item
            for item in values.get("signals", [])
            if item not in {"artifact_deleted", "artifact_delete_attempted"}
        ]
        return cls(**values)


@dataclass(slots=True)
class EvidenceBundle:
    """Evaluation summary plus task-level evidence for a later diagnosis stage."""

    evaluator: str
    source_dir: str
    summary: dict[str, Any]
    signal_counts: dict[str, int]
    cases: list[TaskEvidence]

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluator": self.evaluator,
            "source_dir": self.source_dir,
            "summary": self.summary,
            "signal_counts": self.signal_counts,
            "cases": [case.to_dict() for case in self.cases],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceBundle:
        values = dict(data)
        values["cases"] = [TaskEvidence.from_dict(item) for item in values.get("cases", [])]
        signal_counts = dict(values.get("signal_counts", {}))
        signal_counts.pop("artifact_deleted", None)
        signal_counts.pop("artifact_delete_attempted", None)
        values["signal_counts"] = dict(sorted(signal_counts.items()))
        return cls(**values)


class CodingAgentEvidenceCollector:
    """Join task results, generations, candidates, and sessions."""

    artifact_name = "solution.py"

    def __init__(self, evaluator: str) -> None:
        self.evaluator = evaluator

    def collect(self, output_dir: str | Path) -> EvidenceBundle:
        root = Path(output_dir).resolve()
        summary = read_json(root / "summary.json")
        results = _index_rows(read_jsonl(root / "results.jsonl"), "results")
        generations = _index_rows(read_jsonl(root / "generations.jsonl"), "generations")
        if set(results) != set(generations):
            missing_results = sorted(set(generations).difference(results))
            missing_generations = sorted(set(results).difference(generations))
            raise ValueError(
                f"{self.evaluator} result/generation task mismatch: "
                f"missing_results={missing_results}, missing_generations={missing_generations}"
            )

        cases = [
            self._collect_case(root, row, generations[task_id]) for task_id, row in results.items()
        ]
        cases.sort(key=lambda item: _task_sort_key(item.task_id))
        evaluated = int(summary.get("evaluated", len(cases)))
        if evaluated != len(cases):
            raise ValueError(f"summary evaluated={evaluated}, but found {len(cases)} task results")

        signal_counts: dict[str, int] = {}
        for case in cases:
            for signal in case.signals:
                signal_counts[signal] = signal_counts.get(signal, 0) + 1
        return EvidenceBundle(
            evaluator=self.evaluator,
            source_dir=str(root),
            summary=summary,
            signal_counts=dict(sorted(signal_counts.items())),
            cases=cases,
        )

    def collect_and_write(
        self, output_dir: str | Path, destination: str | Path | None = None
    ) -> EvidenceBundle:
        bundle = self.collect(output_dir)
        target = Path(destination) if destination else Path(output_dir) / "evidence.json"
        write_json(target, bundle.to_dict())
        return bundle

    def _collect_case(
        self,
        root: Path,
        result: dict[str, Any],
        generation: dict[str, Any],
    ) -> TaskEvidence:
        task_id = str(result["task_id"])
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id).strip("_") or "task"
        candidate_file = root / "candidates" / f"{safe_name}.py"
        session_file = root / "sessions" / f"{safe_name}.json"
        messages = generation.get("messages") or []
        tool_sequence, shell_commands, created = self._tool_evidence(messages)
        candidate_present = candidate_file.is_file()
        status = str(result.get("status", "unknown"))
        stop_reason = str(generation.get("stop_reason", result.get("agent_stop_reason", "")))
        signals = _signals(
            status=status,
            stop_reason=stop_reason,
            candidate_present=candidate_present,
            candidate_created=created,
            agent_error=str(generation.get("agent_error", "")),
        )
        error = str(generation.get("agent_error") or result.get("stderr") or "")
        return TaskEvidence(
            task_id=task_id,
            entry_point=str(result.get("entry_point", generation.get("entry_point", ""))),
            status=status,
            passed=bool(result.get("passed", False)),
            stop_reason=stop_reason,
            steps=int(generation.get("steps", result.get("agent_steps", 0))),
            input_tokens=int((generation.get("usage") or {}).get("input_tokens", 0)),
            output_tokens=int((generation.get("usage") or {}).get("output_tokens", 0)),
            generation_seconds=float(generation.get("generation_seconds", 0.0)),
            test_seconds=float(result.get("test_seconds", 0.0)),
            candidate_path=str(candidate_file) if candidate_present else None,
            candidate_present=candidate_present,
            candidate_created=created,
            tool_sequence=tool_sequence,
            shell_commands=shell_commands,
            error=error,
            generation_path=str(root / "generations.jsonl"),
            session_path=str(session_file) if session_file.is_file() else None,
            signals=signals,
        )

    def _tool_evidence(self, messages: list[dict[str, Any]]) -> tuple[list[str], list[str], bool]:
        sequence: list[str] = []
        shell_commands: list[str] = []
        created = False
        for message in messages:
            for call in message.get("tool_calls") or []:
                name = str(call.get("name", ""))
                arguments = call.get("arguments") or {}
                sequence.append(name)
                path = str(arguments.get("path", ""))
                if name == "write_file" and _is_artifact(path, self.artifact_name):
                    created = True
                if name == "run_shell":
                    shell_commands.append(str(arguments.get("command", "")))
        return sequence, shell_commands, created


def _signals(
    *,
    status: str,
    stop_reason: str,
    candidate_present: bool,
    candidate_created: bool,
    agent_error: str,
) -> list[str]:
    signals: list[str] = []
    if status != "pass":
        signals.append(status)
    if stop_reason == "max_steps":
        signals.append("max_steps")
    if not candidate_present:
        signals.append("artifact_missing")
    if candidate_created and not candidate_present:
        signals.append("artifact_created_then_missing")
    if stop_reason == "completed" and not candidate_present:
        signals.append("completed_without_artifact")
    if agent_error:
        signals.append("agent_error")
    return signals


def _is_artifact(path: str, artifact_name: str) -> bool:
    return bool(path) and Path(path).name == artifact_name


def _index_rows(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id", ""))
        if not task_id:
            raise ValueError(f"{label} row is missing task_id")
        if task_id in indexed:
            raise ValueError(f"duplicate {label} task_id: {task_id}")
        indexed[task_id] = row
    return indexed


def _task_sort_key(task_id: str) -> tuple[str, int, str]:
    prefix, separator, suffix = task_id.rpartition("/")
    if separator and suffix.isdigit():
        return prefix, int(suffix), task_id
    return task_id, -1, task_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect structured benchmark evidence")
    parser.add_argument("output_dir", help="benchmark output directory")
    parser.add_argument("--benchmark", choices=("humaneval", "mbpp"), default="humaneval")
    parser.add_argument("--output", help="Evidence JSON path; defaults to OUTPUT_DIR/evidence.json")
    args = parser.parse_args(argv)
    bundle = CodingAgentEvidenceCollector(args.benchmark).collect_and_write(
        args.output_dir, args.output
    )
    print(
        json.dumps(
            {
                "evaluator": bundle.evaluator,
                "cases": len(bundle.cases),
                "signal_counts": bundle.signal_counts,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
