"""Model-guided diagnosis of which evolution layer should change next."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from tinyagent import Message, Model, OpenAICompatibleModel

from .evidence import EvidenceBundle, TaskEvidence
from .memory import EvolutionMemoryEntry, memory_context
from .types import EvaluationReport, EvolutionConfig


class EvolutionLayer(StrEnum):
    MODEL = "model"
    CONTEXT = "context"
    TOOLS = "tools"
    ARCHITECTURE = "architecture"


@dataclass(slots=True)
class Diagnosis:
    primary_layer: EvolutionLayer
    related_layers: list[EvolutionLayer]
    problem: str
    evidence: list[str]
    affected_tasks: list[str]
    proposed_direction: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["primary_layer"] = self.primary_layer.value
        data["related_layers"] = [layer.value for layer in self.related_layers]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Diagnosis:
        required_text = ["problem", "proposed_direction"]
        for field_name in required_text:
            if not str(data.get(field_name, "")).strip():
                raise ValueError(f"diagnosis {field_name} must not be empty")
        evidence = _string_list(data.get("evidence"), "evidence")
        if not evidence:
            raise ValueError("diagnosis evidence must not be empty")
        affected_tasks = _string_list(data.get("affected_tasks", []), "affected_tasks")
        if not affected_tasks:
            raise ValueError("diagnosis affected_tasks must not be empty")
        primary = EvolutionLayer(str(data["primary_layer"]))
        related = [
            EvolutionLayer(item)
            for item in _string_list(data.get("related_layers", []), "related_layers")
        ]
        related = list(dict.fromkeys(layer for layer in related if layer != primary))
        confidence = float(data.get("confidence", 0.0))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("diagnosis confidence must be between 0 and 1")
        return cls(
            primary_layer=primary,
            related_layers=related,
            problem=str(data["problem"]).strip(),
            evidence=evidence,
            affected_tasks=affected_tasks,
            proposed_direction=str(data["proposed_direction"]).strip(),
            confidence=confidence,
        )


@dataclass(slots=True)
class DiagnosisReport:
    source_dir: str
    input_case_count: int
    diagnoses: list[Diagnosis]
    input_tokens: int
    output_tokens: int
    raw_output: str
    attempts: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_dir": self.source_dir,
            "input_case_count": self.input_case_count,
            "diagnoses": [item.to_dict() for item in self.diagnoses],
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "raw_output": self.raw_output,
            "attempts": self.attempts,
        }


class EvidenceDiagnoser:
    """Ask a model to cluster evidence and attribute each problem to an evolution layer."""

    def __init__(self, model: Model, *, max_cases: int = 40, repair_retries: int = 1) -> None:
        if max_cases <= 0:
            raise ValueError("max_cases must be positive")
        if repair_retries < 0:
            raise ValueError("repair_retries must be non-negative")
        self.model = model
        self.max_cases = max_cases
        self.repair_retries = repair_retries

    def diagnose(
        self,
        bundle: EvidenceBundle,
        history: list[EvolutionMemoryEntry] | None = None,
    ) -> DiagnosisReport:
        cases = _select_cases(bundle.cases, self.max_cases)
        payload = {
            "evaluator": bundle.evaluator,
            "summary": bundle.summary,
            "signal_counts": bundle.signal_counts,
            "promotion_objective": {
                "metric": "task_score",
                "meaning": "pass@1",
                "requires_strict_improvement": True,
            },
            "cases": [_compact_case(case) for case in cases],
            "prior_evolution": memory_context(history or []),
        }
        messages = [
            Message("system", DIAGNOSIS_SYSTEM_PROMPT),
            Message(
                "user",
                "Diagnose the following evaluation evidence. "
                "Return only the requested JSON.\n\n"
                + json.dumps(payload, indent=2, ensure_ascii=False),
            ),
        ]
        known_tasks = {case.task_id for case in cases}
        attempts: list[str] = []
        input_tokens = 0
        output_tokens = 0
        last_error: ValueError | None = None
        for attempt_number in range(self.repair_retries + 1):
            response = self.model.complete(messages, [])
            attempts.append(response.message.content)
            input_tokens += response.usage.input_tokens
            output_tokens += response.usage.output_tokens
            try:
                diagnoses = _parse_diagnoses(response.message.content, known_tasks)
            except ValueError as error:
                last_error = error
                if attempt_number == self.repair_retries:
                    break
                messages.extend(
                    [
                        response.message,
                        Message(
                            "user",
                            f"Your JSON was invalid: {error}. Correct it and return only JSON.",
                        ),
                    ]
                )
                continue
            return DiagnosisReport(
                source_dir=bundle.source_dir,
                input_case_count=len(cases),
                diagnoses=diagnoses,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                raw_output=response.message.content,
                attempts=attempts,
            )
        raise ValueError(
            f"diagnosis remained invalid after {len(attempts)} attempt(s): {last_error}"
        ) from last_error


DIAGNOSIS_SYSTEM_PROMPT = """You diagnose failures in a self-evolving software agent.
Cluster the supplied observations into a small number of concrete problems and attribute each
problem to exactly one primary evolution layer. Use related_layers only when another layer is
directly involved.

Layer definitions:
- model: stable reasoning or knowledge limitations with context, tools, and control flow held fixed;
  a stronger model doing better is not by itself evidence that this layer is faulty.
- context: prompts, selected history, memory, compaction, examples, and information presentation.
- tools: tool schemas, descriptions, implementations, skills, and tool-result representations.
- architecture: agent loop, planning, stopping, validation, recovery, state, and orchestration.

Use only supplied observations. Cite task IDs and concrete events in evidence. Prior evolution
records are outcomes, not proof of the current cause: use them to avoid blindly repeating rejected
hypotheses and to preserve improvements that were accepted. Do not propose code or claim hidden
causes. Treat each case's passed field as authoritative. A passed case that stopped at max_steps is
a successful but potentially inefficient case, not an incomplete task; never describe it as missing
a solution. When any failed cases exist, diagnose mechanisms that can improve those failures before
pure efficiency issues from passed cases. Prefer the layer closest to the failed mechanism rather
than listing every layer. Return one to six diagnoses as exactly this JSON object:
{
  "diagnoses": [
    {
      "primary_layer": "model|context|tools|architecture",
      "related_layers": ["model|context|tools|architecture"],
      "problem": "concise problem statement",
      "evidence": ["observable fact"],
      "affected_tasks": ["task id"],
      "proposed_direction": "behavioral improvement direction, not a code patch",
      "confidence": 0.0
    }
  ]
}"""


def diagnose_evaluation(
    config: EvolutionConfig,
    report: EvaluationReport,
    destination: str | Path,
    history: list[EvolutionMemoryEntry] | None = None,
) -> DiagnosisReport:
    evidence_path = Path(
        report.metrics.get("evidence_path", Path(report.output_dir) / "evidence.json")
    )
    bundle = EvidenceBundle.from_dict(json.loads(evidence_path.read_text(encoding="utf-8")))
    model = OpenAICompatibleModel(
        model=config.model,
        base_url=config.base_url,
        temperature=0.0,
        timeout=300.0,
    )
    diagnosis = EvidenceDiagnoser(model).diagnose(bundle, history)
    _write_json(Path(destination), diagnosis.to_dict())
    return diagnosis


def _select_cases(cases: list[TaskEvidence], limit: int) -> list[TaskEvidence]:
    def priority(case: TaskEvidence) -> tuple[int, int, int, str]:
        return (
            0 if not case.passed else 1,
            0 if "artifact_created_then_missing" in case.signals else 1,
            0 if "max_steps" in case.signals else 1,
            case.task_id,
        )

    anomalous = sorted((case for case in cases if case.signals), key=priority)
    selected = anomalous[:limit]
    selected_ids = {case.task_id for case in selected}
    remaining = sorted(
        (case for case in cases if case.task_id not in selected_ids),
        key=lambda case: (-(case.steps), -(case.input_tokens + case.output_tokens), case.task_id),
    )
    return (selected + remaining)[:limit]


def _compact_case(case: TaskEvidence) -> dict[str, Any]:
    notable_commands = [
        command
        for command in case.shell_commands
        if "solution.py" in command and ("rm " in command or "unlink " in command)
    ]
    return {
        "task_id": case.task_id,
        "status": case.status,
        "passed": case.passed,
        "stop_reason": case.stop_reason,
        "steps": case.steps,
        "candidate_present": case.candidate_present,
        "candidate_created": case.candidate_created,
        "artifact_delete_attempted": case.artifact_delete_attempted,
        "tool_sequence": case.tool_sequence,
        "notable_shell_commands": notable_commands,
        "error": case.error[:1000],
        "signals": case.signals,
        "candidate_path": case.candidate_path,
        "candidate_source": _candidate_excerpt(case),
        "session_path": case.session_path,
    }


def _candidate_excerpt(case: TaskEvidence) -> str:
    if case.passed or not case.candidate_path:
        return ""
    path = Path(case.candidate_path)
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8")[:4000]
    except (OSError, UnicodeDecodeError):
        return ""


def _parse_object(text: str) -> dict[str, Any]:
    content = text.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()
    start = content.find("{")
    end = content.rfind("}")
    if start < 0 or end < start:
        raise ValueError("diagnosis response does not contain a JSON object")
    data = json.loads(content[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("diagnosis response must be a JSON object")
    return data


def _parse_diagnoses(text: str, known_tasks: set[str]) -> list[Diagnosis]:
    data = _parse_object(text)
    raw_diagnoses = data.get("diagnoses")
    if not isinstance(raw_diagnoses, list) or not raw_diagnoses:
        raise ValueError("diagnosis response must contain a non-empty diagnoses list")
    if len(raw_diagnoses) > 6 or not all(isinstance(item, dict) for item in raw_diagnoses):
        raise ValueError("diagnosis response must contain one to six diagnosis objects")
    diagnoses = [Diagnosis.from_dict(item) for item in raw_diagnoses]
    referenced_tasks = {task_id for item in diagnoses for task_id in item.affected_tasks}
    unknown_tasks = sorted(referenced_tasks.difference(known_tasks))
    if unknown_tasks:
        raise ValueError(f"diagnosis references tasks outside its evidence: {unknown_tasks}")
    return diagnoses


def _string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"diagnosis {field_name} must be a list of strings")
    return list(dict.fromkeys(item.strip() for item in value if item.strip()))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose benchmark evidence by evolution layer")
    parser.add_argument("output_dir", help="benchmark output directory")
    parser.add_argument("--output", help="Diagnosis JSON path")
    parser.add_argument("--model", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--max-cases", type=int, default=40)
    parser.add_argument("--repair-retries", type=int, default=1)
    args = parser.parse_args(argv)
    evidence_path = Path(args.output_dir) / "evidence.json"
    bundle = EvidenceBundle.from_dict(json.loads(evidence_path.read_text(encoding="utf-8")))
    model = OpenAICompatibleModel(model=args.model, base_url=args.base_url, temperature=0.0)
    report = EvidenceDiagnoser(
        model, max_cases=args.max_cases, repair_retries=args.repair_retries
    ).diagnose(bundle)
    target = Path(args.output) if args.output else Path(args.output_dir) / "diagnosis.json"
    _write_json(target, report.to_dict())
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
