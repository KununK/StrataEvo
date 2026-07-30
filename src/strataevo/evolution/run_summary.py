"""Run-level summaries derived from persisted evolution memory."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .io import write_json
from .memory import EvolutionMemoryEntry


def write_run_summary(
    run_dir: Path,
    entries: list[EvolutionMemoryEntry],
    state: dict[str, Any],
) -> None:
    generations = [_generation_summary(entry) for entry in entries]
    planned_layers = Counter(item["planned_layer"] for item in generations)
    diagnosed_layers = Counter(
        layer for item in generations for layer in item["diagnosed_layers"]
    )
    outcomes = Counter(item["outcome_type"] for item in generations)
    decisions = Counter(item["decision"] for item in generations)
    summary = {
        "run_name": run_dir.name,
        "generation_count": len(generations),
        "baseline_task_score": entries[0].parent_task_score if entries else None,
        "current_task_score": state.get("current_report", {}).get("task_score"),
        "diagnosed_layers": dict(sorted(diagnosed_layers.items())),
        "planned_layers": dict(sorted(planned_layers.items())),
        "outcomes": dict(sorted(outcomes.items())),
        "decisions": dict(sorted(decisions.items())),
        "generations": generations,
    }
    write_json(run_dir / "summary.json", summary)
    _write_markdown(run_dir / "summary.md", summary)


def _generation_summary(entry: EvolutionMemoryEntry) -> dict[str, Any]:
    semantic_noops = sum(
        attempt.get("outcome_type") == "semantic_noop" for attempt in entry.evaluation_attempts
    )
    return {
        "generation": entry.generation,
        "diagnosed_layers": list(
            dict.fromkeys(diagnosis.primary_layer for diagnosis in entry.diagnoses)
        ),
        "planned_layer": entry.plan.get("primary_layer"),
        "candidate_type": _candidate_type(entry),
        "hypothesis": entry.plan.get("hypothesis"),
        "decision": entry.decision,
        "outcome_type": entry.outcome_type,
        "hypothesis_verdict": entry.outcome.hypothesis_verdict,
        "parent_task_score": entry.parent_task_score,
        "candidate_task_score": entry.candidate_task_score,
        "score_delta": entry.outcome.score_delta,
        "changed_paths": entry.changed_paths,
        "semantic_noop_attempts": semantic_noops,
        "reason": entry.reason,
    }


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    rows = [
        "# Evolution Run Summary",
        "",
        f"- Run: `{summary['run_name']}`",
        f"- Generations: {summary['generation_count']}",
        f"- Baseline task score: {_score(summary['baseline_task_score'])}",
        f"- Current task score: {_score(summary['current_task_score'])}",
        f"- Diagnosed layers: `{summary['diagnosed_layers']}`",
        f"- Planned layers: `{summary['planned_layers']}`",
        f"- Decisions: `{summary['decisions']}`",
        "",
        "| Generation | Layer | Hypothesis | Decision | Outcome | Verdict | Parent | Candidate |",
        "| ---: | --- | --- | --- | --- | --- | ---: | ---: |",
    ]
    for item in summary["generations"]:
        rows.append(
            f"| {item['generation']} | {item['planned_layer']} | {_cell(item['hypothesis'])} | "
            f"{item['decision']} | {item['outcome_type']} | {item['hypothesis_verdict']} | "
            f"{_score(item['parent_task_score'])} | {_score(item['candidate_task_score'])} |"
        )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(rows) + "\n", encoding="utf-8")
    temporary.replace(path)


def _candidate_type(entry: EvolutionMemoryEntry) -> str:
    if entry.model_candidate is not None:
        return "model_adapter"
    if entry.context_candidate is not None:
        return "context_profile"
    if entry.tool_candidate is not None:
        return "tool_profile"
    return "source_patch" if entry.changed_paths else "none"


def _cell(value: Any) -> str:
    return str(value or "-").replace("|", "\\|").replace("\n", " ")


def _score(value: Any) -> str:
    return "-" if value is None else f"{float(value):.6f}"
