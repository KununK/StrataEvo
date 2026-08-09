"""Run one isolated, transactional evolution generation."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .decision import EvolutionDecision, decide_evolution
from .layers.architecture import CandidateEvaluationSession
from .layers.context import evolve_context
from .layers.model.evolution import activate_saved_adapter, evolve_model
from .layers.profile import ProfileEvolutionResult
from .layers.tools import evolve_tools
from .memory import EvolutionMemory, EvolutionMemoryEntry
from .mutator import mutate
from .runtime.evaluation import create_evaluator, promotion_decision, validation_commands
from .runtime.repository import GitRepository
from .runtime.structured import StructuredResponseError
from .types import EvaluationReport, EvolutionConfig, GenerationRecord
from .utils.io import read_json, write_json


def run_one_generation(config_path: Path) -> int:
    config = EvolutionConfig.from_dict(read_json(config_path))
    repo = Path(config.repo).resolve()
    run_dir = config_path.parent
    state_path = run_dir / "state.json"
    memory = EvolutionMemory(run_dir / "memory.jsonl")
    git = GitRepository(repo, config.mutable_paths)
    git.ensure_clean()
    if _git_output(repo, ["branch", "--show-current"]).strip() != config.branch:
        raise RuntimeError(f"expected branch {config.branch!r}")

    evaluator = create_evaluator(repo, config)
    state = _load_state(state_path, git, evaluator, run_dir, config.model)
    if config.model_evolution:
        activate_saved_adapter(config, evaluator, state.get("current_adapter"))
    evaluator.set_context(state.get("current_context"))
    evaluator.set_tool_profile(state.get("current_tool_profile"))
    parent = EvaluationReport.from_dict(state["current_report"])
    if parent.task_score >= 1.0:
        state["completed"] = True
        write_json(state_path, state)
        return 0

    generation = int(state["next_generation"])
    directory = run_dir / f"generation-{generation:04d}"
    _prepare_generation_dir(directory)
    parent_commit = git.head()
    decision_path = directory / "decision.json"

    try:
        try:
            decision = decide_evolution(config, parent, decision_path, memory.latest())
        except StructuredResponseError as error:
            record = GenerationRecord(
                generation,
                parent_commit,
                None,
                "decision",
                "rejected",
                "decision_invalid",
                str(error),
                parent.to_dict(),
                None,
            )
            write_json(directory / "record.json", record.to_dict())
            state["next_generation"] = generation + 1
            write_json(state_path, state)
            _print(record)
            return 0
        layer = decision.layer.value
        if layer == "context":
            result = evolve_context(
                config,
                generation,
                directory,
                parent,
                evaluator,
                decision,
                memory.relevant({"context"}),
                parent_context=state.get("current_context"),
                model_name=str(state["current_model"]),
            )
            record = _external_record(result, layer, generation, parent_commit, parent)
            if result.decision == "accepted":
                state["current_context"] = result.candidate
                state["current_report"] = result.candidate_report.to_dict()
        elif layer == "tools":
            result = evolve_tools(
                config,
                generation,
                directory,
                parent,
                evaluator,
                decision,
                memory.relevant({"tools"}),
                parent_profile=state.get("current_tool_profile"),
                model_name=str(state["current_model"]),
            )
            record = _external_record(result, layer, generation, parent_commit, parent)
            if result.decision == "accepted":
                state["current_tool_profile"] = result.candidate
                state["current_report"] = result.candidate_report.to_dict()
        elif layer == "model" and config.model_evolution:
            result = evolve_model(
                config,
                generation,
                directory,
                parent,
                evaluator,
                parent_model=str(state["current_model"]),
                parent_adapter=state.get("current_adapter"),
                decision=decision,
            )
            record = _external_record(result, layer, generation, parent_commit, parent)
            if result.decision == "accepted":
                state["current_model"] = result.candidate["name"]
                state["current_adapter"] = result.candidate
                state["current_report"] = result.candidate_report.to_dict()
        else:
            record = _evolve_architecture(
                repo,
                config,
                generation,
                directory,
                parent_commit,
                parent,
                decision,
                memory,
                git,
                evaluator,
            )
            if record.decision == "accepted":
                state["current_commit"] = record.resulting_commit
                state["current_report"] = record.candidate_report

        _finish(state_path, state, directory, record, memory, decision)
        _print(record)
        return 0
    except BaseException as error:
        if git.changed_paths():
            git.rollback()
        write_json(
            directory / "failure.json",
            {"error_type": type(error).__name__, "error": str(error)},
        )
        raise


def _evolve_architecture(
    repo: Path,
    config: EvolutionConfig,
    generation: int,
    directory: Path,
    parent_commit: str,
    parent: EvaluationReport,
    decision: EvolutionDecision,
    memory: EvolutionMemory,
    git: GitRepository,
    evaluator: Any,
) -> GenerationRecord:
    session = CandidateEvaluationSession(
        repo,
        git,
        evaluator,
        parent,
        directory,
        validation_commands(repo),
        config.max_eval_attempts,
    )
    agent = mutate(
        repo,
        config,
        generation,
        parent,
        decision,
        memory.relevant({"architecture"}),
        validation_commands(repo),
        session.evaluate,
    )
    write_json(
        directory / "agent_result.json",
        {
            "output": agent.output,
            "messages": [message.to_dict() for message in agent.messages],
            "steps": agent.steps,
            "stop_reason": agent.stop_reason,
        },
    )
    session.ensure_evaluated()
    best = session.restore_best()
    if not best or not best.report:
        git.rollback()
        return GenerationRecord(
            generation,
            parent_commit,
            None,
            "architecture",
            "rejected",
            best.outcome if best else "no_change",
            best.reason if best else "meta-agent produced no executable candidate",
            parent.to_dict(),
            None,
            best.changed_paths if best else [],
        )
    candidate = EvaluationReport.from_dict(best.report)
    accepted, reason = promotion_decision(parent, candidate)
    if accepted:
        commit = git.commit(f"evolve: generation {generation} pass@1 {candidate.task_score:.6f}")
    else:
        git.rollback()
        commit = None
    return GenerationRecord(
        generation,
        parent_commit,
        commit,
        "architecture",
        "accepted" if accepted else "rejected",
        "accepted" if accepted else "benchmark_rejected",
        f"candidate screening: {reason}",
        parent.to_dict(),
        candidate.to_dict(),
        best.changed_paths,
    )


def _external_record(
    result: ProfileEvolutionResult,
    layer: str,
    generation: int,
    parent_commit: str,
    parent: EvaluationReport,
) -> GenerationRecord:
    return GenerationRecord(
        generation,
        parent_commit,
        None,
        layer,
        result.decision,
        result.outcome_type,
        result.reason,
        parent.to_dict(),
        result.candidate_report.to_dict() if result.candidate_report else None,
    )


def _finish(
    state_path: Path,
    state: dict[str, Any],
    directory: Path,
    record: GenerationRecord,
    memory: EvolutionMemory,
    decision: EvolutionDecision,
) -> None:
    write_json(directory / "record.json", record.to_dict())
    candidate_score = (
        float(record.candidate_report["task_score"]) if record.candidate_report else None
    )
    memory.append(
        EvolutionMemoryEntry(
            record.generation,
            record.layer,
            decision.hypothesis,
            decision.intervention,
            record.decision,
            record.outcome,
            record.reason,
            float(record.parent_report["task_score"]),
            candidate_score,
            record.changed_paths,
        )
    )
    state["next_generation"] = record.generation + 1
    write_json(state_path, state)


def _load_state(path: Path, git: GitRepository, evaluator: Any, run_dir: Path, model: str) -> dict:
    if path.is_file():
        state = read_json(path)
        if state["current_commit"] != git.head():
            raise RuntimeError("state does not match current Git revision")
        return state
    report = evaluator.evaluate(run_dir / "baseline" / "evaluation")
    state = {
        "next_generation": 1,
        "current_commit": git.head(),
        "current_model": model,
        "current_adapter": None,
        "current_context": None,
        "current_tool_profile": None,
        "current_report": report.to_dict(),
    }
    write_json(path, state)
    return state


def _prepare_generation_dir(path: Path) -> None:
    if path.exists():
        index = 1
        while path.with_name(f"{path.name}-failed-{index:04d}").exists():
            index += 1
        path.rename(path.with_name(f"{path.name}-failed-{index:04d}"))
    path.mkdir(parents=True)


def _git_output(repo: Path, arguments: list[str]) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=repo, capture_output=True, text=True, check=True
    ).stdout


def _print(record: GenerationRecord) -> None:
    score = f" score={record.candidate_report['task_score']:.4f}" if record.candidate_report else ""
    print(
        f"generation={record.generation} layer={record.layer} decision={record.decision} "
        f"outcome={record.outcome}{score} reason={record.reason}",
        flush=True,
    )
