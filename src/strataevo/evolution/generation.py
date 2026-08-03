"""Execute one transactional self-evolution generation."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .diagnosis import DiagnosisReport, diagnose_evaluation
from .layers.architecture import CandidateEvaluationSession
from .layers.context import evolve_context
from .layers.model.evolution import activate_saved_adapter, evolve_model
from .layers.profile import ProfileEvolutionResult
from .layers.tools import evolve_tools
from .memory import EvolutionMemory, EvolutionMemoryEntry
from .mutator import mutate
from .plan import EvolutionPlanReport, plan_evolution
from .runtime.evaluation import Evaluator, create_evaluator, promotion_decision, validation_commands
from .runtime.repository import GitRepository
from .runtime.summary import write_run_summary
from .types import EvaluationReport, EvolutionConfig, GenerationRecord
from .utils.io import read_json, write_json


def run_one_generation(config_path: Path) -> int:
    config = EvolutionConfig.from_dict(read_json(config_path))
    repo = Path(config.repo).resolve()
    run_dir = config_path.parent
    state_path = run_dir / "state.json"
    memory = EvolutionMemory(run_dir / "evolution_memory.jsonl")
    git = GitRepository(repo, config.mutable_paths)
    git.ensure_clean()
    current_branch = _git_output(repo, ["branch", "--show-current"]).strip()
    if current_branch != config.branch:
        raise RuntimeError(f"expected branch {config.branch!r}, found {current_branch!r}")

    evaluator = create_evaluator(repo, config)
    write_json(run_dir / "evaluation_contract.json", evaluator.contract.to_dict())
    state = _load_or_create_state(state_path, git, evaluator, run_dir, config.model)
    write_run_summary(run_dir, memory.load(), state)
    if config.model_evolution:
        activate_saved_adapter(config, evaluator, state.get("current_adapter"))
    if state.get("current_context"):
        evaluator.set_context(state["current_context"])
    if state.get("current_tool_profile"):
        evaluator.set_tool_profile(state["current_tool_profile"])
    parent_report = EvaluationReport.from_dict(state["current_report"])
    if parent_report.task_score >= 1.0:
        state["completed"] = True
        state["completion_reason"] = "task score reached maximum 1.0"
        write_json(state_path, state)
        print("[evolution] stopping: pass@1 already reached 1.0000", flush=True)
        return 0

    commands = validation_commands(repo)
    generation = int(state["next_generation"])
    generation_dir = run_dir / f"generation-{generation:04d}"
    _prepare_generation_dir(generation_dir)
    parent_commit = git.head()
    diagnosis_path = generation_dir / "diagnosis.json"
    plan_path = generation_dir / "plan.json"

    try:
        diagnosis = diagnose_evaluation(config, parent_report, diagnosis_path, memory.latest())
        diagnosed_layers = {
            layer.value
            for item in diagnosis.diagnoses
            for layer in [item.primary_layer, *item.related_layers]
        }
        planning_history = memory.relevant(diagnosed_layers)
        plan_report = plan_evolution(
            config,
            diagnosis,
            parent_report,
            planning_history,
            plan_path,
            evaluator.contract,
        )
        layer = plan_report.plan.primary_layer.value
        if layer == "context":
            result = evolve_context(
                config,
                generation,
                generation_dir,
                parent_report,
                evaluator,
                diagnosis,
                plan_report,
                memory.relevant({"context"}),
                parent_context=state.get("current_context"),
                model_name=str(state.get("current_model", config.model)),
            )
            return _complete_external_generation(
                layer,
                result,
                state_path=state_path,
                state=state,
                generation_dir=generation_dir,
                generation=generation,
                parent_commit=parent_commit,
                diagnosis_path=diagnosis_path,
                diagnosis=diagnosis,
                plan_path=plan_path,
                plan_report=plan_report,
                parent_report=parent_report,
                memory=memory,
            )
        if layer == "tools":
            result = evolve_tools(
                config,
                generation,
                generation_dir,
                parent_report,
                evaluator,
                diagnosis,
                plan_report,
                memory.relevant({"tools"}),
                parent_profile=state.get("current_tool_profile"),
                model_name=str(state.get("current_model", config.model)),
            )
            return _complete_external_generation(
                layer,
                result,
                state_path=state_path,
                state=state,
                generation_dir=generation_dir,
                generation=generation,
                parent_commit=parent_commit,
                diagnosis_path=diagnosis_path,
                diagnosis=diagnosis,
                plan_path=plan_path,
                plan_report=plan_report,
                parent_report=parent_report,
                memory=memory,
            )
        if layer == "model" and config.model_evolution:
            selected_diagnosis = diagnosis.diagnoses[
                plan_report.plan.target_diagnosis
            ]
            result = evolve_model(
                config,
                generation,
                generation_dir,
                parent_report,
                evaluator,
                parent_model=str(state.get("current_model", config.model)),
                parent_adapter=state.get("current_adapter"),
                diagnosis=selected_diagnosis,
                plan=plan_report.plan,
            )
            return _complete_external_generation(
                layer,
                result,
                state_path=state_path,
                state=state,
                generation_dir=generation_dir,
                generation=generation,
                parent_commit=parent_commit,
                diagnosis_path=diagnosis_path,
                diagnosis=diagnosis,
                plan_path=plan_path,
                plan_report=plan_report,
                parent_report=parent_report,
                memory=memory,
            )
        selected_diagnosis = diagnosis.diagnoses[plan_report.plan.target_diagnosis]
        mutation_layers = {
            selected_diagnosis.primary_layer.value,
            *(layer.value for layer in selected_diagnosis.related_layers),
        }
        mutation_history = memory.relevant(mutation_layers)
        candidate_session = CandidateEvaluationSession(
            repo,
            git,
            evaluator,
            parent_report,
            generation_dir,
            commands,
            config.max_eval_attempts,
        )
        agent_result = mutate(
            repo,
            config,
            generation,
            parent_report,
            diagnosis,
            plan_report,
            mutation_history,
            generation_dir,
            commands,
            evaluator.contract,
            candidate_session.evaluate,
        )
        write_json(
            generation_dir / "agent_result.json",
            {
                "output": agent_result.output,
                "messages": [message.to_dict() for message in agent_result.messages],
                "usage": {
                    "input_tokens": agent_result.usage.input_tokens,
                    "output_tokens": agent_result.usage.output_tokens,
                },
                "steps": agent_result.steps,
                "stop_reason": agent_result.stop_reason,
            },
        )
        candidate_session.ensure_evaluated()
        best_attempt = candidate_session.restore_best()
        attempts = candidate_session.to_dicts()
        if best_attempt is None:
            last_attempt = attempts[-1] if attempts else None
            outcome_type = last_attempt["outcome_type"] if last_attempt else "no_change"
            reason = (
                last_attempt["reason"] if last_attempt else "meta-agent produced no source changes"
            )
            record = _record(
                generation,
                parent_commit,
                None,
                "rejected",
                outcome_type,
                reason,
                last_attempt["changed_paths"] if last_attempt else [],
                (
                    Path(last_attempt["patch_path"])
                    if last_attempt and last_attempt["patch_path"]
                    else None
                ),
                diagnosis_path,
                diagnosis,
                plan_path,
                plan_report,
                parent_report,
                None,
                None,
                agent_result,
                attempts,
            )
            _finish_generation(
                state_path,
                state,
                generation_dir,
                record,
                memory,
                diagnosis,
                plan_report,
                agent_result.output,
            )
            _print_generation(record)
            return 0

        changed_paths = best_attempt.changed_paths
        patch_path = Path(best_attempt.patch_path) if best_attempt.patch_path else None
        selection_report = EvaluationReport.from_dict(best_attempt.report)
        accepted, reason = promotion_decision(parent_report, selection_report)
        candidate_report = selection_report
        reason = f"candidate screening: {reason}"
        if accepted:
            resulting_commit = git.commit(
                f"evolve: generation {generation} pass@1 {candidate_report.task_score:.6f}"
            )
            state["current_commit"] = resulting_commit
            state["current_report"] = candidate_report.to_dict()
            decision = "accepted"
        else:
            git.rollback()
            resulting_commit = None
            decision = "rejected"
        record = _record(
            generation,
            parent_commit,
            resulting_commit,
            decision,
            "accepted" if accepted else "benchmark_rejected",
            reason,
            changed_paths,
            patch_path,
            diagnosis_path,
            diagnosis,
            plan_path,
            plan_report,
            parent_report,
            None,
            candidate_report,
            agent_result,
            attempts,
        )
        _finish_generation(
            state_path,
            state,
            generation_dir,
            record,
            memory,
            diagnosis,
            plan_report,
            agent_result.output,
        )
        _print_generation(record)
        return 0
    except KeyboardInterrupt:
        if git.changed_paths():
            git.rollback()
        write_json(
            generation_dir / "failure.json",
            {"error_type": "KeyboardInterrupt", "error": "interrupted by user"},
        )
        print("\n[evolution] interrupted; candidate changes rolled back", flush=True)
        raise
    except Exception as error:
        if git.changed_paths():
            git.rollback()
        write_json(
            generation_dir / "failure.json",
            {"error_type": type(error).__name__, "error": str(error)},
        )
        raise


def _complete_external_generation(
    layer: str,
    result: ProfileEvolutionResult,
    *,
    state_path: Path,
    state: dict[str, Any],
    generation_dir: Path,
    generation: int,
    parent_commit: str,
    diagnosis_path: Path,
    diagnosis: DiagnosisReport,
    plan_path: Path,
    plan_report: EvolutionPlanReport,
    parent_report: EvaluationReport,
    memory: EvolutionMemory,
) -> int:
    candidate_fields = {
        "model_candidate": None,
        "context_candidate": None,
        "tool_candidate": None,
    }
    candidate_fields[
        {
            "model": "model_candidate",
            "context": "context_candidate",
            "tools": "tool_candidate",
        }[layer]
    ] = result.candidate
    if result.decision == "accepted":
        state["current_report"] = result.candidate_report.to_dict()
        if layer == "model":
            state["current_model"] = result.candidate["name"]
            state["current_adapter"] = result.candidate
        elif layer == "context":
            state["current_context"] = result.candidate
        elif layer == "tools":
            state["current_tool_profile"] = result.candidate

    record = _record(
        generation,
        parent_commit,
        None,
        result.decision,
        result.outcome_type,
        result.reason,
        [],
        None,
        diagnosis_path,
        diagnosis,
        plan_path,
        plan_report,
        parent_report,
        result.promotion_parent_report,
        result.candidate_report,
        None,
        [],
        evolution_usage=(
            int(getattr(result, "input_tokens", 0)),
            int(getattr(result, "output_tokens", 0)),
        ),
        **candidate_fields,
    )
    label = "LoRA" if layer == "model" else layer.capitalize()
    _finish_generation(
        state_path,
        state,
        generation_dir,
        record,
        memory,
        diagnosis,
        plan_report,
        f"{label} candidate: {result.reason}",
    )
    _print_generation(record)
    return 0


def _load_or_create_state(
    path: Path,
    git: GitRepository,
    evaluator: Evaluator,
    run_dir: Path,
    model: str,
) -> dict[str, Any]:
    if path.exists():
        state = read_json(path)
        if state["current_commit"] != git.head():
            raise RuntimeError("evolution state does not match the current Git revision")
        if "baseline_task_score" not in state:
            summary_path = run_dir / "baseline" / "evaluation" / "summary.json"
            baseline = read_json(summary_path) if summary_path.is_file() else None
            state["baseline_task_score"] = float(
                baseline["pass_at_1"]
                if baseline
                else state["current_report"]["task_score"]
            )
        state.setdefault("current_model", model)
        state.setdefault("current_adapter", None)
        state.setdefault("current_context", None)
        state.setdefault("current_tool_profile", None)
        return state
    report = evaluator.evaluate(run_dir / "baseline" / "evaluation")
    state = {
        "next_generation": 1,
        "current_commit": git.head(),
        "current_model": model,
        "current_adapter": None,
        "current_context": None,
        "current_tool_profile": None,
        "baseline_task_score": report.task_score,
        "current_report": report.to_dict(),
    }
    write_json(path, state)
    return state


def _record(
    generation: int,
    parent_commit: str,
    resulting_commit: str | None,
    decision: str,
    outcome_type: str,
    reason: str,
    changed_paths: list[str],
    patch_path: Path | None,
    diagnosis_path: Path,
    diagnosis: DiagnosisReport,
    plan_path: Path,
    plan_report: EvolutionPlanReport,
    parent_report: EvaluationReport,
    promotion_parent_report: EvaluationReport | None,
    candidate_report: EvaluationReport | None,
    agent_result: Any,
    evaluation_attempts: list[dict[str, Any]],
    *,
    model_candidate: dict[str, Any] | None = None,
    context_candidate: dict[str, Any] | None = None,
    tool_candidate: dict[str, Any] | None = None,
    evolution_usage: tuple[int, int] = (0, 0),
) -> GenerationRecord:
    return GenerationRecord(
        generation=generation,
        parent_commit=parent_commit,
        resulting_commit=resulting_commit,
        decision=decision,
        outcome_type=outcome_type,
        reason=reason,
        changed_paths=changed_paths,
        patch_path=str(patch_path) if patch_path else None,
        diagnosis_path=str(diagnosis_path),
        diagnosed_layers=list(
            dict.fromkeys(item.primary_layer.value for item in diagnosis.diagnoses)
        ),
        plan_path=str(plan_path),
        planned_layer=plan_report.plan.primary_layer.value,
        plan_hypothesis=plan_report.plan.hypothesis,
        parent_report=parent_report.to_dict(),
        promotion_parent_report=(
            promotion_parent_report.to_dict() if promotion_parent_report else None
        ),
        candidate_report=candidate_report.to_dict() if candidate_report else None,
        agent_stop_reason=(
            agent_result.stop_reason
            if agent_result
            else "context_evolution"
            if context_candidate
            else "tool_evolution"
            if tool_candidate
            else "model_evolution"
        ),
        agent_steps=agent_result.steps if agent_result else 0,
        input_tokens=agent_result.usage.input_tokens if agent_result else evolution_usage[0],
        output_tokens=agent_result.usage.output_tokens if agent_result else evolution_usage[1],
        evaluation_attempts=evaluation_attempts,
        model_candidate=model_candidate,
        context_candidate=context_candidate,
        tool_candidate=tool_candidate,
    )


def _finish_generation(
    state_path: Path,
    state: dict[str, Any],
    generation_dir: Path,
    record: GenerationRecord,
    memory: EvolutionMemory,
    diagnosis: DiagnosisReport,
    plan_report: EvolutionPlanReport,
    agent_output: str,
) -> None:
    entry = EvolutionMemoryEntry.from_generation(record, diagnosis, plan_report, agent_output)
    record.causal_trace = entry.causal_trace
    write_json(generation_dir / "record.json", record.to_dict())
    memory.append(entry)
    state["next_generation"] = record.generation + 1
    write_json(state_path, state)
    write_run_summary(state_path.parent, memory.load(), state)


def _print_generation(record: GenerationRecord) -> None:
    candidate = record.candidate_report
    metrics = f" score={float(candidate['task_score']):.4f}" if candidate else ""
    print(
        f"generation={record.generation} decision={record.decision} "
        f"outcome={record.outcome_type}{metrics} reason={record.reason}",
        flush=True,
    )


def _prepare_generation_dir(path: Path) -> None:
    if path.exists():
        if (path / "record.json").exists():
            raise RuntimeError(
                f"completed generation already exists but state was not advanced: {path}"
            )
        sequence = 1
        while True:
            archive = path.with_name(f"{path.name}-failed-{sequence:04d}")
            if not archive.exists():
                path.rename(archive)
                break
            sequence += 1
    path.mkdir(parents=True, exist_ok=False)


def _git_output(repo: Path, arguments: list[str]) -> str:
    return subprocess.check_output(["git", *arguments], cwd=repo, text=True)
