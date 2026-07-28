"""CLI and generation worker for repository-level recursive self-evolution."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .attempts import CandidateEvaluationSession
from .diagnosis import DiagnosisReport, diagnose_evaluation
from .evaluation import BENCHMARKS, Evaluator, create_evaluator, validation_commands
from .git import GitRepository
from .io import read_json, write_json
from .memory import EvolutionMemory, EvolutionMemoryEntry
from .model_evolution import activate_saved_adapter, evolve_model
from .mutator import mutate
from .plan import EvolutionPlanReport, plan_evolution
from .types import DEFAULT_MUTABLE_PATHS, EvaluationReport, EvolutionConfig, GenerationRecord


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Let StrataEvo evaluate, rewrite, and version its own agent implementation"
    )
    parser.add_argument("--repo", default=".")
    parser.add_argument("--run-name")
    parser.add_argument("--branch", default="evo")
    parser.add_argument("--benchmark", choices=tuple(BENCHMARKS), default="humaneval")
    parser.add_argument("--generations", type=int, default=1)
    parser.add_argument("--model", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--mutator-max-steps", type=int, default=200)
    parser.add_argument("--mutator-rounds", type=int, default=5)
    parser.add_argument("--max-eval-attempts", type=int, default=5)
    parser.add_argument("--eval-limit", type=int)
    parser.add_argument("--eval-offset", type=int, default=0)
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--benchmark-max-steps", type=int, default=12)
    parser.add_argument("--enable-model-evolution", action="store_true")
    parser.add_argument("--force-layer", choices=("model",))
    parser.add_argument("--sft-device", default="1")
    parser.add_argument("--sft-max-steps", type=int, default=20)
    parser.add_argument("--sft-max-samples", type=int, default=32)
    parser.add_argument("--sft-max-length", type=int, default=4096)
    parser.add_argument("--sft-lora-rank", type=int, default=8)
    parser.add_argument("--sft-learning-rate", type=float, default=1e-4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker-config", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker_config:
        return run_one_generation(Path(args.worker_config))
    _validate_args(args)
    repo = Path(args.repo).resolve()
    run_name = args.run_name or datetime.now(UTC).strftime("run-%Y%m%d-%H%M%S")
    run_dir = repo / "evolution" / "runs" / run_name
    config_path = run_dir / "config.json"

    git = GitRepository(repo, DEFAULT_MUTABLE_PATHS)
    git.ensure_clean()
    git.ensure_branch(args.branch)

    if config_path.exists():
        if not args.resume:
            raise RuntimeError(f"run already exists; use --resume: {run_dir}")
        config = EvolutionConfig.from_dict(read_json(config_path))
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        config = EvolutionConfig(
            repo=str(repo),
            run_name=run_name,
            branch=args.branch,
            benchmark=args.benchmark,
            model=args.model,
            base_url=args.base_url,
            mutator_max_steps=args.mutator_max_steps,
            mutator_rounds=args.mutator_rounds,
            max_eval_attempts=args.max_eval_attempts,
            eval_limit=args.eval_limit,
            eval_offset=args.eval_offset,
            eval_workers=args.eval_workers,
            benchmark_max_steps=args.benchmark_max_steps,
            model_evolution=args.enable_model_evolution,
            force_layer=args.force_layer,
            sft_device=args.sft_device,
            sft_max_steps=args.sft_max_steps,
            sft_max_samples=args.sft_max_samples,
            sft_max_length=args.sft_max_length,
            sft_lora_rank=args.sft_lora_rank,
            sft_learning_rate=args.sft_learning_rate,
        )
        write_json(config_path, config.to_dict())

    for _ in range(args.generations):
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "strataevo.evolution.cli",
                    "--worker-config",
                    str(config_path),
                ],
                cwd=repo,
                check=False,
            )
        except KeyboardInterrupt:
            repository = GitRepository(repo, config.mutable_paths)
            if repository.changed_paths():
                repository.rollback()
            print(
                "\n[evolution] interrupted; uncommitted self-modifications rolled back",
                flush=True,
            )
            return 130
        if completed.returncode != 0:
            return completed.returncode
        state_path = run_dir / "state.json"
        if state_path.is_file() and read_json(state_path).get("completed", False):
            break
    return 0


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
    if config.model_evolution:
        activate_saved_adapter(config, evaluator, state.get("current_adapter"))
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
        if plan_report.plan.primary_layer.value == "model" and config.model_evolution:
            model_result = evolve_model(
                config,
                generation,
                generation_dir,
                parent_report,
                evaluator,
                parent_model=str(state.get("current_model", config.model)),
                parent_adapter=state.get("current_adapter"),
            )
            if model_result.decision == "accepted":
                state["current_model"] = model_result.candidate["name"]
                state["current_adapter"] = model_result.candidate
                state["current_report"] = model_result.candidate_report.to_dict()
            record = _record(
                generation,
                parent_commit,
                None,
                model_result.decision,
                model_result.outcome_type,
                model_result.reason,
                [],
                None,
                diagnosis_path,
                diagnosis,
                plan_path,
                plan_report,
                parent_report,
                model_result.promotion_parent_report,
                model_result.candidate_report,
                None,
                [],
                model_candidate=model_result.candidate,
            )
            summary = (
                f"LoRA candidate {model_result.candidate['name']}: "
                f"{model_result.reason}"
            )
            _finish_generation(
                state_path,
                state,
                generation_dir,
                record,
                memory,
                diagnosis,
                plan_report,
                summary,
            )
            _print_generation(record)
            return 0
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
        eligible, reason = _promotion_decision(parent_report, selection_report)
        promotion_parent_report: EvaluationReport | None = None
        candidate_report = selection_report
        if eligible:
            promotion_parent_report, candidate_report = candidate_session.confirm(best_attempt)
            accepted, reason = _promotion_decision(promotion_parent_report, candidate_report)
            reason = f"fresh promotion comparison: {reason}"
        else:
            accepted = False
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
            promotion_parent_report,
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
        state.setdefault("current_model", model)
        state.setdefault("current_adapter", None)
        return state
    report = evaluator.evaluate(run_dir / "baseline" / "evaluation")
    state = {
        "next_generation": 1,
        "current_commit": git.head(),
        "current_model": model,
        "current_adapter": None,
        "current_report": report.to_dict(),
    }
    write_json(path, state)
    return state


def _promotion_decision(
    parent: EvaluationReport,
    candidate: EvaluationReport,
) -> tuple[bool, str]:
    if candidate.task_score <= parent.task_score:
        return False, (
            f"pass@1 {candidate.task_score:.6f} did not exceed parent {parent.task_score:.6f}"
        )
    return True, "pass@1 strictly improved"


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
        agent_stop_reason=agent_result.stop_reason if agent_result else "model_evolution",
        agent_steps=agent_result.steps if agent_result else 0,
        input_tokens=agent_result.usage.input_tokens if agent_result else 0,
        output_tokens=agent_result.usage.output_tokens if agent_result else 0,
        evaluation_attempts=evaluation_attempts,
        model_candidate=model_candidate,
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
    write_json(generation_dir / "record.json", record.to_dict())
    memory.append(
        EvolutionMemoryEntry.from_generation(record, diagnosis, plan_report, agent_output)
    )
    state["next_generation"] = record.generation + 1
    write_json(state_path, state)


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "generations": args.generations,
        "mutator-max-steps": args.mutator_max_steps,
        "mutator-rounds": args.mutator_rounds,
        "max-eval-attempts": args.max_eval_attempts,
        "eval-workers": args.eval_workers,
        "benchmark-max-steps": args.benchmark_max_steps,
        "sft-max-steps": args.sft_max_steps,
        "sft-max-samples": args.sft_max_samples,
        "sft-max-length": args.sft_max_length,
        "sft-lora-rank": args.sft_lora_rank,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if args.eval_offset < 0 or args.eval_limit is not None and args.eval_limit <= 0:
        raise ValueError("eval-offset must be non-negative and eval-limit must be positive")
    if args.sft_learning_rate <= 0:
        raise ValueError("sft-learning-rate must be positive")
    if args.force_layer == "model" and not args.enable_model_evolution:
        raise ValueError("--force-layer model requires --enable-model-evolution")


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


if __name__ == "__main__":
    raise SystemExit(main())
