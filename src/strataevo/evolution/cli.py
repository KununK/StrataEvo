"""Command-line entry point for repository-level self-evolution."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from .generation import _prepare_generation_dir, run_one_generation
from .runtime.evaluation import BENCHMARKS
from .runtime.repository import GitRepository
from .types import DEFAULT_MUTABLE_PATHS, EvolutionConfig
from .utils.io import read_json, write_json

__all__ = ["_prepare_generation_dir", "main", "parse_args", "run_one_generation"]


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
    parser.add_argument("--force-layer", choices=("model", "context", "tools", "architecture"))
    parser.add_argument("--sft-device", default="1")
    parser.add_argument("--sft-epochs", type=int, default=1)
    parser.add_argument("--sft-max-samples", type=int, default=32)
    parser.add_argument("--sft-max-length", type=int, default=4096)
    parser.add_argument("--sft-lora-rank", type=int, default=8)
    parser.add_argument("--sft-learning-rate", type=float, default=1e-4)
    parser.add_argument("--repair-attempts", type=int, default=2)
    parser.add_argument("--repair-temperature", type=float, default=0.2)
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
    runs_dir = f"runs_{args.branch.replace('/', '_')}"
    run_dir = repo / "evolution" / runs_dir / run_name
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
            sft_epochs=args.sft_epochs,
            sft_max_samples=args.sft_max_samples,
            sft_max_length=args.sft_max_length,
            sft_lora_rank=args.sft_lora_rank,
            sft_learning_rate=args.sft_learning_rate,
            repair_attempts=args.repair_attempts,
            repair_temperature=args.repair_temperature,
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


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "generations": args.generations,
        "mutator-max-steps": args.mutator_max_steps,
        "mutator-rounds": args.mutator_rounds,
        "max-eval-attempts": args.max_eval_attempts,
        "eval-workers": args.eval_workers,
        "benchmark-max-steps": args.benchmark_max_steps,
        "sft-epochs": args.sft_epochs,
        "sft-max-samples": args.sft_max_samples,
        "sft-max-length": args.sft_max_length,
        "sft-lora-rank": args.sft_lora_rank,
        "repair-attempts": args.repair_attempts,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if args.eval_offset < 0 or args.eval_limit is not None and args.eval_limit <= 0:
        raise ValueError("eval-offset must be non-negative and eval-limit must be positive")
    if args.sft_learning_rate <= 0:
        raise ValueError("sft-learning-rate must be positive")
    if not 0.0 <= args.repair_temperature <= 2.0:
        raise ValueError("repair-temperature must be between 0 and 2")
    if args.force_layer == "model" and not args.enable_model_evolution:
        raise ValueError("--force-layer model requires --enable-model-evolution")


if __name__ == "__main__":
    raise SystemExit(main())
