"""Freeze a complete evolved state and evaluate it without further evolution."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from strataevo.evolution.layers.model.evolution import VLLMAdapterRuntime


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def load_snapshot(path: Path) -> dict:
    snapshot = json.loads(path.read_text())
    if snapshot.get("adapter"):
        adapter_path = Path(snapshot["adapter"]["path"])
        if not adapter_path.is_absolute():
            snapshot["adapter"]["path"] = str((path.parent / adapter_path).resolve())
    return snapshot


def freeze(repo: Path, output: Path, state_file: Path | None, model: str) -> dict:
    if output.exists():
        raise FileExistsError(f"frozen snapshot already exists: {output}")
    state = json.loads(state_file.read_text()) if state_file else {}
    snapshot = {
        "source_state": str(state_file.resolve()) if state_file else None,
        "commit": git(repo, "rev-parse", state.get("current_commit", "HEAD")),
        "model": state.get("current_model", model),
        "base_model": model,
        "adapter": state.get("current_adapter"),
        "context": state.get("current_context"),
        "tools": state.get("current_tool_profile"),
    }
    if snapshot["adapter"] and not Path(snapshot["adapter"]["path"]).is_dir():
        raise FileNotFoundError("saved adapter directory is missing")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n")
    return snapshot


def export(repo: Path, snapshot_file: Path, output: Path) -> None:
    """Package source plus all inherited state, not the test-time feedback."""
    snapshot = load_snapshot(snapshot_file)
    if "base_model" not in snapshot:
        snapshot["base_model"] = (
            json.loads((Path(snapshot["adapter"]["path"]) / "adapter_config.json").read_text())[
                "base_model_name_or_path"
            ] if snapshot.get("adapter") else snapshot["model"]
        )
    output.mkdir(parents=True, exist_ok=False)
    with (output / "source.tar").open("wb") as archive:
        subprocess.run(["git", "-C", str(repo), "archive", snapshot["commit"]],
                       stdout=archive, check=True)
    snapshot["source_state"] = None
    if snapshot.get("adapter"):
        shutil.copytree(snapshot["adapter"]["path"], output / "adapter")
        snapshot["adapter"]["path"] = "adapter"
    for key in ("context", "tools"):
        if snapshot.get(key):
            snapshot[key].pop("path", None)
            (output / f"{key}.json").write_text(
                json.dumps(snapshot[key], indent=2, ensure_ascii=False) + "\n")
    (output / "frozen_state.json").write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n")
    (output / "README.md").write_text(
        "# Frozen StrataEvo\n\n"
        "Extract source.tar into a separate directory. Use the base model identified in "
        "the base_model field of frozen_state.json; model may be a runtime adapter alias. "
        "If adapter/ exists, load it with vLLM and use its served name. "
        "Pass context.json and tools.json when present via --context-file and "
        "--tool-profile-file. These files are inherited Agent state, not optional examples.\n\n"
        "For a new harness such as SWE, inject the same system/task prompt addenda and "
        "tool description addenda. Do not evolve or choose states using target-test feedback. "
        "Report baseline and final under the same target harness and model settings.\n"
    )


def evaluate(repo: Path, snapshot_file: Path, output: Path, args: argparse.Namespace) -> None:
    snapshot = load_snapshot(snapshot_file)
    output.mkdir(parents=True, exist_ok=False)
    (output / "frozen_state.json").write_text(snapshot_file.read_text())
    checkout = output / "source"
    subprocess.run(["git", "-C", str(repo), "worktree", "add", "--detach",
                    str(checkout), snapshot["commit"]], check=True)
    adapter = snapshot.get("adapter")
    runtime = VLLMAdapterRuntime(args.base_url)
    adapter_name = f"generalization-{output.name}"
    try:
        model = snapshot["model"]
        if adapter:
            runtime.activate(adapter_name, Path(adapter["path"]))
            model = adapter_name
        env = {**os.environ, "PYTHONPATH": str(checkout / "src")}
        for benchmark in args.benchmarks:
            command = [sys.executable, "-m", f"eval.{benchmark}.run",
                       "--model", model, "--base-url", args.base_url,
                       "--output-dir", str(output / benchmark), "--split", "test",
                       "--workers", str(args.workers), "--max-steps", str(args.max_steps),
                       "--temperature", "0", "--no-resume"]
            for key, flag in [("context", "--context-file"), ("tools", "--tool-profile-file")]:
                if snapshot.get(key):
                    path = output / f"{key}.json"
                    path.write_text(json.dumps(snapshot[key], ensure_ascii=False))
                    command.extend([flag, str(path)])
            with (output / f"{benchmark}.log").open("w") as log:
                subprocess.run(command, cwd=checkout, env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            summary = json.loads((output / benchmark / "summary.json").read_text())
            if summary["evaluated"] != summary["selected_tasks"]:
                raise RuntimeError(f"incomplete {benchmark} evaluation; see {output / benchmark}")
            print(summary, flush=True)
    finally:
        if adapter:
            runtime.deactivate(adapter_name)
        # Keep the detached source checkout as part of the experiment's provenance.


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "evaluate", "export"))
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--state-file", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--benchmarks", nargs="+", choices=("mbpp", "humaneval"),
                        default=["mbpp", "humaneval"])
    args = parser.parse_args(argv)
    if args.action == "freeze":
        freeze(args.repo.resolve(), args.snapshot.resolve(), args.state_file, args.model)
    else:
        if args.output_dir is None:
            parser.error("evaluate/export requires --output-dir")
        if args.action == "export":
            export(args.repo.resolve(), args.snapshot.resolve(), args.output_dir.resolve())
        else:
            evaluate(args.repo.resolve(), args.snapshot.resolve(), args.output_dir.resolve(), args)


if __name__ == "__main__":
    main()
