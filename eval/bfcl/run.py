"""Evaluate Tinyagent on the official BFCL V4 multi-turn base suite."""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from eval.coding_agent import (
    add_common_arguments,
    run_benchmark,
    validate_common_arguments,
)
from tinyagent import Agent, Message, Model, ToolRegistry, Usage

from .official import OfficialBFCL, format_call

SYSTEM_PROMPT = """You are an agent completing a multi-turn tool-use task.
Use only the supplied tools when they are needed. Preserve relevant state across turns, use tool
results as evidence, and do not invent successful actions. Finish each user turn with a concise
answer once the requested actions are complete."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Tinyagent on BFCL V4")
    add_common_arguments(
        parser,
        dataset="gorilla-llm/Berkeley-Function-Calling-Leaderboard",
        output_dir="eval/outputs/bfcl/qwen3-coder-agent",
    )
    parser.add_argument("--bfcl-root", help="official Gorilla or bfcl_eval source directory")
    parser.add_argument("--category", default="multi_turn_base", choices=("multi_turn_base",))
    return parser.parse_args(argv)


def load_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.dataset_file:
        raise ValueError("BFCL uses its official paired prompt and ground-truth files")
    official = OfficialBFCL(args.bfcl_root)
    tasks = official.load_tasks(args.category)
    for task in tasks:
        task["_bfcl_root"] = str(official.root.parent)
    end = None if args.limit is None else args.offset + args.limit
    return tasks[args.offset : end]


def run_agent_task(
    task: dict[str, Any],
    model: Model,
    candidate_path: Path,
    *,
    max_steps: int = 12,
    test_timeout: float = 10.0,
    system_prompt: str = SYSTEM_PROMPT,
    user_prompt: str = "",
    tool_description_addenda: dict[str, str] | None = None,
    initial_guidance: str = "",
) -> tuple[dict[str, Any], dict[str, Any]]:
    del candidate_path, test_timeout
    started = time.perf_counter()
    official = OfficialBFCL(task.get("_bfcl_root"))
    tools = ToolRegistry(official.create_tools(task, tool_description_addenda))
    agent = Agent(model=model, tools=tools, system_prompt=system_prompt, max_steps=max_steps)
    history: list[Message] = []
    calls: list[list[list[str]]] = []
    usage = Usage()
    steps = 0
    stop_reason = "completed"
    output = ""
    error = ""
    try:
        for turn_index, turn in enumerate(task["question"]):
            prompt = "\n".join(str(message.get("content", "")) for message in turn)
            if user_prompt:
                prompt += "\n\n" + user_prompt
            if turn_index == 0 and initial_guidance:
                prompt += "\n\n" + initial_guidance
            result = agent.run(prompt, history=history)
            history = result.messages
            usage = usage + result.usage
            steps += result.steps
            stop_reason = result.stop_reason
            output = result.output
            calls.append(_turn_calls(result.messages, prompt))
        checked = official.check(task, calls)
        passed = bool(checked.get("valid"))
        status = "pass" if passed else str(checked.get("error_type", "incorrect_calls"))
        error = "" if passed else str(checked.get("error_message", checked))
    except Exception as exception:
        passed = False
        status = "agent_error"
        error = f"{type(exception).__name__}: {exception}"

    elapsed = time.perf_counter() - started
    generation = {
        "task_id": task["task_id"],
        "prompt": task["prompt"],
        "candidate_path": None,
        "agent_output": output,
        "agent_error": error if status == "agent_error" else "",
        "stop_reason": stop_reason,
        "steps": steps,
        "usage": {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens},
        "messages": [message.to_dict() for message in history],
        "function_calls": calls,
        "generation_seconds": elapsed,
    }
    result = {
        "task_id": task["task_id"],
        "candidate_path": None,
        "artifact_expected": False,
        "agent_stop_reason": stop_reason,
        "agent_steps": steps,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "generation_seconds": elapsed,
        "passed": passed,
        "compile_passed": True,
        "status": status,
        "returncode": 0 if passed else 1,
        "stdout": output,
        "stderr": error,
        "test_seconds": 0.0,
    }
    return generation, result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_common_arguments(args)
    _bypass_loopback_proxy(args.base_url)
    tasks = load_tasks(args)
    return run_benchmark(
        args,
        tasks,
        run_agent_task,
        benchmark="bfcl",
        description="BFCL V4 Agent",
        dataset=args.dataset,
        split=args.category,
        system_prompt=SYSTEM_PROMPT,
        user_prompt="",
    )


def _turn_calls(messages: list[Message], prompt: str) -> list[list[str]]:
    user_index = max(
        index
        for index, message in enumerate(messages)
        if message.role == "user" and message.content == prompt
    )
    return [
        [format_call(call.name, call.arguments) for call in message.tool_calls]
        for message in messages[user_index + 1 :]
        if message.role == "assistant" and message.tool_calls
    ]


def _bypass_loopback_proxy(base_url: str) -> None:
    host = urlsplit(base_url).hostname
    if host not in {"localhost", "127.0.0.1", "::1"}:
        return
    for name in ("NO_PROXY", "no_proxy"):
        values = [item for item in os.environ.get(name, "").split(",") if item]
        if host not in values:
            os.environ[name] = ",".join([*values, host])


if __name__ == "__main__":
    raise SystemExit(main())
