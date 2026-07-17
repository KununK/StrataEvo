"""A practical terminal frontend built entirely on the public API."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .agent import Agent
from .events import Event
from .model import DEFAULT_BASE_URL, DEFAULT_MODEL, OpenAICompatibleModel
from .session import SessionStore
from .tool import ToolRegistry, allow_all, terminal_approval
from .workspace import Workspace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tinyagent", description="A small, understandable coding agent"
    )
    parser.add_argument("prompt", nargs="*", help="task; omit it for an interactive session")
    parser.add_argument("--model", default=os.getenv("TINYAGENT_MODEL", DEFAULT_MODEL))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--workspace", default=".")
    parser.add_argument(
        "--session",
        default="default",
        help="session name, or empty to disable persistence",
    )
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--yes", action="store_true", help="approve all write and shell tool calls")
    parser.add_argument(
        "--verbose", action="store_true", help="show model and tool lifecycle events"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.workspace).resolve()
    if not root.is_dir():
        print(f"error: workspace is not a directory: {root}", file=sys.stderr)
        return 2

    model = OpenAICompatibleModel(model=args.model, base_url=args.base_url)
    registry = ToolRegistry(Workspace(root).tools())
    store = SessionStore(root / ".tinyagent" / "sessions") if args.session else None
    agent = Agent(
        model=model,
        tools=registry,
        max_steps=args.max_steps,
        approval=allow_all if args.yes else terminal_approval,
        session_store=store,
    )
    if args.verbose:
        agent.events.subscribe(_print_event)

    prompt = " ".join(args.prompt).strip()
    if prompt:
        return _run_once(agent, prompt, args.session or None)

    print(f"Tinyagent | model={args.model} | workspace={root}")
    print("Enter a task, or /exit to quit.")
    while True:
        try:
            prompt = input("tinyagent> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if prompt in {"/exit", "/quit"}:
            return 0
        if prompt:
            _run_once(agent, prompt, args.session or None)


def _run_once(agent: Agent, prompt: str, session_id: str | None) -> int:
    try:
        result = agent.run(prompt, session_id=session_id)
    except (RuntimeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if result.output:
        print(result.output)
    if result.stop_reason != "completed":
        print(f"stopped: {result.stop_reason}", file=sys.stderr)
        return 1
    return 0


def _print_event(event: Event) -> None:
    if event.type == "tool_start":
        print(f"[tool] {event.data['name']} {event.data['arguments']}", file=sys.stderr)
    elif event.type == "model_start":
        print(f"[model] step {event.data['step']}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
