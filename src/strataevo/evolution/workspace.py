"""Tools for inspecting and modifying the evolvable project."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

from tinyagent import Tool, tool

HIDDEN_ROOTS = {".git", ".venv", ".tinyagent", ".pytest_cache", ".ruff_cache", "__pycache__"}
PROTECTED_WRITE_ROOTS = {
    ".git",
    ".venv",
    ".tinyagent",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
    "eval/logs",
    "eval/outputs",
    "evolution/runs",
}
SECRET_SUFFIXES = {".key", ".pem"}


class SelfWorkspace:
    def __init__(
        self,
        root: str | Path,
        mutable_paths: list[str],
        validation_commands: list[list[str]],
        *,
        candidate_evaluator: Callable[[], str] | None = None,
        command_timeout: float = 300.0,
        max_output: int = 30_000,
    ) -> None:
        self.root = Path(root).resolve()
        self.mutable_roots = tuple(self._resolve(path) for path in mutable_paths)
        self.validation_commands = validation_commands
        self.candidate_evaluator = candidate_evaluator
        self.command_timeout = command_timeout
        self.max_output = max_output

    def tools(self) -> list[Tool]:
        @tool
        def list_files(path: str = ".") -> str:
            """List one repository directory."""
            target = self._resolve(path)
            if not target.is_dir():
                raise ValueError(f"not a directory: {path}")
            items = [
                str(item.relative_to(self.root))
                for item in sorted(target.iterdir())
                if item.name not in HIDDEN_ROOTS
            ]
            return self._limit("\n".join(items))

        @tool
        def read_file(path: str, start_line: int = 1, end_line: int = 500) -> str:
            """Read a repository text file with line numbers."""
            if start_line < 1 or end_line < start_line:
                raise ValueError("invalid line range")
            lines = self._resolve(path).read_text(encoding="utf-8").splitlines()
            output = "\n".join(
                f"{number:>6}  {lines[number - 1]}"
                for number in range(start_line, min(end_line, len(lines)) + 1)
            )
            return self._limit(output)

        @tool
        def search_files(query: str, path: str = ".") -> str:
            """Search repository text files for a literal string."""
            target = self._resolve(path)
            paths = [target] if target.is_file() else target.rglob("*")
            matches: list[str] = []
            for candidate in paths:
                if not candidate.is_file() or any(part in HIDDEN_ROOTS for part in candidate.parts):
                    continue
                try:
                    for number, line in enumerate(
                        candidate.read_text(encoding="utf-8").splitlines(), 1
                    ):
                        if query in line:
                            matches.append(f"{candidate.relative_to(self.root)}:{number}:{line}")
                except (UnicodeDecodeError, OSError):
                    continue
                if len(matches) >= 200:
                    break
            return self._limit("\n".join(matches) or "No matches")

        @tool(requires_approval=True)
        def write_file(path: str, content: str) -> str:
            """Create or replace a file in the evolvable project."""
            target = self._resolve_mutable(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return f"Wrote {len(content.encode())} bytes to {path}"

        @tool(requires_approval=True)
        def replace_text(path: str, old: str, new: str) -> str:
            """Replace one exact occurrence; use replace_lines after an exact-match failure."""
            target = self._resolve_mutable(path)
            content = target.read_text(encoding="utf-8")
            count = content.count(old)
            if count != 1:
                raise ValueError(f"expected one occurrence, found {count}")
            target.write_text(content.replace(old, new), encoding="utf-8")
            return f"Updated {path}"

        @tool(requires_approval=True)
        def replace_lines(path: str, start_line: int, end_line: int, content: str) -> str:
            """Replace an inclusive line range previously observed with read_file."""
            if start_line < 1 or end_line < start_line:
                raise ValueError("invalid line range")
            target = self._resolve_mutable(path)
            original = target.read_text(encoding="utf-8")
            lines = original.splitlines(keepends=True)
            if end_line > len(lines):
                raise ValueError(f"end_line {end_line} exceeds file length {len(lines)}")
            replacement = content
            if replacement and not replacement.endswith(("\n", "\r")):
                replacement += "\n"
            lines[start_line - 1 : end_line] = replacement.splitlines(keepends=True)
            target.write_text("".join(lines), encoding="utf-8")
            return f"Updated lines {start_line}-{end_line} in {path}"

        @tool(requires_approval=True)
        def delete_file(path: str) -> str:
            """Delete one file from the evolvable project."""
            target = self._resolve_mutable(path)
            if not target.is_file():
                raise ValueError(f"not a file: {path}")
            target.unlink()
            return f"Deleted {path}"

        @tool
        def show_diff() -> str:
            """Show the current uncommitted self-modification diff."""
            completed = subprocess.run(
                ["git", "diff", "HEAD", "--", *self._relative_mutable_paths()],
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            return self._limit(completed.stdout or "No tracked diff")

        @tool
        def run_validation() -> str:
            """Run the fixed syntax, lint, and unit-test checks for the modified agent."""
            outputs = []
            for command in self.validation_commands:
                completed = subprocess.run(
                    command,
                    cwd=self.root,
                    capture_output=True,
                    text=True,
                    timeout=self.command_timeout,
                    check=False,
                )
                output = completed.stdout + completed.stderr
                outputs.append(f"$ {' '.join(command)}\nexit_code={completed.returncode}\n{output}")
                if completed.returncode != 0:
                    break
            return self._limit("\n\n".join(outputs))

        @tool
        def evaluate_candidate() -> str:
            """Evaluate the current self-modification and return benchmark feedback."""
            if self.candidate_evaluator is None:
                raise RuntimeError("candidate evaluation is unavailable")
            return self._limit(self.candidate_evaluator())

        tools = [
            list_files,
            read_file,
            search_files,
            write_file,
            replace_text,
            replace_lines,
            delete_file,
            show_diff,
            run_validation,
        ]
        if self.candidate_evaluator is not None:
            tools.append(evaluate_candidate)
        return tools

    def _resolve(self, path: str) -> Path:
        if not path or "\x00" in path:
            raise ValueError("invalid path")
        target = (self.root / path).resolve()
        if not target.is_relative_to(self.root):
            raise PermissionError(f"path escapes repository: {path}")
        relative = target.relative_to(self.root)
        if relative.parts and relative.parts[0] in {".git", ".venv"}:
            raise PermissionError(f"path is protected runtime state: {path}")
        if any(part == ".env" or part.startswith(".env.") for part in relative.parts):
            raise PermissionError(f"path may contain credentials: {path}")
        if target.suffix.lower() in SECRET_SUFFIXES:
            raise PermissionError(f"path may contain credentials: {path}")
        return target

    def _resolve_mutable(self, path: str) -> Path:
        target = self._resolve(path)
        if not any(target.is_relative_to(root) for root in self.mutable_roots):
            raise PermissionError(f"path is outside evolvable project: {path}")
        relative = target.relative_to(self.root).as_posix()
        if any(
            relative == root or relative.startswith(root + "/") for root in PROTECTED_WRITE_ROOTS
        ):
            raise PermissionError(f"path is protected runtime state: {path}")
        ignored = subprocess.run(
            ["git", "check-ignore", "--quiet", "--", relative],
            cwd=self.root,
            check=False,
        )
        if ignored.returncode == 0:
            raise PermissionError(f"path is ignored runtime state: {path}")
        if ignored.returncode not in {0, 1}:
            raise RuntimeError(f"git check-ignore failed for {path}")
        return target

    def _relative_mutable_paths(self) -> list[str]:
        return [str(path.relative_to(self.root)) for path in self.mutable_roots]

    def _limit(self, output: str) -> str:
        if len(output) <= self.max_output:
            return output
        return output[: self.max_output] + "\n... output truncated"


def remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)
