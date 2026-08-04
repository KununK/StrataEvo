"""Small, auditable tools for a coding agent's workspace."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .tool import Tool, tool


class Workspace:
    """Creates tools restricted to one filesystem root."""

    def __init__(
        self,
        root: str | Path,
        *,
        command_timeout: float = 30.0,
        max_output: int = 30_000,
    ) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(self.root)
        if command_timeout <= 0 or max_output <= 0:
            raise ValueError("command_timeout and max_output must be positive")
        self.command_timeout = command_timeout
        self.max_output = max_output

    def tools(self) -> list[Tool]:
        @tool
        def list_files(path: str = ".") -> str:
            """List files below a workspace directory."""
            target = self._resolve(path)
            if not target.is_dir():
                raise ValueError(f"not a directory: {path}")
            output = "\n".join(
                str(item.relative_to(self.root)) for item in sorted(target.iterdir())
            )
            return self._limit(output)

        @tool
        def read_file(path: str, start_line: int = 1, end_line: int = 400) -> str:
            """Read a UTF-8 text file with line numbers."""
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
            """Search UTF-8 workspace files for a literal string."""
            target = self._resolve(path)
            paths = [target] if target.is_file() else target.rglob("*")
            matches: list[str] = []
            for candidate in paths:
                if not candidate.is_file() or ".git" in candidate.parts:
                    continue
                try:
                    content = candidate.read_text(encoding="utf-8")
                    for number, line in enumerate(content.splitlines(), 1):
                        if query in line:
                            matches.append(f"{candidate.relative_to(self.root)}:{number}:{line}")
                except (UnicodeDecodeError, OSError):
                    continue
                if len(matches) >= 200:
                    break
            return self._limit("\n".join(matches) or "No matches")

        @tool
        def write_file(path: str, content: str) -> str:
            """Create or replace a UTF-8 file in the workspace."""
            target = self._resolve(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return f"Wrote {len(content.encode())} bytes to {path}"

        @tool
        def replace_text(path: str, old: str, new: str) -> str:
            """Replace one exact, unique text occurrence in a file."""
            target = self._resolve(path)
            content = target.read_text(encoding="utf-8")
            count = content.count(old)
            if count != 1:
                raise ValueError(f"expected one occurrence, found {count}")
            target.write_text(content.replace(old, new), encoding="utf-8")
            return f"Updated {path}"

        @tool
        def run_shell(command: str) -> str:
            """Run a shell command in the workspace and return exit code and output."""
            completed = subprocess.run(
                command,
                cwd=self.root,
                shell=True,
                executable="/bin/bash",
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=self.command_timeout,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            output = f"exit_code={completed.returncode}\n{completed.stdout}"
            return self._limit(output)

        return [list_files, read_file, search_files, write_file, replace_text, run_shell]

    def _resolve(self, path: str) -> Path:
        if not path or "\x00" in path:
            raise ValueError("invalid path")
        target = (self.root / Path(path)).resolve()
        if not target.is_relative_to(self.root):
            raise PermissionError(f"path escapes workspace: {path}")
        return target

    def _limit(self, output: str) -> str:
        if len(output) <= self.max_output:
            return output
        return output[: self.max_output] + "\n... output truncated"
