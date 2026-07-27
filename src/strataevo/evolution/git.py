"""Small Git boundary for versioning and rolling back agent generations."""

from __future__ import annotations

import subprocess
from pathlib import Path

from .workspace import PROTECTED_WRITE_ROOTS, remove_path


class GitRepository:
    def __init__(self, root: str | Path, mutable_paths: list[str]) -> None:
        self.root = Path(root).resolve()
        self.mutable_paths = mutable_paths
        if not (self.root / ".git").exists():
            raise ValueError(f"not a Git repository: {self.root}")

    def ensure_clean(self) -> None:
        output = self._run(["status", "--porcelain"], capture=True)
        if output.strip():
            raise RuntimeError("self-evolution requires a clean Git worktree")

    def ensure_branch(self, branch: str) -> None:
        current = self._run(["branch", "--show-current"], capture=True).strip()
        if current == branch:
            return
        exists = (
            subprocess.run(
                ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
                cwd=self.root,
                check=False,
            ).returncode
            == 0
        )
        self._run(["switch", branch] if exists else ["switch", "-c", branch])

    def head(self) -> str:
        return self._run(["rev-parse", "HEAD"], capture=True).strip()

    def changed_paths(self) -> list[str]:
        tracked = self._run(
            ["diff", "--name-only", "--", *self._pathspecs()], capture=True
        ).splitlines()
        staged = self._run(
            ["diff", "--cached", "--name-only", "--", *self._pathspecs()], capture=True
        ).splitlines()
        untracked = self._run(
            ["ls-files", "--others", "--exclude-standard", "--", *self._pathspecs()],
            capture=True,
        ).splitlines()
        return sorted(set(filter(None, [*tracked, *staged, *untracked])))

    def stage(self) -> None:
        self._run(["add", "-A", "--", *self._pathspecs()])

    def staged_diff(self) -> str:
        return self._run(["diff", "--cached", "--binary", "--", *self._pathspecs()], capture=True)

    def commit(self, message: str) -> str:
        self._run(["commit", "-m", message])
        return self.head()

    def apply_patch(self, patch_path: Path) -> None:
        completed = subprocess.run(
            ["git", "apply", "--index", "--binary", str(patch_path)],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"git apply failed: {completed.stderr.strip()}")

    def rollback(self) -> None:
        self._run(["restore", "--staged", "--worktree", "--", *self._pathspecs()])
        untracked = self._run(
            ["ls-files", "--others", "--exclude-standard", "--", *self._pathspecs()],
            capture=True,
        ).splitlines()
        for relative in untracked:
            target = (self.root / relative).resolve()
            if target.is_relative_to(self.root):
                remove_path(target)

    def _pathspecs(self) -> list[str]:
        excluded = [f":(exclude){path}" for path in sorted(PROTECTED_WRITE_ROOTS)]
        return [*self.mutable_paths, *excluded]

    def _run(self, arguments: list[str], *, capture: bool = False) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=self.root,
            capture_output=capture,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() if capture else ""
            raise RuntimeError(f"git {' '.join(arguments)} failed: {detail}")
        return completed.stdout if capture else ""
