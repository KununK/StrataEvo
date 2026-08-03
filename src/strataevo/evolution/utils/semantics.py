"""Conservative detection of candidates that do not change source semantics."""

from __future__ import annotations

import ast
import json
import tomllib
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ChangeSemantics:
    classification: str
    paths: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def classify_changes(
    repo: Path,
    changed_paths: list[str],
    read_parent: Callable[[str], str | None],
) -> ChangeSemantics:
    """Return semantic_noop only when every changed file has a provably equal structure."""
    paths = {
        path: _classify_file(path, read_parent(path), repo / path) for path in changed_paths
    }
    classification = (
        "semantic_noop"
        if paths and all(value == "equivalent" for value in paths.values())
        else "behavior_change"
    )
    return ChangeSemantics(classification, paths)


def _classify_file(path: str, parent: str | None, candidate_path: Path) -> str:
    if parent is None or not candidate_path.is_file():
        return "changed"
    try:
        candidate = candidate_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return "changed"
    suffix = candidate_path.suffix.lower()
    try:
        if suffix == ".py":
            return "equivalent" if _python_tree(parent) == _python_tree(candidate) else "changed"
        if suffix == ".json":
            return "equivalent" if json.loads(parent) == json.loads(candidate) else "changed"
        if suffix == ".toml":
            return "equivalent" if tomllib.loads(parent) == tomllib.loads(candidate) else "changed"
    except (SyntaxError, json.JSONDecodeError, tomllib.TOMLDecodeError):
        return "changed"
    return "equivalent" if parent == candidate else "changed"


def _python_tree(source: str) -> str:
    tree = ast.parse(source)
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        del tree.body[0]
    return ast.dump(tree, include_attributes=False)
