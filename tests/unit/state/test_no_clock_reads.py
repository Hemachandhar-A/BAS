"""IMPLEMENTATION_PLAN.md 7.3 / AGENTS.md rule 8: "Time arrives on the
data. No clock reads in perception/ (except camera.py), state/ or
engine/." Scoped to state/ -- the only one of those three directories P2.1
owns/has created so far; perception/ and engine/ don't exist yet."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STATE_DIR = ROOT / "state"

_FORBIDDEN_MODULES = {"time", "datetime"}


def test_state_package_never_reads_a_clock() -> None:
    offenders: list[str] = []
    for path in sorted(STATE_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in _FORBIDDEN_MODULES:
                        offenders.append(f"{path.name}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] in _FORBIDDEN_MODULES:
                    offenders.append(f"{path.name}: from {node.module} import ...")
    assert not offenders, f"state/ must never read a clock (AGENTS.md rule 8): {offenders}"
