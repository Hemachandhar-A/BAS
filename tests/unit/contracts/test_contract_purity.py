"""IMPLEMENTATION_PLAN.md 7.3: 'import contracts succeeds in an environment
without torch, mediapipe or rfdetr' -- checked statically (by inspecting
contracts.py's own import statements) so this test does not depend on
whether those packages happen to be installed on the machine running it."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import contracts  # the actual runtime import must also succeed

ROOT = Path(__file__).resolve().parents[3]
ALLOWED_THIRD_PARTY = {"pydantic", "numpy", "yaml"}


def test_contracts_module_actually_imports() -> None:
    assert contracts.__file__ is not None


def test_contracts_only_imports_pydantic_numpy_pyyaml_and_stdlib() -> None:
    source = (ROOT / "contracts.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="contracts.py")
    stdlib = set(sys.stdlib_module_names)

    top_level_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            top_level_names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import inside the same module: fine
                continue
            if node.module:
                top_level_names.add(node.module.split(".")[0])

    disallowed = {
        name for name in top_level_names if name not in stdlib and name not in ALLOWED_THIRD_PARTY
    }
    assert not disallowed, (
        f"contracts.py imports {sorted(disallowed)!r}, outside pydantic/numpy/pyyaml/stdlib "
        "(IMPLEMENTATION_PLAN.md 7.3 contract-purity test)"
    )


def test_contracts_defines_no_contracts_package() -> None:
    # IMPLEMENTATION_PLAN.md Part 4: "Do not create a contracts/ package: it
    # would shadow the module silently."
    assert not (ROOT / "contracts").is_dir()
