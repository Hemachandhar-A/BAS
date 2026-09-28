#!/usr/bin/env python3
"""python scripts/install_hooks.py -- installs the git hooks that enforce
IMPLEMENTATION_PLAN.md 7.6: pre-commit runs `check.py --quick`, pre-push
runs the full suite. The hook files themselves are the minimal shell shim
git requires to invoke an executable; all real logic lives in check.py.
"""

from __future__ import annotations

import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOOKS_DIR = ROOT / ".git" / "hooks"

PRE_COMMIT = """#!/bin/sh
cd "$(git rev-parse --show-toplevel)" || exit 1
exec uv run python scripts/check.py --quick
"""

PRE_PUSH = """#!/bin/sh
cd "$(git rev-parse --show-toplevel)" || exit 1
exec uv run python scripts/check.py
"""


def _install(name: str, content: str) -> None:
    path = HOOKS_DIR / name
    path.write_text(content, encoding="utf-8", newline="\n")
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    print(f"installed {path}")


def main() -> int:
    if not HOOKS_DIR.exists():
        print(
            "no .git/hooks directory found -- run this from inside the git repository",
            file=sys.stderr,
        )
        return 1
    _install("pre-commit", PRE_COMMIT)
    _install("pre-push", PRE_PUSH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
