#!/usr/bin/env python3
"""python scripts/check.py [--quick | --status | --video]

Owner: P2 (IMPLEMENTATION_PLAN.md Part 4, 5.9, 7.6). The single entry point
git hooks and CI call -- never a bare shell script (AGENTS.md rule 17).

--quick   ruff + the fast unit suite under tests/unit (excludes slow and
          cached-replay goldens). This is what pre-commit runs.
--status  Feature x pass/fail table derived from the F1..F14 pytest
          markers. Never hand-edited (IMPLEMENTATION_PLAN.md R8). Paste the
          output into a PR per the Part 8 checklist.
--video   cached-replay goldens only (gold_video marker); run at a gate.
(default) the full unit + integration suite, excluding only `slow`. This is
          what pre-push runs.

The offline guard (tests/conftest.py) applies automatically to every pytest
run from this script -- it is not a separate flag.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FEATURES = [f"F{n}" for n in range(1, 15)]

_PASSED_RE = re.compile(r"(\d+) passed")
_FAILED_RE = re.compile(r"(\d+) failed")


def _run(cmd: list[str]) -> int:
    print("+", " ".join(cmd))
    return subprocess.run(cmd, cwd=ROOT).returncode


def cmd_quick() -> int:
    rc = _run([sys.executable, "-m", "ruff", "check", "."])
    rc |= _run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/unit",
            "-m",
            "not slow and not gold_video",
        ]
    )
    return rc


def cmd_full() -> int:
    return _run([sys.executable, "-m", "pytest", "-q", "-m", "not slow and not gold_video"])


def cmd_video() -> int:
    return _run([sys.executable, "-m", "pytest", "-q", "-m", "gold_video"])


def cmd_status() -> int:
    overall_rc = 0
    print(f"{'Feature':<8}{'Passed':>8}{'Failed':>8}  Status")
    for feature in FEATURES:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--no-header", "-m", feature],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        passed = int(m.group(1)) if (m := _PASSED_RE.search(proc.stdout)) else 0
        failed = int(m.group(1)) if (m := _FAILED_RE.search(proc.stdout)) else 0
        total = passed + failed
        if total == 0:
            verdict = "no tests"
        elif failed == 0:
            verdict = "GREEN"
        else:
            verdict = "RED"
            overall_rc = 1
        print(f"{feature:<8}{passed:>8}{failed:>8}  {verdict}")
    return overall_rc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true", help="pre-commit: ruff + fast unit tests")
    group.add_argument("--status", action="store_true", help="feature table derived from markers")
    group.add_argument("--video", action="store_true", help="cached-replay goldens only")
    args = parser.parse_args()

    if args.quick:
        return cmd_quick()
    if args.status:
        return cmd_status()
    if args.video:
        return cmd_video()
    return cmd_full()


if __name__ == "__main__":
    raise SystemExit(main())
