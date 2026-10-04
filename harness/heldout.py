"""harness/heldout.py -- the one-shot replay of the test split (P2.7).

    python scripts/replay.py --from-cache all --split test

The settings are read from ``config/perception.yaml`` and ``config/runtime.yaml``; nothing is
tuned here. ``reports/replay_test.json`` is written once: a second ``--split test`` run is refused
while that file exists, unless ``--allow-repeat-test "<reason>"`` is given (the reason is recorded
in the new report).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class RepeatTestRefused(Exception):
    """The test report already exists and no reason for a repeat was given."""


def guard_repeat_test(report_path: Path | str, allow_repeat_reason: str | None) -> str | None:
    """None for a first replay; the stripped reason for an allowed repeat; else refuse."""
    if not Path(report_path).exists():
        return None
    reason = (allow_repeat_reason or "").strip()
    if not reason:
        raise RepeatTestRefused(
            f"{report_path} already exists: the test split is replayed once; "
            'pass --allow-repeat-test "<reason>" to repeat it (the reason is recorded)'
        )
    return reason


def run_test_split(args: argparse.Namespace) -> int:
    report_path = Path(getattr(args, "report", ROOT / "reports" / "replay_test.json"))
    try:
        guard_repeat_test(report_path, getattr(args, "allow_repeat_test", None))
    except RepeatTestRefused as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0
