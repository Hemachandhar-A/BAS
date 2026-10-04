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
from dataclasses import replace
from pathlib import Path

from contracts import ExpectedDeviation
from harness.metrics import Observation, RunVerdict, verdict

START_STEP = "start_pressed"

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


def start_excluded_verdict(
    expected: list[ExpectedDeviation],
    performed: list[str],
    obs: Observation,
    step_id: str = START_STEP,
) -> RunVerdict:
    """The same MATCH rule (``metrics.verdict``) with ``step_id`` removed from every step_ids
    list of the expected and observed deviations (a deviation left empty is dropped), from
    ``performed_steps`` and from the observed firings. Reporting only: it never replaces the
    strict verdict. Inputs are not modified."""
    want = []
    for d in expected:
        ids = [s for s in d.step_ids if s != step_id]
        if ids:
            want.append(ExpectedDeviation(deviation_type=d.deviation_type, step_ids=ids))
    keep = [i for i, s in enumerate(obs.fired) if s != step_id]
    devs = []
    for kind, ids in obs.deviations:
        left = tuple(s for s in ids if s != step_id)
        if left:
            devs.append((kind, left))
    stripped = replace(
        obs,
        fired=[obs.fired[i] for i in keep],
        fired_frame_index=[obs.fired_frame_index[i] for i in keep],
        fired_t=[obs.fired_t[i] for i in keep],
        deviations=devs,
    )
    return verdict(want, [s for s in performed if s != step_id], stripped)
