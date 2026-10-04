"""harness/heldout.py -- the one-shot replay of the test split (P2.7).

    python scripts/replay.py --from-cache all --split test

The settings are read from ``config/perception.yaml`` and ``config/runtime.yaml``; nothing is
tuned here. ``reports/replay_test.json`` is written once: a second ``--split test`` run is refused
while that file exists, unless ``--allow-repeat-test "<reason>"`` is given (the reason is recorded
in the new report). The verdict uses the strict MATCH of ``harness/metrics.py`` only; the
START-excluded count is reported for information and never changes a verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from contracts import (
    AcceptanceConfig,
    ExpectedDeviation,
    ExperimentDefinition,
    PerceptionConfig,
    ReportHeader,
    RuntimeConfig,
    StateEvent,
)
from engine.sequence import SequenceEngine
from harness.metrics import Observation, RunVerdict, diagnose, observe, verdict
from harness.settings import expected_model_stamp, load_perception_config, load_runtime_config
from harness.tune import RunData, load_runs

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


def cause_category(cause: str) -> str:
    """The fixed vocabulary for one diagnose line (read from the data by metrics.diagnose)."""
    if "merged into one long hold" in cause:
        return "merged presses"
    if "extra press" in cause:
        return "extra press"
    if "missed press" in cause or (
        cause.startswith(f"{START_STEP}:") and "shorter than hysteresis" in cause
    ):
        return "missed START press"
    if "missing box" in cause:
        return "missing container box"
    if "baseline latch" in cause:
        return "baseline latch"
    return "other"


def replay_pos(
    experiment: ExperimentDefinition,
    runtime_config: RuntimeConfig,
    step_ids: list[str],
    t0: float,
) -> float:
    """POS (the engine's own ``RunSummary.pos``) of a sequence of step events."""
    engine = SequenceEngine(experiment, runtime_config)
    engine.start(t0)
    events = []
    for i, sid in enumerate(step_ids, start=1):
        events += engine.on_state_event(
            StateEvent(t=t0 + i, step_id=sid, confidence=1.0, uncertain=False)
        )
    events += engine.finish(t0 + len(step_ids) + 1)
    return next(e.summary.pos for e in events if e.kind == "run_completed" and e.summary)


def _devs(items: Any) -> list[dict[str, Any]]:
    return [
        {"deviation_type": d.deviation_type, "step_ids": list(d.step_ids)}
        if isinstance(d, ExpectedDeviation)
        else {"deviation_type": d[0], "step_ids": list(d[1])}
        for d in items
    ]


def run_row(
    experiment: ExperimentDefinition,
    run: RunData,
    config: PerceptionConfig,
    runtime_config: RuntimeConfig,
) -> dict[str, Any]:
    obs = observe(experiment, run.frames, config, runtime_config)
    v = verdict(run.expected, run.performed, obs)
    s = start_excluded_verdict(run.expected, run.performed, obs)
    causes = [] if v.match else diagnose(experiment, run.frames, config, run.performed, obs)
    if not v.match and not causes:
        causes = ["deviations differ from the expected ones"]
    t0 = run.frames[0].t if run.frames else 0.0
    return {
        "run_id": run.run_id,
        "script_type": run.script_type,
        "frames": len(run.frames),
        "performed_steps": run.performed,
        "observed_fired_steps": obs.fired,
        "expected_deviations": _devs(run.expected),
        "observed_deviations": _devs(obs.deviations),
        "match": v.match,
        "start_excluded_match": s.match,
        "extra_events": v.extra_events,
        "missing_events": v.missing_events,
        "extra_deviations": v.extra_deviations,
        "missing_deviations": v.missing_deviations,
        "uncertain": v.uncertain,
        "pos": replay_pos(experiment, runtime_config, obs.fired, t0),
        "pos_of_performed_steps": replay_pos(experiment, runtime_config, run.performed, t0),
        "cause": "; ".join(causes),
        "cause_categories": [cause_category(c) for c in causes],
        "cache_sha256": run.cache_sha256,
    }


def aggregate(rows: list[dict[str, Any]], max_mismatched_runs: int) -> dict[str, Any]:
    """The verdict depends on the strict ``match`` only; ``start_excluded_match`` is counted for
    information."""
    strict = sum(1 for r in rows if r["match"])
    mismatches = len(rows) - strict
    return {
        "runs": len(rows),
        "strict_matches": strict,
        "strict_mismatches": mismatches,
        "start_excluded_matches": sum(1 for r in rows if r["start_excluded_match"]),
        "max_mismatched_runs": max_mismatched_runs,
        "verdict": "PASS" if mismatches <= max_mismatched_runs else "FAIL",
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_test_split(args: argparse.Namespace) -> int:
    report_path = Path(getattr(args, "report", ROOT / "reports" / "replay_test.json"))
    try:
        repeat_reason = guard_repeat_test(report_path, getattr(args, "allow_repeat_test", None))
    except RepeatTestRefused as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    from perception.cache import CacheMismatch

    experiment_path = Path(getattr(args, "experiment", "config/experiment.json"))
    runtime_path = Path(getattr(args, "runtime_config", ROOT / "config" / "runtime.yaml"))
    perception_path = Path(getattr(args, "perception_config", ROOT / "config" / "perception.yaml"))
    manifest = Path(getattr(args, "manifest", ROOT / "weights" / "MANIFEST.json"))
    acceptance_path = Path(getattr(args, "acceptance", ROOT / "config" / "acceptance.yaml"))
    cache_dir = Path(getattr(args, "cache_dir", "data/cache"))

    experiment = ExperimentDefinition.from_json(experiment_path)
    runtime_config = load_runtime_config(runtime_path)
    perception_config = load_perception_config(perception_path)
    acceptance = AcceptanceConfig.from_yaml(acceptance_path)
    stamp = expected_model_stamp(manifest)
    try:
        runs = load_runs(
            "test",
            cache_dir=cache_dir,
            expected_stamp=stamp,
            expected_fps=runtime_config.target_fps,
            allowed=("test",),
        )
    except CacheMismatch as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    rows = [run_row(experiment, r, perception_config, runtime_config) for r in runs]
    agg = aggregate(rows, acceptance.replay.max_mismatched_runs)

    header = ReportHeader(
        generated_at=datetime.now(UTC),
        model_stamp=stamp,
        input_stamps={
            "experiment_sha256": _sha(experiment_path),
            "runtime_yaml_sha256": _sha(runtime_path),
            "perception_yaml_sha256": _sha(perception_path),
            "weights_manifest_sha256": _sha(manifest),
            "acceptance_yaml_sha256": _sha(acceptance_path),
            **{f"test_cache_{r.run_id}_sha256": r.cache_sha256 for r in runs},
        },
    )
    report = {
        **json.loads(header.model_dump_json()),
        "split": "test",
        "fps": runtime_config.target_fps,
        "config_used": {
            "perception": perception_config.model_dump(),
            "runtime": runtime_config.model_dump(),
        },
        "acceptance": {
            "replay": acceptance.replay.model_dump(),
            "sha256": _sha(acceptance_path),
        },
        "allow_repeat_test_reason": repeat_reason,
        "runs": rows,
        "aggregate": agg,
        "note": "verdict uses the strict match only; start_excluded_match is information only",
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(agg))
    print(f"wrote {report_path}")
    return 0
