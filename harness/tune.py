"""harness/tune.py -- P2.6 threshold tuning on the VAL split only (IMPLEMENTATION_PLAN.md Part 10).

    python scripts/replay.py --tune [--split val]     sweep, write reports/tuning.json and
                                                      config/perception.yaml
    python -m harness.tune --table val|train [--config PATH]    per-run verdict table
    python -m harness.tune --lint                     dynamic experiment lint, train + val

The ``test`` split is never read: ``load_runs`` raises ``SplitRefused`` for anything but ``train``
and ``val`` (and the sweep itself accepts only ``val``), and run ids of other splits are never
used to open a file. ``train`` exists for the information-only table and the lint, which are
labelled as such: the detector was trained on those frames, so train numbers are optimistic and
are never used for selection.

PRE-REGISTERED SELECTION RULE (fixed before any result was seen; do not edit afterwards):
  1. primary: the number of val runs that MATCH (see harness/metrics.py), more is better;
  2. secondary: fewer extra plus missing step events summed over the runs;
  3. tertiary: fewer events flagged uncertain;
  4. final tie-break: the setting closest to the defaults (sum over parameters of the number of
     grid steps between its value and the default's; then the parameter tuple, for determinism).
A setting replaces the defaults ONLY if it has at least ONE more matching val run than the
defaults; otherwise the defaults stay.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from contracts import (
    ExpectedDeviation,
    ExperimentDefinition,
    PerceptionConfig,
    PerceptionFrame,
    ReportHeader,
    RunScript,
    RuntimeConfig,
)
from harness.metrics import diagnose, lint_run, observe, verdict
from harness.settings import (
    expected_model_stamp,
    load_perception_config,
    load_runtime_config,
)

ROOT = Path(__file__).resolve().parents[1]
READABLE_SPLITS = ("train", "val")
GRID: dict[str, list[float | int]] = {
    "detector_conf_floor": [0.20, 0.30, 0.40, 0.50],
    "confirm_conf": [0.50, 0.60, 0.70],
    "hysteresis_frames": [3, 4, 5, 6, 8],
    "release_frames": [3, 5, 8],
    "baseline_frames": [5, 10, 15],
    "touch_margin_frac": [0.05, 0.10, 0.25],
}
DEFAULT_PARAMS: dict[str, float | int] = PerceptionConfig().model_dump()
TOP_N = 20


class SplitRefused(Exception):
    """A split that must not be read here (test, or anything unknown)."""


@dataclass(frozen=True)
class RunData:
    run_id: str
    script_type: str
    performed: list[str]
    expected: list[ExpectedDeviation]
    frames: list[PerceptionFrame]
    cache_sha256: str


@dataclass(frozen=True)
class SettingResult:
    params: dict[str, float | int]
    matches: int
    extra_missing: int
    uncertain: int
    distance: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "params": self.params,
            "matches": self.matches,
            "extra_plus_missing_events": self.extra_missing,
            "uncertain": self.uncertain,
            "distance_to_defaults": self.distance,
        }


# --- loading ---------------------------------------------------------------------------------


def load_runs(
    split: str,
    *,
    manifest_csv: Path | str = "runs/manifest.csv",
    runs_dir: Path | str = "runs",
    cache_dir: Path | str = "data/cache",
    expected_stamp: str,
    expected_fps: float,
    allowed: tuple[str, ...] = READABLE_SPLITS,
) -> list[RunData]:
    """Scripts and caches of the runs of ONE split (train or val), caches validated against the
    expected stamp and fps. A run of any other split is never touched. ``allowed`` is widened
    only by ``harness/heldout.py`` for the one-shot test replay."""
    from perception.cache import cache_path, read_cache

    if split not in allowed:
        raise SplitRefused(f"split {split!r} is refused here; readable: {allowed}")
    with Path(manifest_csv).open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    out: list[RunData] = []
    for row in rows:
        run_id = row["run_id"]
        script = RunScript.model_validate_json(
            (Path(runs_dir) / run_id / "script.json").read_text(encoding="utf-8")
        )
        if script.split != split:
            raise SplitRefused(f"{run_id}: script says split {script.split!r}, not {split!r}")
        path = cache_path(cache_dir, run_id)
        _, frames = read_cache(path, expected_stamp=expected_stamp, expected_fps=expected_fps)
        out.append(
            RunData(
                run_id=run_id,
                script_type=script.script_type,
                performed=list(script.performed_steps),
                expected=list(script.expected_deviations),
                frames=frames,
                cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    return out


# --- evaluation ------------------------------------------------------------------------------


def evaluate(
    experiment: ExperimentDefinition,
    runs: list[RunData],
    config: PerceptionConfig,
    runtime_config: RuntimeConfig,
) -> tuple[int, int, int]:
    """(matching runs, extra plus missing events, uncertain flags) over ``runs``."""
    matches = extra_missing = uncertain = 0
    for run in runs:
        obs = observe(experiment, run.frames, config, runtime_config)
        v = verdict(run.expected, run.performed, obs)
        matches += v.match
        extra_missing += v.extra_events + v.missing_events
        uncertain += v.uncertain
    return matches, extra_missing, uncertain


def iter_settings(grid: dict[str, list[float | int]] = GRID) -> Iterator[dict[str, float | int]]:
    keys = list(grid)
    for combo in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, combo, strict=True))
        try:
            PerceptionConfig(**params)
        except ValidationError:
            continue  # the contract rejects it (for example confirm_conf below the floor)
        yield params


def distance_to_defaults(params: dict[str, float | int]) -> int:
    total = 0
    for key, values in GRID.items():
        total += abs(values.index(params[key]) - values.index(DEFAULT_PARAMS[key]))
    return total


def rank_key(r: SettingResult) -> tuple:
    return (-r.matches, r.extra_missing, r.uncertain, r.distance, tuple(r.params.values()))


def choose(
    results: list[SettingResult], default: SettingResult
) -> tuple[SettingResult, bool, list[SettingResult]]:
    """(chosen, defaults_retained, ranked) by the pre-registered rule."""
    ranked = sorted(results, key=rank_key)
    best = ranked[0]
    if best.matches >= default.matches + 1:
        return best, False, ranked
    return default, True, ranked


def sensitivity(results: list[SettingResult]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for key in GRID:
        rows = []
        for value in GRID[key]:
            subset = [r for r in results if r.params[key] == value]
            if not subset:
                continue
            ms = [r.matches for r in subset]
            rows.append(
                {
                    "value": value,
                    "settings": len(subset),
                    "mean_matches": sum(ms) / len(ms),
                    "min_matches": min(ms),
                    "best_matches": max(ms),
                }
            )
        out[key] = rows
    return out


def sweep(
    experiment: ExperimentDefinition, runs: list[RunData], runtime_config: RuntimeConfig
) -> list[SettingResult]:
    results = []
    for params in iter_settings():
        m, em, u = evaluate(experiment, runs, PerceptionConfig(**params), runtime_config)
        results.append(SettingResult(params, m, em, u, distance_to_defaults(params)))
    return results


# --- tables ----------------------------------------------------------------------------------


def _fmt_dev(devs: Any) -> str:
    items = [
        f"{d.deviation_type}:{'+'.join(d.step_ids)}"
        if isinstance(d, ExpectedDeviation)
        else f"{d[0]}:{'+'.join(d[1])}"
        for d in devs
    ]
    return ", ".join(items) if items else "-"


def run_table(
    experiment: ExperimentDefinition,
    runs: list[RunData],
    config: PerceptionConfig,
    runtime_config: RuntimeConfig,
) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        obs = observe(experiment, run.frames, config, runtime_config)
        v = verdict(run.expected, run.performed, obs)
        cause = "" if v.match else "; ".join(diagnose(experiment, run.frames, config,
                                                      run.performed, obs))  # fmt: skip
        if not v.match and not cause:
            cause = "deviations differ from the expected ones"
        rows.append(
            {
                "run_id": run.run_id,
                "script_type": run.script_type,
                "performed_steps": run.performed,
                "expected": _fmt_dev(run.expected),
                "observed": _fmt_dev(obs.deviations),
                "match": v.match,
                "extra_events": v.extra_events,
                "missing_events": v.missing_events,
                "uncertain": v.uncertain,
                "cause": cause,
            }
        )
    return rows


def format_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| run | type | performed | expected deviations | observed deviations | result "
        "| extra/missing/unc | cause |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['run_id']} | {r['script_type']} | {' '.join(r['performed_steps']) or '-'} "
            f"| {r['expected']} | {r['observed']} | {'MATCH' if r['match'] else 'MISMATCH'} "
            f"| {r['extra_events']}/{r['missing_events']}/{r['uncertain']} | {r['cause']} |"
        )
    return "\n".join(lines)


# --- outputs ---------------------------------------------------------------------------------


def write_perception_yaml(path: Path | str, params: dict[str, float | int], retained: bool) -> None:
    head = "# PerceptionConfig values from `python scripts/replay.py --tune` (P2.6, val only).\n"
    if retained:
        head += "# defaults retained: no setting beat them by one run on val\n"
    else:
        head += "# a setting beat the defaults by one or more matching val runs\n"
    body = "".join(f"{k}: {params[k]}\n" for k in DEFAULT_PARAMS)
    Path(path).write_text(head + body, encoding="utf-8", newline="\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tune_main(args: argparse.Namespace) -> int:
    if args.split != "val":
        print(
            f"error: --tune works on the val split only; {args.split!r} is refused "
            "(never tune on train or test)",
            file=sys.stderr,
        )
        return 2
    experiment_path = Path(getattr(args, "experiment", "config/experiment.json"))
    runtime_path = Path(getattr(args, "runtime_config", ROOT / "config" / "runtime.yaml"))
    manifest = Path(getattr(args, "manifest", ROOT / "weights" / "MANIFEST.json"))
    cache_dir = Path(getattr(args, "cache_dir", "data/cache"))
    out_report = Path(getattr(args, "tuning_report", ROOT / "reports" / "tuning.json"))
    out_config = Path(getattr(args, "out_config", ROOT / "config" / "perception.yaml"))

    experiment = ExperimentDefinition.from_json(experiment_path)
    runtime_config = load_runtime_config(runtime_path)
    stamp = expected_model_stamp(manifest)
    runs = load_runs(
        "val", cache_dir=cache_dir, expected_stamp=stamp, expected_fps=runtime_config.target_fps
    )
    default_cfg = PerceptionConfig()
    dm, dem, du = evaluate(experiment, runs, default_cfg, runtime_config)
    default = SettingResult(dict(DEFAULT_PARAMS), dm, dem, du, 0)
    results = sweep(experiment, runs, runtime_config)
    chosen, retained, ranked = choose(results, default)

    header = ReportHeader(
        generated_at=datetime.now(UTC),
        model_stamp=stamp,
        input_stamps={
            "experiment_sha256": _sha(experiment_path),
            "runtime_yaml_sha256": _sha(runtime_path),
            "weights_manifest_sha256": _sha(manifest),
            **{f"val_cache_{r.run_id}_sha256": r.cache_sha256 for r in runs},
        },
    )
    report = {
        **json.loads(header.model_dump_json()),
        "split": "val",
        "fps": runtime_config.target_fps,
        "val_runs": [r.run_id for r in runs],
        "grid": GRID,
        "settings_evaluated": len(results),
        "selection_rule": __doc__.split("PRE-REGISTERED SELECTION RULE")[1].strip(),
        "default_result": default.as_dict(),
        "top_20": [r.as_dict() for r in ranked[:TOP_N]],
        "chosen": {**chosen.as_dict(), "defaults_retained": retained},
        "sensitivity": sensitivity(results),
    }
    out_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    write_perception_yaml(out_config, chosen.params, retained)
    print(
        f"val runs: {len(runs)}; settings evaluated: {len(results)}; defaults match "
        f"{default.matches}/{len(runs)}; best setting matches {ranked[0].matches}; "
        f"{'defaults retained' if retained else 'new setting chosen'}: {chosen.params}"
    )
    print(f"wrote {out_report} and {out_config}")
    return 0


# --- information / lint CLI -------------------------------------------------------------------


def _load(
    split: str, args: argparse.Namespace
) -> tuple[ExperimentDefinition, RuntimeConfig, list[RunData]]:
    experiment = ExperimentDefinition.from_json(args.experiment)
    rc = load_runtime_config(ROOT / "config" / "runtime.yaml")
    stamp = expected_model_stamp(ROOT / "weights" / "MANIFEST.json")
    return experiment, rc, load_runs(
        split, cache_dir=args.cache_dir, expected_stamp=stamp, expected_fps=rc.target_fps
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--table", choices=READABLE_SPLITS)
    ap.add_argument("--lint", action="store_true", help="dynamic lint on correct runs, train+val")
    ap.add_argument("--config", type=Path, default=None, help="PerceptionConfig yaml")
    ap.add_argument("--experiment", type=Path, default=Path("config/experiment.json"))
    ap.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    args = ap.parse_args(argv)
    config = load_perception_config(args.config) if args.config else PerceptionConfig()
    if args.table:
        experiment, rc, runs = _load(args.table, args)
        label = "val"
        if args.table == "train":
            label = "INFORMATION ONLY (train: optimistic, not for selection)"
        rows = run_table(experiment, runs, config, rc)
        print(f"## {args.table} runs ({label}); config {config.model_dump()}")
        print(format_table(rows))
        print(f"MATCH {sum(r['match'] for r in rows)}/{len(rows)}")
        return 0
    if args.lint:
        bad = 0
        for split in READABLE_SPLITS:
            experiment, rc, runs = _load(split, args)
            for run in (r for r in runs if r.script_type == "correct"):
                for v in lint_run(experiment, run.frames, config, rc):
                    bad += 1
                    print(f"VIOLATION {split} {run.run_id} {v['step_id']}: {v['reason']}")
            n = sum(r.script_type == "correct" for r in runs)
            print(f"lint {split}: {n} correct runs checked")
        print(f"violations: {bad}")
        return 1 if bad else 0
    ap.error("one of --table or --lint is required")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
