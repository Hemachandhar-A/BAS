"""harness/tune.py -- the P2.6 sweep: grid, pre-registered selection rule, split refusal, output
files. Pure helpers are tested on synthetic results; no real cache is opened here."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pytest
import yaml

from contracts import PerceptionConfig
from harness.settings import load_perception_config
from harness.tune import (
    DEFAULT_PARAMS,
    GRID,
    SettingResult,
    SplitRefused,
    choose,
    distance_to_defaults,
    iter_settings,
    load_runs,
    rank_key,
    sensitivity,
    tune_main,
    write_perception_yaml,
)


def _res(params: dict, matches: int, extra_missing: int = 0, uncertain: int = 0) -> SettingResult:
    return SettingResult(
        params=params,
        matches=matches,
        extra_missing=extra_missing,
        uncertain=uncertain,
        distance=distance_to_defaults(params),
    )


def _p(**kw) -> dict:
    return {**DEFAULT_PARAMS, **kw}


def test_grid_is_the_preregistered_one_and_contains_the_defaults() -> None:
    assert GRID == {
        "detector_conf_floor": [0.20, 0.30, 0.40, 0.50],
        "confirm_conf": [0.50, 0.60, 0.70],
        "hysteresis_frames": [3, 4, 5, 6, 8],
        "release_frames": [3, 5, 8],
        "baseline_frames": [5, 10, 15],
        "touch_margin_frac": [0.05, 0.10, 0.25],
    }
    assert DEFAULT_PARAMS == PerceptionConfig().model_dump()
    settings = list(iter_settings())
    assert len(settings) == 4 * 3 * 5 * 3 * 3 * 3
    assert DEFAULT_PARAMS in settings


def test_iter_settings_skips_what_the_contract_rejects() -> None:
    grid = {**GRID, "detector_conf_floor": [0.55], "confirm_conf": [0.50, 0.60]}
    got = list(iter_settings(grid))
    assert got and all(s["confirm_conf"] >= s["detector_conf_floor"] for s in got)
    assert {s["confirm_conf"] for s in got} == {0.60}  # 0.50 < 0.55 is rejected


def test_distance_to_defaults_counts_grid_steps() -> None:
    assert distance_to_defaults(DEFAULT_PARAMS) == 0
    assert distance_to_defaults(_p(hysteresis_frames=6)) == 1
    assert distance_to_defaults(_p(hysteresis_frames=8, release_frames=3)) == 2 + 1


def test_selection_rule_order_matches_then_errors_then_uncertain_then_distance() -> None:
    a = _res(_p(hysteresis_frames=4), matches=6, extra_missing=5)
    b = _res(_p(hysteresis_frames=6), matches=7, extra_missing=9)  # more matches wins
    c = _res(_p(hysteresis_frames=3), matches=7, extra_missing=2)  # fewer errors than b
    d = _res(_p(release_frames=8), matches=7, extra_missing=2, uncertain=3)  # more uncertain
    e = _res(_p(hysteresis_frames=8, release_frames=3), matches=7, extra_missing=2)  # farther
    ranked = sorted([a, b, c, d, e], key=rank_key)
    assert ranked[0] is c  # 7 matches, 2 errors, 0 uncertain, distance 2 (c) vs 3 (e)
    assert [r.matches for r in ranked] == [7, 7, 7, 7, 6]
    assert ranked[-1] is a


def test_defaults_are_kept_unless_beaten_by_at_least_one_matching_run() -> None:
    default = _res(DEFAULT_PARAMS, matches=7, extra_missing=4, uncertain=2)
    same_but_better_elsewhere = _res(_p(hysteresis_frames=4), matches=7, extra_missing=0)
    chosen, retained, ranked = choose([default, same_but_better_elsewhere], default)
    assert retained and chosen is default  # equal matches: defaults stay
    assert ranked[0] is same_but_better_elsewhere  # still ranked first on the secondary rule

    better = _res(_p(hysteresis_frames=4), matches=8, extra_missing=4)
    chosen, retained, _ = choose([default, better], default)
    assert not retained and chosen is better


def test_sensitivity_summarises_each_parameter_value() -> None:
    results = [
        _res(_p(hysteresis_frames=3), 5),
        _res(_p(hysteresis_frames=5), 7),
        _res(_p(hysteresis_frames=5, release_frames=8), 9),
    ]
    s = sensitivity(results)
    h = {row["value"]: row for row in s["hysteresis_frames"]}
    assert h[3]["settings"] == 1 and h[3]["best_matches"] == 5
    assert h[5]["settings"] == 2 and h[5]["best_matches"] == 9
    assert h[5]["mean_matches"] == pytest.approx(8.0)


def test_perception_yaml_roundtrips_through_the_loader(tmp_path: Path) -> None:
    p = tmp_path / "perception.yaml"
    write_perception_yaml(p, _p(hysteresis_frames=4, touch_margin_frac=0.25), retained=False)
    cfg = load_perception_config(p)
    assert cfg.hysteresis_frames == 4 and cfg.touch_margin_frac == 0.25
    write_perception_yaml(p, DEFAULT_PARAMS, retained=True)
    text = p.read_text(encoding="utf-8")
    assert "defaults retained: no setting beat them by one run on val" in text
    assert yaml.safe_load(text) == DEFAULT_PARAMS
    assert load_perception_config(p) == PerceptionConfig()


def test_the_test_split_is_refused_everywhere(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "split", "script_type"])
        w.writerow(["x002", "test", "correct"])
    with pytest.raises(SplitRefused):
        load_runs("test", manifest_csv=manifest, runs_dir=tmp_path, cache_dir=tmp_path,
                  expected_stamp="s", expected_fps=10.0)  # fmt: skip
    ns = argparse.Namespace(split="test", tune=True)
    assert tune_main(ns) == 2
    ns = argparse.Namespace(split="all", tune=True)
    assert tune_main(ns) == 2


def test_load_runs_for_val_never_opens_other_splits(tmp_path: Path) -> None:
    """Only the requested split's runs are read: a manifest row of another split points at a
    cache that does not exist, and loading must not fail on it."""
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "split", "script_type"])
        w.writerow(["a1", "val", "idle"])
        w.writerow(["t1", "test", "correct"])
    runs = tmp_path / "runs" / "a1"
    runs.mkdir(parents=True)
    (runs / "script.json").write_text(
        json.dumps(
            {
                "run_id": "a1", "experiment_id": "e", "script_type": "idle", "split": "val",
                "fps": 30.0, "camera_setup_id": "c", "operator": "o",
                "performed_steps": [], "expected_deviations": [],
            }
        ),
        encoding="utf-8",
    )
    cache = tmp_path / "cache" / "a1"
    cache.mkdir(parents=True)
    (cache / "perception.jsonl").write_text(
        json.dumps({"run_id": "a1", "experiment_id": "e", "fps": 10.0, "model_stamp": "s"}) + "\n"
        + json.dumps({"frame_id": 0, "t": 0.0}) + "\n",
        encoding="utf-8",
    )  # fmt: skip
    got = load_runs(  # fmt: skip
        "val", manifest_csv=manifest, runs_dir=tmp_path / "runs",
                    cache_dir=tmp_path / "cache", expected_stamp="s", expected_fps=10.0)
    assert [r.run_id for r in got] == ["a1"] and len(got[0].frames) == 1
