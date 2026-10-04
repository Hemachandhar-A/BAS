"""harness/heldout.py -- the one-shot test replay: repeat guard, START-excluded metric (reporting
only), report assembly. Synthetic data and temporary files only; no real cache is opened."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from harness.heldout import RepeatTestRefused, guard_repeat_test, run_test_split


def test_guard_allows_the_first_test_replay(tmp_path: Path) -> None:
    assert guard_repeat_test(tmp_path / "replay_test.json", None, tmp_path / ".done") is None


def test_guard_refuses_on_the_marker_alone_whatever_the_report_path(tmp_path: Path) -> None:
    marker = tmp_path / ".replay_test_done"
    marker.write_text("done\n", encoding="utf-8")
    other_report = tmp_path / "another_report.json"  # a different --report; it does not exist
    with pytest.raises(RepeatTestRefused, match="replayed once"):
        guard_repeat_test(other_report, None, marker)
    assert guard_repeat_test(other_report, " operator error ", marker) == "operator error"


def test_deleting_the_report_does_not_reopen_the_split(tmp_path: Path) -> None:
    report, marker = tmp_path / "replay_test.json", tmp_path / ".replay_test_done"
    report.write_text("{}", encoding="utf-8")
    marker.write_text("done\n", encoding="utf-8")
    report.unlink()
    with pytest.raises(RepeatTestRefused):
        guard_repeat_test(report, None, marker)


def test_runner_refuses_on_the_marker_with_a_fresh_report_path(tmp_path: Path) -> None:
    marker = tmp_path / ".replay_test_done"
    marker.write_text("done\n", encoding="utf-8")
    fresh = tmp_path / "fresh.json"
    ns = argparse.Namespace(
        split="test",
        report=fresh,
        marker=marker,
        allow_repeat_test=None,
        cache_dir=tmp_path / "no_such_dir",
        experiment=tmp_path / "no_such_experiment.json",
    )
    assert run_test_split(ns) == 2
    assert not fresh.exists()


def test_guard_refuses_when_the_report_already_exists(tmp_path: Path) -> None:
    report = tmp_path / "replay_test.json"
    report.write_text("{}", encoding="utf-8")
    with pytest.raises(RepeatTestRefused, match="replayed once"):
        guard_repeat_test(report, None, tmp_path / ".done")
    with pytest.raises(RepeatTestRefused):
        guard_repeat_test(report, "   ", tmp_path / ".done")


def test_guard_with_a_reason_returns_it_for_the_new_report(tmp_path: Path) -> None:
    report = tmp_path / "replay_test.json"
    report.write_text("{}", encoding="utf-8")
    reason = guard_repeat_test(report, "  disk full during write  ", tmp_path / ".done")
    assert reason == "disk full during write"


def test_runner_refuses_a_repeat_before_reading_anything(tmp_path: Path) -> None:
    report = tmp_path / "replay_test.json"
    report.write_text('{"keep": true}', encoding="utf-8")
    ns = argparse.Namespace(
        split="test",
        report=report,
        marker=tmp_path / ".done",
        allow_repeat_test=None,
        cache_dir=tmp_path / "no_such_dir",
        experiment=tmp_path / "no_such_experiment.json",
    )
    assert run_test_split(ns) == 2
    assert json.loads(report.read_text(encoding="utf-8")) == {"keep": True}


def test_cli_routes_split_test_to_the_guard(tmp_path: Path) -> None:
    from harness.replay import main

    report = tmp_path / "replay_test.json"
    report.write_text("{}", encoding="utf-8")
    base = ["--split", "test", "--report", str(report), "--marker", str(tmp_path / ".done")]
    assert main(["--from-cache", "all", *base]) == 2
    # a single run id with --split test is refused too
    assert main(["--from-cache", "x002", *base]) == 2
    # the tuner still refuses the test split
    assert main(["--tune", "--split", "test"]) == 2


# --- START-excluded metric (reporting only) -------------------------------------------------

from contracts import ExpectedDeviation  # noqa: E402
from harness.heldout import start_excluded_verdict  # noqa: E402
from harness.metrics import Observation, verdict  # noqa: E402


def _exp(kind: str, *ids: str) -> ExpectedDeviation:
    return ExpectedDeviation(deviation_type=kind, step_ids=list(ids))  # type: ignore[arg-type]


def test_start_excluded_forgives_a_missed_start_press() -> None:
    performed = ["start_pressed", "red_out"]
    obs = Observation(fired=["red_out"], fired_frame_index=[5], fired_t=[0.5], deviations=[])
    assert not verdict([], performed, obs).match
    assert start_excluded_verdict([], performed, obs).match


def test_start_excluded_forgives_an_extra_start_deviation_and_firing() -> None:
    obs = Observation(
        fired=["start_pressed", "start_pressed"],
        fired_frame_index=[1, 9],
        fired_t=[0.1, 0.9],
        deviations=[("repeat", ("start_pressed",))],
    )
    performed = ["start_pressed"]
    assert not verdict([], performed, obs).match
    assert start_excluded_verdict([], performed, obs).match


def test_start_excluded_removes_start_from_step_ids_and_drops_empty_deviations() -> None:
    expected = [_exp("omission", "start_pressed", "red_stowed"), _exp("repeat", "start_pressed")]
    obs = Observation(deviations=[("omission", ("red_stowed",))])
    assert start_excluded_verdict(expected, [], obs).match


def test_start_excluded_still_fails_on_other_steps() -> None:
    obs = Observation(
        fired=["red_out"], fired_frame_index=[1], fired_t=[0.1], deviations=[]
    )
    v = start_excluded_verdict([_exp("omission", "red_in_tray")], ["red_out"], obs)
    assert not v.match


def test_start_excluded_never_mutates_its_inputs_and_other_steps_pass_through() -> None:
    expected = [_exp("omission", "start_pressed", "red_stowed")]
    obs = Observation(
        fired=["start_pressed", "red_out"],
        fired_frame_index=[1, 2],
        fired_t=[0.1, 0.2],
        deviations=[("omission", ("start_pressed", "red_stowed"))],
        uncertain=3,
    )
    performed = ["red_out"]
    v = start_excluded_verdict(expected, performed, obs)
    assert v.match and v.uncertain == 3
    assert expected[0].step_ids == ["start_pressed", "red_stowed"]
    assert obs.fired == ["start_pressed", "red_out"] and performed == ["red_out"]


# --- report assembly ------------------------------------------------------------------------

from contracts import ExperimentDefinition, PerceptionConfig, RuntimeConfig  # noqa: E402
from harness.heldout import aggregate, cause_category, replay_pos, run_row  # noqa: E402
from harness.tune import RunData  # noqa: E402


@pytest.mark.parametrize(
    ("line", "category"),
    [
        (
            "start_pressed: true for at most 3 consecutive frame(s), shorter than "
            "hysteresis_frames=5 (action too short)",
            "missed START press",
        ),
        (
            "start_pressed: missed press: no fingertip inside the grown start_button box",
            "missed START press",
        ),
        ("start_pressed: extra press: fired 2x, performed 1x, at t=1.0", "extra press"),
        (
            "start_pressed: fired 1x but performed more often: (the repeat merged into one "
            "long hold)",
            "merged presses",
        ),
        ("red_out: rule never true; no box for tray (missing box)", "missing container box"),
        ("red_stowed: baseline latch: true in the baseline window", "baseline latch"),
        ("fired order [] differs from performed order []", "other"),
    ],
)
def test_cause_category_maps_diagnose_lines_to_the_fixed_vocabulary(
    line: str, category: str
) -> None:
    assert cause_category(line) == category


def test_aggregate_counts_and_verdict() -> None:
    rows = [
        {"match": True, "start_excluded_match": True},
        {"match": False, "start_excluded_match": True},
        {"match": False, "start_excluded_match": False},
    ]
    agg = aggregate(rows, max_mismatched_runs=0)
    assert agg["runs"] == 3 and agg["strict_matches"] == 1 and agg["strict_mismatches"] == 2
    assert agg["start_excluded_matches"] == 2 and agg["verdict"] == "FAIL"
    assert aggregate(rows, max_mismatched_runs=2)["verdict"] == "PASS"
    assert aggregate(rows[:1], max_mismatched_runs=0)["verdict"] == "PASS"


def test_the_start_excluded_count_never_changes_the_verdict() -> None:
    rows = [{"match": False, "start_excluded_match": True}]
    assert aggregate(rows, max_mismatched_runs=0)["verdict"] == "FAIL"


def test_replay_pos_of_the_canonical_sequence_is_one_and_of_an_empty_run_is_zero() -> None:
    exp = ExperimentDefinition.from_json("config/experiment.json")
    rc = RuntimeConfig()
    assert replay_pos(exp, rc, exp.step_ids, 0.0) == pytest.approx(1.0)
    assert replay_pos(exp, rc, [], 0.0) == pytest.approx(0.0)


def test_run_row_on_an_empty_idle_run_matches() -> None:
    exp = ExperimentDefinition.from_json("config/experiment.json")
    run = RunData("t1", "idle", [], [], [], "0" * 64)
    row = run_row(exp, run, PerceptionConfig(), RuntimeConfig())
    assert row["match"] is True and row["start_excluded_match"] is True
    assert row["cause"] == "" and row["pos"] == 0.0 and row["uncertain"] == 0
