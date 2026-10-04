"""harness/heldout.py -- the one-shot test replay: repeat guard, START-excluded metric (reporting
only), report assembly. Synthetic data and temporary files only; no real cache is opened."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from harness.heldout import RepeatTestRefused, guard_repeat_test, run_test_split


def test_guard_allows_the_first_test_replay(tmp_path: Path) -> None:
    assert guard_repeat_test(tmp_path / "replay_test.json", None) is None


def test_guard_refuses_when_the_report_already_exists(tmp_path: Path) -> None:
    report = tmp_path / "replay_test.json"
    report.write_text("{}", encoding="utf-8")
    with pytest.raises(RepeatTestRefused, match="already exists"):
        guard_repeat_test(report, None)
    with pytest.raises(RepeatTestRefused):
        guard_repeat_test(report, "   ")


def test_guard_with_a_reason_returns_it_for_the_new_report(tmp_path: Path) -> None:
    report = tmp_path / "replay_test.json"
    report.write_text("{}", encoding="utf-8")
    assert guard_repeat_test(report, "  disk full during write  ") == "disk full during write"


def test_runner_refuses_a_repeat_before_reading_anything(tmp_path: Path) -> None:
    report = tmp_path / "replay_test.json"
    report.write_text('{"keep": true}', encoding="utf-8")
    ns = argparse.Namespace(
        split="test",
        report=report,
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
    assert main(["--from-cache", "all", "--split", "test", "--report", str(report)]) == 2
    # a single run id with --split test is refused too
    assert main(["--from-cache", "x002", "--split", "test", "--report", str(report)]) == 2
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
