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
