"""engine/reference.py -- derive_expected_deviations and the --write CLI
that fills script.json's expected_deviations (a derived field, never
hand-edited; IMPLEMENTATION_PLAN.md R8, 5.9)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from contracts import ExpectedDeviation, ExperimentDefinition
from engine.reference import derive_expected_deviations, main

FIXTURE_PATH = Path(__file__).resolve().parents[3] / "fixtures" / "experiment_4step.json"


@pytest.fixture()
def experiment() -> ExperimentDefinition:
    return ExperimentDefinition.from_json(FIXTURE_PATH)


@pytest.mark.F5
def test_correct_run_has_no_deviations(experiment: ExperimentDefinition) -> None:
    assert derive_expected_deviations(experiment, ["s1", "s2", "s3", "s4"]) == []


@pytest.mark.F5
@pytest.mark.F6
def test_omission_derivation_matches_gold_1(experiment: ExperimentDefinition) -> None:
    result = derive_expected_deviations(experiment, ["s1", "s3", "s4"])
    assert result == [ExpectedDeviation(deviation_type="omission", step_ids=["s2"])]


@pytest.mark.F6
def test_swap_derivation_matches_gold_2(experiment: ExperimentDefinition) -> None:
    result = derive_expected_deviations(experiment, ["s1", "s3", "s2", "s4"])
    assert result == [
        ExpectedDeviation(deviation_type="omission", step_ids=["s2"]),
        ExpectedDeviation(deviation_type="out_of_order", step_ids=["s2"]),
    ]


@pytest.mark.F6
def test_repeat_derivation_matches_gold_4(experiment: ExperimentDefinition) -> None:
    result = derive_expected_deviations(experiment, ["s1", "s2", "s2", "s3", "s4"])
    assert result == [ExpectedDeviation(deviation_type="repeat", step_ids=["s2"])]


@pytest.mark.F5
def test_idle_run_has_no_deviations(experiment: ExperimentDefinition) -> None:
    assert derive_expected_deviations(experiment, []) == []


@pytest.mark.F5
def test_determinism(experiment: ExperimentDefinition) -> None:
    performed = ["s1", "s3", "s2", "s4"]
    assert derive_expected_deviations(experiment, performed) == derive_expected_deviations(
        experiment, performed
    )


def _write_script(path: Path, *, expected_deviations: list[dict]) -> None:
    script = {
        "run_id": "fixture-run-1",
        "experiment_id": "fixture_4step",
        "script_type": "skip",
        "split": "train",
        "fps": 15.0,
        "camera_setup_id": "s1",
        "operator": "o1",
        "performed_steps": ["s1", "s3", "s4"],
        "expected_deviations": expected_deviations,
    }
    path.write_text(json.dumps(script), encoding="utf-8")


@pytest.mark.F5
def test_cli_write_updates_stale_script(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "fixture-run-1"
    run_dir.mkdir(parents=True)
    script_path = run_dir / "script.json"
    _write_script(script_path, expected_deviations=[])  # stale: should be [omission s2]

    rc = main(["--experiment", str(FIXTURE_PATH), "--write", str(runs_dir)])
    assert rc == 0

    written = json.loads(script_path.read_text(encoding="utf-8"))
    assert written["expected_deviations"] == [
        {"deviation_type": "omission", "step_ids": ["s2"]}
    ]


@pytest.mark.F5
def test_cli_without_write_reports_but_does_not_modify(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "fixture-run-1"
    run_dir.mkdir(parents=True)
    script_path = run_dir / "script.json"
    _write_script(script_path, expected_deviations=[])
    before = script_path.read_text(encoding="utf-8")

    rc = main(["--experiment", str(FIXTURE_PATH), str(runs_dir)])
    assert rc == 1
    assert script_path.read_text(encoding="utf-8") == before


@pytest.mark.F5
def test_cli_is_a_noop_on_fresh_scripts(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "fixture-run-1"
    run_dir.mkdir(parents=True)
    script_path = run_dir / "script.json"
    _write_script(
        script_path,
        expected_deviations=[{"deviation_type": "omission", "step_ids": ["s2"]}],
    )

    rc = main(["--experiment", str(FIXTURE_PATH), "--write", str(runs_dir)])
    assert rc == 0


@pytest.mark.F5
def test_cli_on_missing_runs_dir_is_a_noop(tmp_path: Path) -> None:
    # runs/ may not exist yet (before the pilot is recorded) -- must not crash.
    rc = main(["--experiment", str(FIXTURE_PATH), "--write", str(tmp_path / "no_such_dir")])
    assert rc == 0


@pytest.mark.F5
def test_cli_reports_one_bad_script_without_crashing_the_batch(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"

    good_dir = runs_dir / "good-run"
    good_dir.mkdir(parents=True)
    _write_script(good_dir / "script.json", expected_deviations=[])

    # a script referencing a step_id that doesn't exist in the experiment
    # -- e.g. a typo in a hand-declared performed_steps list -- must be
    # reported and skipped, not crash the whole --write run.
    bad_dir = runs_dir / "bad-run"
    bad_dir.mkdir(parents=True)
    bad_script = {
        "run_id": "bad-run",
        "experiment_id": "fixture_4step",
        "script_type": "skip",
        "split": "train",
        "fps": 15.0,
        "camera_setup_id": "s1",
        "operator": "o1",
        "performed_steps": ["s1", "s99_typo"],
        "expected_deviations": [],
    }
    (bad_dir / "script.json").write_text(json.dumps(bad_script), encoding="utf-8")

    rc = main(["--experiment", str(FIXTURE_PATH), "--write", str(runs_dir)])
    assert rc == 1

    # the good script was still processed and updated despite the bad one
    written = json.loads((good_dir / "script.json").read_text(encoding="utf-8"))
    assert written["expected_deviations"] == [
        {"deviation_type": "omission", "step_ids": ["s2"]}
    ]
    # the bad script was left untouched, not corrupted
    untouched = json.loads((bad_dir / "script.json").read_text(encoding="utf-8"))
    assert untouched["expected_deviations"] == []


@pytest.mark.F5
def test_cli_reports_malformed_json_without_crashing_the_batch(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"

    good_dir = runs_dir / "good-run"
    good_dir.mkdir(parents=True)
    _write_script(good_dir / "script.json", expected_deviations=[])

    malformed_dir = runs_dir / "malformed-run"
    malformed_dir.mkdir(parents=True)
    # missing several required RunScript fields entirely
    (malformed_dir / "script.json").write_text(
        json.dumps({"run_id": "malformed-run"}), encoding="utf-8"
    )

    rc = main(["--experiment", str(FIXTURE_PATH), "--write", str(runs_dir)])
    assert rc == 1

    written = json.loads((good_dir / "script.json").read_text(encoding="utf-8"))
    assert written["expected_deviations"] == [
        {"deviation_type": "omission", "step_ids": ["s2"]}
    ]


@pytest.mark.F5
def test_cli_reports_truncated_json_without_crashing_the_batch(tmp_path: Path) -> None:
    # e.g. a process killed mid-write, leaving syntactically broken JSON.
    runs_dir = tmp_path / "runs"

    good_dir = runs_dir / "good-run"
    good_dir.mkdir(parents=True)
    _write_script(good_dir / "script.json", expected_deviations=[])

    truncated_dir = runs_dir / "truncated-run"
    truncated_dir.mkdir(parents=True)
    (truncated_dir / "script.json").write_text(
        '{"run_id": "truncated-run", "experiment_i', encoding="utf-8"
    )

    rc = main(["--experiment", str(FIXTURE_PATH), "--write", str(runs_dir)])
    assert rc == 1

    written = json.loads((good_dir / "script.json").read_text(encoding="utf-8"))
    assert written["expected_deviations"] == [
        {"deviation_type": "omission", "step_ids": ["s2"]}
    ]
