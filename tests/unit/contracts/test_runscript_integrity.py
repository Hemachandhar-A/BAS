"""IMPLEMENTATION_PLAN.md 7.3, 'ground-truth integrity': every recorded
run's script.json.expected_deviations must equal engine.reference's
derivation, so a mislabeled clip can never corrupt the golden set.

At G0 no runs are recorded yet (the pilot is P1.1's job) and
engine.reference does not exist yet (P2.2's job) -- this file is the
scaffold both land against, per R3: P1's validator "calls P2's
engine.reference.derive_expected_deviations once it exists (before P2.2
lands it warns instead)."
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import contracts

ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = ROOT / "runs"
EXPERIMENT_PATH = ROOT / "config" / "experiment.json"


def _script_paths() -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    return sorted(RUNS_DIR.glob("*/script.json"))


_SCRIPT_PATHS = _script_paths()
_SCRIPT_IDS = [p.parent.name for p in _SCRIPT_PATHS]


@pytest.mark.parametrize("script_path", _SCRIPT_PATHS, ids=_SCRIPT_IDS)
def test_expected_deviations_match_reference_engine(script_path: Path) -> None:
    try:
        reference = importlib.import_module("engine.reference")
    except ModuleNotFoundError:
        pytest.skip(
            "engine.reference not implemented yet (P2.2) -- ISSUES.md CONTRACT/R3 notwithstanding"
        )
    script = contracts.RunScript.model_validate_json(script_path.read_text(encoding="utf-8"))
    experiment = contracts.ExperimentDefinition.from_json(EXPERIMENT_PATH)
    expected = reference.derive_expected_deviations(experiment, script.performed_steps)
    assert script.expected_deviations == expected, f"{script_path} is stale vs engine.reference"


def test_no_run_appears_in_two_splits() -> None:
    seen: dict[str, str] = {}
    for path in _script_paths():
        script = contracts.RunScript.model_validate_json(path.read_text(encoding="utf-8"))
        if script.run_id in seen and seen[script.run_id] != script.split:
            pytest.fail(
                f"{script.run_id} appears in both {seen[script.run_id]!r} and {script.split!r}"
            )
        seen[script.run_id] = script.split
