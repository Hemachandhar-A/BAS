"""engine/reference.py -- derive_expected_deviations (F5) and the --write
CLI that fills each script.json's expected_deviations, a derived field
never hand-edited (IMPLEMENTATION_PLAN.md R8, 5.9).

derive_expected_deviations replays performed_steps through a throwaway
SequenceEngine -- the same logic that runs live -- so the reference
derivation and the engine can never quietly disagree.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from contracts import ExpectedDeviation, ExperimentDefinition, RunScript, RuntimeConfig, StateEvent
from engine.sequence import SequenceEngine


def derive_expected_deviations(
    experiment: ExperimentDefinition, performed_steps: list[str]
) -> list[ExpectedDeviation]:
    engine = SequenceEngine(experiment, RuntimeConfig())
    engine.start(0.0)
    deviations: list[ExpectedDeviation] = []
    for index, step_id in enumerate(performed_steps):
        events = engine.on_state_event(
            StateEvent(t=float(index), step_id=step_id, confidence=1.0, uncertain=False)
        )
        for event in events:
            if event.kind != "deviation_detected":
                continue
            if event.deviation_type == "omission":
                deviations.append(
                    ExpectedDeviation(deviation_type="omission", step_ids=event.skipped_step_ids)
                )
            else:
                assert event.step_id is not None
                assert event.deviation_type is not None
                deviations.append(
                    ExpectedDeviation(deviation_type=event.deviation_type, step_ids=[event.step_id])
                )
    return deviations


def _iter_scripts(runs_dir: Path) -> list[Path]:
    return sorted(runs_dir.glob("*/script.json"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fill each script.json's expected_deviations (derived, never hand-edited).",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Overwrite stale expected_deviations in place; without it, report only",
    )
    parser.add_argument("runs_dir", nargs="?", default="runs", type=Path)
    parser.add_argument(
        "--experiment",
        default="config/experiment.json",
        type=Path,
        help="Path to the ExperimentDefinition (default: config/experiment.json)",
    )
    args = parser.parse_args(argv)

    experiment = ExperimentDefinition.from_json(args.experiment)
    stale: list[Path] = []
    for path in _iter_scripts(args.runs_dir):
        script = RunScript.model_validate_json(path.read_text(encoding="utf-8"))
        expected = derive_expected_deviations(experiment, script.performed_steps)
        if script.expected_deviations != expected:
            stale.append(path)
            if args.write:
                updated = script.model_copy(update={"expected_deviations": expected})
                path.write_text(updated.model_dump_json(indent=2) + "\n", encoding="utf-8")

    verb = "updated" if args.write else "stale"
    for path in stale:
        print(f"{verb}: {path}")
    if stale and not args.write:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
