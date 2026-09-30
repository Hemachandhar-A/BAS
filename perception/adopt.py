"""perception/adopt.py -- adopts pre-recorded, crew-labeled footage
(data/intake/labels.csv + data/intake/map.csv + data/intake/working/<id>.mp4,
all git-ignored; ISSUES.md 2026-09-30 "crew footage intake" DECISION) into
the runs/ layout Stage 0 (perception/record.py) would have produced
directly, so it feeds F14 stages 2+ the same way a recorded run would.

    python -m perception.adopt [--dry-run] [--force] [--split-file PATH]

Never re-derives expected_deviations itself: calls
engine.reference.derive_expected_deviations (IMPLEMENTATION_PLAN.md R3, a
sanctioned P1 -> P2 call, same as perception/record.py's Stage 1).
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

from contracts import ExperimentDefinition, RunScript, sha256_of_file
from engine.reference import derive_expected_deviations

DEFAULT_EXPERIMENT_PATH = Path("config/experiment.json")
DEFAULT_INTAKE_DIR = Path("data/intake")
DEFAULT_RUNS_ROOT = Path("runs")

# The crew's step codes (labels.csv "steps") are the 1-based index into
# config/experiment.json's step_ids. This fixed order is what the crew
# actually used when writing labels.csv, and is asserted against the live
# experiment before any file is touched (verify_step_order) so a silently
# reordered experiment.json can never mislabel a run.
EXPECTED_STEP_ORDER = [
    "red_out",
    "red_in_tray",
    "yellow_out",
    "yellow_in_tray",
    "start_pressed",
    "red_stowed",
    "yellow_stowed",
]

# Target run counts per split, by script_type: a seeded, stratified,
# provisional split (ISSUES.md, this session's DECISION) covering the 46
# adopted clips exactly (train/val/test).
TARGET_SPLIT_COUNTS: dict[str, tuple[int, int, int]] = {
    "correct": (14, 4, 4),
    "skip": (5, 2, 2),
    "swap": (5, 2, 2),
    "repeat": (1, 1, 1),
    "idle": (1, 1, 1),
}
SPLIT_SEED = 0

_PROVENANCE_FIELDS = [
    "run_id",
    "original_name",
    "sha256",
    "crew_label",
    "label_source",
    "checked_by",
    "notes",
]


class AdoptError(Exception):
    """Raised for anything that must stop the whole adoption batch: a step
    order mismatch, a sha256 mismatch, an existing run directory without
    --force, an inconsistent --split-file. Never coerced."""


# ---------------------------------------------------------------------------
# data/intake/{labels,map}.csv
# ---------------------------------------------------------------------------


class LabelRow:
    def __init__(self, raw: dict[str, str]) -> None:
        self.id = raw["id"]
        self.crew_label = raw["crew_label"]
        self.label_source = raw["label_source"]
        self.steps = raw["steps"]
        self.script_type = raw["script_type"]
        self.checked_by = raw.get("checked_by", "")
        self.notes = raw.get("notes", "")


class MapRow:
    def __init__(self, raw: dict[str, str]) -> None:
        self.id = raw["id"]
        self.original_name = raw["original_name"]
        self.sha256 = raw["sha256"]
        self.fps_measured = float(raw["fps_measured"])


def read_labels(path: Path) -> list[LabelRow]:
    with open(path, newline="", encoding="utf-8") as f:
        return [LabelRow(row) for row in csv.DictReader(f)]


def read_map(path: Path) -> dict[str, MapRow]:
    with open(path, newline="", encoding="utf-8") as f:
        return {row["id"]: MapRow(row) for row in csv.DictReader(f)}


def read_split_file(path: Path) -> dict[str, str]:
    with open(path, newline="", encoding="utf-8") as f:
        return {row["run_id"]: row["split"] for row in csv.DictReader(f)}


# ---------------------------------------------------------------------------
# Step-code parsing and experiment-order verification
# ---------------------------------------------------------------------------


def verify_step_order(experiment: ExperimentDefinition) -> None:
    if experiment.step_ids != EXPECTED_STEP_ORDER:
        raise AdoptError(
            "config/experiment.json step order does not match labels.csv's "
            f"step codes 1-7: expected {EXPECTED_STEP_ORDER}, got {experiment.step_ids}"
        )


def parse_performed_steps(steps_field: str, step_ids: list[str]) -> list[str]:
    """``steps_field`` is a space-separated list of 1-based codes into
    ``step_ids``; ``"0"`` alone means an empty (idle) performed_steps."""
    codes = steps_field.split()
    if codes == ["0"]:
        return []
    if "0" in codes:
        raise AdoptError(
            f"code 0 (idle) must not be combined with other step codes: {steps_field!r}"
        )
    performed: list[str] = []
    for raw_code in codes:
        try:
            code = int(raw_code)
        except ValueError as exc:
            raise AdoptError(f"non-numeric step code {raw_code!r} in {steps_field!r}") from exc
        if not (1 <= code <= len(step_ids)):
            raise AdoptError(f"step code {code} out of range 1-{len(step_ids)} in {steps_field!r}")
        performed.append(step_ids[code - 1])
    return performed


# ---------------------------------------------------------------------------
# Seeded stratified split
# ---------------------------------------------------------------------------


def assign_splits(
    ids_by_type: dict[str, list[str]],
    target_counts: dict[str, tuple[int, int, int]],
    seed: int = SPLIT_SEED,
) -> dict[str, str]:
    """Deterministic, seeded, stratified split: within each script_type, a
    seeded shuffle of the sorted ids, then a straight slice into
    train/val/test per ``target_counts``. Raises if a type's id count
    doesn't match its target sum, so a silent short split is never
    produced, and no id can land in two splits (each is assigned exactly
    once)."""
    assignment: dict[str, str] = {}
    for script_type, ids in ids_by_type.items():
        if script_type not in target_counts:
            raise AdoptError(f"no target split counts for script_type {script_type!r}")
        train_n, val_n, test_n = target_counts[script_type]
        if len(ids) != train_n + val_n + test_n:
            raise AdoptError(
                f"script_type {script_type!r} has {len(ids)} ids, expected "
                f"{train_n + val_n + test_n} ({train_n}/{val_n}/{test_n})"
            )
        shuffled = sorted(ids)
        random.Random(seed).shuffle(shuffled)
        for run_id in shuffled[:train_n]:
            assignment[run_id] = "train"
        for run_id in shuffled[train_n : train_n + val_n]:
            assignment[run_id] = "val"
        for run_id in shuffled[train_n + val_n :]:
            assignment[run_id] = "test"
    return assignment


# ---------------------------------------------------------------------------
# Planning and adopting one run
# ---------------------------------------------------------------------------


class AdoptedRun:
    def __init__(
        self, run_id: str, script_type: str, split: str, performed_steps: list[str]
    ) -> None:
        self.run_id = run_id
        self.script_type = script_type
        self.split = split
        self.performed_steps = performed_steps


def plan_adoption(
    labels: list[LabelRow],
    id_to_map: dict[str, MapRow],
    experiment: ExperimentDefinition,
    split_overrides: dict[str, str] | None = None,
    target_counts: dict[str, tuple[int, int, int]] | None = None,
) -> list[AdoptedRun]:
    for row in labels:
        if row.id not in id_to_map:
            raise AdoptError(f"{row.id} has a labels.csv row but no map.csv row")

    if split_overrides is not None:
        missing = [row.id for row in labels if row.id not in split_overrides]
        if missing:
            raise AdoptError(f"--split-file is missing ids: {missing}")
        assignment = dict(split_overrides)
    else:
        ids_by_type: dict[str, list[str]] = {}
        for row in labels:
            ids_by_type.setdefault(row.script_type, []).append(row.id)
        assignment = assign_splits(ids_by_type, target_counts or TARGET_SPLIT_COUNTS)

    return [
        AdoptedRun(
            row.id,
            row.script_type,
            assignment[row.id],
            parse_performed_steps(row.steps, experiment.step_ids),
        )
        for row in labels
    ]


def adopt_one(
    run: AdoptedRun,
    label: LabelRow,
    map_row: MapRow,
    working_dir: Path,
    runs_root: Path,
    experiment: ExperimentDefinition,
    *,
    force: bool,
    dry_run: bool,
) -> Path:
    src_video = working_dir / f"{run.run_id}.mp4"
    if not src_video.exists():
        raise AdoptError(f"missing source video {src_video}")
    actual_sha256 = sha256_of_file(src_video)
    if actual_sha256 != map_row.sha256:
        raise AdoptError(
            f"{run.run_id}: sha256 mismatch -- map.csv says {map_row.sha256}, "
            f"{src_video} is {actual_sha256}"
        )

    run_dir = runs_root / run.run_id
    if run_dir.exists() and not force:
        raise AdoptError(f"{run_dir} already exists (use --force to overwrite)")

    script = RunScript(
        run_id=run.run_id,
        experiment_id=experiment.experiment_id,
        script_type=run.script_type,
        split=run.split,
        fps=map_row.fps_measured,
        camera_setup_id="unknown",
        operator="unknown",
        performed_steps=run.performed_steps,
        expected_deviations=derive_expected_deviations(experiment, run.performed_steps),
    )

    if dry_run:
        return run_dir

    run_dir.mkdir(parents=True, exist_ok=True)
    dest_video = run_dir / "video.mp4"
    shutil.copyfile(src_video, dest_video)  # byte-for-byte, no transcode
    if sha256_of_file(dest_video) != map_row.sha256:
        raise AdoptError(f"{run.run_id}: copied video's sha256 does not match after copy")
    (run_dir / "script.json").write_text(script.model_dump_json(indent=2), encoding="utf-8")
    return run_dir


def write_provenance(
    path: Path, rows: list[tuple[AdoptedRun, LabelRow, MapRow]]
) -> None:
    """RunScript has no notes field (contracts.py), so provenance -- the
    crew's original filename, hash and label pedigree -- goes in this
    tracked sibling file instead (runs/ is a Data-crew + P1 shared
    directory, IMPLEMENTATION_PLAN.md Part 4; see ISSUES.md DECISION)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_PROVENANCE_FIELDS)
        writer.writeheader()
        for run, label, map_row in rows:
            writer.writerow(
                {
                    "run_id": run.run_id,
                    "original_name": map_row.original_name,
                    "sha256": map_row.sha256,
                    "crew_label": label.crew_label,
                    "label_source": label.label_source,
                    "checked_by": label.checked_by,
                    "notes": label.notes,
                }
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intake-dir", type=Path, default=DEFAULT_INTAKE_DIR)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT_PATH)
    parser.add_argument(
        "--split-file",
        type=Path,
        default=None,
        help="CSV with columns run_id,split -- overrides the seeded stratified split",
    )
    parser.add_argument("--force", action="store_true", help="overwrite an existing runs/<id>/")
    parser.add_argument("--dry-run", action="store_true", help="plan and print only, write nothing")
    args = parser.parse_args(argv)

    try:
        experiment = ExperimentDefinition.from_json(args.experiment)
        verify_step_order(experiment)

        labels = read_labels(args.intake_dir / "labels.csv")
        id_to_map = read_map(args.intake_dir / "map.csv")
        split_overrides = read_split_file(args.split_file) if args.split_file else None
        plan = plan_adoption(labels, id_to_map, experiment, split_overrides)

        label_by_id = {row.id: row for row in labels}
        provenance_rows: list[tuple[AdoptedRun, LabelRow, MapRow]] = []

        print(f"{'run_id':<10} {'type':<8} {'split':<6} steps")
        for run in plan:
            steps_display = " ".join(run.performed_steps) if run.performed_steps else "(idle)"
            print(f"{run.run_id:<10} {run.script_type:<8} {run.split:<6} {steps_display}")
            adopt_one(
                run,
                label_by_id[run.run_id],
                id_to_map[run.run_id],
                args.intake_dir / "working",
                args.runs_root,
                experiment,
                force=args.force,
                dry_run=args.dry_run,
            )
            provenance_rows.append((run, label_by_id[run.run_id], id_to_map[run.run_id]))

        if not args.dry_run:
            write_provenance(args.runs_root / "provenance.csv", provenance_rows)
    except AdoptError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
