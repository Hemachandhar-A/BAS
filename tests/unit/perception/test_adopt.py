"""perception/adopt.py -- adopts the 46 pre-recorded, crew-labeled intake
clips (data/intake/{labels,map}.csv + data/intake/working/<id>.mp4, all
git-ignored) into the runs/ layout Stage 0 (perception/record.py) would
have produced directly. Tested here against a tiny generated clip and a
fake labels.csv/map.csv, never against the real intake tree, per
essential-features.md's F14 done-when ("each run on a fixture")."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from contracts import ExperimentDefinition, RunScript, sha256_of_file
from engine.reference import derive_expected_deviations
from perception import adopt

pytestmark = pytest.mark.F14

EXPERIMENT_PATH = Path(__file__).resolve().parents[3] / "config" / "experiment.json"


@pytest.fixture(scope="module")
def experiment() -> ExperimentDefinition:
    return ExperimentDefinition.from_json(EXPERIMENT_PATH)


def _write_clip(
    path: Path, fps: float = 10.0, n_frames: int = 40, width: int = 32, height: int = 24
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (width, height))
    assert writer.isOpened()
    for i in range(n_frames):
        writer.write(np.full((height, width, 3), 50 + (i * 7) % 150, dtype=np.uint8))
    writer.release()


def _write_intake(
    intake_dir: Path,
    rows: list[dict[str, str]],
) -> None:
    """Writes labels.csv + map.csv from ``rows``, and a tiny generated clip
    per row at data/intake/working/<id>.mp4, with map.csv's sha256/fps
    computed from the actual written file (never hand-typed)."""
    working_dir = intake_dir / "working"
    label_fields = [
        "id",
        "crew_label",
        "label_source",
        "steps",
        "script_type",
        "operator",
        "camera_setup_id",
        "condition",
        "checked_by",
        "notes",
    ]
    map_fields = [
        "id",
        "original_name",
        "group",
        "sha256",
        "width",
        "height",
        "fps_measured",
        "frames",
        "duration_s",
    ]
    map_rows = []
    for row in rows:
        clip_path = working_dir / f"{row['id']}.mp4"
        fps = float(row.get("fps", 10.0))
        n_frames = int(row.get("n_frames", 40))
        _write_clip(clip_path, fps=fps, n_frames=n_frames)
        map_rows.append(
            {
                "id": row["id"],
                "original_name": f"{row['id']}.mp4",
                "group": "good",
                "sha256": sha256_of_file(clip_path),
                "width": 32,
                "height": 24,
                "fps_measured": fps,
                "frames": n_frames,
                "duration_s": n_frames / fps,
            }
        )

    intake_dir.mkdir(parents=True, exist_ok=True)
    with open(intake_dir / "labels.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=label_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in label_fields})

    with open(intake_dir / "map.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=map_fields)
        writer.writeheader()
        for row in map_rows:
            writer.writerow(row)


def _label_row(**overrides) -> dict[str, str]:
    row = {
        "id": "x001",
        "crew_label": "good",
        "label_source": "crew_label_only",
        "steps": "1 2 3 4 5 6 7",
        "script_type": "correct",
        "operator": "unknown",
        "camera_setup_id": "unknown",
        "condition": "unknown",
        "checked_by": "",
        "notes": "",
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# verify_step_order
# ---------------------------------------------------------------------------


def test_verify_step_order_passes_for_the_real_experiment(
    experiment: ExperimentDefinition,
) -> None:
    adopt.verify_step_order(experiment)  # must not raise


def test_verify_step_order_rejects_a_different_order(experiment: ExperimentDefinition) -> None:
    reordered = experiment.model_copy(
        update={"steps": [experiment.steps[1], experiment.steps[0], *experiment.steps[2:]]}
    )
    with pytest.raises(adopt.AdoptError, match="step order"):
        adopt.verify_step_order(reordered)


# ---------------------------------------------------------------------------
# parse_performed_steps
# ---------------------------------------------------------------------------


def test_parse_performed_steps_maps_codes_to_step_ids(experiment: ExperimentDefinition) -> None:
    assert adopt.parse_performed_steps("1 2 3", experiment.step_ids) == [
        "red_out",
        "red_in_tray",
        "yellow_out",
    ]


def test_parse_performed_steps_zero_means_idle(experiment: ExperimentDefinition) -> None:
    assert adopt.parse_performed_steps("0", experiment.step_ids) == []


def test_parse_performed_steps_rejects_out_of_range_code(
    experiment: ExperimentDefinition,
) -> None:
    with pytest.raises(adopt.AdoptError, match="out of range"):
        adopt.parse_performed_steps("1 8", experiment.step_ids)


def test_parse_performed_steps_rejects_non_numeric_code(experiment: ExperimentDefinition) -> None:
    with pytest.raises(adopt.AdoptError, match="non-numeric"):
        adopt.parse_performed_steps("1 x", experiment.step_ids)


def test_parse_performed_steps_rejects_zero_combined_with_others(
    experiment: ExperimentDefinition,
) -> None:
    with pytest.raises(adopt.AdoptError, match="idle"):
        adopt.parse_performed_steps("0 1", experiment.step_ids)


# ---------------------------------------------------------------------------
# assign_splits
# ---------------------------------------------------------------------------


def test_assign_splits_meets_target_counts() -> None:
    ids_by_type = {"correct": ["a", "b", "c", "d"], "idle": ["e"]}
    targets = {"correct": (2, 1, 1), "idle": (1, 0, 0)}
    assignment = adopt.assign_splits(ids_by_type, targets, seed=0)

    assert set(assignment) == {"a", "b", "c", "d", "e"}
    from collections import Counter

    counts = Counter(assignment[i] for i in ["a", "b", "c", "d"])
    assert counts == {"train": 2, "val": 1, "test": 1}
    assert assignment["e"] == "train"


def test_assign_splits_is_deterministic() -> None:
    ids_by_type = {"correct": ["a", "b", "c", "d"]}
    targets = {"correct": (2, 1, 1)}
    first = adopt.assign_splits(ids_by_type, targets, seed=0)
    second = adopt.assign_splits(ids_by_type, targets, seed=0)
    assert first == second


def test_assign_splits_raises_on_count_mismatch() -> None:
    ids_by_type = {"correct": ["a", "b", "c"]}
    targets = {"correct": (2, 1, 1)}  # sums to 4, only 3 ids
    with pytest.raises(adopt.AdoptError, match="expected"):
        adopt.assign_splits(ids_by_type, targets)


# ---------------------------------------------------------------------------
# End to end: main()
# ---------------------------------------------------------------------------


def test_main_adopts_runs_writes_scripts_and_provenance(
    tmp_path: Path, experiment: ExperimentDefinition
) -> None:
    intake_dir = tmp_path / "intake"
    runs_root = tmp_path / "runs"
    _write_intake(
        intake_dir,
        [
            _label_row(id="x001", steps="1 2 3 4 5 6 7", script_type="correct"),
            _label_row(id="x002", steps="0", script_type="idle", crew_label="bad"),
        ],
    )
    split_file = tmp_path / "splits.csv"
    with open(split_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "split"])
        writer.writerow(["x001", "train"])
        writer.writerow(["x002", "val"])

    rc = adopt.main(
        [
            "--intake-dir",
            str(intake_dir),
            "--runs-root",
            str(runs_root),
            "--experiment",
            str(EXPERIMENT_PATH),
            "--split-file",
            str(split_file),
        ]
    )
    assert rc == 0

    src_x001 = intake_dir / "working" / "x001.mp4"
    dest_x001 = runs_root / "x001" / "video.mp4"
    assert dest_x001.exists()
    assert sha256_of_file(dest_x001) == sha256_of_file(src_x001)

    script = RunScript.model_validate_json((runs_root / "x001" / "script.json").read_text())
    assert script.run_id == "x001"
    assert script.experiment_id == experiment.experiment_id
    assert script.script_type == "correct"
    assert script.split == "train"
    assert script.performed_steps == [
        "red_out",
        "red_in_tray",
        "yellow_out",
        "yellow_in_tray",
        "start_pressed",
        "red_stowed",
        "yellow_stowed",
    ]
    assert script.camera_setup_id == "unknown"
    assert script.operator == "unknown"
    assert script.expected_deviations == derive_expected_deviations(
        experiment, script.performed_steps
    )

    idle_script = RunScript.model_validate_json(
        (runs_root / "x002" / "script.json").read_text()
    )
    assert idle_script.performed_steps == []
    assert idle_script.split == "val"
    assert idle_script.expected_deviations == []

    provenance_path = runs_root / "provenance.csv"
    assert provenance_path.exists()
    with open(provenance_path, newline="", encoding="utf-8") as f:
        provenance_rows = {row["run_id"]: row for row in csv.DictReader(f)}
    assert provenance_rows["x001"]["crew_label"] == "good"
    assert provenance_rows["x001"]["original_name"] == "x001.mp4"
    assert provenance_rows["x002"]["crew_label"] == "bad"


def test_main_dry_run_writes_nothing(tmp_path: Path) -> None:
    intake_dir = tmp_path / "intake"
    runs_root = tmp_path / "runs"
    _write_intake(intake_dir, [_label_row(id="x001")])
    split_file = tmp_path / "splits.csv"
    with open(split_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "split"])
        writer.writerow(["x001", "train"])

    rc = adopt.main(
        [
            "--intake-dir",
            str(intake_dir),
            "--runs-root",
            str(runs_root),
            "--experiment",
            str(EXPERIMENT_PATH),
            "--split-file",
            str(split_file),
            "--dry-run",
        ]
    )
    assert rc == 0
    assert not runs_root.exists() or list(runs_root.iterdir()) == []


def test_main_aborts_on_sha256_mismatch(tmp_path: Path) -> None:
    intake_dir = tmp_path / "intake"
    runs_root = tmp_path / "runs"
    _write_intake(intake_dir, [_label_row(id="x001")])
    # Corrupt map.csv's recorded sha256 for x001.
    map_path = intake_dir / "map.csv"
    rows = list(csv.DictReader(open(map_path, newline="", encoding="utf-8")))
    rows[0]["sha256"] = "0" * 64
    with open(map_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    split_file = tmp_path / "splits.csv"
    with open(split_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "split"])
        writer.writerow(["x001", "train"])

    rc = adopt.main(
        [
            "--intake-dir",
            str(intake_dir),
            "--runs-root",
            str(runs_root),
            "--experiment",
            str(EXPERIMENT_PATH),
            "--split-file",
            str(split_file),
        ]
    )
    assert rc == 1
    assert not (runs_root / "x001").exists()


def test_adopt_one_refuses_to_overwrite_without_force(
    tmp_path: Path, experiment: ExperimentDefinition
) -> None:
    intake_dir = tmp_path / "intake"
    runs_root = tmp_path / "runs"
    _write_intake(intake_dir, [_label_row(id="x001")])
    (runs_root / "x001").mkdir(parents=True)

    label = adopt.read_labels(intake_dir / "labels.csv")[0]
    map_row = adopt.read_map(intake_dir / "map.csv")["x001"]
    run = adopt.AdoptedRun("x001", "correct", "train", experiment.step_ids)

    with pytest.raises(adopt.AdoptError, match="already exists"):
        adopt.adopt_one(
            run,
            label,
            map_row,
            intake_dir / "working",
            runs_root,
            experiment,
            force=False,
            dry_run=False,
        )

    # With --force it must succeed.
    adopt.adopt_one(
        run,
        label,
        map_row,
        intake_dir / "working",
        runs_root,
        experiment,
        force=True,
        dry_run=False,
    )
    assert (runs_root / "x001" / "video.mp4").exists()


def test_split_file_missing_an_id_raises(tmp_path: Path, experiment: ExperimentDefinition) -> None:
    intake_dir = tmp_path / "intake"
    _write_intake(
        intake_dir,
        [
            _label_row(id="x001"),
            _label_row(id="x002", crew_label="bad", script_type="idle", steps="0"),
        ],
    )
    labels = adopt.read_labels(intake_dir / "labels.csv")
    id_to_map = adopt.read_map(intake_dir / "map.csv")

    with pytest.raises(adopt.AdoptError, match="missing"):
        adopt.plan_adoption(labels, id_to_map, experiment, split_overrides={"x001": "train"})
