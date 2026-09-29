"""F14 stages 0-1 (essential-features.md section 14): perception/record.py.
The interactive parts of Stage 0 (live preview, countdown, keypress
stop/discard) are thin CLI glue with no logic of their own and aren't
exercised here; everything they delegate to -- capture_run,
build_run_script, write_script_json, probe_video, validate_run, the
manifest writer -- is tested directly against a generated clip and a fake
FrameSource, per essential-features.md's F14 done-when ("each run on a
fixture")."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from contracts import ExperimentDefinition, RunScript
from perception import record

pytestmark = pytest.mark.F14

EXPERIMENT_PATH = Path(__file__).resolve().parents[3] / "config" / "experiment.json"


@pytest.fixture(scope="module")
def experiment() -> ExperimentDefinition:
    return ExperimentDefinition.from_json(EXPERIMENT_PATH)


class _Frame:
    def __init__(self, frame_id: int, t: float, image: np.ndarray) -> None:
        self.frame_id = frame_id
        self.t = t
        self.image = image


class _FakeSource:
    """A minimal FrameSource-shaped fake: yields `images` in order, then
    reports exhausted, with no camera or GUI involved."""

    def __init__(self, images: list[np.ndarray], fps: float = 10.0) -> None:
        self._images = images
        self._index = 0
        self.fps = fps
        self.exhausted = False
        self.closed = False

    def read(self):
        if self._index >= len(self._images):
            self.exhausted = True
            return None
        image = self._images[self._index]
        frame = _Frame(self._index, self._index / self.fps, image)
        self._index += 1
        return frame

    def close(self) -> None:
        self.closed = True


def _make_plan_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=record._PLAN_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in record._PLAN_FIELDS})


def _plan_row(**overrides) -> dict[str, str]:
    row = {
        "run_id": "001-correct",
        "split": "train",
        "script_type": "correct",
        "variant": "c",
        "camera_setup_id": "S1",
        "operator": "O1",
        "planned_performed_steps": "red_out|red_in_tray|yellow_out",
        "intent": "Perform every step in order.",
        "condition": "normal",
        "recorded": "",
        "recorded_by": "",
        "notes": "",
    }
    row.update(overrides)
    return row


def _write_clip(
    path: Path,
    fps: float = 10.0,
    n_frames: int = 20,
    width: int = 32,
    height: int = 24,
    frozen: bool = False,
) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (width, height))
    assert writer.isOpened()
    for i in range(n_frames):
        value = 100 if frozen else 50 + (i * 7) % 150
        writer.write(np.full((height, width, 3), value, dtype=np.uint8))
    writer.release()


# ---------------------------------------------------------------------------
# runs/run_plan.csv
# ---------------------------------------------------------------------------


def test_read_plan_parses_piped_steps_and_pending_rows(tmp_path: Path) -> None:
    plan_path = tmp_path / "run_plan.csv"
    _make_plan_csv(
        plan_path,
        [
            _plan_row(run_id="001-correct"),
            _plan_row(run_id="002-idle", planned_performed_steps="", recorded="2026-09-29"),
        ],
    )
    rows = record.read_plan(plan_path)
    assert [r.run_id for r in rows] == ["001-correct", "002-idle"]
    assert rows[0].planned_performed_steps == ["red_out", "red_in_tray", "yellow_out"]
    assert rows[1].planned_performed_steps == []

    pending = record.pending_rows(rows)
    assert [r.run_id for r in pending] == ["001-correct"]


def test_write_plan_round_trips_recorded_updates(tmp_path: Path) -> None:
    plan_path = tmp_path / "run_plan.csv"
    _make_plan_csv(plan_path, [_plan_row(run_id="001-correct")])
    rows = record.read_plan(plan_path)
    rows[0].recorded = "2026-09-29"
    rows[0].recorded_by = "O1"
    record.write_plan(plan_path, rows)

    reread = record.read_plan(plan_path)
    assert reread[0].recorded == "2026-09-29"
    assert reread[0].recorded_by == "O1"
    assert reread[0].planned_performed_steps == ["red_out", "red_in_tray", "yellow_out"]


# ---------------------------------------------------------------------------
# Stage 0: capture_run, build_run_script, write_script_json
# ---------------------------------------------------------------------------


def test_capture_run_writes_every_frame_until_should_stop(tmp_path: Path) -> None:
    images = [np.full((24, 32, 3), i, dtype=np.uint8) for i in range(10)]
    source = _FakeSource(images, fps=10.0)
    video_path = tmp_path / "run" / "video.mp4"

    count = record.capture_run(
        source, video_path, writer_fps=10.0, should_stop=lambda f: f.frame_id >= 4
    )

    assert count == 5  # frames 0..4 inclusive
    assert video_path.exists()
    cap = cv2.VideoCapture(str(video_path))
    written = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        written += 1
    cap.release()
    assert written == 5


def test_capture_run_stops_on_source_exhaustion(tmp_path: Path) -> None:
    images = [np.zeros((24, 32, 3), dtype=np.uint8) for _ in range(3)]
    source = _FakeSource(images, fps=10.0)
    video_path = tmp_path / "run" / "video.mp4"

    count = record.capture_run(source, video_path, writer_fps=10.0, should_stop=lambda f: False)

    assert count == 3
    assert source.exhausted is True


def test_build_run_script_leaves_expected_deviations_empty(
    experiment: ExperimentDefinition,
) -> None:
    plan_path = None  # unused; build a PlanRow directly
    row = record.PlanRow(_plan_row())
    script = record.build_run_script(
        row, experiment.experiment_id, measured_fps=14.8, performed_steps=["red_out", "red_in_tray"]
    )
    assert script.run_id == "001-correct"
    assert script.experiment_id == experiment.experiment_id
    assert script.fps == pytest.approx(14.8)
    assert script.performed_steps == ["red_out", "red_in_tray"]
    assert script.expected_deviations == []
    del plan_path


def test_write_script_json_round_trips_through_the_contract(
    tmp_path: Path, experiment: ExperimentDefinition
) -> None:
    row = record.PlanRow(_plan_row())
    script = record.build_run_script(row, experiment.experiment_id, 15.0, ["red_out"])
    path = record.write_script_json(tmp_path, script)

    assert path == tmp_path / "001-correct" / "script.json"
    reloaded = RunScript.model_validate_json(path.read_text(encoding="utf-8"))
    assert reloaded == script


# ---------------------------------------------------------------------------
# Stage 1: probe_video
# ---------------------------------------------------------------------------


def test_probe_video_reports_fps_frames_resolution_and_motion(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    _write_clip(clip, fps=10.0, n_frames=20, width=32, height=24)

    probe = record.probe_video(clip)

    assert probe["fps"] == pytest.approx(10.0)
    assert probe["frames"] == 20
    assert probe["duration_s"] == pytest.approx(2.0)
    assert probe["width"] == 32
    assert probe["height"] == 24
    assert probe["mean_frame_diff"] > record._FROZEN_MEAN_ABS_DIFF


def test_probe_video_detects_a_frozen_clip(tmp_path: Path) -> None:
    clip = tmp_path / "frozen.mp4"
    _write_clip(clip, fps=10.0, n_frames=20, frozen=True)

    probe = record.probe_video(clip)

    assert probe["mean_frame_diff"] == pytest.approx(0.0, abs=1e-6)


def test_probe_video_raises_source_error_for_a_missing_file(tmp_path: Path) -> None:
    from contracts import SourceError

    with pytest.raises(SourceError):
        record.probe_video(tmp_path / "nope.mp4")


# ---------------------------------------------------------------------------
# Stage 1: validate_run / cmd_validate
# ---------------------------------------------------------------------------


def _write_run(
    runs_root: Path,
    run_id: str,
    experiment: ExperimentDefinition,
    *,
    fps: float = 2.0,
    n_frames: int = 50,
    width: int = 32,
    height: int = 24,
    script_fps: float | None = None,
    performed_steps: list[str] | None = None,
    camera_setup_id: str = "S1",
    write_video: bool = True,
    write_script: bool = True,
) -> Path:
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True)
    if write_video:
        _write_clip(run_dir / "video.mp4", fps=fps, n_frames=n_frames, width=width, height=height)
    if write_script:
        script = RunScript(
            run_id=run_id,
            experiment_id=experiment.experiment_id,
            script_type="correct",
            split="train",
            fps=script_fps if script_fps is not None else fps,
            camera_setup_id=camera_setup_id,
            operator="O1",
            performed_steps=performed_steps if performed_steps is not None else ["red_out"],
            expected_deviations=[],
        )
        (run_dir / "script.json").write_text(script.model_dump_json(indent=2), encoding="utf-8")
    return run_dir


def test_validate_run_happy_path(tmp_path: Path, experiment: ExperimentDefinition) -> None:
    # fps=2.0, 50 frames -> 25s duration, inside the 20-150s band.
    run_dir = _write_run(tmp_path, "001-correct", experiment, fps=2.0, n_frames=50)

    result = record.validate_run(run_dir, experiment, EXPERIMENT_PATH)

    assert result.ok, result.issues
    assert result.script is not None
    assert result.probe is not None
    assert result.probe["frames"] == 50


def test_validate_run_flags_unknown_step_id(
    tmp_path: Path, experiment: ExperimentDefinition
) -> None:
    run_dir = _write_run(
        tmp_path,
        "001-correct",
        experiment,
        fps=2.0,
        n_frames=50,
        performed_steps=["not_a_real_step"],
    )
    result = record.validate_run(run_dir, experiment, EXPERIMENT_PATH)
    assert not result.ok
    assert any("unknown step ids" in issue for issue in result.issues)


def test_validate_run_flags_stale_expected_deviations_for_valid_step_ids(
    tmp_path: Path, experiment: ExperimentDefinition
) -> None:
    # All step ids here are real (engine/reference.py has landed on this branch),
    # so the staleness check must still run and catch the mismatch -- it must
    # only be skipped when performed_steps itself is invalid.
    run_dir = _write_run(
        tmp_path,
        "001-correct",
        experiment,
        fps=2.0,
        n_frames=50,
        performed_steps=["red_out", "yellow_out"],  # skips red_in_tray
    )
    result = record.validate_run(run_dir, experiment, EXPERIMENT_PATH)
    assert not result.ok
    assert any("expected_deviations is stale" in issue for issue in result.issues)


def test_validate_run_flags_fps_mismatch(tmp_path: Path, experiment: ExperimentDefinition) -> None:
    run_dir = _write_run(
        tmp_path, "001-correct", experiment, fps=2.0, n_frames=50, script_fps=30.0
    )
    result = record.validate_run(run_dir, experiment, EXPERIMENT_PATH)
    assert not result.ok
    assert any("differs from declared" in issue for issue in result.issues)


def test_validate_run_flags_duration_out_of_band(
    tmp_path: Path, experiment: ExperimentDefinition
) -> None:
    # fps=5, 10 frames -> 2s duration, well under the 20s floor.
    run_dir = _write_run(tmp_path, "001-correct", experiment, fps=5.0, n_frames=10)
    result = record.validate_run(run_dir, experiment, EXPERIMENT_PATH)
    assert not result.ok
    assert any("duration" in issue for issue in result.issues)


def test_validate_run_flags_a_frozen_video(
    tmp_path: Path, experiment: ExperimentDefinition
) -> None:
    run_dir = tmp_path / "001-correct"
    run_dir.mkdir()
    _write_clip(run_dir / "video.mp4", fps=2.0, n_frames=50, frozen=True)
    script = RunScript(
        run_id="001-correct",
        experiment_id=experiment.experiment_id,
        script_type="correct",
        split="train",
        fps=2.0,
        camera_setup_id="S1",
        operator="O1",
        performed_steps=["red_out"],
        expected_deviations=[],
    )
    (run_dir / "script.json").write_text(script.model_dump_json(indent=2), encoding="utf-8")

    result = record.validate_run(run_dir, experiment, EXPERIMENT_PATH)
    assert not result.ok
    assert any("frozen" in issue for issue in result.issues)


def test_validate_run_flags_missing_script_and_missing_video(
    tmp_path: Path, experiment: ExperimentDefinition
) -> None:
    no_script_dir = _write_run(tmp_path, "001-no-script", experiment, write_script=False)
    result = record.validate_run(no_script_dir, experiment, EXPERIMENT_PATH)
    assert not result.ok
    assert any("missing script.json" in issue for issue in result.issues)

    no_video_dir = _write_run(tmp_path, "002-no-video", experiment, write_video=False)
    result = record.validate_run(no_video_dir, experiment, EXPERIMENT_PATH)
    assert not result.ok
    assert any("missing video.mp4" in issue for issue in result.issues)


def test_validate_run_rejects_a_malformed_script(
    tmp_path: Path, experiment: ExperimentDefinition
) -> None:
    run_dir = tmp_path / "001-bad-script"
    run_dir.mkdir()
    (run_dir / "script.json").write_text('{"not": "a run script"}', encoding="utf-8")
    result = record.validate_run(run_dir, experiment, EXPERIMENT_PATH)
    assert not result.ok
    assert any("schema validation" in issue for issue in result.issues)


def test_cmd_validate_flags_resolution_mismatch_within_a_session(
    tmp_path: Path, experiment: ExperimentDefinition
) -> None:
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    common = {"fps": 2.0, "n_frames": 50, "width": 32, "height": 24, "camera_setup_id": "S1"}
    _write_run(runs_root, "001-a", experiment, **common)
    _write_run(runs_root, "002-b", experiment, **common)
    odd = {**common, "width": 64, "height": 48}
    _write_run(runs_root, "003-odd", experiment, **odd)

    exit_code = record.cmd_validate(runs_root, EXPERIMENT_PATH)

    assert exit_code == 1
    manifest_path = runs_root / "manifest.csv"
    assert manifest_path.exists()
    with open(manifest_path, newline="", encoding="utf-8") as f:
        rows = {row["run_id"]: row for row in csv.DictReader(f)}
    assert set(rows) == {"001-a", "002-b", "003-odd"}
    assert rows["003-odd"]["width"] == "64"
    assert len(rows["001-a"]["video_sha256"]) == 64  # sha256 hex digest


def test_expected_deviations_stale_is_none_before_engine_reference_lands(
    experiment: ExperimentDefinition,
) -> None:
    # engine/reference.py is P2.2's; this branch hasn't merged it yet, so
    # the staleness check must skip, not fail (AGENTS.md rule 2).
    script = RunScript(
        run_id="001-correct",
        experiment_id=experiment.experiment_id,
        script_type="correct",
        split="train",
        fps=15.0,
        camera_setup_id="S1",
        operator="O1",
        performed_steps=["red_out"],
        expected_deviations=[],
    )
    assert record._expected_deviations_stale(script, EXPERIMENT_PATH) is None
