"""perception/record.py -- Stage 0 (record) and Stage 1 (validate) of the
dataset pipeline (F14; essential-features.md section 14 stages 0-1). The
crew-facing workflow is DATA_COLLECTION.md sections 5 and 7.

    python -m perception.record --plan runs/run_plan.csv   # Stage 0: record
    python -m perception.record --validate runs/            # Stage 1: validate

Stage 0 leaves ``RunScript.expected_deviations`` empty; Stage 1 fills the
staleness check for it via ``engine.reference`` only once that module
exists (AGENTS.md rule 2 -- P1 never guesses P2's shape, and doesn't wait
on it either).
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from pydantic import ValidationError

from contracts import CAPTURE_FPS, ExperimentDefinition, RunScript, SourceError, sha256_of_file
from perception.camera import open_source

logger = logging.getLogger(__name__)

DEFAULT_EXPERIMENT_PATH = Path("config/experiment.json")

_PLAN_FIELDS = [
    "run_id",
    "split",
    "script_type",
    "variant",
    "camera_setup_id",
    "operator",
    "planned_performed_steps",
    "intent",
    "condition",
    "recorded",
    "recorded_by",
    "notes",
]
_MANIFEST_FIELDS = [
    "run_id",
    "split",
    "script_type",
    "fps",
    "frames",
    "duration_s",
    "width",
    "height",
    "operator",
    "camera_setup_id",
    "video_sha256",
]

# Stage 1 acceptance bands: disclosed judgment calls, local to this file --
# none of these cross the P1/P2 boundary, so they don't belong in
# contracts.py (AGENTS.md rule 3).
_FPS_TOLERANCE_FRAC = 0.05
_MIN_DURATION_S = 20.0
_MAX_DURATION_S = 150.0
_LUMINANCE_SAMPLE_FRAMES = 30
_MIN_MEAN_LUMINANCE = 15.0
_MAX_MEAN_LUMINANCE = 240.0
_FROZEN_MEAN_ABS_DIFF = 0.5  # 0-255 scale, across sampled frame pairs


# ---------------------------------------------------------------------------
# runs/run_plan.csv
# ---------------------------------------------------------------------------


class PlanRow:
    """One row of ``runs/run_plan.csv`` (DATA_COLLECTION.md section 4)."""

    def __init__(self, raw: dict[str, str]) -> None:
        self.run_id = raw["run_id"]
        self.split = raw["split"]
        self.script_type = raw["script_type"]
        self.variant = raw.get("variant", "")
        self.camera_setup_id = raw["camera_setup_id"]
        self.operator = raw["operator"]
        self.planned_performed_steps = [
            s for s in raw.get("planned_performed_steps", "").split("|") if s
        ]
        self.intent = raw.get("intent", "")
        self.condition = raw.get("condition", "")
        self.recorded = raw.get("recorded", "")
        self.recorded_by = raw.get("recorded_by", "")
        self.notes = raw.get("notes", "")
        self._raw = dict(raw)

    def to_raw(self) -> dict[str, str]:
        row = dict(self._raw)
        row["recorded"] = self.recorded
        row["recorded_by"] = self.recorded_by
        row["notes"] = self.notes
        return row


def read_plan(path: Path) -> list[PlanRow]:
    with open(path, newline="", encoding="utf-8") as f:
        return [PlanRow(row) for row in csv.DictReader(f)]


def write_plan(path: Path, rows: list[PlanRow]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_PLAN_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_raw())


def pending_rows(rows: list[PlanRow]) -> list[PlanRow]:
    """Rows the crew hasn't ticked off yet (DATA_COLLECTION.md section 7.3)."""
    return [r for r in rows if not r.recorded.strip()]


# ---------------------------------------------------------------------------
# Stage 0 -- record
# ---------------------------------------------------------------------------


def _run_dir(runs_root: Path, run_id: str) -> Path:
    return runs_root / run_id


def _video_path(runs_root: Path, run_id: str) -> Path:
    return _run_dir(runs_root, run_id) / "video.mp4"


class _FrameSourceLike(Protocol):
    exhausted: bool

    def read(self): ...


def capture_run(
    source: _FrameSourceLike,
    video_path: Path,
    writer_fps: float,
    should_stop,
) -> int:
    """Reads frames from ``source`` until it is exhausted or
    ``should_stop(frame)`` returns ``True``, writing each captured frame to
    ``video_path`` (mp4v). Returns the number of frames written. The
    interactive recorder passes a keypress-driven ``should_stop``; tests
    pass a frame-count-driven one against a fixture/fake source, so this
    core loop needs neither a camera nor a GUI to be exercised."""
    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer: cv2.VideoWriter | None = None
    count = 0
    try:
        while True:
            frame = source.read()
            if frame is None:
                if source.exhausted:
                    break
                continue  # transient camera glitch; keep trying
            if writer is None:
                height, width = frame.image.shape[:2]
                writer = cv2.VideoWriter(
                    str(video_path), cv2.VideoWriter.fourcc(*"mp4v"), writer_fps, (width, height)
                )
                if not writer.isOpened():
                    raise SourceError(f"could not open video writer for {video_path}")
            writer.write(frame.image)
            count += 1
            if should_stop(frame):
                break
    finally:
        if writer is not None:
            writer.release()
    return count


def build_run_script(
    row: PlanRow,
    experiment_id: str,
    measured_fps: float,
    performed_steps: list[str],
) -> RunScript:
    """``expected_deviations`` stays empty here; ``engine.reference``
    (F5/P2.2) derives it once it exists (essential-features.md section 14,
    Stage 0)."""
    return RunScript(
        run_id=row.run_id,
        experiment_id=experiment_id,
        script_type=row.script_type,
        split=row.split,
        fps=measured_fps,
        camera_setup_id=row.camera_setup_id,
        operator=row.operator,
        performed_steps=performed_steps,
        expected_deviations=[],
    )


def write_script_json(runs_root: Path, script: RunScript) -> Path:
    run_dir = _run_dir(runs_root, script.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "script.json"
    path.write_text(script.model_dump_json(indent=2), encoding="utf-8")
    return path


class _InteractiveStop:
    """Keypress-driven ``should_stop`` for ``capture_run``: shows the live
    frame, S stops and keeps the take, X stops and discards it
    (DATA_COLLECTION.md section 5, steps 4-5)."""

    def __init__(self, window_name: str) -> None:
        self.window_name = window_name
        self.discarded = False

    def __call__(self, frame) -> bool:
        cv2.imshow(self.window_name, frame.image)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("x"), ord("X")):
            self.discarded = True
            return True
        return key in (ord("s"), ord("S"))


def _show_live_preview(source, window_name: str) -> None:
    print("Live preview: check framing, then press any key to start the countdown.")
    while True:
        frame = source.read()
        if frame is None:
            if source.exhausted:
                raise SourceError("source exhausted during the live preview")
            continue
        cv2.imshow(window_name, frame.image)
        if cv2.waitKey(1) != -1:
            return


def _countdown(seconds: int = 3) -> None:
    for n in range(seconds, 0, -1):
        print(f"Recording starts in {n}...")
        time.sleep(1)


def _confirm_performed_steps(planned: list[str]) -> list[str]:
    shown = " ".join(planned) if planned else "(none -- idle run)"
    print(f"Planned steps: {shown}")
    answer = input(
        "Was it performed exactly as planned? [Enter = yes, or type the actual "
        "step ids separated by spaces]: "
    ).strip()
    return list(planned) if not answer else answer.split()


def cmd_record(plan_path: Path, source_arg: str, experiment_path: Path) -> int:
    experiment = ExperimentDefinition.from_json(experiment_path)
    runs_root = plan_path.parent
    rows = read_plan(plan_path)
    todo = pending_rows(rows)
    if not todo:
        print("No pending runs in the plan.")
        return 0

    for row in todo:
        print(f"\nNext run: {row.run_id}  operator={row.operator}  intent={row.intent}")
        input("Objects on their tape marks. Press Enter for the live preview...")
        source = open_source(source_arg)
        window_name = f"Recording {row.run_id} (S=stop, X=discard)"
        try:
            _show_live_preview(source, window_name)
            _countdown()
            video_path = _video_path(runs_root, row.run_id)
            stopper = _InteractiveStop(window_name)
            capture_run(source, video_path, source.fps or float(CAPTURE_FPS), stopper)
            cv2.destroyWindow(window_name)

            if stopper.discarded:
                video_path.unlink(missing_ok=True)
                print(f"Discarded {row.run_id}.")
                continue

            performed_steps = _confirm_performed_steps(row.planned_performed_steps)
            script = build_run_script(
                row, experiment.experiment_id, source.fps or float(CAPTURE_FPS), performed_steps
            )
            write_script_json(runs_root, script)
            # A plain tick, not a timestamp (AGENTS.md rule 8: no clock reads
            # in perception/ outside camera.py; ISSUES.md 2026-09-29 review).
            row.recorded = "yes"
            row.recorded_by = row.operator
            write_plan(plan_path, rows)
            print(f"Wrote {video_path} and script.json for {row.run_id}.")
        finally:
            source.close()
    return 0


# ---------------------------------------------------------------------------
# Stage 1 -- validate
# ---------------------------------------------------------------------------


def probe_video(path: Path) -> dict:
    """Opens ``path`` and reports what Stage 1 needs to check: measured
    fps, frame count, resolution, and (from up to
    ``_LUMINANCE_SAMPLE_FRAMES`` sampled frames) mean luminance and
    frame-to-frame difference, used to catch a dark/blown-out or frozen
    recording."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SourceError(f"cannot open {path}")
    try:
        reported_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        luminances: list[float] = []
        diffs: list[float] = []
        prev_gray: np.ndarray | None = None
        n_frames = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            n_frames += 1
            if len(luminances) < _LUMINANCE_SAMPLE_FRAMES:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
                luminances.append(float(gray.mean()))
                if prev_gray is not None:
                    diffs.append(float(np.mean(np.abs(gray - prev_gray))))
                prev_gray = gray
    finally:
        cap.release()

    fps_ok = reported_fps and math.isfinite(reported_fps) and reported_fps > 0
    fps = float(reported_fps) if fps_ok else None
    return {
        "fps": fps,
        "frames": n_frames,
        "duration_s": (n_frames / fps) if fps else None,
        "width": width,
        "height": height,
        "mean_luminance": statistics.mean(luminances) if luminances else None,
        "mean_frame_diff": statistics.mean(diffs) if diffs else None,
    }


def _expected_deviations_stale(script: RunScript, experiment_path: Path) -> str | None:
    try:
        from engine.reference import derive_expected_deviations  # type: ignore[import-not-found]
    except ImportError:
        return None  # engine/reference.py hasn't landed yet (P2.2)
    experiment = ExperimentDefinition.from_json(experiment_path)
    expected = derive_expected_deviations(experiment, script.performed_steps)
    if expected != script.expected_deviations:
        return "expected_deviations is stale relative to engine.reference"
    return None


class ValidationResult:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.issues: list[str] = []
        self.script: RunScript | None = None
        self.probe: dict | None = None
        self.video_path: Path | None = None

    @property
    def ok(self) -> bool:
        return not self.issues


def validate_run(
    run_dir: Path,
    experiment: ExperimentDefinition,
    experiment_path: Path,
    min_duration_s: float = _MIN_DURATION_S,
) -> ValidationResult:
    result = ValidationResult(run_dir.name)
    script_path = run_dir / "script.json"
    if not script_path.exists():
        result.issues.append("missing script.json")
        return result

    try:
        script = RunScript.model_validate_json(script_path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        result.issues.append(f"script.json failed schema validation: {exc}")
        return result
    result.script = script

    unknown_steps = sorted(set(script.performed_steps) - set(experiment.step_ids))
    if unknown_steps:
        result.issues.append(f"performed_steps references unknown step ids: {unknown_steps}")

    video_path = run_dir / "video.mp4"
    result.video_path = video_path
    if not video_path.exists():
        result.issues.append("missing video.mp4")
        return result

    try:
        probe = probe_video(video_path)
    except SourceError as exc:
        result.issues.append(f"video does not open: {exc}")
        return result
    result.probe = probe

    if probe["fps"] is None:
        result.issues.append("could not measure the video's fps")
    elif abs(probe["fps"] - script.fps) > _FPS_TOLERANCE_FRAC * script.fps:
        result.issues.append(
            f"measured fps {probe['fps']:.2f} differs from declared {script.fps:.2f} by more "
            f"than {_FPS_TOLERANCE_FRAC:.0%}"
        )

    duration_s = probe["duration_s"]
    if duration_s is not None and not (min_duration_s <= duration_s <= _MAX_DURATION_S):
        result.issues.append(
            f"duration {duration_s:.1f}s outside the "
            f"{min_duration_s:.0f}-{_MAX_DURATION_S:.0f}s band"
        )

    mean_luminance = probe["mean_luminance"]
    if mean_luminance is not None and not (
        _MIN_MEAN_LUMINANCE <= mean_luminance <= _MAX_MEAN_LUMINANCE
    ):
        result.issues.append(f"mean luminance {mean_luminance:.1f} outside the sane band")

    mean_diff = probe["mean_frame_diff"]
    if mean_diff is not None and mean_diff < _FROZEN_MEAN_ABS_DIFF:
        result.issues.append("video looks frozen (near-zero frame-to-frame difference)")

    if not unknown_steps:
        stale = _expected_deviations_stale(script, experiment_path)
        if stale is not None:
            result.issues.append(stale)

    return result


def _flag_resolution_mismatches(results: list[ValidationResult]) -> None:
    """Runs sharing a ``camera_setup_id`` are one recording session and
    must share a resolution (essential-features.md section 14, Stage 1)."""
    by_setup: dict[str, list[tuple[str, tuple[int, int]]]] = {}
    for r in results:
        if r.script is not None and r.probe is not None:
            res = (r.probe["width"], r.probe["height"])
            by_setup.setdefault(r.script.camera_setup_id, []).append((r.run_id, res))

    by_run_id = {r.run_id: r for r in results}
    for setup_id, entries in by_setup.items():
        mode = Counter(res for _, res in entries).most_common(1)[0][0]
        for run_id, res in entries:
            if res != mode:
                by_run_id[run_id].issues.append(
                    f"resolution {res} does not match camera_setup_id {setup_id!r}'s {mode}"
                )


def write_manifest(path: Path, results: list[ValidationResult]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_MANIFEST_FIELDS)
        writer.writeheader()
        for r in results:
            if r.script is None or r.probe is None or r.video_path is None:
                continue
            writer.writerow(
                {
                    "run_id": r.run_id,
                    "split": r.script.split,
                    "script_type": r.script.script_type,
                    "fps": r.probe["fps"] if r.probe["fps"] is not None else r.script.fps,
                    "frames": r.probe["frames"],
                    "duration_s": r.probe["duration_s"],
                    "width": r.probe["width"],
                    "height": r.probe["height"],
                    "operator": r.script.operator,
                    "camera_setup_id": r.script.camera_setup_id,
                    "video_sha256": sha256_of_file(r.video_path),
                }
            )


def cmd_validate(
    runs_root: Path, experiment_path: Path, min_duration_s: float = _MIN_DURATION_S
) -> int:
    experiment = ExperimentDefinition.from_json(experiment_path)
    run_dirs = sorted(p for p in runs_root.iterdir() if p.is_dir())
    results = [
        validate_run(d, experiment, experiment_path, min_duration_s=min_duration_s)
        for d in run_dirs
    ]
    _flag_resolution_mismatches(results)

    exit_code = 0
    for r in results:
        if r.issues:
            exit_code = 1
            for issue in r.issues:
                logger.warning("%s: %s", r.run_id, issue)
        else:
            logger.info("%s: ok", r.run_id)

    write_manifest(runs_root / "manifest.csv", results)
    return exit_code


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--plan", type=Path, help="runs/run_plan.csv -- runs the interactive Stage 0 recorder"
    )
    group.add_argument(
        "--validate", type=Path, help="the runs/ directory -- runs Stage 1 validation"
    )
    parser.add_argument(
        "--source", default="0", help="camera index / file / URL for --plan (default: 0)"
    )
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT_PATH)
    parser.add_argument(
        "--min-duration-s",
        type=float,
        default=_MIN_DURATION_S,
        help=f"Stage 1 minimum clip duration in seconds (default: {_MIN_DURATION_S:.0f}); "
        "used with --validate",
    )
    args = parser.parse_args(argv)

    if args.validate is not None:
        return cmd_validate(args.validate, args.experiment, min_duration_s=args.min_duration_s)
    return cmd_record(args.plan, args.source, args.experiment)


if __name__ == "__main__":
    raise SystemExit(main())
