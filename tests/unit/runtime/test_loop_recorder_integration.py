"""runtime/loop.py -- RuntimeLoop.start_run/reset_run ties the F11
Recorder's lifecycle to the run lifecycle: opened (fresh file named by the
new run_id) when a run starts, closed when it resets -- exercised with a
real Recorder (not a fake) so the produced .avi file is genuinely
playable, matching F11's "one playable file per run in video_dir, named
by run_id"."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import pytest

from contracts import ExperimentDefinition, PerceptionConfig, RuntimeConfig
from engine.sequence import SequenceEngine
from harness.fakes import FakeFrameSource, FakePerception, make_blank_frame
from outputs.tts import FakeSpeaker
from runtime.loop import Router, RuntimeLoop
from state.tracker import StateTracker

FIXTURE_PATH = Path(__file__).resolve().parents[3] / "fixtures" / "experiment_4step.json"


@pytest.mark.F11
def test_start_run_opens_recorder_and_reset_run_produces_a_playable_file(tmp_path: Path) -> None:
    experiment = ExperimentDefinition.from_json(FIXTURE_PATH)
    runtime_config = RuntimeConfig(target_fps=1000.0)
    router = Router(
        experiment,
        runtime_config,
        StateTracker(experiment, PerceptionConfig()),
        SequenceEngine(experiment, runtime_config),
        FakeSpeaker(),
        log_dir=tmp_path,
        run_id_factory=lambda: "recorder-integration",
    )
    frames = [make_blank_frame(i, float(i), height=8, width=8) for i in range(6)]
    source = FakeFrameSource(list(frames), read_delay=0.02)
    loop = RuntimeLoop(source, FakePerception([]), router, runtime_config, video_dir=tmp_path)

    loop.start_threads()
    try:
        run_id = loop.start_run(t=0.0)
        video_path = tmp_path / f"{run_id}.avi"

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not loop.stopped:
            time.sleep(0.01)
    finally:
        loop.reset_run(t=1.0)  # closes the recorder
        loop.stop()

    assert run_id == "recorder-integration"
    assert video_path.exists()
    cap = cv2.VideoCapture(str(video_path))
    try:
        assert cap.isOpened()
        ok, frame = cap.read()
        assert ok
        assert frame.shape == (8, 8, 3)
    finally:
        cap.release()
