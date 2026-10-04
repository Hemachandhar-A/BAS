"""S-I1 review fix-forward: pins for the recorder fps, the run_started video time and the order of
the graceful stop. Fakes and temporary files only."""

from __future__ import annotations

import base64
import json
import threading
import time
from pathlib import Path

import cv2
import pytest

from contracts import ExperimentDefinition, PerceptionConfig, RuntimeConfig
from harness import live
from harness.fakes import FakeFrameSource, FakePerception, make_blank_frame
from outputs.tts import FakeSpeaker

FIXTURE = Path(__file__).resolve().parents[3] / "fixtures" / "experiment_4step.json"
AUTH = {"Authorization": "Basic " + base64.b64encode(b"u:p").decode()}


class _NoServer:
    def serve_forever(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def server_close(self) -> None:
        pass


def _make(tmp_path: Path, frames: int, fps: float | None, target_fps: float = 200.0):
    runtime = RuntimeConfig(
        target_fps=target_fps, log_dir=str(tmp_path / "logs"), video_dir=str(tmp_path / "video")
    )
    settings = live.Settings(ExperimentDefinition.from_json(FIXTURE), runtime, PerceptionConfig())
    source = FakeFrameSource(
        [make_blank_frame(i, i / 20, height=16, width=16) for i in range(frames)],
        fps=fps,
        read_delay=0.005,
    )
    system = live.LiveSystem(
        settings=settings,
        perception=FakePerception([]),
        source=source,
        speaker=FakeSpeaker(),
        username="u",
        password="p",
        server_factory=lambda app, cfg: _NoServer(),
    )
    return system, source


# --- F1.1 -------------------------------------------------------------------------------------


def test_the_recording_is_written_at_the_source_fps_not_target_fps(tmp_path: Path) -> None:
    """target_fps is 200 here and the source is 20 fps: a recorder built with the loop's default
    factory would write a 200 fps file (it plays back 10x too fast, or 3x slow in the real
    setup of 30 fps capture and 10 fps target)."""
    system, _ = _make(tmp_path, frames=40, fps=20.0, target_fps=200.0)
    system.start(auto_start=True)
    try:
        system.wait(exit_when_done=True, poll_s=0.02)
    finally:
        system.shutdown()
    video = next((tmp_path / "video").glob("*.avi"))
    cap = cv2.VideoCapture(str(video))
    try:
        assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(20.0)
    finally:
        cap.release()


def test_recorder_factory_falls_back_to_the_given_fps_when_the_source_has_none(
    tmp_path: Path,
) -> None:
    recorder = live.recorder_factory_for(FakeFrameSource([], fps=None))(tmp_path / "a.avi", 7.0)
    assert recorder._fps == 7.0
    recorder = live.recorder_factory_for(FakeFrameSource([], fps=25.0))(tmp_path / "b.avi", 7.0)
    assert recorder._fps == 25.0


# --- F1.2 -------------------------------------------------------------------------------------


def test_run_started_from_the_dashboard_carries_the_newest_frame_t(tmp_path: Path) -> None:
    """POST /api/run/start through the app: t_video is the newest frame's own t. The process
    uptime (create_app's default run clock) would be a very different number."""
    system, _ = _make(tmp_path, frames=1, fps=20.0)
    system.loop.frame_store.put(make_blank_frame(5, 7.25, height=16, width=16))
    client = system.app.test_client()
    response = client.post("/api/run/start", headers=AUTH)
    assert response.status_code == 200
    run_id = response.get_json()["run_id"]
    system.loop.reset_run(system.run_clock())
    lines = [
        json.loads(x)
        for x in (tmp_path / "logs" / f"{run_id}.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert lines[0]["event_type"] == "run_started" and lines[0]["t_video"] == 7.25
    assert lines[-1]["event_type"] == "run_completed" and lines[-1]["t_video"] == 7.25
    assert time.monotonic() > 7.25 + 1.0  # the uptime would have shown up as a different number


# --- F1.3 -------------------------------------------------------------------------------------


def test_shutdown_stops_the_loop_before_it_finishes_the_run(tmp_path: Path) -> None:
    system, _ = _make(tmp_path, frames=10_000, fps=20.0)
    order: list[str] = []
    real_stop, real_reset = system.loop.stop, system.loop.reset_run

    def stop() -> None:
        order.append("stop")
        real_stop()

    def reset_run(t: float) -> str:
        order.append("reset_run")
        return real_reset(t)

    system.loop.stop = stop  # type: ignore[method-assign]
    system.loop.reset_run = reset_run  # type: ignore[method-assign]
    system.start(auto_start=True)
    time.sleep(0.2)
    system.shutdown()
    assert order == ["stop", "reset_run"]
    log = next((tmp_path / "logs").glob("*.jsonl"))
    kinds = [json.loads(x)["event_type"] for x in log.read_text(encoding="utf-8").splitlines()]
    assert kinds[0] == "run_started" and kinds[-1] == "run_completed"
    assert kinds.count("run_completed") == 1


def test_no_inference_runs_after_the_run_is_completed_at_shutdown(tmp_path: Path) -> None:
    """With the loop stopped first, nothing can process a frame once run_completed is written."""
    system, _ = _make(tmp_path, frames=10_000, fps=20.0)
    processed_after: list[int] = []
    completed = threading.Event()
    real_process = system.router.process_perception_frame

    def spy(frame) -> None:  # noqa: ANN001
        if completed.is_set():
            processed_after.append(frame.frame_id)
        real_process(frame)

    system.router.process_perception_frame = spy  # type: ignore[method-assign]
    real_reset = system.loop.reset_run

    def reset_run(t: float) -> str:
        out = real_reset(t)
        completed.set()
        return out

    system.loop.reset_run = reset_run  # type: ignore[method-assign]
    system.start(auto_start=True)
    time.sleep(0.3)
    system.shutdown()
    time.sleep(0.2)
    assert processed_after == []
