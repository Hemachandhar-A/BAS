"""A two-clip playlist through the live loop with the REAL pipeline: two val clips (x015 and x024,
the first 300 frames of each, written to a temporary folder as mp4) are played one after another;
each clip's events equal the sequential replay of the same clip, the run ids differ, and clip 2
starts clean (it gives the same events as the cache although clip 1 ran before it).

Capture is decimated to every third frame and paced slower than the inference throttle, as in
test_live_wiring.py, so latest-frame-wins drops nothing and the loop sees exactly the frames the
cache holds. The clips are re-encoded to mp4 (lossy), so the comparison is on event kinds, steps
and tags, with times within one processing period. Skipped when weights, clips or caches are
absent."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import cv2
import pytest

from contracts import Frame
from harness import live
from harness.replay import replay_from_video
from outputs.tts import FakeSpeaker

ROOT = Path(__file__).resolve().parents[2]
RUNS = ("x015", "x024")
MANIFEST = ROOT / "weights" / "MANIFEST.json"
RAW_FRAMES = 300  # 10 s of a 30 fps recording; 100 processed frames at target_fps 10
EVERY = 3
FRAME_PERIOD_S = 0.13


class _SlowDecimated:
    """Every third frame, one per FRAME_PERIOD_S (the gate of PacedSource still applies)."""

    def __init__(self, inner: live.PacedSource) -> None:
        self._inner = inner
        self.exhausted = False

    @property
    def fps(self) -> float | None:
        return self._inner.fps

    def read(self) -> Frame | None:
        while True:
            frame = self._inner.read()
            if frame is None:
                self.exhausted = self._inner.exhausted
                return None
            if frame.frame_id % EVERY == 0 or not self._inner._gate.released:
                time.sleep(FRAME_PERIOD_S if self._inner._gate.released else 0.05)
                return frame

    def close(self) -> None:
        self._inner.close()


def _write_clip(src: Path, dst: Path) -> None:
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)  # keep the re-encode as close as possible
    try:
        for _ in range(RAW_FRAMES):
            ok, image = cap.read()
            if not ok:
                break
            writer.write(image)
    finally:
        writer.release()
        cap.release()


def _disable_landmarker_del() -> None:
    from mediapipe.tasks.python.vision import hand_landmarker

    hand_landmarker.HandLandmarker.__del__ = lambda self: None  # type: ignore[method-assign]


@pytest.mark.slow
@pytest.mark.F13
def test_a_two_clip_playlist_matches_the_sequential_replay_of_each_clip(tmp_path: Path) -> None:
    caches = [ROOT / "data" / "cache" / r / "perception.jsonl" for r in RUNS]
    videos = [ROOT / "runs" / r / "video.mp4" for r in RUNS]
    if not (MANIFEST.is_file() and all(p.is_file() for p in caches + videos)):
        pytest.skip("weights, clips or caches not present")
    try:
        from perception.pipeline import load_pipeline

        perception = load_pipeline(MANIFEST)
    except Exception as exc:  # missing weights or an optional dependency
        pytest.skip(f"real pipeline unavailable: {exc}")
    # reset() closes the hand landmarker and drops it; its __del__ then calls close() a second
    # time, which waits forever in MediaPipe's dispatcher when it runs at interpreter exit: the
    # test passes and the process never ends (ISSUES.md, S-I1b). Disable that __del__ here.
    _disable_landmarker_del()

    folder = tmp_path / "clips"
    folder.mkdir()
    for run, video in zip(RUNS, videos, strict=True):
        _write_clip(video, folder / f"{run}.mp4")
    clips = live.parse_playlist(folder, runs_dir=tmp_path / "runs")  # name order: x015, x024
    assert [c.stem for c in clips] == list(RUNS)

    settings = live.load_settings(
        ROOT / "config" / "runtime.yaml",
        ROOT / "config" / "perception.yaml",
        ROOT / "config" / "experiment.json",
    )
    runtime = settings.runtime.model_copy(
        update={"log_dir": str(tmp_path / "logs"), "video_dir": str(tmp_path / "video")}
    )
    settings = live.Settings(settings.experiment, runtime, settings.perception)
    period = 1.0 / runtime.target_fps

    # The reference is the sequential, thread-free replay of the very same clip files (the same
    # decimated frames) through the same pipeline: the clips are lossy re-encodes, so the cache of
    # the original video is not the same pixels (its x015 run gains a late start_pressed).
    wanted = {}
    for clip in clips:
        ref = replay_from_video(
            settings.experiment,
            clip,
            perception_config=settings.perception,
            runtime_config=runtime,
            log_dir=tmp_path / f"reference_{clip.stem}",
            pipeline_factory=lambda: perception,
        )
        assert abs(ref.frames_processed - RAW_FRAMES // EVERY) <= 2
        wanted[clip.stem] = [(e.kind, e.step_id, e.confidence_tag, e.t) for e in ref.engine_events]
    locked = live.LockedPerception(perception)
    opened: list[int] = []

    def on_new_clip(index: int, path: Path) -> None:
        if opened:
            locked.reset()
        opened.append(index)

    from perception.camera import open_source

    source = live.PlaylistSource(
        clips,
        opener=open_source,
        pacer=lambda inner, gate: _SlowDecimated(live.PacedSource(inner, gate=gate)),
        on_new_clip=on_new_clip,
        loop=False,
    )
    label = live.RunLabel(clips[0].stem)
    speaker = FakeSpeaker()

    class _NoServer:
        def serve_forever(self) -> None:
            pass

        def shutdown(self) -> None:
            pass

        def server_close(self) -> None:
            pass

    system = live.LiveSystem(
        settings=settings,
        perception=locked,
        source=source,
        speaker=speaker,
        username="u",
        password="p",
        server_factory=lambda app, cfg: _NoServer(),
        recorder_factory=live.NullRecorder,
        run_id_factory=label,
    )
    lines: list[str] = []
    controller = live.PlaylistController(
        system,
        source,
        label=label,
        pause_between=0.5,
        wait_for_dashboard=False,
        out=lines.append,
        settle_timeout_s=10.0,
    )
    system.start(auto_start=False)
    stop = threading.Event()
    thread = controller.start_thread(stop)
    try:
        end = time.monotonic() + 180
        while controller.reason is None and time.monotonic() < end:
            time.sleep(0.2)
    finally:
        stop.set()
        thread.join(timeout=10)
        system.shutdown()
    assert controller.reason == "playlist finished", lines

    logs = sorted(
        (
            [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]
            for p in (tmp_path / "logs").glob("*.jsonl")
        ),
        key=lambda log: log[0]["t_wall"],
    )
    assert len(logs) == 2
    assert logs[0][0]["run_id"] != logs[1][0]["run_id"]
    assert "x015" in logs[0][0]["run_id"] and "x024" in logs[1][0]["run_id"]
    for run, log in zip(RUNS, logs, strict=True):
        kinds = [e["event_type"] for e in log]
        assert kinds[0] == "run_started" and kinds[-1] == "run_completed"
        assert [e["seq"] for e in log] == list(range(len(log)))
        got = [
            (e["event_type"], e["step_id"], e["confidence_tag"], e["t_video"])
            for e in log
            if e["event_type"] in ("step_confirmed", "deviation_detected")
        ]
        want = [w for w in wanted[run] if w[0] in ("step_confirmed", "deviation_detected")]
        assert [g[:3] for g in got] == [w[:3] for w in want], (run, got, want)
        for g, w in zip(got, want, strict=True):
            assert abs(g[3] - w[3]) <= period, (run, g, w)
    assert wanted[RUNS[1]], "the second clip has events to compare"
