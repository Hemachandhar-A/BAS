"""The live wiring with the REAL pipeline on a real val clip (x015, the first 100 processed
frames at target_fps 10) gives the same events as the cached replay of the same frames.

Capture is paced slower than the inference throttle, so latest-frame-wins drops nothing and the
live loop sees exactly the frames the cache holds (every third frame of the 30 fps recording).
Skipped when the weights, the clip or the cache are absent."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from contracts import Frame
from harness import live
from harness.replay import replay_from_cache
from harness.settings import expected_model_stamp
from outputs.tts import FakeSpeaker

ROOT = Path(__file__).resolve().parents[2]
RUN = "x015"
VIDEO = ROOT / "runs" / RUN / "video.mp4"
CACHE = ROOT / "data" / "cache" / RUN / "perception.jsonl"
MANIFEST = ROOT / "weights" / "MANIFEST.json"
N_FRAMES = 100
EVERY = 3  # 30 fps recording / target_fps 10
FRAME_PERIOD_S = 0.13  # slower than the 0.1 s throttle, with the pipeline's ~50 ms inside it


class _SlowDecimated:
    """Every third frame of the recording, at most N of them, one per FRAME_PERIOD_S."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self._count = 0
        self.fps = inner.fps
        self.exhausted = False

    def read(self) -> Frame | None:
        while self._count < N_FRAMES:
            frame = self._inner.read()
            if frame is None:
                break
            if frame.frame_id % EVERY == 0:
                self._count += 1
                time.sleep(FRAME_PERIOD_S)
                return frame
        self.exhausted = True
        return None

    def close(self) -> None:
        self._inner.close()


@pytest.mark.slow
@pytest.mark.F13
def test_live_loop_matches_the_cached_replay(tmp_path: Path) -> None:
    if not (VIDEO.is_file() and CACHE.is_file() and MANIFEST.is_file()):
        pytest.skip("weights, clip or cache not present")
    try:
        from perception.camera import open_source
        from perception.pipeline import load_pipeline

        perception = load_pipeline(MANIFEST)
    except Exception as exc:  # missing weights or an optional dependency
        pytest.skip(f"real pipeline unavailable: {exc}")

    settings = live.load_settings(
        ROOT / "config" / "runtime.yaml",
        ROOT / "config" / "perception.yaml",
        ROOT / "config" / "experiment.json",
    )
    runtime = settings.runtime.model_copy(
        update={"log_dir": str(tmp_path / "logs"), "video_dir": str(tmp_path / "video")}
    )
    settings = live.Settings(settings.experiment, runtime, settings.perception)
    assert runtime.target_fps == 10

    cached = replay_from_cache(
        settings.experiment,
        RUN,
        cache_dir=ROOT / "data" / "cache",
        perception_config=settings.perception,
        runtime_config=runtime,
        log_dir=tmp_path / "cached_logs",
        expected_stamp=expected_model_stamp(MANIFEST),
        expected_fps=runtime.target_fps,
        max_frames=N_FRAMES,
    )
    want = [(e.kind, e.step_id, e.t, e.confidence_tag) for e in cached.engine_events]
    assert len(want) == 5, want

    speaker = FakeSpeaker()
    system = live.LiveSystem(
        settings=settings,
        perception=perception,
        source=_SlowDecimated(open_source(str(VIDEO))),
        speaker=speaker,
        username="u",
        password="p",
        server_factory=lambda app, cfg: _NoServer(),
    )
    system.start(auto_start=True)
    try:
        assert system.wait(exit_when_done=True, poll_s=0.05) == "source finished"
    finally:
        system.shutdown()

    log = next((tmp_path / "logs").glob("*.jsonl"))
    entries = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    got = [
        (e["event_type"], e["step_id"], e["t_video"], e["confidence_tag"])
        for e in entries
        if e["event_type"] in ("step_confirmed", "deviation_detected")
    ]
    period = 1.0 / runtime.target_fps
    assert [(k, s, tag) for k, s, _, tag in got] == [(k, s, tag) for k, s, _, tag in want]
    for (_, _, t_live, _), (_, _, t_cached, _) in zip(got, want, strict=True):
        assert abs(t_live - t_cached) <= period
    assert [seq["seq"] for seq in entries] == list(range(len(entries)))
    assert speaker.calls  # the first step's cue and a confirmation per step were spoken


class _NoServer:
    def serve_forever(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def server_close(self) -> None:
        pass
