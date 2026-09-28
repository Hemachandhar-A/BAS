"""runtime/loop.py -- RuntimeLoop: the threaded capture/inference
orchestration (essential-features.md section 0). Latest-frame-wins is
inherently racy by design (intermediate frames are meant to be dropped
under load), so these tests pace the fake capture source slower than the
inference throttle to keep frame-order assertions deterministic instead
of asserting exact timing. Feed-loss detection is tested against the
watchdog methods directly (no threads) to avoid coupling to wall-clock
scheduling -- the threaded tests below already cover that the capture and
inference threads actually run and stop cleanly."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pytest

from contracts import ExperimentDefinition, Frame, PerceptionConfig, PerceptionFrame, RuntimeConfig
from engine.sequence import SequenceEngine
from harness.fakes import FakeFrameSource, FakePerception, make_blank_frame
from outputs.tts import FakeSpeaker
from runtime.loop import Router, RuntimeLoop
from state.tracker import StateTracker

FIXTURE_PATH = Path(__file__).resolve().parents[3] / "fixtures" / "experiment_4step.json"


@pytest.fixture()
def experiment() -> ExperimentDefinition:
    return ExperimentDefinition.from_json(FIXTURE_PATH)


def _router(experiment: ExperimentDefinition, tmp_path: Path, run_id: str = "loop-test") -> Router:
    runtime_config = RuntimeConfig(target_fps=1000.0)
    return Router(
        experiment,
        runtime_config,
        StateTracker(experiment, PerceptionConfig()),
        SequenceEngine(experiment, runtime_config),
        FakeSpeaker(),
        log_dir=tmp_path,
        run_id_factory=lambda: run_id,
    )


class _RecordingPerception:
    """Wraps ``FakePerception``, recording the ``frame_id`` of every call
    that is attempted (before any injected failure) so tests can assert on
    delivery order without depending on thread timing."""

    model_stamp = "fake:00000000|hand:00000000|pose:00000000"

    def __init__(self, raise_on: frozenset[int] = frozenset()) -> None:
        self._inner = FakePerception([])
        self._raise_on = raise_on
        self.attempted: list[int] = []

    def process(self, frame: Frame) -> PerceptionFrame:
        self.attempted.append(frame.frame_id)
        if frame.frame_id in self._raise_on:
            raise RuntimeError(f"boom on frame {frame.frame_id}")
        return self._inner.process(frame)

    def reset(self) -> None:
        self._inner.reset()


def _wait_until_stopped(loop: RuntimeLoop, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if loop.stopped:
            return
        time.sleep(0.01)
    raise AssertionError("RuntimeLoop did not stop within the timeout")


@pytest.mark.F13
def test_capture_and_inference_threads_process_frames_in_order_and_stop_on_exhaustion(
    experiment: ExperimentDefinition, tmp_path: Path
) -> None:
    frames = [make_blank_frame(i, float(i)) for i in range(5)]
    # Paced slower than the (very fast) inference throttle below so the
    # inference thread reliably keeps up instead of racing capture.
    source = FakeFrameSource(list(frames), read_delay=0.03)
    perception = _RecordingPerception()
    router = _router(experiment, tmp_path)
    loop = RuntimeLoop(source, perception, router, RuntimeConfig(target_fps=1000.0), tmp_path)

    loop.start_threads()
    _wait_until_stopped(loop)
    loop.stop()

    assert source.closed is True
    # Strictly increasing (never the same frame_id twice, per "skip if same
    # frame_id") and reaching the last frame proves nothing got stuck.
    assert perception.attempted == sorted(perception.attempted)
    assert len(perception.attempted) == len(set(perception.attempted))
    assert perception.attempted[-1] == 4


@pytest.mark.F13
def test_perception_exception_skips_that_frame_and_the_loop_keeps_going(
    experiment: ExperimentDefinition, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    frames = [make_blank_frame(i, float(i)) for i in range(5)]
    source = FakeFrameSource(list(frames), read_delay=0.03)
    perception = _RecordingPerception(raise_on=frozenset({2}))
    router = _router(experiment, tmp_path)
    loop = RuntimeLoop(source, perception, router, RuntimeConfig(target_fps=1000.0), tmp_path)

    with caplog.at_level(logging.WARNING, logger="runtime.loop"):
        loop.start_threads()
        _wait_until_stopped(loop)
        loop.stop()

    assert 2 in perception.attempted  # the failing frame was still attempted
    assert perception.attempted[-1] == 4  # and the loop carried on past it
    assert any("frame 2" in record.getMessage() for record in caplog.records)


@pytest.mark.F13
def test_feed_watchdog_logs_lost_then_restored(
    experiment: ExperimentDefinition, tmp_path: Path
) -> None:
    router = _router(experiment, tmp_path, run_id="feedloss")
    router.start(t=0.0)
    loop = RuntimeLoop(
        source=FakeFrameSource([]),
        perception=FakePerception([]),
        router=router,
        runtime_config=RuntimeConfig(feed_timeout_s=1.0),
        video_dir=tmp_path,
        clock=lambda: 0.0,  # unused directly; the watchdog methods take `now` explicitly
    )
    loop._last_captured_at = 0.0

    loop._check_feed_timeout(0.5)  # under the timeout
    assert loop.feed_ok is True

    loop._check_feed_timeout(1.5)  # >= feed_timeout_s
    assert loop.feed_ok is False

    loop._on_frame_captured(2.0, frame_t=9.0)  # a frame arrives again
    assert loop.feed_ok is True

    lines = [
        json.loads(line)
        for line in (tmp_path / "feedloss.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [entry["event_type"] for entry in lines] == ["run_started", "feed_lost", "feed_restored"]
    assert lines[2]["t_video"] == 9.0
