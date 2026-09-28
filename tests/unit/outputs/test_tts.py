"""outputs/tts.py tests (P2.3; IMPLEMENTATION_PLAN.md Part 10, 5.5; F7).

Every test but the one marked integration below uses ``null_engine_factory``
(no audio device) or ``FakeSpeaker`` (no process at all), per
essential-features.md section 7.6: CI has no audio device. A fixed
``clock`` callable makes the cooldown test deterministic.
"""

from __future__ import annotations

import time

import pytest

from outputs.tts import FakeSpeaker, TTSWorker, null_engine_factory


def _raising_engine_factory():
    """Module-level (picklable under spawn) stand-in for a broken audio
    backend -- pyttsx3.init() raising because no device is present."""
    raise RuntimeError("no audio device")


class _FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self._t = start

    def __call__(self) -> float:
        return self._t

    def advance(self, dt: float) -> None:
        self._t += dt


@pytest.fixture
def worker():
    w = TTSWorker(cooldown_s=5.0, engine_factory=null_engine_factory)
    w.wait_ready(timeout=10.0)
    yield w
    w.close()


# ---------------------------------------------------------------------------
# FakeSpeaker
# ---------------------------------------------------------------------------


@pytest.mark.F7
def test_fake_speaker_records_calls_in_order() -> None:
    speaker = FakeSpeaker()
    speaker.say("Stow the red sample", "info")
    speaker.say("Step skipped: Stow red sample", "alert")
    speaker.say("Experiment complete", "info")
    assert speaker.calls == [
        ("Stow the red sample", "info"),
        ("Step skipped: Stow red sample", "alert"),
        ("Experiment complete", "info"),
    ]


@pytest.mark.F7
def test_fake_speaker_close_never_raises() -> None:
    speaker = FakeSpeaker()
    speaker.close()
    assert speaker.closed


# ---------------------------------------------------------------------------
# TTSWorker -- say() never blocks / never raises
# ---------------------------------------------------------------------------


@pytest.mark.F7
def test_say_returns_quickly_for_info(worker: TTSWorker) -> None:
    start = time.perf_counter()
    worker.say("Stow the red sample", "info")
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0


@pytest.mark.F7
def test_say_never_raises_after_close() -> None:
    worker = TTSWorker(engine_factory=null_engine_factory)
    worker.wait_ready(timeout=10.0)
    worker.close()
    worker.say("Should be dropped silently", "alert")
    worker.say("Should be dropped silently", "info")


@pytest.mark.F7
def test_say_never_raises_if_restart_fails(
    worker: TTSWorker, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*args, **kwargs):
        raise RuntimeError("cannot spawn process")

    monkeypatch.setattr(worker._ctx, "Process", _boom)
    worker.say("alert text", "alert")  # must not raise despite the failed restart


# ---------------------------------------------------------------------------
# TTSWorker -- alert pre-emption and cooldown
# ---------------------------------------------------------------------------


@pytest.mark.F7
def test_alert_restarts_the_worker_process(worker: TTSWorker) -> None:
    process_before = worker._process
    worker.say("Step skipped: Stow red sample", "alert")
    process_after = worker._process
    assert process_before is not process_after
    assert process_after.is_alive()


@pytest.mark.F7
def test_info_does_not_restart_the_worker_process(worker: TTSWorker) -> None:
    process_before = worker._process
    worker.say("Stow the red sample", "info")
    process_after = worker._process
    assert process_before is process_after


@pytest.mark.F7
def test_identical_alert_within_cooldown_is_dropped() -> None:
    clock = _FakeClock()
    worker = TTSWorker(cooldown_s=5.0, engine_factory=null_engine_factory, clock=clock)
    worker.wait_ready(timeout=10.0)
    try:
        worker.say("Out of order: Stow red sample", "alert")
        first_process = worker._process
        clock.advance(1.0)
        worker.say("Out of order: Stow red sample", "alert")
        assert worker._process is first_process  # no restart: cooldown suppressed it
    finally:
        worker.close()


@pytest.mark.F7
def test_identical_alert_after_cooldown_speaks_again() -> None:
    clock = _FakeClock()
    worker = TTSWorker(cooldown_s=5.0, engine_factory=null_engine_factory, clock=clock)
    worker.wait_ready(timeout=10.0)
    try:
        worker.say("Out of order: Stow red sample", "alert")
        first_process = worker._process
        clock.advance(5.1)
        worker.say("Out of order: Stow red sample", "alert")
        assert worker._process is not first_process  # cooldown elapsed: restarts again
    finally:
        worker.close()


@pytest.mark.F7
def test_different_alert_text_is_not_suppressed_by_cooldown() -> None:
    clock = _FakeClock()
    worker = TTSWorker(cooldown_s=5.0, engine_factory=null_engine_factory, clock=clock)
    worker.wait_ready(timeout=10.0)
    try:
        worker.say("Step skipped: Stow red sample", "alert")
        first_process = worker._process
        clock.advance(0.1)
        worker.say("Repeated: Stow red sample", "alert")
        assert worker._process is not first_process
    finally:
        worker.close()


# ---------------------------------------------------------------------------
# TTSWorker -- degrade quietly with no audio device
# ---------------------------------------------------------------------------


@pytest.mark.F7
def test_degrades_quietly_when_engine_init_fails() -> None:
    worker = TTSWorker(engine_factory=_raising_engine_factory)
    assert worker.wait_ready(timeout=10.0)
    try:
        worker.say("hello", "info")  # must not raise even though the engine never came up
        worker.say("alert text", "alert")
    finally:
        worker.close()


# ---------------------------------------------------------------------------
# Integration: the real pyttsx3 backend (essential-features.md section 7.6
# allows exactly one such test; skipped where no audio device is present).
# ---------------------------------------------------------------------------


@pytest.mark.F7
@pytest.mark.slow
def test_real_pyttsx3_backend_initializes_and_speaks() -> None:
    worker = TTSWorker(cooldown_s=0.0)
    try:
        ready = worker.wait_ready(timeout=15.0)
        if not ready:
            pytest.skip("pyttsx3 engine did not become ready on this machine")
        worker.say("Audio self test", "info")
        time.sleep(1.0)  # give the worker a moment to speak before teardown
    finally:
        worker.close()
