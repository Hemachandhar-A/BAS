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


def _slow_null_engine_factory():
    """Module-level (picklable) stand-in for a hanging engine init, so
    tests can exercise ``wait_ready``'s timeout without a real device."""
    time.sleep(2.0)
    return null_engine_factory()


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


@pytest.mark.F7
def test_fake_speaker_ignores_calls_after_close() -> None:
    # Parity with TTSWorker.say(), which also silently drops post-close
    # calls -- a golden asserting "nothing spoken after shutdown" must
    # behave the same against either implementation.
    speaker = FakeSpeaker()
    speaker.say("Stow the red sample", "info")
    speaker.close()
    speaker.say("Should be dropped", "alert")
    assert speaker.calls == [("Stow the red sample", "info")]


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


@pytest.mark.F7
def test_say_recovers_from_a_crashed_worker_process(worker: TTSWorker) -> None:
    # Simulate the worker dying on its own (e.g. a driver fault), bypassing
    # our own _terminate_process() -- self._process still points at the
    # now-dead Process object, exactly like an unexpected crash would.
    worker._process.terminate()
    worker._process.join(timeout=2.0)
    assert not worker._process.is_alive()
    dead_process = worker._process

    worker.say("Stow the red sample", "info")

    assert worker._process is not dead_process
    assert worker._process.is_alive()


@pytest.mark.F7
def test_wait_ready_default_timeout_uses_ready_timeout_s() -> None:
    worker = TTSWorker(engine_factory=_slow_null_engine_factory, ready_timeout_s=0.2)
    try:
        start = time.perf_counter()
        assert worker.wait_ready() is False  # init takes 2s, default timeout is 0.2s
        assert time.perf_counter() - start < 1.5
    finally:
        worker.close()


@pytest.mark.F7
def test_close_lets_a_responsive_worker_exit_without_forced_terminate(
    worker: TTSWorker,
) -> None:
    process = worker._process
    calls: list[None] = []
    original_terminate = process.terminate

    def _spy_terminate() -> None:
        calls.append(None)
        original_terminate()

    process.terminate = _spy_terminate  # type: ignore[method-assign]
    worker.close()
    assert not process.is_alive()
    assert calls == []  # exited via the sentinel, no forced terminate needed


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
