"""runtime/recorder.py -- Recorder (F11; IMPLEMENTATION_PLAN.md 5.6;
essential-features.md #11): one playable file per run, a truncated file
still plays, a recorder failure is logged and never stops the run, and a
full queue drops the newest frame instead of stalling capture."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

import runtime.recorder as recorder_module
from contracts import Frame
from runtime.recorder import Recorder


def _frame(frame_id: int, height: int = 8, width: int = 8) -> Frame:
    image = np.random.default_rng(frame_id).integers(0, 255, (height, width, 3), dtype=np.uint8)
    return Frame(frame_id=frame_id, t=float(frame_id), image=image)


@pytest.mark.F11
def test_recorded_run_produces_one_playable_avi_named_by_run_id(tmp_path: Path) -> None:
    run_id = "test-run-001"
    path = tmp_path / f"{run_id}.avi"
    rec = Recorder(path, fps=10.0)
    rec.open()
    for i in range(5):
        rec.enqueue(_frame(i))
    rec.close()

    assert path.exists()
    assert path.name == f"{run_id}.avi"

    cap = cv2.VideoCapture(str(path))
    try:
        assert cap.isOpened()
        ok, frame = cap.read()
        assert ok
        assert frame is not None
        assert frame.shape == (8, 8, 3)
    finally:
        cap.release()


@pytest.mark.F11
def test_close_before_release_still_leaves_earlier_frames_playable(tmp_path: Path) -> None:
    """MJPG has no global index/trailer to finalize: every write() appends
    a complete frame, so even a run that ends without a clean shutdown
    should leave everything written so far readable. We simulate that by
    reading the file back after a normal close() (release() is called by
    close(), which is the only way to safely reopen it for reading on
    Windows) -- the point under test is that partial writes before any
    crash are not corrupted or buffered away."""
    path = tmp_path / "truncated.avi"
    rec = Recorder(path, fps=10.0)
    rec.open()
    for i in range(3):
        rec.enqueue(_frame(i))
    rec.close()

    cap = cv2.VideoCapture(str(path))
    try:
        assert cap.isOpened()
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert frame_count >= 1
    finally:
        cap.release()


class _FakeWriter:
    """Blocks on write() until released, so the queue backs up
    predictably -- used to test the "queue full -> drop" path without
    relying on real disk timing."""

    def __init__(self, release_event: threading.Event) -> None:
        self._release_event = release_event
        self.writes: list[np.ndarray] = []

    def isOpened(self) -> bool:
        return True

    def write(self, image: np.ndarray) -> None:
        self._release_event.wait(timeout=5.0)
        self.writes.append(image)

    def release(self) -> None:
        pass


@pytest.mark.F11
def test_full_queue_drops_newest_frame_without_blocking_caller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    release_event = threading.Event()
    fake_writer = _FakeWriter(release_event)
    monkeypatch.setattr(recorder_module.cv2, "VideoWriter", lambda *a, **kw: fake_writer)

    rec = Recorder(tmp_path / "full.avi", fps=10.0, queue_maxsize=1)
    with caplog.at_level(logging.WARNING, logger=recorder_module.__name__):
        rec.open()
        rec.enqueue(_frame(0))  # picked up by the worker immediately, blocks in write()
        # Give the worker thread a moment to actually call write() and block.
        import time

        time.sleep(0.2)
        rec.enqueue(_frame(1))  # fills the maxsize=1 queue
        rec.enqueue(_frame(2))  # queue full -> dropped, logged, never blocks
        release_event.set()
        rec.close()

    assert len(fake_writer.writes) == 2  # frame 0 and frame 1; frame 2 was dropped
    assert any("queue full" in record.message.lower() for record in caplog.records)


class _FailingWriter:
    def isOpened(self) -> bool:
        return False

    def write(self, image: np.ndarray) -> None:  # pragma: no cover - never reached
        raise AssertionError("write() must not be called on an unopened writer")

    def release(self) -> None:
        pass


@pytest.mark.F11
def test_writer_open_failure_is_logged_and_never_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(recorder_module.cv2, "VideoWriter", lambda *a, **kw: _FailingWriter())

    rec = Recorder(tmp_path / "fails.avi", fps=10.0)
    with caplog.at_level(logging.WARNING, logger=recorder_module.__name__):
        rec.open()
        rec.enqueue(_frame(0))
        rec.enqueue(_frame(1))  # must be silently dropped after the first failure, not retried
        rec.close()

    assert any("write failed" in record.message.lower() for record in caplog.records)


@pytest.mark.F11
def test_low_free_disk_space_warns_but_still_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class _FakeUsage:
        free = 1024  # far below the 200 MiB threshold

    monkeypatch.setattr(recorder_module.shutil, "disk_usage", lambda _path: _FakeUsage())

    rec = Recorder(tmp_path / "lowdisk.avi", fps=10.0)
    with caplog.at_level(logging.WARNING, logger=recorder_module.__name__):
        rec.open()
        rec.enqueue(_frame(0))
        rec.close()

    assert any("low disk space" in record.message.lower() for record in caplog.records)
    assert (tmp_path / "lowdisk.avi").exists()
