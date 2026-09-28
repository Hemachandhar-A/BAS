"""F1 (essential-features.md section 1): perception/camera.py, tested on a
generated clip (real cv2 round-trip) and a fake VideoCapture (camera-path
behaviour that real hardware can't be scripted to exercise on demand:
CAP_DSHOW retry, the measured-fps fallback, transient read failures)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from contracts import CAPTURE_FPS, CAPTURE_HEIGHT, CAPTURE_WIDTH, SourceError
from perception import camera

pytestmark = pytest.mark.F1


def _make_frame(height: int = 24, width: int = 32, value: int = 0) -> np.ndarray:
    return np.full((height, width, 3), value % 256, dtype=np.uint8)


@pytest.fixture
def generated_clip(tmp_path: Path) -> tuple[Path, int, float]:
    """A tiny real .avi written with a known fps and frame count, so
    open_source can be exercised through a real cv2 decode, not a mock."""
    path = tmp_path / "clip.avi"
    fps = 10.0
    n_frames = 20
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), fps, (32, 24))
    assert writer.isOpened()
    for i in range(n_frames):
        writer.write(_make_frame(value=i))
    writer.release()
    return path, n_frames, fps


class _FakeCapture:
    """A stand-in for cv2.VideoCapture that never touches real hardware."""

    def __init__(
        self,
        frames: list[np.ndarray],
        opened: bool = True,
        fps: float = 0.0,
        width: float = 0.0,
        height: float = 0.0,
    ) -> None:
        self._frames = frames
        self._index = 0
        self._opened = opened
        self._fps = fps
        self._props: dict = {
            (cv2.CAP_PROP_FRAME_WIDTH, "fixed"): width,
            (cv2.CAP_PROP_FRAME_HEIGHT, "fixed"): height,
        }

    def isOpened(self) -> bool:
        return self._opened

    def read(self):
        if self._index >= len(self._frames):
            return False, None
        image = self._frames[self._index]
        self._index += 1
        return True, image

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        # Width/height report whatever the fixture fixed them at, ignoring
        # set() -- simulates a driver that won't grant the requested mode
        # until the caller also requests MJPG.
        if prop in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT):
            return self._props.get((prop, "fixed"), 0.0)
        return self._props.get(prop, 0.0)

    def set(self, prop: int, value: float) -> bool:
        self._props[prop] = value
        return True

    def release(self) -> None:
        self._opened = False


# ---------------------------------------------------------------------------
# Real generated clip: the file-source path end to end.
# ---------------------------------------------------------------------------


def test_open_source_reads_a_generated_clip_with_strictly_increasing_frame_ids(
    generated_clip: tuple[Path, int, float],
) -> None:
    path, n_frames, fps = generated_clip
    src = camera.open_source(str(path))
    try:
        assert src.fps == pytest.approx(fps)
        seen_ids = []
        t_values = []
        while True:
            frame = src.read()
            if frame is None:
                break
            seen_ids.append(frame.frame_id)
            t_values.append(frame.t)
            assert frame.image.shape == (24, 32, 3)
            assert frame.image.dtype == np.uint8
        assert seen_ids == list(range(n_frames))
        assert t_values == [i / fps for i in range(n_frames)]
        assert src.exhausted is True
        assert src.read() is None  # exhausted stays exhausted
    finally:
        src.close()


def test_open_source_raises_source_error_for_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(SourceError):
        camera.open_source(str(tmp_path / "does_not_exist.avi"))


# ---------------------------------------------------------------------------
# Fake camera path: behaviour real hardware can't reliably be made to show.
# ---------------------------------------------------------------------------


def test_open_source_raises_source_error_when_first_read_fails(monkeypatch) -> None:
    monkeypatch.setattr(
        camera.cv2, "VideoCapture", lambda *a: _FakeCapture(frames=[], opened=True, fps=30.0)
    )
    with pytest.raises(SourceError):
        camera.open_source("0")


def test_open_source_retries_with_cap_dshow_when_default_backend_fails(monkeypatch) -> None:
    calls: list[tuple] = []

    def fake_ctor(*args):
        calls.append(args)
        if len(args) == 1:
            return _FakeCapture(frames=[], opened=False)
        return _FakeCapture(frames=[_make_frame()], opened=True, fps=30.0)

    monkeypatch.setattr(camera.cv2, "VideoCapture", fake_ctor)
    src = camera.open_source("0")
    src.close()
    assert len(calls) == 2
    assert calls[0] == (0,)
    assert calls[1] == (0, cv2.CAP_DSHOW)


def test_measured_fps_fallback_when_driver_reports_zero(monkeypatch) -> None:
    frames = [_make_frame(value=i) for i in range(35)]
    monkeypatch.setattr(
        camera.cv2, "VideoCapture", lambda *a: _FakeCapture(frames=frames, opened=True, fps=0.0)
    )
    times = iter(100.0 + 0.1 * i for i in range(200))
    monkeypatch.setattr(camera.time, "monotonic", lambda: next(times))

    src = camera.open_source("0")
    try:
        # 0.1s between reads over the first 30 frames -> ~10 fps.
        assert src.fps == pytest.approx(10.0, rel=0.05)
        seen_ids = []
        while True:
            frame = src.read()
            if frame is None:
                break
            seen_ids.append(frame.frame_id)
        assert seen_ids == list(range(35))
    finally:
        src.close()


def test_camera_transient_read_failure_returns_none_without_exhausting(monkeypatch) -> None:
    class _FlakyCapture(_FakeCapture):
        def read(self):
            if self._index == 1:
                self._index += 1
                return False, None
            return super().read()

    frames = [_make_frame(), _make_frame(), _make_frame()]
    monkeypatch.setattr(
        camera.cv2, "VideoCapture", lambda *a: _FlakyCapture(frames=frames, opened=True, fps=30.0)
    )
    src = camera.open_source("0")
    try:
        first = src.read()
        assert first is not None and first.frame_id == 0
        glitch = src.read()
        assert glitch is None
        assert src.exhausted is False  # transient, not end-of-stream
        recovered = src.read()
        assert recovered is not None and recovered.frame_id == 1
    finally:
        src.close()


def test_capture_mode_falls_back_to_mjpg_when_requested_size_not_granted(monkeypatch) -> None:
    captures: list[_FakeCapture] = []

    def fake_ctor(*a):
        cap = _FakeCapture(
            frames=[_make_frame()],
            opened=True,
            fps=30.0,
            width=640.0,
            height=480.0,
        )
        captures.append(cap)
        return cap

    monkeypatch.setattr(camera.cv2, "VideoCapture", fake_ctor)
    src = camera.open_source("0")
    src.close()
    cap = captures[0]
    assert cap._props[cv2.CAP_PROP_FOURCC] == cv2.VideoWriter.fourcc(*"MJPG")


def test_camera_frame_t_is_monotonic_clock_based(monkeypatch) -> None:
    frames = [_make_frame(), _make_frame()]
    monkeypatch.setattr(
        camera.cv2, "VideoCapture", lambda *a: _FakeCapture(frames=frames, opened=True, fps=30.0)
    )
    times = iter([50.0, 50.25, 50.5])
    monkeypatch.setattr(camera.time, "monotonic", lambda: next(times))

    src = camera.open_source("0")
    try:
        first = src.read()
        assert first.t == pytest.approx(0.0)
        second = src.read()
        assert second.t == pytest.approx(0.25)
    finally:
        src.close()


# ---------------------------------------------------------------------------
# DecimatedSource
# ---------------------------------------------------------------------------


def test_decimated_source_keeps_original_frame_ids(
    generated_clip: tuple[Path, int, float],
) -> None:
    path, n_frames, fps = generated_clip
    inner = camera.open_source(str(path))
    src = camera.DecimatedSource(inner, every=3)
    try:
        seen_ids = []
        while True:
            frame = src.read()
            if frame is None:
                break
            seen_ids.append(frame.frame_id)
        assert seen_ids == [i for i in range(n_frames) if i % 3 == 0]
        assert src.exhausted is True
    finally:
        src.close()


def test_decimated_source_rejects_every_less_than_one(
    generated_clip: tuple[Path, int, float],
) -> None:
    path, _, _ = generated_clip
    inner = camera.open_source(str(path))
    try:
        with pytest.raises(ValueError):
            camera.DecimatedSource(inner, every=0)
    finally:
        inner.close()


def test_capture_constants_match_the_locked_specification() -> None:
    # essential-features.md section 0: 1280x720 @ 30fps is a project-wide
    # constant, defined once in contracts.py.
    assert (CAPTURE_WIDTH, CAPTURE_HEIGHT, CAPTURE_FPS) == (1280, 720, 30)
