"""perception/camera.py -- FrameSource (F1, essential-features.md section 1).

``open_source(source)`` is the factory named in ``contracts.FrameSource``'s
docstring: ``source`` is an all-digit camera index, a file path, or a
stream URL, all handled by ``cv2.VideoCapture`` -- one module, three
inputs, so an IP camera later is a one-line swap. ``DecimatedSource`` wraps
any ``FrameSource`` for the cache builder (F14 stage 9).

The only clock read in the perception layer (other than this module) is
disallowed by IMPLEMENTATION_PLAN.md rule 8; this module is the exception,
per ``contracts.Frame``'s docstring.
"""

from __future__ import annotations

import logging
import math
import statistics
import time

import cv2

from contracts import CAPTURE_FPS, CAPTURE_HEIGHT, CAPTURE_WIDTH, Frame, SourceError

logger = logging.getLogger(__name__)

_FPS_MEASURE_FRAMES = 30
"""essential-features.md section 1, step 4: measure the median inter-frame
interval over the first 30 frames when the driver reports 0/NaN fps."""


def _looks_like_camera_index(source: str) -> bool:
    return source.isdigit()


class CameraSource:
    """Implements ``contracts.FrameSource`` over ``cv2.VideoCapture``.
    Raises ``SourceError`` at construction if the source cannot be opened
    or the first ``read()`` fails; never from ``read()`` itself."""

    def __init__(self, source: str) -> None:
        self._is_camera = _looks_like_camera_index(source)
        cap = self._open(source)
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            raise SourceError(f"could not open source {source!r}")

        if self._is_camera:
            self._request_capture_mode(cap)

        ok, first_image = cap.read()
        first_t = time.monotonic()
        if not ok or first_image is None:
            cap.release()
            raise SourceError(f"source {source!r} opened but the first read() failed")

        self._cap = cap
        self._source = source
        self._frame_id = 0
        self.exhausted = False
        # Buffered (image, monotonic_read_time) pairs: the first frame plus
        # whatever extra frames the fps measurement below consumes, so no
        # frame is ever dropped just because we had to look ahead for fps.
        self._buffer: list[tuple] = [(first_image, first_t)]

        reported_fps = cap.get(cv2.CAP_PROP_FPS)
        if reported_fps and math.isfinite(reported_fps) and reported_fps > 0:
            self.fps: float | None = float(reported_fps)
        else:
            logger.info("source %r reported no usable fps; measuring", source)
            self.fps = self._measure_fps()
        self._start_monotonic = first_t

    def _open(self, source: str) -> cv2.VideoCapture | None:
        if self._is_camera:
            index = int(source)
            cap = cv2.VideoCapture(index)
            if not cap.isOpened():
                # essential-features.md section 1, step 2: Windows often
                # needs CAP_DSHOW when the default backend is slow or fails.
                cap.release()
                cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            return cap
        return cv2.VideoCapture(source)

    def _request_capture_mode(self, cap: cv2.VideoCapture) -> None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
        granted_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        granted_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (granted_w, granted_h) != (CAPTURE_WIDTH, CAPTURE_HEIGHT):
            # Many USB webcams only reach 720p30 in MJPG mode.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
            granted_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            granted_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except cv2.error:
            pass
        logger.info("capture mode granted: %dx%d", granted_w, granted_h)

    def _measure_fps(self) -> float | None:
        timestamps = [self._buffer[0][1]]
        while len(self._buffer) < _FPS_MEASURE_FRAMES:
            ok, image = self._cap.read()
            t = time.monotonic()
            if not ok or image is None:
                break
            self._buffer.append((image, t))
            timestamps.append(t)
        if len(timestamps) < 2:
            return None
        intervals = [b - a for a, b in zip(timestamps, timestamps[1:], strict=False)]
        median_interval = statistics.median(intervals)
        if median_interval <= 0:
            return None
        return 1.0 / median_interval

    def _frame_t(self, monotonic_read_time: float) -> float:
        if self._is_camera:
            return monotonic_read_time - self._start_monotonic
        fps = self.fps or float(CAPTURE_FPS)
        return self._frame_id / fps

    def read(self) -> Frame | None:
        if self.exhausted:
            return None
        if self._buffer:
            image, read_time = self._buffer.pop(0)
        else:
            ok, image = self._cap.read()
            read_time = time.monotonic()
            if not ok or image is None:
                if self._is_camera:
                    return None  # transient failure; exhausted stays False
                self.exhausted = True
                return None
        frame = Frame(frame_id=self._frame_id, t=self._frame_t(read_time), image=image)
        self._frame_id += 1
        return frame

    def close(self) -> None:
        self._cap.release()


def open_source(source: str) -> CameraSource:
    """Factory named in ``contracts.FrameSource``'s docstring."""
    return CameraSource(source)


class DecimatedSource:
    """Wraps any ``FrameSource``, yielding every ``every``-th frame while
    keeping the original ``frame_id`` (so ``t`` stays correct). Used by the
    cache builder (F14 stage 9) to sample a recording at
    ``RuntimeConfig.target_fps``."""

    def __init__(self, src, every: int) -> None:
        if every < 1:
            raise ValueError(f"every must be >= 1, got {every!r}")
        self._src = src
        self._every = every
        self.fps = src.fps
        self.exhausted = src.exhausted

    def read(self) -> Frame | None:
        while True:
            frame = self._src.read()
            self.exhausted = self._src.exhausted
            if frame is None:
                return None
            if frame.frame_id % self._every == 0:
                return frame

    def close(self) -> None:
        self._src.close()
