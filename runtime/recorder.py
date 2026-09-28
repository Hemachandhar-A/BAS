"""runtime/recorder.py -- Recorder (F11; IMPLEMENTATION_PLAN.md 5.6;
essential-features.md #11).

A writer thread with a bounded queue, fed by the capture thread with every
captured ``Frame`` (at capture rate, never the throttled processing rate).
Writes ``cv2.VideoWriter`` MJPG frames to ``{video_dir}/{run_id}.avi`` --
unlike an unfinalized MP4, a truncated AVI/MJPG file generally remains
playable after a crash, because MJPG has no global index/trailer to
finalize: every ``write()`` call appends a complete, independently
decodable frame. The writer opens lazily on the first frame (so the frame
size never has to be guessed up front) and closes on ``close()``.

**A recorder failure is logged once and never stops the run** (F11): if
the ``cv2.VideoWriter`` fails to open, or a ``write()`` call raises,
recording is silently disabled for the rest of that run rather than
raising out of the writer thread or repeatedly re-attempting a doomed
write. If the queue is full (a slow disk), the newest frame is dropped
with a logged warning rather than blocking the capture thread.
"""

from __future__ import annotations

import logging
import queue
import shutil
import threading
from pathlib import Path

import cv2

from contracts import Frame

logger = logging.getLogger(__name__)

_QUEUE_MAXSIZE = 64
_FOURCC = "MJPG"
_MIN_FREE_BYTES = 200 * 1024 * 1024  # 200 MiB -- a disclosed judgment call


class Recorder:
    """One instance per run. ``open()`` at run start, ``close()`` at run
    end/shutdown. Every public method is safe to call from any thread and
    never raises."""

    def __init__(self, path: str | Path, fps: float, queue_maxsize: int = _QUEUE_MAXSIZE) -> None:
        self._path = Path(path)
        self._fps = fps
        self._queue: queue.Queue[Frame | None] = queue.Queue(maxsize=queue_maxsize)
        self._thread: threading.Thread | None = None
        self._writer: cv2.VideoWriter | None = None
        self._opened = False
        self._closed = False

    def open(self) -> None:
        if self._opened:
            return
        self._opened = True
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._check_free_disk()
        except OSError:
            logger.warning("Recorder: could not prepare %s", self._path.parent, exc_info=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def enqueue(self, frame: Frame) -> None:
        if not self._opened or self._closed:
            return
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            logger.warning("Recorder: queue full, dropping frame %d", frame.frame_id)

    def close(self) -> None:
        if not self._opened or self._closed:
            return
        self._closed = True
        self._put_sentinel()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._release_writer()

    def _put_sentinel(self) -> None:
        # Blocks (bounded) rather than stealing a queued slot: at shutdown
        # every already-queued frame should still get written, not be
        # discarded to make room for the sentinel.
        try:
            self._queue.put(None, timeout=5.0)
        except queue.Full:
            logger.warning("Recorder: could not deliver close sentinel", exc_info=True)

    def _check_free_disk(self) -> None:
        free = shutil.disk_usage(self._path.parent).free
        if free < _MIN_FREE_BYTES:
            logger.warning(
                "Recorder: low disk space (%d MiB free) at %s",
                free // (1024 * 1024),
                self._path.parent,
            )

    def _run(self) -> None:
        disabled = False
        while True:
            item = self._queue.get()
            if item is None:
                return
            if disabled:
                continue
            try:
                self._write(item)
            except Exception:
                logger.warning(
                    "Recorder: write failed, disabling recording for this run", exc_info=True
                )
                disabled = True
                self._release_writer()

    def _write(self, frame: Frame) -> None:
        if self._writer is None:
            height, width = frame.image.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*_FOURCC)
            writer = cv2.VideoWriter(str(self._path), fourcc, self._fps, (width, height))
            if not writer.isOpened():
                raise OSError(f"Recorder: could not open VideoWriter at {self._path}")
            self._writer = writer
        self._writer.write(frame.image)

    def _release_writer(self) -> None:
        if self._writer is not None:
            try:
                self._writer.release()
            except Exception:
                logger.warning("Recorder: error releasing VideoWriter", exc_info=True)
            self._writer = None
