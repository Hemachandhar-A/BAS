"""harness/fakes.py -- test doubles for P2's harness (IMPLEMENTATION_PLAN.md
Part 10, P2.4: "harness/fakes.py (a fake Perception producing scripted
PerceptionFrames)"). Also a fake FrameSource so runtime/loop.py's threaded
capture/inference path can be driven deterministically without a camera or
``perception/`` existing yet (P1.1/P1.6 land later; R6 -- P2 builds against
its own fake rather than waiting)."""

from __future__ import annotations

import time

import numpy as np

from contracts import Frame, PerceptionFrame


def make_blank_frame(frame_id: int, t: float, height: int = 4, width: int = 4) -> Frame:
    """A minimal valid ``Frame`` for tests that don't care about pixel
    content -- just the contract shape ``(H, W, 3)`` uint8 BGR."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    return Frame(frame_id=frame_id, t=t, image=image)


class FakePerception:
    """Implements ``contracts.Perception``. Ignores the raw ``Frame`` it is
    given (besides carrying its ``frame_id``/``t`` through) and instead
    replays a scripted list of ``PerceptionFrame``s, one per call to
    ``process`` -- deterministic regardless of real capture timing. Once
    the script is exhausted, further calls return an empty
    ``PerceptionFrame`` (no detections/hands) rather than raising, matching
    a real pipeline that simply sees nothing more of interest."""

    model_stamp = "fake:00000000|hand:00000000|pose:00000000"

    def __init__(self, script: list[PerceptionFrame]) -> None:
        self._script = list(script)
        self._index = 0

    def process(self, frame: Frame) -> PerceptionFrame:
        if self._index < len(self._script):
            scripted = self._script[self._index]
            self._index += 1
            return scripted.model_copy(update={"frame_id": frame.frame_id, "t": frame.t})
        return PerceptionFrame(frame_id=frame.frame_id, t=frame.t)

    def reset(self) -> None:
        self._index = 0


class FakeFrameSource:
    """Implements ``contracts.FrameSource``. Yields ``items`` in order --
    each either a ``Frame`` (a captured frame) or ``None`` (a transient
    read failure, ``exhausted`` stays ``False``); once ``items`` is
    exhausted, ``read()`` returns ``None`` with ``exhausted = True``,
    matching the file-replay semantics of the real Protocol.

    ``read_delay`` paces real captured frames (never the ``None`` gaps or
    the terminal exhausted read) by sleeping that many seconds before
    returning -- only useful for threaded ``RuntimeLoop`` tests that need
    the inference thread to reliably keep up with capture instead of
    racing it (latest-frame-wins is allowed to drop frames under real
    load; a test asserting exactly which frames got processed needs
    capture paced slower than the inference throttle to stay
    deterministic)."""

    def __init__(
        self, items: list[Frame | None], fps: float | None = None, read_delay: float = 0.0
    ) -> None:
        self._items = list(items)
        self._index = 0
        self._read_delay = read_delay
        self.fps = fps
        self.exhausted = False
        self.closed = False

    def read(self) -> Frame | None:
        if self._index >= len(self._items):
            self.exhausted = True
            return None
        item = self._items[self._index]
        self._index += 1
        if item is not None and self._read_delay > 0.0:
            time.sleep(self._read_delay)
        return item

    def close(self) -> None:
        self.closed = True
