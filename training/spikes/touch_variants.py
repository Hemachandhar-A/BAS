"""Pure helpers for the P1.2 addendum "START press rule variants" (measurement
only, no change to state/). A variant decides, per frame, whether a hand
"touches" the START card; ``press_events`` then applies the StateTracker's
per-step lifecycle (baseline latch, hysteresis, release) to that flag
sequence; hold times are given in seconds and converted to frames at the
clip's frame rate. No model, no clock, no I/O.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from contracts import Hand
from training.spikes.postprocess import Box

FINGERTIPS = (4, 8, 12, 16, 20)  # thumb, index, middle, ring, pinky tips
INDEX_TIP = 8


@dataclass(frozen=True)
class TouchVariant:
    name: str
    landmarks: tuple[int, ...] | None  # None = all 21 landmarks
    margin: float  # fraction of the card width / height, as touch_margin_frac


VARIANTS: dict[str, TouchVariant] = {
    "V0": TouchVariant("V0", None, 0.10),  # the current StateTracker rule
    "V1a": TouchVariant("V1a", (INDEX_TIP,), 0.10),
    "V1b": TouchVariant("V1b", (INDEX_TIP,), 0.25),
    "V1c": TouchVariant("V1c", (INDEX_TIP,), 0.50),
    "V2": TouchVariant("V2", FINGERTIPS, 0.10),
}


def touch_flags_from_points(
    hands_per_frame: Sequence[Sequence[Hand]],
    box: Box | None,
    landmarks: tuple[int, ...] | None,
    margin: float,
) -> list[bool]:
    """Per frame: True iff any chosen landmark of any hand lies in ``box``
    grown by ``margin`` x width (left/right) and x height (top/bottom),
    inclusive, as ``state.tracker``. ``landmarks=None`` means all 21. No box
    means never touching."""
    if box is None:
        return [False] * len(hands_per_frame)
    x1, y1, x2, y2 = box
    mx, my = margin * (x2 - x1), margin * (y2 - y1)
    gx1, gy1, gx2, gy2 = x1 - mx, y1 - my, x2 + mx, y2 + my
    flags: list[bool] = []
    for hands in hands_per_frame:
        hit = False
        for hand in hands:
            lms = hand.landmarks_px
            pts = lms if landmarks is None else [lms[i] for i in landmarks]
            if any(gx1 <= x <= gx2 and gy1 <= y <= gy2 for x, y in pts):
                hit = True
                break
        flags.append(hit)
    return flags


def touch_flags(
    hands_per_frame: Sequence[Sequence[Hand]], box: Box | None, variant: str
) -> list[bool]:
    v = VARIANTS[variant]
    return touch_flags_from_points(hands_per_frame, box, v.landmarks, v.margin)


def hold_frames(hold_s: float, fps: float) -> int:
    """Frames covering ``hold_s`` seconds at ``fps``: ceil, at least 1.
    The product is rounded first so 0.75 s x 4 fps is 3, not 3.0000000001 -> 4."""
    return max(1, math.ceil(round(hold_s * fps, 9)))


def press_events(flags: Sequence[bool], baseline: int, hysteresis: int, release: int) -> list[int]:
    """Indices (into ``flags``) at which one step fires, with the lifecycle of
    ``state.tracker.StateTracker``: the first ``baseline`` frames never fire,
    and a step already true in them is latched; a latched or fired step
    re-arms after ``release`` consecutive false frames; an armed step true for
    ``hysteresis`` consecutive frames fires once and disarms."""
    events: list[int] = []
    armed, count, false_streak = True, 0, 0
    for i, truth in enumerate(flags):
        if armed:
            if not truth:
                count = 0
                continue
            if i < baseline:
                armed, count, false_streak = False, 0, 0
                continue
            count += 1
            if count >= hysteresis:
                events.append(i)
                armed, count, false_streak = False, 0, 0
        elif truth:
            false_streak = 0
        else:
            false_streak += 1
            if false_streak >= release:
                armed, false_streak, count = True, 0, 0
    return events


def classify_count(n_events: int, expected: int) -> str:
    """``exact`` / ``missed`` (fewer than expected) / ``extra`` (more). Counts
    only, so a clip cannot be both."""
    if n_events == expected:
        return "exact"
    return "missed" if n_events < expected else "extra"


def tally_labels(rows: Sequence[tuple[str, str]]) -> dict[str, dict[str, int]]:
    """Counts of exact / missed / extra over (script_type, label) rows, under
    ``"all"`` and under each script type present."""
    out: dict[str, dict[str, int]] = {"all": {"exact": 0, "missed": 0, "extra": 0, "n": 0}}
    for script_type, label in rows:
        for key in ("all", script_type):
            d = out.setdefault(key, {"exact": 0, "missed": 0, "extra": 0, "n": 0})
            d[label] += 1
            d["n"] += 1
    return out
