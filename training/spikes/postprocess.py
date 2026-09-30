"""Pure post-processing helpers for the P1.2 detection spike.

No model, no I/O -- these are the only spike functions with unit tests
(training/spikes/detect_spike.py just calls them). Boxes are always
``(x1, y1, x2, y2)`` in original-image pixels, matching ``contracts.Detection``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class RawDetection:
    """One Grounding DINO output before it is assigned to an experiment
    class: ``phrase`` is the raw text span the model matched, ``score`` its
    confidence, ``box`` its pixel box."""

    phrase: str
    score: float
    box: Box


def box_area_frac(box: Box, width: int, height: int) -> float:
    """Fraction of the frame area covered by ``box``."""
    x1, y1, x2, y2 = box
    frame_area = width * height
    if frame_area <= 0:
        return 0.0
    return max(0.0, (x2 - x1)) * max(0.0, (y2 - y1)) / frame_area


def iou(a: Box, b: Box) -> float:
    """Intersection-over-union of two boxes; 0 if they do not overlap or
    either has non-positive area."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def median_box(boxes: list[Box]) -> Box | None:
    """Per-coordinate median box, or None if ``boxes`` is empty."""
    if not boxes:
        return None
    arr = np.asarray(boxes, dtype=np.float64)
    x1, y1, x2, y2 = np.median(arr, axis=0)
    return (float(x1), float(y1), float(x2), float(y2))


def mean_hue_in_box(image_bgr: np.ndarray, box: Box) -> float | None:
    """Mean OpenCV hue (0-179) of the pixels inside ``box``, clipped to the
    image. None if the clipped box is empty."""
    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2 = box
    xi1, yi1 = max(0, int(round(x1))), max(0, int(round(y1)))
    xi2, yi2 = min(width, int(round(x2))), min(height, int(round(y2)))
    if xi2 <= xi1 or yi2 <= yi1:
        return None
    crop = image_bgr[yi1:yi2, xi1:xi2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 0]))


def select_best_in_band(
    detections: list[RawDetection],
    area_band: tuple[float, float] | None,
    width: int,
    height: int,
) -> RawDetection | None:
    """The single highest-scoring detection whose box falls inside
    ``area_band`` (essential-features.md section 14, Stage 3: "keep the
    single highest-scoring box per class"), or None if none qualifies.

    Callers scope ``detections`` to one experiment class themselves (by
    only ever submitting that class's own candidate phrase(s) to the
    detector for this call) -- Grounding DINO's returned text span does not
    reliably echo the submitted phrase verbatim (it drops the trailing
    period and, near the text-match threshold, can return a partial span
    such as "red" or "a"), so matching back to a class by phrase string is
    fragile and unnecessary once the call itself is already class-scoped.
    """
    candidates = detections
    if area_band is not None:
        lo, hi = area_band
        candidates = [d for d in candidates if lo <= box_area_frac(d.box, width, height) <= hi]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.score)
