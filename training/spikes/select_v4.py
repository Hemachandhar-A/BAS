"""Pure selection helpers for rules v4 (P1.4): v3 plus three container rules.

(a) an area CEILING of about 2x the clean median container area, replacing
    v3's band top, so a rotated or blurred (larger) container survives while a
    box over two containers (above ~0.11 of the frame) stays out;
(b) a smaller FLOOR (0.015) that applies only to candidates whose central-patch
    hue and saturation fall inside the class's colour range measured on clean
    frames, so a small visible part of a hand-covered container is kept while
    skin-coloured fragments are not;
(c) a SIZE CAP: a candidate wider or taller than about 1.6x the clean median
    container width or height is rejected (a box stretched over a hand).

No model, no I/O beyond reading pixels from an array passed in. Static
classes and the START white-paper test are unchanged from v3 (select_v3).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from training.spikes.postprocess import Box, RawDetection, box_area_frac, iou
from training.spikes.select_v2 import ContainerSelection, central_box
from training.spikes.select_v3 import SAME_BOX_IOU

CEILING_HARD_MAX = 0.11  # a box over two containers is at least this large
CEILING_FACTOR = 2.0
SIZE_FACTOR = 1.6
SMALL_FLOOR = 0.015
HUE_MARGIN = 2.0
SAT_MARGIN = 10.0
CONTAINER_CLASSES = ("red_box", "yellow_box")

HueSat = tuple[float, float]
Cell = tuple[str, int, str]


@dataclass(frozen=True)
class CleanCell:
    """One container box a reviewer did not mark bad, with its measurements."""

    cls: str
    area: float
    width: float
    height: float
    hue: float  # OpenCV 0-179, circular mean over the central patch
    sat: float  # OpenCV 0-255, mean over the central patch


# --- colour -------------------------------------------------------------------


def signed_hue(hue: float) -> float:
    """OpenCV hue (0-179) mapped to (-90, 90], so red (which wraps at 0/179)
    is one contiguous range around 0."""
    return float(hue) if hue < 90 else float(hue) - 180.0


def in_colour_range(hue: float, sat: float, rng: dict) -> bool:
    """``rng`` = {"hue": [lo, hi] in signed hue, "sat": [lo, hi]}, inclusive."""
    h = signed_hue(hue)
    return rng["hue"][0] <= h <= rng["hue"][1] and rng["sat"][0] <= sat <= rng["sat"][1]


def patch_hue_sat(image_bgr: np.ndarray, box: Box, frac: float = 0.5) -> HueSat | None:
    """Circular-mean hue (0-179) and mean saturation over the central ``frac``
    of ``box`` (clipped to the image; HSV via COLOR_BGR2HSV). None if empty."""
    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2 = central_box(box, frac)
    xi1, yi1 = max(0, int(round(x1))), max(0, int(round(y1)))
    xi2, yi2 = min(width, int(round(x2))), min(height, int(round(y2)))
    if xi2 <= xi1 or yi2 <= yi1:
        return None
    hsv = cv2.cvtColor(image_bgr[yi1:yi2, xi1:xi2], cv2.COLOR_BGR2HSV).reshape(-1, 3)
    angle = hsv[:, 0].astype(np.float64) * (2.0 * np.pi / 180.0)
    mean_angle = np.arctan2(np.sin(angle).mean(), np.cos(angle).mean())
    hue = (np.degrees(mean_angle) % 360.0) / 2.0
    return float(hue % 180.0), float(hsv[:, 1].mean())


def derive_colour_range(hues: list[float], sats: list[float]) -> dict:
    """Colour range from clean cells: 5th to 95th percentile of the signed hue
    and of the saturation, widened by ``HUE_MARGIN`` and ``SAT_MARGIN``."""
    sh = np.asarray([signed_hue(h) for h in hues])
    ss = np.asarray(sats, dtype=np.float64)
    return {
        "hue": [
            float(np.percentile(sh, 5) - HUE_MARGIN),
            float(np.percentile(sh, 95) + HUE_MARGIN),
        ],
        "sat": [
            float(max(0.0, np.percentile(ss, 5) - SAT_MARGIN)),
            float(min(255.0, np.percentile(ss, 95) + SAT_MARGIN)),
        ],
    }


# --- parameters ---------------------------------------------------------------


def derive_v4(cells: list[CleanCell], v3_band: tuple[float, float]) -> dict:
    """The v4 parameters from clean container cells (pooled over both classes
    for area, width and height; per class for colour). Raises if the ceiling
    would reach ``CEILING_HARD_MAX``."""
    if not cells:
        raise ValueError("derive_v4 needs at least one clean cell")
    area_median = float(np.median([c.area for c in cells]))
    ceiling = CEILING_FACTOR * area_median
    if ceiling >= CEILING_HARD_MAX:
        raise ValueError(f"area ceiling {ceiling:.4f} would admit two-container boxes")
    colour = {}
    for cls in CONTAINER_CLASSES:
        own = [c for c in cells if c.cls == cls]
        if not own:
            raise ValueError(f"no clean cell for {cls}")
        colour[cls] = derive_colour_range([c.hue for c in own], [c.sat for c in own])
    width_median = float(np.median([c.width for c in cells]))
    height_median = float(np.median([c.height for c in cells]))
    return {
        "container_band": [float(v3_band[0]), float(v3_band[1])],
        "area_median": area_median,
        "area_ceiling": ceiling,
        "width_median": width_median,
        "width_max": SIZE_FACTOR * width_median,
        "height_median": height_median,
        "height_max": SIZE_FACTOR * height_median,
        "small_floor": SMALL_FLOOR,
        "colour": colour,
        "n_clean": len(cells),
    }


# --- selection ----------------------------------------------------------------


def select_container_v4(
    detections: list[RawDetection],
    params: dict,
    cls: str,
    width: int,
    height: int,
    exclude_boxes: dict[str, Box | None],
    measure: Callable[[Box], HueSat | None],
    exclude_iou_threshold: float = 0.5,
) -> ContainerSelection:
    """Choose one container box in a frame. A candidate is rejected, in this
    order, as ``out_of_band`` (area below ``small_floor`` or above
    ``area_ceiling``), ``too_wide`` / ``too_tall``, ``small_wrong_colour``
    (area below v3's floor and the central patch not inside ``cls``'s colour
    range, or not measurable), ``overlaps_<name>``. The highest score among
    the rest wins."""
    lo3 = params["container_band"][0]
    floor, ceiling = params["small_floor"], params["area_ceiling"]
    rejected: list[tuple[RawDetection, str]] = []
    valid: list[RawDetection] = []
    for det in detections:
        frac = box_area_frac(det.box, width, height)
        if not (floor <= frac <= ceiling):
            rejected.append((det, "out_of_band"))
            continue
        bw, bh = det.box[2] - det.box[0], det.box[3] - det.box[1]
        if bw > params["width_max"]:
            rejected.append((det, "too_wide"))
            continue
        if bh > params["height_max"]:
            rejected.append((det, "too_tall"))
            continue
        if frac < lo3:
            hs = measure(det.box)
            if hs is None or not in_colour_range(hs[0], hs[1], params["colour"][cls]):
                rejected.append((det, "small_wrong_colour"))
                continue
        reason = None
        for name, box in exclude_boxes.items():
            if box is not None and iou(det.box, box) > exclude_iou_threshold:
                reason = f"overlaps_{name}"
                break
        if reason is not None:
            rejected.append((det, reason))
            continue
        valid.append(det)
    if not valid:
        return ContainerSelection(None, rejected)
    chosen = max(valid, key=lambda d: d.score)
    rejected.extend((d, "not_highest_score") for d in valid if d is not chosen)
    return ContainerSelection(chosen, rejected)


# --- the safeguard ------------------------------------------------------------


def compare_to_v3(
    v3: dict[Cell, Box | None], v4: dict[Cell, Box | None], bad_cells: set[Cell]
) -> dict[str, list[Cell]]:
    """Cells whose box differs between v3 and v4 (one is None and the other is
    not, or IoU < ``SAME_BOX_IOU``), split by whether the Lead marked the cell
    bad. A change in an UNMARKED cell trips the safeguard."""
    in_bad: list[Cell] = []
    unmarked: list[Cell] = []
    for cell in sorted(v3):
        a, b = v3[cell], v4.get(cell)
        if a is None and b is None:
            continue
        if a is not None and b is not None and iou(a, b) >= SAME_BOX_IOU:
            continue
        (in_bad if cell in bad_cells else unmarked).append(cell)
    return {"changed_in_bad_cells": in_bad, "changed_in_unmarked_cells": unmarked}


def reintroduced_wrong(
    v4: dict[Cell, Box | None], cells: list[Cell], lead_wrong: dict[Cell, list[Box]]
) -> list[Cell]:
    """Cells whose v4 box is (IoU >= ``SAME_BOX_IOU``) a box the Lead already
    marked ``wrong_box`` in an earlier review. v3 had dropped such a box (the
    frame was excluded); v4 putting it back puts a wrong label in the dataset,
    which is worse than a missing one."""
    out = []
    for cell in cells:
        box = v4.get(cell)
        if box is not None and any(iou(box, w) >= SAME_BOX_IOU for w in lead_wrong.get(cell, [])):
            out.append(cell)
    return out
