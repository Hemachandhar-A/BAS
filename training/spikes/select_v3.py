"""Pure selection helpers for P1.2 checkpoint 1c (selection v3). No model, no
I/O beyond reading pixels from an array passed in. Builds on
``select_v2``: v2's static consensus elected the RED CONTAINER as the
``start_button`` in runs x034 and x044, and the rule "reject containers that
overlap the start_button consensus" then removed the real red box. v3 fixes
the cause (a start_button candidate must look like white paper) and removes
the circular rule (a consensus excludes containers only if it itself passed
that test).
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

from training.spikes.postprocess import Box, RawDetection, iou
from training.spikes.select_v2 import StaticConsensus, central_box, static_consensus

SAME_BOX_IOU = 0.9  # status_change treats boxes this similar as unchanged


def is_white_paper(sat: float, val: float, sat_max: float, val_min: float) -> bool:
    """White paper is pale (low saturation) and bright. ``sat``/``val`` are
    OpenCV-scale means (0-255). Saturation is what separates the START card
    from the red and lime containers; ``val_min`` only guards against dark
    objects."""
    return sat <= sat_max and val >= val_min


def mean_sat_val_in_box(
    image_bgr: np.ndarray, box: Box, frac: float = 0.5
) -> tuple[float, float] | None:
    """Mean OpenCV saturation and value over the central ``frac`` of ``box``
    (clipped to the image). None if the clipped region is empty."""
    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2 = central_box(box, frac)
    xi1, yi1 = max(0, int(round(x1))), max(0, int(round(y1)))
    xi2, yi2 = min(width, int(round(x2))), min(height, int(round(y2)))
    if xi2 <= xi1 or yi2 <= yi1:
        return None
    hsv = cv2.cvtColor(image_bgr[yi1:yi2, xi1:xi2], cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 1])), float(np.mean(hsv[:, :, 2]))


def static_consensus_filtered(
    candidates_by_frame: dict[str, list[RawDetection]],
    area_band: tuple[float, float],
    width: int,
    height: int,
    accept: Callable[[str, RawDetection], bool],
) -> StaticConsensus:
    """v2's ``static_consensus`` on only the candidates ``accept(frame_id,
    det)`` lets through. A frame left with no accepted candidate agreeing
    with the consensus is missing; nothing is invented."""
    kept = {
        frame_id: [d for d in dets if accept(frame_id, d)]
        for frame_id, dets in candidates_by_frame.items()
    }
    return static_consensus(kept, area_band, width, height)


def exclusion_boxes(
    consensus: dict[str, Box | None], start_button_is_white: bool | None
) -> dict[str, Box | None]:
    """Boxes a container candidate may not overlap. outer_box and tray are
    always excluded; the start_button consensus only if it itself passed the
    white-paper test (``True``). ``False`` or ``None`` (never measured) drops
    it, which breaks v2's circular rule."""
    out = dict(consensus)
    if "start_button" in out and start_button_is_white is not True:
        del out["start_button"]
    return out


def tight_band(
    clean_areas: list[float],
    lo_margin: float = 0.85,
    hi_quantile: float = 0.90,
    hi_margin: float = 1.25,
) -> tuple[float, float]:
    """Container area-fraction band from areas measured on clean frames:
    ``min * lo_margin`` to ``quantile(hi_quantile) * hi_margin``. The upper
    end uses a quantile, not the max, so one stretched box in the sample
    cannot widen the band."""
    if not clean_areas:
        raise ValueError("tight_band needs at least one area")
    arr = np.asarray(clean_areas, dtype=np.float64)
    return float(arr.min() * lo_margin), float(np.quantile(arr, hi_quantile) * hi_margin)


def status_change(v2_box: Box | None, v3_box: Box | None) -> tuple[str, str] | None:
    """(v2_status, v3_status) when a frame's box for one class changed
    between v2 and v3, else None. Statuses: ``missing``, ``ok``, and
    ``ok_moved`` (v3 box has IoU < SAME_BOX_IOU with the v2 box)."""
    if v2_box is None and v3_box is None:
        return None
    if v2_box is None:
        return "missing", "ok"
    if v3_box is None:
        return "ok", "missing"
    if iou(v2_box, v3_box) >= SAME_BOX_IOU:
        return None
    return "ok", "ok_moved"


def pick_holdout_runs(
    manifest_rows: list[dict], used_run_ids: set[str], n_runs: int, seed: int
) -> list[str]:
    """Seeded sample of ``n_runs`` TRAIN runs not in ``used_run_ids``
    (never a val or test run). Sorted, so the plan is stable."""
    import random

    pool = sorted(
        r["run_id"]
        for r in manifest_rows
        if r["split"] == "train" and r["run_id"] not in used_run_ids
    )
    if len(pool) < n_runs:
        raise ValueError(f"only {len(pool)} unused train runs, need {n_runs}")
    return sorted(random.Random(seed).sample(pool, k=n_runs))
