"""Pure per-run rules for the auto-labeler (F14 stage 3), no model and no I/O.

Static classes (``outer_box``, ``tray``, ``start_button``) are decided ONCE per
run from a few frames spread across it, by v3's consensus plus the white-paper
test for the START card, and that one box is applied to every frame of the run.
Movable classes (``red_box``, ``yellow_box``) are decided per frame by rules v4
(training/spikes/select_v4.py). v3 is v4 with neutral parameters
(``v3_equivalent_container_params``), so one code path serves whichever rule set
``data/labels/frozen_rules.json`` names.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from training.spikes.postprocess import Box, RawDetection
from training.spikes.select_v2 import ContainerSelection, static_consensus
from training.spikes.select_v3 import (
    exclusion_boxes,
    is_white_paper,
    static_consensus_filtered,
)
from training.spikes.select_v4 import HueSat, select_container_v4

STATIC_CLASSES = ("outer_box", "tray", "start_button")
MOVABLE_CLASSES = ("red_box", "yellow_box")
N_STATIC_FRAMES = 5
_NO_LIMIT = 1.0e9


@dataclass(frozen=True)
class StaticResult:
    """One run's static boxes (None = no frame agreed on any box) and the
    START consensus's own white-paper measurement."""

    boxes: dict[str, Box | None]
    start_button_white: bool | None
    start_button_measure: tuple[float, float] | None


def static_frame_ids(frame_ids: list[int], n: int = N_STATIC_FRAMES) -> list[int]:
    """``n`` frames spread evenly across a run (first and last included); all
    of them if the run has fewer."""
    ids = sorted(frame_ids)
    if len(ids) <= n:
        return ids
    idx = np.unique(np.round(np.linspace(0, len(ids) - 1, n)).astype(int))
    return [ids[i] for i in idx]


def v3_equivalent_container_params(container_band: list[float] | tuple[float, float]) -> dict:
    """v4 container parameters that reproduce v3 exactly: the band is the
    floor and the ceiling, there is no size cap, so the colour test never
    runs."""
    lo, hi = float(container_band[0]), float(container_band[1])
    return {
        "container_band": [lo, hi],
        "area_ceiling": hi,
        "small_floor": lo,
        "width_max": _NO_LIMIT,
        "height_max": _NO_LIMIT,
        "colour": {},
    }


def select_statics(
    static_frames: list[int],
    cands: dict[tuple[int, str], list[RawDetection]],
    width: int,
    height: int,
    rules: dict,
    measure_sv: Callable[[int, Box], tuple[float, float] | None],
) -> StaticResult:
    """v3's static rule over ``static_frames``. ``cands[(frame_id, class)]``
    are the raw candidates; ``measure_sv(frame_id, box)`` is the mean OpenCV
    (saturation, value) over the box's central half."""
    st = rules["static"]
    bands = {k: tuple(v) for k, v in st["static_bands"].items()}
    sat_max, val_min = st["sat_max"], st["val_min"]

    cache: dict[tuple[int, Box], tuple[float, float] | None] = {}

    def measure(fid: int, box: Box) -> tuple[float, float] | None:
        if (fid, box) not in cache:
            cache[(fid, box)] = measure_sv(fid, box)
        return cache[(fid, box)]

    def accept(fid: int, det: RawDetection) -> bool:
        sv = measure(fid, det.box)
        return sv is not None and is_white_paper(sv[0], sv[1], sat_max, val_min)

    cons = {}
    for cls in ("outer_box", "tray"):
        per_frame = {f: cands.get((f, cls), []) for f in static_frames}
        cons[cls] = static_consensus(per_frame, bands[cls], width, height)
    per_frame = {f: cands.get((f, "start_button"), []) for f in static_frames}
    cons["start_button"] = static_consensus_filtered(
        per_frame, bands["start_button"], width, height, accept
    )

    sb = cons["start_button"]
    sb_white: bool | None = None
    sb_meas: tuple[float, float] | None = None
    if sb.consensus_box is not None:
        ms = [
            m
            for f in static_frames
            if sb.per_frame_box[f] is not None and (m := measure(f, sb.consensus_box)) is not None
        ]
        if ms:
            sb_meas = (float(np.median([m[0] for m in ms])), float(np.median([m[1] for m in ms])))
            sb_white = is_white_paper(sb_meas[0], sb_meas[1], sat_max, val_min)
    return StaticResult(
        boxes={c: cons[c].consensus_box for c in STATIC_CLASSES},
        start_button_white=sb_white,
        start_button_measure=sb_meas,
    )


def select_movable(
    cands_by_class: dict[str, list[RawDetection]],
    width: int,
    height: int,
    rules: dict,
    statics: StaticResult,
    measure_hs: Callable[[Box], HueSat | None],
) -> dict[str, ContainerSelection]:
    """Rules v4 for each movable class of one frame. Containers may not
    overlap the run's outer box or tray, nor its START card if the card itself
    passed the white-paper test (v3's non-circular rule)."""
    exclude = exclusion_boxes(dict(statics.boxes), statics.start_button_white)
    return {
        cls: select_container_v4(
            cands_by_class.get(cls, []),
            rules["container"],
            cls,
            width,
            height,
            exclude,
            measure_hs,
        )
        for cls in MOVABLE_CLASSES
    }
