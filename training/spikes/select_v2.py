"""Pure selection helpers for the P1.2 checkpoint-1b re-selection
(IMPLEMENTATION_PLAN.md Part 10, P1.2). No model, no I/O -- these consume
``RawDetection`` candidates already produced by
``training/spikes/run_raw_cache.py`` (training/spikes/run_checkpoint1b.py
does the orchestration). Checkpoint 1 found that its single "highest score
in a 0.1%-60% band" rule (``postprocess.select_best_in_band``) lets the
whole cardboard outer box win the red/yellow container classes in most
frames; these helpers replace that rule with a per-class area band plus
static-object consensus plus overlap rejection against the other classes'
consensus boxes.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from training.spikes.postprocess import Box, RawDetection, box_area_frac, iou, median_box


@dataclass(frozen=True)
class Cluster:
    """A group of candidates, from possibly different frames, whose boxes
    mutually agree (``cluster_by_iou``'s threshold) that they are the same
    physical object."""

    members: tuple[tuple[str, RawDetection], ...]

    @property
    def frame_support(self) -> int:
        """Number of DISTINCT frames contributing to this cluster -- two
        candidates from the same frame (e.g. two phrasings both landing on
        the same box) count once, not twice."""
        return len({frame_id for frame_id, _ in self.members})


@dataclass(frozen=True)
class StaticConsensus:
    """Result of reconciling one static class's candidates across a run's
    frames: the winning cluster's median box, each frame's assigned box
    (the consensus box itself, or None if that frame has no candidate
    agreeing with it), and how many frames supported the winning cluster."""

    consensus_box: Box | None
    per_frame_box: dict[str, Box | None]
    support: int


@dataclass(frozen=True)
class ContainerSelection:
    """Result of choosing one container box in a single frame: the chosen
    candidate (or None if nothing qualified), and every other candidate
    paired with why it did not win."""

    chosen: RawDetection | None
    rejected: list[tuple[RawDetection, str]]


def area_band_filter(
    detections: list[RawDetection], area_band: tuple[float, float], width: int, height: int
) -> list[RawDetection]:
    """Keep only candidates whose box-area fraction falls inside
    ``area_band`` (inclusive)."""
    lo, hi = area_band
    return [d for d in detections if lo <= box_area_frac(d.box, width, height) <= hi]


def cluster_by_iou(
    items: list[tuple[str, RawDetection]], iou_threshold: float = 0.5
) -> list[Cluster]:
    """Greedily group ``(frame_id, RawDetection)`` pairs into clusters whose
    members mutually overlap: each item joins the existing cluster whose
    current median box gives the highest IoU at or above ``iou_threshold``,
    or starts a new cluster if none qualifies. Deterministic given a fixed
    input order -- callers should sort ``items`` themselves."""
    clusters: list[list[tuple[str, RawDetection]]] = []
    for item in items:
        _, det = item
        best_idx: int | None = None
        best_iou = 0.0
        for idx, cluster in enumerate(clusters):
            rep = median_box([d.box for _, d in cluster])
            assert rep is not None
            score = iou(det.box, rep)
            if score >= iou_threshold and score > best_iou:
                best_iou = score
                best_idx = idx
        if best_idx is None:
            clusters.append([item])
        else:
            clusters[best_idx].append(item)
    return [Cluster(members=tuple(c)) for c in clusters]


def static_consensus(
    candidates_by_frame: dict[str, list[RawDetection]],
    area_band: tuple[float, float],
    width: int,
    height: int,
) -> StaticConsensus:
    """Reconcile one static class (``outer_box``, ``tray`` or
    ``start_button``) across a run's frames: cluster the in-band candidates
    by mutual IoU, take the cluster most frames agree on, and assign its
    median box to every frame with a candidate that agrees with it (IoU >=
    0.5) -- static-object smoothing per essential-features.md section 14
    Stage 3. A frame with no agreeing candidate is "missing" (None), never
    a guessed box."""
    filtered_by_frame = {
        frame_id: area_band_filter(dets, area_band, width, height)
        for frame_id, dets in candidates_by_frame.items()
    }
    items = [
        (frame_id, det)
        for frame_id in sorted(filtered_by_frame)
        for det in sorted(filtered_by_frame[frame_id], key=lambda d: d.score, reverse=True)
    ]
    if not items:
        return StaticConsensus(None, dict.fromkeys(candidates_by_frame), 0)

    clusters = cluster_by_iou(items, iou_threshold=0.5)
    winner = max(clusters, key=lambda c: (c.frame_support, len(c.members)))
    consensus_box = median_box([d.box for _, d in winner.members])

    per_frame: dict[str, Box | None] = {}
    for frame_id in candidates_by_frame:
        cands = filtered_by_frame.get(frame_id, [])
        if any(iou(d.box, consensus_box) >= 0.5 for d in cands):
            per_frame[frame_id] = consensus_box
        else:
            per_frame[frame_id] = None
    return StaticConsensus(consensus_box, per_frame, winner.frame_support)


def select_container(
    detections: list[RawDetection],
    area_band: tuple[float, float],
    width: int,
    height: int,
    exclude_boxes: dict[str, Box | None],
    exclude_iou_threshold: float = 0.5,
) -> ContainerSelection:
    """Choose one container box (``red_box`` / ``yellow_box``) in a single
    frame: drop candidates outside ``area_band``, drop any candidate
    overlapping (IoU > ``exclude_iou_threshold``) a named excluded box (the
    run's ``outer_box`` / ``tray`` / ``start_button`` consensus), then keep
    the highest-scoring survivor. Every non-winning candidate is reported
    with the reason it lost."""
    rejected: list[tuple[RawDetection, str]] = []
    valid: list[RawDetection] = []
    lo, hi = area_band
    for det in detections:
        frac = box_area_frac(det.box, width, height)
        if not (lo <= frac <= hi):
            rejected.append((det, "out_of_band"))
            continue
        excluded_reason = None
        for name, box in exclude_boxes.items():
            if box is not None and iou(det.box, box) > exclude_iou_threshold:
                excluded_reason = f"overlaps_{name}"
                break
        if excluded_reason is not None:
            rejected.append((det, excluded_reason))
            continue
        valid.append(det)

    if not valid:
        return ContainerSelection(None, rejected)

    chosen = max(valid, key=lambda d: d.score)
    for det in valid:
        if det is not chosen:
            rejected.append((det, "not_highest_score"))
    return ContainerSelection(chosen, rejected)


def central_box(box: Box, frac: float = 0.5) -> Box:
    """Shrink ``box`` to its central ``frac`` of width and height (frac=0.5
    trims 25% off each side), for measuring colour away from a box's own
    edges where it is most likely to include background or a neighbouring
    object."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    mw, mh = w * (1 - frac) / 2, h * (1 - frac) / 2
    return (x1 + mw, y1 + mh, x2 - mw, y2 - mh)


def mean_hue_sat_in_box(image_bgr: np.ndarray, box: Box) -> tuple[float, float] | None:
    """Mean OpenCV hue (0-179) and saturation (0-255) of the pixels inside
    ``box``, clipped to the image. None if the clipped box is empty."""
    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2 = box
    xi1, yi1 = max(0, int(round(x1))), max(0, int(round(y1)))
    xi2, yi2 = min(width, int(round(x2))), min(height, int(round(y2)))
    if xi2 <= xi1 or yi2 <= yi1:
        return None
    crop = image_bgr[yi1:yi2, xi1:xi2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 0])), float(np.mean(hsv[:, :, 1]))


def single_clean_candidate(
    per_frame_detections: dict[str, list[RawDetection]],
    area_band: tuple[float, float],
    outer_box: Box | None,
    width: int,
    height: int,
) -> dict[str, RawDetection]:
    """Frames where exactly one in-band, non-outer-box candidate exists for
    one colour class -- the only frames a clean hue/saturation measurement
    can be trusted from (checkpoint 1b item 2d), since any frame with zero
    or multiple survivors is ambiguous about which candidate is the real
    container."""
    result: dict[str, RawDetection] = {}
    for frame_id, dets in per_frame_detections.items():
        in_band = area_band_filter(dets, area_band, width, height)
        non_outer = [d for d in in_band if outer_box is None or iou(d.box, outer_box) <= 0.5]
        if len(non_outer) == 1:
            result[frame_id] = non_outer[0]
    return result
