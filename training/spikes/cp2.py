"""Pure helpers for P1.2 checkpoint 2 (zero-shot detector + hands fed through
StateTracker / SequenceEngine). No model, no clock. Orchestration lives in
run_cp2.py."""

from __future__ import annotations

from collections import Counter

import numpy as np

from contracts import Detection, EngineEvent, ExpectedDeviation, Hand, PerceptionFrame
from training.spikes.postprocess import Box, RawDetection, iou

INDEX_FINGERTIP = 8  # MediaPipe hand landmark index


def sample_frame_ids(frame_count: int, fps: float, target_fps: float) -> list[int]:
    """Frame ids closest to a ``target_fps`` grid, strictly increasing, all
    inside the clip."""
    ids: list[int] = []
    k = 0
    while True:
        fid = int(round(k * fps / target_fps))
        if fid >= frame_count:
            break
        if not ids or fid > ids[-1]:
            ids.append(fid)
        k += 1
    return ids


def spread_subset(items: list, n: int) -> list:
    """``n`` items spread evenly over ``items`` (both endpoints included);
    all of them if there are fewer than ``n``."""
    if len(items) <= n:
        return list(items)
    idx = np.linspace(0, len(items) - 1, n).round().astype(int)
    return [items[i] for i in idx]


def make_perception_frame(
    frame_id: int,
    fps: float,
    detections: dict[str, tuple[Box, float] | None],
    hands: list[Hand],
) -> PerceptionFrame:
    """One contracts.PerceptionFrame; ``t = frame_id / fps`` as the contract
    defines it for a file. A class with ``None`` (missing) emits no
    Detection; confidence is clipped into [0, 1]."""
    dets = [
        Detection(label=label, conf=float(min(1.0, max(0.0, conf))), box=tuple(map(float, box)))
        for label, item in detections.items()
        if item is not None
        for box, conf in [item]
    ]
    return PerceptionFrame(frame_id=frame_id, t=frame_id / fps, detections=dets, hands=hands)


def agreeing_median_score(candidates_by_frame: dict[str, list[RawDetection]], box: Box) -> float:
    """Median score of the best candidate per frame that agrees (IoU >= 0.5)
    with ``box``; 0.0 if none does."""
    best = []
    for dets in candidates_by_frame.values():
        scores = [d.score for d in dets if iou(d.box, box) >= 0.5]
        if scores:
            best.append(max(scores))
    return float(np.median(best)) if best else 0.0


def fingertip_in_box(hand: Hand, box: Box, margin_frac: float) -> bool:
    """True iff the index fingertip lies in ``box`` grown by ``margin_frac``
    of its width/height on each side (same growth as the StateTracker)."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x, y = hand.landmarks_px[INDEX_FINGERTIP]
    mx, my = margin_frac * w, margin_frac * h
    return x1 - mx <= x <= x2 + mx and y1 - my <= y <= y2 + my


def deviation_key(event: EngineEvent) -> tuple[str, tuple[str, ...]]:
    """Same keying as ExpectedDeviation: omission -> skipped ids, else the
    observed step."""
    assert event.deviation_type is not None
    if event.deviation_type == "omission":
        return event.deviation_type, tuple(event.skipped_step_ids)
    assert event.step_id is not None
    return event.deviation_type, (event.step_id,)


def compare_deviations(
    expected: list[ExpectedDeviation], produced: list[tuple[str, tuple[str, ...]]]
) -> dict[str, list]:
    """Multiset comparison of expected vs produced (type, step_ids) keys."""
    exp = Counter((e.deviation_type, tuple(e.step_ids)) for e in expected)
    got = Counter(produced)
    return {
        "matched": sorted((exp & got).elements()),
        "missing": sorted((exp - got).elements()),
        "extra": sorted((got - exp).elements()),
    }


def true_runs(flags: list[bool]) -> list[tuple[int, int]]:
    """Maximal runs of consecutive True as (start_index, length)."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i - start))
            start = None
    if start is not None:
        runs.append((start, len(flags) - start))
    return runs
