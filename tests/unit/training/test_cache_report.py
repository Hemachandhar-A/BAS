"""P1.7.2: training/cache_report.py -- per-class detection statistics over perception caches and
the aggregation of a build into reports/cache_build.json. Pure helpers only; no labels are read
(statistics on val and test are descriptive, never an accuracy)."""

from __future__ import annotations

import pytest

from contracts import Detection, PerceptionFrame
from training import cache_report as R

pytestmark = pytest.mark.F14

CLASSES = ["outer_box", "tray", "red_box"]


def fr(i, *dets):
    return PerceptionFrame(
        frame_id=i,
        t=i / 10,
        detections=[Detection(label=lab, conf=c, box=(0, 0, 1, 1)) for lab, c in dets],
    )


def test_class_stats_counts_frames_with_a_detection_at_or_above_the_floor():
    frames = [
        fr(0, ("tray", 0.9), ("tray", 0.4), ("red_box", 0.2)),
        fr(1, ("tray", 0.30)),
        fr(2, ("red_box", 0.29)),
        fr(3),
    ]
    s = R.class_stats(frames, CLASSES, floor=0.30)
    assert s["n_frames"] == 4
    assert s["per_class"]["tray"]["frames_with_detection"] == 2
    assert s["per_class"]["tray"]["fraction_of_frames"] == 0.5
    assert s["per_class"]["tray"]["n_detections_at_floor"] == 3
    assert s["per_class"]["red_box"]["frames_with_detection"] == 0
    assert s["per_class"]["outer_box"]["frames_with_detection"] == 0


def test_confidence_quantiles_are_of_the_best_detection_per_frame_above_the_floor():
    frames = [fr(i, ("tray", c)) for i, c in enumerate([0.5, 0.6, 0.7, 0.8, 0.9])]
    stats = R.class_stats(frames, CLASSES, 0.3)["per_class"]
    assert stats["tray"]["best_conf_quantiles"] == {
        "min": 0.5, "q25": 0.6, "median": 0.7, "q75": 0.8, "max": 0.9,
    }  # fmt: skip
    # a class never seen has no quantiles (null), not zeros
    assert stats["red_box"]["best_conf_quantiles"] is None


def test_class_stats_over_no_frames_does_not_divide_by_zero():
    s = R.class_stats([], CLASSES, 0.3)
    assert s["n_frames"] == 0 and s["per_class"]["tray"]["fraction_of_frames"] is None


def test_split_summary_adds_runs_frames_and_seconds_per_split():
    results = [
        {"run_id": "a", "status": "built", "frames": 10, "seconds": 2.0},
        {"run_id": "b", "status": "built", "frames": 5, "seconds": 1.0},
        {"run_id": "c", "status": "built", "frames": 7, "seconds": 3.0},
        {"run_id": "d", "status": "failed", "error": "x"},
    ]
    splits = {"a": "train", "b": "train", "c": "val", "d": "val"}
    out = R.split_summary(results, splits)
    assert out["train"] == {"runs": 2, "frames": 15, "seconds": 3.0}
    assert out["val"] == {"runs": 1, "frames": 7, "seconds": 3.0}
