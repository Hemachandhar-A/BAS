"""Unit tests for the pure P1.2-spike post-processing helpers
(training/spikes/postprocess.py) -- no model, no I/O, per the spike's own
scope (IMPLEMENTATION_PLAN.md Part 10, P1.2)."""

from __future__ import annotations

import numpy as np
import pytest

from training.spikes.postprocess import (
    RawDetection,
    box_area_frac,
    iou,
    mean_hue_in_box,
    median_box,
    select_best_in_band,
)


def test_box_area_frac_of_the_whole_frame_is_one():
    assert box_area_frac((0, 0, 10, 10), width=10, height=10) == pytest.approx(1.0)


def test_box_area_frac_of_a_quarter_frame():
    assert box_area_frac((0, 0, 5, 5), width=10, height=10) == pytest.approx(0.25)


def test_box_area_frac_zero_area_frame_is_zero():
    assert box_area_frac((0, 0, 1, 1), width=0, height=0) == 0.0


def test_iou_identical_boxes_is_one():
    box = (10.0, 10.0, 30.0, 30.0)
    assert iou(box, box) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_half_overlap():
    a = (0.0, 0.0, 10.0, 10.0)
    b = (5.0, 0.0, 15.0, 10.0)
    # intersection 5x10=50, union 100+100-50=150
    assert iou(a, b) == pytest.approx(50 / 150)


def test_median_box_of_empty_list_is_none():
    assert median_box([]) is None


def test_median_box_of_one_box_is_itself():
    box = (1.0, 2.0, 3.0, 4.0)
    assert median_box([box]) == box


def test_median_box_averages_per_coordinate():
    boxes = [(0.0, 0.0, 10.0, 10.0), (2.0, 2.0, 12.0, 12.0), (4.0, 4.0, 14.0, 14.0)]
    assert median_box(boxes) == (2.0, 2.0, 12.0, 12.0)


def test_mean_hue_in_box_reads_pure_red_hue():
    # OpenCV BGR pure red -> HSV hue 0.
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[:, :] = (0, 0, 255)  # BGR red
    hue = mean_hue_in_box(image, (0, 0, 20, 20))
    assert hue == pytest.approx(0.0, abs=1.0)


def test_mean_hue_in_box_reads_pure_yellow_hue():
    # OpenCV BGR pure yellow (B=0,G=255,R=255) -> HSV hue ~30 (0-179 scale).
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[:, :] = (0, 255, 255)
    hue = mean_hue_in_box(image, (0, 0, 20, 20))
    assert hue == pytest.approx(30.0, abs=1.0)


def test_mean_hue_in_box_empty_clip_is_none():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    assert mean_hue_in_box(image, (25, 25, 30, 30)) is None
    assert mean_hue_in_box(image, (5, 5, 5, 10)) is None


def test_select_best_in_band_keeps_highest_score():
    # phrase text deliberately has no trailing period, matching what
    # Grounding DINO actually returns (it drops the query's own period).
    detections = [
        RawDetection(phrase="a red box", score=0.4, box=(0, 0, 10, 10)),
        RawDetection(phrase="a red container", score=0.7, box=(0, 0, 12, 12)),
    ]
    best = select_best_in_band(detections, area_band=None, width=100, height=100)
    assert best is not None
    assert best.score == pytest.approx(0.7)
    assert best.phrase == "a red container"


def test_select_best_in_band_drops_boxes_outside_area_band():
    # box covers the whole 100x100 frame (area_frac=1.0), band caps at 0.5.
    detections = [RawDetection(phrase="a red box", score=0.9, box=(0, 0, 100, 100))]
    best = select_best_in_band(detections, area_band=(0.0, 0.5), width=100, height=100)
    assert best is None


def test_select_best_in_band_of_empty_list_is_none():
    assert select_best_in_band([], area_band=None, width=100, height=100) is None


def test_select_best_in_band_with_no_band_keeps_every_candidate():
    detections = [
        RawDetection(phrase="a red box", score=0.1, box=(0, 0, 99, 99)),
        RawDetection(phrase="red", score=0.05, box=(0, 0, 1, 1)),
    ]
    best = select_best_in_band(detections, area_band=None, width=100, height=100)
    assert best is not None
    assert best.score == pytest.approx(0.1)
