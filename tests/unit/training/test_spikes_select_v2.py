"""Unit tests for the pure P1.2 checkpoint-1b selection helpers
(training/spikes/select_v2.py) -- no model, no I/O, operating on
training/spikes/postprocess.RawDetection values the raw candidate cache
produced (IMPLEMENTATION_PLAN.md Part 10, P1.2)."""

from __future__ import annotations

import numpy as np
import pytest

from training.spikes.postprocess import RawDetection
from training.spikes.select_v2 import (
    ContainerSelection,
    StaticConsensus,
    area_band_filter,
    central_box,
    cluster_by_iou,
    mean_hue_sat_in_box,
    select_container,
    single_clean_candidate,
    static_consensus,
)


def _det(score: float, box) -> RawDetection:
    return RawDetection(phrase="x", score=score, box=box)


# --- area_band_filter -------------------------------------------------


def test_area_band_filter_keeps_only_in_band_boxes():
    dets = [
        _det(0.9, (0, 0, 10, 10)),  # area_frac 0.01, in band
        _det(0.9, (0, 0, 90, 90)),  # area_frac 0.81, out of band
    ]
    kept = area_band_filter(dets, (0.005, 0.1), width=100, height=100)
    assert kept == [dets[0]]


def test_area_band_filter_of_empty_list_is_empty():
    assert area_band_filter([], (0.0, 1.0), width=100, height=100) == []


# --- cluster_by_iou -----------------------------------------------------


def test_cluster_by_iou_groups_overlapping_boxes_together():
    items = [
        ("f1", _det(0.9, (0, 0, 10, 10))),
        ("f2", _det(0.8, (1, 1, 11, 11))),  # high IoU with f1's box
        ("f3", _det(0.7, (100, 100, 110, 110))),  # disjoint
    ]
    clusters = cluster_by_iou(items, iou_threshold=0.5)
    assert len(clusters) == 2
    sizes = sorted(len(c.members) for c in clusters)
    assert sizes == [1, 2]


def test_cluster_by_iou_frame_support_counts_distinct_frames_not_candidates():
    # two candidates from the SAME frame both land in one cluster: support
    # must still be 1, not 2.
    items = [
        ("f1", _det(0.9, (0, 0, 10, 10))),
        ("f1", _det(0.5, (1, 1, 11, 11))),
    ]
    clusters = cluster_by_iou(items, iou_threshold=0.5)
    assert len(clusters) == 1
    assert clusters[0].frame_support == 1
    assert len(clusters[0].members) == 2


def test_cluster_by_iou_of_empty_list_is_empty():
    assert cluster_by_iou([], iou_threshold=0.5) == []


# --- static_consensus -----------------------------------------------------


def test_static_consensus_picks_the_majority_cluster_and_flags_the_outlier():
    # 4 frames agree on a box near (100,100)-(140,140); 1 frame only has a
    # candidate at a disjoint location (a false match) and must come back
    # "missing" rather than borrowing the consensus box.
    def agree_box(jitter):
        return (100 + jitter, 100 + jitter, 140 + jitter, 140 + jitter)

    candidates_by_frame = {
        "f1": [_det(0.9, agree_box(0))],
        "f2": [_det(0.8, agree_box(1))],
        "f3": [_det(0.8, agree_box(-1))],
        "f4": [_det(0.8, agree_box(2))],
        "f5": [_det(0.9, (400, 400, 440, 440))],
    }
    result = static_consensus(candidates_by_frame, area_band=(0.0, 1.0), width=1000, height=1000)
    assert isinstance(result, StaticConsensus)
    assert result.support == 4
    assert result.consensus_box is not None
    assert result.per_frame_box["f1"] is not None
    assert result.per_frame_box["f2"] is not None
    assert result.per_frame_box["f3"] is not None
    assert result.per_frame_box["f4"] is not None
    assert result.per_frame_box["f5"] is None
    # frames assigned the consensus box all get the SAME (median) box, not
    # their own raw candidate -- static-object smoothing per
    # essential-features.md section 14 Stage 3.
    assert result.per_frame_box["f1"] == result.per_frame_box["f2"] == result.consensus_box


def test_static_consensus_with_no_candidates_anywhere_is_all_missing():
    result = static_consensus({"f1": [], "f2": []}, area_band=(0.0, 1.0), width=100, height=100)
    assert result.consensus_box is None
    assert result.support == 0
    assert result.per_frame_box == {"f1": None, "f2": None}


def test_static_consensus_filters_by_area_band_before_clustering():
    # the only candidates are out of band -> no consensus, not a crash.
    candidates_by_frame = {"f1": [_det(0.9, (0, 0, 99, 99))]}  # area_frac 0.98
    result = static_consensus(candidates_by_frame, area_band=(0.0, 0.5), width=100, height=100)
    assert result.consensus_box is None
    assert result.per_frame_box == {"f1": None}


# --- select_container -----------------------------------------------------


def test_select_container_picks_highest_in_band_remaining_candidate():
    dets = [
        _det(0.4, (0, 0, 10, 10)),
        _det(0.9, (20, 20, 30, 30)),
    ]
    result = select_container(dets, area_band=(0.0, 1.0), width=100, height=100, exclude_boxes={})
    assert isinstance(result, ContainerSelection)
    assert result.chosen is dets[1]
    assert (dets[0], "not_highest_score") in result.rejected


def test_select_container_rejects_out_of_band_candidates():
    dets = [_det(0.9, (0, 0, 99, 99))]  # area_frac 0.98
    result = select_container(dets, area_band=(0.0, 0.5), width=100, height=100, exclude_boxes={})
    assert result.chosen is None
    assert result.rejected == [(dets[0], "out_of_band")]


def test_select_container_rejects_candidates_overlapping_excluded_boxes():
    outer = (0, 0, 50, 50)
    dets = [
        _det(0.9, (1, 1, 49, 49)),  # overlaps outer_box heavily
        _det(0.5, (60, 60, 70, 70)),  # clean, lower score
    ]
    result = select_container(
        dets,
        area_band=(0.0, 1.0),
        width=100,
        height=100,
        exclude_boxes={"outer_box": outer},
    )
    assert result.chosen is dets[1]
    assert (dets[0], "overlaps_outer_box") in result.rejected


def test_select_container_of_empty_list_is_none_chosen():
    result = select_container([], area_band=(0.0, 1.0), width=100, height=100, exclude_boxes={})
    assert result.chosen is None
    assert result.rejected == []


# --- central_box -----------------------------------------------------


def test_central_box_half_of_a_simple_box():
    # box is (0,0)-(20,20); central 50% trims 25% off each side -> (5,5)-(15,15)
    assert central_box((0.0, 0.0, 20.0, 20.0), frac=0.5) == pytest.approx((5.0, 5.0, 15.0, 15.0))


def test_central_box_frac_one_is_unchanged():
    box = (1.0, 2.0, 11.0, 22.0)
    assert central_box(box, frac=1.0) == pytest.approx(box)


# --- mean_hue_sat_in_box -----------------------------------------------------


def test_mean_hue_sat_in_box_reads_pure_red_hue_and_full_saturation():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[:, :] = (0, 0, 255)  # BGR pure red
    result = mean_hue_sat_in_box(image, (0, 0, 20, 20))
    assert result is not None
    hue, sat = result
    assert hue == pytest.approx(0.0, abs=1.0)
    assert sat == pytest.approx(255.0, abs=1.0)


def test_mean_hue_sat_in_box_empty_clip_is_none():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    assert mean_hue_sat_in_box(image, (25, 25, 30, 30)) is None


# --- single_clean_candidate -----------------------------------------------------


def test_single_clean_candidate_keeps_frames_with_exactly_one_in_band_non_outer_candidate():
    outer = (0, 0, 50, 50)
    per_frame = {
        # exactly one clean candidate -> kept
        "f1": [_det(0.9, (60, 60, 70, 70))],
        # two clean candidates -> ambiguous, excluded
        "f2": [_det(0.9, (60, 60, 70, 70)), _det(0.8, (80, 80, 90, 90))],
        # only an outer-overlapping candidate -> zero clean, excluded
        "f3": [_det(0.9, (1, 1, 49, 49))],
        # no candidates at all -> excluded
        "f4": [],
    }
    result = single_clean_candidate(
        per_frame, area_band=(0.0, 1.0), outer_box=outer, width=100, height=100
    )
    assert set(result) == {"f1"}
    assert result["f1"].box == (60, 60, 70, 70)


def test_single_clean_candidate_with_no_outer_box_only_applies_area_band():
    per_frame = {"f1": [_det(0.9, (10, 10, 20, 20))]}
    result = single_clean_candidate(
        per_frame, area_band=(0.0, 1.0), outer_box=None, width=100, height=100
    )
    assert set(result) == {"f1"}
