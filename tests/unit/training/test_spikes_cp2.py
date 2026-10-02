"""Unit tests for the pure P1.2 checkpoint-2 helpers (training/spikes/cp2.py):
frame sampling, static-frame subset, PerceptionFrame assembly, deviation
comparison, fingertip-on-card test."""

from __future__ import annotations

import pytest

from contracts import ExpectedDeviation, Hand
from training.spikes.cp2 import (
    agreeing_median_score,
    compare_deviations,
    deviation_key,
    fingertip_in_box,
    make_perception_frame,
    sample_frame_ids,
    spread_subset,
    true_runs,
)
from training.spikes.postprocess import RawDetection


def test_sample_frame_ids_four_fps_from_thirty():
    ids = sample_frame_ids(frame_count=60, fps=30.0, target_fps=4.0)
    assert ids[0] == 0
    assert ids == sorted(set(ids))
    assert ids[1] in (7, 8)
    assert all(i < 60 for i in ids)
    assert len(ids) == 8  # 2 s at 4 fps


def test_sample_frame_ids_never_past_end():
    ids = sample_frame_ids(frame_count=10, fps=30.0, target_fps=4.0)
    assert max(ids) <= 9


def test_spread_subset_evenly_and_endpoints():
    assert spread_subset(list(range(10)), 5) == [0, 2, 4, 7, 9]
    assert spread_subset([1, 2], 5) == [1, 2]


def _hand(tip=(0.0, 0.0), score=0.9, others=(500.0, 500.0)):
    lms = [others] * 21
    lms[8] = tip
    return Hand(handedness="Right", score=score, landmarks_px=lms)


def test_fingertip_in_box_uses_index_tip_and_margin():
    box = (100.0, 100.0, 200.0, 200.0)
    assert fingertip_in_box(_hand(tip=(150, 150)), box, 0.0)
    assert not fingertip_in_box(_hand(tip=(205, 150)), box, 0.0)
    assert fingertip_in_box(_hand(tip=(205, 150)), box, 0.10)  # grown by 10 px


def test_fingertip_in_box_other_landmarks_do_not_count():
    box = (100.0, 100.0, 200.0, 200.0)
    h = _hand(tip=(0, 0), others=(150, 150))
    assert not fingertip_in_box(h, box, 0.0)


def test_make_perception_frame_t_and_labels():
    pf = make_perception_frame(
        frame_id=15,
        fps=30.0,
        detections={
            "red_box": ((10.0, 20.0, 30.0, 40.0), 0.55),
            "tray": None,
        },
        hands=[],
    )
    assert pf.frame_id == 15
    assert pf.t == pytest.approx(0.5)
    assert [d.label for d in pf.detections] == ["red_box"]
    assert pf.detections[0].conf == 0.55
    assert pf.pose is None


def test_make_perception_frame_clamps_conf():
    pf = make_perception_frame(0, 30.0, {"red_box": ((0, 0, 1, 1), 1.2)}, [])
    assert pf.detections[0].conf == 1.0


def test_agreeing_median_score():
    box = (100.0, 100.0, 200.0, 200.0)
    cands = {
        "a": [RawDetection("p", 0.4, (101, 101, 201, 201)), RawDetection("p", 0.9, (0, 0, 10, 10))],
        "b": [RawDetection("p", 0.6, (99, 99, 199, 199))],
    }
    assert agreeing_median_score(cands, box) == pytest.approx(0.5)
    assert agreeing_median_score({}, box) == 0.0


def test_deviation_key_omission_uses_skipped_ids():
    from contracts import EngineEvent

    ev = EngineEvent(
        t=1.0,
        kind="deviation_detected",
        step_id="start_pressed",
        deviation_type="omission",
        skipped_step_ids=["red_out", "red_in_tray"],
        confidence_tag="confirmed",
    )
    assert deviation_key(ev) == ("omission", ("red_out", "red_in_tray"))
    ev2 = EngineEvent(
        t=1.0,
        kind="deviation_detected",
        step_id="red_out",
        deviation_type="out_of_order",
        confidence_tag="confirmed",
    )
    assert deviation_key(ev2) == ("out_of_order", ("red_out",))


def test_compare_deviations_multiset():
    exp = [
        ExpectedDeviation(deviation_type="omission", step_ids=["a", "b"]),
        ExpectedDeviation(deviation_type="out_of_order", step_ids=["a"]),
    ]
    got = [("out_of_order", ("a",)), ("repeat", ("c",))]
    res = compare_deviations(exp, got)
    assert res["matched"] == [("out_of_order", ("a",))]
    assert res["missing"] == [("omission", ("a", "b"))]
    assert res["extra"] == [("repeat", ("c",))]


def test_true_runs_start_and_length():
    assert true_runs([False, True, True, False, True, False, True, True, True]) == [
        (1, 2),
        (4, 1),
        (6, 3),
    ]
    assert true_runs([]) == []
    assert true_runs([False, False]) == []
    assert true_runs([True, True]) == [(0, 2)]
