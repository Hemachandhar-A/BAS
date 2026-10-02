"""Unit tests for training/autolabel_rules.py: per-run static boxes (v3
consensus plus the white-paper test, applied to every frame of a run) and
per-frame movable boxes (rules v4), with fake measurements."""

from __future__ import annotations

import pytest

from training.autolabel_rules import (
    MOVABLE_CLASSES,
    STATIC_CLASSES,
    select_movable,
    select_statics,
    static_frame_ids,
    v3_equivalent_container_params,
)
from training.spikes.postprocess import RawDetection
from training.spikes.select_v2 import select_container

W, H = 848, 480
FRAME = W * H

OUTER = (60.0, 40.0, 700.0, 340.0)  # area 0.4717 of... scaled below in rules
TRAY = (250.0, 200.0, 520.0, 330.0)
CARD = (480.0, 340.0, 600.0, 425.0)
RED = (200.0, 100.0, 364.0, 218.0)


def _det(box, score=0.5):
    return RawDetection(phrase="p", score=score, box=box)


def _rules():
    return {
        "static": {
            "sat_max": 61.0,
            "val_min": 128.0,
            "static_bands": {
                "outer_box": [0.15, 0.60],
                "tray": [0.019, 0.188],
                "start_button": [0.001, 0.1076],
            },
        },
        "container": {
            "container_band": [0.0324, 0.0694],
            "area_ceiling": 0.092,
            "small_floor": 0.015,
            "width_max": 262.0,
            "height_max": 188.0,
            "colour": {
                "red_box": {"hue": [-1.0, 10.0], "sat": [85.0, 150.0]},
                "yellow_box": {"hue": [23.0, 39.0], "sat": [110.0, 165.0]},
            },
        },
    }


def _white(fid, box):
    return (20.0, 220.0)


def _red(box):
    return (2.0, 130.0)


# --- static_frame_ids ---------------------------------------------------------


def test_static_frame_ids_spread_across_the_run():
    ids = list(range(0, 600, 30))  # 20 frames
    chosen = static_frame_ids(ids, 5)
    assert chosen == [0, 150, 300, 420, 570]


def test_static_frame_ids_short_run_returns_all():
    assert static_frame_ids([0, 30, 60], 5) == [0, 30, 60]


def test_static_frame_ids_deterministic_and_unique():
    ids = list(range(0, 300, 30))
    assert static_frame_ids(ids, 5) == static_frame_ids(ids, 5)
    assert len(set(static_frame_ids(ids, 5))) == 5


# --- select_statics -------------------------------------------------------------


def _cands(frames, per_class):
    return {
        (f, c): [_det(b, s) for b, s in per_class.get(c, [])]
        for f in frames
        for c in STATIC_CLASSES
    }


def test_statics_consensus_box_over_frames():
    frames = [0, 30, 60, 90, 120]
    cands = _cands(
        frames,
        {"outer_box": [(OUTER, 0.6)], "tray": [(TRAY, 0.5)], "start_button": [(CARD, 0.5)]},
    )
    res = select_statics(frames, cands, W, H, _rules(), _white)
    assert res.boxes["outer_box"] == OUTER
    assert res.boxes["tray"] == TRAY
    assert res.boxes["start_button"] == CARD
    assert res.start_button_white is True


def test_statics_start_button_must_look_like_white_paper():
    frames = [0, 30, 60]
    # the "card" candidate is really the red container: saturated
    cands = _cands(frames, {"outer_box": [(OUTER, 0.6)], "start_button": [(RED, 0.5)]})
    res = select_statics(frames, cands, W, H, _rules(), lambda fid, box: (130.0, 230.0))
    assert res.boxes["start_button"] is None
    assert res.start_button_white is None


def test_statics_missing_class_is_none_not_invented():
    frames = [0, 30]
    res = select_statics(frames, _cands(frames, {}), W, H, _rules(), _white)
    assert all(res.boxes[c] is None for c in STATIC_CLASSES)


def test_statics_majority_over_frames_wins_over_a_stray_candidate():
    frames = [0, 30, 60, 90]
    stray = (10.0, 10.0, 600.0, 400.0)
    cands = {(f, c): [] for f in frames for c in STATIC_CLASSES}
    for f in frames:
        cands[(f, "outer_box")] = [_det(OUTER, 0.6)]
    cands[(30, "outer_box")].append(_det(stray, 0.9))
    res = select_statics(frames, cands, W, H, _rules(), _white)
    assert res.boxes["outer_box"] == OUTER


# --- select_movable -------------------------------------------------------------


def _statics(**over):
    boxes = {"outer_box": OUTER, "tray": TRAY, "start_button": CARD}
    boxes.update(over)
    from training.autolabel_rules import StaticResult

    return StaticResult(boxes=boxes, start_button_white=True, start_button_measure=(20.0, 220.0))


def test_movable_picks_a_container_and_reports_missing_as_none():
    sel = select_movable(
        {"red_box": [_det(RED, 0.7)], "yellow_box": []}, W, H, _rules(), _statics(), _red
    )
    assert sel["red_box"].chosen is not None and sel["red_box"].chosen.box == RED
    assert sel["yellow_box"].chosen is None


def test_movable_candidate_on_the_tray_is_excluded():
    sel = select_movable(
        {"red_box": [_det(TRAY, 0.9)], "yellow_box": []}, W, H, _rules(), _statics(), _red
    )
    assert sel["red_box"].chosen is None


def test_movable_start_card_exclusion_only_when_white_passed():
    from training.autolabel_rules import StaticResult

    not_white = StaticResult(
        boxes={"outer_box": OUTER, "tray": TRAY, "start_button": RED},
        start_button_white=False,
        start_button_measure=(130.0, 230.0),
    )
    sel = select_movable(
        {"red_box": [_det(RED, 0.9)], "yellow_box": []}, W, H, _rules(), not_white, _red
    )
    assert sel["red_box"].chosen is not None  # v3's non-circular rule


def test_classes_partition():
    assert set(STATIC_CLASSES) | set(MOVABLE_CLASSES) == {
        "outer_box",
        "tray",
        "start_button",
        "red_box",
        "yellow_box",
    }
    assert not set(STATIC_CLASSES) & set(MOVABLE_CLASSES)


# --- v3 as degenerate v4 ----------------------------------------------------------


@pytest.mark.parametrize(
    "areas",
    [[0.046], [0.0725], [0.0277], [0.050, 0.060], [0.135], [0.031, 0.040]],
)
def test_v3_equivalent_params_reproduce_v2_select_container(areas):
    rules = _rules()
    p = v3_equivalent_container_params(rules["container"]["container_band"])
    dets = []
    for i, a in enumerate(areas):
        w = 164.0
        h = a * FRAME / w
        cx = 150.0 + 190.0 * i
        dets.append(_det((cx - w / 2, 100.0, cx + w / 2, 100.0 + h), 0.9 - 0.1 * i))
    exclude = {"tray": TRAY}
    old = select_container(dets, tuple(p["container_band"]), W, H, exclude)
    rules["container"] = p
    new = select_movable({"red_box": dets, "yellow_box": []}, W, H, rules, _statics(), _red)[
        "red_box"
    ]
    assert (new.chosen is None) == (old.chosen is None)
    if old.chosen is not None:
        assert new.chosen.box == old.chosen.box
