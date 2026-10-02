"""Unit tests for training/spikes/select_v4.py (rules v4 = v3 plus an area
ceiling, a colour-verified smaller floor and a width/height cap), pure code."""

from __future__ import annotations

import numpy as np
import pytest

from training.spikes.postprocess import RawDetection
from training.spikes.select_v4 import (
    CEILING_HARD_MAX,
    CleanCell,
    compare_to_v3,
    derive_v4,
    in_colour_range,
    patch_hue_sat,
    reintroduced_wrong,
    select_container_v4,
    signed_hue,
)

W, H = 848, 480
FRAME = W * H


def _box(area_frac, w_px=164.0, cx=300.0, cy=240.0):
    h_px = area_frac * FRAME / w_px
    return (cx - w_px / 2, cy - h_px / 2, cx + w_px / 2, cy + h_px / 2)


def _det(box, score=0.5):
    return RawDetection(phrase="p", score=score, box=box)


def _params(**over):
    p = {
        "container_band": [0.0324, 0.0694],
        "area_ceiling": 0.092,
        "small_floor": 0.015,
        "width_max": 262.0,
        "height_max": 188.0,
        "colour": {
            "red_box": {"hue": [-1.0, 10.0], "sat": [85.0, 150.0]},
            "yellow_box": {"hue": [23.0, 39.0], "sat": [110.0, 165.0]},
        },
    }
    p.update(over)
    return p


def _measure_const(hue, sat):
    return lambda box: (hue, sat)


# --- colour helpers -----------------------------------------------------------


@pytest.mark.parametrize(
    "h,expected", [(0, 0), (10, 10), (89, 89), (90, -90), (175, -5), (179, -1)]
)
def test_signed_hue_wraps_red(h, expected):
    assert signed_hue(h) == expected


def test_in_colour_range_handles_red_wraparound():
    rng = {"hue": [-3.0, 9.0], "sat": [90.0, 150.0]}
    assert in_colour_range(2.0, 120.0, rng)
    assert in_colour_range(177.0, 120.0, rng)  # signed -3
    assert not in_colour_range(14.0, 120.0, rng)  # skin-ish hue
    assert not in_colour_range(2.0, 60.0, rng)  # too pale
    assert not in_colour_range(2.0, 200.0, rng)


def _solid(bgr, size=(300, 400)):
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    img[:] = bgr
    return img


def test_patch_hue_sat_red_and_lime_bgr_order():
    red = patch_hue_sat(_solid((0, 0, 255)), (50.0, 50.0, 250.0, 150.0))
    lime = patch_hue_sat(_solid((0, 255, 128)), (50.0, 50.0, 250.0, 150.0))
    assert red is not None and lime is not None
    assert signed_hue(red[0]) == pytest.approx(0.0, abs=1.0)
    assert red[1] == pytest.approx(255, abs=1)
    assert 30 <= lime[0] <= 50


def test_patch_hue_sat_red_patch_straddling_zero_is_circular():
    img = _solid((0, 0, 255))
    # half of the pixels slightly magenta-red (hue ~175), half slightly orange (hue ~5)
    img[:, :200] = (40, 0, 255)
    img[:, 200:] = (0, 40, 255)
    hue, _ = patch_hue_sat(img, (0.0, 0.0, 400.0, 300.0), frac=1.0)
    assert abs(signed_hue(hue)) < 3.0  # a plain mean would give ~90


def test_patch_hue_sat_empty_region_is_none():
    assert patch_hue_sat(_solid((0, 0, 255)), (500.0, 500.0, 600.0, 600.0)) is None


# --- derive_v4 ----------------------------------------------------------------


def _clean(n=40):
    rng = np.random.default_rng(0)
    cells = []
    for i in range(n):
        cls = "red_box" if i % 2 == 0 else "yellow_box"
        hue = rng.normal(2.0, 2.0) if cls == "red_box" else rng.normal(35.0, 2.0)
        cells.append(
            CleanCell(
                cls=cls,
                area=0.046 + 0.002 * (i % 5 - 2),
                width=164.0 + (i % 3),
                height=117.0 + (i % 4),
                hue=float(hue),
                sat=float(rng.normal(130.0, 10.0)),
            )
        )
    return cells


def test_derive_uses_medians_and_documented_factors():
    cells = _clean()
    p = derive_v4(cells, v3_band=(0.0324, 0.0694))
    areas = [c.area for c in cells]
    assert p["area_median"] == pytest.approx(float(np.median(areas)))
    assert p["area_ceiling"] == pytest.approx(2.0 * float(np.median(areas)))
    assert p["width_max"] == pytest.approx(1.6 * float(np.median([c.width for c in cells])))
    assert p["height_max"] == pytest.approx(1.6 * float(np.median([c.height for c in cells])))
    assert p["small_floor"] == 0.015
    assert p["container_band"] == [0.0324, 0.0694]
    assert p["n_clean"] == len(cells)


def test_derive_colour_ranges_cover_the_clean_cells_per_class():
    cells = _clean(80)
    p = derive_v4(cells, v3_band=(0.0324, 0.0694))
    for cls in ("red_box", "yellow_box"):
        rng = p["colour"][cls]
        own = [c for c in cells if c.cls == cls]
        inside = sum(in_colour_range(c.hue, c.sat, rng) for c in own)
        assert inside >= 0.9 * len(own)
    # the red range must not swallow the lime range and vice versa
    assert not in_colour_range(35.0, 130.0, p["colour"]["red_box"])
    assert not in_colour_range(2.0, 130.0, p["colour"]["yellow_box"])


def test_derive_refuses_a_ceiling_that_would_admit_two_containers():
    cells = [CleanCell("red_box", 0.07, 160.0, 120.0, 2.0, 130.0)] * 10
    assert 2 * 0.07 > CEILING_HARD_MAX
    with pytest.raises(ValueError, match="ceiling"):
        derive_v4(cells, v3_band=(0.0324, 0.0694))


def test_derive_needs_cells():
    with pytest.raises(ValueError):
        derive_v4([], v3_band=(0.0324, 0.0694))


# --- select_container_v4 ---------------------------------------------------------


def _sel(dets, cls="red_box", exclude=None, measure=None, params=None):
    return select_container_v4(
        dets,
        params or _params(),
        cls,
        W,
        H,
        exclude or {},
        measure or _measure_const(2.0, 120.0),
    )


def test_in_band_candidate_behaves_as_v3():
    d = _det(_box(0.046), 0.6)
    assert _sel([d]).chosen is d


def test_ceiling_admits_a_rotated_larger_box_but_not_two_containers():
    rotated = _det(_box(0.0735, w_px=200.0), 0.66)
    merged = _det(_box(0.135, w_px=200.0), 0.7)
    sel = _sel([rotated, merged])
    assert sel.chosen is rotated
    assert any(r == "out_of_band" and d is merged for d, r in sel.rejected)


def test_above_ceiling_is_out_of_band():
    assert _sel([_det(_box(0.095, w_px=200.0))]).chosen is None


def test_width_cap_rejects_a_stretched_candidate():
    stretched = _det(_box(0.0600, w_px=440.0), 0.9)  # 2.7 x the container width
    sel = _sel([stretched])
    assert sel.chosen is None
    assert sel.rejected[0][1] == "too_wide"


def test_height_cap_rejects_a_tall_candidate():
    tall = _det(_box(0.0600, w_px=120.0), 0.9)  # height 0.06*FRAME/120 = 203 px
    sel = _sel([tall])
    assert sel.chosen is None
    assert sel.rejected[0][1] == "too_tall"


def test_width_cap_applies_to_in_band_candidates_too():
    wide_in_band = _det(_box(0.0690, w_px=300.0), 0.9)
    right = _det(_box(0.046), 0.4)
    assert _sel([wide_in_band, right]).chosen is right


def test_small_candidate_needs_the_class_colour():
    small = _det(_box(0.0277, w_px=110.0), 0.67)
    assert _sel([small], "yellow_box", measure=_measure_const(29.6, 137.0)).chosen is small
    skin = _sel([small], "yellow_box", measure=_measure_const(14.0, 90.0))
    assert skin.chosen is None
    assert skin.rejected[0][1] == "small_wrong_colour"


def test_small_candidate_uses_the_red_range_for_red_box():
    small = _det(_box(0.025, w_px=100.0), 0.5)
    assert _sel([small], "red_box", measure=_measure_const(2.0, 120.0)).chosen is small
    assert _sel([small], "red_box", measure=_measure_const(35.0, 140.0)).chosen is None


def test_small_candidate_with_no_measurable_patch_is_rejected():
    small = _det(_box(0.025, w_px=100.0), 0.5)
    assert _sel([small], measure=lambda box: None).chosen is None


def test_below_the_small_floor_is_out_of_band():
    tiny = _det(_box(0.010, w_px=60.0), 0.9)
    sel = _sel([tiny])
    assert sel.chosen is None and sel.rejected[0][1] == "out_of_band"


def test_exclusion_boxes_still_apply():
    d = _det(_box(0.046), 0.6)
    assert _sel([d], exclude={"tray": d.box}).chosen is None


def test_highest_score_wins_among_valid_and_others_are_reported():
    a, b = _det(_box(0.046), 0.4), _det(_box(0.060, cx=600.0), 0.8)
    sel = _sel([a, b])
    assert sel.chosen is b
    assert (a, "not_highest_score") in sel.rejected


def test_nothing_is_invented():
    assert _sel([]).chosen is None


# --- compare_to_v3 (the safeguard) -----------------------------------------------


def test_compare_lists_changes_split_by_whether_the_lead_marked_the_cell_bad():
    b1 = (100.0, 100.0, 260.0, 220.0)
    b1_shifted = (140.0, 100.0, 300.0, 220.0)
    v3 = {
        ("x1", 1, "red_box"): b1,
        ("x1", 2, "red_box"): b1,
        ("x1", 3, "red_box"): None,
        ("x1", 4, "red_box"): None,
        ("x1", 5, "red_box"): b1,
    }
    v4 = {
        ("x1", 1, "red_box"): b1,  # unchanged
        ("x1", 2, "red_box"): b1_shifted,  # moved, not bad -> unmarked change
        ("x1", 3, "red_box"): b1,  # appeared, marked bad -> allowed
        ("x1", 4, "red_box"): None,  # unchanged
        ("x1", 5, "red_box"): None,  # disappeared, not bad -> unmarked change
    }
    bad = {("x1", 3, "red_box")}
    res = compare_to_v3(v3, v4, bad)
    assert res["changed_in_bad_cells"] == [("x1", 3, "red_box")]
    assert res["changed_in_unmarked_cells"] == [("x1", 2, "red_box"), ("x1", 5, "red_box")]


def test_compare_ignores_changes_above_iou_0_9():
    b = (100.0, 100.0, 260.0, 220.0)
    nudged = (101.0, 100.0, 261.0, 220.0)
    res = compare_to_v3({("x", 1, "red_box"): b}, {("x", 1, "red_box"): nudged}, set())
    assert res["changed_in_bad_cells"] == [] and res["changed_in_unmarked_cells"] == []


# --- reintroduced_wrong -------------------------------------------------------------


def test_reintroduced_wrong_flags_a_v4_box_the_lead_already_called_wrong():
    wrong = (275.0, 185.0, 485.0, 315.0)
    cell = ("x034", 156, "yellow_box")
    other = ("x003", 416, "yellow_box")
    v4 = {cell: (276.0, 186.0, 486.0, 316.0), other: (469.0, 196.0, 572.0, 305.0)}
    assert reintroduced_wrong(v4, [cell, other], {cell: [wrong]}) == [cell]


def test_reintroduced_wrong_ignores_a_different_box_and_missing_boxes():
    cell = ("x034", 156, "yellow_box")
    v4 = {cell: (400.0, 100.0, 560.0, 230.0)}
    assert reintroduced_wrong(v4, [cell], {cell: [(275.0, 185.0, 485.0, 315.0)]}) == []
    assert reintroduced_wrong({cell: None}, [cell], {cell: [(275.0, 185.0, 485.0, 315.0)]}) == []
