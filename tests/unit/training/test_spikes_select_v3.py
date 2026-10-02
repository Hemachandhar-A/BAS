"""Unit tests for the pure P1.2 checkpoint-1c selection helpers
(training/spikes/select_v3.py): white-paper test, filtered static consensus,
non-circular exclusion boxes, tight container band, v2/v3 status diff."""

from __future__ import annotations

import numpy as np
import pytest

from training.spikes.postprocess import RawDetection
from training.spikes.select_v3 import (
    exclusion_boxes,
    is_white_paper,
    mean_sat_val_in_box,
    static_consensus_filtered,
    status_change,
    tight_band,
)

W, H = 848, 480
CARD = (480.0, 340.0, 600.0, 425.0)
CONTAINER = (200.0, 100.0, 365.0, 230.0)


def _det(box, score=0.5):
    return RawDetection(phrase="x", score=score, box=box)


# --- is_white_paper / mean_sat_val_in_box ------------------------------------


def test_white_paper_accepts_pale_bright():
    assert is_white_paper(20.0, 200.0, sat_max=63.0, val_min=130.0)


@pytest.mark.parametrize("sat,val", [(120.0, 230.0), (20.0, 90.0), (63.1, 200.0)])
def test_white_paper_rejects(sat, val):
    assert not is_white_paper(sat, val, sat_max=63.0, val_min=130.0)


def test_mean_sat_val_white_and_red():
    img = np.full((100, 100, 3), 255, dtype=np.uint8)  # BGR white
    s, v = mean_sat_val_in_box(img, (10, 10, 90, 90))
    assert s == pytest.approx(0.0) and v == pytest.approx(255.0)
    img[:] = (0, 0, 255)  # BGR red
    s, v = mean_sat_val_in_box(img, (10, 10, 90, 90))
    assert s == pytest.approx(255.0) and v == pytest.approx(255.0)


def test_mean_sat_val_uses_central_half_only():
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    img[:, :25] = (0, 0, 255)  # red strip outside the central 50% of the full-frame box
    s, _ = mean_sat_val_in_box(img, (0, 0, 100, 100))
    assert s == pytest.approx(0.0)


def test_mean_sat_val_empty_box_is_none():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    assert mean_sat_val_in_box(img, (50, 50, 60, 60)) is None


# --- static_consensus_filtered ----------------------------------------------


def test_filter_removes_rejected_candidate_before_clustering():
    cands = {f"f{i}": [_det(CONTAINER, 0.9), _det(CARD, 0.5)] for i in range(5)}
    res = static_consensus_filtered(cands, (0.001, 0.2), W, H, accept=lambda f, d: d.box == CARD)
    assert res.consensus_box == CARD
    assert all(b == CARD for b in res.per_frame_box.values())


def test_filter_everything_rejected_is_missing_not_invented():
    cands = {"f0": [_det(CONTAINER)], "f1": [_det(CONTAINER)]}
    res = static_consensus_filtered(cands, (0.001, 0.2), W, H, accept=lambda f, d: False)
    assert res.consensus_box is None
    assert res.per_frame_box == {"f0": None, "f1": None}


def test_filter_frame_without_passing_candidate_is_missing():
    cands = {"f0": [_det(CARD)], "f1": [_det(CONTAINER)], "f2": [_det(CARD)]}
    res = static_consensus_filtered(cands, (0.001, 0.2), W, H, accept=lambda f, d: d.box == CARD)
    assert res.per_frame_box["f1"] is None
    assert res.per_frame_box["f0"] == CARD


# --- exclusion_boxes (no circular rule) ---------------------------------------


def test_exclusion_drops_start_button_that_failed_white_test():
    cons = {"outer_box": (1, 2, 3, 4), "tray": (5, 6, 7, 8), "start_button": CONTAINER}
    ex = exclusion_boxes(cons, start_button_is_white=False)
    assert "start_button" not in ex
    assert ex["outer_box"] == (1, 2, 3, 4) and ex["tray"] == (5, 6, 7, 8)


def test_exclusion_keeps_start_button_that_passed():
    cons = {"outer_box": None, "tray": None, "start_button": CARD}
    assert exclusion_boxes(cons, start_button_is_white=True)["start_button"] == CARD


def test_exclusion_drops_start_button_without_measurement():
    cons = {"outer_box": None, "tray": None, "start_button": CARD}
    assert "start_button" not in exclusion_boxes(cons, start_button_is_white=None)


# --- tight_band ---------------------------------------------------------------


def test_tight_band_excludes_two_container_box():
    areas = [0.044, 0.046, 0.047, 0.05, 0.052, 0.055, 0.058, 0.065]
    lo, hi = tight_band(areas)
    assert lo < min(areas) and hi > 0.058
    assert hi < 0.12  # a box spanning both containers (about 0.12-0.13) must not pass


def test_tight_band_needs_data():
    with pytest.raises(ValueError):
        tight_band([])


# --- status_change ------------------------------------------------------------


def test_status_change_unchanged_is_none():
    assert status_change(CARD, CARD) is None
    assert status_change(None, None) is None


def test_status_change_missing_to_ok_and_back():
    assert status_change(None, CARD) == ("missing", "ok")
    assert status_change(CARD, None) == ("ok", "missing")


def test_status_change_moved_box():
    assert status_change(CONTAINER, CARD) == ("ok", "ok_moved")


def test_status_change_tiny_shift_is_unchanged():
    assert status_change(CARD, (481.0, 340.0, 600.0, 425.0)) is None


# --- pick_holdout_runs -----------------------------------------------------------


def test_pick_holdout_runs_train_only_unused_and_deterministic():
    from training.spikes.select_v3 import pick_holdout_runs

    rows = [{"run_id": f"x{i:03d}", "split": "train" if i % 2 else "val"} for i in range(1, 40)]
    used = {"x001", "x003", "x005"}
    a = pick_holdout_runs(rows, used, n_runs=5, seed=1)
    assert a == pick_holdout_runs(rows, used, n_runs=5, seed=1)
    assert len(a) == 5 and a == sorted(a)
    by_id = {r["run_id"]: r for r in rows}
    assert all(by_id[r]["split"] == "train" and r not in used for r in a)


def test_pick_holdout_runs_not_enough_runs_fails():
    from training.spikes.select_v3 import pick_holdout_runs

    rows = [{"run_id": "x001", "split": "train"}, {"run_id": "x002", "split": "test"}]
    with pytest.raises(ValueError):
        pick_holdout_runs(rows, set(), n_runs=2, seed=1)
