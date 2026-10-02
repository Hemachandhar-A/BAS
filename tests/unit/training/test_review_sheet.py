"""Unit tests for training/review_sheet.py: the seeded stratified review draw
(train runs only), flagged-frame pick, sheet layout, pre-filled review CSVs,
the strict CSV loader and the per-class bad fraction."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from training.review_sheet import (
    CSV_HEADER,
    ReviewCsvError,
    bad_fractions,
    draw_review_sample,
    load_label_review,
    pick_flagged,
    prefill_rows,
    render_sheets,
    write_review_csv,
)

CLASSES = ["outer_box", "tray", "red_box", "yellow_box", "start_button"]


def _index(n_train=6, n_val=2, frames=12):
    idx = {}
    for i in range(n_train):
        idx[f"x{i:03d}"] = {"split": "train", "frame_ids": [f * 30 for f in range(frames)]}
    for i in range(n_val):
        idx[f"y{i:03d}"] = {"split": "val", "frame_ids": [f * 30 for f in range(frames)]}
    return idx


# --- draw_review_sample -------------------------------------------------------


def test_sample_is_train_only_seeded_and_has_the_requested_size():
    s = draw_review_sample(_index(), n=20, per_run_min=2, seed=7)
    assert len(s) == 20 and len(set(s)) == 20
    assert all(run.startswith("x") for run, _ in s)  # no val run
    assert s == draw_review_sample(_index(), n=20, per_run_min=2, seed=7)
    assert s != draw_review_sample(_index(), n=20, per_run_min=2, seed=8)
    assert s == sorted(s)


def test_sample_has_at_least_the_minimum_per_run():
    s = draw_review_sample(_index(n_train=6), n=14, per_run_min=2, seed=1)
    per_run = {r: sum(1 for x, _ in s if x == r) for r in {r for r, _ in s}}
    assert len(per_run) == 6 and min(per_run.values()) >= 2


def test_sample_frames_come_from_the_index():
    idx = _index()
    for run, fid in draw_review_sample(idx, n=20, per_run_min=2, seed=3):
        assert fid in idx[run]["frame_ids"]


def test_sample_refuses_a_size_below_the_minimum_coverage():
    with pytest.raises(ValueError):
        draw_review_sample(_index(n_train=6), n=10, per_run_min=2, seed=0)


def test_sample_refuses_more_than_the_pool():
    with pytest.raises(ValueError):
        draw_review_sample(_index(n_train=2, frames=3), n=50, per_run_min=1, seed=0)


# --- pick_flagged -----------------------------------------------------------------


def test_pick_flagged_takes_highest_scores_with_a_per_run_cap_and_skips_excluded():
    scores = {
        ("a", 0): 0.9,
        ("a", 1): 0.8,
        ("a", 2): 0.7,
        ("b", 0): 0.6,
        ("b", 1): 0.5,
        ("c", 0): 0.4,
    }
    got = pick_flagged(scores, exclude={("a", 0)}, n=4, max_per_run=2)
    assert got == [("a", 1), ("a", 2), ("b", 0), ("b", 1)]


def test_pick_flagged_is_deterministic_on_ties():
    scores = {("b", 3): 0.5, ("a", 7): 0.5, ("a", 2): 0.5}
    assert pick_flagged(scores, set(), 3, 3) == pick_flagged(
        dict(reversed(scores.items())), set(), 3, 3
    )


def test_pick_flagged_returns_fewer_when_the_pool_is_small():
    assert pick_flagged({("a", 0): 1.0}, set(), 20, 2) == [("a", 0)]


# --- render_sheets ---------------------------------------------------------------------


def _tile(tag, boxes=None):
    return tag, np.full((480, 848, 3), 90, dtype=np.uint8), boxes or {}


def test_render_sheets_makes_twelve_per_sheet():
    tiles = [_tile(f"x001#{i}") for i in range(25)]
    sheets = render_sheets(tiles, CLASSES, per_sheet=12)
    assert len(sheets) == 3
    assert all(s.ndim == 3 and s.dtype == np.uint8 for s in sheets)
    assert sheets[0].shape == sheets[1].shape  # fixed layout, short last sheet padded


def test_render_sheets_draws_boxes_in_fixed_class_colours():
    box = (100.0, 100.0, 300.0, 250.0)
    plain = render_sheets([_tile("x001#0")], CLASSES)[0]
    with_box = render_sheets([_tile("x001#0", {"red_box": box})], CLASSES)[0]
    assert np.any(plain != with_box)
    other = render_sheets([_tile("x001#0", {"yellow_box": box})], CLASSES)[0]
    assert np.any(with_box != other)  # the two classes use different colours


def test_render_sheets_is_deterministic():
    tiles = [_tile("x001#0", {"tray": (10.0, 10.0, 200.0, 200.0)})]
    assert np.array_equal(render_sheets(tiles, CLASSES)[0], render_sheets(tiles, CLASSES)[0])


# --- prefill_rows / write / load ------------------------------------------------------------


def _cells():
    return [("x001", 0), ("x001", 30), ("x002", 60)]


def test_prefill_has_a_verdict_ok_row_for_every_cell_and_class():
    rows = prefill_rows(_cells(), CLASSES, boxes_found={})
    assert len(rows) == 3 * 5
    assert all(r[3] == "ok" and r[4] == "" and r[5] == "" for r in rows)
    assert [r[2] for r in rows[:5]] == CLASSES


def test_prefill_notes_the_cells_with_no_box_drawn():
    found = {("x001", 0): {"outer_box", "tray", "red_box", "start_button"}}
    rows = prefill_rows([("x001", 0)], CLASSES, boxes_found=found)
    yellow = next(r for r in rows if r[2] == "yellow_box")
    assert "no box" in yellow[6]
    assert all(r[6] == "" for r in rows if r[2] != "yellow_box")


def _write(tmp_path, rows):
    p = tmp_path / "label_review.csv"
    write_review_csv(p, rows)
    return p


def _valid_keys():
    return {(r, f, c) for r, f in _cells() for c in CLASSES}


def test_written_prefill_loads_back_clean(tmp_path):
    p = _write(tmp_path, prefill_rows(_cells(), CLASSES, {}))
    rows = load_label_review(p, CLASSES, _valid_keys())
    assert len(rows) == 15 and all(r.verdict == "ok" for r in rows)
    with p.open(newline="", encoding="utf-8") as f:
        assert next(csv.reader(f)) == CSV_HEADER


def _edit(tmp_path, mutate):
    rows = prefill_rows(_cells(), CLASSES, {})
    mutate(rows)
    return _write(tmp_path, rows)


@pytest.mark.parametrize(
    "mutate,fragment",
    [
        (lambda rows: rows[0].__setitem__(2, "blue_box"), "unknown class"),
        (lambda rows: rows[0].__setitem__(3, "maybe"), "verdict"),
        (lambda rows: rows[0].__setitem__(3, "bad"), "reason"),  # bad needs a reason
        (lambda rows: rows[0].__setitem__(4, "wrong_box"), "reason"),  # ok must not have one
        (
            lambda rows: (rows[0].__setitem__(3, "bad"), rows[0].__setitem__(4, "typo")),
            "reason",
        ),
        (
            lambda rows: (rows[0].__setitem__(3, "bad"), rows[0].__setitem__(4, "missing")),
            "visibility",
        ),
        (
            lambda rows: (
                rows[0].__setitem__(3, "bad"),
                rows[0].__setitem__(4, "missing"),
                rows[0].__setitem__(5, "gone"),
            ),
            "visibility",
        ),
        (lambda rows: rows[0].__setitem__(0, "x001 "), "whitespace"),
        (lambda rows: rows[0].__setitem__(1, "abc"), "frame_id"),
        (lambda rows: rows.append(list(rows[0])), "duplicate"),
        (lambda rows: rows.append(["x999", 0, "tray", "ok", "", "", ""]), "not in the sample"),
    ],
)
def test_loader_rejects_bad_rows(tmp_path, mutate, fragment):
    p = _edit(tmp_path, mutate)
    with pytest.raises(ReviewCsvError, match=fragment):
        load_label_review(p, CLASSES, _valid_keys())


def test_loader_accepts_each_valid_bad_row(tmp_path):
    def mutate(rows):
        rows[0][3:6] = ["bad", "wrong_box", ""]
        rows[1][3:6] = ["bad", "missing", "visible"]
        rows[2][3:6] = ["bad", "missing", "hidden"]
        rows[3][3:6] = ["bad", "duplicate", ""]
        rows[4][3:6] = ["bad", "wrong_class", ""]

    rows = load_label_review(_edit(tmp_path, mutate), CLASSES, _valid_keys())
    assert [r.reason for r in rows[:5]] == [
        "wrong_box",
        "missing",
        "missing",
        "duplicate",
        "wrong_class",
    ]


def test_loader_header_must_match(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("run_id,frame_id,class,verdict,reason\nx001,0,tray,ok,\n", encoding="utf-8")
    with pytest.raises(ReviewCsvError, match="header"):
        load_label_review(p, CLASSES, _valid_keys())


def test_loader_reports_every_problem(tmp_path):
    def mutate(rows):
        rows[0][3] = "maybe"
        rows[1][2] = "blue_box"

    with pytest.raises(ReviewCsvError) as e:
        load_label_review(_edit(tmp_path, mutate), CLASSES, _valid_keys())
    assert "verdict" in str(e.value) and "unknown class" in str(e.value)


# --- bad_fractions ---------------------------------------------------------


def test_bad_fraction_per_class_excludes_hidden_missing_and_applies_the_gate(tmp_path):
    cells = [("x001", i * 30) for i in range(20)]
    rows = prefill_rows(cells, CLASSES, {})
    # 2 bad red (10 %), 3 bad yellow (15 %), one hidden-missing tray (not bad)
    n = {c: 0 for c in CLASSES}
    for r in rows:
        if r[2] == "red_box" and n["red_box"] < 2:
            r[3:6] = ["bad", "wrong_box", ""]
            n["red_box"] += 1
        elif r[2] == "yellow_box" and n["yellow_box"] < 3:
            r[3:6] = ["bad", "missing", "visible"]
            n["yellow_box"] += 1
        elif r[2] == "tray" and n["tray"] < 1:
            r[3:6] = ["bad", "missing", "hidden"]
            n["tray"] += 1
    valid = {(r, f, c) for r, f in cells for c in CLASSES}
    loaded = load_label_review(_write(tmp_path, rows), CLASSES, valid)
    res = bad_fractions(loaded, CLASSES, n_frames=20, max_fraction=0.10)
    assert res["red_box"] == {"bad": 2, "bad_fraction": 0.10, "hidden_missing": 0, "passes": True}
    assert res["yellow_box"]["bad"] == 3 and res["yellow_box"]["passes"] is False
    assert res["tray"] == {"bad": 0, "bad_fraction": 0.0, "hidden_missing": 1, "passes": True}
    assert res["outer_box"]["passes"] is True
