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


# --- review_outcome (the gate numbers recorded in reports/dataset.json) -----------------------


def _rr(run, fid, cls, verdict="ok", reason="", vis="", note=""):
    from training.review_sheet import ReviewRow

    return ReviewRow(run, fid, cls, verdict, reason, vis, note)


def _outcome_fixture():
    """Four frames: a#1 ok, a#2 excluded (red missing), b#1 ok but yellow wrong_box,
    b#2 excluded (yellow missing, hidden). a#1, a#2 stratified; b#1, b#2 flagged."""
    from training.review_sheet import NO_BOX_NOTE

    labels = {
        "a": {
            "1": {"status": "ok", "boxes": {c: [0, 0, 1, 1] for c in CLASSES}, "missing": []},
            "2": {
                "status": "excluded",
                "boxes": {c: [0, 0, 1, 1] for c in CLASSES if c != "red_box"},
                "missing": ["red_box"],
            },
        },
        "b": {
            "1": {"status": "ok", "boxes": {c: [0, 0, 1, 1] for c in CLASSES}, "missing": []},
            "2": {
                "status": "excluded",
                "boxes": {c: [0, 0, 1, 1] for c in CLASSES if c != "yellow_box"},
                "missing": ["yellow_box"],
            },
        },
    }
    rows = []
    for run, fid in [("a", 1), ("a", 2), ("b", 1), ("b", 2)]:
        for c in CLASSES:
            rows.append(_rr(run, fid, c))
    rows = [
        r
        for r in rows
        if (r.run_id, r.frame_id, r.cls)
        not in {("a", 2, "red_box"), ("b", 1, "yellow_box"), ("b", 2, "yellow_box")}
    ]
    rows += [
        _rr("a", 2, "red_box", "bad", "missing", "visible", NO_BOX_NOTE + " | held under the hand"),
        _rr("b", 1, "yellow_box", "bad", "wrong_box"),
        _rr("b", 2, "yellow_box", "bad", "missing", "hidden", NO_BOX_NOTE),
    ]
    return rows, labels


def test_review_outcome_gates_strata_and_kept_frames():
    from training.review_sheet import review_outcome

    rows, labels = _outcome_fixture()
    out = review_outcome(rows, CLASSES, [("a", 1), ("a", 2)], [("b", 1), ("b", 2)], labels)
    assert out["stratified"]["n_frames"] == 2
    assert out["stratified"]["per_class"]["red_box"]["bad"] == 1
    assert out["stratified"]["per_class"]["red_box"]["bad_fraction"] == 0.5
    assert out["stratified"]["gate_passes"] is False
    assert out["flagged"]["n_frames"] == 2
    # b#2 yellow is hidden-and-missing: not bad
    assert out["flagged"]["per_class"]["yellow_box"]["bad"] == 1
    # kept frames = labeler status ok: a#1 and b#1
    assert out["kept_frames"]["n_kept"] == 2
    assert out["kept_frames"]["per_class"]["yellow_box"]["bad"] == 1
    assert out["kept_frames"]["per_class"]["yellow_box"]["bad_fraction"] == 0.5
    assert out["kept_frames"]["per_class"]["red_box"]["bad"] == 0
    assert out["kept_frames_stratified"]["n_kept"] == 1
    assert out["kept_frames_stratified"]["gate_passes"] is True
    assert out["labeler_excluded_frames"] == 2


def test_review_outcome_checks_missing_cells_against_the_labeler_notes():
    from training.review_sheet import review_outcome

    rows, labels = _outcome_fixture()
    out = review_outcome(rows, CLASSES, [("a", 1), ("a", 2)], [("b", 1), ("b", 2)], labels)
    chk = out["missing_cells_check"]
    assert chk["reviewer_missing"] == 2
    assert chk["labeler_no_box_notes"] == 2
    assert chk["labeler_missing"] == 2
    assert chk["equal"] is True
    # a reviewer marks a cell the labeler did have as missing -> not equal
    rows.append(_rr("a", 1, "tray", "bad", "missing", "visible"))
    rows = [
        r
        for r in rows
        if not (r.run_id == "a" and r.frame_id == 1 and r.cls == "tray" and r.verdict == "ok")
    ]
    out = review_outcome(rows, CLASSES, [("a", 1), ("a", 2)], [("b", 1), ("b", 2)], labels)
    assert out["missing_cells_check"]["equal"] is False
    assert out["missing_cells_check"]["only_reviewer"] == [["a", 1, "tray"]]


def test_review_outcome_fails_loudly_on_unlabeled_or_unreviewed_frames():
    from training.review_sheet import review_outcome

    rows, labels = _outcome_fixture()
    labels["a"]["1"]["status"] = "pending"
    with pytest.raises(ValueError, match="not labeled"):
        review_outcome(rows, CLASSES, [("a", 1), ("a", 2)], [("b", 1), ("b", 2)], labels)
    rows, labels = _outcome_fixture()
    with pytest.raises(ValueError, match="no review row"):
        review_outcome(rows[:-3], CLASSES, [("a", 1), ("a", 2)], [("b", 1), ("b", 2)], labels)
