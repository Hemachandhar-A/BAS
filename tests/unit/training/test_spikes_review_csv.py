"""Unit tests for training/spikes/review_csv.py: the strict loader for the
Lead's bad_boxes.csv review sheets (P1.2 checkpoint 1c)."""

from __future__ import annotations

import pytest

from training.spikes.review_csv import ReviewCsvError, load_bad_boxes, summarize

HEADER = "run_id,frame_id,class,reason,visibility,note\n"
FRAMES = {("x001", 1), ("x001", 2), ("x002", 5)}


def _write(tmp_path, body: str, header: str = HEADER):
    p = tmp_path / "bad.csv"
    p.write_text(header + body, encoding="utf-8")
    return p


def test_valid_rows_load(tmp_path):
    p = _write(
        tmp_path,
        "x001,1,red_box,wrong_box,,spans both\n"
        "x001,2,start_button,missing,hidden,under hand\n"
        "x002,5,tray,missing,partial,\n",
    )
    rows = load_bad_boxes(p, FRAMES)
    assert [(r.run_id, r.frame_id, r.cls) for r in rows] == [
        ("x001", 1, "red_box"),
        ("x001", 2, "start_button"),
        ("x002", 5, "tray"),
    ]
    assert rows[1].visibility == "hidden"


def test_header_only_is_empty(tmp_path):
    assert load_bad_boxes(_write(tmp_path, ""), FRAMES) == []


@pytest.mark.parametrize(
    "line",
    [
        "x001,1,blue_box,wrong_box,,",  # unknown class
        "x001,1,red_box,bad,,",  # unknown reason
        "x001,1,red_box,missing,,",  # visibility required for missing
        "x001,1,red_box,missing,seen,",  # visibility outside the set
        "x001,1,red_box,wrong_box,hidden,",  # visibility only allowed with missing
        "x001,1, red_box,wrong_box,,",  # stray whitespace
        "x001,1,red_box,wrong_box ,,",
        "x009,1,red_box,wrong_box,,",  # frame not in cache
        "x001,abc,red_box,wrong_box,,",  # frame_id not an integer
    ],
)
def test_invalid_rows_fail_loudly(tmp_path, line):
    with pytest.raises(ReviewCsvError):
        load_bad_boxes(_write(tmp_path, line + "\n"), FRAMES)


def test_duplicate_key_fails(tmp_path):
    p = _write(tmp_path, "x001,1,red_box,wrong_box,,\nx001,1,red_box,duplicate,,\n")
    with pytest.raises(ReviewCsvError):
        load_bad_boxes(p, FRAMES)


def test_bad_header_fails(tmp_path):
    with pytest.raises(ReviewCsvError):
        load_bad_boxes(_write(tmp_path, "", header="run_id,frame_id,class\n"), FRAMES)


def test_all_errors_reported_together(tmp_path):
    p = _write(tmp_path, "x001,1,blue_box,wrong_box,,\nx001,2,red_box,bad,,\n")
    with pytest.raises(ReviewCsvError) as exc:
        load_bad_boxes(p, FRAMES)
    assert "line 2" in str(exc.value) and "line 3" in str(exc.value)


def test_summarize_separates_hidden_missing(tmp_path):
    p = _write(
        tmp_path,
        "x001,1,red_box,wrong_box,,\n"
        "x001,2,red_box,missing,visible,\n"
        "x002,5,red_box,missing,hidden,\n"
        "x002,5,tray,wrong_class,,\n",
    )
    s = summarize(load_bad_boxes(p, FRAMES), n_frames=60)
    assert s["red_box"] == {"bad": 2, "bad_fraction": 2 / 60, "hidden_missing": 1}
    assert s["tray"]["bad"] == 1
    assert s["yellow_box"] == {"bad": 0, "bad_fraction": 0.0, "hidden_missing": 0}
