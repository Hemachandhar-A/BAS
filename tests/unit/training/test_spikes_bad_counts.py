"""Unit tests for the checkpoint-3 per-class bad counts (training/spikes/
bad_counts.py): v3 verdicts merged from the v2 review and the changed-cell
review; hidden-and-missing is never counted."""

from __future__ import annotations

import pytest

from training.spikes.bad_counts import (
    ChangedRow,
    bad_counts,
    load_changed_review,
    v3_bad_cells,
)
from training.spikes.review_csv import ReviewRow


def _v2(run, fid, cls, reason, vis=""):
    return ReviewRow(run, fid, cls, reason, vis, "")


def _ch(run, fid, cls, verdict, vis=""):
    return ChangedRow(run, fid, cls, verdict, vis)


def test_changed_cell_verdict_overrides_v2():
    v2 = [_v2("a", 1, "red_box", "wrong_box")]
    ch = [_ch("a", 1, "red_box", "ok")]
    assert v3_bad_cells(v2, ch) == set()


def test_unchanged_v2_bad_cell_stays_bad():
    v2 = [_v2("a", 1, "tray", "wrong_box")]
    assert v3_bad_cells(v2, []) == {("a", 1, "tray")}


def test_v2_hidden_missing_not_bad():
    v2 = [_v2("a", 1, "start_button", "missing", "hidden")]
    assert v3_bad_cells(v2, []) == set()


def test_new_bad_cell_from_changed_review():
    ch = [_ch("a", 2, "yellow_box", "bad", "visible")]
    assert v3_bad_cells([], ch) == {("a", 2, "yellow_box")}


def test_changed_review_ok_hidden_not_bad():
    ch = [_ch("a", 2, "start_button", "ok", "hidden")]
    assert v3_bad_cells([], ch) == set()


def test_bad_counts_fractions():
    cells = {("a", 1, "red_box"), ("a", 2, "red_box"), ("b", 1, "yellow_box")}
    out = bad_counts(cells, n_frames=20)
    assert out["red_box"] == {"bad": 2, "bad_fraction": pytest.approx(0.1)}
    assert out["yellow_box"]["bad"] == 1
    assert out["outer_box"]["bad"] == 0


def test_load_changed_review_rejects_bad_verdict(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("run_id,frame_id,class,verdict,reason,visibility,note\na,1,red_box,maybe,,,x\n")
    with pytest.raises(ValueError, match="verdict"):
        load_changed_review(p)


def test_load_changed_review_ok(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text(
        "run_id,frame_id,class,verdict,reason,visibility,note\na,1,red_box,bad,missing,visible,x\n"
    )
    assert load_changed_review(p) == [ChangedRow("a", 1, "red_box", "bad", "visible")]
