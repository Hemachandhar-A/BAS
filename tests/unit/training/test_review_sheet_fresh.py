"""The fresh review sample (F14 stage 4b): drawn from the BUILT train dataset (overlay
applied), a new recorded seed, none of the earlier sample's frames, at least 2 per run
where available; sheets and the prefilled CSV reuse the earlier layout, strict loader
and per-class gate."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from training.review_sheet import (
    CSV_HEADER,
    FRESH_SEED,
    SEED,
    bad_fractions,
    coco_boxes_by_frame,
    draw_fresh_sample,
    fresh_outcome,
    load_label_review,
    main,
    prefill_rows,
    write_review_csv,
)

CLASSES = ["outer_box", "tray", "red_box", "yellow_box", "start_button"]


def _coco(frames_by_run):
    images, anns = [], []
    for run in sorted(frames_by_run):
        for fid in frames_by_run[run]:
            iid = len(images) + 1
            images.append({"id": iid, "file_name": f"{run}_{fid}.jpg", "width": 160, "height": 96})
            for k, _ in enumerate(CLASSES):
                anns.append(
                    {"id": len(anns) + 1, "image_id": iid, "category_id": k,
                     "bbox": [10.0 + k, 10.0, 20.0, 30.0], "area": 600.0, "iscrowd": 0}
                )  # fmt: skip
    cats = [{"id": i, "name": c} for i, c in enumerate(CLASSES)]
    return {"images": images, "annotations": anns, "categories": cats}


def test_fresh_seed_is_new():
    assert FRESH_SEED != SEED


def test_coco_boxes_by_frame_gives_xyxy_per_class():
    out = coco_boxes_by_frame(_coco({"x001": [0, 30]}))
    assert set(out) == {("x001", 0), ("x001", 30)}
    assert out[("x001", 0)]["red_box"] == (12.0, 10.0, 32.0, 40.0)
    assert list(out[("x001", 0)]) == CLASSES


def test_draw_fresh_sample_excludes_old_frames_and_covers_every_run():
    frames = {f"x{i:03d}": [f * 30 for f in range(8)] for i in range(6)}
    exclude = {(r, f) for r in frames for f in (0, 30, 60)}
    cells = draw_fresh_sample(frames, exclude, n=18, per_run_min=2, seed=FRESH_SEED)
    assert len(cells) == 18 == len(set(cells))
    assert not set(cells) & exclude
    for run in frames:
        assert sum(1 for r, _ in cells if r == run) >= 2
    assert cells == sorted(cells)
    assert cells == draw_fresh_sample(frames, exclude, n=18, per_run_min=2, seed=FRESH_SEED)
    assert cells != draw_fresh_sample(frames, exclude, n=18, per_run_min=2, seed=FRESH_SEED + 1)


def test_draw_fresh_sample_takes_what_a_short_run_has():
    frames = {"x001": [0, 30, 60, 90], "x002": [0, 30, 60, 90, 120]}
    exclude = {("x001", 0), ("x001", 30), ("x001", 60)}  # x001 has 1 left
    cells = draw_fresh_sample(frames, exclude, n=5, per_run_min=2, seed=1)
    assert ("x001", 90) in cells and len(cells) == 5


def test_draw_fresh_sample_fails_when_not_enough_frames():
    with pytest.raises(ValueError):
        draw_fresh_sample({"x001": [0, 30]}, set(), n=5, per_run_min=1, seed=1)


def test_fresh_outcome_reuses_the_per_class_gate(tmp_path):
    cells = [("x001", i * 30) for i in range(10)]
    path = tmp_path / "fresh_review.csv"
    rows = prefill_rows(cells, CLASSES, {c: set(CLASSES) for c in cells})
    write_review_csv(path, rows)
    keys = {(r, f, c) for r, f in cells for c in CLASSES}
    loaded = load_label_review(path, CLASSES, keys)
    out = fresh_outcome(loaded, CLASSES, cells)
    assert out["gate_passes"] and out["n_frames"] == 10
    assert out["per_class"] == bad_fractions(loaded, CLASSES, 10)
    # one bad row in ten is exactly 10 %: passes; two is 20 %: fails (red_box only)
    path.write_text(
        path.read_text().replace("x001,0,red_box,ok,,,", "x001,0,red_box,bad,wrong_box,,", 1)
    )
    one = fresh_outcome(load_label_review(path, CLASSES, keys), CLASSES, cells)
    assert one["gate_passes"] and one["per_class"]["red_box"]["bad"] == 1
    path.write_text(
        path.read_text().replace("x001,30,red_box,ok,,,", "x001,30,red_box,bad,wrong_box,,", 1)
    )
    two = fresh_outcome(load_label_review(path, CLASSES, keys), CLASSES, cells)
    assert not two["gate_passes"] and not two["per_class"]["red_box"]["passes"]


def test_cli_fresh_sample_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    frames = {f"x{i:03d}": [f * 30 for f in range(6)] for i in range(3)}
    ds = tmp_path / "data" / "dataset" / "train"
    ds.mkdir(parents=True)
    for run, fids in frames.items():
        for fid in fids:
            cv2.imwrite(str(ds / f"{run}_{fid}.jpg"), np.full((96, 160, 3), 80, np.uint8))
    (ds / "_annotations.coco.json").write_text(json.dumps(_coco(frames)))
    review = tmp_path / "data" / "review"
    review.mkdir(parents=True)
    old = [["x000", 0], ["x001", 0]]
    (review / "sample.json").write_text(json.dumps({"cells": old}))
    corr = tmp_path / "data" / "corrections"
    corr.mkdir()
    (corr / "train.json").write_text(
        json.dumps({"version": 1, "split": "train",
                    "frames": {"x002_30.jpg": {"excluded": False, "verified": True, "boxes": []},
                               "x002_60.jpg": {"excluded": True, "verified": False, "boxes": []}}})
    )  # fmt: skip
    main(["--fresh-sample", "--n", "8"])
    meta = json.loads((review / "fresh_sample.json").read_text())
    cells = [tuple(c) for c in meta["cells"]]
    assert len(cells) == 8 and not set(cells) & {("x000", 0), ("x001", 0)}
    assert meta["seed"] == FRESH_SEED
    assert meta["n_overlay_corrected"] == sum(1 for c in cells if c == ("x002", 30))
    assert (review / "fresh_sheet_01.jpg").exists() and not (review / "fresh_sheet_02.jpg").exists()
    lines = (review / "fresh_review.csv").read_text().splitlines()
    assert lines[0].split(",") == CSV_HEADER and len(lines) == 1 + 8 * 5
    assert all(",ok," in ln for ln in lines[1:])
