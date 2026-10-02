"""Unit tests for training/build_dataset.py (F14 stage 5) on a tiny synthetic
dataset: layout, class order and ids, the overlay semantics of Plan 5.8,
the assertions, and the report with its stamp. The real dataset is not built here."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from training.build_dataset import (
    BuildError,
    build_dataset,
    check_coco,
    dataset_stamp,
)

CLASSES = ["outer_box", "tray", "red_box", "yellow_box", "start_button"]
W, H = 160, 96


def _boxes(shift=0.0):
    return {
        "outer_box": [5.0 + shift, 5.0, 90.0, 90.0],
        "tray": [100.0, 10.0, 150.0, 50.0],
        "red_box": [20.0, 20.0, 50.0, 45.0],
        "yellow_box": [20.0, 55.0, 50.0, 80.0],
        "start_button": [110.0, 70.0, 140.0, 90.0],
    }


# run -> (split, frame ids); a1 train (frames 0, 30, 60), a2 train, v1 val, t1 test
RUNS = {
    "a1": ("train", [0, 30, 60]),
    "a2": ("train", [0, 30]),
    "v1": ("val", [0, 30]),
    "t1": ("test", [0, 30]),
}


def _setup(tmp_path: Path, *, auto_excluded=("a1_60.jpg",), pending=()):
    frames = tmp_path / "frames"
    manifest, index, labels = [], {}, {}
    for run, (split, fids) in RUNS.items():
        manifest.append({"run_id": run, "split": split, "width": str(W), "height": str(H)})
        index[run] = {"split": split, "fps": 30.0, "frame_ids": fids}
        (frames / split).mkdir(parents=True, exist_ok=True)
        labels[run] = {}
        for fid in fids:
            name = f"{run}_{fid}.jpg"
            cv2.imwrite(str(frames / split / name), np.full((H, W, 3), 50 + fid, dtype=np.uint8))
            if name in pending:
                lab = {"status": "pending", "boxes": {}, "missing": [], "reason": ""}
            elif name in auto_excluded:
                lab = {
                    "status": "excluded",
                    "boxes": {},
                    "missing": ["red_box"],
                    "reason": "missing:red_box",
                }
            else:
                lab = {"status": "ok", "boxes": _boxes(), "missing": [], "reason": ""}
            labels[run][str(fid)] = lab
    return dict(
        classes=CLASSES,
        manifest_rows=manifest,
        index=index,
        frame_labels=labels,
        frames_dir=frames,
        corrections_dir=tmp_path / "corrections",
        out_dir=tmp_path / "dataset",
        report_path=tmp_path / "reports" / "dataset.json",
        generated_at="2026-10-02T00:00:00",
    )


def _overlay(tmp_path, split, body):
    d = tmp_path / "corrections"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{split}.json").write_text(json.dumps({"version": 1, "split": split, **body}))


def _coco(tmp_path, folder):
    return json.loads((tmp_path / "dataset" / folder / "_annotations.coco.json").read_text())


# --- layout, classes, ids --------------------------------------------------------------


def test_layout_val_becomes_valid_and_only_ok_frames_are_included(tmp_path):
    build_dataset(**_setup(tmp_path))
    out = tmp_path / "dataset"
    assert sorted(p.name for p in out.iterdir()) == ["test", "train", "valid"]
    train = _coco(tmp_path, "train")
    assert sorted(i["file_name"] for i in train["images"]) == [
        "a1_0.jpg",
        "a1_30.jpg",
        "a2_0.jpg",
        "a2_30.jpg",
    ]
    assert (out / "train" / "a1_0.jpg").exists() and not (out / "train" / "a1_60.jpg").exists()
    assert sorted(i["file_name"] for i in _coco(tmp_path, "valid")["images"]) == [
        "v1_0.jpg",
        "v1_30.jpg",
    ]
    img = cv2.imread(str(out / "train" / "a1_0.jpg"))
    assert img.shape == (H, W, 3)


def test_class_names_equal_experiment_classes_in_order_with_contiguous_ids(tmp_path):
    build_dataset(**_setup(tmp_path))
    for folder in ("train", "valid", "test"):
        coco = _coco(tmp_path, folder)
        assert [c["name"] for c in coco["categories"]] == CLASSES
        assert [c["id"] for c in coco["categories"]] == list(range(5))


def test_round_trip_id_name_mapping_finds_a_known_box(tmp_path):
    build_dataset(**_setup(tmp_path))
    coco = _coco(tmp_path, "train")
    name_of = {c["id"]: c["name"] for c in coco["categories"]}
    img = next(i for i in coco["images"] if i["file_name"] == "a1_0.jpg")
    by_name = {
        name_of[a["category_id"]]: a["bbox"]
        for a in coco["annotations"]
        if a["image_id"] == img["id"]
    }
    assert by_name["red_box"] == [20.0, 20.0, 30.0, 25.0]  # xywh of [20, 20, 50, 45]
    assert by_name["yellow_box"] == [20.0, 55.0, 30.0, 25.0]


# --- overlay semantics (Plan 5.8) ------------------------------------------------------------


def test_frame_overlay_replaces_that_frames_boxes_wholesale(tmp_path):
    _overlay(
        tmp_path,
        "train",
        {
            "frames": {
                "a1_0.jpg": {
                    "excluded": False,
                    "verified": True,
                    "boxes": [{"class": "red_box", "xyxy": [1.0, 2.0, 11.0, 12.0]}],
                }
            }
        },
    )
    report = build_dataset(**_setup(tmp_path))
    coco = _coco(tmp_path, "train")
    img = next(i for i in coco["images"] if i["file_name"] == "a1_0.jpg")
    anns = [a for a in coco["annotations"] if a["image_id"] == img["id"]]
    assert (
        len(anns) == 1 and anns[0]["category_id"] == 2 and anns[0]["bbox"] == [1.0, 2.0, 10.0, 10.0]
    )
    assert report["splits"]["train"]["frames_verified"] == 1
    assert report["splits"]["train"]["frames_corrected"] == 1


def test_a_confirmed_unchanged_frame_is_verified_but_not_corrected(tmp_path):
    boxes = [{"class": c, "xyxy": b} for c, b in _boxes().items()]
    _overlay(
        tmp_path,
        "train",
        {"frames": {"a1_0.jpg": {"excluded": False, "verified": True, "boxes": boxes}}},
    )
    report = build_dataset(**_setup(tmp_path))
    assert report["splits"]["train"]["frames_verified"] == 1
    assert report["splits"]["train"]["frames_corrected"] == 0


def test_overlay_can_bring_back_an_auto_excluded_frame(tmp_path):
    boxes = [{"class": c, "xyxy": b} for c, b in _boxes().items()]
    _overlay(
        tmp_path,
        "train",
        {"frames": {"a1_60.jpg": {"excluded": False, "verified": False, "boxes": boxes}}},
    )
    report = build_dataset(**_setup(tmp_path))
    assert "a1_60.jpg" in [i["file_name"] for i in _coco(tmp_path, "train")["images"]]
    assert report["splits"]["train"]["frames_corrected"] == 1
    assert report["splits"]["train"]["frames_excluded_auto"] == 0


def test_overlay_excluded_drops_the_frame(tmp_path):
    _overlay(
        tmp_path,
        "train",
        {"frames": {"a2_30.jpg": {"excluded": True, "verified": False, "boxes": []}}},
    )
    report = build_dataset(**_setup(tmp_path))
    names = [i["file_name"] for i in _coco(tmp_path, "train")["images"]]
    assert "a2_30.jpg" not in names and "a2_0.jpg" in names
    assert report["splits"]["train"]["frames_excluded_overlay"] == 1
    assert not (tmp_path / "dataset" / "train" / "a2_30.jpg").exists()


def test_static_override_applies_to_every_frame_of_the_run_except_own_entries(tmp_path):
    new_tray = [101.0, 11.0, 151.0, 51.0]
    own = [{"class": "tray", "xyxy": [0.0, 0.0, 5.0, 5.0]}]
    _overlay(
        tmp_path,
        "train",
        {
            "static_overrides": {"a1": {"tray": new_tray}},
            "frames": {"a1_30.jpg": {"excluded": False, "verified": False, "boxes": own}},
        },
    )
    report = build_dataset(**_setup(tmp_path))
    coco = _coco(tmp_path, "train")
    ids = {i["file_name"]: i["id"] for i in coco["images"]}

    def tray_of(name):
        return [
            a["bbox"]
            for a in coco["annotations"]
            if a["image_id"] == ids[name] and a["category_id"] == 1
        ]

    assert tray_of("a1_0.jpg") == [[101.0, 11.0, 50.0, 40.0]]
    assert tray_of("a1_30.jpg") == [[0.0, 0.0, 5.0, 5.0]]  # its own entry wins
    assert tray_of("a2_0.jpg") == [[100.0, 10.0, 50.0, 40.0]]  # other runs untouched
    assert (
        report["splits"]["train"]["frames_static_overridden"] == 1
    )  # a1_0 only (a1_60 is excluded)


# --- assertions ----------------------------------


@pytest.mark.parametrize(
    "body,fragment",
    [
        (
            {
                "frames": {
                    "a1_0.jpg": {
                        "excluded": False,
                        "verified": True,
                        "boxes": [{"class": "blue_box", "xyxy": [0, 0, 5, 5]}],
                    }
                }
            },
            "class",
        ),
        (
            {"frames": {"a1_999.jpg": {"excluded": False, "verified": True, "boxes": []}}},
            "does not exist",
        ),
        ({"frames": {"v1_0.jpg": {"excluded": False, "verified": True, "boxes": []}}}, "split"),
        ({"static_overrides": {"v1": {"tray": [0, 0, 5, 5]}}}, "split"),
        ({"static_overrides": {"a1": {"blue_box": [0, 0, 5, 5]}}}, "class"),
        ({"static_overrides": {"nope": {"tray": [0, 0, 5, 5]}}}, "run"),
    ],
)
def test_overlay_assertions(tmp_path, body, fragment):
    _overlay(tmp_path, "train", body)
    with pytest.raises(BuildError, match=fragment):
        build_dataset(**_setup(tmp_path))


def test_overlay_whose_split_field_disagrees_with_its_file_name_is_rejected(tmp_path):
    d = tmp_path / "corrections"
    d.mkdir()
    (d / "train.json").write_text(json.dumps({"version": 1, "split": "val", "frames": {}}))
    with pytest.raises(BuildError, match="split"):
        build_dataset(**_setup(tmp_path))


def test_overlay_version_must_be_one(tmp_path):
    d = tmp_path / "corrections"
    d.mkdir()
    (d / "train.json").write_text(json.dumps({"version": 2, "split": "train", "frames": {}}))
    with pytest.raises(BuildError, match="version"):
        build_dataset(**_setup(tmp_path))


def test_a_run_in_two_splits_is_rejected(tmp_path):
    kw = _setup(tmp_path)
    kw["manifest_rows"].append({"run_id": "a1", "split": "val", "width": str(W), "height": str(H)})
    with pytest.raises(BuildError, match="two splits"):
        build_dataset(**kw)


def test_train_may_only_hold_train_runs(tmp_path):
    kw = _setup(tmp_path)
    kw["index"]["v1"]["split"] = "train"  # the frame index disagrees with the manifest
    with pytest.raises(BuildError, match="manifest"):
        build_dataset(**kw)


def test_pending_frames_block_the_build(tmp_path):
    with pytest.raises(BuildError, match="not labeled"):
        build_dataset(**_setup(tmp_path, pending=("a1_30.jpg",)))


def test_wrong_class_set_is_rejected(tmp_path):
    kw = _setup(tmp_path)
    kw["classes"] = ["tray", "outer_box", "red_box", "yellow_box", "start_button"]  # order changed
    # labels still use the same names, so the build works; only duplicates are fatal
    build_dataset(**kw)
    kw["classes"] = CLASSES[:2] + CLASSES[:1] + CLASSES[3:]
    with pytest.raises(BuildError, match="classes"):
        build_dataset(**kw)


def test_check_coco_catches_non_contiguous_ids_and_name_mismatch():
    good = {
        "images": [{"id": 1, "file_name": "a1_0.jpg", "width": W, "height": H}],
        "annotations": [],
        "categories": [{"id": i, "name": c} for i, c in enumerate(CLASSES)],
    }
    check_coco(good, CLASSES)
    bad_ids = json.loads(json.dumps(good))
    bad_ids["categories"][3]["id"] = 7
    with pytest.raises(BuildError, match="contiguous"):
        check_coco(bad_ids, CLASSES)
    bad_names = json.loads(json.dumps(good))
    bad_names["categories"][0]["name"] = "tray"
    with pytest.raises(BuildError, match="names"):
        check_coco(bad_names, CLASSES)
    oob = json.loads(json.dumps(good))
    oob["annotations"] = [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 0,
            "bbox": [0, 0, W + 5, 10],
            "area": 1,
            "iscrowd": 0,
        }
    ]
    with pytest.raises(BuildError, match="outside"):
        check_coco(oob, CLASSES)


# --- report and stamp ----------------------------------


def test_report_counts_per_split_class_and_run(tmp_path):
    report = build_dataset(**_setup(tmp_path))
    assert report["classes"] == CLASSES
    tr = report["splits"]["train"]
    assert tr["images"] == 4 and tr["annotations"] == 20
    assert tr["images_per_run"] == {"a1": 2, "a2": 2}
    assert tr["annotations_per_class"] == {c: 4 for c in CLASSES}
    assert tr["frames_excluded_auto"] == 1 and tr["frames_excluded_overlay"] == 0
    assert report["splits"]["valid"]["images"] == 2 and report["splits"]["test"]["images"] == 2
    assert json.loads((tmp_path / "reports" / "dataset.json").read_text()) == report
    assert report["generated_at"] == "2026-10-02T00:00:00"


def test_stamp_is_stable_and_changes_with_annotations_and_overlays(tmp_path):
    a = build_dataset(**_setup(tmp_path / "one"))["dataset_stamp"]
    b = build_dataset(**_setup(tmp_path / "one"))["dataset_stamp"]
    assert a == b and len(a) == 16

    _overlay(
        tmp_path / "two",
        "train",
        {"frames": {"a2_30.jpg": {"excluded": True, "verified": False, "boxes": []}}},
    )
    c = build_dataset(**_setup(tmp_path / "two"))["dataset_stamp"]
    assert c != a

    kw = _setup(tmp_path / "three")
    kw["frame_labels"]["a1"]["0"]["boxes"]["tray"] = [100.0, 10.0, 151.0, 50.0]
    assert build_dataset(**kw)["dataset_stamp"] != a


def test_stamp_covers_the_overlay_files_even_when_annotations_are_identical(tmp_path):
    # a verified-only entry does not change a single box, but the overlay file is stamped
    kw = _setup(tmp_path / "one")
    base = build_dataset(**kw)["dataset_stamp"]
    boxes = [{"class": c, "xyxy": b} for c, b in _boxes().items()]
    _overlay(
        tmp_path / "two",
        "train",
        {"frames": {"a1_0.jpg": {"excluded": False, "verified": True, "boxes": boxes}}},
    )
    with_overlay = build_dataset(**_setup(tmp_path / "two"))["dataset_stamp"]
    assert with_overlay != base


def test_dataset_stamp_is_order_independent_of_the_file_list(tmp_path):
    p1, p2 = tmp_path / "a.json", tmp_path / "b.json"
    p1.write_text("1")
    p2.write_text("2")
    assert dataset_stamp([p1, p2]) == dataset_stamp([p2, p1])


def test_review_outcome_is_recorded_only_when_given(tmp_path):
    kw = _setup(tmp_path)
    assert build_dataset(**kw)["label_review"] == {"status": "not_recorded"}
    outcome = {"red_box": {"bad": 1, "bad_fraction": 0.05, "hidden_missing": 0, "passes": True}}
    kw2 = _setup(tmp_path / "two")
    assert build_dataset(**kw2, label_review=outcome)["label_review"] == outcome
