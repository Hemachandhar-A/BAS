"""F14 Stage 6 YOLO11n fallback path: training/coco_to_yolo.py (pure converter, a tiny
synthetic dataset end to end, and the dataset yaml)."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
import yaml

from training import coco_to_yolo as c2y

pytestmark = pytest.mark.F14

CLASSES = ["outer_box", "tray", "red_box", "yellow_box", "start_button"]


def _coco(images, annotations):
    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i, "name": n} for i, n in enumerate(CLASSES)],
    }


def test_box_is_normalised_to_centre_and_size():
    # xywh (100, 50, 200, 100) in a 400 x 200 image: centre (200, 100) -> (0.5, 0.5)
    assert c2y.xywh_to_yolo([100, 50, 200, 100], 400, 200) == pytest.approx((0.5, 0.5, 0.5, 0.5))
    assert c2y.xywh_to_yolo([0, 0, 40, 20], 400, 200) == pytest.approx((0.05, 0.05, 0.1, 0.1))


def test_class_index_is_position_in_category_list_not_the_raw_id():
    cats = [{"id": 7, "name": "b"}, {"id": 3, "name": "a"}]
    idx, names = c2y.class_index(cats)
    assert names == ["a", "b"]  # sorted by id, the order COCO ids define
    assert idx == {3: 0, 7: 1}


def test_zero_based_ids_keep_experiment_order():
    idx, names = c2y.class_index(_coco([], [])["categories"])
    assert names == CLASSES
    assert [idx[i] for i in range(5)] == [0, 1, 2, 3, 4]


def test_lines_per_image_include_empty_images():
    coco = _coco(
        [
            {"id": 1, "file_name": "a_1.jpg", "width": 400, "height": 200},
            {"id": 2, "file_name": "a_2.jpg", "width": 400, "height": 200},
        ],
        [{"id": 1, "image_id": 1, "category_id": 3, "bbox": [100, 50, 200, 100]}],
    )
    out = c2y.coco_to_yolo_lines(coco)
    assert out["a_1.jpg"] == ["3 0.500000 0.500000 0.500000 0.500000"]
    assert out["a_2.jpg"] == []
    assert set(out) == {"a_1.jpg", "a_2.jpg"}


def test_unknown_category_id_raises():
    coco = _coco(
        [{"id": 1, "file_name": "a.jpg", "width": 10, "height": 10}],
        [{"id": 1, "image_id": 1, "category_id": 9, "bbox": [0, 0, 5, 5]}],
    )
    with pytest.raises(c2y.ConvertError):
        c2y.coco_to_yolo_lines(coco)


def test_box_outside_image_raises():
    coco = _coco(
        [{"id": 1, "file_name": "a.jpg", "width": 10, "height": 10}],
        [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [8, 0, 5, 5]}],
    )
    with pytest.raises(c2y.ConvertError):
        c2y.coco_to_yolo_lines(coco)


def test_dataset_yaml_names_follow_class_order_and_leave_out_test(tmp_path):
    text = c2y.dataset_yaml(tmp_path, CLASSES)
    d = yaml.safe_load(text)
    assert d["names"] == {i: n for i, n in enumerate(CLASSES)}
    assert d["train"] == "images/train" and d["val"] == "images/valid"
    assert "test" not in d
    assert d["path"] == tmp_path.resolve().as_posix()


def _make_split(root, split, runs):
    folder = root / split
    folder.mkdir(parents=True)
    images, anns = [], []
    for run, fid, cat in runs:
        name = f"{run}_{fid}.jpg"
        cv2.imwrite(str(folder / name), np.full((20, 40, 3), 128, np.uint8))
        images.append({"id": len(images) + 1, "file_name": name, "width": 40, "height": 20})
        if cat is not None:
            anns.append(
                {
                    "id": len(anns) + 1,
                    "image_id": len(images),
                    "category_id": cat,
                    "bbox": [10, 5, 20, 10],
                }
            )
    (folder / "_annotations.coco.json").write_text(json.dumps(_coco(images, anns)))


def test_write_yolo_dataset_end_to_end(tmp_path):
    src = tmp_path / "dataset"
    _make_split(src, "train", [("r1", 0, 2), ("r1", 1, None)])
    _make_split(src, "valid", [("r2", 0, 3)])
    _make_split(src, "test", [("r3", 0, 1)])
    out = tmp_path / "yolo"
    yaml_path = c2y.write_yolo_dataset(src, out, CLASSES)
    assert yaml_path == out / "dataset.yaml"
    assert (out / "images" / "train" / "r1_0.jpg").is_file()
    label = (out / "labels" / "train" / "r1_0.txt").read_text().split()
    assert label == ["2", "0.500000", "0.500000", "0.500000", "0.500000"]
    assert (out / "labels" / "train" / "r1_1.txt").read_text() == ""  # empty image, empty file
    assert (out / "labels" / "valid" / "r2_0.txt").read_text().startswith("3 ")
    assert not (out / "images" / "test").exists()  # never converted: no test use in training
    d = yaml.safe_load(yaml_path.read_text())
    assert d["names"][2] == "red_box"


def test_write_yolo_dataset_refuses_other_class_order(tmp_path):
    src = tmp_path / "dataset"
    _make_split(src, "train", [("r1", 0, 2)])
    _make_split(src, "valid", [("r2", 0, 3)])
    with pytest.raises(c2y.ConvertError):
        c2y.write_yolo_dataset(src, tmp_path / "y", list(reversed(CLASSES)))
