"""F14 Stage 5/6 check (ISSUES.md 2026-10-02 P1.4: "whether RF-DETR wants a Roboflow-style
placeholder id 0 is unverified"): how the installed rfdetr reads COCO category ids, using
rfdetr's own dataset helpers on our synthetic dataset. CocoDetection itself needs
pycocotools (a ``rfdetr[train]`` extra, not installed here), so the mapping helpers it uses
are called directly."""

from __future__ import annotations

import pytest

pytest.importorskip("rfdetr")

from rfdetr import RFDETR  # noqa: E402
from rfdetr.datasets.coco import _train_split_cat2label, filter_parent_categories  # noqa: E402

from training import finetune as ft  # noqa: E402

pytestmark = pytest.mark.F14

CLASSES = ["outer_box", "tray", "red_box", "yellow_box", "start_button"]


def test_zero_based_ids_map_to_label_indices_unchanged_and_nothing_is_dropped(tmp_path):
    root = ft.make_synthetic_dataset(tmp_path / "ds", CLASSES, seed=0)
    assert _train_split_cat2label(root) == {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    assert RFDETR._load_classes(str(root)) == CLASSES
    assert RFDETR._detect_num_classes_for_training(str(root)) == 5


def test_a_roboflow_placeholder_is_dropped_only_because_of_its_supercategory():
    roboflow = [
        {"id": 0, "name": "root", "supercategory": "none"},
        {"id": 1, "name": "a", "supercategory": "root"},
        {"id": 2, "name": "b", "supercategory": "root"},
    ]
    assert [c["name"] for c in filter_parent_categories(roboflow, {1, 2})] == ["a", "b"]
    # our categories carry no supercategory, so id 0 is a real class and is kept
    ours = [{"id": i, "name": n} for i, n in enumerate(CLASSES)]
    assert [c["name"] for c in filter_parent_categories(ours, {0, 1, 2, 3, 4})] == CLASSES
