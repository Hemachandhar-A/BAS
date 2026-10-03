"""training/verify_dataset.py: the dataset on disk is whole and is the one the report names.
Each check has a test that breaks exactly that thing."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from training import verify_dataset as V
from training.build_dataset import dataset_stamp
from training.finetune import make_synthetic_dataset

CLASSES = ["outer_box", "tray", "red_box", "yellow_box", "start_button"]


def _setup(tmp_path):
    """A valid dataset (3 train, 2 valid, 1 test image), an overlay and a matching report."""
    ds = make_synthetic_dataset(tmp_path / "dataset", CLASSES)
    corr = tmp_path / "corrections"
    corr.mkdir()
    overlay = corr / "train.json"
    overlay.write_bytes(json.dumps({"version": 1, "split": "train", "frames": {}}).encode())
    return ds, corr, _write_report(tmp_path, ds, corr)


def _write_report(tmp_path, ds, corr, **override):
    import hashlib

    anns = [ds / s / "_annotations.coco.json" for s in ("train", "valid", "test")]
    report = {
        "version": 1,
        "classes": CLASSES,
        "dataset_stamp": dataset_stamp(anns + [corr / "train.json"]),
        "overlay_files": {
            "train": hashlib.sha256((corr / "train.json").read_bytes()).hexdigest()[:16]
        },
        "splits": {
            "train": {"images": 3, "annotations": 15},
            "valid": {"images": 2, "annotations": 10},
            "test": {"images": 1, "annotations": 5},
        },
    } | override
    path = tmp_path / "dataset_report.json"
    path.write_bytes(json.dumps(report, indent=1).encode())
    return path


def _run(ds, corr, report, **kw):
    return V.verify(ds, report, corr, CLASSES, **kw)


def test_a_valid_dataset_has_no_problems_and_reports_counts(tmp_path):
    ds, corr, report = _setup(tmp_path)
    result = _run(ds, corr, report)
    assert result.problems == []
    assert result.counts == {"train": 3, "valid": 2, "test": 1}
    assert result.stamp == json.loads(report.read_text())["dataset_stamp"]


def test_a_json_that_does_not_parse_is_named(tmp_path):
    ds, corr, report = _setup(tmp_path)
    (ds / "valid" / "_annotations.coco.json").write_bytes(b"{ not json")
    problems = _run(ds, corr, report).problems
    assert any("valid/_annotations.coco.json" in p and "parse" in p for p in problems)


def test_nul_bytes_in_a_json_or_csv_are_found_in_every_place_scanned(tmp_path):
    ds, corr, report = _setup(tmp_path)
    ann = ds / "train" / "_annotations.coco.json"
    ann.write_bytes(ann.read_bytes() + b"\x00\x00")
    extra = tmp_path / "manifest.csv"
    extra.write_bytes(b"run_id,split\n" + b"\x00" * 10)
    problems = _run(ds, corr, report, csv_paths=[extra]).problems
    assert any("NUL" in p and "train/_annotations.coco.json" in p for p in problems)
    assert any("NUL" in p and "manifest.csv" in p for p in problems)
    # the report and the overlay are scanned too
    report.write_bytes(report.read_bytes() + b"\x00")
    (corr / "train.json").write_bytes((corr / "train.json").read_bytes() + b"\x00")
    problems = _run(ds, corr, report).problems
    assert any("NUL" in p and report.name in p for p in problems)
    assert any("NUL" in p and "corrections" in p for p in problems)


def test_an_image_file_without_an_annotation_or_the_reverse_is_a_count_mismatch(tmp_path):
    ds, corr, report = _setup(tmp_path)
    cv2.imwrite(str(ds / "train" / "extra_1.jpg"), np.zeros((120, 160, 3), np.uint8))
    assert any("extra_1.jpg" in p for p in _run(ds, corr, report).problems)
    (ds / "train" / "extra_1.jpg").unlink()
    (ds / "valid" / "dry4_4.jpg").unlink()
    problems = _run(ds, corr, report).problems
    assert any("dry4_4.jpg" in p and "missing" in p for p in problems)
    assert any("image count" in p and "valid" in p for p in problems)


def test_an_image_that_does_not_open_is_named(tmp_path):
    ds, corr, report = _setup(tmp_path)
    (ds / "test" / "dry6_6.jpg").write_bytes(b"not a jpeg")
    assert any("dry6_6.jpg" in p and "open" in p for p in _run(ds, corr, report).problems)


def test_an_image_whose_size_differs_from_the_annotation_is_named(tmp_path):
    ds, corr, report = _setup(tmp_path)
    cv2.imwrite(str(ds / "test" / "dry6_6.jpg"), np.zeros((50, 50, 3), np.uint8))
    assert any("dry6_6.jpg" in p and "size" in p for p in _run(ds, corr, report).problems)


def test_every_image_has_one_box_per_class(tmp_path):
    ds, corr, report = _setup(tmp_path)
    path = ds / "train" / "_annotations.coco.json"
    coco = json.loads(path.read_text())
    coco["annotations"] = coco["annotations"][1:]  # image 1 loses its first box
    path.write_bytes(json.dumps(coco).encode())
    problems = _run(ds, corr, report).problems
    assert any("dry1_1.jpg" in p and "4 boxes" in p for p in problems)


def test_a_duplicated_class_on_one_image_is_a_problem_even_with_five_boxes(tmp_path):
    ds, corr, report = _setup(tmp_path)
    path = ds / "train" / "_annotations.coco.json"
    coco = json.loads(path.read_text())
    coco["annotations"][0]["category_id"] = 1  # image 1 now has tray twice and no outer_box
    path.write_bytes(json.dumps(coco).encode())
    assert any("dry1_1.jpg" in p and "outer_box" in p for p in _run(ds, corr, report).problems)


def test_class_names_must_equal_experiment_classes_in_order(tmp_path):
    ds, corr, report = _setup(tmp_path)
    path = ds / "valid" / "_annotations.coco.json"
    coco = json.loads(path.read_text())
    coco["categories"][2]["name"], coco["categories"][3]["name"] = "yellow_box", "red_box"
    path.write_bytes(json.dumps(coco).encode())
    assert any("class names" in p and "valid" in p for p in _run(ds, corr, report).problems)


def test_report_classes_must_also_equal_experiment_classes(tmp_path):
    ds, corr, _ = _setup(tmp_path)
    report = _write_report(tmp_path, ds, corr, classes=CLASSES[::-1])
    assert any("report classes" in p for p in _run(ds, corr, report).problems)


def test_a_changed_annotation_changes_the_stamp_and_is_reported(tmp_path):
    ds, corr, report = _setup(tmp_path)
    path = ds / "test" / "_annotations.coco.json"
    coco = json.loads(path.read_text())
    coco["info"]["description"] = "changed"
    path.write_bytes(json.dumps(coco).encode())
    assert any("stamp" in p for p in _run(ds, corr, report).problems)


def test_a_changed_overlay_is_reported_by_hash_and_by_stamp(tmp_path):
    ds, corr, report = _setup(tmp_path)
    (corr / "train.json").write_bytes(
        json.dumps({"version": 1, "split": "train", "frames": {"a_1.jpg": {}}}).encode()
    )
    problems = _run(ds, corr, report).problems
    assert any("overlay" in p and "train" in p for p in problems)
    assert any("stamp" in p for p in problems)


def test_a_missing_overlay_file_named_by_the_report_is_a_problem(tmp_path):
    ds, corr, report = _setup(tmp_path)
    (corr / "train.json").unlink()
    assert any("overlay" in p and "missing" in p for p in _run(ds, corr, report).problems)


def test_report_image_counts_must_match_the_disk(tmp_path):
    ds, corr, _ = _setup(tmp_path)
    report = _write_report(
        tmp_path,
        ds,
        corr,
        splits={"train": {"images": 594}, "valid": {"images": 2}, "test": {"images": 1}},
    )
    assert any(
        "report" in p and "train" in p and "594" in p for p in _run(ds, corr, report).problems
    )


def test_expected_counts_can_be_pinned(tmp_path):
    ds, corr, report = _setup(tmp_path)
    assert _run(ds, corr, report, expect_train=3).problems == []
    assert any("expected 593" in p for p in _run(ds, corr, report, expect_train=593).problems)


def test_a_missing_split_folder_is_reported_without_a_crash(tmp_path):
    ds, corr, report = _setup(tmp_path)
    import shutil

    shutil.rmtree(ds / "test")
    assert any("test" in p and "missing" in p for p in _run(ds, corr, report).problems)


@pytest.mark.parametrize("flag", [0, 1])
def test_cli_exit_code_and_output(tmp_path, capsys, flag):
    ds, corr, report = _setup(tmp_path)
    if flag:
        (ds / "valid" / "dry4_4.jpg").unlink()
    args = [
        "--dataset-dir", str(ds), "--report", str(report), "--corrections", str(corr),
        "--no-default-csvs",
    ]  # fmt: skip
    code = V.main(args)
    out = capsys.readouterr()
    assert code == flag
    if flag:
        assert "PROBLEMS" in out.out and "dry4_4.jpg" in out.out
    else:
        assert "OK" in out.out and "train 3" in out.out
