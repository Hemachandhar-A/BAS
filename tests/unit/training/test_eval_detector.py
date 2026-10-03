"""F14 Stage 7 (essential-features.md section 14): training/eval_detector.py. The test-split
refusal first, then the metric plumbing on a tiny synthetic ground truth with predictions of
known precision and recall, the gold subset, the report fields, and the CLI flow with the
detector faked. No mAP is re-implemented: supervision computes it."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from training import eval_detector as ev
from training import finetune as ft
from training import gpu_preflight as gp

pytestmark = pytest.mark.F14

CLASSES = ["outer_box", "tray", "red_box", "yellow_box", "start_button"]


# --- the test split is refused until acceptance.yaml exists ------------------------------------


def test_test_split_is_refused_without_acceptance_file(tmp_path):
    with pytest.raises(ev.Refusal) as e:
        ev.acceptance_gate("test", tmp_path / "acceptance.yaml")
    assert "acceptance.yaml" in str(e.value)


def test_test_split_with_acceptance_file_returns_its_sha256(tmp_path):
    f = tmp_path / "acceptance.yaml"
    f.write_bytes(b"detector: {}\n")
    assert ev.acceptance_gate("test", f) == hashlib.sha256(b"detector: {}\n").hexdigest()


def test_valid_split_needs_no_acceptance_file(tmp_path):
    assert ev.acceptance_gate("valid", tmp_path / "nope.yaml") is None


def test_cli_refuses_test_before_loading_any_model(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ev, "load_predictor", lambda *a, **k: pytest.fail("model was loaded"))
    code = ev.main(
        ["--model", "rfdetr", "--weights", str(tmp_path / "w.pth"), "--dataset-dir", str(tmp_path),
         "--split", "test", "--out", str(tmp_path / "r.json"),
         "--acceptance", str(tmp_path / "missing.yaml")]
    )  # fmt: skip
    assert code == 2
    assert "acceptance.yaml" in capsys.readouterr().err
    assert not (tmp_path / "r.json").exists()


# --- gold subset ----------------------------------------------------------------------------


def test_verified_files_picks_only_verified_true_frames():
    overlay = {
        "version": 1,
        "split": "val",
        "frames": {
            "a_1.jpg": {"verified": True, "boxes": []},
            "a_2.jpg": {"verified": False},
            "a_3.jpg": {"excluded": True},
            "a_4.jpg": {"verified": True, "excluded": True},
        },
    }
    assert ev.verified_files(overlay) == {"a_1.jpg"}
    assert ev.verified_files(None) == set()


def test_overlay_split_name_maps_valid_folder_to_val():
    assert ev.overlay_split("valid") == "val"
    assert ev.overlay_split("test") == "test"


# --- metric plumbing with known precision and recall ------------------------------------------


def _gt(boxes):  # boxes: list of (xyxy, class_id)
    return (
        np.array([b[0] for b in boxes], float).reshape(-1, 4),
        np.array([b[1] for b in boxes], int),
    )


def _pred(boxes):  # boxes: list of (xyxy, conf, class_id)
    return (
        np.array([b[0] for b in boxes], float).reshape(-1, 4),
        np.array([b[1] for b in boxes], float),
        np.array([b[2] for b in boxes], int),
    )


A = [0, 0, 10, 10]
B = [20, 20, 40, 40]
C = [50, 50, 60, 60]
FAR = [100, 100, 110, 110]


def _scene():
    """img1: class0 at A, class1 at B. img2: class0 at C.
    Predictions (img1): class0 at A (0.9, hit), class1 NOT predicted above the floor, but a
    correct class1 box at 0.1 (below floor 0.3). img2: class0 at C (0.8, hit) and a class0
    false positive at FAR (0.5)."""
    gt = {
        "i1.jpg": _gt([(A, 0), (B, 1)]),
        "i2.jpg": _gt([(C, 0)]),
    }
    pred = {
        "i1.jpg": _pred([(A, 0.9, 0), (B, 0.1, 1)]),
        "i2.jpg": _pred([(C, 0.8, 0), (FAR, 0.5, 0)]),
    }
    return gt, pred


def test_precision_recall_at_floor_and_map_with_known_values():
    gt, pred = _scene()
    r = ev.evaluate_subset(gt, pred, set(gt), CLASSES, floor=0.3)
    c0, c1 = r["per_class"]["outer_box"], r["per_class"]["tray"]
    assert c0["precision"] == pytest.approx(2 / 3) and c0["recall"] == pytest.approx(1.0)
    assert (c0["n_gt"], c0["n_pred_at_floor"]) == (2, 3)
    # tray: its only correct box is below the floor, so at the floor it is a miss
    assert c1["recall"] == pytest.approx(0.0) and c1["precision"] is None
    assert (c1["n_gt"], c1["n_pred_at_floor"]) == (1, 0)
    # mAP uses the low-confidence predictions too: tray is found at 0.1, so its AP50 is 1,
    # class 0's false positive ranks last, so its AP50 is 1 as well
    assert r["map50"] == pytest.approx(1.0, abs=1e-6)
    assert r["map50_95"] == pytest.approx(1.0, abs=1e-6)
    assert r["n_images"] == 2


def test_class_with_no_ground_truth_has_no_recall():
    gt, pred = _scene()
    r = ev.evaluate_subset(gt, pred, set(gt), CLASSES, floor=0.3)
    c = r["per_class"]["red_box"]
    assert c["recall"] is None and c["precision"] is None and c["n_gt"] == 0


def test_subset_restricts_images():
    gt, pred = _scene()
    r = ev.evaluate_subset(gt, pred, {"i2.jpg"}, CLASSES, floor=0.3)
    assert r["n_images"] == 1
    c0 = r["per_class"]["outer_box"]
    assert c0["precision"] == pytest.approx(0.5) and c0["recall"] == pytest.approx(1.0)
    assert r["per_class"]["tray"]["n_gt"] == 0


def test_empty_subset_reports_none_not_zero():
    gt, pred = _scene()
    r = ev.evaluate_subset(gt, pred, set(), CLASSES, floor=0.3)
    assert r["n_images"] == 0 and r["map50"] is None and r["map50_95"] is None
    assert all(c["recall"] is None for c in r["per_class"].values())


def test_wrong_class_is_both_a_miss_and_a_false_positive():
    gt = {"i1.jpg": _gt([(A, 0)])}
    pred = {"i1.jpg": _pred([(A, 0.9, 2)])}
    r = ev.evaluate_subset(gt, pred, set(gt), CLASSES, floor=0.3)
    assert r["per_class"]["outer_box"]["recall"] == pytest.approx(0.0)
    assert r["per_class"]["red_box"]["precision"] == pytest.approx(0.0)


# --- report fields and every metric twice -----------------------------------------------------


def test_report_has_every_metric_for_all_and_gold_and_the_provenance_fields():
    gt, pred = _scene()
    rep = ev.build_report(
        gt=gt, pred=pred, gold={"i1.jpg"}, classes=CLASSES, floor=0.3, model="rfdetr",
        split="valid", weights_sha256="ab" * 32, dataset_stamp="stamp1",
        generated_at="2026-10-03T00:00:00", acceptance_sha256=None,
    )  # fmt: skip
    assert rep["subsets"]["all"]["n_images"] == 2
    assert rep["subsets"]["gold"]["n_images"] == 1
    for sub in ("all", "gold"):
        assert {"map50", "map50_95", "per_class", "n_images"} <= set(rep["subsets"][sub])
    assert rep["generated_at"] == "2026-10-03T00:00:00"
    assert rep["weights_sha256"] == "ab" * 32
    assert rep["dataset_stamp"] == "stamp1"
    assert rep["detector_conf_floor"] == 0.3
    assert rep["acceptance_sha256"] is None and rep["split"] == "valid"


def test_report_without_gold_frames_says_so_instead_of_inventing_numbers():
    gt, pred = _scene()
    rep = ev.build_report(
        gt=gt, pred=pred, gold=set(), classes=CLASSES, floor=0.3, model="rfdetr", split="valid",
        weights_sha256="x", dataset_stamp=None, generated_at="t", acceptance_sha256=None,
    )  # fmt: skip
    assert rep["subsets"]["gold"]["n_images"] == 0
    assert rep["subsets"]["gold"]["map50"] is None
    assert "no verified" in rep["subsets"]["gold"]["note"]


def test_config_floor_is_read_from_perception_config_not_hardcoded():
    from contracts import PerceptionConfig

    assert ev.config_floor() == PerceptionConfig().detector_conf_floor


# --- reading the dataset ground truth ---------------------------------------------------------


def test_load_ground_truth_converts_xywh_to_xyxy_and_keeps_empty_images(tmp_path):
    root = ft.make_synthetic_dataset(tmp_path / "ds", CLASSES, seed=0)
    p = root / "valid" / "_annotations.coco.json"
    coco = json.loads(p.read_text())
    keep = [a for a in coco["annotations"] if a["image_id"] != 1]
    coco["annotations"] = keep  # image 1 now has no boxes
    p.write_text(json.dumps(coco))
    gt = ev.load_ground_truth(root / "valid")
    assert len(gt) == 2
    empty = gt[coco["images"][0]["file_name"]]
    assert empty[0].shape == (0, 4) and empty[1].shape == (0,)
    name, ann = coco["images"][1]["file_name"], coco["annotations"][0]
    xyxy, cls = gt[name]
    x, y, w, h = ann["bbox"]
    assert xyxy[0].tolist() == [x, y, x + w, y + h] and cls[0] == ann["category_id"]


# --- CLI flow with the detector faked ---------------------------------------------------------


def test_cli_end_to_end_on_valid_with_a_fake_detector(tmp_path, monkeypatch):
    root = ft.make_synthetic_dataset(tmp_path / "ds", CLASSES, seed=0)
    gt = ev.load_ground_truth(root / "valid")

    def fake_loader(model, weights, device, classes, **kw):
        # a perfect detector: returns the ground truth of whichever image it is shown
        by_content = {}
        for name, (xyxy, cls) in gt.items():
            by_content[name] = (xyxy, np.full(len(cls), 0.9), cls)
        order = iter(sorted(by_content))

        def predict(bgr):
            return by_content[next(order)]

        return predict

    monkeypatch.setattr(ev, "load_predictor", fake_loader)
    weights = tmp_path / "detector.pth"
    weights.write_bytes(b"w")
    corr = tmp_path / "corr"
    corr.mkdir()
    first = sorted(gt)[0]
    (corr / "val.json").write_text(
        json.dumps({"version": 1, "split": "val", "frames": {first: {"verified": True}}})
    )
    report = tmp_path / "reports" / "detector_eval.json"
    code = ev.main(
        ["--model", "rfdetr", "--weights", str(weights), "--dataset-dir", str(root),
         "--split", "valid", "--out", str(report), "--corrections-dir", str(corr),
         "--dataset-report", str(tmp_path / "none.json"), "--device", "cpu",
         "--acceptance", str(tmp_path / "no_acceptance.yaml")]
    )  # fmt: skip
    assert code == 0
    rep = json.loads(report.read_text())
    assert rep["weights_sha256"] == hashlib.sha256(b"w").hexdigest()
    assert rep["subsets"]["all"]["n_images"] == 2 and rep["subsets"]["gold"]["n_images"] == 1
    assert rep["subsets"]["all"]["map50"] == pytest.approx(1.0, abs=1e-6)
    assert rep["subsets"]["gold"]["per_class"]["tray"]["recall"] == pytest.approx(1.0)
    assert rep["split"] == "valid" and rep["acceptance_sha256"] is None


def test_cli_test_split_records_acceptance_sha(tmp_path, monkeypatch):
    root = ft.make_synthetic_dataset(tmp_path / "ds", CLASSES, seed=0)
    monkeypatch.setattr(ev, "load_predictor", lambda *a, **k: lambda bgr: _pred([]))
    weights = tmp_path / "d.pth"
    weights.write_bytes(b"w")
    acc = tmp_path / "acceptance.yaml"
    acc.write_bytes(b"detector: {}\n")
    out = tmp_path / "r.json"
    code = ev.main(
        ["--model", "rfdetr", "--weights", str(weights), "--dataset-dir", str(root),
         "--split", "test", "--out", str(out), "--acceptance", str(acc),
         "--corrections-dir", str(tmp_path / "nocorr"), "--device", "cpu"]
    )  # fmt: skip
    assert code == 0
    assert (
        json.loads(out.read_text())["acceptance_sha256"]
        == hashlib.sha256(b"detector: {}\n").hexdigest()
    )


# --- the real RF-DETR predict path (random 5-class head: plumbing only) -----------------------


@pytest.mark.slow
@pytest.mark.skipif(
    not gp.default_weights_path("rfdetr").is_file(), reason="RF-DETR weights not cached"
)
def test_rfdetr_predictor_returns_arrays_in_the_expected_shape():
    from rfdetr import RFDETRNano

    m = RFDETRNano(num_classes=5, pretrain_weights=str(gp.default_weights_path("rfdetr")))
    m.model.class_names = CLASSES
    predict = ev.rfdetr_predictor(m, floor=0.0, n_classes=5)
    bgr = np.random.default_rng(0).integers(0, 255, (120, 160, 3), dtype=np.uint8)
    xyxy, conf, cls = predict(bgr)
    assert xyxy.ndim == 2 and xyxy.shape[1] == 4
    assert len(xyxy) == len(conf) == len(cls)
    assert cls.size == 0 or (cls.min() >= 0 and cls.max() < 5)
