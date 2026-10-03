"""F2 (essential-features.md section 2): perception/detector.py -- the Detector protocol, the
two backends, and load_detector with its precedence, refusals and warning. Everything runs
against fakes; the real-weights equivalence test is in test_detector_equivalence.py."""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from contracts import DETECTOR_MIN_CONF, Detection
from perception import detector as D

pytestmark = pytest.mark.F2

CLASSES = ["outer_box", "tray", "red_box", "yellow_box", "start_button"]
IMG = np.zeros((48, 64, 3), np.uint8)


# --- ordering and conversion ---------------------------------------------------------------


def test_detections_are_ordered_by_conf_then_label_then_box():
    xyxy = np.array([[5, 5, 9, 9], [1, 1, 4, 4], [0, 0, 2, 2], [3, 3, 8, 8], [0, 0, 1, 1]], float)
    conf = np.array([0.5, 0.9, 0.5, 0.5, 0.5])
    cls = np.array([2, 0, 2, 1, 2])
    out = D.to_detections(xyxy, conf, cls, CLASSES)
    assert [d.conf for d in out] == [0.9, 0.5, 0.5, 0.5, 0.5]
    assert [d.label for d in out] == ["outer_box", "red_box", "red_box", "red_box", "tray"]
    assert [d.box for d in out][1:4] == [(0, 0, 1, 1), (0, 0, 2, 2), (5, 5, 9, 9)]
    shuffled = D.to_detections(xyxy[::-1], conf[::-1], cls[::-1], CLASSES)
    assert shuffled == out  # input order never matters


def test_conversion_drops_ids_outside_the_classes_and_clips_conf_to_unit():
    xyxy = np.array([[0, 0, 1, 1], [0, 0, 2, 2], [0, 0, 3, 3]], float)
    out = D.to_detections(xyxy, np.array([0.7, 1.0000001, 0.8]), np.array([5, 1, -1]), CLASSES)
    assert [(d.label, d.conf) for d in out] == [("tray", 1.0)]
    assert isinstance(out[0], Detection)


def test_empty_arrays_give_an_empty_list():
    assert D.to_detections(np.zeros((0, 4)), np.zeros(0), np.zeros(0, int), CLASSES) == []


def test_swapped_corners_are_normalised_so_the_contract_accepts_them():
    out = D.to_detections(np.array([[9.0, 8.0, 1.0, 2.0]]), np.array([0.5]), np.array([0]), CLASSES)
    assert out[0].box == (1.0, 2.0, 9.0, 8.0)


# --- the two prediction functions (shared with training/eval_detector.py) -----------------


class _T:
    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class FakeYolo:
    names = dict(enumerate(CLASSES))

    def __init__(self, rows=()):
        self.rows = list(rows)
        self.calls = []

    def predict(self, img, **kw):
        self.calls.append((img, kw))
        boxes = types.SimpleNamespace(
            xyxy=_T(np.array([r[0] for r in self.rows], np.float32).reshape(-1, 4)),
            conf=_T(np.array([r[1] for r in self.rows], np.float32)),
            cls=_T(np.array([r[2] for r in self.rows], np.float32)),
        )
        return [types.SimpleNamespace(boxes=boxes)]


def test_yolo_predict_passes_bgr_unchanged_with_conf_size_and_device():
    m = FakeYolo([([1, 2, 3, 4], 0.8, 2)])
    xyxy, conf, cls = D.yolo_predict(m, IMG, conf=0.1, imgsz=384, device="cpu")
    img, kw = m.calls[0]
    assert img is IMG  # BGR ndarray as Ultralytics expects, not converted
    assert kw == {"conf": 0.1, "imgsz": 384, "device": "cpu", "verbose": False}
    assert xyxy.shape == (1, 4) and cls.dtype.kind == "i" and conf.dtype == np.float64


class FakeRfdetr:
    class_names = dict(enumerate(CLASSES))

    def __init__(self, xyxy, conf, cls):
        self.out = types.SimpleNamespace(
            xyxy=np.array(xyxy, float).reshape(-1, 4),
            confidence=np.array(conf, float),
            class_id=np.array(cls, int),
        )
        self.seen = None

    def predict(self, rgb, threshold, include_source_image):
        self.seen = (rgb, threshold, include_source_image)
        return self.out


def test_rfdetr_predict_converts_bgr_to_rgb_at_the_boundary_and_drops_the_spare_slot():
    img = np.zeros((4, 4, 3), np.uint8)
    img[..., 0] = 10  # blue channel in BGR
    m = FakeRfdetr([[0, 0, 1, 1], [0, 0, 2, 2]], [0.9, 0.8], [1, 5])
    xyxy, conf, cls = D.rfdetr_predict(m, img, 0.1, n_classes=5)
    rgb, thr, inc = m.seen
    assert rgb[..., 2].max() == 10 and rgb[..., 0].max() == 0 and rgb.flags["C_CONTIGUOUS"]
    assert img[..., 0].max() == 10  # the caller's array is untouched
    assert (thr, inc) == (0.1, False)
    assert cls.tolist() == [1] and conf.tolist() == [0.9]  # id 5 is the spare slot


def test_backends_return_the_same_detection_list_through_the_shared_conversion():
    rows = [([1, 1, 5, 5], 0.75, 3), ([2, 2, 6, 6], 0.875, 0)]
    y = D.YoloDetector(None, CLASSES, "a" * 64, model=FakeYolo(rows))
    r = D.RfdetrDetector(
        None, CLASSES, "b" * 64, model=FakeRfdetr([r[0] for r in rows], [0.75, 0.875], [3, 0])
    )
    assert y.detect(IMG) == r.detect(IMG)
    assert [d.label for d in y.detect(IMG)] == ["outer_box", "yellow_box"]


def test_yolo_detector_uses_the_floor_and_input_size_and_is_repeatable():
    m = FakeYolo([([1, 1, 5, 5], 0.6, 3)])
    det = D.YoloDetector(None, CLASSES, "a" * 64, model=m, device="cpu")
    a, b = det.detect(IMG), det.detect(IMG)
    assert a == b
    assert m.calls[0][1]["conf"] == DETECTOR_MIN_CONF and m.calls[0][1]["imgsz"] == 384
    assert (det.name, det.stamp_label, det.sha256) == ("yolo11n", "yolo11n", "a" * 64)


def test_yolo_detector_refuses_checkpoint_classes_that_differ_in_order():
    m = FakeYolo()
    m.names = {0: "tray", 1: "outer_box", 2: "red_box", 3: "yellow_box", 4: "start_button"}
    with pytest.raises(D.DetectorRefusal, match="class"):
        D.YoloDetector(None, CLASSES, "a" * 64, model=m)


def test_device_auto_falls_back_to_cpu_without_cuda(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert D.resolve_device("auto") == "cpu" and D.resolve_device("cpu") == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert D.resolve_device("auto") == "cuda"


# --- load_detector: precedence, refusals, warning ------------------------------------------


def _entry(name, file, data, classes=None, validated=True):
    return {
        "name": name,
        "file": file,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "classes": list(classes or CLASSES),
        "validated_for_pipeline": validated,
    }


@pytest.fixture
def setup(tmp_path):
    (tmp_path / "w").mkdir()
    (tmp_path / "w" / "y.pt").write_bytes(b"yolo-bytes")
    (tmp_path / "w" / "r.pth").write_bytes(b"rfdetr-bytes")
    cfg = tmp_path / "experiment.json"
    cfg.write_text(json.dumps({"classes": CLASSES}))
    manifest = {
        "active_detector": "yolo11n",
        "detectors": [
            _entry("yolo11n", "y.pt", b"yolo-bytes"),
            _entry("rfdetr_nano", "r.pth", b"rfdetr-bytes", validated=False),
        ],
    }
    mp = tmp_path / "w" / "MANIFEST.json"
    mp.write_text(json.dumps(manifest))

    made = []

    def factory_for(kind):
        def make(path, classes, sha256, **kw):
            made.append((kind, Path(path).name, sha256))
            if kind == "yolo11n":
                return D.YoloDetector(path, classes, sha256, model=FakeYolo())
            return D.RfdetrDetector(path, classes, sha256, model=FakeRfdetr([], [], []))

        return make

    backends = {"yolo11n": factory_for("yolo11n"), "rfdetr_nano": factory_for("rfdetr_nano")}
    return types.SimpleNamespace(mp=mp, cfg=cfg, made=made, backends=backends, tmp=tmp_path)


def _load(s, name=None, environ=None):
    return D.load_detector(
        manifest_path=s.mp, name=name, config_path=s.cfg, environ=environ or {}, backends=s.backends
    )


def test_precedence_is_argument_then_environment_then_manifest(setup):
    assert _load(setup).name == "yolo11n"  # manifest active_detector
    assert _load(setup, environ={"SIH_DETECTOR": "rfdetr_nano"}).name == "rfdetr_nano"
    both = _load(setup, name="yolo11n", environ={"SIH_DETECTOR": "rfdetr_nano"})
    assert both.name == "yolo11n"  # explicit argument beats the environment
    assert _load(setup, environ={"SIH_DETECTOR": ""}).name == "yolo11n"  # empty = unset


def test_the_hyphenated_stamp_label_is_accepted_as_a_name_and_maps_to_the_stamp(setup):
    d = _load(setup, name="rfdetr-nano")
    assert (d.name, d.stamp_label) == ("rfdetr_nano", "rfdetr-nano")
    assert d.sha256 == hashlib.sha256(b"rfdetr-bytes").hexdigest()


def test_an_unknown_name_is_refused_and_lists_the_choices(setup):
    with pytest.raises(D.DetectorRefusal, match="yolo11n.*rfdetr_nano"):
        _load(setup, name="yolov99")
    assert setup.made == []


def test_a_weights_hash_mismatch_is_refused_before_anything_is_loaded(setup):
    (setup.tmp / "w" / "y.pt").write_bytes(b"tampered")
    with pytest.raises(D.DetectorRefusal, match="sha256"):
        _load(setup)
    assert setup.made == []


def test_missing_weights_are_refused(setup):
    (setup.tmp / "w" / "y.pt").unlink()
    with pytest.raises(D.DetectorRefusal, match="not found"):
        _load(setup)


def test_manifest_classes_that_differ_from_the_config_are_refused(setup):
    cfg = json.loads(setup.cfg.read_text())
    cfg["classes"] = [*CLASSES[1:], CLASSES[0]]  # same set, different order
    setup.cfg.write_text(json.dumps(cfg))
    with pytest.raises(D.DetectorRefusal, match="classes"):
        _load(setup)
    assert setup.made == []


def test_a_detector_not_validated_for_the_pipeline_logs_a_warning(setup, caplog):
    with caplog.at_level(logging.WARNING, logger="perception.detector"):
        _load(setup)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="perception.detector"):
        _load(setup, name="rfdetr_nano")
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(msgs) == 1 and "validated_for_pipeline" in msgs[0] and "rfdetr_nano" in msgs[0]


def test_the_real_manifest_and_config_agree_on_classes_and_name_every_detector():
    root = Path(__file__).resolve().parents[3]
    m = json.loads((root / "weights" / "MANIFEST.json").read_text())
    cfg = json.loads((root / "config" / "experiment.json").read_text())
    assert m["active_detector"] in {d["name"] for d in m["detectors"]}
    for d in m["detectors"]:
        assert d["classes"] == cfg["classes"]
        assert d["name"] in D.STAMP_LABELS


# --- import hygiene ------------------------------------------------------------------------


def test_importing_the_module_imports_neither_ultralytics_nor_rfdetr():
    code = (
        "import sys, perception.detector\n"
        "bad = [m for m in ('ultralytics', 'rfdetr') if m in sys.modules]\n"
        "assert not bad, bad\n"
    )
    root = Path(__file__).resolve().parents[3]
    r = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
