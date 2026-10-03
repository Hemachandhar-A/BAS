"""F2: the runtime YoloDetector returns what the evaluation measured. The reference below is the
evaluation's original inline prediction (``conf=0.001``, ``imgsz=384``, the result then filtered
at ``DETECTOR_MIN_CONF``); the new path must give the same boxes, classes and confidences on 20
valid images. Skips cleanly when the weights, ultralytics or the dataset are absent."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from contracts import DETECTOR_MIN_CONF

pytestmark = [pytest.mark.F2, pytest.mark.slow]

ROOT = Path(__file__).resolve().parents[3]
WEIGHTS = ROOT / "weights" / "detector_yolo11n.pt"
VALID = ROOT / "data" / "dataset" / "valid"


def test_yolo_detector_equals_the_original_evaluation_path_on_20_valid_images():
    pytest.importorskip("ultralytics")
    cv2 = pytest.importorskip("cv2")
    if not WEIGHTS.is_file() or not VALID.is_dir():
        pytest.skip("yolo11n weights or the valid split are not present in this checkout")
    from ultralytics import YOLO

    from perception.detector import load_detector

    names = sorted(p.name for p in VALID.glob("*.jpg"))
    assert len(names) >= 20
    picked = names[:: len(names) // 20][:20]
    det = load_detector(ROOT / "weights" / "MANIFEST.json", name="yolo11n",
                        config_path=ROOT / "config" / "experiment.json", device="cpu")  # fmt: skip
    classes = json.loads((ROOT / "config" / "experiment.json").read_text())["classes"]
    ref_model = YOLO(str(WEIGHTS))

    total = 0
    for n in picked:
        bgr = cv2.imread(str(VALID / n), cv2.IMREAD_COLOR)
        r = ref_model.predict(bgr, conf=0.001, imgsz=384, device="cpu", verbose=False)[0]
        xyxy = r.boxes.xyxy.cpu().numpy().astype(float).reshape(-1, 4)
        conf = r.boxes.conf.cpu().numpy().astype(float)
        cls = r.boxes.cls.cpu().numpy().astype(int)
        keep = conf >= DETECTOR_MIN_CONF
        rows = zip(xyxy[keep], conf[keep], cls[keep], strict=True)
        ref = [(classes[int(k)], float(c), tuple(b)) for b, c, k in rows]
        ref.sort(key=lambda t: (-t[1], t[0], t[2]))
        got = det.detect(bgr)
        assert len(got) == len(ref), (n, len(got), len(ref))
        for g, (label, c, box) in zip(got, ref, strict=True):
            assert g.label == label
            assert g.conf == pytest.approx(c, abs=1e-4)
            assert np.allclose(g.box, box, atol=1e-4)
        total += len(got)
    assert total > 0
    print(f"equivalence: {len(picked)} images, {total} detections compared")
