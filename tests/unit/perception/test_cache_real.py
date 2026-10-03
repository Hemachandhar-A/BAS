"""P1.7.3: on the real caches. A cache built with YOLO11n is refused by a reader that expects the
RF-DETR stamp, and rebuilding a run gives identical bytes. Skips cleanly when the weights, the
videos or the caches are absent (all git-ignored)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from contracts import compose_model_stamp
from perception import cache as C

pytestmark = [pytest.mark.F2, pytest.mark.F14, pytest.mark.slow]

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "weights" / "MANIFEST.json"
RUN = "x015"


def _needs():
    pytest.importorskip("ultralytics")
    pytest.importorskip("mediapipe")
    need = [
        ROOT / "weights" / "detector_yolo11n.pt",
        ROOT / "weights" / "hand_landmarker.task",
        ROOT / "runs" / RUN / "video.mp4",
        C.cache_path(ROOT / "data" / "cache", RUN),
    ]
    if not all(p.is_file() for p in need):
        pytest.skip("weights, video or cache not present in this checkout")


def _stamps():
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    by = {d["name"]: d for d in m["detectors"]}
    hand = m["hand"]["sha256"]
    yolo = compose_model_stamp(by["yolo11n"]["sha256"], hand, "none", detector_name="yolo11n")
    rf = compose_model_stamp(by["rfdetr_nano"]["sha256"], hand, "none", detector_name="rfdetr-nano")
    return yolo, rf


def test_a_yolo_cache_is_refused_for_the_rfdetr_stamp_and_accepted_for_its_own():
    _needs()
    yolo, rf = _stamps()
    p = C.cache_path(ROOT / "data" / "cache", RUN)
    assert C.read_header(p, expected_stamp=yolo).model_stamp == yolo
    with pytest.raises(C.CacheMismatch, match="model_stamp"):
        C.read_cache(p, expected_stamp=rf)


def test_rebuilding_a_run_gives_identical_bytes(tmp_path):
    _needs()
    from perception.pipeline import load_pipeline

    pipe = load_pipeline(detector="yolo11n")
    existing = C.cache_path(ROOT / "data" / "cache", RUN)
    hdr = C.read_header(existing)
    r = C.build_run(
        RUN, ROOT / "runs" / RUN / "video.mp4", pipe, fps=hdr.fps,
        experiment_id=hdr.experiment_id, out_root=tmp_path,
    )  # fmt: skip
    assert r["status"] == "built"
    rebuilt = C.cache_path(tmp_path, RUN)
    a, b = existing.read_bytes(), rebuilt.read_bytes()
    assert hashlib.sha256(a).hexdigest() == hashlib.sha256(b).hexdigest() and a == b
