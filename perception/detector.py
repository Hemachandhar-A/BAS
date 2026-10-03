"""perception/detector.py -- the object detector behind the pipeline (F2,
essential-features.md section 2). Owner: P1.

One prediction code path for the runtime *and* the evaluation: ``yolo_predict`` and
``rfdetr_predict`` return plain ``(xyxy, conf, class_id)`` arrays, and
``training/eval_detector.py`` imports them, so the numbers in ``reports/detector_eval_*.json``
describe what the pipeline runs. ``to_detections`` turns the arrays into contract
``Detection`` objects in a fixed order (confidence descending, then label, then box).

``load_detector`` picks the detector at **launch**, never during a run: the explicit ``name``
argument, else the environment variable ``SIH_DETECTOR``, else ``active_detector`` in
``weights/MANIFEST.json``. It checks the weights' sha256 against the manifest and the manifest's
classes against ``config/experiment.json`` (in order), and refuses on any mismatch.

Importing this module imports neither ``ultralytics`` (AGPL, optional group) nor ``rfdetr``: each
backend imports its library inside its own constructor. Images are BGR everywhere; RGB only at
the RF-DETR boundary. No clock reads, no randomness.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from contracts import DETECTOR_MIN_CONF, Detection, sha256_of_file

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST_PATH = Path("weights/MANIFEST.json")
DEFAULT_CONFIG_PATH = Path("config/experiment.json")
ENV_VAR = "SIH_DETECTOR"
INPUT_SIZE = 384
"""Both detectors were fine-tuned at 384 (``input_size`` in the manifest)."""
STAMP_LABELS = {"yolo11n": "yolo11n", "rfdetr_nano": "rfdetr-nano"}
"""Manifest detector name -> label for ``contracts.compose_model_stamp`` (underscore vs hyphen)."""

Arrays = tuple[np.ndarray, np.ndarray, np.ndarray]


class DetectorError(Exception):
    """A detector could not be built or used."""


class DetectorRefusal(DetectorError):
    """The detector must not be used (unknown name, hash or class mismatch, missing weights)."""


@runtime_checkable
class Detector(Protocol):
    """What the pipeline needs from a detector. ``detect`` takes a BGR ``uint8`` image and
    returns every detection at or above ``DETECTOR_MIN_CONF`` with a manifest class label, a box
    in original pixels and ``conf`` in [0, 1], ordered deterministically."""

    name: str
    sha256: str
    stamp_label: str

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]: ...

    def reset(self) -> None: ...


# --- pure helpers ---------------------------------------------------------------------------


def resolve_device(device: str) -> str:
    """``auto`` means CUDA when torch sees one, else CPU; an explicit device is kept."""
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def to_detections(
    xyxy: np.ndarray, conf: np.ndarray, class_id: np.ndarray, classes: list[str]
) -> list[Detection]:
    """Arrays -> ``Detection`` list. Ids outside ``0..len(classes)-1`` are dropped (the RF-DETR
    head has one spare slot); corners are put in order; ``conf`` is clipped to [0, 1]. The result
    is ordered by confidence descending, then label, then box, whatever the input order."""
    out: list[Detection] = []
    boxes = np.asarray(xyxy, float).reshape(-1, 4)
    for b, c, k in zip(boxes, np.asarray(conf, float), np.asarray(class_id, int), strict=True):
        if not 0 <= int(k) < len(classes):
            continue
        x1, y1, x2, y2 = (float(v) for v in b)
        out.append(
            Detection(
                label=classes[int(k)],
                conf=min(1.0, max(0.0, float(c))),
                box=(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)),
            )
        )
    out.sort(key=lambda d: (-d.conf, d.label, d.box))
    return out


def yolo_predict(model: Any, bgr: np.ndarray, *, conf: float, imgsz: int, device: Any) -> Arrays:
    """One Ultralytics prediction. The BGR ndarray goes in as it is (Ultralytics expects BGR for
    numpy input). Returns (xyxy (N,4) float, conf (N,) float, class_id (N,) int)."""
    r = model.predict(bgr, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
    return (
        r.boxes.xyxy.cpu().numpy().astype(float).reshape(-1, 4),
        r.boxes.conf.cpu().numpy().astype(float),
        r.boxes.cls.cpu().numpy().astype(int),
    )


def rfdetr_predict(model: Any, bgr: np.ndarray, floor: float, n_classes: int | None) -> Arrays:
    """One RF-DETR prediction. BGR becomes RGB here, at the model boundary, in a fresh
    contiguous buffer. ``n_classes`` drops ids outside ``0..n-1`` (the head's spare slot)."""
    rgb = np.ascontiguousarray(bgr[..., ::-1])
    det = model.predict(rgb, threshold=floor, include_source_image=False)
    n = len(det.xyxy)
    conf = det.confidence if det.confidence is not None else np.ones(n)
    cls = det.class_id if det.class_id is not None else np.zeros(n, int)
    xyxy, conf, cls = (
        np.asarray(det.xyxy, float).reshape(-1, 4),
        np.asarray(conf, float),
        np.asarray(cls, int),
    )
    if n_classes is not None:
        keep = (cls >= 0) & (cls < n_classes)
        xyxy, conf, cls = xyxy[keep], conf[keep], cls[keep]
    return xyxy, conf, cls


def _class_names(model: Any) -> list[str]:
    names = getattr(model, "class_names", None)
    if names is None:
        names = getattr(model, "names", None)
    if isinstance(names, Mapping):
        return [v for _, v in sorted(names.items())]
    return list(names or [])


# --- backends -------------------------------------------------------------------------------


class YoloDetector:
    """YOLO11n through Ultralytics (AGPL-3.0). ``model`` lets tests inject a fake; otherwise
    the weights are loaded here, importing ``ultralytics`` only now."""

    name = "yolo11n"

    def __init__(
        self,
        weights: Path | str | None,
        classes: list[str],
        sha256: str,
        *,
        imgsz: int = INPUT_SIZE,
        conf: float = DETECTOR_MIN_CONF,
        device: str = "auto",
        model: Any = None,
    ) -> None:
        if model is None:
            from ultralytics import YOLO

            model = YOLO(str(weights))
        names = _class_names(model)
        if names != list(classes):
            raise DetectorRefusal(f"checkpoint classes {names} != experiment classes {classes}")
        self._model = model
        self.classes = list(classes)
        self.sha256 = sha256
        self.stamp_label = STAMP_LABELS[self.name]
        self._imgsz = imgsz
        self._conf = conf
        dev = resolve_device(device)
        self._device: Any = 0 if dev == "cuda" else dev

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        xyxy, conf, cls = yolo_predict(
            self._model, frame_bgr, conf=self._conf, imgsz=self._imgsz, device=self._device
        )
        return to_detections(xyxy, conf, cls, self.classes)

    def reset(self) -> None:
        """Stateless between frames: nothing to clear."""


class RfdetrDetector:
    """RF-DETR-Nano (Apache-2.0). ``model`` lets tests inject a fake."""

    name = "rfdetr_nano"

    def __init__(
        self,
        weights: Path | str | None,
        classes: list[str],
        sha256: str,
        *,
        conf: float = DETECTOR_MIN_CONF,
        device: str = "auto",
        model: Any = None,
    ) -> None:
        if model is None:
            from rfdetr import RFDETR

            model = RFDETR.from_checkpoint(
                str(weights), trust_checkpoint=True, device=resolve_device(device)
            )
        names = _class_names(model)
        if names != list(classes):
            raise DetectorRefusal(f"checkpoint classes {names} != experiment classes {classes}")
        self._model = model
        self.classes = list(classes)
        self.sha256 = sha256
        self.stamp_label = STAMP_LABELS[self.name]
        self._conf = conf

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        xyxy, conf, cls = rfdetr_predict(self._model, frame_bgr, self._conf, len(self.classes))
        return to_detections(xyxy, conf, cls, self.classes)

    def reset(self) -> None:
        """Stateless between frames: nothing to clear."""


BACKENDS: dict[str, Callable[..., Detector]] = {
    "yolo11n": YoloDetector,
    "rfdetr_nano": RfdetrDetector,
}


# --- the launch-time factory ----------------------------------------------------------------


def _canonical(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def load_detector(
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    name: str | None = None,
    *,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    environ: Mapping[str, str] | None = None,
    device: str = "auto",
    backends: Mapping[str, Callable[..., Detector]] | None = None,
) -> Detector:
    """Build the detector chosen at launch. Precedence: ``name``, then ``$SIH_DETECTOR``, then
    the manifest's ``active_detector``. Refuses (``DetectorRefusal``) on an unknown name, missing
    weights, a weights sha256 that differs from the manifest, or manifest classes that differ
    from ``config/experiment.json`` in order. Warns when ``validated_for_pipeline`` is false."""
    env = os.environ if environ is None else environ
    mpath = Path(manifest_path)
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    entries = {_canonical(d["name"]): d for d in manifest["detectors"]}

    chosen = (name or "").strip() or (env.get(ENV_VAR) or "").strip() or manifest["active_detector"]
    entry = entries.get(_canonical(chosen))
    if entry is None:
        raise DetectorRefusal(
            f"unknown detector {chosen!r}; the manifest lists "
            f"{[d['name'] for d in entries.values()]}"
        )
    dname = entry["name"]

    config_classes = json.loads(Path(config_path).read_text(encoding="utf-8"))["classes"]
    if list(entry["classes"]) != list(config_classes):
        raise DetectorRefusal(
            f"manifest classes {entry['classes']} != experiment classes {config_classes} "
            "(same names in the same order are required)"
        )
    weights = mpath.parent / entry["file"]
    if not weights.is_file():
        raise DetectorRefusal(f"weights not found: {weights}")
    actual = sha256_of_file(weights)
    if actual != entry["sha256"]:
        raise DetectorRefusal(
            f"weights sha256 {actual} != manifest sha256 {entry['sha256']} for {dname} ({weights})"
        )
    if not entry.get("validated_for_pipeline", False):
        logger.warning(
            "detector %s has validated_for_pipeline=false in the manifest: it has not been "
            "accepted for the pipeline (loading it anyway because it was asked for)",
            dname,
        )
    factory = (backends or BACKENDS).get(dname)
    if factory is None:
        raise DetectorRefusal(f"no backend for detector {dname!r}")
    kwargs: dict[str, Any] = {} if backends is not None else {"device": device}
    if backends is None and "input_size" in entry and dname == "yolo11n":
        kwargs["imgsz"] = int(entry["input_size"])
    detector = factory(weights, list(entry["classes"]), entry["sha256"], **kwargs)
    logger.info("detector %s loaded (stamp label %s)", dname, detector.stamp_label)
    return detector
