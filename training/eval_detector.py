"""F14 stage 7: evaluate a fine-tuned detector on a split of the dataset.

  python -m training.eval_detector --model rfdetr|yolo11n --weights PATH --dataset-dir data/dataset
        --split valid|test --out reports/detector_eval.json
        [--corrections-dir data/corrections] [--acceptance config/acceptance.yaml]
        [--dataset-report reports/dataset.json] [--device auto] [--trust-checkpoint]

Per-class precision and recall at ``detector_conf_floor`` (read from ``PerceptionConfig``, never
hardcoded) at IoU 0.5, and COCO-style mAP50 and mAP50-95 computed on predictions down to a very
low confidence, as mAP requires. All of it comes from ``supervision.metrics``; nothing is
re-implemented here. **Every metric is reported twice**: on all labelled frames of the split
(agreement with the auto-labeler plus the crew's corrections) and on the **verified gold subset**
(frames with ``verified: true`` in ``data/corrections/<split>.json``); the gold numbers are the
ones checked against ``config/acceptance.yaml``. A subset with no frames reports ``null`` and a
note, never a zero.

``--split test`` is **refused** unless ``config/acceptance.yaml`` already exists (it must be
written before anyone looks at ``test``, AGENTS.md rule 12), and the file's sha256 is then
recorded in the report. The report also carries ``generated_at``, the weights sha256, the
dataset stamp and the confidence floor used. Images are read as BGR; they are converted to RGB
only at the RF-DETR boundary (YOLO takes BGR). Classes the detector reports must equal
``config/experiment.json`` in order, or it refuses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from training.autolabel import MOVABLE_CLASSES, _intersects

REPORT_VERSION = 1
MAP_CONF_FLOOR = 0.001
"""Predictions are collected down to this confidence for mAP (it integrates over the whole
precision-recall curve); precision and recall use only those at or above the config floor."""
OVERLAY_SPLIT = {"train": "train", "valid": "val", "test": "test"}

Predict = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]


class Refusal(Exception):
    """The evaluation must not run (test hygiene, or a class-order mismatch)."""


# --- gates and small pure helpers -------------------------------------------------------------


def acceptance_gate(split: str, acceptance_path: Path) -> str | None:
    """sha256 of ``acceptance.yaml`` if it exists, else None; for ``test`` its absence refuses."""
    p = Path(acceptance_path)
    if p.is_file():
        return hashlib.sha256(p.read_bytes()).hexdigest()
    if split == "test":
        raise Refusal(
            f"refusing --split test: config/acceptance.yaml ({p}) does not exist. "
            "The thresholds must be written before anyone looks at test (AGENTS.md rule 12)."
        )
    return None


def overlay_split(folder: str) -> str:
    return OVERLAY_SPLIT[folder]


def verified_files(overlay: dict | None) -> set[str]:
    """File names of the gold frames: ``verified: true`` and not excluded."""
    if not overlay:
        return set()
    return {
        name
        for name, e in overlay.get("frames", {}).items()
        if e.get("verified") is True and not e.get("excluded")
    }


def config_floor() -> float:
    from contracts import PerceptionConfig

    return float(PerceptionConfig().detector_conf_floor)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --- ground truth and metrics -----------------------------------------------------------------


def load_ground_truth(split_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """file_name -> (xyxy (N,4) float, class_id (N,) int); every image present, maybe empty."""
    coco = json.loads((Path(split_dir) / "_annotations.coco.json").read_text(encoding="utf-8"))
    names = {i["id"]: i["file_name"] for i in coco["images"]}
    boxes: dict[str, list[list[float]]] = {n: [] for n in names.values()}
    cls: dict[str, list[int]] = {n: [] for n in names.values()}
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        boxes[names[a["image_id"]]].append([x, y, x + w, y + h])
        cls[names[a["image_id"]]].append(int(a["category_id"]))
    return {n: (np.array(boxes[n], float).reshape(-1, 4), np.array(cls[n], int)) for n in boxes}


def _detections(xyxy: np.ndarray, class_id: np.ndarray, conf: np.ndarray | None = None) -> Any:
    import supervision as sv

    return sv.Detections(
        xyxy=np.asarray(xyxy, float).reshape(-1, 4),
        class_id=np.asarray(class_id, int),
        confidence=None if conf is None else np.asarray(conf, float),
    )


def evaluate_subset(
    gt: dict[str, tuple[np.ndarray, np.ndarray]],
    pred: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    files: set[str],
    classes: list[str],
    floor: float,
) -> dict[str, Any]:
    """Metrics over the images in ``files``. ``pred`` holds every prediction down to a low
    confidence; precision/recall use those with confidence >= ``floor``."""
    from supervision.metrics import AveragingMethod, MeanAveragePrecision, Precision, Recall

    names = sorted(set(files) & set(gt))
    per_class: dict[str, dict[str, Any]] = {
        c: {"precision": None, "recall": None, "n_gt": 0, "n_pred_at_floor": 0} for c in classes
    }
    result: dict[str, Any] = {
        "n_images": len(names),
        "map50": None,
        "map50_95": None,
        "per_class": per_class,
    }
    if not names:
        return result

    targets, preds_all, preds_floor = [], [], []
    for n in names:
        gxyxy, gcls = gt[n]
        pxyxy, pconf, pcls = pred[n]
        keep = pconf >= floor
        targets.append(_detections(gxyxy, gcls))
        preds_all.append(_detections(pxyxy, pcls, pconf))
        preds_floor.append(_detections(pxyxy[keep], pcls[keep], pconf[keep]))
        for cid, c in enumerate(classes):
            per_class[c]["n_gt"] += int((gcls == cid).sum())
            per_class[c]["n_pred_at_floor"] += int((pcls[keep] == cid).sum())
    if sum(c["n_gt"] for c in per_class.values()) == 0:
        return result  # nothing to score against: leave everything null

    m = MeanAveragePrecision()
    m.update(preds_all, targets)
    mr = m.compute()
    result["map50"], result["map50_95"] = float(mr.map50), float(mr.map50_95)

    for metric, key, attr in (
        (Precision(averaging_method=AveragingMethod.MACRO), "precision", "precision_per_class"),
        (Recall(averaging_method=AveragingMethod.MACRO), "recall", "recall_per_class"),
    ):
        metric.update(preds_floor, targets)
        r = metric.compute()
        table = getattr(r, attr)  # (n matched classes, n IoU thresholds); column 0 is IoU 0.5
        rows = {int(c): float(table[i][0]) for i, c in enumerate(r.matched_classes)}
        for cid, c in enumerate(classes):
            if key == "precision" and per_class[c]["n_pred_at_floor"] == 0:
                continue  # no predictions at the floor: precision is undefined, not 0
            if key == "recall" and per_class[c]["n_gt"] == 0:
                continue  # nothing to recall
            per_class[c][key] = rows.get(cid, 0.0)
    return result


def _parse_frame_name(name: str) -> tuple[str, int] | None:
    """``<run>_<frame id>.jpg`` -> (run, frame id); None for any other name."""
    stem = name[:-4] if name.endswith(".jpg") else name
    run, _, fid = stem.rpartition("_")
    return (run, int(fid)) if run and fid.isdigit() else None


def split_by_hand(
    gt: dict[str, tuple[np.ndarray, np.ndarray]],
    hands: dict[tuple[str, int], list[tuple[float, ...]]],
    files: set[str],
    classes: list[str],
) -> tuple[set[str], set[str], set[str]]:
    """(hand over a container, hand elsewhere, no hand data) for the frames in ``files``. The
    first group is build_dataset's ``with_hand_over_container``: a hand box intersects a ground
    truth red or yellow box. A frame without a hand record is never guessed into either group."""
    movable = {classes.index(c) for c in MOVABLE_CLASSES}
    over: set[str] = set()
    away: set[str] = set()
    none: set[str] = set()
    for name in sorted(set(files) & set(gt)):
        key = _parse_frame_name(name)
        if key is None or key not in hands:
            none.add(name)
            continue
        xyxy, cls = gt[name]
        boxes = [tuple(b) for b, c in zip(xyxy, cls, strict=True) if int(c) in movable]
        hit = any(_intersects(h, b) for h in hands[key] for b in boxes)
        (over if hit else away).add(name)
    return over, away, none


def red_yellow_confusion(
    gt: dict[str, tuple[np.ndarray, np.ndarray]],
    pred: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    files: set[str],
    classes: list[str],
    floor: float,
) -> dict[str, dict[str, int]]:
    """For every ground-truth red and yellow box: the class of the best-IoU (>= 0.5) prediction
    at or above ``floor``, or ``missed``. Rows are the true class, columns what was predicted."""
    import supervision as sv

    out = {
        c: {"red_box": 0, "yellow_box": 0, "other_class": 0, "missed": 0} for c in MOVABLE_CLASSES
    }
    ids = {classes.index(c): c for c in MOVABLE_CLASSES}
    for name in sorted(set(files) & set(gt)):
        gxyxy, gcls = gt[name]
        pxyxy, pconf, pcls = pred[name]
        keep = pconf >= floor
        pxyxy, pcls = pxyxy[keep], pcls[keep]
        for i, c in enumerate(gcls):
            if int(c) not in ids:
                continue
            row = out[ids[int(c)]]
            if len(pxyxy) == 0:
                row["missed"] += 1
                continue
            iou = sv.box_iou_batch(np.asarray(gxyxy[i : i + 1], float), pxyxy)[0]
            j = int(np.argmax(iou))
            if iou[j] < 0.5:
                row["missed"] += 1
            else:
                row[
                    ids.get(int(pcls[j]), "other_class") if int(pcls[j]) in ids else "other_class"
                ] += 1
    return out


def eligibility(report: dict, min_recall: float, min_map50: float) -> dict[str, Any]:
    """Pre-registered rule (1): every class recall >= ``min_recall`` and mAP50 >= ``min_map50``
    on the gold subset. A missing number fails; it never passes."""
    gold = report["subsets"]["gold"]
    classes = {}
    for c, m in gold["per_class"].items():
        r = m["recall"]
        classes[c] = {
            "recall": r,
            "threshold": min_recall,
            "pass": r is not None and r >= min_recall,
        }
    mp = gold["map50"]
    map_row = {"map50": mp, "threshold": min_map50, "pass": mp is not None and mp >= min_map50}
    return {
        "classes": classes,
        "map50": map_row,
        "eligible": all(r["pass"] for r in classes.values()) and map_row["pass"],
    }


def eligibility_table(name: str, e: dict[str, Any]) -> str:
    def f(v: float | None) -> str:
        return "n/a" if v is None else f"{v:.4f}"

    lines = [
        f"{name}: rule (1) on the gold subset",
        f"  {'check':<14}{'value':>8}{'needs':>8}  result",
    ]
    for c, r in e["classes"].items():
        verdict = "PASS" if r["pass"] else "FAIL"
        lines.append(f"  {'recall ' + c:<14}{f(r['recall']):>8}{r['threshold']:>8}  {verdict}")
    m = e["map50"]
    lines.append(
        f"  {'mAP50':<14}{f(m['map50']):>8}{m['threshold']:>8}  {'PASS' if m['pass'] else 'FAIL'}"
    )
    lines.append(f"  => {'ELIGIBLE' if e['eligible'] else 'NOT ELIGIBLE'}")
    return "\n".join(lines)


def build_report(
    *,
    gt: dict,
    pred: dict,
    gold: set[str],
    classes: list[str],
    floor: float,
    model: str,
    split: str,
    weights_sha256: str,
    dataset_stamp: str | None,
    generated_at: str,
    acceptance_sha256: str | None,
    hands: dict[tuple[str, int], list[tuple[float, ...]]] | None = None,
) -> dict[str, Any]:
    all_sub = evaluate_subset(gt, pred, set(gt), classes, floor)
    gold_sub = evaluate_subset(gt, pred, gold, classes, floor)
    all_sub["note"] = "all labelled frames of the split: agreement with the (corrected) labels"
    gold_sub["note"] = (
        "verified gold frames (verified: true in the corrections overlay); "
        "the numbers checked against acceptance.yaml"
        if gold_sub["n_images"]
        else "no verified gold frames in this split: nothing to report here"
    )
    all_sub["red_yellow_confusion"] = red_yellow_confusion(gt, pred, set(gt), classes, floor)
    gold_sub["red_yellow_confusion"] = red_yellow_confusion(gt, pred, gold, classes, floor)
    if hands is not None:
        groups = dict(
            zip(
                ("with_hand_over_container", "hand_away", "no_hand_data"),
                split_by_hand(gt, hands, gold, classes),
                strict=True,
            )
        )
        gold_sub["by_hand"] = {
            k: evaluate_subset(gt, pred, files, classes, floor) for k, files in groups.items()
        }
    return {
        "version": REPORT_VERSION,
        "generated_at": generated_at,
        "model": model,
        "split": split,
        "classes": list(classes),
        "weights_sha256": weights_sha256,
        "dataset_stamp": dataset_stamp,
        "detector_conf_floor": floor,
        "map_conf_floor": MAP_CONF_FLOOR,
        "iou_for_precision_recall": 0.5,
        "acceptance_sha256": acceptance_sha256,
        "subsets": {"all": all_sub, "gold": gold_sub},
    }


# --- detectors (the model boundary) -----------------------------------------------------------


def rfdetr_predictor(
    model: Any, floor: float = MAP_CONF_FLOOR, n_classes: int | None = None
) -> Predict:
    """``n_classes``: drop detections whose class id is outside 0..n-1. The head has one spare
    slot (num_classes + 1 outputs) that a trained model never fires; a random head can."""

    def predict(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rgb = np.ascontiguousarray(bgr[..., ::-1])  # RGB only at the model boundary
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

    return predict


def load_predictor(
    model: str, weights: Path, device: str, classes: list[str], trust_checkpoint: bool = False
) -> Predict:
    if model == "rfdetr":
        from rfdetr import RFDETR

        kw: dict[str, Any] = {} if device == "auto" else {"device": device}
        m = RFDETR.from_checkpoint(str(weights), trust_checkpoint=trust_checkpoint, **kw)
        names = list(getattr(m, "class_names", None) or [])
        if isinstance(getattr(m, "class_names", None), dict):
            names = [v for _, v in sorted(m.class_names.items())]
        if names != list(classes):
            raise Refusal(f"checkpoint class names {names} != experiment classes {list(classes)}")
        return rfdetr_predictor(m, n_classes=len(classes))
    if model == "yolo11n":
        from ultralytics import YOLO

        y = YOLO(str(weights))
        names = [n for _, n in sorted(y.names.items())]
        if names != list(classes):
            raise Refusal(f"checkpoint class names {names} != experiment classes {list(classes)}")
        dev = "cpu" if device == "cpu" else (0 if device in ("auto", "cuda") else device)

        def predict(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            r = y.predict(bgr, conf=MAP_CONF_FLOOR, imgsz=384, device=dev, verbose=False)[0]
            return (
                r.boxes.xyxy.cpu().numpy().astype(float).reshape(-1, 4),
                r.boxes.conf.cpu().numpy().astype(float),
                r.boxes.cls.cpu().numpy().astype(int),
            )

        return predict
    raise ValueError(f"unknown model {model!r}")


# --- CLI --------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=("rfdetr", "yolo11n"), required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--split", choices=("valid", "test"), required=True)
    ap.add_argument("--out", type=Path, default=Path("reports/detector_eval.json"))
    ap.add_argument("--corrections-dir", type=Path, default=Path("data/corrections"))
    ap.add_argument("--acceptance", type=Path, default=Path("config/acceptance.yaml"))
    ap.add_argument("--dataset-report", type=Path, default=Path("reports/dataset.json"))
    ap.add_argument("--config", type=Path, default=Path("config/experiment.json"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--trust-checkpoint", action="store_true")
    args = ap.parse_args(argv)

    try:
        acceptance_sha = acceptance_gate(args.split, args.acceptance)
    except Refusal as e:
        print(str(e), file=sys.stderr)
        return 2
    if not args.weights.is_file():
        print(f"weights not found: {args.weights}", file=sys.stderr)
        return 3

    import cv2

    from training.autolabel import _hands_cache, load_classes
    from training.finetune import read_dataset_stamp

    classes = load_classes(args.config)
    split_dir = args.dataset_dir / args.split
    gt = load_ground_truth(split_dir)
    overlay_path = args.corrections_dir / f"{overlay_split(args.split)}.json"
    overlay = (
        json.loads(overlay_path.read_text(encoding="utf-8")) if overlay_path.is_file() else None
    )
    gold = verified_files(overlay) & set(gt)

    try:
        predict = load_predictor(
            args.model, args.weights, args.device, classes, trust_checkpoint=args.trust_checkpoint
        )
    except Refusal as e:
        print(f"refusing: {e}", file=sys.stderr)
        return 2

    pred = {}
    for name in sorted(gt):
        bgr = cv2.imread(str(split_dir / name), cv2.IMREAD_COLOR)  # BGR
        if bgr is None:
            print(f"cannot read image {split_dir / name}", file=sys.stderr)
            return 3
        pred[name] = predict(bgr)

    report = build_report(
        gt=gt,
        pred=pred,
        gold=gold,
        classes=classes,
        floor=config_floor(),
        model=args.model,
        split=args.split,
        weights_sha256=sha256_file(args.weights),
        dataset_stamp=read_dataset_stamp(args.dataset_report),
        generated_at=datetime.now().isoformat(timespec="seconds"),
        acceptance_sha256=acceptance_sha,
        hands=_hands_cache(),
    )
    acc_text = args.acceptance.read_text(encoding="utf-8") if acceptance_sha else ""
    acc = yaml.safe_load(acc_text) or {}
    det = acc.get("detector", {})
    if "min_recall_per_class" in det and "min_map50" in det:
        report["eligibility"] = eligibility(
            report, float(det["min_recall_per_class"]), float(det["min_map50"])
        )
        print(eligibility_table(args.model, report["eligibility"]), file=sys.stderr)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
