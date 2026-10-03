"""F14 stage 5: build the RF-DETR dataset from the auto-labels plus the review overlays.

  python -m training.build_dataset [--review-done]

Reads ``data/frames/index.json`` (sampled frames), ``data/labels/frame_labels.json``
(auto-label status and boxes per frame, written by ``training.autolabel --build``),
``runs/manifest.csv`` (the split of every run) and ``data/corrections/<split>.json``
(the label editor's overlay, Plan 5.8; the auto-labels are never edited). Writes
``data/dataset/{train,valid,test}/`` (our ``val`` is RF-DETR's ``valid``): the JPEGs and
``_annotations.coco.json`` per split, plus ``reports/dataset.json``.

Overlay (Plan 5.8): ``frames["<run>_<frame>.jpg"]`` REPLACES that frame's boxes wholesale
(and can bring back a frame the auto-labeler excluded); ``excluded: true`` drops the frame;
``verified: true`` marks a gold frame; ``static_overrides[run][class]`` replaces that
class's box on every auto-labeled frame of the run unless the frame has its own entry. A
static override never resurrects a frame the labeler excluded: only a frame entry of its own
can bring such a frame back, so an excluded frame stays excluded under a static override.
An overlay box must be a real box inside the image (x1 < x2, y1 < y2, within the frame's
width and height); a reversed, zero-size or out-of-image box, a box that does not have four
numbers, or a class that appears twice in one entry is a ``BuildError`` naming the frame (or the
run and class of a static override). Nothing is clipped silently.

Stamp: ``dataset_stamp`` hashes the three ``_annotations.coco.json`` files and the overlay
files, NOT the JPEG bytes. The images are unmodified copies of the frame cache, which is a
deterministic function of the run videos (``video_sha256`` in ``runs/manifest.csv``), and the
annotation files name every image and its size, so a changed image set changes the stamp; a
re-encoded frame with the same name would not. That is accepted to keep the stamp cheap and
independent of the JPEG encoder; ``reports/dataset.json`` records the image counts per split.

Report: ``reports/dataset.json`` keeps its ``label_review`` block across rebuilds; a build
without ``--review-done`` never replaces a recorded block with ``not_recorded``.

Assertions: no run in two splits; the frame index agrees with the manifest (so ``train`` holds
only train runs); no frame left unlabeled; overlay version 1, its ``split`` equals its file
name, every overlay class is in ``experiment.classes``, every overlay frame exists and belongs
to the overlay's split, every static override names a run of that split; class names equal
``experiment.classes`` in order and ids are contiguous from 0. Split by run, never by frame.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

from training.autolabel import MOVABLE_CLASSES, _intersects, build_coco

SPLITS = ("train", "val", "test")
FOLDER = {"train": "train", "val": "valid", "test": "test"}
OVERLAY_VERSION = 1


class BuildError(Exception):
    """A dataset assertion failed."""


# --- pure helpers ---------------------------------------------------------------------


def check_coco(coco: dict, classes: list[str]) -> None:
    """Class names equal ``classes`` in order, category ids are 0..n-1, every box
    lies inside its image."""
    cats = coco["categories"]
    if [c["name"] for c in cats] != list(classes):
        raise BuildError(
            f"category names {[c['name'] for c in cats]} != experiment classes {list(classes)}"
        )
    if [c["id"] for c in cats] != list(range(len(classes))):
        raise BuildError(f"category ids are not contiguous from 0: {[c['id'] for c in cats]}")
    dims = {i["id"]: (i["width"], i["height"]) for i in coco["images"]}
    for a in coco["annotations"]:
        w, h = dims[a["image_id"]]
        x, y, bw, bh = a["bbox"]
        if x < 0 or y < 0 or x + bw > w + 1e-6 or y + bh > h + 1e-6:
            raise BuildError(f"annotation {a['id']} has a box outside its image")


def dataset_stamp(files: list[Path]) -> str:
    """sha256 prefix (16 hex) over the given files, each as ``<parent>/<name>`` plus its
    bytes, in sorted order of that label."""
    labelled = sorted((f"{Path(p).parent.name}/{Path(p).name}", Path(p)) for p in files)
    h = hashlib.sha256()
    for label, p in labelled:
        h.update(label.encode("utf-8") + b"\0")
        h.update(p.read_bytes() + b"\0")
    return h.hexdigest()[:16]


def _parse_file_name(name: str) -> tuple[str, int]:
    stem = name[:-4] if name.endswith(".jpg") else name
    run, _, fid = stem.rpartition("_")
    return run, int(fid)


def _boxes_equal(a: dict[str, list[float]], b: dict[str, list[float]]) -> bool:
    return {k: list(v) for k, v in a.items()} == {k: list(v) for k, v in b.items()}


def check_overlay_box(box: object, width: int, height: int, where: str) -> None:
    """``box`` must be four finite numbers with x1 < x2 and y1 < y2, inside the image.
    ``where`` names the frame (or run and class) in the message."""
    try:
        x1, y1, x2, y2 = (float(v) for v in box)  # type: ignore[union-attr]
        four = len(box) == 4  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise BuildError(f"{where}: box {box!r} must be four numbers [x1, y1, x2, y2]") from None
    if not four or not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        raise BuildError(f"{where}: box {box!r} must be four finite numbers [x1, y1, x2, y2]")
    if x2 < x1 or y2 < y1:
        raise BuildError(f"{where}: reversed box {list(box)} (needs x1 < x2 and y1 < y2)")  # type: ignore[call-overload]
    if x2 == x1 or y2 == y1:
        raise BuildError(f"{where}: zero-size box {list(box)}")  # type: ignore[call-overload]
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        raise BuildError(f"{where}: box {list(box)} is outside the {width}x{height} image")  # type: ignore[call-overload]


def merge_label_review(existing_report: dict | None, new: dict | None) -> dict:
    """The ``label_review`` block of the new report: ``new`` if given, else the block already
    recorded in the existing report, else ``not_recorded``. A recorded block is never replaced
    by ``not_recorded``."""
    if new is not None:
        return new
    old = (existing_report or {}).get("label_review")
    if isinstance(old, dict) and old.get("status") != "not_recorded" and old:
        return old
    return {"status": "not_recorded"}


def hand_container_share(
    final: dict[str, dict[str, list[float]]],
    hands: dict[tuple[str, int], list[tuple[float, ...]]],
) -> dict:
    """Proxy for the hard case: of the kept frames (``final``), how many have a hand box that
    intersects a red or yellow box (``MOVABLE_CLASSES``). Frames with no hand data are counted
    separately, never guessed."""
    with_hand = no_data = 0
    for name, boxes in final.items():
        run, fid = _parse_file_name(name)
        if (run, fid) not in hands:
            no_data += 1
            continue
        if any(
            _intersects(h, boxes[c])
            for h in hands[(run, fid)]
            for c in MOVABLE_CLASSES
            if c in boxes
        ):
            with_hand += 1
    return {
        "kept_frames": len(final),
        "with_hand_over_container": with_hand,
        "no_hand_data": no_data,
    }


def _load_overlay(path: Path, split: str) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != OVERLAY_VERSION:
        raise BuildError(f"{path.name}: overlay version must be {OVERLAY_VERSION}")
    if data.get("split") != split:
        raise BuildError(f"{path.name}: overlay split {data.get('split')!r} != {split!r}")
    return data


def _validate_overlay(
    overlay: dict,
    split: str,
    classes: list[str],
    files_of_split: set[str],
    all_files: set[str],
    runs_of_split: set[str],
    all_runs: set[str],
    dims: dict[str, tuple[int, int]],
) -> None:
    for name, entry in overlay.get("frames", {}).items():
        if name not in all_files:
            raise BuildError(f"overlay frame {name} does not exist in the sampled frames")
        if name not in files_of_split:
            raise BuildError(f"overlay for split {split} touches {name}, which is in another split")
        run, _ = _parse_file_name(name)
        seen: set[str] = set()
        for box in entry.get("boxes", []):
            if box["class"] not in classes:
                raise BuildError(f"overlay class {box['class']!r} is not in experiment.classes")
            if box["class"] in seen:
                raise BuildError(f"overlay frame {name}: duplicate class {box['class']!r}")
            seen.add(box["class"])
            check_overlay_box(box["xyxy"], *dims[run], f"overlay frame {name} {box['class']}")
    for run, per_class in overlay.get("static_overrides", {}).items():
        if run not in all_runs:
            raise BuildError(f"static override names run {run}, which does not exist")
        if run not in runs_of_split:
            raise BuildError(f"static override for run {run} is in another split than {split}")
        for cls, box in per_class.items():
            if cls not in classes:
                raise BuildError(f"static override class {cls!r} is not in experiment.classes")
            check_overlay_box(box, *dims[run], f"static override {run} {cls}")


def apply_overlay(
    auto: dict[str, dict[str, list[float]]],
    overlay: dict | None,
    files: set[str],
) -> tuple[dict[str, dict[str, list[float]]], dict]:
    """Final boxes per frame file from the auto boxes (``auto``: only frames that were
    auto-labeled ok) and an overlay. ``files`` = every sampled frame of the split.
    Returns (final, stats)."""
    overlay = overlay or {}
    entries = overlay.get("frames", {})
    statics = overlay.get("static_overrides", {})
    final: dict[str, dict[str, list[float]]] = {}
    static_overridden = 0
    for name in sorted(files):
        if name in entries:
            continue
        if name not in auto:
            continue
        boxes = {c: list(b) for c, b in auto[name].items()}
        run, _ = _parse_file_name(name)
        if run in statics:
            before = {c: list(b) for c, b in boxes.items()}
            for cls, box in statics[run].items():
                boxes[cls] = [float(v) for v in box]
            if not _boxes_equal(before, boxes):
                static_overridden += 1
        final[name] = boxes
    corrected = verified = excluded_overlay = 0
    for name, entry in sorted(entries.items()):
        if entry.get("excluded"):
            excluded_overlay += 1
            continue
        boxes = {b["class"]: [float(v) for v in b["xyxy"]] for b in entry.get("boxes", [])}
        final[name] = boxes
        if entry.get("verified"):
            verified += 1
        if name not in auto or not _boxes_equal(auto[name], boxes):
            corrected += 1
    stats = {
        "frames_corrected": corrected,
        "frames_verified": verified,
        "frames_excluded_overlay": excluded_overlay,
        "frames_static_overridden": static_overridden,
        "frames_excluded_auto": len(files - set(auto) - set(entries)),
        "frames_labeler_excluded": len(files - set(auto)),
        "frames_recovered": sum(
            1 for n, e in entries.items() if not e.get("excluded") and n not in auto
        ),
    }
    return final, stats


# --- the build ---------------------------------------------------------------------------


def build_dataset(
    *,
    classes: list[str],
    manifest_rows: list[dict],
    index: dict[str, dict],
    frame_labels: dict[str, dict[str, dict]],
    frames_dir: Path,
    corrections_dir: Path,
    out_dir: Path,
    report_path: Path,
    generated_at: str,
    label_review: dict | None = None,
    hands: dict[tuple[str, int], list[tuple[float, ...]]] | None = None,
) -> dict:
    classes = list(classes)
    if not classes or len(set(classes)) != len(classes):
        raise BuildError(f"classes must be non-empty and unique, got {classes}")

    split_of_run: dict[str, str] = {}
    dims: dict[str, tuple[int, int]] = {}
    for row in manifest_rows:
        run, split = row["run_id"], row["split"]
        if split not in SPLITS:
            raise BuildError(f"run {run}: unknown split {split!r}")
        if split_of_run.setdefault(run, split) != split:
            raise BuildError(f"run {run} is in two splits")
        dims[run] = (int(row["width"]), int(row["height"]))
    for run, info in index.items():
        if split_of_run.get(run) != info["split"]:
            raise BuildError(f"run {run}: frame index split {info['split']!r} != manifest")

    files_by_split: dict[str, set[str]] = {s: set() for s in SPLITS}
    for run, info in index.items():
        for fid in info["frame_ids"]:
            files_by_split[info["split"]].add(f"{run}_{fid}.jpg")
    all_files = set().union(*files_by_split.values())
    runs_by_split = {s: {r for r, i in index.items() if i["split"] == s} for s in SPLITS}

    auto_by_split: dict[str, dict[str, dict[str, list[float]]]] = {s: {} for s in SPLITS}
    unlabeled = []
    for run, info in index.items():
        for fid in info["frame_ids"]:
            lab = frame_labels.get(run, {}).get(str(fid))
            if lab is None or lab["status"] == "pending":
                unlabeled.append(f"{run}_{fid}.jpg")
            elif lab["status"] == "ok":
                auto_by_split[info["split"]][f"{run}_{fid}.jpg"] = {
                    c: list(b) for c, b in lab["boxes"].items()
                }
    if unlabeled:
        raise BuildError(
            f"{len(unlabeled)} frames are not labeled yet, e.g. {sorted(unlabeled)[:3]}"
        )

    overlays: dict[str, dict] = {}
    overlay_files: list[Path] = []
    for split in SPLITS:
        path = Path(corrections_dir) / f"{split}.json"
        if path.exists():
            overlays[split] = _load_overlay(path, split)
            overlay_files.append(path)
            _validate_overlay(
                overlays[split],
                split,
                classes,
                files_by_split[split],
                all_files,
                runs_by_split[split],
                set(split_of_run),
                dims,
            )

    out_dir = Path(out_dir)
    ann_files: list[Path] = []
    report_splits: dict[str, dict] = {}
    for split in SPLITS:
        final, stats = apply_overlay(
            auto_by_split[split], overlays.get(split), files_by_split[split]
        )
        labels_by_run: dict[str, dict[int, dict]] = {}
        for name, boxes in final.items():
            run, fid = _parse_file_name(name)
            labels_by_run.setdefault(run, {})[fid] = {
                "status": "ok",
                "boxes": {c: tuple(b) for c, b in boxes.items()},
            }
        coco = build_coco(split, labels_by_run, dims, classes)
        check_coco(coco, classes)
        for img in coco["images"]:
            run, _ = _parse_file_name(img["file_name"])
            if split_of_run[run] != split:
                raise BuildError(
                    f"{img['file_name']} is in {FOLDER[split]} but run {run} is {split_of_run[run]}"
                )
        folder = out_dir / FOLDER[split]
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        for img in coco["images"]:
            shutil.copy2(Path(frames_dir) / split / img["file_name"], folder / img["file_name"])
        ann = folder / "_annotations.coco.json"
        ann.write_text(json.dumps(coco), encoding="utf-8")
        ann_files.append(ann)

        cat_name = {c["id"]: c["name"] for c in coco["categories"]}
        per_class = Counter(cat_name[a["category_id"]] for a in coco["annotations"])
        per_run = Counter(_parse_file_name(i["file_name"])[0] for i in coco["images"])
        run_of_image = {i["id"]: _parse_file_name(i["file_name"])[0] for i in coco["images"]}
        ann_per_run = Counter(run_of_image[a["image_id"]] for a in coco["annotations"])
        report_splits[FOLDER[split]] = {
            "frames_sampled": len(files_by_split[split]),
            "images": len(coco["images"]),
            "annotations": len(coco["annotations"]),
            "images_per_run": dict(sorted(per_run.items())),
            "annotations_per_class": {c: per_class[c] for c in classes},
            "annotations_per_run": dict(sorted(ann_per_run.items())),
            **stats,
        }
        if hands is not None:
            report_splits[FOLDER[split]]["hand_over_container"] = hand_container_share(final, hands)

    report = {
        "version": 1,
        "generated_at": generated_at,
        "classes": classes,
        "dataset_stamp": dataset_stamp(ann_files + overlay_files),
        "overlay_files": {
            p.stem: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in overlay_files
        },
        "splits": report_splits,
    }
    report_path = Path(report_path)
    existing = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None
    report["label_review"] = merge_label_review(existing, label_review)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


# --- CLI ----------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    from training.autolabel import (
        FRAMES_DIR,
        LABELS_DIR,
        _hands_cache,
        load_classes,
        load_index,
    )
    from training.review_sheet import compute_outcome

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("data/dataset"))
    ap.add_argument("--report", type=Path, default=Path("reports/dataset.json"))
    ap.add_argument("--corrections", type=Path, default=Path("data/corrections"))
    ap.add_argument("--manifest", type=Path, default=Path("runs/manifest.csv"))
    ap.add_argument(
        "--review-done",
        action="store_true",
        help="the crew has filled label_review.csv: record its per-class bad fractions",
    )
    args = ap.parse_args(argv)

    classes = load_classes()
    with args.manifest.open(newline="", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))
    index = load_index()
    labels = json.loads((LABELS_DIR / "frame_labels.json").read_text(encoding="utf-8"))
    review = None
    if args.review_done:
        review = compute_outcome()
    report = build_dataset(
        classes=classes,
        manifest_rows=manifest,
        index=index,
        frame_labels=labels,
        frames_dir=FRAMES_DIR,
        corrections_dir=args.corrections,
        out_dir=args.out,
        report_path=args.report,
        generated_at=datetime.now().isoformat(timespec="seconds"),
        label_review=review,
        hands=_hands_cache(),
    )
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
