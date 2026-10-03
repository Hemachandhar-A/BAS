"""Is the dataset on disk whole, and is it the one ``reports/dataset.json`` names?

  python -m training.verify_dataset [--dataset-dir data/dataset] [--report reports/dataset.json]
        [--corrections data/corrections] [--config config/experiment.json]
        [--expect-train N] [--no-default-csvs]

Checks (every problem is listed, then the exit code is 1; 0 and one line when all is well):
every JSON under the dataset folder, the report and the overlay files parses; no NUL bytes in any
JSON or CSV (a dataset build was once found NUL-filled minutes after it was written; cause
unknown); per split the images on disk are exactly the images in the annotation file; every image
opens and has the annotated size; every image has exactly one box per class; the category names
equal ``experiment.classes`` in order, with ids 0..n-1; the report's classes and per-split image
counts agree with the disk; the overlay files the report names exist and have the recorded hash;
and the dataset stamp recomputed from the annotation and overlay files equals the report's.
Run it after every build and again just before the dataset is packed or trained on.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from training.build_dataset import dataset_stamp

SPLITS = ("train", "valid", "test")
ANN = "_annotations.coco.json"


@dataclass
class Result:
    problems: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    stamp: str | None = None


def _scan_text_files(files: list[tuple[str, Path]], problems: list[str]) -> None:
    """NUL and parse check of (label, path) pairs; ``.json`` files must parse."""
    for label, path in files:
        try:
            data = path.read_bytes()
        except OSError as e:
            problems.append(f"{label}: cannot be read ({e})")
            continue
        if b"\x00" in data:
            problems.append(f"{label}: contains NUL bytes ({data.count(0)} of {len(data)} bytes)")
            continue
        if path.suffix == ".json":
            try:
                json.loads(data.decode("utf-8"))
            except (ValueError, UnicodeDecodeError) as e:
                problems.append(f"{label}: does not parse as JSON ({e})")


def _check_split(split: str, folder: Path, classes: list[str], problems: list[str]) -> int | None:
    """Checks one split folder; returns its annotated image count (None if unreadable)."""
    import cv2

    if not folder.is_dir():
        problems.append(f"{split}: folder missing ({folder})")
        return None
    ann_path = folder / ANN
    if not ann_path.is_file():
        problems.append(f"{split}: annotation file missing ({ann_path})")
        return None
    try:
        coco = json.loads(ann_path.read_bytes().decode("utf-8"))
        images, annotations, cats = coco["images"], coco["annotations"], coco["categories"]
    except (ValueError, KeyError, UnicodeDecodeError):
        return None  # already reported by the parse scan
    names = [c["name"] for c in cats]
    if names != list(classes):
        problems.append(f"{split}: class names {names} != experiment.classes {list(classes)}")
    if [c["id"] for c in cats] != list(range(len(cats))):
        problems.append(f"{split}: category ids are not 0..n-1: {[c['id'] for c in cats]}")
    cat_name = {c["id"]: c["name"] for c in cats}

    listed = {i["file_name"] for i in images}
    on_disk = {p.name for p in folder.glob("*.jpg")}
    for name in sorted(listed - on_disk):
        problems.append(f"{split}: {name} is in the annotations but the file is missing")
    for name in sorted(on_disk - listed):
        problems.append(f"{split}: {name} is on disk but not in the annotations")
    if len(on_disk) != len(images):
        problems.append(
            f"{split}: image count {len(on_disk)} on disk != {len(images)} in the annotations"
        )

    per_image: dict[int, list[str]] = {}
    for a in annotations:
        per_image.setdefault(a["image_id"], []).append(cat_name.get(a["category_id"], "?"))
    for img in images:
        name = img["file_name"]
        if name in on_disk:
            arr = cv2.imread(str(folder / name))
            if arr is None:
                problems.append(f"{split}: {name} cannot be opened as an image")
            elif (arr.shape[1], arr.shape[0]) != (img["width"], img["height"]):
                problems.append(
                    f"{split}: {name} size {arr.shape[1]}x{arr.shape[0]} != annotation "
                    f"{img['width']}x{img['height']}"
                )
        got = per_image.get(img["id"], [])
        if len(got) != len(classes):
            problems.append(f"{split}: {name} has {len(got)} boxes, expected {len(classes)}")
        wrong = sorted(set(classes) - set(got)) + sorted({c for c in got if got.count(c) > 1})
        if wrong:
            problems.append(f"{split}: {name} needs one box per class; problem classes: {wrong}")
    return len(images)


def verify(
    dataset_dir: Path,
    report_path: Path,
    corrections_dir: Path,
    classes: list[str],
    *,
    csv_paths: list[Path] | tuple[Path, ...] = (),
    expect_train: int | None = None,
) -> Result:
    dataset_dir, report_path, corrections_dir = (
        Path(dataset_dir),
        Path(report_path),
        Path(corrections_dir),
    )
    res = Result()
    problems = res.problems

    scan: list[tuple[str, Path]] = []
    if dataset_dir.is_dir():
        for p in sorted(dataset_dir.rglob("*")):
            if p.suffix in (".json", ".csv"):
                scan.append((p.relative_to(dataset_dir).as_posix(), p))
    scan.append((report_path.name, report_path))
    if corrections_dir.is_dir():
        scan += [
            (f"{corrections_dir.name}/{p.name}", p) for p in sorted(corrections_dir.glob("*.json"))
        ]
    scan += [(p.name, Path(p)) for p in csv_paths if Path(p).exists()]
    _scan_text_files(scan, problems)

    for split in SPLITS:
        n = _check_split(split, dataset_dir / split, classes, problems)
        if n is not None:
            res.counts[split] = n
    if expect_train is not None and res.counts.get("train") != expect_train:
        problems.append(f"train: expected {expect_train} images, found {res.counts.get('train')}")

    try:
        report = json.loads(report_path.read_bytes().decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        problems.append(f"{report_path.name}: cannot be loaded, so the stamp is not checked")
        return res
    if report.get("classes") != list(classes):
        problems.append(f"report classes {report.get('classes')} != experiment.classes {classes}")
    for split, n in res.counts.items():
        said = report.get("splits", {}).get(split, {}).get("images")
        if said != n:
            problems.append(f"report says {said} images in {split}, the disk has {n}")

    files = [dataset_dir / s / ANN for s in SPLITS]
    missing = [str(f) for f in files if not f.is_file()]
    for name, sha in sorted(report.get("overlay_files", {}).items()):
        f = corrections_dir / f"{name}.json"
        if not f.is_file():
            problems.append(f"overlay {name}: {f} is missing but the report names it")
            missing.append(str(f))
            continue
        got = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
        if got != sha:
            problems.append(f"overlay {name} changed: sha256 prefix {got} != report {sha}")
        files.append(f)
    if missing:
        problems.append(f"stamp not recomputed, files missing: {missing}")
        return res
    res.stamp = dataset_stamp(files)
    if res.stamp != report.get("dataset_stamp"):
        problems.append(f"dataset stamp {res.stamp} != report {report.get('dataset_stamp')}")
    return res


def main(argv: list[str] | None = None) -> int:
    from training.autolabel import load_classes

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset-dir", type=Path, default=Path("data/dataset"))
    ap.add_argument("--report", type=Path, default=Path("reports/dataset.json"))
    ap.add_argument("--corrections", type=Path, default=Path("data/corrections"))
    ap.add_argument("--config", type=Path, default=Path("config/experiment.json"))
    ap.add_argument("--expect-train", type=int, default=None)
    ap.add_argument(
        "--no-default-csvs",
        action="store_true",
        help="do not also scan runs/manifest.csv and data/label_review.csv for NUL bytes",
    )
    args = ap.parse_args(argv)

    csvs: list[Path] = []
    if not args.no_default_csvs:
        csvs = [p for p in (Path("runs/manifest.csv"), Path("data/label_review.csv")) if p.exists()]
    res = verify(
        args.dataset_dir,
        args.report,
        args.corrections,
        load_classes(args.config),
        csv_paths=csvs,
        expect_train=args.expect_train,
    )
    if res.problems:
        print(f"PROBLEMS ({len(res.problems)}):")
        for p in res.problems:
            print(f"  - {p}")
        return 1
    counts = ", ".join(f"{s} {n}" for s, n in res.counts.items())
    print(f"OK: {counts}; stamp {res.stamp} equals the report; no NUL bytes; every image opens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
