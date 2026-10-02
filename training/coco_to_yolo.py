"""Pure COCO -> YOLO label converter for the YOLO11n fallback (F14 stage 6).

  python -m training.coco_to_yolo --dataset-dir data/dataset --out-dir <dir>

Reads ``<dataset-dir>/{train,valid}/_annotations.coco.json`` (the layout ``build_dataset``
writes; COCO ids 0-based, class order = ``experiment.classes``) and writes the layout
Ultralytics wants, ``<out>/images/<split>/*.jpg`` + ``<out>/labels/<split>/*.txt`` (a label
file per image, **empty for an image with no boxes**) and ``<out>/dataset.yaml``. Class index
in a label line is the category's position in the id-sorted category list, so with our 0-based
ids it equals the id. The ``test`` split is never converted: training has no use for it.
Images are copied, not linked (symlinks are not portable to Windows).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

SPLITS = ("train", "valid")


class ConvertError(Exception):
    """The COCO data cannot be converted faithfully."""


def xywh_to_yolo(bbox: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    """COCO xywh in pixels -> YOLO (cx, cy, w, h) normalised to 0..1."""
    x, y, w, h = bbox
    return ((x + w / 2) / width, (y + h / 2) / height, w / width, h / height)


def class_index(categories: list[dict]) -> tuple[dict[int, int], list[str]]:
    ordered = sorted(categories, key=lambda c: c["id"])
    return {c["id"]: i for i, c in enumerate(ordered)}, [c["name"] for c in ordered]


def coco_to_yolo_lines(coco: dict) -> dict[str, list[str]]:
    """file_name -> YOLO label lines. Every image gets an entry, empty if it has no boxes."""
    idx, _ = class_index(coco["categories"])
    dims = {i["id"]: (i["width"], i["height"]) for i in coco["images"]}
    names = {i["id"]: i["file_name"] for i in coco["images"]}
    out: dict[str, list[str]] = {n: [] for n in names.values()}
    for a in coco["annotations"]:
        if a["category_id"] not in idx:
            raise ConvertError(f"annotation {a.get('id')} has unknown category {a['category_id']}")
        w, h = dims[a["image_id"]]
        x, y, bw, bh = a["bbox"]
        if x < 0 or y < 0 or x + bw > w + 1e-6 or y + bh > h + 1e-6 or bw <= 0 or bh <= 0:
            raise ConvertError(f"annotation {a.get('id')} has a degenerate or outside box")
        cx, cy, nw, nh = xywh_to_yolo([x, y, bw, bh], w, h)
        out[names[a["image_id"]]].append(
            f"{idx[a['category_id']]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
        )
    return out


def dataset_yaml(root: Path, classes: list[str]) -> str:
    return yaml.safe_dump(
        {
            "path": Path(root).resolve().as_posix(),
            "train": "images/train",
            "val": "images/valid",
            "names": dict(enumerate(classes)),
        },
        sort_keys=False,
    )


def write_yolo_dataset(dataset_dir: Path, out_dir: Path, classes: list[str]) -> Path:
    dataset_dir, out_dir = Path(dataset_dir), Path(out_dir)
    for split in SPLITS:
        ann = dataset_dir / split / "_annotations.coco.json"
        coco = json.loads(ann.read_text(encoding="utf-8"))
        _, names = class_index(coco["categories"])
        if names != list(classes):
            raise ConvertError(f"{split}: class names {names} != expected {list(classes)}")
        lines = coco_to_yolo_lines(coco)
        img_dir, lab_dir = out_dir / "images" / split, out_dir / "labels" / split
        for d in (img_dir, lab_dir):
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)
        for name, rows in lines.items():
            shutil.copy2(dataset_dir / split / name, img_dir / name)
            (lab_dir / (Path(name).stem + ".txt")).write_text(
                "".join(r + "\n" for r in rows), encoding="utf-8"
            )
    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text(dataset_yaml(out_dir, classes), encoding="utf-8")
    return yaml_path


def main(argv: list[str] | None = None) -> None:
    from training.autolabel import load_classes

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset-dir", type=Path, default=Path("data/dataset"))
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    print(write_yolo_dataset(args.dataset_dir, args.out_dir, load_classes()))


if __name__ == "__main__":
    main()
