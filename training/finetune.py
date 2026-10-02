"""F14 stage 6: fine-tune the detector on a borrowed GPU (environment-agnostic).

  python -m training.finetune --model rfdetr|yolo11n --dataset-dir data/dataset --output-dir OUT
        [--epochs E] [--lr X] [--batch-size 4] [--grad-accum 4] [--seed 0]
        [--device auto|cpu|cuda|cuda:N] [--resume] [--dry-run] [--weights PATH]
        [--dataset-report reports/dataset.json] [--config config/experiment.json]
        [--allow-download]

Every path is explicit, so the same command runs unchanged on any machine. Before anything
trains it refuses (exit 2) if the dataset class names differ from ``config/experiment.json``
in order, if a split folder, an annotation file or an image is missing, if a run id appears in
two splits, or if the COCO ids are not 0..n-1 (``build_dataset.check_coco`` and its run-split
parsing are called, not copied). It stops (exit 3) if the pretrained weights are not already
on disk: nothing is downloaded without ``--allow-download`` (the Lead's yes).

**RF-DETR-Nano** (Apache-2.0): ``RFDETRNano().train(dataset_dir, epochs, batch_size=4,
grad_accum_steps=4, lr, output_dir, early_stopping=True, seed)``, effective batch 16. The
default epochs and learning rate are pure functions of the number of train images (Stage 6):
< 500 images 100-200 epochs, 500-2,000: 50-100, 2,000-10,000: 30-50; lr 5e-5 below 1,000
images, else 1e-4. The default is the *upper* end of the range because early stopping on
validation mAP bounds the cost (a judgment call). The best-EMA checkpoint
(``checkpoint_best_ema.pth``) is copied to ``<output-dir>/detector.pth``.

**YOLO11n** (AGPL-3.0, benchmarked fallback only): the COCO labels are converted by
``training.coco_to_yolo``, Ultralytics trains with the same seed (AdamW at the same lr, batch =
effective batch), and ``best.pt`` is copied to ``<output-dir>/detector_yolo11n.pt``.

Both write ``<output-dir>/train_summary.json``. ``--dry-run`` builds the model and runs 1-2
iterations on a 6-image synthetic dataset: it proves the script runs end to end and says
nothing about accuracy. RF-DETR's ``train()`` needs the ``rfdetr[train]`` extras
(pytorch_lightning, torchmetrics, ...); without them a dry run can only time the network
forward/backward and says so.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import random
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from training import gpu_preflight as gp
from training.build_dataset import BuildError, _parse_file_name, check_coco

SPLIT_FOLDERS = ("train", "valid", "test")
TRAIN_EXTRAS = ("pytorch_lightning", "torchmetrics", "pycocotools")
RFDETR_BEST = "checkpoint_best_ema.pth"
DRY_NOTE = (
    "dry run on 6 synthetic images: proves the script runs end to end here and says nothing "
    "about accuracy"
)


class DatasetError(BuildError):
    """The dataset folder is incomplete or unreadable."""


# --- Stage 6 rule: epochs and learning rate from the number of train images ----------------


def epoch_range(n_train: int) -> tuple[int, int]:
    """Stage 6 (RF-DETR docs): < 500 images 100-200 epochs, 500-2,000 -> 50-100, 2,000-10,000
    -> 30-50. Bands are [lo, hi): 500 is in the second, 2,000 in the third. Stage 6 says
    nothing at 10,000 or above, so that case must be decided by the caller."""
    if n_train <= 0:
        raise ValueError(f"n_train must be positive, got {n_train}")
    if n_train < 500:
        return (100, 200)
    if n_train < 2000:
        return (50, 100)
    if n_train < 10_000:
        return (30, 50)
    raise ValueError(f"Stage 6 gives no epoch range for {n_train} images: pass --epochs")


def default_epochs(n_train: int) -> int:
    return epoch_range(n_train)[1]


def default_lr(n_train: int) -> float:
    """5e-5 below about 1,000 images, else the RF-DETR default 1e-4."""
    return 5e-5 if n_train < 1000 else 1e-4


def effective_batch(batch_size: int, grad_accum: int) -> int:
    return batch_size * grad_accum


# --- dataset checks and a synthetic dataset -------------------------------------------------


def check_dataset(dataset_dir: Path, classes: list[str]) -> dict[str, Any]:
    """Refuse an incomplete or inconsistent dataset. Returns counts."""
    dataset_dir = Path(dataset_dir)
    split_of_run: dict[str, str] = {}
    counts: dict[str, int] = {}
    for split in SPLIT_FOLDERS:
        folder = dataset_dir / split
        ann = folder / "_annotations.coco.json"
        if not folder.is_dir():
            raise DatasetError(f"split folder missing: {folder}")
        if not ann.is_file():
            raise DatasetError(f"annotation file missing: {ann}")
        coco = json.loads(ann.read_text(encoding="utf-8"))
        check_coco(coco, classes)  # names in order, ids 0..n-1, boxes inside the image
        for img in coco["images"]:
            if not (folder / img["file_name"]).is_file():
                raise DatasetError(f"image listed in {ann} is missing: {img['file_name']}")
            run, _ = _parse_file_name(img["file_name"])
            if split_of_run.setdefault(run, split) != split:
                raise BuildError(f"run {run} appears in two splits ({split_of_run[run]}, {split})")
        counts[split] = len(coco["images"])
    return {
        "n_train": counts["train"],
        "n_valid": counts["valid"],
        "n_test": counts["test"],
        "classes": list(classes),
    }


def make_synthetic_dataset(root: Path, classes: list[str], seed: int = 0) -> Path:
    """Six images (3 train, 2 valid, 1 test), one run each, coloured rectangles as boxes, in
    the exact layout ``build_dataset`` writes. For plumbing checks only."""
    import cv2

    rng = np.random.default_rng(seed)
    root = Path(root)
    layout = {"train": 3, "valid": 2, "test": 1}
    w, h = 160, 120
    n = 0
    for split, count in layout.items():
        folder = root / split
        folder.mkdir(parents=True, exist_ok=True)
        images, anns = [], []
        for _ in range(count):
            n += 1
            name = f"dry{n}_{n}.jpg"
            img = np.full((h, w, 3), 90, np.uint8)
            images.append({"id": len(images) + 1, "file_name": name, "width": w, "height": h})
            for cid, _cls in enumerate(classes):
                bw, bh = int(rng.integers(20, 50)), int(rng.integers(15, 40))
                x, y = int(rng.integers(0, w - bw)), int(rng.integers(0, h - bh))
                colour = tuple(int(c) for c in rng.integers(0, 255, 3))
                cv2.rectangle(img, (x, y), (x + bw, y + bh), colour, -1)
                anns.append(
                    {
                        "id": len(anns) + 1,
                        "image_id": len(images),
                        "category_id": cid,
                        "bbox": [x, y, bw, bh],
                        "area": bw * bh,
                        "iscrowd": 0,
                    }
                )
            cv2.imwrite(str(folder / name), img)
        coco = {
            "info": {"description": "synthetic dry-run data"},
            "images": images,
            "annotations": anns,
            "categories": [{"id": i, "name": c} for i, c in enumerate(classes)],
        }
        (folder / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")
    return root


# --- small helpers --------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_dataset_stamp(report_path: Path) -> str | None:
    p = Path(report_path)
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8")).get("dataset_stamp")


def _best_column(csv_path: Path, wanted: list[str]) -> float | None:
    if not csv_path.is_file():
        return None
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return None
        names = {n.strip(): n for n in reader.fieldnames}
        col = next((names[w] for w in wanted if w in names), None)
        if col is None:
            return None
        vals = []
        for row in reader:
            try:
                vals.append(float(row[col]))
            except (TypeError, ValueError):
                continue  # a train-only row has no validation value
    return max(vals) if vals else None


def best_map_rfdetr(output_dir: Path) -> float | None:
    """Best validation mAP@50:95 of the EMA model from rfdetr's ``metrics.csv``."""
    return _best_column(Path(output_dir) / "metrics.csv", ["val/ema_mAP_50_95", "val/mAP_50_95"])


def best_map_yolo(run_dir: Path) -> float | None:
    """Best mAP50-95(B) from Ultralytics' ``results.csv`` (column names are space padded)."""
    return _best_column(Path(run_dir) / "results.csv", ["metrics/mAP50-95(B)"])


def find_resume_checkpoint(output_dir: Path) -> Path | None:
    out = Path(output_dir)
    last = out / "last.ckpt"
    if last.is_file():
        return last
    numbered = []
    for p in out.glob("checkpoint_*.ckpt"):
        m = re.fullmatch(r"checkpoint_(\d+)\.ckpt", p.name)
        if m:
            numbered.append((int(m.group(1)), p))
    return max(numbered)[1] if numbered else None


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def missing_training_extras() -> list[str]:
    return [m for m in TRAIN_EXTRAS if importlib.util.find_spec(m) is None]


class TrainingExtrasMissing(Exception):
    """rfdetr's training stack (the ``rfdetr[train]`` extras) is not installed."""


def build_summary(**kw: Any) -> dict[str, Any]:
    keys = (
        "model versions seed epochs lr batch_size grad_accum_steps effective_batch n_train "
        "n_valid seconds_per_iteration_measured total_seconds best_val_map dataset_stamp "
        "classes detector_file detector_sha256 device_name dry_run notes"
    ).split()
    missing = [k for k in keys if k not in kw]
    if missing:
        raise KeyError(f"summary fields missing: {missing}")
    return {k: kw[k] for k in keys} | {k: v for k, v in kw.items() if k not in keys}


def resolve_weights_or_stop(model: str, explicit: Path | None, allow_download: bool) -> Path:
    return gp.resolve_weights(model, explicit, allow_download)


def measure_iteration(
    model: str, batch_size: int, device: str, weights: Path | None, seed: int
) -> dict[str, Any]:
    n = gp.DEFAULT_GPU_ITERATIONS if device.startswith("cuda") else gp.DEFAULT_CPU_ITERATIONS
    return gp.measure(model, batch_size, device, weights, n, seed)


# --- the real trainers (not run in unit tests; see the faked flow and the dry runs) --------------


def rfdetr_train_kwargs(
    dataset_dir: Path,
    output_dir: Path,
    epochs: int,
    lr: float,
    batch_size: int,
    grad_accum_steps: int,
    seed: int,
    classes: list[str],
) -> dict[str, Any]:
    """The Stage 6 ``RFDETRNano().train(...)`` arguments (tensorboard off: it is an optional
    extra and ``metrics.csv`` is always written)."""
    return {
        "dataset_dir": str(dataset_dir),
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "lr": lr,
        "output_dir": str(output_dir),
        "early_stopping": True,
        "seed": seed,
        "class_names": list(classes),
        "tensorboard": False,
    }


def train_rfdetr(
    *,
    dataset_dir: Path,
    output_dir: Path,
    epochs: int,
    lr: float,
    batch_size: int,
    grad_accum_steps: int,
    seed: int,
    device: str,
    resume: bool,
    weights: Path,
    classes: list[str],
) -> dict[str, Any]:
    missing = missing_training_extras()
    if missing:
        raise TrainingExtrasMissing(
            f"rfdetr's training stack is not installed (missing: {', '.join(missing)}). "
            "It is the `rfdetr[train]` extras group; the lockfile's `train` group does not "
            "include it yet (ISSUES.md, 2026-10-03 P1.5 prep)."
        )
    from rfdetr import RFDETRNano

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    kw: dict[str, Any] = {}
    ckpt = find_resume_checkpoint(out) if resume else None
    if resume and ckpt is None:
        raise FileNotFoundError(f"--resume: no checkpoint (last.ckpt / checkpoint_N.ckpt) in {out}")
    if ckpt is not None:
        kw["resume"] = str(ckpt)
    if device != "auto":
        kw["device"] = device
    model = RFDETRNano(
        pretrain_weights=str(weights), **{k: v for k, v in kw.items() if k == "device"}
    )
    model.train(
        **rfdetr_train_kwargs(
            dataset_dir, out, epochs, lr, batch_size, grad_accum_steps, seed, classes
        ),
        **kw,
    )
    best = out / RFDETR_BEST
    if not best.is_file():
        raise FileNotFoundError(
            f"{best} not found after training; files: {sorted(p.name for p in out.iterdir())}"
        )
    return {"checkpoint": best, "best_map": best_map_rfdetr(out)}


def _yolo_device(device: str) -> str:
    if device == "auto":
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    return (
        device.split(":")[1]
        if device.startswith("cuda:")
        else ("0" if device == "cuda" else device)
    )


def train_yolo(
    *,
    dataset_dir: Path,
    output_dir: Path,
    epochs: int,
    lr: float,
    batch_size: int,
    grad_accum_steps: int,
    seed: int,
    device: str,
    resume: bool,
    weights: Path,
    classes: list[str],
) -> dict[str, Any]:
    from ultralytics import YOLO

    from training.coco_to_yolo import write_yolo_dataset

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    yaml_path = write_yolo_dataset(dataset_dir, out / "yolo_data", classes)
    run_dir = out / "yolo_train"
    last = run_dir / "weights" / "last.pt"
    if resume:
        if not last.is_file():
            raise FileNotFoundError(f"--resume: {last} not found")
        YOLO(str(last)).train(resume=True)
    else:
        YOLO(str(weights)).train(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=gp.RESOLUTION,
            batch=effective_batch(batch_size, grad_accum_steps),
            optimizer="AdamW",  # honours lr0; "auto" would pick its own rate
            lr0=lr,
            seed=seed,
            deterministic=True,
            device=_yolo_device(device),
            project=str(out.resolve()),
            name="yolo_train",
            exist_ok=True,
            patience=20,
            plots=False,
            workers=2,
        )
    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"{best} not found after training")
    return {"checkpoint": best, "best_map": best_map_yolo(run_dir)}


# --- CLI --------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=gp.MODELS, required=True)
    ap.add_argument("--dataset-dir", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--allow-download", action="store_true")
    ap.add_argument("--dataset-report", type=Path, default=Path("reports/dataset.json"))
    ap.add_argument("--config", type=Path, default=Path("config/experiment.json"))
    args = ap.parse_args(argv)

    from training.autolabel import load_classes

    classes = load_classes(args.config)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    dry = args.dry_run
    if dry:
        dataset_dir = make_synthetic_dataset(out / "_dry_run_dataset", classes, args.seed)
    elif args.dataset_dir is None:
        print("--dataset-dir is required unless --dry-run", file=sys.stderr)
        return 2
    else:
        dataset_dir = args.dataset_dir
    try:
        info = check_dataset(dataset_dir, classes)
    except BuildError as e:
        print(f"refusing to train: {e}", file=sys.stderr)
        return 2

    try:
        weights = resolve_weights_or_stop(args.model, args.weights, args.allow_download)
    except gp.WeightsMissing as e:
        msg = f"{e}" + ("\n(dry run skipped)" if dry else "")
        print(msg, file=sys.stderr)
        return 3

    n_train = info["n_train"]
    if dry:
        epochs, lr, batch_size, accum = 1, args.lr or default_lr(n_train), 2, 1
    else:
        epochs = args.epochs if args.epochs is not None else default_epochs(n_train)
        lr = args.lr if args.lr is not None else default_lr(n_train)
        batch_size, accum = args.batch_size, args.grad_accum
    device = gp.resolve_device(args.device)
    minfo = gp.machine_info(args.device)
    print(
        f"{args.model}: {n_train} train / {info['n_valid']} valid images, epochs {epochs}, "
        f"lr {lr}, batch {batch_size} x accum {accum} = {effective_batch(batch_size, accum)}, "
        f"seed {args.seed}, device {device} ({minfo['gpu_name'] or 'CPU'}, "
        f"torch {minfo['torch']} [{minfo['torch_build']}])"
    )

    timing = measure_iteration(args.model, batch_size, device, weights, args.seed)
    print(f"measured {timing['seconds_per_iteration']:.2f} s / iteration (forward+backward)")

    trainer = train_rfdetr if args.model == "rfdetr" else train_yolo
    kwargs = dict(
        dataset_dir=dataset_dir,
        output_dir=out,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        grad_accum_steps=accum,
        seed=args.seed,
        device=args.device,
        resume=args.resume,
        weights=weights,
        classes=classes,
    )
    notes = [DRY_NOTE] if dry else []
    t0 = time.perf_counter()
    result: dict[str, Any] | None
    try:
        result = trainer(**kwargs)
    except TrainingExtrasMissing as e:
        if not dry:
            print(f"cannot train: {e}", file=sys.stderr)
            return 4
        result = None
        notes.append(
            f"PARTIAL: {e} train() was NOT called; only the 2 forward+backward iterations above "
            "ran."
        )
    total = time.perf_counter() - t0

    det_name = "detector.pth" if args.model == "rfdetr" else "detector_yolo11n.pt"
    det_file = det_sha = best_map = None
    if result is not None:
        dest = out / det_name
        shutil.copy2(result["checkpoint"], dest)
        det_file, det_sha, best_map = det_name, sha256_file(dest), result["best_map"]

    summary = build_summary(
        model=args.model,
        versions={"torch": minfo["torch"], "torch_build": minfo["torch_build"],
                  "rfdetr": minfo["rfdetr"], "ultralytics": minfo["ultralytics"],
                  "python": platform.python_version()},
        seed=args.seed,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        grad_accum_steps=accum,
        effective_batch=effective_batch(batch_size, accum),
        n_train=n_train,
        n_valid=info["n_valid"],
        seconds_per_iteration_measured=timing["seconds_per_iteration"],
        total_seconds=total,
        best_val_map=best_map,
        dataset_stamp=None if dry else read_dataset_stamp(args.dataset_report),
        classes=classes,
        detector_file=det_file,
        detector_sha256=det_sha,
        device_name=minfo["gpu_name"] or f"CPU ({platform.processor() or platform.machine()})",
        dry_run=dry,
        notes=notes,
    )  # fmt: skip
    (out / "train_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
