"""F14 Stage 6 (essential-features.md section 14): training/finetune.py. Pure helpers,
dataset refusals, the summary file shape and the end-to-end flow with the trainers faked.
The real-model dry runs are behind skips when the pretrained weights are not cached."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from training import finetune as ft
from training import gpu_preflight as gp
from training.build_dataset import BuildError

pytestmark = pytest.mark.F14

CLASSES = ["outer_box", "tray", "red_box", "yellow_box", "start_button"]


# --- epochs and learning rate: Stage 6, tested at the boundaries -------------------------------


@pytest.mark.parametrize(
    "n,expected",
    [
        (1, (100, 200)),
        (470, (100, 200)),
        (499, (100, 200)),
        (500, (50, 100)),
        (620, (50, 100)),
        (1999, (50, 100)),
        (2000, (30, 50)),
        (9999, (30, 50)),
    ],
)
def test_epoch_range_by_dataset_size(n, expected):
    assert ft.epoch_range(n) == expected


@pytest.mark.parametrize("n", [0, -1])
def test_epoch_range_rejects_nonpositive(n):
    with pytest.raises(ValueError):
        ft.epoch_range(n)


def test_epoch_range_beyond_stage_6_requires_explicit_epochs():
    with pytest.raises(ValueError) as e:
        ft.epoch_range(10_000)
    assert "--epochs" in str(e.value)


def test_default_epochs_is_the_upper_end_of_the_range_because_early_stopping_bounds_it():
    assert ft.default_epochs(470) == 200
    assert ft.default_epochs(620) == 100
    assert ft.default_epochs(2500) == 50


@pytest.mark.parametrize("n,lr", [(1, 5e-5), (470, 5e-5), (999, 5e-5), (1000, 1e-4), (2500, 1e-4)])
def test_default_lr_is_5e_5_below_1000_images(n, lr):
    assert ft.default_lr(n) == lr


def test_effective_batch_is_batch_times_accumulation():
    assert ft.effective_batch(4, 4) == 16


# --- a tiny synthetic dataset -----------------------------------------------------------------


def _ds(tmp_path: Path) -> Path:
    return ft.make_synthetic_dataset(tmp_path / "ds", CLASSES, seed=0)


def test_synthetic_dataset_has_six_images_and_passes_the_checks(tmp_path):
    root = _ds(tmp_path)
    info = ft.check_dataset(root, CLASSES)
    assert (info["n_train"], info["n_valid"], info["n_test"]) == (3, 2, 1)
    assert sum(info[k] for k in ("n_train", "n_valid", "n_test")) == 6
    assert info["classes"] == CLASSES


def test_synthetic_dataset_is_deterministic(tmp_path):
    a = ft.make_synthetic_dataset(tmp_path / "a", CLASSES, seed=3)
    b = ft.make_synthetic_dataset(tmp_path / "b", CLASSES, seed=3)
    for split in ("train", "valid", "test"):
        assert (a / split / "_annotations.coco.json").read_bytes() == (
            b / split / "_annotations.coco.json"
        ).read_bytes()


def test_refuses_when_class_names_differ_in_order(tmp_path):
    root = _ds(tmp_path)
    with pytest.raises(BuildError):
        ft.check_dataset(root, list(reversed(CLASSES)))


def test_refuses_when_a_split_folder_is_missing(tmp_path):
    import shutil

    root = _ds(tmp_path)
    shutil.rmtree(root / "valid")
    with pytest.raises(ft.DatasetError) as e:
        ft.check_dataset(root, CLASSES)
    assert "valid" in str(e.value)


def test_refuses_when_an_annotation_file_is_missing(tmp_path):
    root = _ds(tmp_path)
    (root / "test" / "_annotations.coco.json").unlink()
    with pytest.raises(ft.DatasetError):
        ft.check_dataset(root, CLASSES)


def test_refuses_when_an_image_file_is_missing(tmp_path):
    root = _ds(tmp_path)
    next((root / "train").glob("*.jpg")).unlink()
    with pytest.raises(ft.DatasetError):
        ft.check_dataset(root, CLASSES)


def test_refuses_when_a_run_id_is_in_two_splits(tmp_path):
    root = _ds(tmp_path)
    train = json.loads((root / "train" / "_annotations.coco.json").read_text())
    valid_p = root / "valid" / "_annotations.coco.json"
    valid = json.loads(valid_p.read_text())
    # rename a valid image so it belongs to a run that is also in train
    train_run = train["images"][0]["file_name"].rpartition("_")[0]
    old = valid["images"][0]["file_name"]
    new = f"{train_run}_9999.jpg"
    (root / "valid" / old).rename(root / "valid" / new)
    valid["images"][0]["file_name"] = new
    valid_p.write_text(json.dumps(valid))
    with pytest.raises(BuildError) as e:
        ft.check_dataset(root, CLASSES)
    assert train_run in str(e.value)


def test_refuses_non_contiguous_category_ids_via_build_dataset_check(tmp_path):
    root = _ds(tmp_path)
    p = root / "train" / "_annotations.coco.json"
    coco = json.loads(p.read_text())
    coco["categories"][0]["id"] = 1  # now 1,1,2,3,4: not 0..n-1
    p.write_text(json.dumps(coco))
    with pytest.raises(BuildError):
        ft.check_dataset(root, CLASSES)


# --- small pure helpers -----------------------------------------------------------------------


def test_sha256_file(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"abc")
    assert ft.sha256_file(f) == hashlib.sha256(b"abc").hexdigest()


def test_read_dataset_stamp_present_and_absent(tmp_path):
    p = tmp_path / "dataset.json"
    assert ft.read_dataset_stamp(p) is None
    p.write_text(json.dumps({"dataset_stamp": "abc123"}))
    assert ft.read_dataset_stamp(p) == "abc123"


def test_best_map_rfdetr_reads_max_ema_column(tmp_path):
    with (tmp_path / "metrics.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "val/ema_mAP_50_95", "train/loss"])
        w.writerow([0, "0.10", "5.0"])
        w.writerow([1, "", "4.0"])  # a train-only row has no validation value
        w.writerow([2, "0.35", "3.0"])
        w.writerow([3, "0.30", "2.5"])
    assert ft.best_map_rfdetr(tmp_path) == pytest.approx(0.35)


def test_best_map_rfdetr_missing_file_or_column_is_none(tmp_path):
    assert ft.best_map_rfdetr(tmp_path) is None
    (tmp_path / "metrics.csv").write_text("epoch,loss\n0,1\n")
    assert ft.best_map_rfdetr(tmp_path) is None


def test_best_map_yolo_reads_padded_column_names(tmp_path):
    (tmp_path / "results.csv").write_text(
        "                  epoch,  metrics/mAP50(B),  metrics/mAP50-95(B)\n"
        "                      1,            0.5,            0.20\n"
        "                      2,            0.6,            0.31\n"
    )
    assert ft.best_map_yolo(tmp_path) == pytest.approx(0.31)


def test_find_resume_checkpoint_prefers_last_then_highest_epoch(tmp_path):
    assert ft.find_resume_checkpoint(tmp_path) is None
    (tmp_path / "checkpoint_9.ckpt").write_bytes(b"1")
    (tmp_path / "checkpoint_10.ckpt").write_bytes(b"1")
    assert ft.find_resume_checkpoint(tmp_path).name == "checkpoint_10.ckpt"
    (tmp_path / "last.ckpt").write_bytes(b"1")
    assert ft.find_resume_checkpoint(tmp_path).name == "last.ckpt"


def test_seed_everything_makes_random_streams_repeat():
    import random

    import numpy as np
    import torch

    ft.seed_everything(7)
    a = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    ft.seed_everything(7)
    b = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert a == b


# --- summary file shape and the flow with fake trainers ---------------------------------------

SUMMARY_KEYS = {
    "model", "versions", "seed", "epochs", "lr", "batch_size", "grad_accum_steps",
    "effective_batch", "n_train", "n_valid", "seconds_per_iteration_measured",
    "total_seconds", "best_val_map", "dataset_stamp", "classes", "detector_file",
    "detector_sha256", "device_name", "dry_run", "notes",
}  # fmt: skip


def _fake_trainers(monkeypatch, tmp_path):
    def fake_rfdetr(**kw):
        out = Path(kw["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        ckpt = out / "checkpoint_best_ema.pth"
        ckpt.write_bytes(b"weights-v1")
        (out / "metrics.csv").write_text("epoch,val/ema_mAP_50_95\n0,0.25\n1,0.4\n")
        return {"checkpoint": ckpt, "best_map": 0.4}

    monkeypatch.setattr(ft, "train_rfdetr", fake_rfdetr)
    monkeypatch.setattr(
        ft,
        "measure_iteration",
        lambda *a, **k: {"seconds_per_iteration": 1.5, "n_timed": 2, "warmup_dropped": False},
    )
    monkeypatch.setattr(ft, "resolve_weights_or_stop", lambda *a, **k: tmp_path / "w.pth")


def test_main_writes_detector_and_summary_with_all_fields(tmp_path, monkeypatch):
    _fake_trainers(monkeypatch, tmp_path)
    root = _ds(tmp_path)
    out = tmp_path / "out"
    stamp = tmp_path / "dataset.json"
    stamp.write_text(json.dumps({"dataset_stamp": "feedbeef"}))
    code = ft.main(
        [
            "--model", "rfdetr", "--dataset-dir", str(root), "--output-dir", str(out),
            "--dataset-report", str(stamp)
        ]
    )  # fmt: skip
    # the synthetic dataset uses the real experiment classes in order, so the check passes
    assert code == 0
    det = out / "detector.pth"
    assert det.read_bytes() == b"weights-v1"
    s = json.loads((out / "train_summary.json").read_text())
    assert SUMMARY_KEYS <= set(s)
    assert s["detector_sha256"] == hashlib.sha256(b"weights-v1").hexdigest()
    assert s["dataset_stamp"] == "feedbeef"
    assert s["best_val_map"] == 0.4
    assert s["n_train"] == 3 and s["n_valid"] == 2
    assert s["epochs"] == 200 and s["lr"] == 5e-5  # 3 images: the < 500 band
    assert (s["batch_size"], s["grad_accum_steps"], s["effective_batch"]) == (4, 4, 16)
    assert s["seconds_per_iteration_measured"] == 1.5
    assert s["classes"] == CLASSES and s["seed"] == 0 and s["dry_run"] is False


def test_main_flags_override_epochs_and_lr(tmp_path, monkeypatch):
    _fake_trainers(monkeypatch, tmp_path)
    root = _ds(tmp_path)
    out = tmp_path / "out"
    ft.main(
        ["--model", "rfdetr", "--dataset-dir", str(root), "--output-dir", str(out),
         "--epochs", "7", "--lr", "0.001", "--seed", "5"]
    )  # fmt: skip
    s = json.loads((out / "train_summary.json").read_text())
    assert (s["epochs"], s["lr"], s["seed"]) == (7, 0.001, 5)


def test_main_refuses_before_training_when_dataset_is_wrong(tmp_path, monkeypatch):
    called = []
    _fake_trainers(monkeypatch, tmp_path)
    monkeypatch.setattr(ft, "train_rfdetr", lambda **kw: called.append(1))
    root = _ds(tmp_path)
    (root / "train" / "_annotations.coco.json").unlink()
    code = ft.main(
        ["--model", "rfdetr", "--dataset-dir", str(root), "--output-dir", str(tmp_path / "o")]
    )
    assert code == 2 and called == []


def test_main_stops_when_weights_are_not_cached(tmp_path, monkeypatch, capsys):
    root = _ds(tmp_path)
    monkeypatch.setattr(ft.gp, "default_weights_path", lambda m: tmp_path / "absent.pth")
    code = ft.main(
        ["--model", "rfdetr", "--dataset-dir", str(root), "--output-dir", str(tmp_path / "o")]
    )
    assert code == 3
    assert "absent.pth" in capsys.readouterr().err


def test_missing_training_extras_lists_names(monkeypatch):
    real = ft.importlib.util.find_spec
    monkeypatch.setattr(
        ft.importlib.util, "find_spec", lambda n: None if n == "pytorch_lightning" else real(n)
    )
    assert "pytorch_lightning" in ft.missing_training_extras()


# --- real-model dry run, only if the weights are cached ----------------------------------------


def _have(model: str) -> bool:
    try:
        return gp.default_weights_path(model).is_file()
    except Exception:
        return False


@pytest.mark.slow
@pytest.mark.skipif(not _have("yolo11n"), reason="YOLO11n weights not cached locally")
def test_dry_run_yolo11n_on_cpu_end_to_end(tmp_path):
    out = tmp_path / "out"
    code = ft.main(["--model", "yolo11n", "--output-dir", str(out), "--dry-run", "--device", "cpu"])
    assert code == 0
    s = json.loads((out / "train_summary.json").read_text())
    assert s["dry_run"] is True and s["n_train"] == 3
    assert (out / "detector_yolo11n.pt").is_file()
    assert "accuracy" in " ".join(s["notes"])


@pytest.mark.slow
@pytest.mark.skipif(not _have("rfdetr"), reason="RF-DETR-Nano weights not cached locally")
def test_rfdetr_train_kwargs_are_accepted_by_the_installed_train_config(tmp_path):
    from rfdetr import RFDETRNano

    kw = ft.rfdetr_train_kwargs(tmp_path, tmp_path / "o", 7, 5e-5, 4, 4, 3, CLASSES)
    cfg = RFDETRNano(pretrain_weights=str(gp.default_weights_path("rfdetr"))).get_train_config(**kw)
    assert (cfg.epochs, cfg.batch_size, cfg.grad_accum_steps, cfg.lr) == (7, 4, 4, 5e-5)
    assert cfg.early_stopping is True and cfg.seed == 3 and cfg.class_names == CLASSES
