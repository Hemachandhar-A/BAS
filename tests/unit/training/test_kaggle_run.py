"""The pure parts of the Kaggle script kernel (training/kaggle/run_training.py): pins from the
lockfile, the torch-never-touched rule, unpacking and verification, the time guard, the
command plans, the report builders. The real run happens on Kaggle; nothing here needs a GPU,
the network or the Kaggle CLI."""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest

from training.kaggle import run_training as R

ROOT = Path(__file__).resolve().parents[3]
LOCK = (ROOT / "uv.lock").read_text(encoding="utf-8")


# --- pins ---------------------------------------------------------------------------------


def test_pins_come_from_the_real_lock_and_never_name_torch():
    versions = R.parse_lock(LOCK)
    pins = R.pins_from_lock(LOCK)
    assert pins[0] == f"rfdetr[train]=={versions['rfdetr']}"
    assert f"supervision=={versions['supervision']}" in pins
    assert f"ultralytics=={versions['ultralytics']}" in pins
    for p in pins:
        assert "==" in p
        base = R.norm(p.split("[")[0].split("==")[0])
        assert base not in R.NEVER_INSTALL


def test_optional_pins_appear_only_when_the_lock_names_them():
    lock = (
        '[[package]]\nname = "rfdetr"\nversion = "1.2.3"\n'
        '[[package]]\nname = "supervision"\nversion = "0.1.0"\n'
        '[[package]]\nname = "ultralytics"\nversion = "8.0.0"\n'
    )
    assert R.pins_from_lock(lock) == [
        "rfdetr[train]==1.2.3",
        "supervision==0.1.0",
        "ultralytics==8.0.0",
    ]
    more = lock + '[[package]]\nname = "pytorch_lightning"\nversion = "2.6.1"\n'
    more += '[[package]]\nname = "torch"\nversion = "9.9.9"\n'
    more += '[[package]]\nname = "pycocotools"\nversion = "2.0.8"\n'
    pins = R.pins_from_lock(more)
    assert "pycocotools==2.0.8" in pins and not any(p.startswith("torch") for p in pins)
    assert "pytorch-lightning==2.6.1" in pins  # name normalised, as pip accepts it


def test_a_lock_without_a_required_package_is_an_error():
    with pytest.raises(ValueError, match="rfdetr"):
        R.pins_from_lock('[[package]]\nname = "supervision"\nversion = "0.1"\n')


def test_constraints_freeze_the_installed_torch_stack_and_pip_never_upgrades():
    text = R.constraints_text({"torch": "2.9.0+cu126", "torchvision": "0.24.0", "numpy": "2.0"})
    assert text.splitlines() == ["torch==2.9.0+cu126", "torchvision==0.24.0"]
    cmd = R.pip_install_cmd(["rfdetr[train]==1.11.0"], Path("c.txt"))
    assert "-c" in cmd and "c.txt" in cmd and "rfdetr[train]==1.11.0" in cmd
    for bad in ("--upgrade", "-U", "--force-reinstall", "torch", "torchvision"):
        assert bad not in cmd


def test_conflict_detection_reads_pips_wording():
    assert R.looks_like_conflict("ERROR: ResolutionImpossible: for help visit ...")
    assert R.looks_like_conflict("The user requested torch==2.2 but conflicting dependencies")
    assert not R.looks_like_conflict("Successfully installed rfdetr-1.11.0")


def test_filter_freeze_keeps_relevant_packages_with_normalised_names():
    freeze = "numpy==2.0.2\nPyYAML==6.0\nrequests==2.32\ntorch==2.9.0\nPytorch_Lightning==2.6.1\n"
    out = R.filter_freeze(freeze)
    assert out == ["numpy==2.0.2", "Pytorch_Lightning==2.6.1", "PyYAML==6.0", "torch==2.9.0"]


# --- mode and generated block --------------------------------------------------------------


def test_set_mode_changes_only_the_mode_line():
    src = Path(R.__file__).read_text(encoding="utf-8")
    full = R.set_mode(src, "FULL")
    assert 'MODE = "FULL"' in full and 'MODE = "SMOKE"' not in full
    assert R.set_mode(full, "SMOKE") == src
    with pytest.raises(ValueError):
        R.set_mode(src, "TRAIN")
    with pytest.raises(ValueError):
        R.set_mode("x = 1", "FULL")


def test_inject_package_fills_the_block_and_the_result_still_compiles():
    src = Path(R.__file__).read_text(encoding="utf-8")
    package = {"dataset_zip_sha256": "ab" * 32, "n_train": 593, "note": 'it\'s a "test"'}
    out = R.inject_package(src, package)
    compile(out, "run_training.py", "exec")
    ns: dict = {}
    exec(compile(out, "run_training.py", "exec"), ns)  # noqa: S102 - the module under test
    assert ns["PACKAGE"] == package and ns["MODE"] == "SMOKE"
    again = R.inject_package(out, {"n_train": 1})  # re-injecting replaces, never duplicates
    assert again.count("\n# BEGIN GENERATED PACKAGE") == 1
    ns2: dict = {}
    exec(compile(again, "run_training.py", "exec"), ns2)  # noqa: S102
    assert ns2["PACKAGE"] == {"n_train": 1}


def test_the_template_has_no_credential_and_imports_only_the_standard_library_at_top():
    src = Path(R.__file__).read_text(encoding="utf-8")
    assert "kaggle.json" not in src and "KAGGLE_KEY" not in src
    top = [ln for ln in src.splitlines() if ln.startswith(("import ", "from "))]
    assert all(
        ln.split()[1].split(".")[0]
        in {
            "__future__",
            "hashlib",
            "json",
            "os",
            "re",
            "shutil",
            "subprocess",
            "sys",
            "threading",
            "time",
            "zipfile",
            "pathlib",
        }
        for ln in top
    )


# --- unpacking and verification ------------------------------------------------------------


def test_find_sentinel_works_for_both_mount_layouts(tmp_path):
    a = tmp_path / "input" / "sih26174-dataset"
    b = tmp_path / "input" / "datasets" / "hemachandhara" / "sih26174-code"
    for d, name in ((a, R.DATASET_SENTINEL), (b, R.CODE_SENTINEL)):
        d.mkdir(parents=True)
        (d / name).write_text("{}")
    assert R.find_sentinel(tmp_path / "input", R.DATASET_SENTINEL) == a
    assert R.find_sentinel(tmp_path / "input", R.CODE_SENTINEL) == b
    assert R.find_sentinel(tmp_path / "input", "nope.json") is None
    assert R.find_sentinel(tmp_path / "missing", "x") is None


def test_unzip_safe_extracts_and_refuses_a_path_that_escapes(tmp_path):
    good = tmp_path / "good.zip"
    with zipfile.ZipFile(good, "w") as z:
        z.writestr("train/a.jpg", b"1")
    assert R.unzip_safe(good, tmp_path / "out") == 1
    assert (tmp_path / "out" / "train" / "a.jpg").read_bytes() == b"1"
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as z:
        z.writestr("../escape.txt", b"x")
    with pytest.raises(ValueError, match="unsafe"):
        R.unzip_safe(evil, tmp_path / "out2")
    assert not (tmp_path / "escape.txt").exists()


def test_check_files_names_missing_changed_and_extra_files(tmp_path):
    (tmp_path / "a.txt").write_bytes(b"A")
    (tmp_path / "b.txt").write_bytes(b"B")
    (tmp_path / "zzz.txt").write_bytes(b"E")
    expected = {
        "a.txt": R.sha256_file(tmp_path / "a.txt"),
        "b.txt": "0" * 64,
        "c.txt": "1" * 64,
    }
    problems = R.check_files(tmp_path, expected)
    assert "sha256 differs: b.txt" in problems
    assert "missing: c.txt" in problems
    assert "unexpected file: zzz.txt" in problems
    assert not any("a.txt" in p for p in problems)


# --- time guard ------------------------------------------------------------------------------


def test_time_guard_budget_shrinks_with_the_clock_and_stops_new_work():
    now = [1000.0]
    g = R.TimeGuard(limit_s=10_000, safety_s=1_000, clock=lambda: now[0])
    assert g.budget() == 9_000 and g.can_start(min_s=3_600)
    now[0] += 6_000
    assert g.elapsed() == 6_000 and g.budget() == 3_000
    assert not g.can_start(min_s=3_600)
    now[0] += 50_000
    assert g.budget() == 0.0  # never negative


def test_run_logged_stops_a_process_at_the_timeout_and_logs_its_output(tmp_path, capsys):
    log = tmp_path / "x.log"
    code = "import time,sys; print('hello', flush=True); time.sleep(60)"
    r = R.run_logged([sys.executable, "-c", code], log, cwd=tmp_path, timeout_s=3)
    assert r["timed_out"] is True and r["seconds"] < 30
    assert b"hello" in log.read_bytes()
    ok = R.run_logged([sys.executable, "-c", "print('done')"], log, cwd=tmp_path, timeout_s=30)
    assert ok["returncode"] == 0 and ok["timed_out"] is False
    assert "done" in capsys.readouterr().out


# --- command plans ---------------------------------------------------------------------------


def test_preflight_requires_cuda_and_uses_the_report_numbers():
    cmd = R.preflight_cmd("rfdetr", Path("/w/rf.pth"), 593, 100)
    assert "--require-cuda" in cmd
    assert cmd[cmd.index("--n-train") + 1] == "593" and cmd[cmd.index("--epochs") + 1] == "100"
    assert cmd[cmd.index("--model") + 1] == "rfdetr" and cmd[cmd.index("--weights") + 1] == str(
        Path("/w/rf.pth")
    )


def test_full_command_leaves_epochs_and_lr_to_the_s_e_rule_and_resumes_only_when_asked():
    base = R.finetune_cmd("rfdetr", Path("d"), Path("o"), Path("w.pth"), Path("r.json"), False)
    assert "--epochs" not in base and "--lr" not in base and "--resume" not in base
    assert base[base.index("--seed") + 1] == "0"
    assert "--dry-run" not in base
    assert (
        R.finetune_cmd("yolo11n", Path("d"), Path("o"), Path("w"), Path("r"), True)[-1]
        == "--resume"
    )


def test_dry_run_command_is_a_dry_run_with_seed_zero():
    cmd = R.dry_run_cmd("yolo11n", Path("w.pt"), Path("o"), Path("r.json"))
    assert "--dry-run" in cmd and cmd[cmd.index("--seed") + 1] == "0"


def test_checkpoint_detection_per_model(tmp_path):
    rf, yo = tmp_path / "rf", tmp_path / "yo"
    assert not R.has_checkpoint("rfdetr", rf) and not R.has_checkpoint("yolo11n", yo)
    rf.mkdir()
    (rf / "checkpoint_best_ema.pth").write_bytes(b"1")
    assert not R.has_checkpoint("rfdetr", rf)  # best weights are not a resume checkpoint
    (rf / "checkpoint_10.ckpt").write_bytes(b"1")
    assert R.has_checkpoint("rfdetr", rf)
    (yo / "yolo_train" / "weights").mkdir(parents=True)
    (yo / "yolo_train" / "weights" / "last.pt").write_bytes(b"1")
    assert R.has_checkpoint("yolo11n", yo)


def test_restore_from_an_attached_earlier_output(tmp_path):
    prev = tmp_path / "input" / "sih26174-train" / "outputs" / "rfdetr"
    prev.mkdir(parents=True)
    (prev / "last.ckpt").write_bytes(b"ckpt")
    (prev / "metrics.csv").write_text("epoch\n1\n")
    out = tmp_path / "work" / "outputs" / "rfdetr"
    assert R.restore_from_inputs("rfdetr", out, tmp_path / "input") == prev
    assert (out / "last.ckpt").read_bytes() == b"ckpt" and (out / "metrics.csv").exists()
    assert R.restore_from_inputs("rfdetr", out, tmp_path / "input") is None  # already there
    assert R.restore_from_inputs("yolo11n", tmp_path / "w2", tmp_path / "input") is None


def test_model_status_distinguishes_finished_failed_timeout_and_not_started(tmp_path):
    out = tmp_path / "rfdetr"
    out.mkdir()
    (out / "detector.pth").write_bytes(b"w")
    (out / "train_summary.json").write_text(json.dumps({"best_val_map": 0.5}))
    fin = R.model_status("rfdetr", out, 0, False, 12.0)
    assert fin["state"] == "finished" and fin["best_val_map"] == 0.5
    assert fin["detector_sha256"] == R.sha256_file(out / "detector.pth")
    assert R.model_status("rfdetr", out, 1, False, 1.0)["state"] == "failed"
    assert R.model_status("rfdetr", out, None, True, 1.0)["state"] == "stopped_by_time_guard"
    assert R.model_status("yolo11n", tmp_path / "y", None, False, 0.0)["state"] == "not_started"
    (out / "last.ckpt").write_bytes(b"c")
    assert R.model_status("rfdetr", out, None, True, 1.0)["resumable"] is True


# --- smoke report ----------------------------------------------------------------------------

PREFLIGHT = """model:            rfdetr
device used:      cuda
GPU:              Tesla T4  (14.74 GB VRAM)
CUDA available:   True
torch:            2.9.0+cu126  [CUDA build (cuda 12.6)]
rfdetr:           1.11.0
timed iterations: 4 at micro-batch 4 (first dropped as warm-up)
seconds / iteration (measured): 0.42
-- ESTIMATE for 593 images x 100 epochs (149 iterations per epoch) --
seconds / epoch:  62.6  (62.6 s)
total:            6258 s  (1.74 h)
note: an estimate only
"""


def test_parse_preflight_reads_every_number():
    p = R.parse_preflight(PREFLIGHT)
    assert p["gpu_name"] == "Tesla T4" and p["vram_gb"] == 14.74
    assert p["torch"] == "2.9.0+cu126" and p["torch_build"] == "CUDA build (cuda 12.6)"
    assert p["seconds_per_iteration"] == 0.42 and p["seconds_per_epoch_est"] == 62.6
    assert p["total_seconds_est"] == 6258 and p["iterations_per_epoch"] == 149
    assert R.parse_preflight("nothing here")["seconds_per_iteration"] is None


def test_smoke_report_gives_minutes_flags_failures_and_the_budget():
    pf = {m: {**R.parse_preflight(PREFLIGHT), "returncode": 0} for m in R.MODELS}
    dry = {m: {"returncode": 0, "seconds": 40.0} for m in R.MODELS}
    machine = {
        "gpu_name": "Tesla T4",
        "vram_gb": 14.7,
        "cuda_available": True,
        "torch": "2.9.0",
        "torch_build": "cuda 12.6",
        "python": "3.12",
    }
    rep = R.smoke_report(
        machine=machine, n_train=593, epochs=100, preflight=pf, dry_runs=dry,
        smoke_seconds=300.0, pip_seconds=120.0, dataset_stamp="s", problems=[],
    )  # fmt: skip
    m = rep["models"]["rfdetr"]
    assert m["seconds_per_iteration"] == 0.42
    assert m["estimated_minutes_per_epoch"] == pytest.approx(62.6 / 60)
    assert m["estimated_total_minutes"] == pytest.approx(6258 / 60)
    assert rep["estimated_total_minutes_both_models"] == pytest.approx(2 * 6258 / 60)
    assert rep["ok"] and rep["within_budget"] and rep["gpu_name"] == "Tesla T4"
    json.dumps(rep)
    slow = R.smoke_report(
        machine=machine, n_train=593, epochs=100, preflight=pf, dry_runs=dry,
        smoke_seconds=700.0, pip_seconds=None, dataset_stamp="s", problems=["x failed"],
    )  # fmt: skip
    assert not slow["within_budget"] and not slow["ok"]
    missing = R.smoke_report(
        machine=machine, n_train=593, epochs=100, preflight={}, dry_runs={},
        smoke_seconds=1.0, pip_seconds=None, dataset_stamp=None, problems=[],
    )  # fmt: skip
    assert missing["estimated_total_minutes_both_models"] is None


def test_sha256_sums_lists_only_existing_files(tmp_path):
    (tmp_path / "a").write_bytes(b"x")
    text = R.sha256_sums(tmp_path, ["a", "missing"])
    assert text == f"{R.sha256_file(tmp_path / 'a')}  a\n"


def test_child_env_allows_hub_downloads_because_the_kernel_has_internet():
    assert R.child_env()["HF_HUB_OFFLINE"] == "0"


def test_parse_preflight_reads_the_real_format_report_output():
    """Guards against drift: the text comes from gpu_preflight.format_report itself."""
    from training import gpu_preflight as gp

    info = {
        "device": "cuda", "gpu_name": "Tesla T4", "vram_gb": 14.74, "cuda_available": True,
        "torch_build": "cuda 12.6", "torch": "2.9.0+cu126", "rfdetr": "1.11.0",
    }  # fmt: skip
    timing = gp.summarise_times([1.0, 0.5, 0.5, 0.5, 0.5])
    est = gp.extrapolate(timing["seconds_per_iteration"], 593, 4, 100)
    p = R.parse_preflight(gp.format_report("rfdetr", info, timing, est, batch_size=4))
    assert p["gpu_name"] == "Tesla T4" and p["vram_gb"] == 14.74
    assert p["torch"] == "2.9.0+cu126" and p["torch_build"] == "CUDA build (cuda 12.6)"
    assert p["seconds_per_iteration"] == pytest.approx(0.5, abs=0.01)
    assert p["iterations_per_epoch"] == 149
    assert p["seconds_per_epoch_est"] == pytest.approx(est["seconds_per_epoch"], abs=0.1)
    assert p["total_seconds_est"] == pytest.approx(est["total_seconds"], abs=1)
