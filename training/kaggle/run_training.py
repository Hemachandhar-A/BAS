"""Kaggle SCRIPT kernel: fine-tune the detector (RF-DETR-Nano, then YOLO11n) on a Kaggle GPU.

One constant selects what runs. Change only the line ``MODE = ...`` below (the packer does it
with ``python -m training.kaggle_pack --set-mode FULL``):

  SMOKE  prints the machine, unpacks and verifies the data, installs the pinned packages,
         times both models on the GPU (``gpu_preflight --require-cuda``), runs
         ``finetune --dry-run`` on a tiny synthetic dataset, writes ``smoke_report.json`` and
         stops. Meant to take well under 10 minutes after the pip install.
  FULL   trains RF-DETR-Nano and then YOLO11n (seed 0, epochs and learning rate by the S-E
         rule inside ``training.finetune``), each with ``train_summary.json``, the best
         weights, a log and sha256s, all under ``/kaggle/working/outputs``. Resumable: a
         ``--resume`` is passed when a checkpoint is already there (or in an attached earlier
         run's output). A total-time guard stops cleanly before the session limit and writes
         ``run_status.json`` saying what finished.

This file is stdlib-only at import time so its pure helpers are unit-tested on a laptop
(tests/unit/training/test_kaggle_run.py); torch and the project code are imported inside
functions, after the datasets have been unpacked. Nothing here reads or writes a credential.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

MODE = "SMOKE"  # "SMOKE" or "FULL" -- the only line to change between the two runs

# BEGIN GENERATED PACKAGE (training.kaggle_pack fills this block; do not edit by hand)
PACKAGE: dict | None = None
# END GENERATED PACKAGE

SEED = 0
INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
OUT = WORK / "outputs"
DATASET_SENTINEL = "dataset_manifest.json"
CODE_SENTINEL = "code_manifest.json"
# Kaggle documents up to 9 h for a GPU session (unverified by the author of this script); the
# guard stops FULL this long before that, so the outputs are written and the kernel ends clean.
SESSION_LIMIT_S = 9 * 3600
SAFETY_S = 25 * 60
MIN_START_S = 30 * 60  # do not start a model with less than this left
SMOKE_BUDGET_S = 10 * 60
RELEVANT_PACKAGES = (
    "torch", "torchvision", "torchaudio", "rfdetr", "supervision", "ultralytics", "pycocotools",
    "torchmetrics", "pytorch-lightning", "transformers", "peft", "numpy", "opencv-python",
    "opencv-python-headless", "pillow", "pydantic", "pyyaml", "scipy",
)  # fmt: skip
# pinned from uv.lock when the lock names them; rfdetr[train] brings the rest of its stack
LOCK_REQUIRED = ("rfdetr", "supervision", "ultralytics")
LOCK_OPTIONAL = ("pycocotools", "torchmetrics", "pytorch-lightning")
NEVER_INSTALL = ("torch", "torchvision", "torchaudio")
MODELS = ("rfdetr", "yolo11n")
DETECTOR_FILE = {"rfdetr": "detector.pth", "yolo11n": "detector_yolo11n.pt"}
WEIGHT_FILE = {"rfdetr": "rf-detr-nano.pth", "yolo11n": "yolo11n.pt"}


# --- small pure helpers -----------------------------------------------------------------


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fmt_seconds(s: float) -> str:
    s = int(round(s))
    return f"{s // 3600}h{(s % 3600) // 60:02d}m{s % 60:02d}s"


def set_mode(source: str, mode: str) -> str:
    """``source`` with the ``MODE = ...`` line set to ``mode`` (SMOKE or FULL)."""
    if mode not in ("SMOKE", "FULL"):
        raise ValueError(f"mode must be SMOKE or FULL, got {mode!r}")
    new, n = re.subn(r'(?m)^MODE = "[A-Z]+"', f'MODE = "{mode}"', source, count=1)
    if n != 1:
        raise ValueError("the MODE line was not found")
    return new


def inject_package(source: str, package: dict) -> str:
    """``source`` with the generated PACKAGE block holding ``package`` (plain JSON data)."""
    m_begin = re.search(r"(?m)^# BEGIN GENERATED PACKAGE", source)
    m_end = re.search(r"(?m)^# END GENERATED PACKAGE", source)
    if not m_begin or not m_end:
        raise ValueError("the generated PACKAGE block markers were not found")
    begin, i, j = "# BEGIN GENERATED PACKAGE", m_begin.start(), m_end.start()
    head = source[:i]
    tail = source[j:]
    block = (
        begin
        + " (training.kaggle_pack fills this block; do not edit by hand)\n"
        + "PACKAGE: dict | None = json.loads(r'''"
        + json.dumps(package, indent=1, sort_keys=True)
        + "''')\n"
    )
    return head + block + tail


def filter_freeze(freeze_text: str, names: tuple[str, ...] = RELEVANT_PACKAGES) -> list[str]:
    """The lines of a ``pip freeze`` for the packages that matter here (name matching ignores
    case and ``-``/``_``/``.``)."""
    wanted = {norm(n) for n in names}
    out = []
    for line in freeze_text.splitlines():
        m = re.match(r"([A-Za-z0-9_.\-]+)", line.strip())
        if m and norm(m.group(1)) in wanted:
            out.append(line.strip())
    return sorted(out, key=str.lower)


# --- pins from uv.lock (pure) -----------------------------------------------------------


def parse_lock(lock_text: str) -> dict[str, str]:
    """name -> version of every package in a ``uv.lock``."""
    import tomllib

    data = tomllib.loads(lock_text)
    return {norm(p["name"]): p["version"] for p in data["package"] if "version" in p}


def pins_from_lock(lock_text: str) -> list[str]:
    """pip requirement strings pinned to the locked versions: ``rfdetr[train]``, ``supervision``
    and ``ultralytics`` (required; a lock without them is an error), plus
    ``pycocotools``, ``torchmetrics`` and ``pytorch-lightning`` when the lock names them.
    torch, torchvision and torchaudio are never returned: the Kaggle image's own build is used.
    The lockfile's ``train`` group lacks rfdetr's ``[train]`` extras (ISSUES.md), so those are
    left to pip within rfdetr's own bounds; the versions pip picks are in the run's pip freeze."""
    versions = parse_lock(lock_text)
    missing = [n for n in LOCK_REQUIRED if n not in versions]
    if missing:
        raise ValueError(f"uv.lock does not name {missing}")
    pins = [f"rfdetr[train]=={versions['rfdetr']}"]
    pins += [f"{n}=={versions[n]}" for n in ("supervision", "ultralytics")]
    pins += [f"{n}=={versions[n]}" for n in LOCK_OPTIONAL if n in versions]
    assert not any(norm(re.split(r"[\[=<>]", p)[0]) in NEVER_INSTALL for p in pins)
    return pins


def constraints_text(installed: dict[str, str]) -> str:
    """A pip constraints file that freezes torch, torchvision and torchaudio at the versions
    already installed, so no install can replace the image's CUDA build."""
    lines = [f"{n}=={installed[n]}" for n in NEVER_INSTALL if n in installed]
    return "\n".join(lines) + "\n"


def pip_install_cmd(pins: list[str], constraints_path: Path) -> list[str]:
    return [
        sys.executable, "-m", "pip", "install", "--no-input", "-c", str(constraints_path), *pins
    ]  # fmt: skip


def looks_like_conflict(pip_output: str) -> bool:
    low = pip_output.lower()
    return any(
        k in low
        for k in (
            "resolutionimpossible",
            "conflicting dependencies",
            "cannot install",
            "no matching distribution",
        )
    )


# --- time guard (pure, clock injected) ---------------------------------------------------


class TimeGuard:
    def __init__(self, limit_s: float, safety_s: float, clock=time.monotonic):
        self.limit, self.safety, self.clock = limit_s, safety_s, clock
        self.t0 = clock()

    def elapsed(self) -> float:
        return self.clock() - self.t0

    def budget(self) -> float:
        """Seconds a step may still run (limit - safety - elapsed), never negative."""
        return max(0.0, self.limit - self.safety - self.elapsed())

    def can_start(self, min_s: float = MIN_START_S) -> bool:
        return self.budget() >= min_s


# --- unpacking and verification (pure + file system) ------------------------------------


def find_sentinel(root: Path, filename: str) -> Path | None:
    """The directory under ``root`` (any depth) holding ``filename``; Kaggle's mount layout
    for attached datasets has changed between image versions, so nothing is hard-coded."""
    root = Path(root)
    if not root.is_dir():
        return None
    hits = sorted(root.rglob(filename), key=lambda p: (len(p.parts), str(p)))
    return hits[0].parent if hits else None


def unzip_safe(zip_path: Path, dest: Path) -> int:
    """Extract ``zip_path`` into ``dest``; refuses a member that would land outside it."""
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            target = (dest / info.filename).resolve()
            if dest != target and dest not in target.parents:
                raise ValueError(f"{zip_path.name}: unsafe member {info.filename!r}")
        z.extractall(dest)
        return len(z.infolist())


def check_files(root: Path, expected: dict[str, str]) -> list[str]:
    """Problems between the files under ``root`` and ``expected`` (relative posix path ->
    sha256): missing, changed, and extra files."""
    root = Path(root)
    problems = []
    for rel, sha in sorted(expected.items()):
        p = root / rel
        if not p.is_file():
            problems.append(f"missing: {rel}")
        elif sha256_file(p) != sha:
            problems.append(f"sha256 differs: {rel}")
    on_disk = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    problems += [f"unexpected file: {x}" for x in sorted(on_disk - set(expected))]
    return problems


def parse_preflight(text: str) -> dict:
    """The numbers in ``gpu_preflight``'s report."""

    def num(pattern: str) -> float | None:
        m = re.search(pattern, text, re.M)
        return float(m.group(1)) if m else None

    gpu = re.search(r"^GPU:\s+(.+?)\s{2}\(([\d.]+) GB VRAM\)", text, re.M)
    torch_line = re.search(r"^torch:\s+(\S+)\s+\[(.+?)\]", text, re.M)
    return {
        "gpu_name": gpu.group(1) if gpu else None,
        "vram_gb": float(gpu.group(2)) if gpu else None,
        "torch": torch_line.group(1) if torch_line else None,
        "torch_build": torch_line.group(2) if torch_line else None,
        "seconds_per_iteration": num(r"seconds / iteration \(measured\):\s+([\d.]+)"),
        "seconds_per_epoch_est": num(r"^seconds / epoch:\s+([\d.]+)"),
        "total_seconds_est": num(r"^total:\s+([\d.]+) s"),
        "iterations_per_epoch": num(r"\((\d+) iterations per epoch\)"),
    }


def smoke_report(
    *,
    machine: dict,
    n_train: int,
    epochs: int,
    preflight: dict[str, dict],
    dry_runs: dict[str, dict],
    smoke_seconds: float,
    pip_seconds: float | None,
    dataset_stamp: str | None,
    problems: list[str],
) -> dict:
    models = {}
    for m in MODELS:
        pf, dr = preflight.get(m, {}), dry_runs.get(m, {})
        per_epoch, total = pf.get("seconds_per_epoch_est"), pf.get("total_seconds_est")
        models[m] = {
            "seconds_per_iteration": pf.get("seconds_per_iteration"),
            "estimated_minutes_per_epoch": None if per_epoch is None else per_epoch / 60,
            "estimated_total_minutes": None if total is None else total / 60,
            "estimate_note": f"{n_train} images, {epochs} epochs; a lower bound",
            "preflight_returncode": pf.get("returncode"),
            "dry_run_returncode": dr.get("returncode"),
            "dry_run_seconds": dr.get("seconds"),
        }
    totals = [v["estimated_total_minutes"] for v in models.values()]
    return {
        "mode": "SMOKE",
        "gpu_name": machine.get("gpu_name"),
        "vram_gb": machine.get("vram_gb"),
        "cuda_available": machine.get("cuda_available"),
        "torch": machine.get("torch"),
        "torch_build": machine.get("torch_build"),
        "python": machine.get("python"),
        "n_train": n_train,
        "epochs": epochs,
        "dataset_stamp": dataset_stamp,
        "models": models,
        "estimated_total_minutes_both_models": (
            None if any(t is None for t in totals) else sum(totals)
        ),
        "smoke_seconds": smoke_seconds,
        "smoke_budget_seconds": SMOKE_BUDGET_S,
        "within_budget": smoke_seconds <= SMOKE_BUDGET_S,
        "pip_install_seconds": pip_seconds,
        "problems": problems,
        "ok": not problems,
    }


# --- command plans (pure) ----------------------------------------------------------------


def preflight_cmd(model: str, weights: Path, n_train: int, epochs: int) -> list[str]:
    return [
        sys.executable, "-m", "training.gpu_preflight", "--model", model, "--batch-size", "4",
        "--n-train", str(n_train), "--epochs", str(epochs), "--require-cuda",
        "--weights", str(weights), "--seed", str(SEED),
    ]  # fmt: skip


def dry_run_cmd(model: str, weights: Path, out_dir: Path, report: Path) -> list[str]:
    return [
        sys.executable, "-m", "training.finetune", "--model", model, "--dry-run",
        "--output-dir", str(out_dir), "--weights", str(weights), "--seed", str(SEED),
        "--dataset-report", str(report),
    ]  # fmt: skip


def finetune_cmd(
    model: str, dataset_dir: Path, out_dir: Path, weights: Path, report: Path, resume: bool
) -> list[str]:
    """FULL: no --epochs and no --lr, so training.finetune applies the S-E rule (epochs and
    learning rate from the number of train images)."""
    cmd = [
        sys.executable, "-m", "training.finetune", "--model", model,
        "--dataset-dir", str(dataset_dir), "--output-dir", str(out_dir),
        "--weights", str(weights), "--dataset-report", str(report), "--seed", str(SEED),
    ]  # fmt: skip
    if resume:
        cmd.append("--resume")
    return cmd


def has_checkpoint(model: str, out_dir: Path) -> bool:
    out = Path(out_dir)
    if model == "yolo11n":
        return (out / "yolo_train" / "weights" / "last.pt").is_file()
    return (out / "last.ckpt").is_file() or any(
        re.fullmatch(r"checkpoint_\d+\.ckpt", p.name) for p in out.glob("checkpoint_*.ckpt")
    )


def restore_from_inputs(model: str, out_dir: Path, input_root: Path) -> Path | None:
    """If an earlier run's output is attached as an input (``outputs/<model>`` with a
    checkpoint), copy that folder to ``out_dir`` so ``--resume`` can continue it."""
    out_dir = Path(out_dir)
    if has_checkpoint(model, out_dir):
        return None
    root = Path(input_root)
    if not root.is_dir():
        return None
    for cand in sorted(root.rglob(model), key=lambda p: (len(p.parts), str(p))):
        if cand.is_dir() and cand.parent.name == "outputs" and has_checkpoint(model, cand):
            shutil.copytree(cand, out_dir, dirs_exist_ok=True)
            return cand
    return None


def model_status(
    model: str, out_dir: Path, returncode: int | None, timed_out: bool, seconds: float
) -> dict:
    out = Path(out_dir)
    det = out / DETECTOR_FILE[model]
    summary = out / "train_summary.json"
    best_map = None
    if summary.is_file():
        try:
            best_map = json.loads(summary.read_text(encoding="utf-8")).get("best_val_map")
        except ValueError:
            pass
    if timed_out:
        state = "stopped_by_time_guard"
    elif returncode == 0 and det.is_file():
        state = "finished"
    elif returncode is None:
        state = "not_started"
    else:
        state = "failed"
    return {
        "model": model,
        "state": state,
        "returncode": returncode,
        "seconds": seconds,
        "detector_file": det.name if det.is_file() else None,
        "detector_sha256": sha256_file(det) if det.is_file() else None,
        "train_summary": summary.name if summary.is_file() else None,
        "best_val_map": best_map,
        "resumable": has_checkpoint(model, out),
        "log": f"{model}_train.log",
    }


def sha256_sums(root: Path, names: list[str]) -> str:
    lines = [f"{sha256_file(root / n)}  {n}" for n in names if (root / n).is_file()]
    return "\n".join(lines) + "\n"


# --- running things ---------------------------------------------------------------------


def capture(cmd: list[str], timeout_s: float = 60) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        return (r.stdout or "") + (r.stderr or "")
    except (OSError, subprocess.SubprocessError) as e:
        return f"<{cmd[0]} failed: {e}>"


def run_logged(
    cmd: list[str], log: Path, *, cwd: Path, timeout_s: float | None, env: dict | None = None
) -> dict:
    """Run ``cmd``, echo its output live and append it to ``log``. On ``timeout_s`` the process
    is terminated (then killed) and ``timed_out`` is True."""
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    timed_out = False
    with Path(log).open("ab") as lf:
        proc = subprocess.Popen(
            cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )

        def pump() -> None:
            assert proc.stdout is not None
            for line in iter(proc.stdout.readline, b""):
                sys.stdout.write(line.decode("utf-8", "replace"))
                sys.stdout.flush()
                lf.write(line)
                lf.flush()

        t = threading.Thread(target=pump, daemon=True)
        t.start()
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.terminate()
            try:
                proc.wait(timeout=90)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        t.join(timeout=15)
    return {
        "returncode": proc.returncode,
        "timed_out": timed_out,
        "seconds": time.monotonic() - t0,
    }


def child_env() -> dict:
    """Internet is on in this kernel, so allow Hugging Face downloads (training.autolabel sets
    HF_HUB_OFFLINE=1 only when the variable is unset)."""
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def machine_info() -> dict:
    info = {"python": sys.version.split()[0], "cuda_available": False}
    try:
        import torch

        info.update(
            torch=torch.__version__,
            torch_build=f"cuda {torch.version.cuda}" if torch.version.cuda else "cpu",
        )
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            p = torch.cuda.get_device_properties(0)
            info.update(gpu_name=p.name, vram_gb=round(p.total_memory / 1024**3, 2))
            info["compute_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
    except Exception as e:  # noqa: BLE001 - the banner must never stop the run
        info["torch_error"] = repr(e)
    return info


def banner() -> dict:
    print("=" * 78)
    print(f"MODE {MODE}   started {time.strftime('%Y-%m-%d %H:%M:%S')}")
    info = machine_info()
    for k in (
        "python",
        "torch",
        "torch_build",
        "cuda_available",
        "gpu_name",
        "vram_gb",
        "compute_capability",
    ):
        print(f"{k:18}: {info.get(k)}")
    if "torch_error" in info:
        print("torch import failed:", info["torch_error"])
    print("-- nvidia-smi --")
    print(capture(["nvidia-smi"]).strip())
    print("-- pip freeze (relevant packages) --")
    print("\n".join(filter_freeze(capture([sys.executable, "-m", "pip", "freeze"], 120))))
    print("=" * 78, flush=True)
    return info


def stop(message: str, code: int) -> None:
    print(f"\nSTOP: {message}", flush=True)
    sys.exit(code)


def installed_versions() -> dict[str, str]:
    import importlib.metadata as md

    out = {}
    for n in NEVER_INSTALL:
        try:
            out[n] = md.version(n)
        except md.PackageNotFoundError:
            pass
    return out


# --- the run ----------------------------------------------------------------------------


def prepare(guard: TimeGuard) -> dict:
    """Locate, verify and unpack the two datasets. Returns the paths and manifests."""
    assert PACKAGE is not None, "this script was not generated by training.kaggle_pack"
    ds_dir = find_sentinel(INPUT, DATASET_SENTINEL)
    code_dir = find_sentinel(INPUT, CODE_SENTINEL)
    if ds_dir is None or code_dir is None:
        stop(
            f"datasets not found under {INPUT} (dataset sentinel {ds_dir}, code sentinel "
            f"{code_dir}); attach sih26174-dataset and sih26174-code as inputs",
            2,
        )
    ds_manifest = json.loads((ds_dir / DATASET_SENTINEL).read_text(encoding="utf-8"))
    code_manifest = json.loads((code_dir / CODE_SENTINEL).read_text(encoding="utf-8"))

    data_root = WORK / "data" / "dataset"
    zip_path = ds_dir / ds_manifest["zip"]["name"]
    if zip_path.is_file():
        got = sha256_file(zip_path)
        print(f"dataset zip sha256 {got}")
        if got != PACKAGE["dataset_zip_sha256"] or got != ds_manifest["zip"]["sha256"]:
            stop(f"dataset zip sha256 {got} != packed {PACKAGE['dataset_zip_sha256']}", 2)
        print(f"unzipped {unzip_safe(zip_path, data_root)} entries -> {data_root}")
    else:
        print("dataset.zip is not present; Kaggle unpacked it: copying the tree")
        for split in ("train", "valid", "test"):
            if (ds_dir / split).is_dir():
                shutil.copytree(ds_dir / split, data_root / split, dirs_exist_ok=True)
    problems = check_files(data_root, ds_manifest["files"])
    if problems:
        stop(f"dataset files differ from the packed manifest: {problems[:5]} ({len(problems)})", 2)

    code_root = WORK / "code"
    code_zip = code_dir / code_manifest["zip"]["name"]
    if code_zip.is_file():
        got = sha256_file(code_zip)
        print(f"code zip sha256 {got}")
        if got != PACKAGE["code_zip_sha256"] or got != code_manifest["zip"]["sha256"]:
            stop(f"code zip sha256 {got} != packed {PACKAGE['code_zip_sha256']}", 2)
        print(f"unzipped {unzip_safe(code_zip, code_root)} entries -> {code_root}")
    else:
        print("code.zip is not present; Kaggle unpacked it: copying the tree")
        for rel in code_manifest["files"]:
            if (code_dir / rel).is_file():
                (code_root / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(code_dir / rel, code_root / rel)
    problems = check_files(code_root, code_manifest["files"])
    if problems:
        stop(f"code files differ from the packed manifest: {problems[:5]} ({len(problems)})", 2)

    weights = {}
    for name, meta in code_manifest["weights"].items():
        p = code_dir / name
        if not p.is_file() or sha256_file(p) != meta["sha256"]:
            stop(f"weights file {name} is missing or its sha256 differs", 2)
        weights[name] = p
        print(f"weights {name}: sha256 ok, {p.stat().st_size} bytes, licence {meta['licence']}")
    return {"data_root": data_root, "code_root": code_root, "weights": weights}


def verify_unzipped(code_root: Path, data_root: Path) -> None:
    print("\n== verify_dataset on the unzipped data ==", flush=True)
    cmd = [
        sys.executable, "-m", "training.verify_dataset", "--dataset-dir", str(data_root),
        "--report", str(code_root / "reports" / "dataset.json"),
        "--corrections", str(code_root / "data" / "corrections"), "--no-default-csvs",
        "--expect-train", str(PACKAGE["n_train"]),
    ]  # fmt: skip
    r = run_logged(cmd, OUT / "verify_dataset.log", cwd=code_root, timeout_s=300, env=child_env())
    if r["returncode"] != 0:
        stop("verify_dataset reported problems on the unzipped data (see above)", 2)


def install_pins(code_root: Path) -> float:
    print("\n== install pinned packages (torch and torchvision are never touched) ==", flush=True)
    pins = pins_from_lock((code_root / "uv.lock").read_text(encoding="utf-8"))
    print("pins:", *pins, sep="\n  ")
    before = installed_versions()
    print("torch stack before:", before)
    cons = WORK / "constraints.txt"
    cons.write_text(constraints_text(before), encoding="utf-8")
    t0 = time.monotonic()
    log = OUT / "pip_install.log"
    r = run_logged(pip_install_cmd(pins, cons), log, cwd=code_root, timeout_s=1800, env=child_env())
    seconds = time.monotonic() - t0
    if r["returncode"] != 0:
        text = log.read_text(encoding="utf-8", errors="replace")
        if looks_like_conflict(text):
            stop(
                "pip cannot satisfy the pins together with Kaggle's preinstalled torch "
                f"{before} (constraints kept torch fixed). The conflict is printed above. "
                "Nothing was reinstalled. Report it; do not force a torch change on Kaggle.",
                3,
            )
        stop(f"pip install failed (exit {r['returncode']}); see above", 3)
    after = installed_versions()
    if after != before:
        stop(f"torch stack changed during install: {before} -> {after}", 3)
    print(f"pip install finished in {fmt_seconds(seconds)}; torch stack unchanged {after}")
    return seconds


def read_report(code_root: Path) -> dict:
    return json.loads((code_root / "reports" / "dataset.json").read_text(encoding="utf-8"))


def run_smoke(paths: dict, machine: dict, pip_seconds: float | None) -> int:
    code_root, weights = paths["code_root"], paths["weights"]
    sys.path.insert(0, str(code_root))
    from training.finetune import default_epochs

    report = read_report(code_root)
    n_train = report["splits"]["train"]["images"]
    epochs = default_epochs(n_train)
    print(f"\n== SMOKE: n_train {n_train}, epochs {epochs} (S-E rule) ==", flush=True)
    t0 = time.monotonic()
    problems: list[str] = []
    preflight: dict[str, dict] = {}
    dry: dict[str, dict] = {}
    for model in MODELS:
        w = weights[WEIGHT_FILE[model]]
        print(f"\n-- gpu_preflight {model} --", flush=True)
        log = OUT / f"smoke_preflight_{model}.log"
        r = run_logged(
            preflight_cmd(model, w, n_train, epochs),
            log,
            cwd=code_root,
            timeout_s=SMOKE_BUDGET_S,
            env=child_env(),
        )
        parsed = parse_preflight(log.read_text(encoding="utf-8", errors="replace"))
        preflight[model] = {**parsed, "returncode": r["returncode"]}
        if r["returncode"] != 0:
            problems.append(f"gpu_preflight {model} exited {r['returncode']}")
    for model in MODELS:
        w = weights[WEIGHT_FILE[model]]
        print(f"\n-- finetune --dry-run {model} (6 synthetic images) --", flush=True)
        r = run_logged(
            dry_run_cmd(model, w, OUT / "smoke" / model, code_root / "reports" / "dataset.json"),
            OUT / f"smoke_dry_run_{model}.log",
            cwd=code_root,
            timeout_s=SMOKE_BUDGET_S,
            env=child_env(),
        )
        dry[model] = {"returncode": r["returncode"], "seconds": r["seconds"]}
        if r["returncode"] != 0:
            problems.append(f"finetune --dry-run {model} exited {r['returncode']}")
    smoke_seconds = time.monotonic() - t0
    rep = smoke_report(
        machine=machine, n_train=n_train, epochs=epochs, preflight=preflight, dry_runs=dry,
        smoke_seconds=smoke_seconds, pip_seconds=pip_seconds,
        dataset_stamp=report.get("dataset_stamp"), problems=problems,
    )  # fmt: skip
    (OUT / "smoke_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print("\n" + "=" * 78 + "\nSMOKE REPORT\n" + json.dumps(rep, indent=1))
    print(f"\nsmoke part took {fmt_seconds(smoke_seconds)} (budget {fmt_seconds(SMOKE_BUDGET_S)})")
    print("SMOKE " + ("OK" if rep["ok"] else "FAILED: " + "; ".join(problems)), flush=True)
    return 0 if rep["ok"] else 1


def run_full(paths: dict, machine: dict, guard: TimeGuard) -> int:
    code_root, data_root, weights = paths["code_root"], paths["data_root"], paths["weights"]
    report_path = code_root / "reports" / "dataset.json"
    status: dict = {
        "mode": "FULL", "machine": machine, "seed": SEED,
        "dataset_stamp": read_report(code_root).get("dataset_stamp"),
        "models": [], "time_guard": {"limit_s": SESSION_LIMIT_S, "safety_s": SAFETY_S},
    }  # fmt: skip

    def save() -> None:
        status["elapsed_s"] = guard.elapsed()
        (OUT / "run_status.json").write_text(json.dumps(status, indent=1), encoding="utf-8")

    save()
    for model in MODELS:
        out_dir = OUT / model
        restored = restore_from_inputs(model, out_dir, INPUT)
        if restored:
            print(f"{model}: restored an earlier run's folder {restored} for --resume")
        if (out_dir / DETECTOR_FILE[model]).is_file() and (
            out_dir / "train_summary.json"
        ).is_file():
            print(f"{model}: already finished in an earlier run, not retrained")
            status["models"].append(model_status(model, out_dir, 0, False, 0.0))
            save()
            continue
        if not guard.can_start():
            print(f"{model}: not started, {fmt_seconds(guard.budget())} left before the guard")
            status["models"].append(model_status(model, out_dir, None, False, 0.0))
            save()
            continue
        resume = has_checkpoint(model, out_dir)
        cmd = finetune_cmd(
            model, data_root, out_dir, weights[WEIGHT_FILE[model]], report_path, resume
        )
        print(
            f"\n== FULL: {model} ({'resume' if resume else 'fresh'}), "
            f"guard budget {fmt_seconds(guard.budget())} ==",
            flush=True,
        )
        r = run_logged(
            cmd,
            OUT / f"{model}_train.log",
            cwd=code_root,
            timeout_s=guard.budget(),
            env=child_env(),
        )
        st = model_status(model, out_dir, r["returncode"], r["timed_out"], r["seconds"])
        status["models"].append(st)
        save()
        print(f"{model}: {st['state']} after {fmt_seconds(r['seconds'])}", flush=True)
    names = [f"{m}/{DETECTOR_FILE[m]}" for m in MODELS] + [
        f"{m}/train_summary.json" for m in MODELS
    ]
    (OUT / "SHA256SUMS.txt").write_text(sha256_sums(OUT, names), encoding="utf-8")
    size_gb = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1024**3
    status["outputs_gb"] = round(size_gb, 2)
    save()
    print("\n" + "=" * 78 + "\nFULL RUN STATUS\n" + json.dumps(status, indent=1))
    for st in status["models"]:
        print(f"  {st['model']:8} {st['state']:22} detector sha256 {st['detector_sha256']}")
    stopped = [
        s["model"]
        for s in status["models"]
        if s["state"] in ("stopped_by_time_guard", "not_started")
    ]
    if stopped:
        print(
            f"Unfinished (time guard): {stopped}. Attach this run's output as an input and "
            "push again to resume."
        )
    return 1 if any(s["state"] == "failed" for s in status["models"]) else 0


def main() -> int:
    guard = TimeGuard(SESSION_LIMIT_S, SAFETY_S)
    OUT.mkdir(parents=True, exist_ok=True)
    machine = banner()
    if not machine.get("cuda_available"):
        stop(
            "CUDA is not available in this kernel: enable a GPU accelerator in the kernel settings",
            2,
        )
    paths = prepare(guard)
    sys.path.insert(0, str(paths["code_root"]))
    verify_unzipped(paths["code_root"], paths["data_root"])
    pip_seconds = install_pins(paths["code_root"])
    machine = machine_info()
    if MODE == "SMOKE":
        return run_smoke(paths, machine, pip_seconds)
    if MODE == "FULL":
        return run_full(paths, machine, guard)
    stop(f"MODE must be SMOKE or FULL, got {MODE!r}", 2)
    return 2


if __name__ == "__main__":
    sys.exit(main())
