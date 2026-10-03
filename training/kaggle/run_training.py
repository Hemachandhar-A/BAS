"""Kaggle SCRIPT kernel: fine-tune the detector (RF-DETR-Nano, then YOLO11n) on a Kaggle GPU.

One constant selects what runs. Change only the line ``MODE = ...`` below (the packer does it
with ``python -m training.kaggle_pack --set-mode FULL``):

  SMOKE  prints the machine, unpacks and verifies the data, installs the pinned packages
         (dataset verification and the pip step are hard stops), then runs five stages, each in
         its own guard with a 5-minute timeout, and CONTINUES after a failure: gpu_preflight
         for both models, ``finetune --dry-run`` for both on a tiny synthetic dataset, and a
         3-iteration real RF-DETR training probe on the real dataset. Prints a stage table,
         writes ``smoke_report.json`` and ends with SMOKE OK or SMOKE PARTIAL.
  FULL   trains RF-DETR-Nano and then YOLO11n (seed 0, epochs and learning rate by the S-E
         rule inside ``training.finetune``), each with ``train_summary.json``, the best
         weights, a log and sha256s, all under ``/kaggle/working/outputs``. Resumable: a
         ``--resume`` is passed when a checkpoint is already there (or in an attached earlier
         run's output). A total-time guard stops cleanly before the session limit and writes
         ``run_status.json`` saying what finished.

Datasets and code are staged under a temp folder (/kaggle/temp, else /tmp/work), never under
/kaggle/working: that folder is the kernel OUTPUT and keeps only ``outputs/`` (checked at the
end: more than 2 GB fails the run).

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
WORK = Path("/kaggle/working")  # the kernel OUTPUT: only outputs/ lives here
OUT = WORK / "outputs"
TEMP_CANDIDATES = (Path("/kaggle/temp"), Path("/tmp/work"))  # datasets and code are staged here
TEMP_NEED_BYTES = 4 * 1024**3
OUTPUT_LIMIT_BYTES = 2 * 1024**3
STAGE_TIMEOUT_S = 300
TAIL_LINES = 30
PROBE_ITERATIONS = 3
DATASET_SENTINEL = "dataset_manifest.json"
CODE_SENTINEL = "code_manifest.json"
# Kaggle documents up to 9 h for a GPU session (unverified by the author of this script); the
# guard stops FULL this long before that, so the outputs are written and the kernel ends clean.
SESSION_LIMIT_S = 9 * 3600
SAFETY_S = 25 * 60
MIN_START_S = 30 * 60  # do not start a model with less than this left
SMOKE_BUDGET_S = 10 * 60
MAX_DEPTH = 5  # levels under /kaggle/input that discovery looks at
LIST_DEPTH = 4  # levels of the start-of-run listing
LIST_LINES = 200
BIG_FOLDER = 10  # a folder with more files than this is listed as a count and a total size
PROBLEM_LIMIT = 10
RELEVANT_PACKAGES = (
    "torch", "torchvision", "torchaudio", "rfdetr", "supervision", "ultralytics", "pycocotools",
    "torchmetrics", "pytorch-lightning", "transformers", "peft", "numpy", "opencv-python",
    "opencv-python-headless", "pillow", "pydantic", "pyyaml", "scipy",
)  # fmt: skip
# pinned from uv.lock when the lock names them; rfdetr[train] brings the rest of its stack
LOCK_REQUIRED = ("rfdetr", "supervision", "ultralytics")
LOCK_OPTIONAL = ("pycocotools", "torchmetrics", "pytorch-lightning")
NEVER_INSTALL = ("torch", "torchvision", "torchaudio")
# The constraints file freezes ONLY the compiled core, at the versions the Kaggle image has.
# Pure-Python training libraries (torchmetrics, pytorch-lightning, peft, transformers,
# pycocotools, pydantic) are NOT frozen: smoke run 2 failed because the image's torchmetrics
# 1.9.0 was frozen while rfdetr[train] 1.11.0 needs torchmetrics<1.9.0. pip's default
# only-if-needed strategy leaves them alone unless a pin requires a change.
FROZEN = (
    *NEVER_INSTALL, "numpy", "scipy", "pillow", "opencv-python", "opencv-python-headless",
    "opencv-contrib-python",
)  # fmt: skip
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
    """A pip constraints file that freezes every package of ``FROZEN`` (the core: torch,
    torchvision, torchaudio, numpy, scipy, pillow, opencv) that is installed at its installed
    version, so no install can replace the image's CUDA build or its core stack. Nothing else
    is frozen. A pin that cannot be met together with these stops the run, and
    ``conflicting_packages`` names the culprits."""
    lines = [f"{n}=={installed[n]}" for n in FROZEN if n in installed]
    return "\n".join(lines) + "\n"


def drop_installed_optional(
    pins: list[str], installed: dict[str, str]
) -> tuple[list[str], list[str]]:
    """``pins`` without the optional ones the image already has (the constraints file freezes
    those at the installed version, so a different locked version could only conflict), and
    a note for each dropped pin."""
    optional = {norm(n) for n in LOCK_OPTIONAL}
    keep, notes = [], []
    for p in pins:
        base = norm(re.split(r"[\[=<>]", p)[0])
        if base in optional and base in installed:
            notes.append(f"kept the installed {base} {installed[base]} instead of {p}")
        else:
            keep.append(p)
    return keep, notes


def changed_packages(before: dict[str, str], after: dict[str, str]) -> list[str]:
    return sorted(n for n in set(before) | set(after) if before.get(n) != after.get(n))


def conflicting_packages(pip_output: str) -> list[str]:
    """The package names pip's error text blames, first mention first: ``Cannot install A and
    B``, ``No matching distribution found for X``, ``Could not find a version ... X``,
    ``The user requested X`` and ``X 1.0 requires Y<2, but you have Y 2.1 which is
    incompatible``."""

    def base(req: str) -> str:
        return norm(re.split(r"[\[=<>!~;,\s(]", req.strip())[0])

    found: list[str] = []
    for m in re.finditer(r"Cannot install (.+?)(?: because|\n|$)", pip_output):
        found += [base(r) for r in re.split(r"\s+and\s+|,\s*", m.group(1))]
    for m in re.finditer(r"No matching distribution found for (\S+)", pip_output):
        found.append(base(m.group(1)))
    for m in re.finditer(r"find a version that satisfies the requirement (\S+)", pip_output):
        found.append(base(m.group(1)))
    for m in re.finditer(r"The user requested (\S+)", pip_output):
        found.append(base(m.group(1)))
    for m in re.finditer(
        r"^\s*([A-Za-z0-9_.\-]+) \S+ requires ([A-Za-z0-9_.\-]+)[^\n]*which is incompatible",
        pip_output,
        re.M,
    ):
        found += [norm(m.group(1)), norm(m.group(2))]
    out: list[str] = []
    for n in found:
        if n and n not in out:
            out.append(n)
    return out


def pip_install_cmd(pins: list[str], constraints_path: Path, dry_run: bool = False) -> list[str]:
    cmd = [sys.executable, "-m", "pip", "install", "--no-input", "-c", str(constraints_path)]
    if dry_run:
        cmd.append("--dry-run")
    return [*cmd, *pins]


def pip_supports_dry_run(install_help: str) -> bool:
    """Does ``pip install --help`` list ``--dry-run`` (pip 22.2 and later)?"""
    return "--dry-run" in install_help


def parse_dry_run_plan(pip_output: str) -> list[tuple[str, str]]:
    """(normalised name, version) of every package on pip's ``Would install ...`` line(s)."""
    plan: list[tuple[str, str]] = []
    for m in re.finditer(r"(?m)^Would install (.+)$", pip_output):
        for token in m.group(1).split():
            parts = re.match(r"(.+?)-(\d.*)$", token)
            plan.append((norm(parts.group(1)), parts.group(2)) if parts else (norm(token), ""))
    return plan


def plan_core_changes(plan: list[tuple[str, str]]) -> list[str]:
    """The core packages (``FROZEN``) a plan would install or replace, sorted."""
    core = {norm(n) for n in FROZEN}
    return sorted({name for name, _v in plan if name in core})


def apply_plan(installed: dict[str, str], plan: list[tuple[str, str]]) -> dict[str, str]:
    """``installed`` as it would be after ``plan``."""
    return {**installed, **dict(plan)}


def describe_changes(before: dict[str, str], after: dict[str, str]) -> list[str]:
    """One line per package added or changed between two ``name -> version`` maps."""
    lines = []
    for name in sorted(after):
        if name not in before:
            lines.append(f"added   {name} {after[name]}")
        elif before[name] != after[name]:
            lines.append(f"changed {name} {before[name]} -> {after[name]}")
    return lines


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


# --- temp folder and output size (pure, probes injected) -----------------------------------


def free_bytes(path: Path, must_exist: bool = False) -> int | None:
    """Free bytes on the volume holding ``path`` (its nearest existing parent, unless
    ``must_exist``); None if it cannot be told."""
    p = Path(path)
    if must_exist and not p.exists():
        return None
    while not p.exists() and p != p.parent:
        p = p.parent
    try:
        return shutil.disk_usage(p).free
    except OSError:
        return None


def choose_temp_root(
    free, need_bytes: int = TEMP_NEED_BYTES, candidates: tuple[Path, ...] = TEMP_CANDIDATES
) -> tuple[Path, str]:
    """Where the datasets and the code are staged: the first candidate with at least
    ``need_bytes`` free, and a sentence saying which and why. ``free(path_string)`` returns
    the free bytes or None when the candidate is absent (the real probe needs /kaggle/temp to
    exist; /tmp/work is created on demand). Raises ``DiscoveryError`` naming every candidate
    when none has room."""
    notes = []
    for c in candidates:
        got = free(c.as_posix())
        if got is None:
            notes.append(f"{c.as_posix()} is not available")
        elif got < need_bytes:
            notes.append(
                f"{c.as_posix()} too small ({size_text(got)} free, need {size_text(need_bytes)})"
            )
        else:
            why = f"using {c.as_posix()} ({size_text(got)} free)"
            return c, why + (f"; {'; '.join(notes)}" if notes else "")
    raise DiscoveryError("no temp folder with room: " + "; ".join(notes))


def temp_free_probe(path: str) -> int | None:
    """The real probe for ``choose_temp_root``: /kaggle/temp must exist, /tmp/work need not."""
    return free_bytes(Path(path), must_exist=path == TEMP_CANDIDATES[0].as_posix())


def dir_size(root: Path) -> int:
    """Total bytes of the files under ``root`` (links not followed); 0 if it is missing."""
    total = 0
    for cur, _dirs, files in os.walk(root):
        for f in files:
            try:
                total += os.lstat(os.path.join(cur, f)).st_size
            except OSError:
                pass
    return total


def output_size_check(root: Path, limit_bytes: int = OUTPUT_LIMIT_BYTES) -> tuple[bool, list[str]]:
    """(ok, lines): the total size of ``root`` against ``limit_bytes``, with the biggest
    entries named so a failure says what to remove."""
    root = Path(root)
    total = dir_size(root)
    ok = total <= limit_bytes
    head = f"output size of {root}: {size_text(total)} (limit {size_text(limit_bytes)}): " + (
        "OK" if ok else "EXCEEDS THE LIMIT: the kernel output is too large to download"
    )
    lines = [head]
    if root.is_dir() and (not ok or total > limit_bytes // 4):
        entries = []
        for e in root.iterdir():  # two levels, so outputs/<big thing> is named
            entries.append((dir_size(e) if e.is_dir() else e.lstat().st_size, e.name))
            if e.is_dir():
                for c in e.iterdir():
                    n = dir_size(c) if c.is_dir() else c.lstat().st_size
                    entries.append((n, f"{e.name}/{c.name}"))
        for n, name in sorted(entries, reverse=True)[:6]:
            lines.append(f"  {size_text(n):>10}  {name}")
    return ok, lines


# --- stages (collect-all smoke) ------------------------------------------------------------


def tail_lines(text: str, n: int = TAIL_LINES) -> list[str]:
    return text.splitlines()[-n:] if n > 0 else []


def subprocess_result(r: dict, log: Path, detail: dict | None = None) -> dict:
    """The stage-function result of a ``run_logged`` outcome."""
    if r.get("timed_out"):
        message = f"timed out after {fmt_seconds(r['seconds'])} ({r['seconds']:.0f} s)"
    elif r["returncode"] != 0:
        message = f"exit {r['returncode']}"
    else:
        message = ""
    return {
        "ok": r["returncode"] == 0 and not r.get("timed_out"),
        "log": log,
        "message": message,
        "detail": {"returncode": r["returncode"], "seconds": r["seconds"], **(detail or {})},
    }


def run_stage(name: str, fn, clock=time.monotonic) -> dict:
    """Run ``fn()`` (returns ``{"ok": bool, "log": Path|None, "message": str, "detail": dict}``)
    inside its own guard: any exception is a failed stage, never an escape. The record has
    ``name``, ``status`` (ok or failed), ``ok``, ``seconds``, ``message``, ``detail`` and, on
    failure, ``tail``: the last 30 lines of the stage's log (or its traceback)."""
    t0 = clock()
    tail: list[str] = []
    try:
        res = fn()
        ok = bool(res.get("ok"))
        message, detail, log = res.get("message", ""), res.get("detail", {}), res.get("log")
        if not ok:
            if log and Path(log).is_file():
                tail = tail_lines(Path(log).read_text(encoding="utf-8", errors="replace"))
            if not tail:
                tail = [message or "no output"]
    except Exception as e:  # noqa: BLE001 - one stage must never stop the others
        import traceback

        ok, detail = False, {}
        message = f"{type(e).__name__}: {e}"
        tail = tail_lines(traceback.format_exc())
    return {
        "name": name,
        "status": "ok" if ok else "failed",
        "ok": ok,
        "seconds": round(clock() - t0, 1),
        "message": message,
        "detail": detail,
        "tail": [] if ok else tail,
    }


def skipped_stage(name: str, why: str) -> dict:
    return {
        "name": name, "status": "skipped", "ok": False, "seconds": 0.0, "message": why,
        "detail": {}, "tail": [],
    }  # fmt: skip


def stage_table(stages: list[dict]) -> list[str]:
    lines = [f"{'stage':24} {'status':8} {'seconds':>8}  note"]
    for st in stages:
        lines.append(
            f"{st['name']:24} {st['status']:8} {st['seconds']:8.1f}  {st.get('message') or ''}"
        )
    return lines


def smoke_verdict(stages: list[dict]) -> str:
    """``SMOKE OK`` only if every stage is ok, else ``SMOKE PARTIAL: <names of the others>``
    (a skipped stage counts: it did not pass)."""
    if not stages:
        return "SMOKE PARTIAL: no stage ran"
    failing = [st["name"] for st in stages if st["status"] != "ok"]
    return "SMOKE OK" if not failing else "SMOKE PARTIAL: " + ", ".join(failing)


# --- the RF-DETR training probe ------------------------------------------------------------

# Runs in a child process (``python -c``), cwd = the code folder: three real training
# iterations of RF-DETR-Nano on the real dataset at batch 4, timed per iteration (data loading
# included), with the GPU memory peak. It wraps pytorch_lightning's Trainer so that train()
# stops after N batches with no validation; everything else is the project's own call.
PROBE_SOURCE = r"""
import json, sys, time, traceback
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

dataset_dir, weights, out_dir, n_iter = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
stamps = []


class Probe(Callback):
    def on_train_start(self, trainer, module):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        stamps.append(time.perf_counter())

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        torch.cuda.synchronize()
        stamps.append(time.perf_counter())
        print(f"PROBE iteration {len(stamps) - 1}: {stamps[-1] - stamps[-2]:.3f} s", flush=True)


_orig_init = pl.Trainer.__init__


def _patched_init(self, *args, **kwargs):
    kwargs.update(limit_train_batches=n_iter, limit_val_batches=0, num_sanity_val_steps=0)
    kwargs["callbacks"] = [*(kwargs.get("callbacks") or []), Probe()]
    _orig_init(self, *args, **kwargs)


pl.Trainer.__init__ = _patched_init
error = None
try:
    from rfdetr import RFDETRNano
    from training.autolabel import load_classes
    from training.finetune import rfdetr_train_kwargs, seed_everything

    classes = load_classes(Path("config/experiment.json"))
    seed_everything(0)
    kw = rfdetr_train_kwargs(Path(dataset_dir), Path(out_dir), 1, 5e-5, 4, 1, 0, classes)
    RFDETRNano(pretrain_weights=weights).train(**kw)
except Exception as e:  # train() may complain after the probe batches (no validation ran)
    error = repr(e)
    traceback.print_exc()
deltas = [b - a for a, b in zip(stamps, stamps[1:])]
steady = deltas[1:] or deltas
result = {
    "iterations": len(deltas),
    "seconds_each": [round(d, 3) for d in deltas],
    "seconds_first": round(deltas[0], 3) if deltas else None,
    "seconds_per_iteration": round(sum(steady) / len(steady), 3) if steady else None,
    "gpu_peak_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
    "gpu_reserved_peak_mb": round(torch.cuda.max_memory_reserved() / 1024**2, 1),
    "batch_size": 4,
    "error_after_probe": error if len(deltas) >= n_iter else None,
}
print("PROBE_RESULT " + json.dumps(result), flush=True)
sys.exit(0 if len(deltas) >= n_iter else 1)
"""


def probe_cmd(dataset_dir: Path, weights: Path, out_dir: Path, iterations: int) -> list[str]:
    return [
        sys.executable, "-c", PROBE_SOURCE, str(dataset_dir), str(weights), str(out_dir),
        str(iterations),
    ]  # fmt: skip


def parse_probe(text: str) -> dict | None:
    """The JSON of the last ``PROBE_RESULT`` line, or None."""
    found = None
    for line in text.splitlines():
        if line.startswith("PROBE_RESULT "):
            try:
                found = json.loads(line[len("PROBE_RESULT ") :])
            except ValueError:
                found = None
    return found


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


class DiscoveryError(Exception):
    """A dataset or weights file could not be located or verified; the message says what was
    examined."""


def walk_dirs(root: Path, max_depth: int = MAX_DEPTH) -> list[Path]:
    """``root`` and every directory below it, at most ``max_depth`` levels down (root is level
    0), in a fixed order. Links are not followed."""
    root = Path(root)
    if not root.is_dir():
        return []
    out = []
    for cur, dirs, _files in os.walk(root):
        dirs.sort()
        out.append(Path(cur))
        if len(Path(cur).relative_to(root).parts) >= max_depth:
            dirs[:] = []
    return out


def find_files_named(root: Path, name: str, max_depth: int = MAX_DEPTH) -> list[Path]:
    """Every file called ``name`` at most ``max_depth`` levels under ``root``, shallowest first."""
    hits = [d / name for d in walk_dirs(root, max_depth) if (d / name).is_file()]
    return sorted(hits, key=lambda p: (len(p.parts), str(p)))


def find_sentinels(root: Path, filename: str) -> list[Path]:
    """The directories (shallowest first) holding ``filename``; Kaggle's mount layout for
    attached datasets has changed between image versions, so nothing is hard-coded."""
    return [p.parent for p in find_files_named(root, filename)]


def find_sentinel(root: Path, filename: str) -> Path | None:
    hits = find_sentinels(root, filename)
    return hits[0] if hits else None


def read_manifest(root: Path, filename: str, label: str) -> dict:
    """The manifest called ``filename`` found under ``root``. Several copies must be
    byte-identical (the same dataset mounted twice); otherwise it is ambiguous."""
    dirs = find_sentinels(root, filename)
    if not dirs:
        raise DiscoveryError(
            f"{label}: {filename} not found within {MAX_DEPTH} levels under {root}; attach the "
            "dataset as an input of the kernel"
        )
    blobs = {d: (d / filename).read_bytes() for d in dirs}
    if len(set(blobs.values())) > 1:
        raise DiscoveryError(
            f"{label}: {filename} is ambiguous, different copies in: " + ", ".join(map(str, dirs))
        )
    print(
        f"{label} manifest: {dirs[0] / filename}"
        + (f" ({len(dirs)} identical copies)" if len(dirs) > 1 else "")
    )
    return json.loads(blobs[dirs[0]].decode("utf-8"))


def find_roots(root: Path, expected, max_depth: int = MAX_DEPTH) -> dict:
    """Which directory under ``root`` (at most ``max_depth`` levels) holds EVERY path of
    ``expected`` (relative posix paths, a dict or any iterable)? The first path in sorted
    order is a cheap probe: only a directory holding it is checked in full. ``examined`` lists
    each directory looked at in full (those with the probe, or, when none has it, those with
    one of the manifest's top-level folders): ``found`` of ``total`` files and the first ten
    ``missing`` paths."""
    rels = sorted(expected)
    if not rels:
        raise DiscoveryError("the manifest lists no files")
    root = Path(root)
    dirs = walk_dirs(root, max_depth)
    tops = sorted({r.split("/")[0] for r in rels if "/" in r})

    def examine(d: Path) -> dict:
        missing = [r for r in rels if not (d / r).is_file()]
        return {
            "root": str(d),
            "found": len(rels) - len(missing),
            "total": len(rels),
            "missing": missing[:PROBLEM_LIMIT],
        }

    examined = [examine(d) for d in dirs if (d / rels[0]).is_file()]
    if not examined:
        examined = [examine(d) for d in dirs if any((d / t).is_dir() for t in tops)]
    return {
        "root": str(root),
        "root_exists": root.is_dir(),
        "probe": rels[0],
        "total": len(rels),
        "dirs_walked": len(dirs),
        "max_depth": max_depth,
        "examined": examined,
        "matches": [Path(x["root"]) for x in examined if x["found"] == x["total"]],
    }


def format_examined(res: dict) -> list[str]:
    lines = [
        f"examined {res['dirs_walked']} directories (at most {res['max_depth']} levels) under "
        f"{res['root']}; probe {res['probe']!r}"
    ]
    if not res["examined"]:
        lines.append("  no directory held the probe file or a top-level folder of the manifest")
    for x in res["examined"][:PROBLEM_LIMIT]:
        lines.append(f"  {x['root']}: {x['found']} of {x['total']} manifest files found")
        if x["missing"]:
            lines.append("    first missing: " + ", ".join(x["missing"]))
    return lines


def single_root(res: dict, what: str, input_root: Path) -> Path:
    """The one matching directory of a ``find_roots`` result; raises with the evidence when
    there is none or more than one."""
    if not res["root_exists"]:
        raise DiscoveryError(f"{what}: the input folder {input_root} does not exist")
    n = len(res["matches"])
    if n == 1:
        return res["matches"][0]
    if n > 1:
        raise DiscoveryError(
            f"{what}: ambiguous, {n} directories hold every manifest file: "
            + "; ".join(map(str, res["matches"]))
        )
    raise DiscoveryError(
        f"{what}: no directory under {input_root} holds all {res['total']} manifest files.\n"
        + "\n".join(format_examined(res))
    )


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


def stage_tree(
    input_root: Path,
    dest: Path,
    expected,
    zip_name: str,
    zip_sha256s: list[str],
    label: str,
) -> Path:
    """Put the files of ``expected`` under a directory the script can write to and return it.
    The files are looked for as a tree anywhere under ``input_root`` (Kaggle unpacks a zip
    dataset itself; the mount path and any extra folder vary): exactly one directory holding
    all of them is copied to ``dest`` (only the manifest's files); two are ambiguous. If there
    is no tree, a file called ``zip_name`` is checked against ``zip_sha256s``, unzipped into
    ``dest`` and searched the same way (the zip may hold an extra top folder)."""
    input_root, dest = Path(input_root), Path(dest)
    res = find_roots(input_root, expected)
    if res["matches"]:
        src = single_root(res, label, input_root)
        for rel in sorted(expected):
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / rel, dest / rel)
        print(f"{label}: tree found at {src}; {len(list(expected))} files copied to {dest}")
        return dest
    zips = find_files_named(input_root, zip_name)
    if not zips:
        single_root(res, label, input_root)  # raises, with what was examined
    zip_path = zips[0]
    got = sha256_file(zip_path)
    print(f"{label}: no tree; {zip_path} sha256 {got}")
    if any(got != s for s in zip_sha256s):
        raise DiscoveryError(
            f"{label}: {zip_path} sha256 {got} differs from the packed {zip_sha256s}"
        )
    print(f"{label}: unzipped {unzip_safe(zip_path, dest)} entries -> {dest}")
    inner = find_roots(dest, expected)
    return single_root(inner, f"{label} (unzipped)", dest)


def locate_weights(input_root: Path, weights_meta: dict[str, dict]) -> dict[str, Path]:
    """Each weights file by name anywhere under ``input_root``, verified by sha256."""
    out = {}
    for name, meta in weights_meta.items():
        hits = find_files_named(input_root, name)
        if not hits:
            raise DiscoveryError(
                f"weights file {name} not found within {MAX_DEPTH} levels of {input_root}"
            )
        good = [h for h in hits if sha256_file(h) == meta["sha256"]]
        if not good:
            raise DiscoveryError(
                f"weights file {name}: found {[str(h) for h in hits]} but the sha256 differs from "
                f"the packed {meta['sha256']}"
            )
        out[name] = good[0]
    return out


# --- start-of-run diagnostics (pure) -------------------------------------------------------


def size_text(n: float) -> str:
    if n < 1024:
        return f"{int(n)} B"
    for unit, div, fmt in (("KB", 1024, ".1f"), ("MB", 1024**2, ".1f"), ("GB", 1024**3, ".2f")):
        if n < div * 1024 or unit == "GB":
            return f"{n / div:{fmt}} {unit}"
    return f"{n} B"  # pragma: no cover


def list_tree(root: Path, max_depth: int = LIST_DEPTH, max_lines: int = LIST_LINES) -> list[str]:
    """A depth-limited listing of ``root``: at most ``max_depth`` levels and ``max_lines``
    lines, folders before files, sizes for files, and one count-and-total line for a folder
    with more than ``BIG_FOLDER`` files."""
    root = Path(root)
    if not root.is_dir():
        return [f"{root} does not exist"]
    lines = [f"{root}/"]
    truncated = False

    def emit(d: Path, level: int) -> None:
        nonlocal truncated
        if truncated:
            return
        try:
            entries = sorted(os.scandir(d), key=lambda e: e.name)
        except OSError as e:
            lines.append("  " * level + f"<unreadable: {e}>")
            return
        dirs = [e for e in entries if e.is_dir(follow_symlinks=False)]
        files = [e for e in entries if not e.is_dir(follow_symlinks=False)]
        pad = "  " * level
        for e in dirs:
            if len(lines) >= max_lines:
                truncated = True
                return
            if level >= max_depth:
                lines.append(f"{pad}{e.name}/  (not expanded)")
                continue
            lines.append(f"{pad}{e.name}/")
            emit(Path(e.path), level + 1)
        if len(files) > BIG_FOLDER:
            total = sum(e.stat(follow_symlinks=False).st_size for e in files)
            if len(lines) >= max_lines:
                truncated = True
                return
            lines.append(f"{pad}[{len(files)} files, {size_text(total)}]")
            return
        for e in files:
            if len(lines) >= max_lines:
                truncated = True
                return
            lines.append(f"{pad}{e.name}  {size_text(e.stat(follow_symlinks=False).st_size)}")

    emit(root, 1)
    if truncated:
        lines.append(f"... truncated at {max_lines} lines")
    return lines


def free_disk_line(path: Path) -> str:
    try:
        u = shutil.disk_usage(path)
    except OSError as e:
        return f"free disk at {path}: unavailable ({e})"
    return f"free disk at {path}: {u.free / 1024**3:.1f} GB free of {u.total / 1024**3:.1f} GB"


def print_start_diagnostics() -> None:
    print("-- /kaggle/input listing (at most 4 levels, 200 lines) --")
    print("\n".join(list_tree(INPUT)))
    print(free_disk_line(WORK), flush=True)


# --- one GPU (pure) ------------------------------------------------------------------------


def single_gpu_env(env: dict) -> dict:
    """A copy of ``env`` that lets a child process see only GPU 0. Kaggle's T4 x2 is two
    devices; training on one keeps the numbers comparable (effective batch 4 x 4 = 16)."""
    out = dict(env)
    out["CUDA_VISIBLE_DEVICES"] = "0"
    return out


def gpu_note(gpus_seen: int) -> str:
    if gpus_seen <= 0:
        return "no GPU seen"
    used = "using it" if gpus_seen == 1 else "using 1 (CUDA_VISIBLE_DEVICES=0)"
    note = f"{gpus_seen} GPU{'s' if gpus_seen != 1 else ''} seen; {used}"
    if gpus_seen > 1:
        who = "the second is" if gpus_seen == 2 else "the others are"
        note += f"; {who} deliberately unused"
    return note + "; effective batch stays 4 x 4 = 16"


def annotate_summary(path: Path, gpus_seen: int) -> bool:
    """Add ``gpus_seen`` and ``gpu_used`` to a ``train_summary.json`` (training.finetune
    writes it, and knows nothing of the kernel's GPU choice). False if it is absent or bad."""
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(data, dict):
        return False
    data["gpus_seen"] = gpus_seen
    data["gpu_used"] = 1 if gpus_seen > 0 else 0
    p.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return True


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
    stages: list[dict] | None = None,
    extra: dict | None = None,
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
        "gpus_seen": machine.get("gpus_seen", 0),
        "gpu_used": 1 if machine.get("gpus_seen", 0) > 0 else 0,
        "gpu_names": machine.get("gpu_names"),
        "gpu_note": gpu_note(machine.get("gpus_seen", 0)),
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
        "stages": stages or [],
        "verdict": smoke_verdict(stages) if stages else None,
        **(extra or {}),
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
    """The environment of every child process: one GPU only (``single_gpu_env``); internet is
    on in this kernel, so allow Hugging Face downloads (training.autolabel sets
    HF_HUB_OFFLINE=1 only when the variable is unset)."""
    env = single_gpu_env(dict(os.environ))
    env["HF_HUB_OFFLINE"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def machine_info() -> dict:
    info = {"python": sys.version.split()[0], "cuda_available": False, "gpus_seen": 0}
    try:
        import torch

        info.update(
            torch=torch.__version__,
            torch_build=f"cuda {torch.version.cuda}" if torch.version.cuda else "cpu",
        )
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["gpus_seen"] = torch.cuda.device_count()
            info["gpu_names"] = [
                torch.cuda.get_device_properties(i).name for i in range(info["gpus_seen"])
            ]
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
        "gpus_seen",
        "gpu_names",
    ):
        print(f"{k:18}: {info.get(k)}")
    print(f"{'gpu policy':18}: {gpu_note(info.get('gpus_seen') or 0)}")
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
    """Versions of the packages the constraints file freezes (``FROZEN``) that are installed."""
    import importlib.metadata as md

    out = {}
    for n in FROZEN:
        try:
            out[n] = md.version(n)
        except md.PackageNotFoundError:
            pass
    return out


# --- the run ----------------------------------------------------------------------------


def prepare(guard: TimeGuard, tmp_root: Path) -> dict:
    """Locate, verify and stage the two datasets wherever Kaggle mounted them (see
    ``stage_tree``) under ``tmp_root``, never under /kaggle/working (the kernel output). Any
    discovery problem stops the run with what was examined."""
    assert PACKAGE is not None, "this script was not generated by training.kaggle_pack"
    try:
        ds_manifest = read_manifest(INPUT, DATASET_SENTINEL, "dataset")
        code_manifest = read_manifest(INPUT, CODE_SENTINEL, "code")
        data_root = stage_tree(
            INPUT,
            tmp_root / "data" / "dataset",
            ds_manifest["files"],
            ds_manifest["zip"]["name"],
            [PACKAGE["dataset_zip_sha256"], ds_manifest["zip"]["sha256"]],
            "dataset",
        )
        problems = check_files(data_root, ds_manifest["files"])
        if problems:
            stop(
                f"dataset files differ from the packed manifest in {data_root}: "
                f"{problems[:PROBLEM_LIMIT]} ({len(problems)})",
                2,
            )
        print(f"dataset OK: {len(ds_manifest['files'])} files verified by sha256 in {data_root}")
        code_root = stage_tree(
            INPUT,
            tmp_root / "code",
            code_manifest["files"],
            code_manifest["zip"]["name"],
            [PACKAGE["code_zip_sha256"], code_manifest["zip"]["sha256"]],
            "code",
        )
        problems = check_files(code_root, code_manifest["files"])
        if problems:
            stop(
                f"code files differ from the packed manifest in {code_root}: "
                f"{problems[:PROBLEM_LIMIT]} ({len(problems)})",
                2,
            )
        print(f"code OK: {len(code_manifest['files'])} files verified by sha256 in {code_root}")
        for name, meta in code_manifest["weights"].items():
            if PACKAGE["weights"].get(name) != meta["sha256"]:
                stop(
                    f"weights {name}: manifest sha256 differs from the one packed in this script", 2
                )
        weights = locate_weights(INPUT, code_manifest["weights"])
    except DiscoveryError as e:
        stop(str(e), 2)
    for name, p in weights.items():
        meta = code_manifest["weights"][name]
        print(
            f"weights {name}: {p}, sha256 ok, {p.stat().st_size} bytes, licence {meta['licence']}"
        )
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


def installed_all() -> dict[str, str]:
    """name -> version of every installed distribution (names normalised)."""
    import importlib.metadata as md

    out = {}
    for d in md.distributions():
        name = d.metadata["Name"]
        if name:
            out[norm(name)] = d.version
    return out


def stop_on_pip_failure(r: dict, text: str, frozen: list[str], what: str) -> None:
    names = conflicting_packages(text)
    if looks_like_conflict(text):
        stop(
            f"pip cannot satisfy the pins together with the frozen core ({what}; frozen: "
            f"{sorted(frozen)}). Conflicting packages named by pip: "
            f"{names or 'see the pip output above'}. Nothing was reinstalled. Report it; "
            "do not force a package change on Kaggle.",
            3,
        )
    stop(f"pip install failed ({what}, exit {r['returncode']}); see above", 3)


def install_pins(code_root: Path) -> dict:
    """pip install of the pins with the core frozen. First ``--dry-run`` (when this pip has
    it): the plan is printed and a plan that would change a core package stops the run. After
    the real install the core versions are asserted unchanged and every package pip added or
    changed is printed."""
    print(
        "\n== install pinned packages (torch and the core stack stay as Kaggle has them) ==",
        flush=True,
    )
    core_before = installed_versions()
    all_before = installed_all()
    print("core frozen at the installed versions:", json.dumps(core_before, indent=1), sep="\n")
    pins, notes = drop_installed_optional(
        pins_from_lock((code_root / "uv.lock").read_text(encoding="utf-8")), all_before
    )
    print("\n".join(notes))
    print("pins:", *pins, sep="\n  ")
    cons = OUT / "constraints.txt"
    cons.write_text(constraints_text(core_before), encoding="utf-8")
    plan_lines: list[str] | None = None
    help_text = capture([sys.executable, "-m", "pip", "install", "--help"], 60)
    if pip_supports_dry_run(help_text):
        print("\n-- pip install --dry-run: the plan --", flush=True)
        log = OUT / "pip_dry_run.log"
        r = run_logged(
            pip_install_cmd(pins, cons, dry_run=True), log, cwd=code_root, timeout_s=900,
            env=child_env(),
        )  # fmt: skip
        text = log.read_text(encoding="utf-8", errors="replace")
        if r["returncode"] != 0:
            stop_on_pip_failure(r, text, list(core_before), "dry run")
        plan = parse_dry_run_plan(text)
        plan_lines = describe_changes(all_before, apply_plan(all_before, plan))
        print(f"pip plan: {len(plan)} packages would be installed")
        print("\n".join(f"  {x}" for x in plan_lines) or "  (nothing changes)")
        core = plan_core_changes(plan)
        if core:
            stop(
                f"the pip plan would change core packages {core}; nothing was installed. "
                "Report it; do not force a package change on Kaggle.",
                3,
            )
        print("plan leaves the core alone; installing", flush=True)
    else:
        print("this pip has no --dry-run: installing without a plan (core checked afterwards)")
    t0 = time.monotonic()
    log = OUT / "pip_install.log"
    r = run_logged(pip_install_cmd(pins, cons), log, cwd=code_root, timeout_s=1800, env=child_env())
    seconds = time.monotonic() - t0
    text = log.read_text(encoding="utf-8", errors="replace")
    if r["returncode"] != 0:
        stop_on_pip_failure(r, text, list(core_before), "install")
    names = conflicting_packages(text)
    if names:
        print(f"WARNING: pip reports incompatible requirements involving: {names}")
    moved = changed_packages(core_before, installed_versions())
    if moved:
        stop(
            f"core packages changed during install: {moved}: {core_before} -> "
            f"{installed_versions()}",
            3,
        )
    changes = describe_changes(all_before, installed_all())
    print(f"pip install finished in {fmt_seconds(seconds)}; core packages unchanged")
    print("packages pip added or changed:")
    print("\n".join(f"  {x}" for x in changes) or "  (none)")
    (OUT / "pip_freeze.txt").write_text(
        capture([sys.executable, "-m", "pip", "freeze"], 120), encoding="utf-8"
    )
    return {"seconds": seconds, "changes": changes, "plan": plan_lines}


def read_report(code_root: Path) -> dict:
    return json.loads((code_root / "reports" / "dataset.json").read_text(encoding="utf-8"))


def run_smoke(paths: dict, machine: dict, pip: dict) -> int:
    """Five guarded stages (see the module docstring); a failed stage is recorded and the
    rest still run. Returns 0 only if every stage passed and the outputs are small enough."""
    code_root, data_root, weights = paths["code_root"], paths["data_root"], paths["weights"]
    scratch = paths["tmp_root"] / "smoke"  # training scratch: never under /kaggle/working
    sys.path.insert(0, str(code_root))
    from training.finetune import default_epochs

    report = read_report(code_root)
    n_train = report["splits"]["train"]["images"]
    epochs = default_epochs(n_train)
    print(f"\n== SMOKE: n_train {n_train}, epochs {epochs} (S-E rule) ==", flush=True)
    t0 = time.monotonic()
    preflight: dict[str, dict] = {}
    dry: dict[str, dict] = {}

    def preflight_stage(model: str):
        def fn() -> dict:
            log = OUT / f"smoke_preflight_{model}.log"
            r = run_logged(
                preflight_cmd(model, weights[WEIGHT_FILE[model]], n_train, epochs), log,
                cwd=code_root, timeout_s=STAGE_TIMEOUT_S, env=child_env(),
            )  # fmt: skip
            parsed = parse_preflight(log.read_text(encoding="utf-8", errors="replace"))
            preflight[model] = {**parsed, "returncode": r["returncode"]}
            return subprocess_result(r, log, parsed)

        return fn

    def dry_run_stage(model: str):
        def fn() -> dict:
            out_dir, w = scratch / model, weights[WEIGHT_FILE[model]]
            log = OUT / f"smoke_dry_run_{model}.log"
            r = run_logged(
                dry_run_cmd(model, w, out_dir, code_root / "reports" / "dataset.json"),
                log, cwd=code_root, timeout_s=STAGE_TIMEOUT_S, env=child_env(),
            )  # fmt: skip
            dry[model] = {"returncode": r["returncode"], "seconds": r["seconds"]}
            summary = out_dir / "train_summary.json"  # the only file kept: weights stay in temp
            if summary.is_file():
                keep = OUT / "smoke" / model
                keep.mkdir(parents=True, exist_ok=True)
                shutil.copy2(summary, keep / summary.name)
            return subprocess_result(r, log)

        return fn

    def probe_stage() -> dict:
        log, w = OUT / "smoke_probe_rfdetr.log", weights[WEIGHT_FILE["rfdetr"]]
        r = run_logged(
            probe_cmd(data_root, w, scratch / "probe", PROBE_ITERATIONS),
            log, cwd=code_root, timeout_s=STAGE_TIMEOUT_S, env=child_env(),
        )  # fmt: skip
        probe = parse_probe(log.read_text(encoding="utf-8", errors="replace"))
        res = subprocess_result(r, log, {"probe": probe})
        if probe is None and res["ok"]:
            res.update(ok=False, message="the probe printed no PROBE_RESULT line")
        elif probe is not None:
            line = (
                f"PROBE rfdetr batch 4, {probe['iterations']} iterations: "
                f"{probe['seconds_per_iteration']} s per iteration "
                f"(each {probe['seconds_each']}), GPU memory peak {probe['gpu_peak_mb']} MB "
                f"allocated / {probe['gpu_reserved_peak_mb']} MB reserved"
            )
            print(line, flush=True)
            res["message"] = (
                res["message"]
                or f"{probe['seconds_per_iteration']} s/iter, {probe['gpu_peak_mb']} MB peak"
            )
        return res

    stages: list[dict] = []
    for name, fn in (
        ("preflight_rfdetr", preflight_stage("rfdetr")),
        ("preflight_yolo11n", preflight_stage("yolo11n")),
        ("dry_run_rfdetr", dry_run_stage("rfdetr")),
        ("dry_run_yolo11n", dry_run_stage("yolo11n")),
    ):
        print(f"\n-- stage {name} (timeout {STAGE_TIMEOUT_S} s) --", flush=True)
        stages.append(run_stage(name, fn))
        print(f"-- stage {name}: {stages[-1]['status']} in {stages[-1]['seconds']} s", flush=True)
    name = "train_probe_rfdetr"
    if stages[2]["ok"]:
        print(f"\n-- stage {name} (timeout {STAGE_TIMEOUT_S} s) --", flush=True)
        stages.append(run_stage(name, probe_stage))
        print(f"-- stage {name}: {stages[-1]['status']} in {stages[-1]['seconds']} s", flush=True)
    else:
        stages.append(skipped_stage(name, "dry_run_rfdetr did not pass"))

    smoke_seconds = time.monotonic() - t0
    problems = [f"{st['name']}: {st['message'] or st['status']}" for st in stages if not st["ok"]]
    print("\n" + "=" * 78 + "\nSTAGE TABLE")
    print("\n".join(stage_table(stages)))
    for st in stages:
        if st["tail"]:
            print(f"\n-- last {TAIL_LINES} lines of {st['name']} ({st['status']}) --")
            print("\n".join(st["tail"]))
    size_ok, size_lines = output_size_check(WORK, OUTPUT_LIMIT_BYTES)
    rep = smoke_report(
        machine=machine, n_train=n_train, epochs=epochs, preflight=preflight, dry_runs=dry,
        smoke_seconds=smoke_seconds, pip_seconds=pip.get("seconds"),
        dataset_stamp=report.get("dataset_stamp"), problems=problems, stages=stages,
        extra={
            "pip_changes": pip.get("changes"),
            "pip_plan": pip.get("plan"),
            "temp_root": str(paths["tmp_root"]),
            "temp_note": paths["tmp_note"],
            "outputs_size_ok": size_ok,
            "outputs_size_text": size_lines[0],
        },
    )  # fmt: skip
    (OUT / "smoke_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print("\n" + "=" * 78 + "\nSMOKE REPORT\n" + json.dumps(rep, indent=1))
    print(f"\nsmoke part took {fmt_seconds(smoke_seconds)} (budget {fmt_seconds(SMOKE_BUDGET_S)})")
    print("\n".join(size_lines))
    if not size_ok:
        print(
            "OUTPUT TOO LARGE: the run fails on size alone (the stages above are unaffected); "
            "remove the largest entries named above or write them to the temp folder",
            flush=True,
        )
    print(rep["verdict"], flush=True)  # the last line: SMOKE OK or SMOKE PARTIAL: <stages>
    return 0 if rep["ok"] and size_ok else 1


def run_full(paths: dict, machine: dict, guard: TimeGuard) -> int:
    code_root, data_root, weights = paths["code_root"], paths["data_root"], paths["weights"]
    report_path = code_root / "reports" / "dataset.json"
    status: dict = {
        "mode": "FULL", "machine": machine, "seed": SEED,
        "gpus_seen": machine.get("gpus_seen", 0),
        "gpu_used": 1 if machine.get("gpus_seen", 0) > 0 else 0,
        "gpu_note": gpu_note(machine.get("gpus_seen", 0)),
        "dataset_stamp": read_report(code_root).get("dataset_stamp"),
        "models": [], "time_guard": {"limit_s": SESSION_LIMIT_S, "safety_s": SAFETY_S},
    }  # fmt: skip

    def save() -> None:
        status["elapsed_s"] = guard.elapsed()
        (OUT / "run_status.json").write_text(json.dumps(status, indent=1), encoding="utf-8")

    save()
    print(f"GPU policy: {status['gpu_note']}", flush=True)
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
        annotate_summary(out_dir / "train_summary.json", machine.get("gpus_seen", 0))
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
    print_start_diagnostics()
    machine = banner()
    if not machine.get("cuda_available"):
        stop(
            "CUDA is not available in this kernel: enable a GPU accelerator in the kernel settings",
            2,
        )
    try:
        tmp_root, tmp_note = choose_temp_root(temp_free_probe)
    except DiscoveryError as e:
        stop(str(e), 2)
    tmp_root.mkdir(parents=True, exist_ok=True)
    print(f"temp folder for datasets, code and training scratch: {tmp_note}")
    print(f"{WORK} keeps only outputs/ (limit {size_text(OUTPUT_LIMIT_BYTES)})", flush=True)
    paths = prepare(guard, tmp_root)
    paths.update(tmp_root=tmp_root, tmp_note=tmp_note)
    sys.path.insert(0, str(paths["code_root"]))
    verify_unzipped(paths["code_root"], paths["data_root"])
    pip = install_pins(paths["code_root"])
    machine = machine_info()
    if MODE == "SMOKE":
        return run_smoke(paths, machine, pip)
    if MODE == "FULL":
        rc = run_full(paths, machine, guard)
        size_ok, size_lines = output_size_check(WORK, OUTPUT_LIMIT_BYTES)
        print("\n".join(size_lines))
        if not size_ok:
            print("OUTPUT TOO LARGE: the run fails on size alone; the training itself finished")
        return rc if size_ok else max(rc, 1)
    stop(f"MODE must be SMOKE or FULL, got {MODE!r}", 2)
    return 2


if __name__ == "__main__":
    sys.exit(main())
