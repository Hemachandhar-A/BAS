"""Prepare (never upload) the Kaggle package for the detector fine-tune.

  python -m training.kaggle_pack [--out data/kaggle_upload] [--username hemachandhara]
        [--rfdetr-weights PATH] [--yolo-weights PATH] [--allow-dirty]
  python -m training.kaggle_pack --set-mode FULL [--out data/kaggle_upload]
  python -m training.kaggle_pack --print-commands [--out data/kaggle_upload]

NO NETWORK: this module only reads the repo, the dataset and the cached weights, and writes
under ``--out`` (git-ignored ``data/``). It never runs the Kaggle CLI; the commands the Lead runs
later are printed by ``--print-commands`` and written to ``COMMANDS.md``.

Layout written (three Kaggle objects, one folder each):

  dataset/  dataset.zip (data/dataset/{train,valid,test}, deterministic), dataset_manifest.json
            (zip sha256 + sha256 of every file inside), dataset-metadata.json
  code/     code.zip (``git archive HEAD`` of what training needs), code_manifest.json,
            rf-detr-nano.pth, yolo11n.pt, dataset-metadata.json
  kernel/   run_training.py (the script kernel, PACKAGE block filled), kernel-metadata.json

plus ``package_manifest.json`` and ``COMMANDS.md``. Before packing, ``verify_dataset`` must pass
and the archived paths must be committed (the archive is of HEAD, whose commit is recorded).
No credential is read, written or embedded: the Lead's own Kaggle CLI login is used later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from training.kaggle.run_training import inject_package, pins_from_lock, set_mode

USERNAME = "hemachandhara"
DATASET_SLUG = "sih26174-dataset"
CODE_SLUG = "sih26174-code"
KERNEL_SLUG = "sih26174-train"
DEFAULT_OUT = Path("data/kaggle_upload")
SPLIT_FOLDERS = ("train", "valid", "test")
# training imports contracts (config load) and nothing from perception/, state/ or engine/:
# training.autolabel imports perception.hands only inside a function that finetune never calls
CODE_PATHS = (
    "training",
    "config",
    "contracts.py",
    "pyproject.toml",
    "uv.lock",
    "reports/dataset.json",
    "data/corrections/train.json",
)
ZIP_DATE = (1980, 1, 1, 0, 0, 0)
WEIGHTS = {
    "rf-detr-nano.pth": {
        "model": "RF-DETR-Nano (COCO-pretrained)",
        "licence": "Apache-2.0",
        "role": "primary detector",
    },
    "yolo11n.pt": {
        "model": "YOLO11n (COCO-pretrained)",
        "licence": "AGPL-3.0",
        "role": "benchmarked fallback only (AGENTS.md rule 15; log the licence decision)",
    },
}
SECRET_NAME = re.compile(r"(kaggle\.json|\.env$|token|secret|credential|id_rsa)", re.I)
SECRET_TEXT = re.compile(r'(KGAT_[A-Za-z0-9]+|"key"\s*:\s*"[0-9a-f]{20,}")')


class PackError(Exception):
    """The package cannot be built as asked."""


# --- pure helpers -----------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_dataset_files(dataset_dir: Path) -> list[tuple[str, Path]]:
    """(archive name, path) for every file of the three split folders, sorted by name."""
    entries = []
    for split in SPLIT_FOLDERS:
        folder = Path(dataset_dir) / split
        if not folder.is_dir():
            raise PackError(f"split folder missing: {folder}")
        entries += [(f"{split}/{p.name}", p) for p in folder.iterdir() if p.is_file()]
    return sorted(entries)


def deterministic_zip(dest: Path, entries: list[tuple[str, Path]]) -> dict:
    """Write ``entries`` to ``dest``: sorted, stored (no compression, so the bytes do not
    depend on the zlib build), fixed timestamp and permissions. The same files always give
    the same zip bytes. Returns ``{"sha256", "size_bytes", "files": {name: sha256}}``."""
    files: dict[str, str] = {}
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_STORED) as z:
        for name, path in sorted(entries):
            data = Path(path).read_bytes()
            info = zipfile.ZipInfo(name, date_time=ZIP_DATE)
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            z.writestr(info, data)
            files[name] = hashlib.sha256(data).hexdigest()
    return {
        "sha256": sha256_file(dest),
        "size_bytes": Path(dest).stat().st_size,
        "files": files,
    }


def zip_file_hashes(zip_path: Path) -> dict[str, str]:
    """name -> sha256 of every file member of a zip (directory entries are skipped)."""
    out = {}
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if not info.is_dir():
                out[info.filename] = hashlib.sha256(z.read(info)).hexdigest()
    return out


def dataset_metadata(user: str) -> dict:
    """Private is the default of ``kaggle datasets create`` (``-u`` would make it public). The
    licence name ``other`` is one of the CLI's own valid names (its ``datasets create`` help
    lists all, cc, gpl, odb, other); the data is the project's own recordings, not for release."""
    return {
        "title": "SIH26174 dataset",
        "id": f"{user}/{DATASET_SLUG}",
        "licenses": [{"name": "other"}],
        "description": (
            "Private. Frames of the Sample Transfer experiment with COCO boxes for five "
            "classes (train, valid, test split by run). Project data, not for redistribution."
        ),
    }


def code_metadata(user: str) -> dict:
    return {
        "title": "SIH26174 code",
        "id": f"{user}/{CODE_SLUG}",
        "licenses": [{"name": "other"}],
        "description": (
            "Private. Project code (training/, config/, contracts.py, lockfile), the dataset "
            "report and overlay, and the pretrained weights: RF-DETR-Nano (Apache-2.0) and "
            "YOLO11n (AGPL-3.0, fallback only)."
        ),
    }


def kernel_metadata(user: str) -> dict:
    """Only keys of the template ``kaggle kernels init`` writes (read from the installed CLI)."""
    return {
        "id": f"{user}/{KERNEL_SLUG}",
        "title": "SIH26174 train",
        "code_file": "run_training.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [f"{user}/{DATASET_SLUG}", f"{user}/{CODE_SLUG}"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def weights_manifest(files: dict[str, Path]) -> dict:
    out = {}
    for name, path in files.items():
        meta = WEIGHTS[name]
        out[name] = {
            "sha256": sha256_file(path),
            "size_bytes": Path(path).stat().st_size,
            "licence": meta["licence"],
            "model": meta["model"],
            "role": meta["role"],
        }
    return out


def secret_problems(folder: Path) -> list[str]:
    """Names or contents under ``folder`` that look like a credential (the package must hold
    none). Only small text files are read."""
    problems = []
    for p in sorted(Path(folder).rglob("*")):
        if not p.is_file():
            continue
        if SECRET_NAME.search(p.name):
            problems.append(f"{p}: file name looks like a credential")
        elif p.suffix in (".json", ".py", ".md", ".txt", ".toml") and p.stat().st_size < 2_000_000:
            if SECRET_TEXT.search(p.read_text(encoding="utf-8", errors="replace")):
                problems.append(f"{p}: content looks like a credential")
    return problems


def n_images(report: dict) -> dict[str, int]:
    return {s: report["splits"][s]["images"] for s in SPLIT_FOLDERS}


# --- git (local only) ---------------------------------------------------------------------


def git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise PackError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout


def dirty_paths(repo: Path, paths: tuple[str, ...] = CODE_PATHS) -> list[str]:
    out = git(repo, "status", "--porcelain", "--", *paths)
    return [ln for ln in out.splitlines() if ln.strip()]


def git_archive(repo: Path, dest: Path, paths: tuple[str, ...] = CODE_PATHS) -> str:
    """``git archive --format=zip HEAD`` of ``paths`` into ``dest``; returns the commit hash."""
    git(repo, "archive", "--format=zip", "-o", str(Path(dest).resolve()), "HEAD", "--", *paths)
    return git(repo, "rev-parse", "HEAD").strip()


# --- the package ------------------------------------------------------------------------


def build_package(
    repo: Path,
    out: Path,
    *,
    user: str = USERNAME,
    rfdetr_weights: Path,
    yolo_weights: Path,
    allow_dirty: bool = False,
    dataset_dir: Path | None = None,
    corrections_dir: Path | None = None,
    report_path: Path | None = None,
    classes: list[str] | None = None,
) -> dict:
    from training.autolabel import load_classes
    from training.verify_dataset import verify

    repo, out = Path(repo), Path(out)
    dataset_dir = Path(dataset_dir or repo / "data" / "dataset")
    corrections_dir = Path(corrections_dir or repo / "data" / "corrections")
    report_path = Path(report_path or repo / "reports" / "dataset.json")
    classes = classes or load_classes(repo / "config" / "experiment.json")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    counts = n_images(report)
    res = verify(dataset_dir, report_path, corrections_dir, classes, expect_train=counts["train"])
    if res.problems:
        raise PackError("verify_dataset reports problems:\n  " + "\n  ".join(res.problems))
    dirty = dirty_paths(repo)
    if dirty and not allow_dirty:
        raise PackError(
            "uncommitted changes in the archived paths (commit first):\n  " + "\n  ".join(dirty)
        )

    for sub in ("dataset", "code", "kernel"):
        if (out / sub).exists():
            shutil.rmtree(out / sub)
        (out / sub).mkdir(parents=True)

    # dataset/
    ds_zip = out / "dataset" / "dataset.zip"
    z = deterministic_zip(ds_zip, collect_dataset_files(dataset_dir))
    if z["files"] != {n: sha256_file(p) for n, p in collect_dataset_files(dataset_dir)}:
        raise PackError("dataset files changed while packing")
    ds_manifest = {
        "zip": {"name": "dataset.zip", "sha256": z["sha256"], "size_bytes": z["size_bytes"]},
        "dataset_stamp": report["dataset_stamp"],
        "images": counts,
        "files": z["files"],
    }
    (out / "dataset" / "dataset_manifest.json").write_text(
        json.dumps(ds_manifest, indent=1), encoding="utf-8"
    )
    (out / "dataset" / "dataset-metadata.json").write_text(
        json.dumps(dataset_metadata(user), indent=1), encoding="utf-8"
    )

    # code/
    code_zip = out / "code" / "code.zip"
    commit = git_archive(repo, code_zip)
    code_files = zip_file_hashes(code_zip)
    weights = {"rf-detr-nano.pth": Path(rfdetr_weights), "yolo11n.pt": Path(yolo_weights)}
    for name, path in weights.items():
        if not path.is_file():
            raise PackError(f"pretrained weights not found: {path}")
        shutil.copy2(path, out / "code" / name)
    code_manifest = {
        "zip": {
            "name": "code.zip",
            "sha256": sha256_file(code_zip),
            "size_bytes": code_zip.stat().st_size,
        },
        "git_commit": commit,
        "paths": list(CODE_PATHS),
        "files": code_files,
        "weights": weights_manifest({n: out / "code" / n for n in weights}),
    }
    (out / "code" / "code_manifest.json").write_text(
        json.dumps(code_manifest, indent=1), encoding="utf-8"
    )
    (out / "code" / "dataset-metadata.json").write_text(
        json.dumps(code_metadata(user), indent=1), encoding="utf-8"
    )

    # kernel/
    template = Path(__file__).with_name("kaggle") / "run_training.py"
    package = {
        "dataset_zip_sha256": ds_manifest["zip"]["sha256"],
        "code_zip_sha256": code_manifest["zip"]["sha256"],
        "dataset_stamp": report["dataset_stamp"],
        "n_train": counts["train"],
        "n_valid": counts["valid"],
        "n_test": counts["test"],
        "git_commit": commit,
        "weights": {n: m["sha256"] for n, m in code_manifest["weights"].items()},
    }
    script = inject_package(template.read_text(encoding="utf-8"), package)
    (out / "kernel" / "run_training.py").write_text(script, encoding="utf-8", newline="\n")
    (out / "kernel" / "kernel-metadata.json").write_text(
        json.dumps(kernel_metadata(user), indent=1), encoding="utf-8"
    )

    problems = secret_problems(out)
    if problems:
        raise PackError("credential-looking content in the package:\n  " + "\n  ".join(problems))

    lock_text = zipfile.ZipFile(code_zip).read("uv.lock").decode("utf-8")
    manifest = {
        "git_commit": commit,
        "dataset_stamp": report["dataset_stamp"],
        "images": counts,
        "username": user,
        "pins": pins_from_lock(lock_text),
        "never_installed": ["torch", "torchvision", "torchaudio"],
        "files": _listing(out),
    }
    (out / "package_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (out / "COMMANDS.md").write_text(commands_markdown(user, out), encoding="utf-8")
    return manifest


def _listing(out: Path) -> dict:
    return {
        p.relative_to(out).as_posix(): {"size_bytes": p.stat().st_size, "sha256": sha256_file(p)}
        for p in sorted(Path(out).rglob("*"))
        if p.is_file() and p.name not in ("package_manifest.json", "COMMANDS.md")
    }


# --- the commands for the Lead (printed, never run) --------------------------------------

NEEDS_YES = "needs the Lead's explicit yes (network)"
KAGGLE = "uv tool run kaggle"


def commands(user: str, out: Path) -> list[tuple[str, str]]:
    """(label, command) in the order the Lead runs them; every one needs the Lead's yes."""
    o = Path(out).as_posix()
    kernel = f"{user}/{KERNEL_SLUG}"
    return [
        ("1. create the private dataset", f"{KAGGLE} datasets create -p {o}/dataset"),
        ("2. create the private code dataset", f"{KAGGLE} datasets create -p {o}/code"),
        ("3. push the kernel in SMOKE mode", f"{KAGGLE} kernels push -p {o}/kernel"),
        (
            "4. kernel status (repeat until it says complete or error)",
            f"{KAGGLE} kernels status {kernel}",
        ),
        ("5. download the smoke output", f"{KAGGLE} kernels output {kernel} -p {o}/smoke_output"),
        (
            "6. switch the kernel script to FULL (local edit, no network)",
            f"python -m training.kaggle_pack --set-mode FULL --out {o}",
        ),
        ("7. push again in FULL mode", f"{KAGGLE} kernels push -p {o}/kernel"),
        ("8. kernel status", f"{KAGGLE} kernels status {kernel}"),
        ("9. download the training output", f"{KAGGLE} kernels output {kernel} -p {o}/full_output"),
    ]


def commands_markdown(user: str, out: Path) -> str:
    lines = [
        "# Kaggle commands (prepared, NOT run)",
        "",
        "Every step is marked; all but the local edit in step 6 contact Kaggle.",
        "",
    ]
    for label, cmd in commands(user, out):
        mark = "local, no network" if "kaggle_pack" in cmd else NEEDS_YES
        lines += [f"**{label}** ({mark})", "", "```bash", cmd, "```", ""]
    lines += [
        "If the pushed kernel lands on a P100: open the kernel on kaggle.com, "
        "Settings (right side panel), Accelerator, choose GPU T4 x2 (or GPU T4), save, "
        "then Run again.",
        "",
    ]
    return "\n".join(lines)


# --- CLI ----------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--username", default=USERNAME)
    ap.add_argument("--rfdetr-weights", type=Path, default=None)
    ap.add_argument("--yolo-weights", type=Path, default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--set-mode", choices=("SMOKE", "FULL"))
    g.add_argument("--print-commands", action="store_true")
    args = ap.parse_args(argv)

    script = args.out / "kernel" / "run_training.py"
    if args.set_mode:
        if not script.is_file():
            print(f"{script} does not exist; pack first", file=sys.stderr)
            return 2
        script.write_text(
            set_mode(script.read_text(encoding="utf-8"), args.set_mode),
            encoding="utf-8",
            newline="\n",
        )
        print(f"{script}: MODE = {args.set_mode}")
        return 0
    if args.print_commands:
        print(commands_markdown(args.username, args.out))
        return 0

    from training import gpu_preflight as gp

    try:
        manifest = build_package(
            Path.cwd(),
            args.out,
            user=args.username,
            rfdetr_weights=args.rfdetr_weights or gp.default_weights_path("rfdetr"),
            yolo_weights=args.yolo_weights or gp.default_weights_path("yolo11n"),
            allow_dirty=args.allow_dirty,
        )
    except PackError as e:
        print(f"cannot pack: {e}", file=sys.stderr)
        return 2
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=1))
    for name, meta in manifest["files"].items():
        print(f"{meta['size_bytes']:>12}  {meta['sha256']}  {name}")
    print("\nNothing was uploaded. Commands: python -m training.kaggle_pack --print-commands")
    return 0


if __name__ == "__main__":
    sys.exit(main())
