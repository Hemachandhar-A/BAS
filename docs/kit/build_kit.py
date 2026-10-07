"""docs/kit/build_kit.py -- stage the two Kaggle bundles of the judge replication kit.

Maintainer tool (see docs/KIT.md). Run from the repo root with the venv interpreter:

    .venv\\Scripts\\python.exe docs/kit/build_kit.py            # both bundles, into kit_staging/
    .venv\\Scripts\\python.exe docs/kit/build_kit.py --with-weights  # bundle B has weights

It only COPIES files (never moves, re-encodes, trims or stitches the originals), uses no network
and uploads nothing. It fails loudly (exit 1) when a weights file does not match the sha256 in
weights/MANIFEST.json, when a clip does not match runs/manifest.csv, or when a source is missing.
Bundle A (bas-demo-kit) is flat: Kaggle's default ``--dir-mode skip`` ignores sub-folders.
The committed hash list assets/ASSETS.sha256 is written from bundle A.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from contracts import sha256_of_file  # noqa: E402  (the repo's own helper, same hash everywhere)

DEMO_SLUG = "<kaggle-user>/bas-demo-kit"
FULL_SLUG = "<kaggle-user>/bas-dataset-full"

# (run id, staged name, story role). All four are val-split originals: no --allow-heldout needed.
DEMO_CLIPS = [
    ("x016", "01_x016_clean.mp4", "clean run"),
    ("x032", "02_x032_skip.mp4", "skipped step (START)"),
    ("x025", "03_x025_swap.mp4", "swapped order (yellow before red)"),
    ("x036", "04_x036_idle.mp4", "idle, a hand roams, no step is performed"),
]

PLAYLIST = """\
# bas-demo-kit playlist: one clip per line, relative to this file. Put it next to the clips in
# demo_videos/ and run:  python scripts/demo.py --playlist demo_videos/playlist.txt ...
# Every clip is its own run. NEVER stitch, trim or re-encode a clip.
01_x016_clean.mp4
02_x032_skip.mp4
03_x025_swap.mp4
04_x036_idle.mp4
# Not in the kit (see README section 6): x015 (a clean run on which the LIVE path raises a false
# omission alert, the cached replay does not), x022 (a correct run on which the START press is
# missed: a documented false alarm), x040 (val repeat, 8.6 s, the repeat is not detected),
# x038 (repeat, TEST split: needs --allow-heldout, a Lead decision). Get them from bas-dataset-full.
"""

KIT_README = """\
bas-demo-kit -- what a judge needs to RUN the demo (BAS, advisory step-checking assistant)

WHAT THIS IS
  Two model files and four short recorded clips. Nothing else is needed from Kaggle to run the
  recorded-clip demo; the code comes from the git repository.

WHERE TO UNZIP
  Copy the files into the cloned repository (the folder that holds README.md and scripts/):
    weights/detector_yolo11n.pt
    weights/hand_landmarker.task
    demo_videos/01_x016_clean.mp4  02_x032_skip.mp4  03_x025_swap.mp4  04_x036_idle.mp4
    demo_videos/playlist.txt
  (Create weights/ and demo_videos/ if they are missing. This dataset is flat on purpose.)
  Then:  python scripts/verify_assets.py     -> must end with READY
  Every file is also listed with its sha256 in SHA256SUMS.txt (same format as sha256sum).

LICENCES (facts, not legal advice)
  * detector_yolo11n.pt: Ultralytics YOLO11n fine-tuned weights, AGPL-3.0.
    Alternative, Apache-2.0: RF-DETR-Nano (weights/detector_rfdetr_nano.pth in bas-dataset-full
    with --with-weights, or from the project). Its status: trained and evaluated on the validation
    gold frames only; NOT validated for the pipeline and NOT evaluated on test.
  * hand_landmarker.task: MediaPipe hand landmarker, Apache-2.0 (weights/MANIFEST.json).
  * Clips: recorded by the project crew. DECISION - Lead: consent of the performer and the licence
    of the footage are not recorded in the repository.

DATASET LICENCE / VISIBILITY: DECISION - Lead (placeholder in dataset-metadata.json).
"""

DATASET_META = {
    "demo": {
        "title": "BAS demo kit",
        "id": DEMO_SLUG,
        "licenses": [{"name": "other"}],  # PLACEHOLDER: DECISION - Lead (docs/KIT.md)
        "description": "Files to run the BAS recorded-clip demo: two model files, four clips, a "
        "playlist and SHA256SUMS.txt. See KIT_README.txt. Licence and visibility: DECISION - Lead.",
    },
    "full": {
        "title": "BAS dataset full",
        "id": FULL_SLUG,
        "licenses": [{"name": "other"}],  # PLACEHOLDER: DECISION - Lead (docs/KIT.md)
        "description": "BAS Sample Transfer: 46 recorded runs, scripts, labels, COCO dataset and "
        "reports to reproduce the results. See DATASET_CARD.md. Licence, consent: DECISION - Lead.",
    },
}


class BuildError(Exception):
    pass


def _manifest() -> dict:
    return json.loads((ROOT / "weights" / "MANIFEST.json").read_text(encoding="utf-8"))


def _weights_expected() -> dict[str, dict]:
    """file name -> {sha256, size_bytes} from weights/MANIFEST.json."""
    m = _manifest()
    out = {d["file"]: d for d in m["detectors"]}
    out[m["hand"]["file"]] = m["hand"]
    return out


def _run_hashes() -> dict[str, dict]:
    with (ROOT / "runs" / "manifest.csv").open(encoding="utf-8", newline="") as fh:
        return {r["run_id"]: r for r in csv.DictReader(fh)}


def _copy(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise BuildError(f"source missing: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_weight(name: str, dest_dir: Path) -> None:
    exp = _weights_expected()[name]
    src = ROOT / "weights" / name
    got = sha256_of_file(src) if src.is_file() else "(missing)"
    if got != exp["sha256"]:
        raise BuildError(f"weights/{name}: sha256 {got} != MANIFEST.json {exp['sha256']}")
    _copy(src, dest_dir / name)


def write_sums(bundle: Path) -> list[tuple[str, str, int]]:
    rows = []
    for p in sorted(bundle.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS.txt":
            rows.append((sha256_of_file(p), p.relative_to(bundle).as_posix(), p.stat().st_size))
    (bundle / "SHA256SUMS.txt").write_text(
        "".join(f"{h}  {rel}\n" for h, rel, _ in rows), encoding="utf-8", newline="\n"
    )
    return rows


def write_meta(bundle: Path, key: str) -> None:
    (bundle / "dataset-metadata.json").write_text(
        json.dumps(DATASET_META[key], indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def build_demo(stage: Path) -> list[tuple[str, str, int]]:
    bundle = stage / "bas-demo-kit"
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True)
    runs = _run_hashes()
    for run_id, staged, _ in DEMO_CLIPS:
        src = ROOT / "runs" / run_id / "video.mp4"
        got = sha256_of_file(src) if src.is_file() else "(missing)"
        if got != runs[run_id]["video_sha256"]:
            raise BuildError(f"runs/{run_id}/video.mp4 sha256 {got} != runs/manifest.csv")
        if runs[run_id]["split"] == "test":
            raise BuildError(f"{run_id} is a test-split clip: not allowed in the default kit")
        _copy(src, bundle / staged)
    _copy_weight("detector_yolo11n.pt", bundle)
    _copy_weight("hand_landmarker.task", bundle)
    (bundle / "playlist.txt").write_text(PLAYLIST, encoding="utf-8", newline="\n")
    (bundle / "KIT_README.txt").write_text(KIT_README, encoding="utf-8", newline="\n")
    write_meta(bundle, "demo")
    return write_sums(bundle)


def build_full(stage: Path, with_weights: bool) -> list[tuple[str, str, int]]:
    bundle = stage / "bas-dataset-full"
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True)
    skipped: list[str] = []

    def take(rel: str) -> None:
        src = ROOT / rel
        if src.is_dir():
            for p in sorted(src.rglob("*")):
                if p.is_file():
                    _copy(p, bundle / p.relative_to(ROOT))
        elif src.is_file():
            _copy(src, bundle / rel)
        else:
            skipped.append(rel)

    for run_id, row in sorted(_run_hashes().items()):
        src = ROOT / "runs" / run_id / "video.mp4"
        got = sha256_of_file(src) if src.is_file() else "(missing)"
        if got != row["video_sha256"]:
            raise BuildError(f"runs/{run_id}/video.mp4 sha256 {got} != runs/manifest.csv")
        take(f"runs/{run_id}/video.mp4")
        take(f"runs/{run_id}/script.json")
    for rel in (
        "runs/manifest.csv",
        "runs/provenance.csv",
        "runs/run_plan.csv",
        "data/dataset",
        "data/corrections/train.json",
        "data/corrections/val.json",
        "data/corrections/test.json",
        "data/label_review.csv",
        "reports/dataset.json",
        "reports/training",
    ):
        take(rel)
    if with_weights:
        for name in _weights_expected():
            _copy_weight(name, bundle / "weights")
        take("weights/MANIFEST.json")
    card = ROOT / "docs" / "DATASET_CARD.md"
    if card.is_file():
        _copy(card, bundle / "DATASET_CARD.md")
    else:
        skipped.append("docs/DATASET_CARD.md")
    write_meta(bundle, "full")
    rows = write_sums(bundle)
    if skipped:
        print("NOT STAGED (does not exist): " + ", ".join(skipped))
    return rows


def write_assets_list(rows: list[tuple[str, str, int]]) -> Path:
    """assets/ASSETS.sha256: destination path in the repo -> sha256, from bundle A."""
    dest = {"detector_yolo11n.pt": "weights", "hand_landmarker.task": "weights"}
    lines = [
        "# assets/ASSETS.sha256 -- expected sha256 of the demo kit (scripts/verify_assets.py).",
        "# Generated by docs/kit/build_kit.py from the staged bundle.",
        "# Format: <sha256>  <repo-relative path>.",
        f"# dataset: {DEMO_SLUG}",
        f"# dataset-full: {FULL_SLUG}",
        "# The dataset slugs above are placeholders until the Lead publishes the bundles.",
    ]
    for h, rel, _ in rows:
        if rel in ("SHA256SUMS.txt", "KIT_README.txt", "dataset-metadata.json"):
            continue
        folder = dest.get(rel, "demo_videos")
        lines.append(f"{h}  {folder}/{rel}")
    out = ROOT / "assets" / "ASSETS.sha256"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--stage", default=str(ROOT / "kit_staging"), help="staging folder (git-ignored)"
    )
    ap.add_argument("--with-weights", action="store_true", help="bundle B also carries all weights")
    ap.add_argument("--only", choices=["demo", "full"], help="build one bundle only")
    args = ap.parse_args(argv)
    stage = Path(args.stage)
    try:
        if args.only != "full":
            rows = build_demo(stage)
            out = write_assets_list(rows)
            total = sum(s for _, _, s in rows)
            print(f"bas-demo-kit: {len(rows)} files, {total / 1e6:.1f} MB; hash list -> {out}")
            for h, rel, s in rows:
                print(f"  {h[:12]}  {s:>10}  {rel}")
        if args.only != "demo":
            rows = build_full(stage, args.with_weights)
            total = sum(s for _, _, s in rows)
            print(f"bas-dataset-full: {len(rows)} files, {total / 1e6:.1f} MB")
    except BuildError as exc:
        print(f"BUILD FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
