"""The only tool that edits the hand-made overlay ``data/corrections/<split>.json``.

  python -m training.corrections exclude --split train --frame x012_570.jpg [--force]
                                 [--corrections data/corrections]

Adds ``{"excluded": true, "verified": false, "boxes": []}`` for the frame. Before anything is
written the current file is copied to ``<file>.bak-<sha256 prefix>-<timestamp>``; the new
content goes to a temporary file that is validated with ``build_dataset``'s own checks and only
then replaces the original, so a failed validation leaves the overlay exactly as it was.
Refuses (exit 2) an unknown split, a missing overlay, a file with NUL bytes, a frame that is
not one of the split's sampled frames, and a frame that is already corrected (a non-excluded
entry) unless ``--force``. Excluding an already excluded frame changes nothing. The overlay is
hand work that exists nowhere else: nothing here deletes a backup.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from training.build_dataset import SPLITS, BuildError, validate_overlay_path


class CorrectionError(Exception):
    """The requested change was refused."""


def sha256_prefix(data: bytes, n: int = 12) -> str:
    return hashlib.sha256(data).hexdigest()[:n]


def known_frames(index: dict[str, dict], split: str) -> set[str]:
    return {
        f"{run}_{fid}.jpg"
        for run, info in index.items()
        if info["split"] == split
        for fid in info["frame_ids"]
    }


def exclude_frame(
    overlay: dict, frame: str, known: set[str], *, force: bool = False
) -> tuple[dict, str]:
    """A copy of ``overlay`` with ``frame`` excluded, and what happened:
    ``excluded``, ``excluded_forced`` (a corrected entry replaced) or ``already_excluded``."""
    if frame not in known:
        raise CorrectionError(
            f"unknown frame {frame}: not a sampled frame of split {overlay.get('split')}"
        )
    new = copy.deepcopy(overlay)
    entry = new.setdefault("frames", {}).get(frame)
    if entry is not None and entry.get("excluded"):
        return new, "already_excluded"
    if entry is not None and not force:
        raise CorrectionError(
            f"{frame} is already corrected in the overlay (verified={entry.get('verified')}); "
            "pass --force to replace it with an exclusion"
        )
    new["frames"][frame] = {"excluded": True, "verified": False, "boxes": []}
    return new, "excluded" if entry is None else "excluded_forced"


def backup(path: Path, now: str | None = None) -> Path:
    path = Path(path)
    data = path.read_bytes()
    stamp = now or datetime.now().strftime("%Y%m%dT%H%M%S")
    dest = path.with_name(f"{path.name}.bak-{sha256_prefix(data)}-{stamp}")
    dest.write_bytes(data)
    if dest.read_bytes() != data:
        raise CorrectionError(f"backup {dest} does not match the original; nothing was changed")
    return dest


def _summary(overlay: dict, status: str, bak: str | None) -> dict:
    entries = overlay.get("frames", {})
    excluded = sum(1 for e in entries.values() if e.get("excluded"))
    return {
        "status": status,
        "backup": bak,
        "entries": len(entries),
        "excluded": excluded,
        "corrected": len(entries) - excluded,
    }


def run_exclude(
    corrections_dir: Path,
    split: str,
    frame: str,
    classes: list[str],
    manifest_rows: list[dict],
    index: dict[str, dict],
    *,
    force: bool = False,
    now: str | None = None,
) -> dict:
    """Exclude ``frame`` in ``<corrections_dir>/<split>.json``. Returns
    ``{"status", "backup" (file name or None), "entries", "excluded", "corrected"}``."""
    if split not in SPLITS:
        raise CorrectionError(
            f"unknown split {split!r}; use one of {list(SPLITS)} (val, not valid)"
        )
    path = Path(corrections_dir) / f"{split}.json"
    if not path.is_file():
        raise CorrectionError(f"no overlay for split {split}: {path} does not exist")
    raw = path.read_bytes()
    if b"\x00" in raw:
        raise CorrectionError(f"{path} contains NUL bytes; restore it from a backup first")
    validate_overlay_path(path, split, classes, manifest_rows, index)  # the file as it is now
    overlay = json.loads(raw.decode("utf-8"))
    new, status = exclude_frame(overlay, frame, known_frames(index, split), force=force)
    if status == "already_excluded":
        return _summary(new, status, None)
    bak = backup(path, now)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(json.dumps(new, indent=1).encode("utf-8"))  # LF, like the original
    try:
        validate_overlay_path(tmp, split, classes, manifest_rows, index)
    except BuildError:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)
    return _summary(new, status, bak.name)


def main(argv: list[str] | None = None) -> int:
    from training.autolabel import load_classes, load_index

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("exclude", help="exclude one frame from training")
    ex.add_argument("--split", required=True)
    ex.add_argument("--frame", required=True, help="file name, e.g. x012_570.jpg")
    ex.add_argument("--force", action="store_true")
    ex.add_argument("--corrections", type=Path, default=Path("data/corrections"))
    ex.add_argument("--manifest", type=Path, default=Path("runs/manifest.csv"))
    args = ap.parse_args(argv)

    with args.manifest.open(newline="", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))
    try:
        out = run_exclude(
            args.corrections,
            args.split,
            args.frame,
            load_classes(),
            manifest,
            load_index(),
            force=args.force,
        )
    except (CorrectionError, BuildError) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
