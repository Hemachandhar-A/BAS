"""F14 stage 2: sample and de-duplicate frames from every run (train, val and
test) into ``data/frames/<split>/<run_id>_<frame_id>.jpg`` at native
resolution. Deterministic: no randomness, no clock.

Per run: candidates are every ``stride_for(src_fps)``-th frame (about 2 per
second); a candidate is kept only if the mean absolute difference of its 32x32
grayscale thumbnail from the LAST KEPT frame exceeds ``DIFF_THRESHOLD`` (6/255)
or at least ``MAX_GAP_S`` (1 s) has passed since it. If more than ``CAP``
frames survive, an even subsample (first and last kept) is taken, so a long
run is covered end to end rather than cut off.

Run: python -m training.sample_frames [--runs-dir runs] [--out data/frames]
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np

DIFF_THRESHOLD = 6 / 255
MAX_GAP_S = 1.0
CAP = 120
THUMB = 32
JPEG_QUALITY = 95


def stride_for(src_fps: float) -> int:
    """Candidate stride: about two candidates per second."""
    return max(1, round(src_fps / 2))


def thumbnail(frame_bgr: np.ndarray) -> np.ndarray:
    """32x32 float32 grayscale thumbnail (frame is BGR; area interpolation)."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (THUMB, THUMB), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32)


def thumb_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute difference of two thumbnails on a 0-1 scale (value / 255)."""
    return float(np.abs(a - b).mean()) / 255.0


def select_frame_ids(
    candidates: Iterable[tuple[int, np.ndarray]], src_fps: float, cap: int = CAP
) -> list[int]:
    """Frame ids to keep from ``(frame_id, thumbnail)`` candidates in frame
    order. The first candidate is always kept."""
    kept: list[int] = []
    ref: np.ndarray | None = None
    ref_id = 0
    for frame_id, thumb in candidates:
        if ref is None:
            keep = True
        else:
            keep = (
                thumb_diff(thumb, ref) > DIFF_THRESHOLD
                or (frame_id - ref_id) / src_fps >= MAX_GAP_S
            )
        if keep:
            kept.append(frame_id)
            ref, ref_id = thumb, frame_id
    if len(kept) > cap:
        idx = np.unique(np.round(np.linspace(0, len(kept) - 1, cap)).astype(int))
        kept = [kept[i] for i in idx]
    return kept


def sample_run(run_id: str, video_path: Path, out_dir: Path, src_fps: float) -> list[int]:
    """Sample one run and write its JPEGs into ``out_dir``. Returns the kept
    frame ids. Frames are decoded sequentially; frame ids are 0-based decode
    order. Stale ``<run_id>_*.jpg`` files from an earlier pass are removed so a
    re-run leaves exactly the current selection."""
    stride = stride_for(src_fps)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    candidates: list[tuple[int, np.ndarray]] = []
    frames: dict[int, np.ndarray] = {}
    try:
        frame_id = 0
        while True:
            if frame_id % stride:
                if not cap.grab():
                    break
            else:
                ok, frame = cap.read()
                if not ok:
                    break
                candidates.append((frame_id, thumbnail(frame)))
                frames[frame_id] = frame
            frame_id += 1
    finally:
        cap.release()
    kept = select_frame_ids(candidates, src_fps)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob(f"{run_id}_*.jpg"):
        stale.unlink()
    for fid in kept:
        path = out_dir / f"{run_id}_{fid}.jpg"
        cv2.imwrite(str(path), frames[fid], [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return kept


def read_manifest(runs_dir: Path) -> list[dict]:
    with (runs_dir / "manifest.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path, default=Path("data/frames"))
    args = ap.parse_args(argv)

    index: dict[str, dict] = {}
    for row in read_manifest(args.runs_dir):
        run_id, split = row["run_id"], row["split"]
        ids = sample_run(
            run_id, args.runs_dir / run_id / "video.mp4", args.out / split, float(row["fps"])
        )
        index[run_id] = {
            "split": split,
            "fps": float(row["fps"]),
            "n_source_frames": int(row["frames"]),
            "frame_ids": ids,
        }
        print(f"{run_id} {split}: {len(ids)} frames")
    (args.out / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    per_split: dict[str, int] = {}
    for info in index.values():
        per_split[info["split"]] = per_split.get(info["split"], 0) + len(info["frame_ids"])
    print("frames per split:", per_split, "total", sum(per_split.values()))


if __name__ == "__main__":
    main()
