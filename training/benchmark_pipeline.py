"""P1.7.1: end-to-end throughput of the real perception pipeline on this machine.

  python -m training.benchmark_pipeline [--detector yolo11n] [--clips 3] [--warmup 10]
        [--frames 200] [--out reports/benchmark_pipeline.json]

For each of the first ``--clips`` validation runs, frames are decoded one after another
(``open_source``) and passed through ``PerceptionPipeline.process`` (detector + hands),
sequentially; the time of each frame is decode plus process, measured with ``time.perf_counter``.
The first ``--warmup`` frames of each clip are dropped. Only validation clips are read; nothing
is evaluated against labels. The result is the median and p95 milliseconds per frame and the fps
they imply (``1000 / median``, ``1000 / p95``); it does not include tracking, streaming or speech.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from training.benchmark_valid import latency_stats

logger = logging.getLogger(__name__)


def summarize(seconds: list[float]) -> dict[str, float]:
    s = latency_stats(seconds)
    s["implied_fps"] = 1000.0 / s["median_ms"]
    s["implied_fps_p95"] = 1000.0 / s["p95_ms"]
    return s


def pick_clips(manifest_csv: Path, n: int) -> list[str]:
    with Path(manifest_csv).open(newline="", encoding="utf-8") as f:
        val = [r["run_id"] for r in csv.DictReader(f) if r["split"] == "val"]
    if len(val) < n:
        raise ValueError(f"only {len(val)} validation runs in {manifest_csv}, {n} asked for")
    return val[:n]


def time_clip(pipeline: Any, video: Path, warmup: int, frames: int) -> list[float]:
    from perception.camera import open_source

    src = open_source(str(video))
    pipeline.reset()
    out: list[float] = []
    try:
        for i in range(warmup + frames):
            t0 = time.perf_counter()
            fr = src.read()
            if fr is None:
                break
            pipeline.process(fr)
            if i >= warmup:
                out.append(time.perf_counter() - t0)
    finally:
        src.close()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--detector", default=None)
    ap.add_argument("--clips", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path, default=Path("reports/benchmark_pipeline.json"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    from perception.pipeline import load_pipeline

    pipeline = load_pipeline(detector=args.detector)
    per_clip: dict[str, Any] = {}
    allt: list[float] = []
    for run_id in pick_clips(args.runs_dir / "manifest.csv", args.clips):
        t = time_clip(pipeline, args.runs_dir / run_id / "video.mp4", args.warmup, args.frames)
        per_clip[run_id] = summarize(t)
        allt += t
        logger.info("%s: %s", run_id, per_clip[run_id])
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_stamp": pipeline.model_stamp,
        "detector": pipeline.detector_name,
        "what": "sequential decode + detector + hands per frame, no tracking/streaming/speech",
        "warmup_frames_dropped_per_clip": args.warmup,
        "frames_per_clip_requested": args.frames,
        "machine": {"processor": platform.processor(), "platform": platform.platform()},
        "per_clip": per_clip,
        "all_clips": summarize(allt),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report["all_clips"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
