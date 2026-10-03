"""P1.7.2: build every perception cache at the target fps, time it, and write
``reports/cache_build.json``.

  python -m training.cache_report [--fps 10] [--detector NAME] [--out-root data/cache]
        [--report reports/cache_build.json] [--split all] [--force] [--log PATH]

This wraps ``perception.cache.build_runs`` (perception reads no clock, so the timing lives here)
and then summarises per split: runs, frames, seconds, and for ``val`` and ``test`` descriptive
per-class detection statistics: the number and share of frames with a detection of the class at
or above ``detector_conf_floor`` and the quantiles of the best such confidence per frame. Labels
are never read, nothing is compared to ground truth and nothing is tuned on these numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from contracts import DETECTOR_MIN_CONF, PerceptionConfig, PerceptionFrame
from perception import cache as C

logger = logging.getLogger(__name__)


def _quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    v = sorted(values)
    q = statistics.quantiles(v, n=4, method="inclusive") if len(v) > 1 else [v[0]] * 3
    return {"min": v[0], "q25": q[0], "median": q[1], "q75": q[2], "max": v[-1]}


def class_stats(frames: list[PerceptionFrame], classes: list[str], floor: float) -> dict[str, Any]:
    best: dict[str, list[float]] = {c: [] for c in classes}
    n_det = dict.fromkeys(classes, 0)
    for fr in frames:
        top: dict[str, float] = {}
        for d in fr.detections:
            if d.conf >= floor and d.label in best:
                n_det[d.label] += 1
                top[d.label] = max(top.get(d.label, 0.0), d.conf)
        for c, v in top.items():
            best[c].append(v)
    n = len(frames)
    return {
        "n_frames": n,
        "floor": floor,
        "per_class": {
            c: {
                "frames_with_detection": len(best[c]),
                "fraction_of_frames": (len(best[c]) / n) if n else None,
                "n_detections_at_floor": n_det[c],
                "best_conf_quantiles": _quantiles(best[c]),
            }
            for c in classes
        },
    }


def split_summary(results: list[dict], splits: dict[str, str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in results:
        if r["status"] == "failed":
            continue
        s = out.setdefault(splits[r["run_id"]], {"runs": 0, "frames": 0, "seconds": 0.0})
        s["runs"] += 1
        s["frames"] += r["frames"]
        s["seconds"] += r.get("seconds", 0.0)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fps", type=float, default=C.DEFAULT_TARGET_FPS)
    ap.add_argument("--detector", default=None)
    ap.add_argument("--split", choices=C.SPLITS, default="all")
    ap.add_argument("--run", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out-root", type=Path, default=C.DEFAULT_CACHE_ROOT)
    ap.add_argument("--runs-dir", type=Path, default=C.DEFAULT_RUNS_DIR)
    ap.add_argument("--report", type=Path, default=Path("reports/cache_build.json"))
    ap.add_argument("--log", type=Path, default=None)
    args = ap.parse_args(argv)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        handlers=handlers,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from perception.pipeline import load_pipeline
    from training.autolabel import load_classes

    manifest = args.runs_dir / "manifest.csv"
    runs = C.select_runs(manifest, args.runs_dir, split=args.split, run=args.run)
    with manifest.open(newline="", encoding="utf-8") as f:
        splits = {r["run_id"]: r["split"] for r in csv.DictReader(f)}
    pipeline = load_pipeline(detector=args.detector)
    exp_id = C.experiment_id()
    logger.info("model_stamp %s; %d runs at %s fps", pipeline.model_stamp, len(runs), args.fps)

    last = [time.perf_counter()]

    def stamp_time(r: dict) -> None:
        now = time.perf_counter()
        r["seconds"] = round(now - last[0], 3)
        last[0] = now

    t_all = time.perf_counter()
    results = C.build_runs(
        runs,
        pipeline,
        fps=args.fps,
        experiment_id=exp_id,
        out_root=args.out_root,
        force=args.force,
        on_run=stamp_time,
    )
    total_s = round(time.perf_counter() - t_all, 3)

    classes = load_classes(Path("config/experiment.json"))
    floor = float(PerceptionConfig().detector_conf_floor)
    stats = {}
    for sp in ("val", "test"):
        frames: list[PerceptionFrame] = []
        for r in results:
            if r["status"] != "failed" and splits[r["run_id"]] == sp:
                frames += C.read_cache(r["path"], expected_stamp=pipeline.model_stamp)[1]
        stats[sp] = class_stats(frames, classes, floor)
    keys = ("status", "frames", "skipped_frames", "every", "seconds", "error")
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_stamp": pipeline.model_stamp,
        "fps": args.fps,
        "config": {
            "experiment_id": exp_id,
            "detector": pipeline.detector_name,
            "detector_min_conf": DETECTOR_MIN_CONF,
            "detector_conf_floor_for_statistics": floor,
            "decimation": "every round(source_fps / fps)-th frame, ids and t of the recording",
            "split_selected": args.split,
            "out_root": str(args.out_root),
        },
        "total_wall_seconds": total_s,
        "per_split": split_summary(results, splits),
        "per_run": [
            {"run_id": r["run_id"], "split": splits[r["run_id"]]}
            | {k: r[k] for k in keys if k in r}
            for r in results
        ],
        "class_statistics": {
            "note": "descriptive only: no labels read, nothing compared or tuned on val or test",
            "val": stats["val"],
            "test": stats["test"],
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1), encoding="utf-8")
    failed = [r["run_id"] for r in results if r["status"] == "failed"]
    logger.info("report written to %s; failed runs: %s", args.report, failed or "none")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
