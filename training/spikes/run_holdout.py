"""P1.2 checkpoint 1c: fresh hold-out for selection v3. Train runs NOT among
the 12 used so far, seed 1, 5 runs x 4 frames = 20 frames spread across each
run (5 x 4 rather than 10 x 2 so the static-object consensus has several
frames per run, as in the 60-frame cache). ONE phrasing per class. The v3
rules and thresholds are loaded from data/spikes/v3/frozen_params.json and
are not re-derived or tuned here.

  python -m training.spikes.run_holdout plan     # choose runs/frames, write plan.json
  python -m training.spikes.run_holdout cache    # model; resumable; HF_HUB_OFFLINE=1
  python -m training.spikes.run_holdout report   # apply frozen v3, write sheets
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from training.spikes.run_checkpoint1 import (  # noqa: E402
    RUNS_DIR,
    _read_frame,
    _sample_frame_indices,
)
from training.spikes.run_checkpoint1b import (  # noqa: E402
    ALL_CLASSES,
    RAW_CACHE_PATH,
    _draw_v2_boxes,
    frame_dims,
    frames_by_run,
    index_by_run_frame_class,
    load_raw_cache,
    write_contact_sheets_v2,
)
from training.spikes.run_checkpoint1c import (  # noqa: E402
    FROZEN_PATH,
    _load_frame_cached,
    select_v3_frames,
    winning_phrases,
)
from training.spikes.select_v3 import pick_holdout_runs  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("p1.2-holdout")

OUT_DIR = Path("data/spikes/holdout")
PLAN_PATH = OUT_DIR / "plan.json"
CACHE_PATH = OUT_DIR / "raw_candidates.jsonl"
SEED = 1
N_RUNS, FRAMES_PER_RUN = 5, 4
BOX_THRESHOLD = TEXT_THRESHOLD = SCORE_FLOOR = 0.20
MAX_CANDIDATES = 10

# One phrasing per class, from the 60-frame cache (see run_checkpoint1c's
# phrase tally: the phrasing of the best-scoring candidate agreeing with the
# final box, most frames). red/yellow fixed by the Lead.
PHRASES = {
    "outer_box": "a brown cardboard box.",
    "tray": "a grey book.",
    "start_button": "a white index card.",
    "red_box": "a red container.",
    "yellow_box": "a lime green container.",
}


def make_plan() -> dict:
    with (RUNS_DIR / "manifest.csv").open() as f:
        rows = list(csv.DictReader(f))
    used = set(frames_by_run(load_raw_cache(RAW_CACHE_PATH)))
    chosen = pick_holdout_runs(rows, used, N_RUNS, SEED)
    by_id = {r["run_id"]: r for r in rows}
    plan = {"seed": SEED, "excluded_used_runs": sorted(used), "runs": {}}
    for i, run_id in enumerate(chosen):
        assert by_id[run_id]["split"] == "train"
        n = int(by_id[run_id]["frames"])
        plan["runs"][run_id] = {
            "script_type": by_id[run_id]["script_type"],
            "frame_count": n,
            "frames": _sample_frame_indices(n, FRAMES_PER_RUN, seed=SEED + i),
        }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLAN_PATH.write_text(json.dumps(plan, indent=2))
    return plan


def run_cache() -> None:
    from training.spikes.detector import GroundingDinoSpike

    plan = json.loads(PLAN_PATH.read_text())
    done = set()
    if CACHE_PATH.exists():
        for line in CACHE_PATH.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["run_id"], r["frame_id"], r["phrase"]))
    detector = GroundingDinoSpike(box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD)
    latencies = []
    with CACHE_PATH.open("a") as out:
        for run_id, info in plan["runs"].items():
            for fid in info["frames"]:
                pending = [(c, p) for c, p in PHRASES.items() if (run_id, fid, p) not in done]
                if not pending:
                    continue
                frame = _read_frame(RUNS_DIR / run_id / "video.mp4", fid)
                if frame is None:
                    log.warning("cannot read %s#%d", run_id, fid)
                    continue
                h, w = frame.shape[:2]
                rgb = np.ascontiguousarray(frame[..., ::-1])
                for cls, phrase in pending:
                    t0 = time.perf_counter()
                    res = detector.detect(rgb, [phrase])
                    dt = time.perf_counter() - t0
                    latencies.append(dt)
                    cands = [
                        d
                        for d in sorted(res.detections, key=lambda d: -d.score)
                        if d.score >= SCORE_FLOOR
                    ][:MAX_CANDIDATES]
                    out.write(
                        json.dumps(
                            {
                                "run_id": run_id,
                                "frame_id": fid,
                                "class": cls,
                                "phrase": phrase,
                                "width": w,
                                "height": h,
                                "box_threshold": BOX_THRESHOLD,
                                "text_threshold": TEXT_THRESHOLD,
                                "score_floor": SCORE_FLOOR,
                                "seconds": dt,
                                "candidates": [
                                    {"box": list(d.box), "score": d.score} for d in cands
                                ],
                            }
                        )
                        + "\n"
                    )
                    out.flush()
                    if len(latencies) % 10 == 0:
                        log.info("%d calls, last %.2fs", len(latencies), dt)
    if latencies:
        log.info(
            "Done: %d calls, mean %.2fs, median %.2fs, total %.1f min",
            len(latencies),
            np.mean(latencies),
            np.median(latencies),
            sum(latencies) / 60,
        )


def run_report() -> None:
    params = json.loads(FROZEN_PATH.read_text())
    records = load_raw_cache(CACHE_PATH)
    by_rfc = index_by_run_frame_class(records)
    by_run_frames = frames_by_run(records)
    dims = frame_dims(records)
    load_frame = _load_frame_cached()
    tiles, tally = [], {c: Counter() for c in ALL_CLASSES}
    out = {"params": params, "per_frame": [], "per_run": {}}
    for run_id in sorted(by_run_frames):
        res = select_v3_frames(run_id, by_run_frames[run_id], by_rfc, dims, params, load_frame)
        winning_phrases(res["boxes"], run_id, by_rfc, tally)
        out["per_run"][run_id] = {
            "start_button_white": res["start_button_white"],
            "start_button_measure": res["start_button_measure"],
        }
        for fid in by_run_frames[run_id]:
            boxes = res["boxes"][fid]
            out["per_frame"].append(
                {
                    "run_id": run_id,
                    "frame_id": fid,
                    "boxes": boxes,
                    "rejected": res["rejected"][fid],
                }
            )
            tiles.append((f"{run_id}#{fid}", _draw_v2_boxes(load_frame(run_id, fid), boxes)))
    out["missing"] = {
        c: sum(1 for e in out["per_frame"] if e["boxes"][c] is None) for c in ALL_CLASSES
    }
    out["n_frames"] = len(out["per_frame"])
    out["mean_seconds_per_call"] = float(np.mean([r["seconds"] for r in records]))
    (OUT_DIR / "report_holdout.json").write_text(json.dumps(out, indent=2, default=str))
    sheets = write_contact_sheets_v2(tiles, OUT_DIR, per_sheet=10, prefix="contact_sheet_holdout_")
    bad = OUT_DIR / "bad_boxes.csv"
    if not bad.exists():
        bad.write_text("run_id,frame_id,class,reason,visibility,note\n")
    log.info("sheets %s; missing %s; n=%d", sheets, out["missing"], out["n_frames"])
    log.info("mean seconds/call %.2f", out["mean_seconds_per_call"])


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "plan":
        p = make_plan()
        log.info(json.dumps(p, indent=1))
    elif mode == "cache":
        run_cache()
    elif mode == "report":
        run_report()
    else:
        raise SystemExit(__doc__)
    cv2.destroyAllWindows()
