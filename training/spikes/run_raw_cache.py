"""P1.2 checkpoint 1b, item 1: raw candidate cache (IMPLEMENTATION_PLAN.md
Part 10, P1.2 packet). Time-boxed, exploratory, like run_checkpoint1.py --
no tests here (model + I/O, not a pure helper); see
training/spikes/select_v2.py for the tested selection logic that consumes
this cache.

Reruns the identical 12 train runs x 5 frames x 10 phrasings as checkpoint 1
(same seed, same frame list, reused from run_checkpoint1._select_runs /
_sample_frame_indices rather than re-derived -- AGENTS.md rule 1), but
records EVERY candidate box Grounding DINO returns per call (up to
MAX_CANDIDATES, score >= SCORE_FLOOR), not just checkpoint 1's single
"best in band" winner. Checkpoint 1's report.json only kept the post-hoc
winner under the too-wide 0.1%-60% band, which is exactly the number that
needs re-deriving -- hence a fresh model pass instead of reuse.

Output: data/spikes/v2/raw_candidates.jsonl, one JSON line per
(run_id, frame_id, phrase). Resumable: lines already present are skipped,
so an interrupted run loses no completed work.

Run: python -m training.spikes.run_raw_cache
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from training.spikes.detector import GroundingDinoSpike  # noqa: E402
from training.spikes.prompts import PROMPT_CANDIDATES  # noqa: E402
from training.spikes.run_checkpoint1 import (  # noqa: E402
    FRAMES_PER_RUN,
    RUNS_DIR,
    SEED,
    _read_frame,
    _sample_frame_indices,
    _select_runs,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("p1.2-spike-v2-cache")

OUT_DIR = Path("data/spikes/v2")
OUT_PATH = OUT_DIR / "raw_candidates.jsonl"

# Lower than checkpoint 1's 0.25/0.20: we want the losing (smaller, correct)
# container candidates that checkpoint 1's own thresholds may have already
# screened out, so select_v2.py has real candidates to choose among instead
# of only the whole-cardboard-box winner.
BOX_THRESHOLD = 0.20
TEXT_THRESHOLD = 0.20
SCORE_FLOOR = 0.20
MAX_CANDIDATES = 10


def _existing_keys(path: Path) -> set[tuple[str, int, str]]:
    keys: set[tuple[str, int, str]] = set()
    if not path.exists():
        return keys
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            keys.add((rec["run_id"], rec["frame_id"], rec["phrase"]))
    return keys


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_ids = _select_runs(SEED)
    log.info("Selected %d train runs: %s", len(run_ids), run_ids)

    done = _existing_keys(OUT_PATH)
    log.info("%d (run,frame,phrase) triples already cached; resuming", len(done))

    log.info(
        "Loading Grounding DINO tiny (box_threshold=%.2f, text_threshold=%.2f, "
        "score_floor=%.2f, HF_HUB_OFFLINE=%s)...",
        BOX_THRESHOLD,
        TEXT_THRESHOLD,
        SCORE_FLOOR,
        os.environ.get("HF_HUB_OFFLINE"),
    )
    detector = GroundingDinoSpike(box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD)

    latencies: list[float] = []
    total_calls = 0

    with OUT_PATH.open("a") as out:
        for run_index, run_id in enumerate(run_ids):
            video_path = RUNS_DIR / run_id / "video.mp4"
            cap = cv2.VideoCapture(str(video_path))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            indices = _sample_frame_indices(frame_count, FRAMES_PER_RUN, seed=SEED + run_index)

            for frame_id in indices:
                pending = [
                    (cls, phrase)
                    for cls, cls_phrases in PROMPT_CANDIDATES.items()
                    for phrase in cls_phrases
                    if (run_id, frame_id, phrase) not in done
                ]
                if not pending:
                    continue

                frame_bgr = _read_frame(video_path, frame_id)
                if frame_bgr is None:
                    log.warning("Could not read frame %d of %s; skipping", frame_id, run_id)
                    continue
                height, width = frame_bgr.shape[:2]
                image_rgb = np.ascontiguousarray(frame_bgr[..., ::-1])

                for cls, phrase in pending:
                    start = time.perf_counter()
                    result = detector.detect(image_rgb, [phrase])
                    elapsed = time.perf_counter() - start
                    latencies.append(elapsed)
                    total_calls += 1

                    cands = sorted(result.detections, key=lambda d: d.score, reverse=True)
                    cands = [d for d in cands if d.score >= SCORE_FLOOR][:MAX_CANDIDATES]

                    rec = {
                        "run_id": run_id,
                        "frame_id": frame_id,
                        "class": cls,
                        "phrase": phrase,
                        "width": width,
                        "height": height,
                        "box_threshold": BOX_THRESHOLD,
                        "text_threshold": TEXT_THRESHOLD,
                        "score_floor": SCORE_FLOOR,
                        "seconds": elapsed,
                        "candidates": [{"box": list(d.box), "score": d.score} for d in cands],
                    }
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    done.add((run_id, frame_id, phrase))

                    if total_calls % 25 == 0:
                        log.info(
                            "%d calls done (last %s#%d %r: %.2fs, %d candidates)",
                            total_calls,
                            run_id,
                            frame_id,
                            phrase,
                            elapsed,
                            len(cands),
                        )

    if latencies:
        log.info(
            "Done this session: %d calls, mean %.2fs/call, median %.2fs/call, "
            "total %.1f min",
            len(latencies),
            float(np.mean(latencies)),
            float(np.median(latencies)),
            sum(latencies) / 60.0,
        )
    else:
        log.info("Nothing to do; all (run,frame,phrase) triples already cached.")


if __name__ == "__main__":
    main()
