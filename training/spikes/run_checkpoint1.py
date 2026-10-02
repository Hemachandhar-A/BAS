"""P1.2 checkpoint 1: Grounding DINO tiny feasibility spike
(IMPLEMENTATION_PLAN.md Part 10, P1.2 packet). Time-boxed, exploratory:
samples frames from `train`-split runs only, runs Grounding DINO tiny
zero-shot detection with the candidate prompts in
`training/spikes/prompts.py`, applies the plan's post-processing
(training/spikes/postprocess.py), and writes:

  - data/spikes/report.json        -- every number in this module's docstring
  - data/spikes/contact_sheet_NN.jpg -- 12 frames per sheet, boxes drawn

Run: python -m training.spikes.run_checkpoint1
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path

import cv2
import numpy as np

from training.spikes.detector import GroundingDinoSpike
from training.spikes.postprocess import (
    RawDetection,
    box_area_frac,
    iou,
    mean_hue_in_box,
    median_box,
    select_best_in_band,
)
from training.spikes.prompts import PROMPT_CANDIDATES, all_phrases

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("p1.2-spike")

SEED = 0
FRAMES_PER_RUN = 5
RUNS_DIR = Path("runs")
OUT_DIR = Path("data/spikes")

# A broad sanity band, not the real per-class band (that is checkpoint 3's
# recommendation, made from the area fractions this run measures). Only
# used here to drop degenerate boxes (near-zero or near-whole-frame).
SANITY_AREA_BAND: dict[str, tuple[float, float]] = {
    cls: (0.001, 0.60) for cls in PROMPT_CANDIDATES
}

# Static objects per essential-features.md section 14 Stage 3: "the outer
# box, tray and START button never move" within a run.
STATIC_CLASSES = ("outer_box", "tray", "start_button")


def _select_runs(seed: int) -> list[str]:
    """~12 train runs, seeded, spread across every script_type present in
    train (checkpoint 1 requires including the idle and skip clips)."""
    import csv

    with (RUNS_DIR / "manifest.csv").open() as f:
        rows = list(csv.DictReader(f))
    train = [r for r in rows if r["split"] == "train"]
    by_type: dict[str, list[str]] = {}
    for r in train:
        by_type.setdefault(r["script_type"], []).append(r["run_id"])

    rng = random.Random(seed)
    selected: list[str] = []
    # every idle and repeat run in train (there is exactly one of each) --
    # too few to sample from.
    selected += by_type.get("idle", [])
    selected += by_type.get("repeat", [])
    # a spread sample of skip and swap (both present, both small).
    selected += rng.sample(by_type.get("skip", []), k=min(3, len(by_type.get("skip", []))))
    selected += rng.sample(by_type.get("swap", []), k=min(3, len(by_type.get("swap", []))))
    # fill the rest from correct up to 12 total.
    remaining = max(0, 12 - len(selected))
    correct_runs = by_type.get("correct", [])
    selected += rng.sample(correct_runs, k=min(remaining, len(correct_runs)))
    selected.sort()
    return selected


def _sample_frame_indices(frame_count: int, n: int, seed: int) -> list[int]:
    """n indices spread across the whole clip (not just the first second),
    seeded jitter around evenly spaced anchors so different runs don't all
    sample the identical relative position."""
    rng = random.Random(seed)
    anchors = np.linspace(0, frame_count - 1, n + 2)[1:-1]
    jitter_span = max(1, frame_count // (n * 4))
    indices = []
    for a in anchors:
        jitter = rng.randint(-jitter_span, jitter_span)
        idx = int(np.clip(round(a) + jitter, 0, frame_count - 1))
        indices.append(idx)
    return indices


def _read_frame(video_path: Path, frame_index: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def _model_cache_info() -> dict:
    """Locate the cached Grounding DINO tiny weights and hash the file(s)
    actually present (safetensors is preferred by transformers when both
    formats exist in the repo; only the present file(s) are downloaded)."""
    from huggingface_hub import scan_cache_dir

    info: dict = {"repo_id": "IDEA-Research/grounding-dino-tiny", "files": []}
    cache = scan_cache_dir()
    for repo in cache.repos:
        if repo.repo_id != "IDEA-Research/grounding-dino-tiny":
            continue
        for revision in repo.revisions:
            for file in revision.files:
                path = Path(file.file_path)
                if not path.is_file():
                    continue
                sha256 = hashlib.sha256()
                with path.open("rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        sha256.update(chunk)
                info["files"].append(
                    {
                        "name": path.name,
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256.hexdigest(),
                    }
                )
            info["total_size_bytes"] = repo.size_on_disk
    return info


def _draw_boxes(frame_bgr: np.ndarray, best: dict[str, RawDetection]) -> np.ndarray:
    canvas = frame_bgr.copy()
    colors = {
        "outer_box": (255, 128, 0),
        "tray": (128, 128, 128),
        "red_box": (0, 0, 255),
        "yellow_box": (0, 200, 200),
        "start_button": (255, 0, 255),
    }
    for cls, det in best.items():
        x1, y1, x2, y2 = (int(round(v)) for v in det.box)
        color = colors.get(cls, (0, 255, 0))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{cls} {det.score:.2f}"
        cv2.putText(canvas, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return canvas


def _write_contact_sheets(
    tiles: list[tuple[str, np.ndarray]], out_dir: Path, per_sheet: int = 12
) -> list[str]:
    paths: list[str] = []
    for sheet_start in range(0, len(tiles), per_sheet):
        chunk = tiles[sheet_start : sheet_start + per_sheet]
        thumb_w, thumb_h = 320, 180
        cols = 4
        rows = (len(chunk) + cols - 1) // cols
        sheet = np.full((rows * thumb_h, cols * thumb_w, 3), 40, dtype=np.uint8)
        for i, (caption, img) in enumerate(chunk):
            thumb = cv2.resize(img, (thumb_w, thumb_h))
            cv2.putText(
                thumb, caption, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
            )
            r, c = divmod(i, cols)
            sheet[r * thumb_h : (r + 1) * thumb_h, c * thumb_w : (c + 1) * thumb_w] = thumb
        idx = sheet_start // per_sheet + 1
        path = out_dir / f"contact_sheet_{idx:02d}.jpg"
        cv2.imwrite(str(path), sheet)
        paths.append(str(path))
    return paths


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_ids = _select_runs(SEED)
    log.info("Selected %d train runs: %s", len(run_ids), run_ids)

    phrases = all_phrases()
    log.info("Prompt candidates tried:")
    for cls, cands in PROMPT_CANDIDATES.items():
        log.info("  %s: %s", cls, cands)

    log.info("Loading Grounding DINO tiny (downloads on first run)...")
    detector = GroundingDinoSpike()

    frame_records: list[dict] = []
    tiles: list[tuple[str, np.ndarray]] = []
    latencies: list[float] = []
    # per-phrase raw-detection counts (any box returned, before the area
    # filter), for checkpoint 3's "which wording wins".
    phrase_raw_detection_count: dict[str, int] = dict.fromkeys(phrases, 0)

    for run_index, run_id in enumerate(run_ids):
        video_path = RUNS_DIR / run_id / "video.mp4"
        cap = cv2.VideoCapture(str(video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        indices = _sample_frame_indices(frame_count, FRAMES_PER_RUN, seed=SEED + run_index)

        for frame_id in indices:
            frame_bgr = _read_frame(video_path, frame_id)
            if frame_bgr is None:
                log.warning("Could not read frame %d of %s; skipping", frame_id, run_id)
                continue
            height, width = frame_bgr.shape[:2]
            image_rgb = np.ascontiguousarray(frame_bgr[..., ::-1])

            # One call per CLASS, per PHRASING -- not one combined call for
            # all 5 classes' 10 phrases together. That was tried first and
            # dropped: Grounding DINO's returned text spans blend across
            # phrase boundaries in a long multi-class query (e.g. a query
            # of "a cardboard box. a red box." on one frame came back
            # labelled "cardboard box red box" / "a a red box" -- fragments
            # that do not cleanly map to either submitted phrase). Querying
            # one phrase at a time removes the ambiguity, at the cost of
            # len(phrases) forward passes per frame instead of one.
            per_class: dict = {}
            frame_seconds = 0.0
            raw_count = 0
            for cls, cls_phrases in PROMPT_CANDIDATES.items():
                cls_raw: list = []
                for phrase in cls_phrases:
                    result = detector.detect(image_rgb, [phrase])
                    latencies.append(result.seconds)
                    frame_seconds += result.seconds
                    raw_count += len(result.detections)
                    if result.detections:
                        phrase_raw_detection_count[phrase] += 1
                    cls_raw.extend(result.detections)
                det = select_best_in_band(cls_raw, SANITY_AREA_BAND.get(cls), width, height)
                if det is not None:
                    entry = {
                        "phrase": det.phrase,
                        "score": det.score,
                        "box": det.box,
                        "area_frac": box_area_frac(det.box, width, height),
                    }
                    if cls in ("red_box", "yellow_box"):
                        entry["mean_hue"] = mean_hue_in_box(frame_bgr, det.box)
                    per_class[cls] = entry

            frame_records.append(
                {
                    "run_id": run_id,
                    "frame_id": frame_id,
                    "width": width,
                    "height": height,
                    "seconds": frame_seconds,
                    "raw_detection_count": raw_count,
                    "best_per_class": per_class,
                }
            )
            caption = f"{run_id}#{frame_id} " + ",".join(sorted(per_class))
            best_boxes = {
                cls: RawDetection(
                    phrase=entry["phrase"], score=entry["score"], box=entry["box"]
                )
                for cls, entry in per_class.items()
            }
            tiles.append((caption, _draw_boxes(frame_bgr, best_boxes)))
            log.info(
                "%s frame %d: found %s (%.1f ms total, %d calls)",
                run_id,
                frame_id,
                sorted(per_class),
                frame_seconds * 1000,
                len(phrases),
            )

    # --- aggregate metrics -------------------------------------------------
    n_frames = len(frame_records)
    classes = list(PROMPT_CANDIDATES)

    found_rate = {
        cls: sum(1 for r in frame_records if cls in r["best_per_class"]) / n_frames
        for cls in classes
    }
    area_fracs = {
        cls: [
            r["best_per_class"][cls]["area_frac"]
            for r in frame_records
            if cls in r["best_per_class"]
        ]
        for cls in classes
    }
    hue_values = {
        cls: [
            r["best_per_class"][cls]["mean_hue"]
            for r in frame_records
            if cls in r["best_per_class"] and r["best_per_class"][cls].get("mean_hue") is not None
        ]
        for cls in ("red_box", "yellow_box")
    }

    # red/yellow confusion: both found in the same frame with high IoU
    # between their boxes (the two prompts landed on the same physical
    # object rather than two distinct containers).
    red_yellow_confusions = []
    for r in frame_records:
        bpc = r["best_per_class"]
        if "red_box" in bpc and "yellow_box" in bpc:
            overlap = iou(tuple(bpc["red_box"]["box"]), tuple(bpc["yellow_box"]["box"]))
            if overlap > 0.5:
                red_yellow_confusions.append(
                    {"run_id": r["run_id"], "frame_id": r["frame_id"], "iou": overlap}
                )

    # static-object box stability: per-run IoU of each frame's box against
    # that run's own median box.
    stability: dict[str, dict] = {}
    for cls in STATIC_CLASSES:
        per_run_ious: dict[str, list[float]] = {}
        for run_id in run_ids:
            boxes = [
                tuple(r["best_per_class"][cls]["box"])
                for r in frame_records
                if r["run_id"] == run_id and cls in r["best_per_class"]
            ]
            med = median_box(boxes)
            if med is None:
                continue
            per_run_ious[run_id] = [iou(b, med) for b in boxes]
        stability[cls] = per_run_ious

    latency_stats = {
        "per_call": {
            "mean_s": float(np.mean(latencies)) if latencies else None,
            "median_s": float(np.median(latencies)) if latencies else None,
            "min_s": float(np.min(latencies)) if latencies else None,
            "max_s": float(np.max(latencies)) if latencies else None,
            "n": len(latencies),
        },
        "per_frame_all_10_phrasings": {
            "mean_s": (
                float(np.mean([r["seconds"] for r in frame_records])) if frame_records else None
            ),
            "median_s": (
                float(np.median([r["seconds"] for r in frame_records])) if frame_records else None
            ),
            "n": n_frames,
        },
    }

    phrase_raw_detection_rate = {
        phrase: count / n_frames if n_frames else None
        for phrase, count in phrase_raw_detection_count.items()
    }

    log.info("Hashing cached model files for the sha256/size record...")
    model_info = _model_cache_info()

    contact_sheets = _write_contact_sheets(tiles, OUT_DIR)

    report = {
        "seed": SEED,
        "run_ids": run_ids,
        "n_frames": n_frames,
        "prompts_tried": PROMPT_CANDIDATES,
        "phrase_raw_detection_rate": phrase_raw_detection_rate,
        "query_methodology_note": (
            "One Grounding DINO forward call per (class, phrasing), not one "
            "combined call for all 10 phrases: a first attempt submitted "
            "all 5 classes' phrasings in a single query and the model's "
            "returned text spans blended across phrase boundaries (e.g. "
            "'a cardboard box. a red box.' came back labelled 'cardboard "
            "box red box' / 'a a red box' on a test frame, neither of "
            "which cleanly matches either submitted phrase). Querying one "
            "phrase at a time removes that ambiguity at the cost of "
            "len(phrases) forward passes per frame instead of one; "
            "training/autolabel.py (Stage 3, real pipeline) should use "
            "either the same one-call-per-class approach or verify "
            "text_labels parsing more carefully if it wants one call."
        ),
        "sanity_area_band": SANITY_AREA_BAND,
        "found_rate": found_rate,
        "area_frac_samples": area_fracs,
        "hue_samples": hue_values,
        "red_yellow_confusions": red_yellow_confusions,
        "static_box_stability_iou": stability,
        "latency_stats": latency_stats,
        "model": model_info,
        "contact_sheets": contact_sheets,
    }
    report_path = OUT_DIR / "report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    log.info("Wrote %s", report_path)
    log.info("Wrote %d contact sheet(s): %s", len(contact_sheets), contact_sheets)


if __name__ == "__main__":
    main()
