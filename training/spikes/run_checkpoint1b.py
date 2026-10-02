"""P1.2 checkpoint 1b, items 2-4 (IMPLEMENTATION_PLAN.md Part 10, P1.2
packet): consumes data/spikes/v2/raw_candidates.jsonl (written by
training/spikes/run_raw_cache.py) and the pure selection helpers in
training/spikes/select_v2.py to re-derive per-class area bands, pick final
boxes with the v2 rule, compare against the v1 (checkpoint-1) rule on the
SAME cached candidates, and write the v2 contact sheets and
data/spikes/v2/bad_boxes.csv. No model here -- model + I/O is
run_raw_cache.py; tested pure logic is select_v2.py. Time-boxed,
exploratory, like run_checkpoint1.py (no tests on this orchestration file
itself, per that file's own stated scope).

Run: python -m training.spikes.run_checkpoint1b
"""

from __future__ import annotations

import csv
import json
import logging
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from training.spikes.postprocess import RawDetection, box_area_frac, iou, median_box
from training.spikes.postprocess import select_best_in_band as select_best_in_band_v1
from training.spikes.run_checkpoint1 import RUNS_DIR, _read_frame
from training.spikes.select_v2 import (
    central_box,
    mean_hue_sat_in_box,
    select_container,
    single_clean_candidate,
    static_consensus,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("p1.2-checkpoint1b")

RAW_CACHE_PATH = Path("data/spikes/v2/raw_candidates.jsonl")
CHECKPOINT1_REPORT_PATH = Path("data/spikes/report.json")
OUT_DIR = Path("data/spikes/v2")

STATIC_CLASSES = ("outer_box", "tray", "start_button")
CONTAINER_CLASSES = ("red_box", "yellow_box")
ALL_CLASSES = (*STATIC_CLASSES, *CONTAINER_CLASSES)

V1_SANITY_BAND = (0.001, 0.60)  # checkpoint 1's own (too-wide) band, for the side-by-side table
GAP_MIN = 0.03  # minimum gap size (area-fraction units) to call a split "clear"
SKIN_SAT_THRESHOLD = 100.0  # heuristic only -- see ISSUES.md caveat


# --- loading --------------------------------------------------------------


def load_raw_cache(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def index_by_run_frame_class(records: list[dict]) -> dict[tuple[str, int, str], list[RawDetection]]:
    """Merge the two phrasings per class into one candidate list per
    (run_id, frame_id, class); ``RawDetection.phrase`` is set to the
    SUBMITTED phrase (not the model's echoed span) so "which phrasing wins"
    can be tallied directly."""
    out: dict[tuple[str, int, str], list[RawDetection]] = {}
    for rec in records:
        key = (rec["run_id"], rec["frame_id"], rec["class"])
        dets = [
            RawDetection(phrase=rec["phrase"], score=c["score"], box=tuple(c["box"]))
            for c in rec["candidates"]
        ]
        out.setdefault(key, []).extend(dets)
    return out


def frames_by_run(records: list[dict]) -> dict[str, list[int]]:
    by_run: dict[str, set[int]] = {}
    for rec in records:
        by_run.setdefault(rec["run_id"], set()).add(rec["frame_id"])
    return {run_id: sorted(frames) for run_id, frames in by_run.items()}


def frame_dims(records: list[dict]) -> dict[tuple[str, int], tuple[int, int]]:
    return {(r["run_id"], r["frame_id"]): (r["width"], r["height"]) for r in records}


# --- area bands (item 2a) --------------------------------------------------


def pick_area_band(fracs: list[float], anchor: float) -> tuple[tuple[float, float], dict]:
    """Derive an area-fraction band from the raw cache's full candidate
    population, anchored on ``anchor`` -- checkpoint 1's own minimum
    observed area fraction for this class (report.json). That minimum
    reliably falls inside the class's TRUE cluster for all five classes
    (confirmed against report.json before trusting this): for red_box /
    yellow_box the true cluster is the SMALL one (the real container; the
    large one is the whole-cardboard-box collision), but for outer_box
    itself the true cluster is the LARGE one -- a plain "smallest gap below
    wins" rule gets outer_box backwards, which is why this anchors on a
    real measurement instead of assuming small-is-correct.

    Splits the sorted population into clusters wherever a consecutive gap
    is at least GAP_MIN, then returns the cluster containing (or nearest
    to) the anchor, widened by a small margin."""
    xs = sorted(fracs)
    if not xs:
        return (0.001, 0.60), {"method": "fallback_no_data", "anchor": anchor}

    clusters: list[list[float]] = [[xs[0]]]
    for x in xs[1:]:
        if x - clusters[-1][-1] >= GAP_MIN:
            clusters.append([x])
        else:
            clusters[-1].append(x)

    def distance_to(cluster: list[float]) -> float:
        if cluster[0] <= anchor <= cluster[-1]:
            return 0.0
        return min(abs(anchor - cluster[0]), abs(anchor - cluster[-1]))

    target = min(clusters, key=distance_to)
    lo = max(0.001, target[0] * 0.8)
    hi = min(0.95, target[-1] * 1.2)
    return (lo, hi), {
        "method": "anchored_cluster",
        "anchor": anchor,
        "cluster_min": target[0],
        "cluster_max": target[-1],
        "cluster_n": len(target),
        "n_clusters_total": len(clusters),
        "n_total_candidates": len(xs),
    }


def derive_area_bands(
    by_rfc: dict[tuple[str, int, str], list[RawDetection]],
    dims: dict[tuple[str, int], tuple[int, int]],
    checkpoint1_report: dict,
) -> tuple[dict[str, tuple[float, float]], dict[str, dict]]:
    bands: dict[str, tuple[float, float]] = {}
    details: dict[str, dict] = {}
    for cls in ALL_CLASSES:
        fracs = []
        for (run_id, frame_id, c), dets in by_rfc.items():
            if c != cls:
                continue
            width, height = dims[(run_id, frame_id)]
            fracs.extend(box_area_frac(d.box, width, height) for d in dets)
        anchor = min(checkpoint1_report["area_frac_samples"][cls])
        band, detail = pick_area_band(fracs, anchor)
        bands[cls] = band
        details[cls] = detail
    return bands, details


# --- v2 selection per run ---------------------------------------------------


def run_v2_selection(
    run_id: str,
    frame_ids: list[int],
    by_rfc: dict[tuple[str, int, str], list[RawDetection]],
    dims: dict[tuple[str, int], tuple[int, int]],
    bands: dict[str, tuple[float, float]],
) -> dict:
    width, height = dims[(run_id, frame_ids[0])]

    static_results = {}
    for cls in STATIC_CLASSES:
        candidates_by_frame = {fid: by_rfc.get((run_id, fid, cls), []) for fid in frame_ids}
        static_results[cls] = static_consensus(candidates_by_frame, bands[cls], width, height)

    exclude_boxes = {cls: static_results[cls].consensus_box for cls in STATIC_CLASSES}

    container_results: dict[str, dict[int, object]] = {cls: {} for cls in CONTAINER_CLASSES}
    for cls in CONTAINER_CLASSES:
        for fid in frame_ids:
            dets = by_rfc.get((run_id, fid, cls), [])
            container_results[cls][fid] = select_container(
                dets, bands[cls], width, height, exclude_boxes=exclude_boxes
            )

    return {
        "static": static_results,
        "container": container_results,
        "exclude_boxes": exclude_boxes,
    }


# --- v1 selection (for the side-by-side table) ------------------------------


def run_v1_selection(
    run_id: str,
    frame_ids: list[int],
    by_rfc: dict[tuple[str, int, str], list[RawDetection]],
    dims: dict[tuple[str, int], tuple[int, int]],
) -> dict[str, dict[int, RawDetection | None]]:
    width, height = dims[(run_id, frame_ids[0])]
    out: dict[str, dict[int, RawDetection | None]] = {cls: {} for cls in ALL_CLASSES}
    for cls in ALL_CLASSES:
        for fid in frame_ids:
            dets = by_rfc.get((run_id, fid, cls), [])
            out[cls][fid] = select_best_in_band_v1(dets, V1_SANITY_BAND, width, height)
    return out


# --- proxy metrics (item 3) -------------------------------------------------


def proxy_metrics_v1(
    run_ids: list[str],
    by_run_frames: dict[str, list[int]],
    v1_by_run: dict[str, dict[str, dict[int, RawDetection | None]]],
) -> dict:
    red_yellow_overlap = 0
    container_outer_overlap = 0
    start_button_overlap = 0
    n_frames = 0
    for run_id in run_ids:
        v1 = v1_by_run[run_id]
        for fid in by_run_frames[run_id]:
            n_frames += 1
            red, yellow = v1["red_box"][fid], v1["yellow_box"][fid]
            if red is not None and yellow is not None and iou(red.box, yellow.box) > 0.5:
                red_yellow_overlap += 1
            outer = v1["outer_box"][fid]
            if outer is not None:
                for cls in CONTAINER_CLASSES:
                    det = v1[cls][fid]
                    if det is not None and iou(det.box, outer.box) > 0.5:
                        container_outer_overlap += 1
                        break
            sb = v1["start_button"][fid]
            tray = v1["tray"][fid]
            if sb is not None:
                hit = False
                for cls in CONTAINER_CLASSES:
                    det = v1[cls][fid]
                    if det is not None and iou(det.box, sb.box) > 0.5:
                        hit = True
                if tray is not None and iou(tray.box, sb.box) > 0.5:
                    hit = True
                if hit:
                    start_button_overlap += 1

    # static stability: IoU of each run's own v1 picks against that run's median.
    static_ious = []
    for cls in STATIC_CLASSES:
        for run_id in run_ids:
            boxes = [
                v1_by_run[run_id][cls][fid].box
                for fid in by_run_frames[run_id]
                if v1_by_run[run_id][cls][fid] is not None
            ]
            med = median_box(boxes)
            if med is None:
                continue
            static_ious.extend(iou(b, med) for b in boxes)

    return {
        "n_frames": n_frames,
        "red_yellow_overlap_iou_gt_0.5": red_yellow_overlap,
        "container_outer_overlap_iou_gt_0.5": container_outer_overlap,
        "start_button_overlap_iou_gt_0.5": start_button_overlap,
        "mean_static_box_iou_vs_run_median": float(np.mean(static_ious)) if static_ious else None,
    }


def proxy_metrics_v2(
    run_ids: list[str],
    by_run_frames: dict[str, list[int]],
    v2_by_run: dict[str, dict],
) -> dict:
    red_yellow_overlap = 0
    container_outer_overlap = 0
    start_button_overlap = 0
    n_frames = 0
    missing_counts = dict.fromkeys(ALL_CLASSES, 0)
    static_match_ious = []

    for run_id in run_ids:
        v2 = v2_by_run[run_id]
        for cls in STATIC_CLASSES:
            sc = v2["static"][cls]
            for fid in by_run_frames[run_id]:
                if sc.per_frame_box[fid] is None:
                    missing_counts[cls] += 1

        for fid in by_run_frames[run_id]:
            n_frames += 1
            red_sel = v2["container"]["red_box"][fid]
            yellow_sel = v2["container"]["yellow_box"][fid]
            if red_sel.chosen is None:
                missing_counts["red_box"] += 1
            if yellow_sel.chosen is None:
                missing_counts["yellow_box"] += 1
            if red_sel.chosen is not None and yellow_sel.chosen is not None:
                if iou(red_sel.chosen.box, yellow_sel.chosen.box) > 0.5:
                    red_yellow_overlap += 1

            outer_box = v2["exclude_boxes"]["outer_box"]
            if outer_box is not None:
                for sel in (red_sel, yellow_sel):
                    if sel.chosen is not None and iou(sel.chosen.box, outer_box) > 0.5:
                        container_outer_overlap += 1
                        break

            sb_box = v2["exclude_boxes"]["start_button"]
            tray_box = v2["exclude_boxes"]["tray"]
            if sb_box is not None:
                hit = False
                for sel in (red_sel, yellow_sel):
                    if sel.chosen is not None and iou(sel.chosen.box, sb_box) > 0.5:
                        hit = True
                if tray_box is not None and iou(tray_box, sb_box) > 0.5:
                    hit = True
                if hit:
                    start_button_overlap += 1

        for cls in STATIC_CLASSES:
            sc = v2["static"][cls]
            if sc.consensus_box is None:
                continue
            for fid in by_run_frames[run_id]:
                dets = sc.per_frame_box[fid]
                if dets is not None:
                    static_match_ious.append(1.0)  # smoothed box IS the consensus, by construction

    return {
        "n_frames": n_frames,
        "red_yellow_overlap_iou_gt_0.5": red_yellow_overlap,
        "container_outer_overlap_iou_gt_0.5": container_outer_overlap,
        "start_button_overlap_iou_gt_0.5": start_button_overlap,
        "missing_counts": missing_counts,
        "mean_static_box_iou_vs_run_median": (
            float(np.mean(static_match_ious)) if static_match_ious else None
        ),
        "static_iou_note": (
            "v2 assigns the SMOOTHED consensus box itself to every matching frame "
            "(essential-features.md Stage 3), so this is trivially 1.0 for every "
            "matched frame -- not comparable in magnitude to the v1 column, which "
            "reports the raw per-frame pick's IoU against its own run median. "
            "See missing_counts for v2's actual stability signal (an unmatched "
            "frame is 'missing', not a low-IoU box)."
        ),
    }


# --- hue/saturation (item 2d) -----------------------------------------------


def hue_saturation_table(
    run_ids: list[str],
    by_run_frames: dict[str, list[int]],
    by_rfc: dict[tuple[str, int, str], list[RawDetection]],
    dims: dict[tuple[str, int], tuple[int, int]],
    bands: dict[str, tuple[float, float]],
    v2_by_run: dict[str, dict],
) -> dict:
    samples: dict[str, list[tuple[str, int, float, float]]] = {cls: [] for cls in CONTAINER_CLASSES}
    frame_cache: dict[tuple[str, int], np.ndarray] = {}

    for run_id in run_ids:
        outer_box = v2_by_run[run_id]["exclude_boxes"]["outer_box"]
        for cls in CONTAINER_CLASSES:
            per_frame = {fid: by_rfc.get((run_id, fid, cls), []) for fid in by_run_frames[run_id]}
            width, height = dims[(run_id, by_run_frames[run_id][0])]
            clean = single_clean_candidate(per_frame, bands[cls], outer_box, width, height)
            for fid, det in clean.items():
                key = (run_id, fid)
                if key not in frame_cache:
                    frame_cache[key] = _read_frame(RUNS_DIR / run_id / "video.mp4", fid)
                frame_bgr = frame_cache[key]
                if frame_bgr is None:
                    continue
                hs = mean_hue_sat_in_box(frame_bgr, central_box(det.box, 0.5))
                if hs is not None:
                    samples[cls].append((run_id, fid, hs[0], hs[1]))

    result = {}
    for cls in CONTAINER_CLASSES:
        vals = samples[cls]
        n = len(vals)
        if n == 0:
            result[cls] = {"n": 0}
            continue
        hues = [v[2] for v in vals]
        sats = [v[3] for v in vals]
        possible_skin = sum(1 for s in sats if s < SKIN_SAT_THRESHOLD)
        result[cls] = {
            "n": n,
            "mean_hue": float(np.mean(hues)),
            "median_hue": float(np.median(hues)),
            "min_hue": float(np.min(hues)),
            "max_hue": float(np.max(hues)),
            "mean_sat": float(np.mean(sats)),
            "median_sat": float(np.median(sats)),
            "min_sat": float(np.min(sats)),
            "max_sat": float(np.max(sats)),
            "possible_skin_frames_sat_lt_100": possible_skin,
            "frames": [f"{r}#{f}" for r, f, _, _ in vals],
        }
    return result


# --- contact sheets v2 (item 4) ---------------------------------------------

COLORS = {
    "outer_box": (255, 128, 0),
    "tray": (128, 128, 128),
    "red_box": (0, 0, 255),
    "yellow_box": (0, 200, 200),
    "start_button": (255, 0, 255),
}


def _draw_v2_boxes(frame_bgr: np.ndarray, boxes: dict[str, tuple | None]) -> np.ndarray:
    canvas = frame_bgr.copy()
    for cls, box in boxes.items():
        if box is None:
            continue
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        color = COLORS[cls]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, cls, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return canvas


def _legend_bar(width: int, height: int = 40) -> np.ndarray:
    bar = np.full((height, width, 3), 20, dtype=np.uint8)
    x = 10
    for cls, color in COLORS.items():
        cv2.rectangle(bar, (x, 10), (x + 20, 30), color, -1)
        cv2.putText(bar, cls, (x + 26, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        x += 26 + 12 * len(cls) + 30
    return bar


def write_contact_sheets_v2(
    tiles: list[tuple[str, np.ndarray]], out_dir: Path, per_sheet: int = 12
) -> list[str]:
    paths: list[str] = []
    thumb_w, thumb_h = 320, 220
    cols = 4
    for sheet_start in range(0, len(tiles), per_sheet):
        chunk = tiles[sheet_start : sheet_start + per_sheet]
        rows = (len(chunk) + cols - 1) // cols
        legend = _legend_bar(cols * thumb_w)
        sheet = np.full((legend.shape[0] + rows * thumb_h, cols * thumb_w, 3), 40, dtype=np.uint8)
        sheet[: legend.shape[0]] = legend
        for i, (tag, img) in enumerate(chunk):
            thumb = cv2.resize(img, (thumb_w, thumb_h - 24))
            tile = np.full((thumb_h, thumb_w, 3), 40, dtype=np.uint8)
            tile[: thumb_h - 24] = thumb
            cv2.putText(
                tile, tag, (6, thumb_h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1
            )
            r, c = divmod(i, cols)
            y0 = legend.shape[0] + r * thumb_h
            sheet[y0 : y0 + thumb_h, c * thumb_w : (c + 1) * thumb_w] = tile
        idx = sheet_start // per_sheet + 1
        path = out_dir / f"contact_sheet_v2_{idx:02d}.jpg"
        cv2.imwrite(str(path), sheet)
        paths.append(str(path))
    return paths


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_raw_cache(RAW_CACHE_PATH)
    log.info("Loaded %d raw-candidate records from %s", len(records), RAW_CACHE_PATH)

    by_rfc = index_by_run_frame_class(records)
    by_run_frames = frames_by_run(records)
    dims = frame_dims(records)
    run_ids = sorted(by_run_frames)

    checkpoint1_report = json.loads(CHECKPOINT1_REPORT_PATH.read_text())
    bands, band_details = derive_area_bands(by_rfc, dims, checkpoint1_report)
    log.info("Derived area bands: %s", bands)
    log.info("Area band derivation detail: %s", band_details)

    v2_by_run = {
        run_id: run_v2_selection(run_id, by_run_frames[run_id], by_rfc, dims, bands)
        for run_id in run_ids
    }
    v1_by_run = {
        run_id: run_v1_selection(run_id, by_run_frames[run_id], by_rfc, dims) for run_id in run_ids
    }

    metrics_v1 = proxy_metrics_v1(run_ids, by_run_frames, v1_by_run)
    metrics_v2 = proxy_metrics_v2(run_ids, by_run_frames, v2_by_run)
    log.info("v1 proxy metrics: %s", metrics_v1)
    log.info("v2 proxy metrics: %s", metrics_v2)

    hue_table = hue_saturation_table(run_ids, by_run_frames, by_rfc, dims, bands, v2_by_run)
    hue_table_no_frames = {
        k: {kk: vv for kk, vv in v.items() if kk != "frames"} for k, v in hue_table.items()
    }
    log.info("Hue/saturation table: %s", hue_table_no_frames)

    # which phrasing wins most often, per container class (static classes'
    # winning box is a median, not attributable to one phrase/call).
    phrase_tally = {cls: Counter() for cls in CONTAINER_CLASSES}
    for run_id in run_ids:
        for cls in CONTAINER_CLASSES:
            for fid in by_run_frames[run_id]:
                sel = v2_by_run[run_id]["container"][cls][fid]
                if sel.chosen is not None:
                    phrase_tally[cls][sel.chosen.phrase] += 1
    log.info("Winning phrase tally: %s", {k: dict(v) for k, v in phrase_tally.items()})

    # per-frame, per-class chosen box or missing + rejection reasons, for bad_boxes review context.
    per_frame_output = []
    for run_id in run_ids:
        v2 = v2_by_run[run_id]
        for fid in by_run_frames[run_id]:
            entry = {"run_id": run_id, "frame_id": fid, "classes": {}}
            for cls in STATIC_CLASSES:
                box = v2["static"][cls].per_frame_box[fid]
                entry["classes"][cls] = {"box": box, "status": "ok" if box else "missing"}
            for cls in CONTAINER_CLASSES:
                sel = v2["container"][cls][fid]
                entry["classes"][cls] = {
                    "box": sel.chosen.box if sel.chosen else None,
                    "status": "ok" if sel.chosen else "missing",
                    "rejected": [(d.box, reason) for d, reason in sel.rejected],
                }
            per_frame_output.append(entry)

    report_v2 = {
        "area_bands": bands,
        "area_band_derivation": band_details,
        "proxy_metrics_v1": metrics_v1,
        "proxy_metrics_v2": metrics_v2,
        "hue_saturation": hue_table,
        "winning_phrase_tally": {k: dict(v) for k, v in phrase_tally.items()},
        "per_frame": per_frame_output,
    }
    (OUT_DIR / "report_v2.json").write_text(json.dumps(report_v2, indent=2, default=str))
    log.info("Wrote %s", OUT_DIR / "report_v2.json")

    # contact sheets: draw final v2 boxes only.
    tiles = []
    for run_id in run_ids:
        v2 = v2_by_run[run_id]
        for fid in by_run_frames[run_id]:
            frame_bgr = _read_frame(RUNS_DIR / run_id / "video.mp4", fid)
            if frame_bgr is None:
                continue
            boxes = {cls: v2["static"][cls].per_frame_box[fid] for cls in STATIC_CLASSES}
            for cls in CONTAINER_CLASSES:
                sel = v2["container"][cls][fid]
                boxes[cls] = sel.chosen.box if sel.chosen else None
            tag = f"{run_id}#{fid}"
            tiles.append((tag, _draw_v2_boxes(frame_bgr, boxes)))

    sheets = write_contact_sheets_v2(tiles, OUT_DIR)
    log.info("Wrote %d contact sheet(s): %s", len(sheets), sheets)

    bad_boxes_path = OUT_DIR / "bad_boxes.csv"
    with bad_boxes_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "frame_id", "class", "reason"])
    log.info("Wrote %s (header only, for the operator to fill in)", bad_boxes_path)


if __name__ == "__main__":
    main()
