"""P1.2 addendum (S-C): START press rule variants on TRAIN clips. Measurement
only: nothing in state/, engine/, perception/, contracts.py or config/ is
touched. Orchestration (model + I/O), like run_cp2.py; the tested pure logic
is touch_variants.py (and cp2.py / select_v3.py, reused).

  python -m training.spikes.run_cp3 plan
  python -m training.spikes.run_cp3 cache     # card model calls; resumable; HF_HUB_OFFLINE=1
  python -m training.spikes.run_cp3 cards     # v3 static consensus -> cards.json (no model)
  python -m training.spikes.run_cp3 hands     # MediaPipe on every frame at 4 and 8 fps
  python -m training.spikes.run_cp3 analyze   # offline: validity check, variant tables

Only runs whose manifest split is "train" are ever read.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from contracts import Hand, PerceptionConfig  # noqa: E402
from training.spikes.cp2 import (  # noqa: E402
    agreeing_median_score,
    sample_frame_ids,
    spread_subset,
)
from training.spikes.run_checkpoint1 import RUNS_DIR  # noqa: E402
from training.spikes.run_checkpoint1b import (  # noqa: E402
    frame_dims,
    index_by_run_frame_class,
    load_raw_cache,
)
from training.spikes.run_checkpoint1c import FROZEN_PATH, select_v3_frames  # noqa: E402
from training.spikes.run_cp2 import (  # noqa: E402
    BOX_THRESHOLD,
    N_STATIC_FRAMES,
    TEXT_THRESHOLD,
    _FrameStore,
    _read_frames,
)
from training.spikes.run_holdout import MAX_CANDIDATES, PHRASES, SCORE_FLOOR  # noqa: E402
from training.spikes.touch_variants import (  # noqa: E402
    VARIANTS,
    classify_count,
    hold_frames,
    press_events,
    tally_labels,
    touch_flags,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("p1.2-cp3")

OUT_DIR = Path("data/spikes/cp3")
CP2_DIR = Path("data/spikes/cp2")
PLAN_PATH = OUT_DIR / "plan.json"
CARDS_PATH = OUT_DIR / "cards.json"
FPS_LIST = (4, 8)
HOLDS_S = (0.25, 0.5, 0.75, 1.0)
CLASS = "start_button"
# cp2's real StateTracker start_pressed events (data/spikes/cp2/<run>/report.json)
VALIDITY_RUNS = ("x011", "x019", "x001", "x037", "x043")
# Tracker defaults (PerceptionConfig); baseline/release as time at the rate cp2 ran at.
BASELINE_S = 2.5  # 10 frames at 4 fps
RELEASE_S = 1.25  # 5 frames at 4 fps


def train_rows() -> list[dict]:
    with (RUNS_DIR / "manifest.csv").open() as f:
        return [r for r in csv.DictReader(f) if r["split"] == "train"]


def make_plan() -> dict:
    plan: dict = {"fps_list": list(FPS_LIST), "runs": {}}
    for r in train_rows():
        run_id = r["run_id"]
        assert r["split"] == "train", run_id
        script = json.loads((RUNS_DIR / run_id / "script.json").read_text())
        fc = int(r["frames"])
        ids = {str(f): sample_frame_ids(fc, script["fps"], f) for f in FPS_LIST}
        plan["runs"][run_id] = {
            "script_type": script["script_type"],
            "fps": script["fps"],
            "frame_count": fc,
            "expected_presses": script["performed_steps"].count("start_pressed"),
            "performed_steps": script["performed_steps"],
            "frames": ids,
            "static_frames": spread_subset(ids["4"], N_STATIC_FRAMES),
        }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLAN_PATH.write_text(json.dumps(plan, indent=1))
    return plan


def _done_keys(path: Path) -> set:
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["frame_id"], r["phrase"]))
    return done


def _seed_from_cp2(run_id: str, path: Path) -> int:
    """cp2 already ran the identical call (same frames, phrase, thresholds) on
    five of these clips: copy those start_button lines instead of recomputing."""
    src = CP2_DIR / run_id / "raw.jsonl"
    if not src.exists() or path.exists():
        return 0
    n = 0
    with path.open("w") as out:
        for line in src.read_text().splitlines():
            if line.strip() and json.loads(line)["class"] == CLASS:
                out.write(line + "\n")
                n += 1
    return n


def run_cache() -> None:
    from training.spikes.detector import GroundingDinoSpike

    plan = json.loads(PLAN_PATH.read_text())
    detector = None
    latencies: list[float] = []
    for run_id, info in plan["runs"].items():
        run_dir = OUT_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "raw.jsonl"
        seeded = _seed_from_cp2(run_id, path)
        if seeded:
            log.info("%s: %d lines copied from cp2 (same call)", run_id, seeded)
        done = _done_keys(path)
        static = set(info["static_frames"])
        phrase = PHRASES[CLASS]
        pending_ids = {f for f in static if (f, phrase) not in done}
        if not pending_ids:
            log.info("RUN DONE %s (cached)", run_id)
            continue
        if detector is None:
            detector = GroundingDinoSpike(
                box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD
            )
        cap = cv2.VideoCapture(str(RUNS_DIR / run_id / "video.mp4"))
        fid = -1
        with path.open("a") as out:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                fid += 1
                if fid not in pending_ids:
                    continue
                h, w = frame.shape[:2]
                rgb = np.ascontiguousarray(frame[..., ::-1])  # BGR -> RGB at the model boundary
                t0 = time.perf_counter()
                res = detector.detect(rgb, [phrase])
                dt = time.perf_counter() - t0
                latencies.append(dt)
                cands = [
                    d
                    for d in sorted(res.detections, key=lambda d: -d.score)
                    if d.score >= SCORE_FLOOR
                ][:MAX_CANDIDATES]
                rec = {
                    "run_id": run_id,
                    "frame_id": fid,
                    "class": CLASS,
                    "phrase": phrase,
                    "width": w,
                    "height": h,
                    "seconds": dt,
                    "candidates": [{"box": list(d.box), "score": d.score} for d in cands],
                }
                out.write(json.dumps(rec) + "\n")
                out.flush()
        cap.release()
        log.info("RUN DONE %s (%d calls so far)", run_id, len(latencies))
    if latencies:
        log.info(
            "ALL DONE: %d calls, mean %.2fs, total %.1f min",
            len(latencies),
            np.mean(latencies),
            sum(latencies) / 60,
        )


def run_cards() -> None:
    params = json.loads(FROZEN_PATH.read_text())
    plan = json.loads(PLAN_PATH.read_text())
    out: dict = {}
    for run_id, info in plan["runs"].items():
        records = load_raw_cache(OUT_DIR / run_id / "raw.jsonl")
        by_rfc = index_by_run_frame_class(records)
        dims = frame_dims(records)
        static_ids = info["static_frames"]
        images = _FrameStore(_read_frames(RUNS_DIR / run_id / "video.mp4", set(static_ids)))
        res = select_v3_frames(run_id, static_ids, by_rfc, dims, params, images.get_frame)
        sb = res["static"][CLASS]
        box = sb.consensus_box
        conf = (
            agreeing_median_score({f: by_rfc.get((run_id, f, CLASS), []) for f in static_ids}, box)
            if box is not None
            else 0.0
        )
        found = box is not None and res["start_button_white"] is True
        out[run_id] = {
            "found": bool(found),
            "box": list(box) if box is not None else None,
            "conf": conf,
            "white": res["start_button_white"],
            "white_sat_val": res["start_button_measure"],
            "frame_support": sb.support,
            "n_static_frames": len(static_ids),
        }
        log.info("%s card: %s", run_id, out[run_id])
    CARDS_PATH.write_text(json.dumps(out, indent=1))
    log.info("cards found: %d of %d", sum(v["found"] for v in out.values()), len(out))


def run_hands() -> None:
    from perception.hands import HandTracker

    plan = json.loads(PLAN_PATH.read_text())
    tpath = OUT_DIR / "hands_timing.json"
    timing: dict = json.loads(tpath.read_text()) if tpath.exists() else {}
    for run_id, info in plan["runs"].items():
        wanted = {f for fps in FPS_LIST for f in info["frames"][str(fps)]}
        images = _read_frames(RUNS_DIR / run_id / "video.mp4", wanted)
        for fps in FPS_LIST:
            path = OUT_DIR / run_id / f"hands_{fps}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                continue
            ids = info["frames"][str(fps)]
            ht = HandTracker()  # fresh landmarker per run and rate, VIDEO mode
            lines = []
            t0 = time.perf_counter()
            for fid in ids:
                hands = ht.process(images[fid], fid / info["fps"])
                lines.append(
                    json.dumps(
                        {
                            "run_id": run_id,
                            "target_fps": fps,
                            "frame_id": fid,
                            "t": fid / info["fps"],
                            "hands": [h.model_dump() for h in hands],
                        }
                    )
                )
            dt = time.perf_counter() - t0
            ht.close()
            path.write_text("\n".join(lines) + "\n")
            timing[f"{run_id}@{fps}"] = {"frames": len(ids), "seconds": dt}
            log.info(
                "%s @%d fps: %d frames, %.1f s (%.1f fps)", run_id, fps, len(ids), dt, len(ids) / dt
            )
    tpath.write_text(json.dumps(timing, indent=1))


def load_hands(run_id: str, fps: int) -> tuple[list[int], list[list[Hand]]]:
    ids: list[int] = []
    hands: list[list[Hand]] = []
    for line in (OUT_DIR / run_id / f"hands_{fps}.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            assert r["target_fps"] == fps
            ids.append(r["frame_id"])
            hands.append([Hand.model_validate(h) for h in r["hands"]])
    return ids, hands


def _usable_box(card: dict, floor: float) -> list[float] | None:
    """The tracker ignores a detection below detector_conf_floor, so the START
    rule would be false for the whole clip."""
    if not card["found"] or card["conf"] < floor:
        return None
    return card["box"]


def run_analyze() -> None:
    pcfg = PerceptionConfig()
    plan = json.loads(PLAN_PATH.read_text())
    cards = json.loads(CARDS_PATH.read_text())
    runs = plan["runs"]
    usable = {r: _usable_box(cards[r], pcfg.detector_conf_floor) for r in runs if cards[r]["found"]}
    below_floor = [r for r in runs if cards[r]["found"] and usable[r] is None]
    not_found = [r for r in runs if not cards[r]["found"]]
    out: dict = {
        "n_train_runs": len(runs),
        "card_not_found": not_found,
        "card_conf_below_floor": below_floor,
        "validity": {},
        "hand_share": {},
        "hands_fps": {},
        "tip_in_box_share": {},
        "table": {},
        "per_clip": {},
    }

    # --- validity: V0 with tracker defaults must reproduce cp2's real tracker events
    ok_all = True
    for r in VALIDITY_RUNS:
        rep = json.loads((CP2_DIR / r / "report.json").read_text())
        want = [round(e["t"], 2) for e in rep["state_events"] if e["step_id"] == "start_pressed"]
        ids, hands = load_hands(r, 4)
        flags = touch_flags(hands, usable.get(r), "V0")
        got_i = press_events(
            flags, pcfg.baseline_frames, pcfg.hysteresis_frames, pcfg.release_frames
        )
        got = [round(ids[i] / runs[r]["fps"], 2) for i in got_i]
        ok = got == want
        ok_all &= ok
        out["validity"][r] = {"cp2_real_tracker": want, "reproduced": got, "same": ok}
    out["validity_ok"] = ok_all
    log.info("VALIDITY %s %s", "OK" if ok_all else "FAILED", json.dumps(out["validity"]))
    if not ok_all:
        (OUT_DIR / "results.json").write_text(json.dumps(out, indent=1))
        raise SystemExit("validity check failed: stop and report")

    # --- hands: share of frames with a detected hand; hands-only fps
    timing = json.loads((OUT_DIR / "hands_timing.json").read_text())
    for fps in FPS_LIST:
        n = with_hand = 0
        frames = secs = 0.0
        for r in runs:
            _, hands = load_hands(r, fps)
            n += len(hands)
            with_hand += sum(1 for h in hands if h)
            frames += timing[f"{r}@{fps}"]["frames"]
            secs += timing[f"{r}@{fps}"]["seconds"]
        out["hand_share"][str(fps)] = {"frames": n, "with_hand": with_hand, "share": with_hand / n}
        out["hands_fps"][str(fps)] = frames / secs

    # --- variants x hold x fps over usable clips
    use = [r for r in runs if usable.get(r) is not None]
    out["n_usable"] = len(use)
    n_correct = sum(runs[r]["script_type"] == "correct" for r in use)
    out["n_usable_correct"] = n_correct
    for fps in FPS_LIST:
        data = {r: load_hands(r, fps) for r in use}
        # share of frames whose index fingertip is inside the grown card box, per margin
        for v in ("V1a", "V1b", "V1c"):
            tot = tip = 0
            for r in use:
                fl = touch_flags(data[r][1], usable[r], v)
                tot += len(fl)
                tip += sum(fl)
            out["tip_in_box_share"][f"{v}@{fps}"] = {
                "frames": tot,
                "inside": tip,
                "share": tip / tot,
            }
        for variant in VARIANTS:
            flags = {r: touch_flags(data[r][1], usable[r], variant) for r in use}
            for hold_s in HOLDS_S:
                h = hold_frames(hold_s, fps)
                for mode in ("seconds", "frames"):
                    # baseline/release: fixed in seconds (default) or the tracker's frame counts
                    if mode == "seconds":
                        base, rel = hold_frames(BASELINE_S, fps), hold_frames(RELEASE_S, fps)
                    else:
                        base, rel = pcfg.baseline_frames, pcfg.release_frames
                    rows = []
                    clips = {}
                    for r in use:
                        ev = press_events(flags[r], base, h, rel)
                        lab = classify_count(len(ev), runs[r]["expected_presses"])
                        rows.append((runs[r]["script_type"], lab))
                        clips[r] = {
                            "n": len(ev),
                            "expected": runs[r]["expected_presses"],
                            "label": lab,
                            "t": [round(data[r][0][i] / runs[r]["fps"], 2) for i in ev],
                        }
                    key = f"{variant}|H={hold_s}|fps={fps}|{mode}"
                    out["table"][key] = tally_labels(rows)
                    out["per_clip"][key] = clips
    (OUT_DIR / "results.json").write_text(json.dumps(out, indent=1))
    log.info("wrote %s", OUT_DIR / "results.json")


def _tip_distance(hands: list[Hand], box: list[float]) -> float | None:
    """Smallest distance (px) from an index fingertip to the card box, 0 if
    inside the plain (ungrown) box; None if no hand."""
    best = None
    x1, y1, x2, y2 = box
    for h in hands:
        x, y = h.landmarks_px[8]
        d = float(np.hypot(max(x1 - x, 0, x - x2), max(y1 - y, 0, y - y2)))
        best = d if best is None else min(best, d)
    return best


def run_report(fps: int = 8) -> None:
    """Per-clip diagnostics from the caches (no model): why a clip is missed
    or extra. Writes data/spikes/cp3/diagnostics.md."""
    from training.spikes.cp2 import true_runs

    plan = json.loads(PLAN_PATH.read_text())
    cards = json.loads(CARDS_PATH.read_text())
    lines = [
        "clip | type | exp | frames | hand frames | V0 frames | tip(8) in 0.10/0.50 | "
        "V2 frames | longest run V1c / V2 (s) | min tip dist px (card w) | note"
    ]
    for r, info in plan["runs"].items():
        ids, hands = load_hands(r, fps)
        box = cards[r]["box"]
        w = box[2] - box[0]
        n_hand = sum(1 for h in hands if h)
        f = {v: touch_flags(hands, box, v) for v in ("V0", "V1a", "V1c", "V2")}
        runs = {v: max((n for _, n in true_runs(f[v])), default=0) / fps for v in ("V1c", "V2")}
        dists = [d for h in hands if (d := _tip_distance(h, box)) is not None]
        dmin = min(dists) if dists else None
        base = hold_frames(BASELINE_S, fps)
        note = []
        if any(f["V2"][:base]) or any(f["V1c"][:base]):
            note.append("fingertip near card in the baseline window")
        if not any(f["V2"]) and f["V0"] and any(f["V0"]):
            note.append("only palm/arm landmarks reach the card (no fingertip)")
        if not any(f["V0"]):
            note.append("no landmark reaches the card")
        lines.append(
            f"{r} | {info['script_type']} | {info['expected_presses']} | {len(hands)} | {n_hand} | "
            f"{sum(f['V0'])} | {sum(f['V1a'])}/{sum(f['V1c'])} | {sum(f['V2'])} | "
            f"{runs['V1c']:.2f} / {runs['V2']:.2f} | "
            f"{'-' if dmin is None else f'{dmin:.0f} ({dmin / w:.2f})'} | {'; '.join(note)}"
        )
    (OUT_DIR / "diagnostics.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "plan":
        p = make_plan()
        log.info(json.dumps({r: i["expected_presses"] for r, i in p["runs"].items()}))
    elif mode == "cache":
        run_cache()
    elif mode == "cards":
        run_cards()
    elif mode == "hands":
        run_hands()
    elif mode == "analyze":
        run_analyze()
    elif mode == "report":
        run_report()
    else:
        raise SystemExit(__doc__)
