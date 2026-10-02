"""P1.2 checkpoint 2: frozen v3 zero-shot detector + MediaPipe hands on 5
train clips (3 correct, 1 skip, 1 swap) at ~4 fps, fed through P2's
StateTracker and SequenceEngine. Feasibility check on 5 clips, not accuracy.
Frozen v3 parameters are loaded from data/spikes/v3/frozen_params.json and are
not re-derived or tuned. Orchestration (model + I/O), like run_holdout.py;
the tested pure logic is cp2.py.

  python -m training.spikes.run_cp2 plan
  python -m training.spikes.run_cp2 cache      # model; resumable; HF_HUB_OFFLINE=1
  python -m training.spikes.run_cp2 assemble [run ...]  # hands + tracker + engine + report
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

from contracts import (  # noqa: E402
    EngineEvent,
    ExpectedDeviation,
    ExperimentDefinition,
    PerceptionConfig,
    RuntimeConfig,
)
from training.spikes.cp2 import (  # noqa: E402
    agreeing_median_score,
    compare_deviations,
    deviation_key,
    fingertip_in_box,
    make_perception_frame,
    sample_frame_ids,
    spread_subset,
    true_runs,
)
from training.spikes.run_checkpoint1 import RUNS_DIR  # noqa: E402
from training.spikes.run_checkpoint1b import (  # noqa: E402
    CONTAINER_CLASSES,
    STATIC_CLASSES,
    frame_dims,
    index_by_run_frame_class,
    load_raw_cache,
)
from training.spikes.run_checkpoint1c import FROZEN_PATH, select_v3_frames  # noqa: E402
from training.spikes.run_holdout import MAX_CANDIDATES, PHRASES, SCORE_FLOOR  # noqa: E402
from training.spikes.select_v2 import select_container  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("p1.2-cp2")

OUT_DIR = Path("data/spikes/cp2")
PLAN_PATH = OUT_DIR / "plan.json"
TARGET_FPS = 4.0
N_STATIC_FRAMES = 5
BOX_THRESHOLD = TEXT_THRESHOLD = 0.20
# 3 correct, 1 skip, 1 swap: train runs not among the 12 of checkpoints 1/1b/1c
# nor the 5 hold-out runs; the shortest unused clips (cheapest at ~4 s/call).
RUNS = ("x011", "x019", "x001", "x037", "x043")


def make_plan() -> dict:
    with (RUNS_DIR / "manifest.csv").open() as f:
        rows = {r["run_id"]: r for r in csv.DictReader(f)}
    plan: dict = {"target_fps": TARGET_FPS, "runs": {}}
    for run_id in RUNS:
        assert rows[run_id]["split"] == "train", run_id
        script = json.loads((RUNS_DIR / run_id / "script.json").read_text())
        ids = sample_frame_ids(int(rows[run_id]["frames"]), script["fps"], TARGET_FPS)
        plan["runs"][run_id] = {
            "script_type": script["script_type"],
            "fps": script["fps"],
            "frame_count": int(rows[run_id]["frames"]),
            "performed_steps": script["performed_steps"],
            "expected_deviations": script["expected_deviations"],
            "frames": ids,
            "static_frames": spread_subset(ids, N_STATIC_FRAMES),
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


def run_cache() -> None:
    from training.spikes.detector import GroundingDinoSpike

    plan = json.loads(PLAN_PATH.read_text())
    detector = GroundingDinoSpike(box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD)
    latencies: list[float] = []
    for run_id, info in plan["runs"].items():
        run_dir = OUT_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "raw.jsonl"
        done = _done_keys(path)
        wanted = set(info["frames"])
        static = set(info["static_frames"])
        cap = cv2.VideoCapture(str(RUNS_DIR / run_id / "video.mp4"))
        fid = -1
        with path.open("a") as out:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                fid += 1
                if fid not in wanted:
                    continue
                classes = CONTAINER_CLASSES + (STATIC_CLASSES if fid in static else ())
                pending = [(c, PHRASES[c]) for c in classes if (fid, PHRASES[c]) not in done]
                if not pending:
                    continue
                h, w = frame.shape[:2]
                rgb = np.ascontiguousarray(frame[..., ::-1])  # BGR -> RGB at the model boundary
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
                        log.info(
                            "%s: frame %d, %d calls, last %.2fs", run_id, fid, len(latencies), dt
                        )
        cap.release()
        log.info("RUN DONE %s", run_id)
    if latencies:
        log.info(
            "ALL DONE: %d calls, mean %.2fs, median %.2fs, total %.1f min",
            len(latencies),
            np.mean(latencies),
            np.median(latencies),
            sum(latencies) / 60,
        )


def _read_frames(video: Path, wanted: set[int]) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    out, fid = {}, -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fid += 1
        if fid in wanted:
            out[fid] = frame
    cap.release()
    return out


class _FrameStore:
    """Adapts a frame dict to the (run_id, frame_id) loader select_v3_frames expects."""

    def __init__(self, frames: dict[int, np.ndarray]) -> None:
        self._frames = frames

    def get_frame(self, run_id: str, frame_id: int) -> np.ndarray | None:
        return self._frames.get(frame_id)

    def __getitem__(self, frame_id: int) -> np.ndarray:
        return self._frames[frame_id]


def run_assemble(only: list[str]) -> None:
    from engine.sequence import SequenceEngine
    from perception.hands import HandTracker
    from state import tracker as tracker_mod
    from state.tracker import StateTracker

    params = json.loads(FROZEN_PATH.read_text())
    exp = ExperimentDefinition.from_json("config/experiment.json")
    pcfg, rcfg = PerceptionConfig(), RuntimeConfig()
    plan = json.loads(PLAN_PATH.read_text())
    summary: dict = {"perception_config": pcfg.model_dump(), "runs": {}}
    hand_frames = hand_seconds = 0.0

    for run_id, info in plan["runs"].items():
        if only and run_id not in only:
            continue
        run_dir = OUT_DIR / run_id
        fps, ids = info["fps"], info["frames"]
        records = load_raw_cache(run_dir / "raw.jsonl")
        by_rfc = index_by_run_frame_class(records)
        dims = frame_dims(records)
        width, height = dims[(run_id, ids[0])]
        static_ids = info["static_frames"]
        container_band = tuple(params["container_band"])

        images = _FrameStore(_read_frames(RUNS_DIR / run_id / "video.mp4", set(ids)))
        res = select_v3_frames(run_id, static_ids, by_rfc, dims, params, images.get_frame)
        static_box, static_conf = {}, {}
        for cls in STATIC_CLASSES:
            box = res["static"][cls].consensus_box
            static_box[cls] = box
            static_conf[cls] = (
                agreeing_median_score(
                    {f: by_rfc.get((run_id, f, cls), []) for f in static_ids}, box
                )
                if box is not None
                else 0.0
            )

        # hands on the same frames (fresh landmarker per run, VIDEO mode)
        ht = HandTracker()
        hands_by_frame = {}
        t0 = time.perf_counter()
        for fid in ids:
            hands_by_frame[fid] = ht.process(images[fid], fid / fps)
        h_dt = time.perf_counter() - t0
        ht.close()
        hand_frames += len(ids)
        hand_seconds += h_dt

        tracker = StateTracker(exp, pcfg)
        engine = SequenceEngine(exp, rcfg)
        engine.start(0.0, run_id)
        frames_out, state_events, engine_events, pf_lines = [], [], [], []
        card = static_box["start_button"]
        for fid in ids:
            dets: dict = {
                c: (static_box[c], static_conf[c]) if static_box[c] else None
                for c in STATIC_CLASSES
            }
            reasons = {}
            for cls in CONTAINER_CLASSES:
                sel = select_container(
                    by_rfc.get((run_id, fid, cls), []),
                    container_band,
                    width,
                    height,
                    res["exclude"],
                )
                dets[cls] = (sel.chosen.box, sel.chosen.score) if sel.chosen else None
                if sel.chosen is None:
                    reasons[cls] = sorted({r for _, r in sel.rejected}) or ["no_candidate"]
            pf = make_perception_frame(fid, fps, dets, hands_by_frame[fid])
            pf_lines.append(pf.model_dump_json())
            # diagnostics only: per-frame step truth via the tracker's own (private) evaluator
            filt = tracker_mod._floor_filter(pf.detections, pcfg.detector_conf_floor)
            truth = {
                s.step_id: tracker_mod._evaluate_step(s, filt, pf.hands, pcfg)[0] for s in exp.steps
            }
            tip_on = bool(
                card and any(fingertip_in_box(h, card, pcfg.touch_margin_frac) for h in pf.hands)
            )
            frames_out.append(
                {
                    "frame_id": fid,
                    "t": pf.t,
                    "n_hands": len(pf.hands),
                    "missing": [c for c in CONTAINER_CLASSES if dets[c] is None],
                    "missing_reason": reasons,
                    "below_floor": [
                        d.label for d in pf.detections if d.conf < pcfg.detector_conf_floor
                    ],
                    "conf": {d.label: round(d.conf, 3) for d in pf.detections},
                    "step_truth": [k for k, v in truth.items() if v],
                    "fingertip_on_card": tip_on,
                }
            )
            for ev in tracker.update(pf):
                state_events.append(ev.model_dump())
                for ee in engine.on_state_event(ev):
                    engine_events.append(ee.model_dump())
        for ee in engine.finish(ids[-1] / fps):
            engine_events.append(ee.model_dump())

        produced = [
            deviation_key(EngineEvent.model_validate(e))
            for e in engine_events
            if e["kind"] == "deviation_detected"
        ]
        expected = [ExpectedDeviation.model_validate(e) for e in info["expected_deviations"]]
        out = {
            "script_type": info["script_type"],
            "performed_steps": info["performed_steps"],
            "expected_deviations": info["expected_deviations"],
            "n_frames": len(ids),
            "static_boxes": static_box,
            "static_conf": static_conf,
            "start_button_white": res["start_button_white"],
            "state_events": state_events,
            "engine_events": engine_events,
            "snapshot": [s.model_dump() for s in engine.snapshot()],
            "deviations": compare_deviations(expected, produced),
            "hands_seconds": h_dt,
            "hands_fps": len(ids) / h_dt,
            "frames": frames_out,
        }
        (run_dir / "perception.jsonl").write_text("\n".join(pf_lines) + "\n")
        (run_dir / "report.json").write_text(json.dumps(out, indent=1, default=str))
        summary["runs"][run_id] = {k: out[k] for k in ("script_type", "deviations", "hands_fps")}
        log.info("%s done: fired %s", run_id, [e["step_id"] for e in state_events])
    summary["hands_fps_overall"] = hand_frames / hand_seconds if hand_seconds else None
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    log.info("hands-only fps overall %s", summary["hands_fps_overall"])


def run_tables() -> None:
    """Per-clip diagnostic tables from the saved report.json files (no model)."""
    pcfg = PerceptionConfig()
    lines: list[str] = []
    for run_id in RUNS:
        path = OUT_DIR / run_id / "report.json"
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        fr = r["frames"]
        lines.append(f"## {run_id} ({r['script_type']}), {len(fr)} frames at ~{TARGET_FPS:g} fps")
        lines.append(f"performed: {r['performed_steps']}")
        lines.append(
            "fired (t): "
            + ", ".join(
                f"{e['step_id']}@{e['t']:.2f}{'?' if e['uncertain'] else ''}"
                for e in r["state_events"]
            )
        )
        lines.append(f"expected deviations: {r['expected_deviations']}")
        lines.append("deviations vs expected: " + json.dumps(r["deviations"]))
        for e in r["engine_events"]:
            if e["kind"] != "step_confirmed":
                lines.append(
                    f"  engine t={e['t']:.2f} {e['kind']} {e['deviation_type']} "
                    f"step={e['step_id']} skipped={e['skipped_step_ids']} tag={e['confidence_tag']}"
                )
        steps = [
            "red_out",
            "red_in_tray",
            "yellow_out",
            "yellow_in_tray",
            "start_pressed",
            "red_stowed",
            "yellow_stowed",
        ]
        lines.append(f"step truth runs (start t, frames) [hysteresis={pcfg.hysteresis_frames}]:")
        for sid in steps:
            runs = true_runs([sid in f["step_truth"] for f in fr])
            lines.append(
                f"  {sid}: " + " ".join(f"({fr[i]['t']:.2f},{n})" for i, n in runs)
                if runs
                else f"  {sid}: never true"
            )
        for cls in CONTAINER_CLASSES:
            miss = true_runs([cls in f["missing"] for f in fr])
            low = sum(cls in f["below_floor"] for f in fr)
            lines.append(
                f"  {cls}: missing {sum(n for _, n in miss)} frames in runs "
                + " ".join(f"({fr[i]['t']:.2f},{n})" for i, n in miss)
                + f"; below floor {low} frames"
            )
            reasons: dict[str, int] = {}
            for f in fr:
                for why in f["missing_reason"].get(cls, []):
                    reasons[why] = reasons.get(why, 0) + 1
            lines.append(f"    missing reasons (frame counts): {reasons}")
        st = true_runs(["start_pressed" in f["step_truth"] for f in fr])
        lines.append("  START truth runs (frames with the index fingertip on the card):")
        for i, n in st:
            tips = sum(f["fingertip_on_card"] for f in fr[i : i + n])
            lines.append(f"    t={fr[i]['t']:.2f} n={n} fingertip_on_card={tips}")
        lines.append(f"  hands-only fps {r['hands_fps']:.1f}; static conf {r['static_conf']}")
        lines.append("")
    (OUT_DIR / "tables.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "plan":
        log.info(json.dumps({r: len(i["frames"]) for r, i in make_plan()["runs"].items()}))
    elif mode == "cache":
        run_cache()
    elif mode == "assemble":
        run_assemble(sys.argv[2:])
    elif mode == "tables":
        run_tables()
    else:
        raise SystemExit(__doc__)
