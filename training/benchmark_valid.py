"""S-F2a: CPU latency of the two fine-tuned detectors and the MediaPipe hand tracker on frames
drawn from the VALID split (never test), and the pipeline view that pre-registered rule (2) reads.

  python -m training.benchmark_valid [--runs 2] [--warmup 20] [--timed 150] [--out PATH]

Each candidate is timed per frame with ``time.perf_counter`` around its own call only (the image
is already decoded: BGR in, RGB only inside the RF-DETR predictor, as in ``eval_detector``). The
pipeline view is ``median(detector) + median(hands) + overhead``; with the detector on every
k-th frame the average frame time is ``median(detector)/k + median(hands) + overhead`` while the
slowest frame stays ``median(detector) + median(hands) + overhead``. The budget is the pipeline
fps in ``config/acceptance.yaml`` (125 ms at 8 fps). Nothing is downloaded; nothing here chooses
a detector.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from training.benchmark_cpu import _percentile

OVERHEAD_MS = 5.0
BUDGET_FPS = 8.0
SEED = 0


def latency_stats(seconds: list[float]) -> dict[str, float]:
    if not seconds:
        raise ValueError("no timings")
    ms = [s * 1000.0 for s in seconds]
    return {
        "n": len(ms),
        "median_ms": statistics.median(ms),
        "p95_ms": _percentile(ms, 95.0),
        "mean_ms": statistics.fmean(ms),
    }


def pipeline_view(
    det_median_ms: float,
    hands_median_ms: float,
    overhead_ms: float = OVERHEAD_MS,
    budget_fps: float = BUDGET_FPS,
) -> dict[str, dict[str, Any]]:
    budget_ms = 1000.0 / budget_fps
    out: dict[str, dict[str, Any]] = {}
    for k in (1, 2, 3):
        avg = det_median_ms / k + hands_median_ms + overhead_ms
        slow = det_median_ms + hands_median_ms + overhead_ms
        out[f"detector_every_{k}"] = {
            "avg_frame_ms": avg,
            "fps": 1000.0 / avg,
            "frame_ms_with_detector": slow,
            "meets_budget": avg <= budget_ms + 1e-9,
        }
    return out


def draw_frames(names: list[str], n: int, seed: int = SEED) -> list[str]:
    """``n`` file names in a seeded random order, cycling through the shuffled list if short."""
    order = sorted(names)
    random.Random(seed).shuffle(order)
    return [order[i % len(order)] for i in range(n)]


def time_calls(
    fn: Callable[[Any], Any],
    frames: list[Any],
    warmup: int,
    timed: int,
    clock: Callable[[], float] = time.perf_counter,
) -> list[float]:
    """Run ``fn`` on the first ``warmup`` frames untimed, then on the next ``timed`` frames,
    returning each timed call's seconds."""
    if len(frames) < warmup + timed:
        raise ValueError(f"need {warmup + timed} frames, got {len(frames)}")
    for f in frames[:warmup]:
        fn(f)
    out = []
    for f in frames[warmup : warmup + timed]:
        t0 = clock()
        fn(f)
        out.append(clock() - t0)
    return out


# --- machine facts ----------------------------------------------------------------------------


def _run(cmd: list[str]) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=False)
        return r.stdout.strip() or None
    except Exception:  # noqa: BLE001 - machine facts are best effort
        return None


def _on_ac_power() -> bool | None:
    if os.name != "nt":
        return None
    try:
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [
                ("ACLineStatus", ctypes.c_ubyte),
                ("BatteryFlag", ctypes.c_ubyte),
                ("BatteryLifePercent", ctypes.c_ubyte),
                ("SystemStatusFlag", ctypes.c_ubyte),
                ("BatteryLifeTime", ctypes.c_ulong),
                ("BatteryFullLifeTime", ctypes.c_ulong),
            ]

        s = Status()
        if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)):
            return None
        return {0: False, 1: True}.get(s.ACLineStatus)
    except Exception:  # noqa: BLE001
        return None


def machine_facts() -> dict[str, Any]:
    cpu = platform.processor() or None
    plan = None
    if os.name == "nt":
        cpu = (
            _run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"])
            or cpu
        )
        plan = _run(["powercfg", "/getactivescheme"])
    try:
        import torch

        threads = torch.get_num_threads()
        torch_version = torch.__version__
    except Exception:  # noqa: BLE001
        threads, torch_version = None, None
    return {
        "cpu": cpu,
        "logical_cores": os.cpu_count(),
        "power_plan": plan,
        "on_ac_power": _on_ac_power(),
        "torch_threads": threads,
        "torch_version": torch_version,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }


# --- candidates -------------------------------------------------------------------------------


def _candidates(weights_rf: Path, weights_yolo: Path, hand_model: Path, floor: float):
    """name -> callable(BGR frame); built once, all on CPU."""
    from rfdetr import RFDETR
    from ultralytics import YOLO

    from perception.hands import HandTracker
    from training.eval_detector import rfdetr_predictor

    cands: dict[str, Callable[[np.ndarray], Any]] = {}
    notes: dict[str, str] = {}

    rf = RFDETR.from_checkpoint(str(weights_rf), trust_checkpoint=True, device="cpu")
    cands["rfdetr_nano"] = rfdetr_predictor(rf, floor=floor, n_classes=5)

    rf_opt = RFDETR.from_checkpoint(str(weights_rf), trust_checkpoint=True, device="cpu")
    try:
        opt = getattr(rf_opt, "optimize_for_inference", None) or rf_opt.inference
        opt()
        cands["rfdetr_nano_optimized"] = rfdetr_predictor(rf_opt, floor=floor, n_classes=5)
        notes["rfdetr_nano_optimized"] = "extra row: model.optimize_for_inference() applied"
    except Exception as e:  # noqa: BLE001
        notes["rfdetr_nano_optimized"] = f"skipped: {type(e).__name__}: {e}"

    y = YOLO(str(weights_yolo))
    cands["yolo11n"] = lambda bgr: y.predict(
        bgr, conf=floor, imgsz=384, device="cpu", verbose=False
    )

    tracker = HandTracker(hand_model)
    state = {"t": 0.0}

    def hands(bgr: np.ndarray) -> Any:
        state["t"] += 0.1
        return tracker.process(bgr, state["t"])

    cands["mediapipe_hands"] = hands
    return cands, notes


def run_once(cands: dict, order: list[str], frames: list, warmup: int, timed: int) -> dict:
    res = {}
    for name in order:
        if name not in cands:
            continue
        res[name] = latency_stats(time_calls(cands[name], frames, warmup, timed))
        time.sleep(2.0)  # cooldown between blocks
    return res


def views(results: dict) -> dict:
    hands = results["mediapipe_hands"]["median_ms"]
    return {
        name: pipeline_view(r["median_ms"], hands)
        for name, r in results.items()
        if name != "mediapipe_hands"
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rfdetr", type=Path, default=Path("weights/detector_rfdetr_nano.pth"))
    ap.add_argument("--yolo", type=Path, default=Path("weights/detector_yolo11n.pt"))
    ap.add_argument("--hands", type=Path, default=Path("weights/hand_landmarker.task"))
    ap.add_argument("--valid-dir", type=Path, default=Path("data/dataset/valid"))
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--timed", type=int, default=150)
    ap.add_argument("--out", type=Path, default=Path("reports/benchmark_cpu_valid.json"))
    args = ap.parse_args(argv)

    import cv2

    from training.eval_detector import config_floor, sha256_file

    floor = config_floor()
    names = draw_frames(
        [p.name for p in args.valid_dir.glob("*.jpg")], args.warmup + args.timed, SEED
    )
    frames = [cv2.imread(str(args.valid_dir / n), cv2.IMREAD_COLOR) for n in names]
    if any(f is None for f in frames):
        print("cannot read a valid image", file=sys.stderr)
        return 3
    cands, notes = _candidates(args.rfdetr, args.yolo, args.hands, floor)
    base = ["rfdetr_nano", "rfdetr_nano_optimized", "yolo11n", "mediapipe_hands"]
    print("BENCHMARK STARTING: the Lead must not use the laptop for about 15 minutes", flush=True)
    runs = []
    for i in range(args.runs):
        order = base if i % 2 == 0 else base[::-1]  # reversed on odd runs: exposes position drift
        res = run_once(cands, order, frames, args.warmup, args.timed)
        runs.append(
            {
                "run": i + 1,
                "order": order,
                "facts": machine_facts(),
                "results": res,
                "pipeline_view": views(res),
            }
        )
        print(json.dumps(runs[-1]["results"], indent=1), flush=True)
    report = {
        "version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "split_used": "valid",
        "input_px": [int(frames[0].shape[1]), int(frames[0].shape[0])],
        "model_resolution": 384,
        "warmup": args.warmup,
        "timed": args.timed,
        "overhead_ms": OVERHEAD_MS,
        "budget_fps": BUDGET_FPS,
        "detector_conf_floor": floor,
        "frames_seed": SEED,
        "weights_sha256": {"rfdetr": sha256_file(args.rfdetr), "yolo11n": sha256_file(args.yolo)},
        "notes": notes,
        "runs": runs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
