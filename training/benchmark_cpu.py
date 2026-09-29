"""training/benchmark_cpu.py -- Stage 8 of the dataset pipeline (F14;
essential-features.md section 14, Stage 8): CPU latency of the detector
candidates on this laptop's CPU -- the "full per-frame path" (BGR -> RGB,
predict), for RF-DETR-Nano under PyTorch (with and without
``optimize_for_inference``) and under ONNX Runtime, and YOLO11n as the
benchmarked fallback (context.md section 6, ISSUES.md 2026-09-28 "detector:
RF-DETR-Nano vs YOLO11n").

    python -m training.benchmark_cpu [--allow-download] [--out PATH]

Hands are never measured here -- that is a separate, still-open path
(no ``.task`` model file is vendored yet; ISSUES.md 2026-09-28 "MediaPipe
model files must be vendored").

**Real candidates are opt-in** (``--allow-download``): constructing
``RFDETRNano()`` or ``ultralytics.YOLO("yolo11n.pt")`` auto-downloads a
COCO-pretrained checkpoint on first use if one isn't already cached
locally (outside this repo -- ``~/.roboflow/models`` and Ultralytics' own
cache dir respectively), which this module treats as an explicit-permission
action (IMPLEMENTATION_PLAN.md R7; the exact sources, sizes and licenses
are logged in ISSUES.md, 2026-09-29). Without the flag, a missing/unwilling
candidate is simply skipped and reported as unavailable -- nothing is
downloaded or installed as a side effect of running this script.

This module's own numbers are provisional before P1.5 (weights are
COCO-pretrained, not the fine-tuned head; the class count differs) --
write ``--out`` to a git-ignored path, never ``reports/`` (P2-owned),
until P1.5 actually runs this for real.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_WARMUP_FRAMES = 20
DEFAULT_TIMED_FRAMES = 200
DEFAULT_FRAME_WIDTH = 1280  # contracts.CAPTURE_WIDTH -- essential-features.md section 0
DEFAULT_FRAME_HEIGHT = 720  # contracts.CAPTURE_HEIGHT
HANDS_NOTE = "not measured (no .task model file vendored -- ISSUES.md 2026-09-28)"


def _bgr_to_contiguous_rgb(image: np.ndarray) -> np.ndarray:
    # The same BGR->RGB conversion the real pipeline pays for at the model
    # boundary (AGENTS.md rule 16). A bare `image[..., ::-1]` view is not
    # C-contiguous and native code can misread it (ISSUES.md, 2026-09-29,
    # the hands.py finding) -- ascontiguousarray, always.
    return np.ascontiguousarray(image[..., ::-1])


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        raise ValueError("no timed frames to compute a percentile from")
    ordered = sorted(values)
    k = (len(ordered) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


@dataclass
class LatencyResult:
    name: str
    warmup_frames: int
    timed_frames: int
    mean_ms: float
    p95_ms: float
    fps: float
    per_frame_ms: list[float] = field(repr=False)

    def to_report_dict(self) -> dict[str, Any]:
        # per_frame_ms is for the caller's own analysis, not the summary
        # report (essential-features.md Stage 8 asks for mean/p95/fps).
        d = asdict(self)
        d.pop("per_frame_ms")
        return d


def benchmark_predict(
    name: str,
    predict_fn: Callable[[np.ndarray], object],
    frames: list[np.ndarray],
    warmup_frames: int = DEFAULT_WARMUP_FRAMES,
    timed_frames: int = DEFAULT_TIMED_FRAMES,
    now: Callable[[], float] = time.perf_counter,
) -> LatencyResult:
    """Times ``_bgr_to_contiguous_rgb`` + ``predict_fn(rgb)`` per frame,
    over ``warmup_frames`` (excluded) then ``timed_frames`` (reported).
    ``frames`` is cycled if shorter than ``warmup_frames + timed_frames``
    -- a fixed-size model's latency doesn't depend much on frame content
    (essential-features.md section 0), only ``predict_fn``'s own behavior
    (e.g. a variable number of detections feeding post-processing) does,
    which is exactly what timing many frames and reporting p95, not just
    one frame, is for.
    """
    if not frames:
        raise ValueError("frames must be non-empty")
    if warmup_frames < 0 or timed_frames < 1:
        raise ValueError("warmup_frames must be >= 0 and timed_frames >= 1")

    per_frame_ms: list[float] = []
    total = warmup_frames + timed_frames
    for i in range(total):
        frame = frames[i % len(frames)]
        start = now()
        rgb = _bgr_to_contiguous_rgb(frame)
        predict_fn(rgb)
        elapsed_ms = (now() - start) * 1000.0
        if i >= warmup_frames:
            per_frame_ms.append(elapsed_ms)

    mean_ms = statistics.mean(per_frame_ms)
    return LatencyResult(
        name=name,
        warmup_frames=warmup_frames,
        timed_frames=timed_frames,
        mean_ms=mean_ms,
        p95_ms=_percentile(per_frame_ms, 95),
        fps=(1000.0 / mean_ms) if mean_ms > 0 else float("inf"),
        per_frame_ms=per_frame_ms,
    )


def _os_info(
    system: str | None = None,
    getwindowsversion: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """``platform.release()``/``platform.platform()`` report "10" for
    Windows 11 too -- Microsoft kept major.minor at 10.0 and only the
    build number changed (11 starts at build 22000), so
    ``platform.platform()`` alone is not authoritative on Windows
    (ISSUES.md, 2026-09-29). ``windows_build`` (from
    ``sys.getwindowsversion()``) is the authoritative field there;
    ``platform_string`` is kept only for reference. ``system``/
    ``getwindowsversion`` are injectable for tests without a real
    Windows machine."""
    system = system if system is not None else platform.system()
    info: dict[str, Any] = {"system": system, "platform_string": platform.platform()}
    if system == "Windows":
        get_version = getwindowsversion if getwindowsversion is not None else sys.getwindowsversion
        build = get_version().build
        info["windows_build"] = build
        info["windows_release_label"] = "Windows 11" if build >= 22000 else "Windows 10"
    return info


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "cpu_model": platform.processor() or platform.machine(),
        "logical_cpus": os.cpu_count(),
        "os": _os_info(),
        "python_version": platform.python_version(),
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["torch_threads"] = torch.get_num_threads()
        info["torch_cuda_available"] = torch.cuda.is_available()
    except ImportError:
        info["torch_version"] = None
    return info


# ---------------------------------------------------------------------------
# Real candidates -- each opt-in behind --allow-download (see module
# docstring). Never called by this module's own tests.
# ---------------------------------------------------------------------------


def _rfdetr_pytorch_candidates() -> list[tuple[str, Callable[[np.ndarray], object]]]:
    try:
        from rfdetr import RFDETRNano
    except ImportError:
        logger.info("rfdetr not importable; skipping RF-DETR-Nano PyTorch candidates")
        return []

    from contracts import DETECTOR_MIN_CONF  # essential-features.md section 2, step 3

    model = RFDETRNano()  # COCO-pretrained; downloads if not cached (ISSUES.md 2026-09-29)
    plain = ("rfdetr_nano_pytorch", lambda rgb: model.predict(rgb, threshold=DETECTOR_MIN_CONF))

    optimized_model = RFDETRNano()
    # essential-features.md section 2, step 2 calls this
    # optimize_for_inference(); the installed rfdetr (1.11.0) renamed it
    # to inference() -- verified against the installed version's actual
    # signature, not assumed (ISSUES.md, 2026-09-29).
    optimized_model.inference()
    optimized = (
        "rfdetr_nano_pytorch_optimized",
        lambda rgb: optimized_model.predict(rgb, threshold=DETECTOR_MIN_CONF),
    )
    return [plain, optimized]


def _rfdetr_onnx_candidate() -> tuple[str, Callable[[np.ndarray], object]] | None:
    try:
        import onnxruntime  # noqa: F401
        from rfdetr import RFDETRNano
    except ImportError:
        logger.info("onnxruntime and/or rfdetr[onnx] not importable; skipping ONNX candidate")
        return None
    model = RFDETRNano()
    onnx_path = model.export()  # essential-features.md section 2, step 6
    session = onnxruntime.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    return (
        "rfdetr_nano_onnxruntime",
        lambda rgb: session.run(None, {input_name: rgb}),
    )


_YOLO_CACHE_PATH = Path.home() / ".cache" / "ultralytics_benchmark" / "yolo11n.pt"
"""An explicit, absolute, out-of-repo path. A bare "yolo11n.pt" downloads
into the current working directory (confirmed the hard way: it landed at
the repo root, ISSUES.md 2026-09-29) rather than any per-user cache --
never pass a bare relative name to YOLO() from inside this repo."""


def _yolo11n_candidate() -> tuple[str, Callable[[np.ndarray], object]] | None:
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.info("ultralytics not importable; skipping YOLO11n candidate")
        return None
    _YOLO_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(_YOLO_CACHE_PATH))  # COCO-pretrained; downloads if not cached
    return ("yolo11n_pytorch", lambda rgb: model.predict(rgb, verbose=False))


def collect_real_candidates(
    allow_download: bool,
) -> list[tuple[str, Callable[[np.ndarray], object]]]:
    if not allow_download:
        logger.info("--allow-download not set; no real candidate will be constructed")
        return []
    candidates = list(_rfdetr_pytorch_candidates())
    onnx_candidate = _rfdetr_onnx_candidate()
    if onnx_candidate is not None:
        candidates.append(onnx_candidate)
    yolo_candidate = _yolo11n_candidate()
    if yolo_candidate is not None:
        candidates.append(yolo_candidate)
    return candidates


def build_report(
    results: list[LatencyResult],
    warmup_frames: int,
    timed_frames: int,
    frame_width: int,
    frame_height: int,
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "environment": environment_info(),
        "warmup_frames": warmup_frames,
        "timed_frames": timed_frames,
        "frame_size": [frame_width, frame_height],
        "hands": HANDS_NOTE,
        "results": [r.to_report_dict() for r in results],
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP_FRAMES)
    parser.add_argument("--frames", type=int, default=DEFAULT_TIMED_FRAMES)
    parser.add_argument("--width", type=int, default=DEFAULT_FRAME_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_FRAME_HEIGHT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="permit constructing real candidates that may download a COCO-pretrained "
        "checkpoint if one isn't already cached locally (see the module docstring)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/benchmark_cpu.json"),
        help="write results here (use a git-ignored path for provisional runs, never "
        "reports/, which is P2-owned -- see the module docstring)",
    )
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    frames = [
        rng.integers(0, 256, size=(args.height, args.width, 3), dtype=np.uint8)
        for _ in range(8)
    ]

    candidates = collect_real_candidates(args.allow_download)
    if not candidates:
        logger.warning("no candidates available -- nothing to benchmark")
    results = [
        benchmark_predict(name, fn, frames, args.warmup, args.frames) for name, fn in candidates
    ]

    report = build_report(results, args.warmup, args.frames, args.width, args.height)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
