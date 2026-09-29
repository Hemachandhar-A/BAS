"""training/benchmark_cpu.py -- Stage 8 of the dataset pipeline (F14;
essential-features.md section 14, Stage 8): CPU latency of the detector
candidates on this laptop's CPU -- the "full per-frame path" (BGR -> RGB,
predict), for RF-DETR-Nano under PyTorch (with and without
``optimize_for_inference``, renamed to ``.inference()`` on the installed
rfdetr -- see ISSUES.md 2026-09-29) and under ONNX Runtime, and YOLO11n as
the benchmarked fallback (context.md section 6, ISSUES.md 2026-09-28
"detector: RF-DETR-Nano vs YOLO11n").

    python -m training.benchmark_cpu [--allow-download] [--num-threads N] [--out PATH]

Every row's timed content is the same shape -- resize to the model's
input resolution, ImageNet-style normalize, run the network, decode to
scored boxes -- so the candidates are comparable:
- RF-DETR PyTorch (plain / ``.inference()``-optimized): ``RFDETRNano.predict()``
  does all four stages internally (verified by reading the installed
  rfdetr's ``detr.py`` -- resize via ``torchvision.transforms.functional.resize``
  with ``antialias=False``, ``F.normalize`` with ImageNet mean/std, the
  network forward, then ``PostProcess`` decodes and thresholds).
- RF-DETR ONNX Runtime: the same four stages, done explicitly here by
  reusing rfdetr's own shipped runtime helpers
  (``rfdetr.export._runtime.preprocess.preprocess_to_nchw`` and
  ``rfdetr.export._runtime.decode.decode_detections``) rather than
  reimplementing the resize/normalize/decode math (AGENTS.md rule 1 --
  these are exactly the functions ``rfdetr``'s own ONNX inference helper
  calls; they are documented to be bit-exact with ``predict()``'s
  preprocessing and to mirror ``PostProcess.forward`` exactly). These are
  underscore-prefixed ("private") modules of the installed rfdetr==1.11.0
  and are not a public/stable API guarantee across versions -- re-verify
  on any rfdetr upgrade.
- YOLO11n: ``ultralytics.YOLO.predict()`` likewise does all four stages
  internally.

Hands are never measured here -- that is a separate, still-open path
(no ``.task`` model file is vendored yet; ISSUES.md 2026-09-28 "MediaPipe
model files must be vendored").

**Real candidates are opt-in** (``--allow-download``): constructing
``RFDETRNano()`` or ``ultralytics.YOLO(...)`` auto-downloads a
COCO-pretrained checkpoint on first use if one isn't already cached
locally (outside this repo -- ``~/.roboflow/models`` and an explicit
``~/.cache/ultralytics_benchmark/`` respectively -- a bare relative
``YOLO("yolo11n.pt")`` downloads into the *current working directory*,
confirmed the hard way, ISSUES.md 2026-09-29), which this module treats
as an explicit-permission action (IMPLEMENTATION_PLAN.md R7; the exact
sources, sizes and licenses are logged in ISSUES.md, 2026-09-29). Without
the flag, a missing/unwilling candidate is simply skipped and reported as
unavailable -- nothing is downloaded or installed as a side effect of
running this script. The ONNX candidate additionally needs ``onnxruntime``,
which is not in this project's lockfile at all (an R7 request, ISSUES.md
2026-09-29, open) -- it silently skips itself when unimportable, same as
any other missing candidate.

**Harness controls** (ISSUES.md 2026-09-29 P2 review, "run-order/drift
labelling is weaker than it reads" -- run 2 was faster than run 1 for
every candidate in the first provisional pass, in fixed order, which
confounds *position in the run sequence* with *which candidate ran*):
candidate order is shuffled independently each repeat round (fixed seed,
so the run is reproducible), at least 3 repeats, each repeat re-runs its
own warm-up, a short cooldown separates every timed block, and the
current ``torch.get_num_threads()`` (and, for the ONNX candidate, its
configured ``intra_op_num_threads``) is recorded on every single run, not
just once. The report separates a per-candidate summary (its own repeat
values) from a per-slot "position effect" summary (mean latency at
run-sequence position 1, 2, 3, ... across every candidate that happened
to land there) so drift by position and drift by candidate don't get
tangled together the way they did in the first pass.

This module's own numbers are provisional before P1.5 (weights are
COCO-pretrained, not the fine-tuned head; the class count differs; no
detector is chosen and no ``target_fps`` is set here) -- every report
carries ``"provisional": true`` and a ``"not_measured"`` list, and
``--out`` defaults to a git-ignored local path, never ``reports/``
(P2-owned), until P1.5 actually runs this for real.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import random
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
DEFAULT_REPEATS = 3
DEFAULT_COOLDOWN_S = 1.0
DEFAULT_FRAME_WIDTH = 1280  # contracts.CAPTURE_WIDTH -- essential-features.md section 0
DEFAULT_FRAME_HEIGHT = 720  # contracts.CAPTURE_HEIGHT
DEFAULT_OUT_PATH = Path("data/benchmark_cpu_provisional.json")
"""Never reports/benchmark_cpu.json -- that path is P2-owned and reserved
for P1.5's real run (ISSUES.md 2026-09-29 P2 review, finding 3)."""

HANDS_NOTE = "not measured (no .task model file vendored -- ISSUES.md 2026-09-28)"
NOT_MEASURED = [
    HANDS_NOTE,
    "the fine-tuned detector (COCO-pretrained weights only; different class count "
    "than config/experiment.json's five classes)",
    "real Sample Transfer footage (synthetic random frames only)",
]

_YOLO_CACHE_PATH = Path.home() / ".cache" / "ultralytics_benchmark" / "yolo11n.pt"
"""An explicit, absolute, out-of-repo path. A bare "yolo11n.pt" downloads
into the current working directory (confirmed the hard way: it landed at
the repo root, ISSUES.md 2026-09-29) rather than any per-user cache --
never pass a bare relative name to YOLO() from inside this repo."""

_ONNX_EXPORT_CACHE_DIR = Path.home() / ".cache" / "rfdetr_benchmark"
"""Explicit, absolute, out-of-repo export directory. RFDETRNano.export()'s
own default (output_dir="output") is CWD-relative and would write into
the repo root the same way yolo11n.pt did (ISSUES.md 2026-09-29 P2
review, finding 1) -- never use the default from inside this repo."""


def _bgr_to_contiguous_rgb(image: np.ndarray) -> np.ndarray:
    # The same BGR->RGB conversion the real pipeline pays for at the model
    # boundary (AGENTS.md rule 16). A bare `image[..., ::-1]` view is not
    # C-contiguous and native code can misread it (ISSUES.md, 2026-09-29,
    # the hands.py finding) -- ascontiguousarray, always.
    return np.ascontiguousarray(image[..., ::-1])


def _rgb_to_contiguous_bgr(image: np.ndarray) -> np.ndarray:
    """The inverse of ``_bgr_to_contiguous_rgb``. Ultralytics documents
    ndarray input as BGR (``engine/predictor.py``), but the harness's
    shared timing loop hands every candidate an RGB frame (ISSUES.md
    2026-09-29 P2 review, finding 2) -- the YOLO candidate converts back
    before calling ``predict``. The P1.6 detector wrapper must not copy
    this: RGB is for RF-DETR/MediaPipe only (AGENTS.md rule 16); YOLO is
    not part of that wrapper's real candidate set."""
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
# ONNX <-> PyTorch parity (essential-features.md doesn't cover this; the
# design is this session's own, per the operator's phase_1 instruction).
# Unit-tested against fake arrays now; Phase 2 feeds it real model output.
# ---------------------------------------------------------------------------


@dataclass
class ParityResult:
    max_box_abs_diff: float
    max_logit_abs_diff: float
    within_tolerance: bool
    box_atol: float
    logit_atol: float


def check_raw_output_parity(
    pytorch_boxes: np.ndarray,
    pytorch_logits: np.ndarray,
    onnx_boxes: np.ndarray,
    onnx_logits: np.ndarray,
    box_atol: float = 1e-3,
    logit_atol: float = 1e-2,
) -> ParityResult:
    """Compares the *raw* ``(pred_boxes, pred_logits)`` tensors PyTorch and
    ONNX Runtime each produce for the *same* preprocessed input tensor --
    before decoding -- so a mismatch is attributed to the export/runtime,
    not to this module's decode call (which is the same
    ``rfdetr.export._runtime.decode.decode_detections`` function fed
    either source in Phase 2, so it cannot itself be the source of a
    divergence). ``box_atol``/``logit_atol`` are disclosed judgment
    calls, not sourced -- normalized cxcywh boxes live in a small [0,1]-ish
    range so a tight absolute tolerance is reasonable; raw logits can be
    an order of magnitude larger, hence the looser bound. Phase 2 must
    STOP and report before measuring anything if ``within_tolerance`` is
    False (this session's own instruction)."""
    box_diff = float(np.max(np.abs(pytorch_boxes - onnx_boxes))) if pytorch_boxes.size else 0.0
    logit_diff = float(np.max(np.abs(pytorch_logits - onnx_logits))) if pytorch_logits.size else 0.0
    return ParityResult(
        max_box_abs_diff=box_diff,
        max_logit_abs_diff=logit_diff,
        within_tolerance=(box_diff <= box_atol and logit_diff <= logit_atol),
        box_atol=box_atol,
        logit_atol=logit_atol,
    )


def extract_pytorch_raw_outputs(
    model: Any, preprocessed_tensor: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Phase-2 helper: runs the bare forward pass RFDETR.predict() itself
    dispatches to (verified by reading detr.py's predict(), around its
    ``if self._is_optimized_for_inference: ... else: ...`` branch) on an
    already-preprocessed ``(1, C, H, W)`` tensor, returning
    ``(pred_boxes, pred_logits)`` as numpy arrays with the batch
    dimension squeezed, for `check_raw_output_parity`. Duck-typed against
    RFDETR's documented internal shape (``model._is_optimized_for_inference``,
    ``model.model.model`` / ``model.model.inference_model``) rather than a
    public method -- this is Phase-2-only glue, exercised here only
    against a fake stand-in; re-verify against the real object before
    Phase 2 feeds it a real model."""
    import torch

    with torch.no_grad():
        if model._is_optimized_for_inference:
            output = model.model.inference_model(preprocessed_tensor)
        else:
            output = model.model.model(preprocessed_tensor)
    if isinstance(output, dict):
        boxes, logits = output["pred_boxes"], output["pred_logits"]
    else:
        boxes, logits = output[0], output[1]
    to_numpy = lambda t: t.detach().cpu().numpy()[0]  # noqa: E731
    return to_numpy(boxes), to_numpy(logits)


# ---------------------------------------------------------------------------
# Real candidates -- each opt-in behind --allow-download (see module
# docstring). Never called by this module's own tests.
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    name: str
    predict_fn: Callable[[np.ndarray], object]
    onnx_intra_op_threads: int | None = None


def _rfdetr_pytorch_candidates() -> list[Candidate]:
    try:
        from rfdetr import RFDETRNano
    except ImportError:
        logger.info("rfdetr not importable; skipping RF-DETR-Nano PyTorch candidates")
        return []

    from contracts import DETECTOR_MIN_CONF  # essential-features.md section 2, step 3

    model = RFDETRNano()  # COCO-pretrained; downloads if not cached (ISSUES.md 2026-09-29)
    plain = Candidate(
        "rfdetr_nano_pytorch", lambda rgb: model.predict(rgb, threshold=DETECTOR_MIN_CONF)
    )

    optimized_model = RFDETRNano()
    # essential-features.md section 2, step 2 calls this
    # optimize_for_inference(); the installed rfdetr (1.11.0) renamed it
    # to inference() -- verified against the installed version's actual
    # signature, not assumed (ISSUES.md, 2026-09-29).
    optimized_model.inference()
    optimized = Candidate(
        "rfdetr_nano_pytorch_optimized",
        lambda rgb: optimized_model.predict(rgb, threshold=DETECTOR_MIN_CONF),
    )
    return [plain, optimized]


def _rfdetr_onnx_candidate(num_threads: int) -> Candidate | None:
    try:
        import onnxruntime as ort
        from PIL import Image as PILImage
        from rfdetr import RFDETRNano
        from rfdetr.export._runtime.decode import decode_detections
        from rfdetr.export._runtime.preprocess import preprocess_to_nchw
    except ImportError:
        logger.info(
            "onnxruntime and/or rfdetr's export runtime helpers not importable; "
            "skipping ONNX candidate"
        )
        return None

    from contracts import DETECTOR_MIN_CONF

    _ONNX_EXPORT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model = RFDETRNano()
    # fp16=False: the export default is fp16=True, tuned for GPU tensor
    # cores (context.md section 6's own caveat -- every published RF-DETR
    # figure is GPU/TensorRT). CPU execution providers generally have no
    # fast fp16 kernels, so fp16 would benchmark ONNX Runtime's fp16
    # fallback path, not a fair CPU comparison against the float32
    # PyTorch candidates above.
    onnx_path = model.export(
        output_dir=str(_ONNX_EXPORT_CACHE_DIR), format="onnx", fp16=False, verbose=False
    )

    options = ort.SessionOptions()
    options.intra_op_num_threads = num_threads
    # Matches rfdetr's own _create_onnx_session (export/_onnx/inference.py):
    # spinning contends for CPU with this process's own preprocessing.
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    session = ort.InferenceSession(
        str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"]
    )

    input_meta = session.get_inputs()[0]
    _, channels, height, width = input_meta.shape
    input_name = input_meta.name
    output_names = [o.name for o in session.get_outputs()]
    boxes_idx = next(i for i, n in enumerate(output_names) if "dets" in n)
    logits_idx = next(i for i, n in enumerate(output_names) if "labels" in n)

    def predict_fn(rgb: np.ndarray) -> object:
        pil_img = PILImage.fromarray(rgb)
        tensor = preprocess_to_nchw(pil_img, height, width, channels)
        raw = session.run(None, {input_name: tensor})
        boxes_cwh, logits = raw[boxes_idx][0], raw[logits_idx][0]
        # background_class_id=None: this is a sparse COCO-pretrained
        # checkpoint (90 categories via a non-contiguous id table, no
        # reserved background slot to exclude) -- per decode_detections'
        # own docstring ("Pass None for sparse COCO checkpoints").
        return decode_detections(
            boxes_cwh,
            logits,
            pil_img.size,
            threshold=DETECTOR_MIN_CONF,
            background_class_id=None,
        )

    return Candidate("rfdetr_nano_onnxruntime", predict_fn, onnx_intra_op_threads=num_threads)


def _yolo11n_candidate() -> Candidate | None:
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.info("ultralytics not importable; skipping YOLO11n candidate")
        return None
    _YOLO_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(_YOLO_CACHE_PATH))  # COCO-pretrained; downloads if not cached

    def predict_fn(rgb: np.ndarray) -> object:
        # Ultralytics expects BGR ndarray input; the shared timing loop
        # hands every candidate RGB (ISSUES.md 2026-09-29 P2 review,
        # finding 2). Convert back rather than leave it wrong -- cheap,
        # and correctness here doesn't cost the other candidates anything.
        return model.predict(_rgb_to_contiguous_bgr(rgb), verbose=False)

    return Candidate("yolo11n_pytorch", predict_fn)


def collect_real_candidates(allow_download: bool, num_threads: int) -> list[Candidate]:
    if not allow_download:
        logger.info("--allow-download not set; no real candidate will be constructed")
        return []
    candidates = list(_rfdetr_pytorch_candidates())
    onnx_candidate = _rfdetr_onnx_candidate(num_threads)
    if onnx_candidate is not None:
        candidates.append(onnx_candidate)
    yolo_candidate = _yolo11n_candidate()
    if yolo_candidate is not None:
        candidates.append(yolo_candidate)
    return candidates


# ---------------------------------------------------------------------------
# Harness controls: shuffled repeats, per-run thread recording, a
# position-effect summary separate from the per-candidate summary
# (ISSUES.md 2026-09-29 P2 review, finding 4).
# ---------------------------------------------------------------------------


def run_suite(
    candidates: list[Candidate],
    frames: list[np.ndarray],
    warmup_frames: int,
    timed_frames: int,
    repeats: int,
    seed: int,
    cooldown_s: float,
    num_threads: int,
) -> list[dict[str, Any]]:
    """Runs every candidate ``repeats`` times. Each repeat round visits
    the candidates in a freshly shuffled order (one `random.Random(seed)`
    advanced across rounds, so the whole run is reproducible from
    `seed`), so no candidate is systematically "always first" or "always
    last" the way the first provisional pass was (RF-DETR first, YOLO
    last, every run). ``torch.set_num_threads(num_threads)`` is
    (re-)asserted before every single run, not just once, since importing
    `ultralytics` was observed to silently change it mid-session
    (ISSUES.md 2026-09-29). Returns one record per (round, candidate)
    run, in the order actually executed.
    """
    try:
        import torch
    except ImportError:
        torch = None  # pragma: no cover -- exercised only with a fake torch in tests

    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    global_position = 0
    for round_idx in range(1, repeats + 1):
        order = list(range(len(candidates)))
        rng.shuffle(order)
        for slot, idx in enumerate(order, start=1):
            candidate = candidates[idx]
            global_position += 1
            if torch is not None:
                torch.set_num_threads(num_threads)
                threads_now = torch.get_num_threads()
            else:
                threads_now = None
            result = benchmark_predict(
                candidate.name, candidate.predict_fn, frames, warmup_frames, timed_frames
            )
            records.append(
                {
                    "candidate": candidate.name,
                    "round": round_idx,
                    "slot": slot,
                    "global_position": global_position,
                    "torch_threads": threads_now,
                    "onnx_intra_op_threads": candidate.onnx_intra_op_threads,
                    **result.to_report_dict(),
                }
            )
            if cooldown_s > 0 and global_position < repeats * len(candidates):
                time.sleep(cooldown_s)
    return records


def summarize_by_candidate(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_name: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_name.setdefault(r["candidate"], []).append(r)
    summary: dict[str, Any] = {}
    for name, rows in by_name.items():
        means = [r["mean_ms"] for r in rows]
        summary[name] = {
            "repeats": len(rows),
            "mean_ms_per_repeat": means,
            "mean_of_means_ms": statistics.mean(means),
            "spread_ms": max(means) - min(means),
            "fps_per_repeat": [r["fps"] for r in rows],
        }
    return summary


def summarize_by_position(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean latency at each in-round slot (1st candidate run that round,
    2nd, ...), pooled across every candidate that happened to land there
    -- since candidate order is shuffled per round, a slot isn't tied to
    one candidate, so a trend here points at warm-up/thermal drift by
    position rather than a slow candidate."""
    by_slot: dict[int, list[float]] = {}
    for r in records:
        by_slot.setdefault(r["slot"], []).append(r["mean_ms"])
    return {
        str(slot): {"mean_ms_avg": statistics.mean(values), "n": len(values)}
        for slot, values in sorted(by_slot.items())
    }


def build_report(
    records: list[dict[str, Any]],
    warmup_frames: int,
    timed_frames: int,
    frame_width: int,
    frame_height: int,
    repeats: int,
    seed: int,
    cooldown_s: float,
    num_threads: int,
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "provisional": True,
        "environment": environment_info(),
        "controls": {
            "repeats": repeats,
            "seed": seed,
            "cooldown_s": cooldown_s,
            "num_threads_requested": num_threads,
            "candidate_order": "shuffled independently each repeat round, fixed seed",
        },
        "warmup_frames": warmup_frames,
        "timed_frames": timed_frames,
        "frame_size": [frame_width, frame_height],
        "not_measured": list(NOT_MEASURED),
        "per_run": records,
        "per_candidate_summary": summarize_by_candidate(records),
        "position_effect": summarize_by_position(records),
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP_FRAMES)
    parser.add_argument("--frames", type=int, default=DEFAULT_TIMED_FRAMES)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cooldown", type=float, default=DEFAULT_COOLDOWN_S)
    parser.add_argument(
        "--num-threads",
        type=int,
        default=os.cpu_count() or 1,
        help="torch.set_num_threads / ONNX Runtime intra_op_num_threads for this run. "
        "Run once with the physical core count and once with the logical count "
        "(this default) -- see the module docstring.",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_FRAME_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_FRAME_HEIGHT)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="permit constructing real candidates that may download a COCO-pretrained "
        "checkpoint if one isn't already cached locally (see the module docstring)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_PATH,
        help="write results here -- a git-ignored path by default; never point this at "
        "reports/, which is P2-owned (see the module docstring)",
    )
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    frames = [
        rng.integers(0, 256, size=(args.height, args.width, 3), dtype=np.uint8) for _ in range(8)
    ]

    candidates = collect_real_candidates(args.allow_download, args.num_threads)
    if not candidates:
        logger.warning("no candidates available -- nothing to benchmark")
    records = run_suite(
        candidates,
        frames,
        args.warmup,
        args.frames,
        args.repeats,
        args.seed,
        args.cooldown,
        args.num_threads,
    )

    report = build_report(
        records,
        args.warmup,
        args.frames,
        args.width,
        args.height,
        args.repeats,
        args.seed,
        args.cooldown,
        args.num_threads,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
