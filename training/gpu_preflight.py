"""F14 stage 6 preflight: what machine is this, and how long will the fine-tune take?

  python -m training.gpu_preflight --model rfdetr|yolo11n --batch-size 4 --n-train N --epochs E
                                   [--require-cuda] [--iterations K] [--weights PATH]
                                   [--device auto|cpu|cuda|cuda:N] [--allow-download]

Prints the GPU name and VRAM, whether CUDA is available, whether the installed torch is a CPU
or a CUDA build, and the torch and rfdetr versions; then times K forward + backward iterations
of the real model at the training micro-batch on synthetic input and extrapolates seconds per
epoch and the total for N training images and E epochs. **The extrapolation is an estimate**:
it is the measured seconds per iteration times the number of micro-batches, and leaves out
validation, data loading, the optimizer step and the loss/matcher (the backward runs on a
surrogate loss over the raw network outputs, which exercises the same kernels and memory but
is not the real criterion). Run it with ``--require-cuda`` on a borrowed GPU: it exits with
code 2 and a clear message if there is no CUDA device (typically a CPU-only torch wheel).

Never downloads: the pretrained weights must already be on disk (the cache the benchmark uses,
or ``--weights``); otherwise it stops and says so. ``--allow-download`` (RF-DETR only) lets
rfdetr fetch its own checkpoint, which needs the Lead's yes (IMPLEMENTATION_PLAN.md R7).
"""

from __future__ import annotations

import argparse
import importlib.metadata
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

MODELS = ("rfdetr", "yolo11n")
RESOLUTION = 384  # RF-DETR-Nano native resolution; the YOLO11n fallback is trained at the same
DEFAULT_CPU_ITERATIONS = 2
DEFAULT_GPU_ITERATIONS = 5
EXCLUDES = "validation, data loading, optimizer step, the matcher/criterion, checkpointing"


class WeightsMissing(Exception):
    """A pretrained weights file is not on disk and downloading was not allowed."""


# --- pure helpers -------------------------------------------------------------------------


def iterations_per_epoch(n_train: int, batch_size: int) -> int:
    """Micro-batches in one epoch (gradient accumulation does not change this count)."""
    if n_train <= 0 or batch_size <= 0:
        raise ValueError(f"n_train and batch_size must be positive, got {n_train}, {batch_size}")
    return math.ceil(n_train / batch_size)


def extrapolate(
    seconds_per_iteration: float, n_train: int, batch_size: int, epochs: int
) -> dict[str, Any]:
    its = iterations_per_epoch(n_train, batch_size)
    per_epoch = seconds_per_iteration * its
    return {
        "is_estimate": True,
        "iterations_per_epoch": its,
        "seconds_per_epoch": per_epoch,
        "total_seconds": per_epoch * epochs,
        "epochs": epochs,
        "n_train": n_train,
        "excludes": EXCLUDES,
    }


def summarise_times(times: list[float]) -> dict[str, Any]:
    """Mean seconds per iteration. With 3 or more samples the first is dropped as warm-up
    (cuDNN autotune, allocator growth); with fewer, everything is kept."""
    if not times:
        raise ValueError("no timings")
    drop = len(times) >= 3
    used = times[1:] if drop else times
    return {
        "seconds_per_iteration": sum(used) / len(used),
        "n_timed": len(used),
        "warmup_dropped": drop,
        "all_seconds": list(times),
    }


def time_iterations(
    step: Callable[[], None],
    n: int,
    *,
    clock: Callable[[], float] = time.perf_counter,
    sync: Callable[[], None] = lambda: None,
) -> list[float]:
    """Time ``n`` calls of ``step``; ``sync`` runs after each so GPU work is included."""
    out = []
    for _ in range(n):
        t0 = clock()
        step()
        sync()
        out.append(clock() - t0)
    return out


def _fmt_duration(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds / 3600:.2f} h"
    if seconds >= 120:
        return f"{seconds / 60:.1f} min"
    return f"{seconds:.1f} s"


def format_report(model: str, info: dict, timing: dict, est: dict, *, batch_size: int) -> str:
    gpu = (
        f"{info['gpu_name']}  ({info['vram_gb']:.2f} GB VRAM)"
        if info["gpu_name"]
        else "no CUDA device"
    )
    build = "CPU build" if info["torch_build"] == "cpu" else f"CUDA build ({info['torch_build']})"
    lines = [
        f"model:            {model}",
        f"device used:      {info['device']}",
        f"GPU:              {gpu}",
        f"CUDA available:   {info['cuda_available']}",
        f"torch:            {info['torch']}  [{build}]",
        f"rfdetr:           {info['rfdetr']}",
        f"timed iterations: {timing['n_timed']} at micro-batch {batch_size}"
        + (" (first dropped as warm-up)" if timing["warmup_dropped"] else ""),
        f"seconds / iteration (measured): {timing['seconds_per_iteration']:.2f}",
        f"-- ESTIMATE for {est['n_train']} images x {est['epochs']} epochs "
        f"({est['iterations_per_epoch']} iterations per epoch) --",
        f"seconds / epoch:  {est['seconds_per_epoch']:.1f}  "
        f"({_fmt_duration(est['seconds_per_epoch'])})",
        f"total:            {est['total_seconds']:.0f} s  ({_fmt_duration(est['total_seconds'])})",
        f"note: an estimate only; it leaves out {est['excludes']}. The backward pass ran on a "
        "surrogate loss over the raw network outputs, not the real criterion.",
    ]
    if info["torch_build"] == "cpu":
        lines.append(
            "WARNING: this torch is a CPU build, so a GPU would not be used even if present; "
            "install the CUDA build from the PyTorch selector (training/README_GPU.md)."
        )
    return "\n".join(lines)


# --- machine and weights ------------------------------------------------------------------


def _version(dist: str) -> str:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def resolve_device(device: str = "auto") -> str:
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def machine_info(device: str = "auto") -> dict[str, Any]:
    import torch

    cuda = torch.cuda.is_available()
    dev = resolve_device(device)
    name = vram = None
    if cuda:
        idx = torch.device(dev).index or 0
        props = torch.cuda.get_device_properties(idx)
        name, vram = props.name, props.total_memory / 1024**3
    return {
        "gpu_name": name,
        "vram_gb": vram,
        "cuda_available": cuda,
        "torch_build": f"cuda {torch.version.cuda}" if torch.version.cuda else "cpu",
        "torch": torch.__version__,
        "rfdetr": _version("rfdetr"),
        "ultralytics": _version("ultralytics"),
        "device": dev,
    }


def default_weights_path(model: str) -> Path:
    """Where the pretrained checkpoint is looked up when ``--weights`` is not given: the same
    local caches ``training.benchmark_cpu`` already uses."""
    if model == "rfdetr":
        from rfdetr.assets.model_weights import get_model_cache_dir

        return Path(get_model_cache_dir()) / "rf-detr-nano.pth"
    if model == "yolo11n":
        return Path.home() / ".cache" / "ultralytics_benchmark" / "yolo11n.pt"
    raise ValueError(f"unknown model {model!r}")


def require_file(path: Path, what: str) -> Path:
    path = Path(path)
    if not path.is_file():
        raise WeightsMissing(
            f"{what} pretrained weights not found at {path}. Nothing is downloaded "
            "automatically: copy the file there (training/README_GPU.md, step A3) or pass "
            "--weights PATH. Downloading needs the Lead's yes."
        )
    return path


def resolve_weights(model: str, explicit: Path | None, allow_download: bool) -> Path:
    if explicit is not None:
        return require_file(explicit, model)
    path = default_weights_path(model)
    if allow_download and model == "rfdetr":
        return path  # rfdetr fetches its own checkpoint into this cache path if absent
    return require_file(path, model)


# --- the real model step ------------------------------------------------------------------


def build_step(
    model: str, batch_size: int, device: str, weights: Path | None, seed: int = 0
) -> Callable[[], None]:
    """A zero-argument callable running one forward + backward at ``batch_size`` on synthetic
    input (BGR/RGB is irrelevant to the cost). The loss is a surrogate over the raw outputs."""
    import torch

    torch.manual_seed(seed)
    x = torch.randn(batch_size, 3, RESOLUTION, RESOLUTION, device=device)
    if model == "rfdetr":
        from rfdetr import RFDETRNano

        kw: dict[str, Any] = {"device": device}
        if weights is not None:
            kw["pretrain_weights"] = str(weights)
        net = RFDETRNano(**kw).model.model.to(device)

        def loss_of(out: Any) -> Any:
            return out["pred_logits"].float().mean() + out["pred_boxes"].float().mean()

    elif model == "yolo11n":
        from ultralytics import YOLO

        path = weights if weights is not None else default_weights_path("yolo11n")
        net = YOLO(str(path)).model.to(device)

        def loss_of(out: Any) -> Any:
            outs = out if isinstance(out, (list, tuple)) else [out]
            return sum(o.float().mean() for o in outs if hasattr(o, "float"))

    else:
        raise ValueError(f"unknown model {model!r}")
    net.train()

    def step() -> None:
        net.zero_grad(set_to_none=True)
        loss_of(net(x)).backward()

    return step


def measure(
    model: str,
    batch_size: int,
    device: str,
    weights: Path | None,
    iterations: int,
    seed: int = 0,
) -> dict[str, Any]:
    import torch

    step = build_step(model, batch_size, device, weights, seed)
    sync = torch.cuda.synchronize if device.startswith("cuda") else (lambda: None)
    return summarise_times(time_iterations(step, iterations, sync=sync))


# --- CLI ----------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--n-train", type=int, required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--require-cuda", action="store_true")
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-download", action="store_true")
    args = ap.parse_args(argv)

    info = machine_info(args.device)
    if args.require_cuda and not info["cuda_available"]:
        why = (
            "the installed torch is a CPU build"
            if info["torch_build"] == "cpu"
            else "no usable NVIDIA driver/device was found"
        )
        print(
            f"--require-cuda: CUDA is not available ({why}; torch {info['torch']}, "
            f"{'CPU build' if info['torch_build'] == 'cpu' else info['torch_build']}). "
            "Install the CUDA build of torch from the PyTorch selector for this machine "
            "(training/README_GPU.md, step A5) and re-run.",
            file=sys.stderr,
        )
        return 2

    try:
        weights = resolve_weights(args.model, args.weights, args.allow_download)
    except WeightsMissing as e:
        print(str(e), file=sys.stderr)
        return 3

    iterations = args.iterations or (
        DEFAULT_GPU_ITERATIONS if info["device"].startswith("cuda") else DEFAULT_CPU_ITERATIONS
    )
    timing = measure(args.model, args.batch_size, info["device"], weights, iterations, args.seed)
    est = extrapolate(timing["seconds_per_iteration"], args.n_train, args.batch_size, args.epochs)
    print(format_report(args.model, info, timing, est, batch_size=args.batch_size))
    return 0


if __name__ == "__main__":
    sys.exit(main())
