"""perception/cache.py -- perception caches (F14 stage 9, IMPLEMENTATION_PLAN.md 5.2). Owner: P1.

A cache is ``<root>/<run_id>/perception.jsonl``: line 1 a ``PerceptionCacheHeader``, every later
line a ``PerceptionFrame``. It is valid only while its ``model_stamp`` (and ``fps``) equal what
the reader expects; the reader refuses anything else with a message that shows both values.
Files are written to a temporary name and renamed, so a file that exists is a complete one and
an interrupted build never leaves half a cache. Bytes are deterministic (Unix newlines, the
pydantic JSON encoder), so the same video and weights rebuild an identical file.

  python -m perception.cache build [--split train|val|test|all] [--run ID] [--fps F]
        [--detector NAME] [--force] [--out-root data/cache] [--runs-dir runs] [--log PATH]

A run whose cache already has the expected header and an intact last line is skipped (resume);
``--force`` rebuilds. A frame that fails in perception is logged with its id and skipped; a run
whose video cannot be opened is reported and the others go on. This module reads no clock: the
timing and the report of a whole build live in ``training/cache_report.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from contracts import CAPTURE_FPS, PerceptionCacheHeader, PerceptionFrame, SourceError

logger = logging.getLogger(__name__)

DEFAULT_CACHE_ROOT = Path("data/cache")
DEFAULT_RUNS_DIR = Path("runs")
DEFAULT_TARGET_FPS = 10.0
CACHE_NAME = "perception.jsonl"
SPLITS = ("train", "val", "test", "all")


class CacheError(Exception):
    """A cache file is missing or malformed."""


class CacheMismatch(CacheError):
    """The cache is well formed but was built with a different model_stamp or fps."""


def cache_path(root: Path | str, run_id: str) -> Path:
    return Path(root) / run_id / CACHE_NAME


# --- writer ---------------------------------------------------------------------------------


@contextmanager
def _atomic_text(path: Path) -> Iterator[Any]:
    """Write to ``<path>.tmp``; rename over ``path`` only if the block finished."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as f:
            yield f
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_cache(
    path: Path | str, hdr: PerceptionCacheHeader, frames: Iterable[PerceptionFrame]
) -> int:
    """Header line, then one line per frame, atomically. Returns the number of frames."""
    n = 0
    with _atomic_text(Path(path)) as f:
        f.write(hdr.model_dump_json() + "\n")
        for fr in frames:
            f.write(fr.model_dump_json() + "\n")
            n += 1
    return n


# --- reader ---------------------------------------------------------------------------------


def _check_header(
    hdr: PerceptionCacheHeader, path: Path, expected_stamp: str | None, expected_fps: float | None
) -> None:
    if expected_stamp is not None and hdr.model_stamp != expected_stamp:
        raise CacheMismatch(
            f"{path}: cache model_stamp {hdr.model_stamp!r} != expected {expected_stamp!r}; "
            "rebuild the cache with this detector and weights"
        )
    if expected_fps is not None and not math.isclose(hdr.fps, expected_fps, rel_tol=1e-9):
        raise CacheMismatch(f"{path}: cache fps {hdr.fps} != expected fps {expected_fps}")


def _parse_header(line: str, path: Path) -> PerceptionCacheHeader:
    if not line.strip():
        raise CacheError(f"{path}: empty file (no header line)")
    try:
        return PerceptionCacheHeader.model_validate_json(line)
    except ValidationError as e:
        raise CacheError(f"{path}: line 1 is not a valid cache header: {e}") from e


def read_header(
    path: Path | str, *, expected_stamp: str | None = None, expected_fps: float | None = None
) -> PerceptionCacheHeader:
    p = Path(path)
    if not p.is_file():
        raise CacheError(f"{p}: cache not found")
    with p.open(encoding="utf-8", newline="\n") as f:
        hdr = _parse_header(f.readline(), p)
    _check_header(hdr, p, expected_stamp, expected_fps)
    return hdr


def read_cache(
    path: Path | str, *, expected_stamp: str | None = None, expected_fps: float | None = None
) -> tuple[PerceptionCacheHeader, list[PerceptionFrame]]:
    """Validate the header (and refuse a stamp or fps that differs from what the caller expects)
    and return it with every frame. Frame ids must strictly increase."""
    p = Path(path)
    hdr = read_header(p, expected_stamp=expected_stamp, expected_fps=expected_fps)
    frames: list[PerceptionFrame] = []
    with p.open(encoding="utf-8", newline="\n") as f:
        f.readline()
        for n, line in enumerate(f, start=2):
            if not line.strip():
                continue
            try:
                fr = PerceptionFrame.model_validate_json(line)
            except ValidationError as e:
                raise CacheError(f"{p}: line {n} is not a valid PerceptionFrame: {e}") from e
            if frames and fr.frame_id <= frames[-1].frame_id:
                raise CacheError(f"{p}: line {n}: frame ids must increase ({fr.frame_id})")
            frames.append(fr)
    return hdr, frames


def is_complete(path: Path | str, stamp: str, fps: float) -> bool:
    """True when the file exists with the expected header and an intact last line. Because
    files are renamed into place only when finished, this is a cheap resume check."""
    p = Path(path)
    try:
        read_header(p, expected_stamp=stamp, expected_fps=fps)
        data = p.read_bytes()
    except (CacheError, OSError):
        return False
    if not data.endswith(b"\n"):
        return False
    lines = data.splitlines()
    if len(lines) < 2:
        return True  # header only: a run with no frames
    try:
        PerceptionFrame.model_validate_json(lines[-1])
    except ValidationError:
        return False
    return True


# --- building -------------------------------------------------------------------------------


def _every(source_fps: float | None, target_fps: float) -> int:
    return max(1, round((source_fps or float(CAPTURE_FPS)) / target_fps))


def build_run(
    run_id: str,
    video: Path | str,
    pipeline: Any,
    *,
    fps: float,
    experiment_id: str,
    out_root: Path | str = DEFAULT_CACHE_ROOT,
    force: bool = False,
    source_factory: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Build one run's cache. The source is opened with ``open_source`` and wrapped in
    ``DecimatedSource`` (every ``round(source_fps / fps)``-th frame; the frame ids and ``t``
    stay those of the recording). Returns a result dict with ``status`` ``built`` or
    ``skipped``."""
    if not fps > 0:
        raise ValueError(f"fps must be > 0, got {fps!r}")
    from perception.camera import DecimatedSource, open_source

    out = cache_path(out_root, run_id)
    stamp = pipeline.model_stamp
    if not force and is_complete(out, stamp, fps):
        logger.info("run %s: cache is complete for this stamp and fps, skipping", run_id)
        hdr, frames = read_cache(out)
        return {"run_id": run_id, "status": "skipped", "frames": len(frames), "path": str(out)}

    src = (source_factory or open_source)(str(video))
    try:
        every = _every(src.fps, fps)
        dec = DecimatedSource(src, every)
        pipeline.reset()  # hand-landmarker timestamps restart for every run
        stats = {"skipped_frames": 0}

        def frames() -> Iterator[PerceptionFrame]:
            while (fr := dec.read()) is not None:
                try:
                    out_frame = pipeline.process(fr)
                except Exception:
                    stats["skipped_frames"] += 1
                    logger.warning("run %s: skipping frame %d", run_id, fr.frame_id, exc_info=True)
                    continue
                yield out_frame

        hdr = PerceptionCacheHeader(
            run_id=run_id, experiment_id=experiment_id, fps=fps, model_stamp=stamp
        )
        n = write_cache(out, hdr, frames())
    finally:
        src.close()
    logger.info(
        "run %s: %d frames written (every %d), %d skipped",
        run_id,
        n,
        every,
        stats["skipped_frames"],
    )
    return {
        "run_id": run_id,
        "status": "built",
        "frames": n,
        "skipped_frames": stats["skipped_frames"],
        "every": every,
        "source_fps": src.fps,
        "path": str(out),
    }


def build_runs(
    runs: Iterable[tuple[str, Path]],
    pipeline: Any,
    *,
    fps: float,
    experiment_id: str,
    out_root: Path | str = DEFAULT_CACHE_ROOT,
    force: bool = False,
    source_factory: Callable[[str], Any] | None = None,
    on_run: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """``build_run`` for each ``(run_id, video)``. A run that cannot be built is recorded as
    ``failed`` with its error and the rest continue. ``on_run`` sees each result as it ends."""
    results = []
    for run_id, video in runs:
        try:
            r = build_run(
                run_id, video, pipeline, fps=fps, experiment_id=experiment_id,
                out_root=out_root, force=force, source_factory=source_factory,
            )  # fmt: skip
        except (SourceError, OSError) as e:
            logger.error("run %s: cannot build: %s", run_id, e)
            r = {"run_id": run_id, "status": "failed", "error": str(e)}
        results.append(r)
        if on_run is not None:
            on_run(r)
    return results


def select_runs(
    manifest_csv: Path | str,
    runs_dir: Path | str = DEFAULT_RUNS_DIR,
    *,
    split: str = "all",
    run: str | None = None,
) -> list[tuple[str, Path]]:
    """``(run_id, video path)`` from ``runs/manifest.csv``, in manifest order."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    with Path(manifest_csv).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if run is not None:
        rows = [r for r in rows if r["run_id"] == run]
        if not rows:
            raise ValueError(f"run {run!r} is not in {manifest_csv}")
    elif split != "all":
        rows = [r for r in rows if r["split"] == split]
    return [(r["run_id"], Path(runs_dir) / r["run_id"] / "video.mp4") for r in rows]


# --- CLI ------------------------------------------------------------------------------------


def experiment_id(config_path: Path | str = Path("config/experiment.json")) -> str:
    return str(json.loads(Path(config_path).read_text(encoding="utf-8"))["experiment_id"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m perception.cache", description=__doc__.splitlines()[0]
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build perception caches for recorded runs")
    b.add_argument("--split", choices=SPLITS, default="all")
    b.add_argument("--run", default=None, help="one run id (overrides --split)")
    b.add_argument("--fps", type=float, default=DEFAULT_TARGET_FPS)
    b.add_argument(
        "--detector", default=None, help="manifest name; else $SIH_DETECTOR, else active"
    )
    b.add_argument("--force", action="store_true")
    b.add_argument("--out-root", type=Path, default=DEFAULT_CACHE_ROOT)
    b.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    b.add_argument("--manifest", type=Path, default=None, help="default <runs-dir>/manifest.csv")
    b.add_argument("--log", type=Path, default=None, help="also write the log to this file")
    args = ap.parse_args(argv)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        handlers=handlers,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from perception.pipeline import load_pipeline

    runs = select_runs(
        args.manifest or args.runs_dir / "manifest.csv",
        args.runs_dir,
        split=args.split,
        run=args.run,
    )
    pipeline = load_pipeline(detector=args.detector)
    logger.info(
        "model_stamp %s; %d run(s) at %.3f fps into %s",
        pipeline.model_stamp,
        len(runs),
        args.fps,
        args.out_root,
    )
    results = build_runs(
        runs, pipeline, fps=args.fps, experiment_id=experiment_id(),
        out_root=args.out_root, force=args.force,
    )  # fmt: skip
    failed = [r["run_id"] for r in results if r["status"] == "failed"]
    logger.info(
        "done: %d built, %d skipped, %d failed %s",
        sum(r["status"] == "built" for r in results),
        sum(r["status"] == "skipped" for r in results),
        len(failed),
        failed or "",
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
