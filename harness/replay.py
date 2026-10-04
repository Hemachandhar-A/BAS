"""harness/replay.py -- replay a recorded/cached/scripted run through
StateTracker + SequenceEngine + Router (the same routing runtime/loop.py
uses live), producing a ``TEMP_1_ReplayResult``.

CONTRACT gap (ISSUES.md, 2026-09-28 P2.4): IMPLEMENTATION_PLAN.md 5.9 and
Part 10 both name ``ReplayResult`` as this function's return type, but
``contracts.py`` does not define it. Per AGENTS.md rule 3, this module
works against a clearly prefixed local stub (``TEMP_1_ReplayResult``)
instead of guessing a shape into ``contracts.py``; a human should confirm
its final fields in a future contract PR (likely P2.6/P2.7, once tuning
and test-split reporting know what they need from it).

File replay has no threads: it reads sequentially and never sleeps
(essential-features.md section 0). Three modes, matching the command
surface (IMPLEMENTATION_PLAN.md 5.9):

    --from-cache RUN_ID   replay data/cache/<run_id>/perception.jsonl; refuses a
                           cache whose header fps differs from the configured
                           target_fps or whose model_stamp differs from the
                           active detector's expected stamp
    --video PATH           replay a video file through the real pipeline built
                           by perception.pipeline.load_pipeline() (detector
                           chosen by SIH_DETECTOR, then the manifest), with the
                           same frame decimation to target_fps the cache
                           builder uses (R3/R6)
    --script PATH          replay a RunScript's performed_steps directly as
                           StateEvents, bypassing StateTracker entirely --
                           the same shortcut engine/reference.py uses --
                           so the full Router (log + speak) can be
                           exercised without needing any PerceptionFrame
                           data at all (e.g. GOLD-1 "through the loop").
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from contracts import (
    CAPTURE_FPS,
    EngineEvent,
    ExperimentDefinition,
    Perception,
    PerceptionCacheHeader,
    PerceptionConfig,
    PerceptionFrame,
    RunScript,
    RunSummary,
    RuntimeConfig,
    Speaker,
    StateEvent,
)
from engine.sequence import SequenceEngine
from harness.settings import expected_model_stamp, load_perception_config, load_runtime_config
from outputs.tts import FakeSpeaker
from runtime.loop import Router
from state.tracker import StateTracker


class TEMP_1_ReplayResult(BaseModel):
    """Local stub standing in for the ``ReplayResult`` type
    IMPLEMENTATION_PLAN.md refers to but ``contracts.py`` does not yet
    define (see the module docstring's CONTRACT note)."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    run_id: str
    frames_processed: int
    engine_events: list[EngineEvent]
    summary: RunSummary | None


def _summary_of(events: list[EngineEvent]) -> RunSummary | None:
    return next((e.summary for e in events if e.kind == "run_completed"), None)


def replay_scripted(
    experiment: ExperimentDefinition,
    performed_steps: list[str],
    runtime_config: RuntimeConfig | None = None,
    run_id: str = "scripted-replay",
    log_dir: str | Path = "runs_out/logs",
    speaker: Speaker | None = None,
) -> TEMP_1_ReplayResult:
    """Bypasses ``StateTracker`` entirely: ``performed_steps`` are fed
    straight to the ``Engine`` as ``StateEvent``s (``confidence=1.0``,
    ``uncertain=False``), exactly like ``engine/reference.py``'s
    ``derive_expected_deviations`` -- but through the full ``Router`` so
    logging and speaking are exercised too."""
    runtime_config = runtime_config or RuntimeConfig()
    engine = SequenceEngine(experiment, runtime_config)
    speaker = speaker or FakeSpeaker()
    events: list[EngineEvent] = []
    router = Router(
        experiment,
        runtime_config,
        tracker=None,
        engine=engine,
        speaker=speaker,
        log_dir=log_dir,
        run_id_factory=lambda: run_id,
        on_engine_event=events.append,
    )
    router.start(t=0.0)
    for i, step_id in enumerate(performed_steps, start=1):
        router.process_state_event(
            StateEvent(t=float(i), step_id=step_id, confidence=1.0, uncertain=False)
        )
    return TEMP_1_ReplayResult(
        run_id=run_id,
        frames_processed=len(performed_steps),
        engine_events=events,
        summary=_summary_of(events),
    )


def _read_cache(path: Path) -> tuple[PerceptionCacheHeader, list[PerceptionFrame]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"empty perception cache: {path}")
    header = PerceptionCacheHeader.model_validate_json(lines[0])
    frames = [PerceptionFrame.model_validate_json(line) for line in lines[1:] if line.strip()]
    return header, frames


def _check_cache_header(
    header: PerceptionCacheHeader,
    path: Path,
    expected_stamp: str | None,
    expected_fps: float | None,
) -> None:
    from perception.cache import CacheMismatch  # P1's exception type (sanctioned call)

    if expected_stamp is not None and header.model_stamp != expected_stamp:
        raise CacheMismatch(
            f"{path}: cache model_stamp {header.model_stamp!r} != expected {expected_stamp!r}; "
            "rebuild the cache with this detector and weights"
        )
    if expected_fps is not None and not math.isclose(header.fps, expected_fps, rel_tol=1e-9):
        raise CacheMismatch(
            f"{path}: cache fps {header.fps} != configured target_fps {expected_fps}"
        )


def replay_from_cache(
    experiment: ExperimentDefinition,
    run_id: str,
    cache_dir: str | Path = "data/cache",
    perception_config: PerceptionConfig | None = None,
    runtime_config: RuntimeConfig | None = None,
    log_dir: str | Path = "runs_out/logs",
    speaker: Speaker | None = None,
    expected_stamp: str | None = None,
    expected_fps: float | None = None,
    max_frames: int | None = None,
) -> TEMP_1_ReplayResult:
    """``max_frames`` keeps only the first N cached frames. ``expected_stamp`` and
    ``expected_fps`` (when given, as ``main`` always does) make the replay refuse a cache built
    with another ``model_stamp`` or at another fps (``CacheMismatch``)."""
    perception_config = perception_config or PerceptionConfig()
    runtime_config = runtime_config or RuntimeConfig()
    cache_path = Path(cache_dir) / run_id / "perception.jsonl"
    header, frames = _read_cache(cache_path)
    _check_cache_header(header, cache_path, expected_stamp, expected_fps)
    if max_frames is not None:
        frames = frames[:max_frames]
    if header.experiment_id != experiment.experiment_id:
        raise ValueError(
            f"cache {cache_path} was built for experiment {header.experiment_id!r}, "
            f"not {experiment.experiment_id!r}"
        )
    tracker = StateTracker(experiment, perception_config)
    engine = SequenceEngine(experiment, runtime_config)
    speaker = speaker or FakeSpeaker()
    events: list[EngineEvent] = []
    router = Router(
        experiment,
        runtime_config,
        tracker,
        engine,
        speaker,
        log_dir,
        run_id_factory=lambda: run_id,
        on_engine_event=events.append,
    )
    router.start(t=frames[0].t if frames else 0.0)
    for frame in frames:
        router.process_perception_frame(frame)
    return TEMP_1_ReplayResult(
        run_id=run_id,
        frames_processed=len(frames),
        engine_events=events,
        summary=_summary_of(events),
    )




def decimation_every(source_fps: float | None, target_fps: float) -> int:
    """The rule ``perception.cache.build_run`` uses: every ``round(source_fps / target_fps)``-th
    frame of the recording, never fewer than one (an unknown source fps counts as the capture
    rate), so a ``--video`` replay sees the frames the cache holds."""
    return max(1, round((source_fps or float(CAPTURE_FPS)) / target_fps))


def replay_from_video(
    experiment: ExperimentDefinition,
    video_path: str | Path,
    perception_config: PerceptionConfig | None = None,
    runtime_config: RuntimeConfig | None = None,
    log_dir: str | Path = "runs_out/logs",
    speaker: Speaker | None = None,
    run_id: str | None = None,
    pipeline_factory: Callable[[], Perception] | None = None,
    max_frames: int | None = None,
) -> TEMP_1_ReplayResult:
    """Opens ``video_path`` via ``perception.camera.open_source``, keeps every
    ``decimation_every``-th frame (frame ids and ``t`` stay those of the recording, as in the
    cache) and processes it with the pipeline from ``pipeline_factory`` -- by default
    ``perception.pipeline.load_pipeline()``, so the detector is chosen like at launch
    (``$SIH_DETECTOR``, then the manifest). ``max_frames`` stops after that many kept frames.
    Perception modules are imported lazily (R3, sanctioned cross-boundary call)."""
    from perception.camera import open_source  # type: ignore[import-not-found]

    if pipeline_factory is None:
        from perception.pipeline import load_pipeline  # type: ignore[import-not-found]

        pipeline_factory = load_pipeline

    perception_config = perception_config or PerceptionConfig()
    runtime_config = runtime_config or RuntimeConfig()
    run_id = run_id or Path(video_path).stem

    # ``source`` is opened first and everything else wrapped in try/finally
    # from this point on, so a later construction failure (e.g. the pipeline
    # raising) still closes the already-open source instead of leaking a
    # file/camera handle.
    source = open_source(str(video_path))
    try:
        every = decimation_every(source.fps, runtime_config.target_fps)
        perception = pipeline_factory()
        perception.reset()  # hand-landmarker timestamps restart, as in the cache builder
        tracker = StateTracker(experiment, perception_config)
        engine = SequenceEngine(experiment, runtime_config)
        speaker = speaker or FakeSpeaker()
        events: list[EngineEvent] = []
        router = Router(
            experiment,
            runtime_config,
            tracker,
            engine,
            speaker,
            log_dir,
            run_id_factory=lambda: run_id,
            on_engine_event=events.append,
        )
        frames_processed = 0
        router.start(t=0.0)
        while max_frames is None or frames_processed < max_frames:
            frame = source.read()
            if frame is None:
                if source.exhausted:
                    break
                continue
            if frame.frame_id % every != 0:
                continue
            frames_processed += 1
            try:
                pframe = perception.process(frame)
            except Exception:
                # Matches runtime/loop.py's live behavior (essential-features.md
                # section 0, "Errors"): a single bad frame is skipped, not fatal
                # to the whole replay.
                continue
            router.process_perception_frame(pframe)
    finally:
        source.close()
    return TEMP_1_ReplayResult(
        run_id=run_id,
        frames_processed=frames_processed,
        engine_events=events,
        summary=_summary_of(events),
    )


ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--from-cache", metavar="RUN_ID")
    group.add_argument("--video", metavar="PATH")
    group.add_argument("--script", metavar="PATH")
    group.add_argument(
        "--tune",
        action="store_true",
        help="P2.6 sweep of the PerceptionConfig values on the val split only",
    )
    parser.add_argument(
        "--experiment",
        default="config/experiment.json",
        type=Path,
        help="Path to the ExperimentDefinition (default: config/experiment.json)",
    )
    parser.add_argument("--runtime-config", default=ROOT / "config" / "runtime.yaml", type=Path)
    parser.add_argument(
        "--perception-config", default=ROOT / "config" / "perception.yaml", type=Path
    )
    parser.add_argument("--manifest", default=ROOT / "weights" / "MANIFEST.json", type=Path)
    parser.add_argument("--cache-dir", default="data/cache", type=Path)
    parser.add_argument("--max-frames", type=int, default=None, help="--video: stop after N frames")
    parser.add_argument(
        "--split",
        default="val",
        help="--tune: must be val (any other split is refused); "
        "--from-cache all --split test: the one-shot test replay (harness/heldout.py)",
    )
    parser.add_argument(
        "--allow-repeat-test",
        metavar="REASON",
        default=None,
        help="--split test: allow a second test replay although the report or the marker exists; "
        "the reason is recorded in the new report",
    )
    parser.add_argument("--report", default=ROOT / "reports" / "replay_test.json", type=Path)
    parser.add_argument(
        "--marker",
        default=ROOT / "reports" / ".replay_test_done",
        type=Path,
        help="--split test: the fixed marker the repeat guard checks besides the report",
    )
    parser.add_argument("--acceptance", default=ROOT / "config" / "acceptance.yaml", type=Path)
    args = parser.parse_args(argv)

    if args.tune:
        from harness.tune import tune_main

        return tune_main(args)

    if args.split == "test":
        from harness.heldout import run_test_split

        if args.from_cache != "all":
            print("error: --split test needs --from-cache all", file=sys.stderr)
            return 2
        return run_test_split(args)

    experiment = ExperimentDefinition.from_json(args.experiment)
    runtime_config = load_runtime_config(args.runtime_config)
    perception_config = (
        load_perception_config(args.perception_config)
        if Path(args.perception_config).is_file()
        else PerceptionConfig()
    )

    if args.from_cache:
        from perception.cache import CacheMismatch

        try:
            result = replay_from_cache(
                experiment,
                args.from_cache,
                cache_dir=args.cache_dir,
                perception_config=perception_config,
                runtime_config=runtime_config,
                expected_stamp=expected_model_stamp(args.manifest),
                expected_fps=runtime_config.target_fps,
                max_frames=args.max_frames,
            )
        except CacheMismatch as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    elif args.video:
        try:
            result = replay_from_video(
                experiment,
                args.video,
                perception_config=perception_config,
                runtime_config=runtime_config,
                max_frames=args.max_frames,
            )
        except ImportError as exc:
            print(f"error: --video needs perception/ (not available): {exc}", file=sys.stderr)
            return 1
    else:
        script = RunScript.model_validate_json(Path(args.script).read_text(encoding="utf-8"))
        result = replay_scripted(
            experiment, script.performed_steps, runtime_config, run_id=script.run_id
        )

    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
