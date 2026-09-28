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

    --from-cache RUN_ID   replay data/cache/<run_id>/perception.jsonl
                           (P1.7; not yet built -- this path is exercised
                           once caches exist)
    --video PATH           replay a video file through a live Perception
                           (perception/camera.py, perception/pipeline.py --
                           imported lazily since P1 may not have landed
                           them yet, R3/R6)
    --script PATH          replay a RunScript's performed_steps directly as
                           StateEvents, bypassing StateTracker entirely --
                           the same shortcut engine/reference.py uses --
                           so the full Router (log + speak) can be
                           exercised without needing any PerceptionFrame
                           data at all (e.g. GOLD-1 "through the loop").
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from contracts import (
    EngineEvent,
    ExperimentDefinition,
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


def replay_from_cache(
    experiment: ExperimentDefinition,
    run_id: str,
    cache_dir: str | Path = "data/cache",
    perception_config: PerceptionConfig | None = None,
    runtime_config: RuntimeConfig | None = None,
    log_dir: str | Path = "runs_out/logs",
    speaker: Speaker | None = None,
) -> TEMP_1_ReplayResult:
    perception_config = perception_config or PerceptionConfig()
    runtime_config = runtime_config or RuntimeConfig()
    cache_path = Path(cache_dir) / run_id / "perception.jsonl"
    header, frames = _read_cache(cache_path)
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


def replay_from_video(
    experiment: ExperimentDefinition,
    video_path: str | Path,
    perception_config: PerceptionConfig | None = None,
    runtime_config: RuntimeConfig | None = None,
    log_dir: str | Path = "runs_out/logs",
    speaker: Speaker | None = None,
    run_id: str | None = None,
) -> TEMP_1_ReplayResult:
    """Opens ``video_path`` via P1's ``perception.camera.open_source`` and
    processes it with P1's ``perception.pipeline.PerceptionPipeline`` --
    both imported lazily: as of P2.4 neither has landed (P1.1/P1.6), so
    this path only runs once they exist (R3, sanctioned cross-boundary
    call)."""
    from perception.camera import open_source  # type: ignore[import-not-found]
    from perception.pipeline import PerceptionPipeline  # type: ignore[import-not-found]

    perception_config = perception_config or PerceptionConfig()
    runtime_config = runtime_config or RuntimeConfig()
    run_id = run_id or Path(video_path).stem

    # ``source`` is opened first and everything else wrapped in try/finally
    # from this point on, so a later construction failure (e.g.
    # PerceptionPipeline raising) still closes the already-open source
    # instead of leaking a file/camera handle.
    source = open_source(str(video_path))
    try:
        perception = PerceptionPipeline(perception_config)
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
        while True:
            frame = source.read()
            if frame is None:
                if source.exhausted:
                    break
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--from-cache", metavar="RUN_ID")
    group.add_argument("--video", metavar="PATH")
    group.add_argument("--script", metavar="PATH")
    parser.add_argument(
        "--experiment",
        default="config/experiment.json",
        type=Path,
        help="Path to the ExperimentDefinition (default: config/experiment.json)",
    )
    args = parser.parse_args(argv)

    experiment = ExperimentDefinition.from_json(args.experiment)

    if args.from_cache:
        result = replay_from_cache(experiment, args.from_cache)
    elif args.video:
        try:
            result = replay_from_video(experiment, args.video)
        except ImportError as exc:
            print(f"error: --video needs perception/ (not available yet): {exc}", file=sys.stderr)
            return 1
    else:
        script = RunScript.model_validate_json(Path(args.script).read_text(encoding="utf-8"))
        result = replay_scripted(experiment, script.performed_steps, run_id=script.run_id)

    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
