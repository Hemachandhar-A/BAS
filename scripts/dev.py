#!/usr/bin/env python3
"""python scripts/dev.py -- run the live system against the webcam, served
by the dashboard (IMPLEMENTATION_PLAN.md 5.9).

Requires perception/camera.py and perception/pipeline.py (P1.1/P1.6);
until those land this exits with a clear message instead of a bare
ImportError traceback (R3: the sanctioned cross-boundary call, not owned
by this session).

Wires FrameSource (webcam) -> Perception -> StateTracker -> SequenceEngine
-> Router (JsonlLogger + TTSWorker) -> RuntimeLoop, starts the capture and
inference threads, and serves server/app.py's Flask dashboard (F10, F12).
The operator starts/resets the run from the dashboard (POST /api/run/start,
/api/run/reset) -- this script no longer starts one automatically, now that
a server exists to hit those routes (P2.5). Runs until Ctrl+C.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from contracts import ExperimentDefinition, PerceptionConfig, RuntimeConfig
from engine.sequence import SequenceEngine
from outputs.tts import TTSWorker
from runtime.loop import Router, RuntimeLoop
from server.app import RecentAlerts, create_app, resolve_bind_host, run_app
from server.env import load_env_file, stream_credentials
from state.tracker import StateTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> int:
    try:
        from perception.pipeline import PerceptionPipeline

        from perception.camera import open_source
    except ImportError as exc:
        print(
            "scripts/dev.py needs perception/camera.py and perception/pipeline.py "
            f"(P1.1/P1.6), which are not available yet: {exc}",
            file=sys.stderr,
        )
        return 1

    load_env_file()
    try:
        username, password = stream_credentials()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    experiment = ExperimentDefinition.from_json("config/experiment.json")
    perception_config = (
        PerceptionConfig.from_yaml("config/perception.yaml")
        if Path("config/perception.yaml").exists()
        else PerceptionConfig()
    )
    runtime_config = (
        RuntimeConfig.from_yaml("config/runtime.yaml")
        if Path("config/runtime.yaml").exists()
        else RuntimeConfig()
    )

    source = open_source(runtime_config.source)
    # Everything from here on that can raise (PerceptionPipeline, TTSWorker,
    # Router, RuntimeLoop, create_app, start_threads, run_app) must still
    # close `source` (and `speaker`, once it exists) on the way out --
    # nothing else owns them until RuntimeLoop is successfully constructed,
    # and even then RuntimeLoop.stop() is the one thing that actually closes
    # `source`. A bare try/except around only the PerceptionPipeline
    # construction step (the original shape here) left every later failure
    # point leaking the camera; one try/finally covering the whole wiring
    # sequence, keyed on whether `loop` was reached, covers all of them.
    loop: RuntimeLoop | None = None
    speaker: TTSWorker | None = None
    exit_code = 0
    try:
        perception = PerceptionPipeline(perception_config)
        tracker = StateTracker(experiment, perception_config)
        engine = SequenceEngine(experiment, runtime_config)
        speaker = TTSWorker()
        recent_alerts = RecentAlerts(cap=20)
        router = Router(
            experiment,
            runtime_config,
            tracker,
            engine,
            speaker,
            runtime_config.log_dir,
            on_engine_event=recent_alerts,
        )
        loop = RuntimeLoop(source, perception, router, runtime_config, runtime_config.video_dir)
        app = create_app(
            experiment=experiment,
            runtime_config=runtime_config,
            router=router,
            loop=loop,
            log_dir=runtime_config.log_dir,
            username=username,
            password=password,
            recent_alerts=recent_alerts.snapshot,
        )

        loop.start_threads()
        logger.info(
            "dashboard on %s:%d -- open it and press Start (Ctrl+C here to stop)",
            resolve_bind_host(runtime_config),
            runtime_config.port,
        )
        run_app(app, runtime_config)  # blocks until Ctrl+C
        if loop.fatal_error is not None:
            logger.error("dev run stopped after an unexpected error: %s", loop.fatal_error)
            exit_code = 1
    except Exception:
        logger.exception("scripts/dev.py: run failed")
        exit_code = 1
    finally:
        if loop is not None:
            # loop.stop() (which closes source/recorder and joins the
            # threads) must run even if reset_run() itself failed -- a
            # failure in one shutdown step must never skip the others.
            try:
                loop.reset_run(t=0.0)
            except Exception:
                logger.warning("scripts/dev.py: reset_run failed during shutdown", exc_info=True)
            loop.stop()
        else:
            try:
                source.close()
            except Exception:
                logger.warning("scripts/dev.py: error closing frame source", exc_info=True)
        if speaker is not None:
            speaker.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
