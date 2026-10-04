#!/usr/bin/env python3
"""python scripts/dev.py -- run the live system against the webcam, served by the dashboard
(IMPLEMENTATION_PLAN.md 5.9).

The wiring lives in harness/live.py and is shared with scripts/demo.py: the perception pipeline
comes from perception.pipeline.load_pipeline() (so $SIH_DETECTOR and weights/MANIFEST.json choose
the detector, and a weights hash that differs from the manifest refuses the start), the settings
from config/runtime.yaml and config/perception.yaml. The operator presses Start on the dashboard
(POST /api/run/start). Runs until Ctrl+C.

    python scripts/dev.py                    camera 0
    python scripts/dev.py --source 1         another camera
    python scripts/dev.py --source clip.mp4  a file or a stream URL (digits mean a camera index)

Only the standard library is imported at the top, because the TTS worker is a spawned process that
re-imports this script (see scripts/demo.py).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `python scripts/dev.py` works


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--source",
        default=None,
        help="camera index (digits), or a file path / stream URL (default: runtime.yaml's source)",
    )
    parser.add_argument("--port", type=int, default=None, help="override runtime.yaml's port")
    parser.add_argument("--tls", choices=("auto", "on", "off"), default="auto")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    from harness import live
    from harness.settings import load_runtime_config

    runtime_path = live.ROOT / "config" / "runtime.yaml"
    source = args.source
    if source is None:
        try:
            source = load_runtime_config(runtime_path).source
        except FileNotFoundError:
            source = "0"  # the self-check will name the missing file
    options = live.LiveOptions(mode="dev", source=source, port=args.port, tls=args.tls)
    return live.run_live(options, print)


if __name__ == "__main__":
    raise SystemExit(main())
