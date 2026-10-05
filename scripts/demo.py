#!/usr/bin/env python3
"""python scripts/demo.py -- the rehearsed demo configuration (IMPLEMENTATION_PLAN.md 5.9).

One command starts the whole system (capture -> perception -> state -> engine -> log, voice,
dashboard, recorder) with the settings in config/runtime.yaml and config/perception.yaml, after
a startup self-check in which every item is OK or FAIL with a reason (no password is printed).

    python scripts/demo.py                       camera 0 at the camera module's requested mode
    python scripts/demo.py --source 1            another camera
    python scripts/demo.py --source clip.mp4     a file, or a stream URL, instead of a camera
    python scripts/demo.py --replay-run x016     play runs/x016/video.mp4 in REAL TIME through the
                                                 same live loop (the backup mode, remote judges)
    python scripts/demo.py --playlist demo_videos   a folder of recorded clips, one after another
    python scripts/demo.py --check               only the self-check; exit 1 on any FAIL
    python scripts/demo.py --tts-test            speak "Audio check" through the real speaker path

A file (--replay-run, --source FILE, --playlist) does not start by itself: the first frame is shown
at once as a still preview, and the run and the voice start together when the dashboard asks
(open the URL printed after READY, which ends in ?autostart=1) or when Start is pressed. Use
--no-wait-for-dashboard (or --auto-start) to start at once: tests, soak, headless runs. The voice
is held back by --speech-delay seconds (default 0.2) so it lines up with the frame on the screen.

PLAYLIST (demo only). --playlist PATH is a folder of video files (.mp4 .avi .mov .mkv, in name
order) or a text file with one path per line (relative paths are resolved against the file's
folder). Every clip is a SEPARATE run through the same live loop (fresh perception, tracker and
engine, a new run_id, its own log, its own run_completed); the last frame is held during
--pause-between; the server, the dashboard and the voice stay up. DO NOT STITCH CLIPS INTO ONE
FILE: the engine would treat it as one run and the second clip would be judged as the
continuation of the first.

TLS is switched on automatically when `python scripts/gen_cert.py` has written certs/cert.pem and
certs/key.pem (or when tls_cert / tls_key are set in config/runtime.yaml); without it the server
binds 127.0.0.1 and the URL is http://127.0.0.1:PORT. The credentials come from the git-ignored
.env file. Ctrl+C stops cleanly (run finished, recorder file closed, log flushed, TTS stopped).

This module imports only the standard library at the top: the TTS worker is a spawned process that
re-imports the main script, and every cue that restarts it would otherwise pay for the
heavy imports.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `python scripts/demo.py` works


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--source",
        metavar="SOURCE",
        help="a camera index (digits), a video file path or a stream URL (default: camera 0)",
    )
    src.add_argument(
        "--playlist",
        type=Path,
        metavar="PATH",
        help="a folder of video clips (name order) or a text file with one clip path per line;\n"
        "each clip is its own run, played one after another (never stitch clips into one file).\n"
        "A clip under runs/<test id>/ needs --allow-heldout",
    )
    src.add_argument(
        "--replay-run",
        metavar="RUN_ID",
        help="play runs/RUN_ID/video.mp4 in real time (paced by the frame times) through the\n"
        "same live loop; the run starts automatically",
    )
    parser.add_argument(
        "--allow-heldout",
        action="store_true",
        help="allow --replay-run for a run of the test split. Demonstrating a held-out run is\n"
        "honest and allowed; this flag only prevents accidents (a held-out run chosen by mistake)",
    )
    parser.add_argument("--check", action="store_true", help="run only the self-check, then exit")
    parser.add_argument(
        "--tts-test", action="store_true", help='speak "Audio check" once and report, then exit'
    )
    parser.add_argument(
        "--tls", choices=("auto", "on", "off"), default="auto", help="default: auto"
    )
    parser.add_argument("--port", type=int, default=None, help="override runtime.yaml's port")
    auto = parser.add_mutually_exclusive_group()
    auto.add_argument(
        "--auto-start",
        dest="auto_start",
        action="store_true",
        default=None,
        help="start the run at once, without waiting for the dashboard (see also\n"
        "--no-wait-for-dashboard)",
    )
    auto.add_argument(
        "--no-auto-start",
        dest="auto_start",
        action="store_false",
        help="wait for Start on the dashboard (the default for a camera; for a file the first\n"
        "frame is held until Start, see --no-wait-for-dashboard)",
    )
    flow = parser.add_argument_group("demo flow")
    flow.add_argument(
        "--no-wait-for-dashboard",
        dest="wait_for_dashboard",
        action="store_false",
        help="do not hold on the first frame: start the run at once (tests, soak, headless runs)",
    )
    flow.add_argument(
        "--speech-delay",
        type=float,
        default=0.2,
        metavar="SECONDS",
        help="hold each spoken cue this long so it lines up with the frame on the screen\n"
        "(default 0.2; 0 disables). Tune by eye: raise it if the voice runs ahead of the video",
    )
    flow.add_argument(
        "--pause-between",
        type=float,
        default=5.0,
        metavar="SECONDS",
        help="--playlist: how long the last frame is held between clips (default 5)",
    )
    flow.add_argument(
        "--advance",
        choices=("auto", "enter"),
        default="auto",
        help="--playlist: 'auto' starts the next clip after the pause; 'enter' waits for the\n"
        "presenter to press Enter in this terminal (the pause is then ignored)",
    )
    flow.add_argument("--shuffle", action="store_true", help="--playlist: seeded shuffle")
    flow.add_argument("--seed", type=int, default=0, help="--shuffle seed (default 0)")
    flow.add_argument(
        "--once", action="store_true", help="--playlist: stop after one pass (default: loop)"
    )
    flow.add_argument(
        "--no-record", dest="record", action="store_false", help="write no recorder .avi file"
    )
    measure = parser.add_argument_group("measurement and soak (used by the S-I1 measurements)")
    measure.add_argument(
        "--loop",
        action="store_true",
        help="replay the file again and again, a fresh run per pass (soak)",
    )
    measure.add_argument(
        "--max-speed",
        action="store_true",
        help="no pacing and no throttle: how fast can the pipeline go "
        "(events are not meaningful in this mode)",
    )
    measure.add_argument(
        "--unthrottled",
        action="store_true",
        help="keep real-time pacing but lift the target_fps throttle: the pipeline's own speed,\n"
        "capped by the source's frame rate",
    )
    measure.add_argument(
        "--duration",
        type=float,
        default=None,
        metavar="SECONDS",
        help="stop after this many seconds",
    )
    measure.add_argument(
        "--exit-when-done",
        action="store_true",
        help="exit when the file has been played (default: keep serving)",
    )
    measure.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        metavar="FILE.json",
        help="write throughput, drops, latency, CPU and memory samples here",
    )
    measure.add_argument(
        "--sample-every",
        type=float,
        default=30.0,
        metavar="SECONDS",
        help="memory sampling interval for --metrics-out (default 30)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("werkzeug").setLevel(logging.WARNING)  # one line per request is noise

    from harness import live

    if args.tts_test:
        item = live.tts_test()
        print(item.line())
        print("Confirm by ear that you heard: Audio check")
        return 0 if item.ok else 1

    options = live.LiveOptions(
        mode="demo",
        source=args.source,
        replay_run=args.replay_run,
        allow_heldout=args.allow_heldout,
        loop_source=args.loop,
        max_speed=args.max_speed,
        unthrottled=args.unthrottled,
        duration_s=args.duration,
        exit_when_done=args.exit_when_done,
        auto_start=args.auto_start,
        metrics_out=args.metrics_out,
        sample_every_s=args.sample_every,
        tls=args.tls,
        port=args.port,
        check_only=args.check,
        wait_for_dashboard=args.wait_for_dashboard,
        speech_delay_s=args.speech_delay,
        playlist=args.playlist,
        pause_between_s=args.pause_between,
        advance=args.advance,
        shuffle=args.shuffle,
        seed=args.seed,
        once=args.once,
        record=args.record,
    )
    return live.run_live(options, print)


if __name__ == "__main__":
    raise SystemExit(main())
