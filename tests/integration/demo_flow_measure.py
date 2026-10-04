"""Measurements for the demo flow (S-I1b, block M1). NOT a test (the name does not match
test_*.py, pytest never collects it): run it by hand with the project interpreter,

    python tests/integration/demo_flow_measure.py [--out runs_out/measure/demo_flow.json]

It starts the real LiveSystem (real pipeline, a real werkzeug server on a free loopback port, no
TLS, throwaway credentials that exist only in this process) on the val clip runs/x015, gated, and
talks to it over HTTP like a browser would. Nothing is printed that contains a credential.

Measured, once with --speech-delay 0 and once with 0.2:
  * page request -> first /video_feed bytes (and the preview: the stream shows frame 0 while the
    run is idle, before anything is started);
  * POST /api/run/start -> first speech call, and -> first advancing frame (the store and the
    stream);
  * for every engine event that speaks: the moment the first /video_feed frame at or after the
    event's frame reached the client, and the moment its speech was handed to the speaker; the
    skew is speech minus picture (positive: the voice comes after the picture).

Frames on the wire are identified by a 16-bit marker painted into a strip at the bottom-left of
each frame by a measuring wrapper around the source (the clip's own pixels elsewhere are
untouched); the marker is read back from the decoded JPEG.
"""

from __future__ import annotations

import argparse
import base64
import http.client
import json
import socket
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from contracts import Frame  # noqa: E402
from harness import live  # noqa: E402

CLIP = ROOT / "runs" / "x015" / "video.mp4"
MANIFEST = ROOT / "weights" / "MANIFEST.json"
STOP_AT_T = 10.5  # seconds of the clip: x015's events are at 2.4 .. 9.3 s
USER, PASSWORD = "measure_user", "measure_pass_0"  # throwaway, in-process only
BLOCK = 8


class MarkedSource:
    """Paints the frame id (16 bits, 8 px blocks, bottom-left) into every frame."""

    def __init__(self, inner) -> None:  # noqa: ANN001
        self._inner = inner

    @property
    def fps(self):  # noqa: ANN201
        return self._inner.fps

    @property
    def exhausted(self) -> bool:
        return self._inner.exhausted

    def read(self) -> Frame | None:
        frame = self._inner.read()
        if frame is None:
            return None
        image = frame.image.copy()
        h = image.shape[0]
        for bit in range(16):
            on = (frame.frame_id >> bit) & 1
            image[h - BLOCK : h, bit * BLOCK : (bit + 1) * BLOCK] = 255 if on else 0
        return Frame(frame_id=frame.frame_id, t=frame.t, image=image)

    def close(self) -> None:
        self._inner.close()


def read_marker(image: np.ndarray) -> int:
    h = image.shape[0]
    value = 0
    for bit in range(16):
        block = image[h - BLOCK + 2 : h - 2, bit * BLOCK + 2 : (bit + 1) * BLOCK - 2]
        if float(block.mean()) > 127:
            value |= 1 << bit
    return value


class TimedSpeaker:
    def __init__(self) -> None:
        self.calls: list[tuple[float, str, str]] = []
        self.closed = False

    def say(self, text: str, priority: str) -> None:
        self.calls.append((time.perf_counter(), text, priority))

    def close(self) -> None:
        self.closed = True


class FeedClient(threading.Thread):
    """Reads /video_feed like an <img> would; records (arrival time, marker id) per part."""

    def __init__(self, port: int) -> None:
        super().__init__(daemon=True)
        self.port = port
        self.parts: list[tuple[float, int]] = []
        self.first_bytes_at: float | None = None
        self.started_at: float | None = None
        self.halt = threading.Event()

    def run(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        self.started_at = time.perf_counter()
        conn.request("GET", "/video_feed", headers=_auth())
        resp = conn.getresponse()
        buf = b""
        while not self.halt.is_set():
            chunk = resp.read1(65536)
            if not chunk:
                break
            if self.first_bytes_at is None:
                self.first_bytes_at = time.perf_counter()
            buf += chunk
            while True:
                start = buf.find(b"\xff\xd8")
                end = buf.find(b"\xff\xd9", start + 2) if start >= 0 else -1
                if start < 0 or end < 0:
                    break
                arrived = time.perf_counter()
                jpg, buf = buf[start : end + 2], buf[end + 2 :]
                image = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                if image is not None:
                    self.parts.append((arrived, read_marker(image)))
        conn.close()


def _auth() -> dict[str, str]:
    return {"Authorization": "Basic " + base64.b64encode(f"{USER}:{PASSWORD}".encode()).decode()}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def request(port: int, method: str, path: str) -> tuple[int, bytes, float]:
    t0 = time.perf_counter()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request(method, path, headers=_auth())
    resp = conn.getresponse()
    body = resp.read()
    t1 = time.perf_counter()
    conn.close()
    return resp.status, body, t1 - t0


def until(predicate, timeout: float = 30.0) -> bool:  # noqa: ANN001
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.001)
    return False


def measure_once(perception, speech_delay: float, work: Path) -> dict:  # noqa: ANN001
    settings = live.load_settings(
        ROOT / "config" / "runtime.yaml",
        ROOT / "config" / "perception.yaml",
        ROOT / "config" / "experiment.json",
    )
    port = free_port()
    runtime = settings.runtime.model_copy(
        update={
            "port": port,
            "log_dir": str(work / "logs"),
            "video_dir": str(work / "video"),
            "tls_cert": None,
            "tls_key": None,
        }
    )
    settings = live.Settings(settings.experiment, runtime, settings.perception)
    from perception.camera import open_source

    perception.reset()
    gate = live.StartGate()
    source = MarkedSource(live.PacedSource(open_source(str(CLIP)), gate=gate))
    speaker = TimedSpeaker()
    spoken = live.DelayedSpeaker(speaker, speech_delay) if speech_delay > 0 else speaker
    system = live.LiveSystem(
        settings=settings,
        perception=perception,
        source=source,
        speaker=spoken,
        username=USER,
        password=PASSWORD,
        recorder_factory=live.NullRecorder,
    )
    system.start(auto_start=False)
    halt = threading.Event()
    releaser = threading.Thread(
        target=live.release_gate_when_running,
        args=(system.router, lambda: gate, halt, 0.005),
        daemon=True,
    )
    releaser.start()
    out: dict = {"speech_delay_s": speech_delay}
    try:
        until(lambda: system.loop.frame_store.get() is not None)
        t_page = time.perf_counter()
        status, _, page_s = request(port, "GET", "/?autostart=1")
        feed = FeedClient(port)
        feed.start()
        until(lambda: feed.first_bytes_at is not None)
        out["page_status"] = status
        out["page_to_first_feed_bytes_ms"] = round((feed.first_bytes_at - t_page) * 1000, 1)
        out["page_load_ms"] = round(page_s * 1000, 1)
        time.sleep(1.0)  # the preview: idle, the still first frame, no speech, no log
        preview = [m for _, m in feed.parts]
        status_idle = json.loads(request(port, "GET", "/api/status")[1])
        out["preview_frames_on_wire"] = len(preview)
        out["preview_marker_ids"] = sorted(set(preview))
        out["preview_run_state"] = status_idle["run_state"]
        out["speech_calls_before_start"] = len(speaker.calls)

        n_before = len(feed.parts)
        t_post = time.perf_counter()
        code, body, _ = request(port, "POST", "/api/run/start")
        out["post_status"] = code
        until(lambda: speaker.calls)
        out["post_to_first_speech_ms"] = round((speaker.calls[0][0] - t_post) * 1000, 1)
        first_id = system.loop.frame_store.get().frame_id
        t_store = None
        while t_store is None:
            frame = system.loop.frame_store.get()
            if frame is not None and frame.frame_id > first_id:
                t_store = time.perf_counter()
            else:
                time.sleep(0.0005)
        out["post_to_first_advancing_frame_store_ms"] = round((t_store - t_post) * 1000, 1)
        until(lambda: any(m > first_id for _, m in feed.parts[n_before:]))
        t_wire = next(a for a, m in feed.parts[n_before:] if m > first_id)
        out["post_to_first_advancing_frame_wire_ms"] = round((t_wire - t_post) * 1000, 1)

        until(lambda: system.loop.frame_store.get().t >= STOP_AT_T, timeout=60)
        time.sleep(0.5)
        events = system.tally.take()
        # speaker calls: the first phrase, then one per speaking event, in order
        speaking = [e for e in events if e.speak is not None]
        calls = speaker.calls[1:]
        fps = source.fps or 30.0
        skews = []
        for event, call in zip(speaking, calls, strict=False):
            frame_id = round(event.t * fps)
            seen = next((a for a, m in feed.parts if m >= frame_id), None)
            if seen is None:
                continue
            skews.append(
                {
                    "event": f"{event.kind}:{event.step_id}",
                    "t_video": round(event.t, 2),
                    "speech_minus_picture_ms": round((call[0] - seen) * 1000, 1),
                    "priority": call[2],
                }
            )
        out["event_skews"] = skews
        values = [s["speech_minus_picture_ms"] for s in skews]
        if values:
            out["skew_ms_median"] = round(statistics.median(values), 1)
            out["skew_ms_min"], out["skew_ms_max"] = min(values), max(values)
        feed.halt.set()
    finally:
        halt.set()
        system.shutdown()
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    default_out = ROOT / "runs_out" / "measure" / "demo_flow.json"
    parser.add_argument("--out", type=Path, default=default_out)
    args = parser.parse_args()
    from perception.pipeline import load_pipeline

    perception = load_pipeline(MANIFEST)
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for delay in (0.0, 0.2):
            work = Path(tmp) / f"delay_{delay}"
            work.mkdir()
            results.append(measure_once(perception, delay, work))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
