"""harness/live.py -- the live system's wiring and preflight, shared by scripts/dev.py and
scripts/demo.py (IMPLEMENTATION_PLAN.md 5.6 and 5.9; essential-features.md section 0, F10, F13).

Nothing here changes what perception, state, engine or runtime *do*; it only injects: the real
``Perception`` (``perception.pipeline.load_pipeline``), the settings read from
``config/runtime.yaml`` / ``config/perception.yaml``, the ``Speaker``, and wrappers around the
``FrameSource`` (real-time pacing, looping, measuring) and the ``Router`` (measuring).

Library code logs through ``logging``; the scripts print. A user-facing report is returned as
data (``CheckItem`` lists, ``LiveMetrics.report()``), never printed from here.
"""

from __future__ import annotations

import csv
import json
import logging
import multiprocessing as mp
import os
import socket
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from contracts import (
    ExperimentDefinition,
    Frame,
    FrameSource,
    PerceptionConfig,
    PerceptionFrame,
    RuntimeConfig,
    sha256_of_file,
)
from perception.detector import ENV_VAR
from runtime.loop import Router

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDER_NOTE = "'changeme' placeholder"
UNTHROTTLED_FPS = 1000.0
"""``--max-speed`` replaces ``target_fps`` with this, so the inference loop runs as fast as the
pipeline allows (a throughput measurement mode; the tuned hysteresis no longer applies)."""


class StartupRefused(RuntimeError):
    """The system will not start (a failed self-check item, a refused option)."""


class HeldoutRefused(StartupRefused):
    """A run of the test split was named without ``--allow-heldout``."""


# --------------------------------------------------------------------------------------------
# Self-check items
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckItem:
    name: str
    ok: bool
    detail: str

    def line(self) -> str:
        return f"[{'OK' if self.ok else 'FAIL'}] {self.name}: {self.detail}"


def format_report(items: list[CheckItem]) -> str:
    return "\n".join(i.line() for i in items)


def all_ok(items: list[CheckItem]) -> bool:
    return all(i.ok for i in items)


def chosen_detector_entry(
    manifest: Mapping[str, Any],
    detector: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """The manifest entry the pipeline would load: argument, then ``$SIH_DETECTOR``, then the
    manifest's ``active_detector`` (the precedence of ``perception.detector.load_detector``)."""
    env = os.environ if environ is None else environ
    chosen = (
        (detector or "").strip() or (env.get(ENV_VAR) or "").strip() or manifest["active_detector"]
    )

    def canon(name: str) -> str:
        return name.strip().lower().replace("-", "_")

    for entry in manifest["detectors"]:
        if canon(entry["name"]) == canon(chosen):
            return dict(entry)
    raise ValueError(
        f"unknown detector {chosen!r}; the manifest lists "
        f"{[d['name'] for d in manifest['detectors']]}"
    )


def check_weights(
    manifest_path: Path | str,
    detector: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> list[CheckItem]:
    """Two items: the chosen detector's weights and the hand model, each hashed against the
    manifest (the pipeline checks again when it loads; this one refuses before loading)."""
    mpath = Path(manifest_path)
    try:
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        entry = chosen_detector_entry(manifest, detector, environ)
    except (OSError, ValueError, KeyError) as exc:
        return [CheckItem("weights manifest", False, f"{mpath}: {exc}")]
    items = []
    for name, spec in (
        (f"detector weights ({entry['name']})", entry),
        ("hand model", manifest.get("hand") or {}),
    ):
        items.append(_check_hashed_file(name, mpath.parent, spec))
    return items


def _check_hashed_file(name: str, base: Path, spec: Mapping[str, Any]) -> CheckItem:
    try:
        path = base / spec["file"]
        expected = spec["sha256"]
    except KeyError as exc:
        return CheckItem(name, False, f"the manifest entry has no {exc}")
    if not path.is_file():
        return CheckItem(name, False, f"file not found: {path}")
    actual = sha256_of_file(path)
    if actual != expected:
        return CheckItem(
            name, False, f"sha256 {actual[:12]} differs from the manifest {expected[:12]}"
        )
    return CheckItem(name, True, f"{path.name}, sha256 {actual[:12]} matches the manifest")


@dataclass(frozen=True)
class Settings:
    experiment: ExperimentDefinition
    runtime: RuntimeConfig
    perception: PerceptionConfig


def load_settings(
    runtime_path: Path | str,
    perception_path: Path | str,
    experiment_path: Path | str,
) -> Settings:
    from harness.settings import load_perception_config, load_runtime_config

    return Settings(
        experiment=ExperimentDefinition.from_json(experiment_path),
        runtime=load_runtime_config(runtime_path),
        perception=load_perception_config(perception_path),
    )


def check_configs(
    runtime_path: Path | str, perception_path: Path | str, experiment_path: Path | str
) -> tuple[CheckItem, Settings | None]:
    try:
        settings = load_settings(runtime_path, perception_path, experiment_path)
    except Exception as exc:  # a missing file, a bad key, a bad value: all the same to the operator
        return CheckItem("config files", False, f"{type(exc).__name__}: {exc}"), None
    return (
        CheckItem(
            "config files",
            True,
            f"{Path(runtime_path).name}, {Path(perception_path).name}, "
            f"{Path(experiment_path).name} load "
            f"(target_fps {settings.runtime.target_fps:g}, {len(settings.experiment.steps)} steps)",
        ),
        settings,
    )


def check_env(
    env_path: Path | str = ROOT / ".env",
) -> tuple[CheckItem, tuple[str, str] | None]:
    """``STREAM_USER`` and ``STREAM_PASSWORD`` set, non-empty and not the ``changeme``
    placeholder. The detail names where they come from and never contains a value."""
    from server.env import load_env_file, stream_credentials

    path = Path(env_path)
    keys = ("STREAM_USER", "STREAM_PASSWORD")
    in_environment = [k for k in keys if os.environ.get(k)]  # a real variable wins over the file
    if not path.is_file():
        origin = "process environment (no .env file)"
    elif len(in_environment) == len(keys):
        origin = "process environment (it overrides the .env file)"
    elif in_environment:
        origin = f".env file ({path}) and the process environment"
    else:
        origin = f".env file ({path})"
    in_file = path.is_file()
    load_env_file(path)
    try:
        creds = stream_credentials()
    except RuntimeError as exc:
        text = str(exc)
        reason = (
            f"the {PLACEHOLDER_NOTE} is still set"
            if "changeme" in text
            else "STREAM_USER and STREAM_PASSWORD must both be set and non-empty"
        )
        where = f"{path} not found; " if not in_file else ""
        return CheckItem("credentials", False, f"{where}{reason}"), None
    return CheckItem(
        "credentials", True, f"STREAM_USER and STREAM_PASSWORD set, from the {origin}"
    ), creds


def check_port(host: str, port: int) -> CheckItem:
    """The port is free: nothing accepts a connection on it and a plain bind succeeds."""
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.5)
        if probe.connect_ex((probe_host, port)) == 0:
            return CheckItem("port", False, f"{probe_host}:{port} is already in use")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host or "127.0.0.1", port))
        except OSError as exc:
            return CheckItem("port", False, f"cannot bind {host}:{port}: {exc}")
    return CheckItem("port", True, f"{host}:{port} is free")


def check_tls(runtime_config: RuntimeConfig) -> CheckItem:
    cert, key = runtime_config.tls_cert, runtime_config.tls_key
    if not cert and not key:
        return CheckItem("TLS", True, "off (no tls_cert/tls_key): the server binds 127.0.0.1")
    if not (cert and key):
        return CheckItem("TLS", False, "only one of tls_cert / tls_key is set; both are needed")
    missing = [p for p in (cert, key) if not Path(p).is_file()]
    if missing:
        return CheckItem(
            "TLS", False, f"file(s) not found: {', '.join(missing)} (python scripts/gen_cert.py)"
        )
    return CheckItem("TLS", True, f"on: {cert}, {key}")


def tls_auto(
    runtime_config: RuntimeConfig, certs_dir: Path | str = ROOT / "certs"
) -> RuntimeConfig:
    """``--tls auto``: when the config sets no certificate and ``certs/cert.pem`` + ``key.pem``
    (what ``scripts/gen_cert.py`` writes) both exist, switch TLS on."""
    if runtime_config.tls_cert or runtime_config.tls_key:
        return runtime_config
    cert, key = Path(certs_dir) / "cert.pem", Path(certs_dir) / "key.pem"
    if cert.is_file() and key.is_file():
        return runtime_config.model_copy(update={"tls_cert": str(cert), "tls_key": str(key)})
    return runtime_config


def tls_is_on(runtime_config: RuntimeConfig) -> bool:
    return bool(runtime_config.tls_cert and runtime_config.tls_key)


# --- TTS ------------------------------------------------------------------------------------


def _tts_probe(engine_factory: Callable[[], Any], result: Any) -> None:
    """Child process: build the engine the way ``outputs.tts._worker_main`` does (without
    speaking) and report. A module-level function so the spawn start method can pickle it."""
    try:
        engine = engine_factory()
        engine.setProperty("rate", 170)
        voices = engine.getProperty("voices") or []
        result.put(("ok", f"{len(voices)} voice(s)"))
    except Exception as exc:
        result.put(("fail", f"{type(exc).__name__}: {exc}"))


def check_tts(
    engine_factory: Callable[[], Any] | None = None, timeout_s: float = 20.0
) -> CheckItem:
    """The TTS engine initialises, probed in a short-lived child process like the real worker
    (``outputs/tts.py`` degrades silently when it cannot, which is exactly what to catch here)."""
    from outputs.tts import default_engine_factory

    factory = engine_factory or default_engine_factory
    ctx = mp.get_context("spawn")
    result = ctx.Queue()
    proc = ctx.Process(target=_tts_probe, args=(factory, result), daemon=True)
    proc.start()
    try:
        status, detail = result.get(timeout=timeout_s)
    except Exception:
        status, detail = "fail", f"no answer within {timeout_s:g}s"
    finally:
        proc.join(timeout=2.0)
        if proc.is_alive():
            proc.terminate()
    return CheckItem(
        "TTS engine", status == "ok", detail if status != "ok" else f"initialised ({detail})"
    )


class ReportingEngineFactory:
    """An engine factory for ``TTSWorker`` that reports through a queue what the engine did:
    ``("init", None)``, ``("spoke", text)`` or ``("fail", message)``. ``--tts-test`` uses it to
    drive the *real* speaker path and still learn whether the engine accepted the utterance."""

    def __init__(self, inner: Callable[[], Any], report: Any) -> None:
        self._inner = inner
        self._report = report

    def __call__(self) -> Any:
        try:
            engine = self._inner()
        except Exception as exc:
            self._report.put(("fail", f"{type(exc).__name__}: {exc}"))
            raise
        self._report.put(("init", None))
        return _ReportingEngine(engine, self._report)


class _ReportingEngine:
    def __init__(self, engine: Any, report: Any) -> None:
        self._engine = engine
        self._report = report
        self._last = ""

    def setProperty(self, name: str, value: Any) -> None:  # noqa: N802 - pyttsx3's API
        self._engine.setProperty(name, value)

    def getProperty(self, name: str) -> Any:  # noqa: N802
        return self._engine.getProperty(name)

    def say(self, text: str) -> None:
        self._last = text
        self._engine.say(text)

    def stop(self) -> None:
        self._engine.stop()

    def runAndWait(self) -> None:  # noqa: N802
        try:
            self._engine.runAndWait()
        except Exception as exc:
            self._report.put(("fail", f"{type(exc).__name__}: {exc}"))
            raise
        self._report.put(("spoke", self._last))


def tts_test(
    text: str = "Audio check",
    engine_factory: Callable[[], Any] | None = None,
    timeout_s: float = 20.0,
) -> CheckItem:
    """Speak ``text`` once through ``TTSWorker`` (priority ``info``) and report whether the
    engine accepted it. Whether it was *heard* is for the operator to confirm by ear."""
    from outputs.tts import TTSWorker, default_engine_factory

    ctx = mp.get_context("spawn")
    report = ctx.Queue()
    worker = TTSWorker(
        engine_factory=ReportingEngineFactory(engine_factory or default_engine_factory, report)
    )
    try:
        worker.say(text, "info")
        deadline = time.monotonic() + timeout_s
        seen_init = False
        while time.monotonic() < deadline:
            try:
                kind, detail = report.get(timeout=max(0.05, deadline - time.monotonic()))
            except Exception:
                break
            if kind == "init":
                seen_init = True
            elif kind == "spoke":
                time.sleep(0.3)  # let the audio finish before close() terminates the worker
                return CheckItem("TTS test", True, f"the engine accepted and played {detail!r}")
            elif kind == "fail":
                return CheckItem("TTS test", False, f"the engine failed: {detail}")
        reason = "initialised but did not finish speaking" if seen_init else "did not initialise"
        return CheckItem("TTS test", False, f"{reason} within {timeout_s:g}s")
    finally:
        worker.close()


# --- sources --------------------------------------------------------------------------------


def run_split(run_id: str, runs_dir: Path | str = ROOT / "runs") -> str | None:
    """The split of a recorded run: ``runs/<id>/script.json``, else ``runs/manifest.csv``.
    ``None`` when neither knows the id."""
    runs = Path(runs_dir)
    script = runs / run_id / "script.json"
    if script.is_file():
        try:
            return json.loads(script.read_text(encoding="utf-8")).get("split")
        except (OSError, ValueError):
            pass
    manifest = runs / "manifest.csv"
    if manifest.is_file():
        with manifest.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("run_id") == run_id:
                    return row.get("split")
    return None


def resolve_source(
    source: str | None,
    replay_run: str | None,
    *,
    default_source: str = "0",
    runs_dir: Path | str = ROOT / "runs",
    allow_heldout: bool = False,
) -> str:
    """The string for ``open_source``. ``replay_run`` names ``runs/<id>/video.mp4``. A run of the
    test split is refused unless ``allow_heldout``: demonstrating a held-out run is honest and
    allowed, the flag only prevents accidents. A ``--source`` path that points into a run folder
    is held to the same rule."""
    runs = Path(runs_dir)
    if replay_run and source:
        raise StartupRefused("--source and --replay-run are alternatives; give one")
    if replay_run:
        video = runs / replay_run / "video.mp4"
        if not video.is_file():
            raise StartupRefused(f"no recording for run {replay_run!r}: {video} not found")
        _refuse_heldout(replay_run, runs, allow_heldout)
        return str(video)
    chosen = default_source if source is None else source
    if not chosen.isdigit():
        path = Path(chosen)
        try:
            inside = path.resolve().parent.parent == runs.resolve()
        except OSError:
            inside = False
        if inside:
            _refuse_heldout(path.resolve().parent.name, runs, allow_heldout)
    return chosen


def _refuse_heldout(run_id: str, runs: Path, allow_heldout: bool) -> None:
    split = run_split(run_id, runs)
    if split == "test" and not allow_heldout:
        raise HeldoutRefused(
            f"run {run_id!r} belongs to the test split; pass --allow-heldout to replay it "
            "(showing a held-out run is honest and allowed; the flag only prevents accidents)"
        )


class _SourceWrapper:
    """Common base: delegates ``fps``, ``exhausted`` and ``close`` to ``inner``."""

    def __init__(self, inner: FrameSource) -> None:
        self._inner = inner

    @property
    def fps(self) -> float | None:
        return self._inner.fps

    @property
    def exhausted(self) -> bool:
        return self._inner.exhausted

    def close(self) -> None:
        self._inner.close()


class PrefetchedSource(_SourceWrapper):
    """Hands back one already-read frame first (the self-check reads a frame to learn the
    granted mode), then the wrapped source."""

    def __init__(self, first: Frame, inner: FrameSource) -> None:
        super().__init__(inner)
        self._first: Frame | None = first

    def read(self) -> Frame | None:
        if self._first is not None:
            frame, self._first = self._first, None
            return frame
        return self._inner.read()


class PacedSource(_SourceWrapper):
    """Real-time pacing for a file replay: ``read()`` returns a frame no earlier than its ``t``
    after the first read, so alerts and speech arrive at the natural speed of the recording, like
    a camera. If the reader falls more than ``resync_after_s`` behind, the schedule is re-anchored
    instead of delivering a burst."""

    def __init__(
        self,
        inner: FrameSource,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        resync_after_s: float = 1.0,
    ) -> None:
        super().__init__(inner)
        self._clock = clock
        self._sleep = sleep
        self._resync_after_s = resync_after_s
        self._origin: float | None = None  # clock() value at which t == 0

    def read(self) -> Frame | None:
        frame = self._inner.read()
        if frame is None:
            return None
        now = self._clock()
        if self._origin is None:
            self._origin = now - frame.t
        due = self._origin + frame.t
        if now - due > self._resync_after_s:
            self._origin = now - frame.t
            due = now
        while (wait := due - self._clock()) > 0.0:
            self._sleep(min(wait, 0.1))
        return frame


class LoopingSource(_SourceWrapper):
    """Plays a file again and again (the soak): when the inner source is exhausted it is closed,
    ``on_loop(n)`` is called, ``factory()`` opens a new one, and frame ids and ``t`` carry on
    from where they were, so downstream sees one unbroken, increasing timeline."""

    def __init__(
        self,
        factory: Callable[[], FrameSource],
        on_loop: Callable[[int], None] | None = None,
    ) -> None:
        super().__init__(factory())
        self._factory = factory
        self._on_loop = on_loop
        self.loops = 0
        self._id_offset = 0
        self._t_offset = 0.0
        self._last_id = -1
        self._last_t = 0.0

    @property
    def exhausted(self) -> bool:
        return False

    def read(self) -> Frame | None:
        frame = self._inner.read()
        if frame is None:
            if not self._inner.exhausted:
                return None
            self._rewind()
            frame = self._inner.read()
            if frame is None:
                if self._inner.exhausted:
                    raise StartupRefused("the looped source yields no frames")
                return None
        out = Frame(
            frame_id=frame.frame_id + self._id_offset,
            t=frame.t + self._t_offset,
            image=frame.image,
        )
        self._last_id, self._last_t = out.frame_id, out.t
        return out

    def _rewind(self) -> None:
        period = 1.0 / (self._inner.fps or 30.0)
        self._inner.close()
        self._id_offset = self._last_id + 1
        self._t_offset = self._last_t + period
        self.loops += 1
        if self._on_loop is not None:
            self._on_loop(self.loops)
        self._inner = self._factory()


# --- measuring ------------------------------------------------------------------------------


def resident_memory_mb() -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process().memory_info().rss / (1024 * 1024)


class LiveMetrics:
    """Throughput, drops, latency (frame arrival -> status update) and resources of a live
    run. Thread-safe; diagnostic only, never read by any decision."""

    _MAX_PENDING = 5000

    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        cpu_clock: Callable[[], float] = time.process_time,
        memory: Callable[[], float | None] = resident_memory_mb,
    ) -> None:
        self._clock = clock
        self._cpu_clock = cpu_clock
        self._memory = memory
        self._lock = threading.Lock()
        self._arrivals: dict[int, float] = {}
        self._latencies: list[float] = []
        self._captured = 0
        self._processed = 0
        self._first_status: float | None = None
        self._last_status: float | None = None
        self._t0 = clock()
        self._cpu0 = cpu_clock()
        self._samples: list[dict[str, float | None]] = []

    def frame_arrived(self, frame_id: int) -> None:
        with self._lock:
            self._captured += 1
            self._arrivals[frame_id] = self._clock()
            while len(self._arrivals) > self._MAX_PENDING:
                self._arrivals.pop(next(iter(self._arrivals)))

    def status_updated(self, frame_id: int) -> None:
        now = self._clock()
        with self._lock:
            self._processed += 1
            if self._first_status is None:
                self._first_status = now
            self._last_status = now
            arrived = self._arrivals.pop(frame_id, None)
            if arrived is not None:
                self._latencies.append(now - arrived)

    def sample_resources(self) -> None:
        with self._lock:
            self._samples.append(
                {"t_s": round(self._clock() - self._t0, 1), "rss_mb": self._memory()}
            )

    @staticmethod
    def _percentile(sorted_values: list[float], q: float) -> float | None:
        if not sorted_values:
            return None
        index = max(0, min(len(sorted_values) - 1, round(q * (len(sorted_values) - 1))))
        return sorted_values[index]

    def report(self) -> dict[str, Any]:
        with self._lock:
            lat = sorted(self._latencies)
            wall = self._clock() - self._t0
            window = (
                (self._last_status - self._first_status)
                if self._first_status is not None and self._last_status is not None
                else 0.0
            )
            achieved = (
                (self._processed - 1) / window if window > 0 and self._processed > 1 else None
            )
            ms = lambda v: None if v is None else round(v * 1000.0, 1)  # noqa: E731
            return {
                "wall_s": round(wall, 2),
                "frames_captured": self._captured,
                "frames_processed": self._processed,
                "frames_dropped": self._captured - self._processed,
                "achieved_fps": None if achieved is None else round(achieved, 2),
                "fps_over_wall": round(self._processed / wall, 2) if wall > 0 else None,
                "latency_ms_p50": ms(self._percentile(lat, 0.50)),
                "latency_ms_p95": ms(self._percentile(lat, 0.95)),
                "latency_ms_max": ms(lat[-1] if lat else None),
                "cpu_percent_of_one_core": round(100.0 * (self._cpu_clock() - self._cpu0) / wall, 1)
                if wall > 0
                else None,
                "resources": list(self._samples),
            }


class MeteredSource(_SourceWrapper):
    def __init__(self, inner: FrameSource, metrics: LiveMetrics) -> None:
        super().__init__(inner)
        self._metrics = metrics

    def read(self) -> Frame | None:
        frame = self._inner.read()
        if frame is not None:
            self._metrics.frame_arrived(frame.frame_id)
        return frame


class MeteredRouter(Router):
    """A ``Router`` that tells ``LiveMetrics`` when a frame's status has been updated."""

    def __init__(self, *args: Any, metrics: LiveMetrics, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._metrics = metrics

    def process_perception_frame(self, frame: PerceptionFrame) -> None:
        super().process_perception_frame(frame)
        self._metrics.status_updated(frame.frame_id)


# --- the camera / file source check ---------------------------------------------------------

Opener = Callable[[str], FrameSource]


def _default_opener(source: str) -> FrameSource:
    from perception.camera import open_source

    return open_source(source)


def check_source(
    source: str, opener: Opener | None = None, tries: int = 60
) -> tuple[CheckItem, FrameSource | None]:
    """Opens ``source`` and reads one frame, so the *granted* mode is known (a camera is asked
    for 1280 x 720 at 30 fps; a driver may grant less). The frame is handed back inside a
    ``PrefetchedSource``, so nothing is lost and the camera is opened only once."""
    from contracts import CAPTURE_FPS, CAPTURE_HEIGHT, CAPTURE_WIDTH

    is_camera = source.isdigit()
    name = "camera" if is_camera else "video source"
    label = f"camera {source}" if is_camera else source
    try:
        opened = (opener or _default_opener)(source)
    except Exception as exc:
        return CheckItem(name, False, f"cannot open {label}: {type(exc).__name__}: {exc}"), None
    first = None
    for _ in range(tries):
        first = opened.read()
        if first is not None:
            break
        if opened.exhausted:
            break
        time.sleep(0.05)
    if first is None:
        opened.close()
        return CheckItem(name, False, f"{label} opened but delivered no frame"), None
    height, width = first.image.shape[:2]
    fps = opened.fps
    fps_text = f"{fps:.1f} fps" if fps else "fps unknown"
    detail = f"{label} opened: granted {width}x{height}, {fps_text}"
    if is_camera:
        detail += f" (requested {CAPTURE_WIDTH}x{CAPTURE_HEIGHT} at {CAPTURE_FPS} fps)"
        if (width, height) != (CAPTURE_WIDTH, CAPTURE_HEIGHT):
            detail += "; the driver granted a different mode"
    return CheckItem(name, True, detail), PrefetchedSource(first, opened)


# --------------------------------------------------------------------------------------------
# The live system
# --------------------------------------------------------------------------------------------


def frame_clock(loop: Any) -> Callable[[], float]:
    """The ``t`` handed to run start / reset: the newest frame's own ``t`` (0.0 before the first
    frame), so ``run_started`` and an aborted run's ``run_completed`` carry video time rather
    than the process uptime ``server.app.create_app`` defaults to (AGENTS.md rule 8)."""

    def now() -> float:
        frame = loop.frame_store.get()
        return 0.0 if frame is None else frame.t

    return now


def recorder_factory_for(source: FrameSource) -> Callable[[Path, float], Any]:
    """The recorder receives every *captured* frame, so its file must be written at the source's
    frame rate, not ``target_fps`` (the loop's default), or it would play back slowed down."""
    from runtime.recorder import Recorder

    def make(path: Path, fps: float) -> Recorder:
        return Recorder(path, source.fps or fps)

    return make


class LiveSystem:
    """capture thread -> LatestFrameStore -> recorder queue; inference thread -> Perception ->
    StateTracker -> Engine -> Router -> JsonlLogger / Speaker / status; Flask threads read only
    the latest-frame store and the status. Built from injected parts, so tests can use fakes."""

    def __init__(
        self,
        *,
        settings: Settings,
        perception: Any,
        source: FrameSource,
        speaker: Any,
        username: str,
        password: str,
        metrics: LiveMetrics | None = None,
        server_factory: Callable[[Any, RuntimeConfig], Any] | None = None,
    ) -> None:
        from engine.sequence import SequenceEngine
        from runtime.loop import RuntimeLoop
        from server.app import RecentAlerts, create_app
        from state.tracker import StateTracker

        self.settings = settings
        self.runtime_config = settings.runtime
        self.speaker = speaker
        self.metrics = metrics
        self.source = source
        self._stopped = False
        self._serving = False

        experiment = settings.experiment
        tracker = StateTracker(experiment, settings.perception)
        engine = SequenceEngine(experiment, settings.runtime)
        self.recent_alerts = RecentAlerts(cap=20)
        router_args = (
            experiment,
            settings.runtime,
            tracker,
            engine,
            speaker,
            settings.runtime.log_dir,
        )
        if metrics is not None:
            self.router: Router = MeteredRouter(
                *router_args, metrics=metrics, on_engine_event=self.recent_alerts
            )
        else:
            self.router = Router(*router_args, on_engine_event=self.recent_alerts)
        self.loop = RuntimeLoop(
            source,
            perception,
            self.router,
            settings.runtime,
            settings.runtime.video_dir,
            recorder_factory=recorder_factory_for(source),
        )
        self.run_clock = frame_clock(self.loop)
        self.app = create_app(
            experiment=experiment,
            runtime_config=settings.runtime,
            router=self.router,
            loop=self.loop,
            log_dir=settings.runtime.log_dir,
            username=username,
            password=password,
            recent_alerts=self.recent_alerts.snapshot,
            run_clock=self.run_clock,
        )
        self._server_factory = server_factory or _make_server
        self.server: Any = None
        self._server_thread: threading.Thread | None = None

    def restart_run(self) -> str:
        """Finishes the current run (an aborted one is logged as such) and starts a fresh one."""
        self.loop.reset_run(self.run_clock())
        return self.loop.start_run(self.run_clock())

    def start(self, *, auto_start: bool) -> None:
        if auto_start:
            self.loop.start_run(0.0)
        self.loop.start_threads()
        self.server = self._server_factory(self.app, self.runtime_config)
        self._server_thread = threading.Thread(
            target=self.server.serve_forever, name="http-server", daemon=True
        )
        self._server_thread.start()
        self._serving = True

    def wait(
        self,
        *,
        duration_s: float | None = None,
        exit_when_done: bool = False,
        stop_event: threading.Event | None = None,
        sample_every_s: float = 30.0,
        poll_s: float = 0.25,
        clock: Callable[[], float] = time.monotonic,
    ) -> str:
        """Blocks until something ends the session and says what: ``stopped`` (the event),
        ``duration``, ``source finished`` (only with ``exit_when_done``), or ``fatal error``."""
        started = clock()
        next_sample = started
        while True:
            now = clock()
            if self.loop.fatal_error is not None:
                return "fatal error"
            if stop_event is not None and stop_event.is_set():
                return "stopped"
            if duration_s is not None and now - started >= duration_s:
                return "duration"
            if exit_when_done and self.loop.stopped:
                return "source finished"
            if self.metrics is not None and now >= next_sample:
                self.metrics.sample_resources()
                next_sample = now + sample_every_s
            time.sleep(poll_s)

    def shutdown(self) -> None:
        """Every step runs even if an earlier one fails: the server stops, the threads join (the
        recorder file closes, the source closes), a running run is finished (``run_completed``
        aborted), the TTS process stops. Idempotent."""
        if self._stopped:
            return
        self._stopped = True
        if self.metrics is not None:
            try:
                self.metrics.sample_resources()
            except Exception:
                logger.warning("live: final resource sample failed", exc_info=True)
        if self._serving and self.server is not None:
            try:
                self.server.shutdown()
                if self._server_thread is not None:
                    self._server_thread.join(timeout=5.0)
                self.server.server_close()
            except Exception:
                logger.warning("live: error stopping the HTTP server", exc_info=True)
        # The threads stop first: with the inference thread still running, a frame could be
        # logged after run_completed. The newest frame stays in the store, so run_clock() still
        # gives the aborted run's end time.
        try:
            self.loop.stop()
        except Exception:
            logger.warning("live: loop.stop failed", exc_info=True)
        try:
            self.loop.reset_run(self.run_clock())
        except Exception:
            logger.warning("live: reset_run failed during shutdown", exc_info=True)
        try:
            self.speaker.close()
        except Exception:
            logger.warning("live: speaker.close failed", exc_info=True)


def _make_server(app: Any, runtime_config: RuntimeConfig) -> Any:
    from werkzeug.serving import make_server

    from server.app import resolve_bind_host

    ssl_context = None
    if tls_is_on(runtime_config):
        ssl_context = (runtime_config.tls_cert, runtime_config.tls_key)
    return make_server(
        resolve_bind_host(runtime_config),
        runtime_config.port,
        app,
        threaded=True,
        ssl_context=ssl_context,
    )


# --------------------------------------------------------------------------------------------
# run_live: preflight, wiring, banner, run, clean stop
# --------------------------------------------------------------------------------------------


@dataclass
class LiveOptions:
    mode: str = "dev"  # "dev" or "demo": only the banner differs
    source: str | None = None
    replay_run: str | None = None
    allow_heldout: bool = False
    loop_source: bool = False
    max_speed: bool = False
    unthrottled: bool = False
    duration_s: float | None = None
    exit_when_done: bool = False
    auto_start: bool | None = None  # None: on for a replay, off for a camera
    metrics_out: Path | None = None
    sample_every_s: float = 30.0
    tls: str = "auto"  # auto | on | off
    port: int | None = None
    check_only: bool = False
    runtime_path: Path = ROOT / "config" / "runtime.yaml"
    perception_path: Path = ROOT / "config" / "perception.yaml"
    experiment_path: Path = ROOT / "config" / "experiment.json"
    manifest_path: Path = ROOT / "weights" / "MANIFEST.json"
    env_path: Path = ROOT / ".env"
    runs_dir: Path = ROOT / "runs"


def effective_runtime(options: LiveOptions, runtime: RuntimeConfig) -> RuntimeConfig:
    """The ``RuntimeConfig`` the run uses: the file, plus ``--port``, ``--tls``, ``--max-speed``."""
    update: dict[str, Any] = {}
    if options.port is not None:
        update["port"] = options.port
    if options.max_speed or options.unthrottled:
        update["target_fps"] = UNTHROTTLED_FPS
    cfg = runtime.model_copy(update=update) if update else runtime
    if options.tls == "off":
        cfg = cfg.model_copy(update={"tls_cert": None, "tls_key": None})
    elif options.tls == "auto":
        cfg = tls_auto(cfg)
    return cfg


def public_url(runtime_config: RuntimeConfig) -> str:
    scheme = "https" if tls_is_on(runtime_config) else "http"
    return f"{scheme}://127.0.0.1:{runtime_config.port}/"


def build_source(
    source_arg: str,
    options: LiveOptions,
    first: FrameSource | None,
    metrics: LiveMetrics | None,
    on_loop: Callable[[int], None] | None,
) -> FrameSource:
    """Wraps the opened source: looping (the soak), real-time pacing (a file), metering."""
    is_file = not source_arg.isdigit()
    src: FrameSource
    if options.loop_source:
        if not is_file:
            raise StartupRefused("--loop needs a file (--replay-run or --source FILE)")
        if first is not None:
            first.close()  # the loop opens its own; the checked one was only for the report
        src = LoopingSource(lambda: _default_opener(source_arg), on_loop=on_loop)
    else:
        assert first is not None
        src = first
    if is_file and not options.max_speed:
        src = PacedSource(src)
    if metrics is not None:
        src = MeteredSource(src, metrics)
    return src


def run_live(
    options: LiveOptions,
    out: Callable[[str], None],
    *,
    stop_event: threading.Event | None = None,
) -> int:
    """Returns the process exit code. All user-facing text goes through ``out``."""
    try:
        source_arg = resolve_source(
            options.source,
            options.replay_run,
            default_source="0",
            runs_dir=options.runs_dir,
            allow_heldout=options.allow_heldout,
        )
    except StartupRefused as exc:
        out(f"refused: {exc}")
        return 2

    items: list[CheckItem] = []

    def record(item: CheckItem) -> CheckItem:
        items.append(item)
        out(item.line())
        return item

    out(f"self-check ({options.mode}):")
    for item in check_weights(options.manifest_path):
        record(item)
    cfg_item, settings = check_configs(
        options.runtime_path, options.perception_path, options.experiment_path
    )
    record(cfg_item)
    cred_item, creds = check_env(options.env_path)
    record(cred_item)
    if settings is None:
        out("refusing to start: fix the FAIL items above")
        return 1
    runtime = effective_runtime(options, settings.runtime)
    settings = Settings(settings.experiment, runtime, settings.perception)
    record(check_tls(runtime))
    from server.app import resolve_bind_host

    bind_host = resolve_bind_host(runtime)
    record(check_port(bind_host, runtime.port))

    source_name = "camera" if source_arg.isdigit() else "video source"
    first: FrameSource | None = None
    if all_ok(items):
        src_item, first = check_source(source_arg)
        record(src_item)
    else:
        record(CheckItem(source_name, False, "not tried: an earlier item failed"))
    record(check_tts())

    perception: Any = None
    entry: dict[str, Any] = {}
    if all_ok(items):
        try:
            from perception.pipeline import load_pipeline

            perception = load_pipeline(options.manifest_path, enable_pose=runtime.enable_pose)
            manifest = json.loads(options.manifest_path.read_text(encoding="utf-8"))
            entry = chosen_detector_entry(manifest)
            record(
                CheckItem(
                    "perception pipeline",
                    True,
                    f"detector {entry['name']} sha256 {entry['sha256'][:12]}, "
                    f"model_stamp {perception.model_stamp}",
                )
            )
        except Exception as exc:
            record(CheckItem("perception pipeline", False, f"{type(exc).__name__}: {exc}"))
    else:
        record(CheckItem("perception pipeline", False, "not tried: an earlier item failed"))

    if not all_ok(items) or options.check_only:
        if first is not None:
            first.close()
        if not all_ok(items):
            out(
                "NOT READY: fix the FAIL items above"
                if options.check_only
                else "refusing to start: fix the FAIL items above"
            )
            return 1
        out("self-check passed")
        return 0

    assert creds is not None and perception is not None
    logger.info(
        "starting: detector %s (sha256 %s), model_stamp %s, target_fps %g, source %s, "
        "TLS %s, bind %s:%d",
        entry["name"],
        entry["sha256"],
        perception.model_stamp,
        runtime.target_fps,
        source_arg,
        "on" if tls_is_on(runtime) else "off",
        bind_host,
        runtime.port,
    )
    metrics = LiveMetrics() if options.metrics_out is not None else None
    system: LiveSystem | None = None
    restart: list[Callable[[], Any]] = []
    try:
        wrapped = build_source(
            source_arg,
            options,
            first,
            metrics,
            on_loop=lambda n: restart[0]() if restart else None,
        )
        system = LiveSystem(
            settings=settings,
            perception=perception,
            source=wrapped,
            speaker=_make_speaker(runtime),
            username=creds[0],
            password=creds[1],
            metrics=metrics,
        )
        restart.append(system.restart_run)
        auto_start = options.auto_start
        if auto_start is None:
            auto_start = not source_arg.isdigit()
        system.start(auto_start=auto_start)
        realtime = not source_arg.isdigit() and not options.max_speed
        out(f"READY  {public_url(runtime)}")
        out(f"  login:   credentials from the {credentials_origin(cred_item)} (values not shown)")
        out(f"  logs:    {runtime.log_dir}")
        out(f"  video:   {runtime.video_dir}")
        out(f"  source:  {source_arg}" + ("  (played in real time)" if realtime else ""))
        start_note = "run started automatically" if auto_start else "press Start on the dashboard"
        out(f"  {start_note}; Ctrl+C stops cleanly")
        reason = system.wait(
            duration_s=options.duration_s,
            exit_when_done=options.exit_when_done,
            stop_event=stop_event,
            sample_every_s=options.sample_every_s,
        )
        out(f"stopping ({reason})")
        return 1 if reason == "fatal error" else 0
    except KeyboardInterrupt:
        out("stopping (Ctrl+C)")
        return 0
    except Exception as exc:
        logger.exception("live run failed")
        out(f"failed: {type(exc).__name__}: {exc}")
        return 1
    finally:
        if system is not None:
            system.shutdown()
        elif first is not None:
            first.close()
        if metrics is not None and options.metrics_out is not None:
            _write_metrics(options.metrics_out, metrics, runtime, source_arg, perception, system)
            out(f"metrics written to {options.metrics_out}")


def credentials_origin(item: CheckItem) -> str:
    """'.env file (path)' or 'process environment ...', cut out of the credentials item."""
    return item.detail.split(", from the ", 1)[-1]


def _make_speaker(runtime: RuntimeConfig) -> Any:
    from outputs.tts import TTSWorker

    return TTSWorker(cooldown_s=runtime.alert_cooldown_s)


def _write_metrics(
    path: Path,
    metrics: LiveMetrics,
    runtime: RuntimeConfig,
    source: str,
    perception: Any,
    system: LiveSystem | None,
) -> None:
    report = metrics.report()
    report["target_fps"] = runtime.target_fps
    report["source"] = source
    report["model_stamp"] = getattr(perception, "model_stamp", None)
    if system is not None:
        report["loops"] = _count_loops(system.source)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")


def _count_loops(source: Any) -> int | None:
    node = source
    while node is not None:
        if isinstance(node, LoopingSource):
            return node.loops
        node = getattr(node, "_inner", None)
    return None
