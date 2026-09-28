"""runtime/loop.py -- RuntimeLoop, LatestFrameStore, Router (F5-F9, F11,
F13; IMPLEMENTATION_PLAN.md 5.6; essential-features.md section 0 and #13).

Threading model (essential-features.md section 0):

    capture thread   : source.read() -> LatestFrameStore (latest-frame-wins)
                        + recorder queue (F11)
    inference thread : take newest frame (skip if same frame_id), throttled
                        to target_fps -> Perception.process ->
                        Router.process_perception_frame
    router            : StateTracker -> Engine -> LogSink / Speaker

``Router`` is the synchronous, thread-free per-frame/per-run business logic
shared by the live ``RuntimeLoop`` and ``harness/replay.py`` (which drives
it sequentially, with no threads and no sleeps -- essential-features.md
section 0, "File replay (harness) has no threads"). It also owns run_id
generation and the per-run ``JsonlLogger`` (a fresh log file per run_id,
IMPLEMENTATION_PLAN.md 5.2).

Run-state note (ISSUES.md, 2026-09-28 P2.2 DECISION): ``contracts.Engine``
has no method that ever re-produces ``run_state == "idle"`` once a run has
started -- only a freshly constructed ``SequenceEngine`` starts idle, and
``finish()``/natural completion both leave it at ``"completed"``. ``Router``
closes that gap itself (rather than in ``contracts.py``): it tracks its own
``_idle_armed`` flag, set at construction and by ``reset()``, cleared by
``start()``, and reports ``"idle"`` whenever it is set regardless of what
the underlying ``Engine.run_state`` says. This is the resolution the P2.2
entry asked P2.4 to make.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from contracts import (
    ContractViolation,
    Engine,
    EngineEvent,
    ExperimentDefinition,
    Frame,
    FrameSource,
    LogEntry,
    LogEventType,
    Perception,
    PerceptionFrame,
    RunState,
    RuntimeConfig,
    Speaker,
    StateEvent,
    StateTracker,
)
from outputs.logger import JsonlLogger
from runtime.recorder import Recorder

logger = logging.getLogger(__name__)


def default_run_id() -> str:
    """Live runtime run ids: a readable UTC timestamp plus a short random
    suffix, so two resets within the same second never collide (a
    disclosed judgment call -- ISSUES.md, 2026-09-28 P2.4; recorded runs
    use the plan's own ``<split>-<n>`` ids, RunScript, unrelated to this)."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"live-{stamp}-{uuid.uuid4().hex[:6]}"


class LatestFrameStore:
    """Thread-safe holder for the single newest ``Frame``. ``put`` always
    overwrites; ``get`` never blocks. The "latest-frame-wins" store of
    essential-features.md section 0: if inference is slower than capture,
    intermediate frames are dropped, never queued, so latency never grows."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: Frame | None = None

    def put(self, frame: Frame) -> None:
        with self._lock:
            self._frame = frame

    def get(self) -> Frame | None:
        with self._lock:
            return self._frame


def _display_names(experiment: ExperimentDefinition) -> dict[str, str]:
    return {s.step_id: s.display_name for s in experiment.steps}


def _log_detail(event: EngineEvent, names: dict[str, str]) -> str:
    """The fixed one-liner logged independently of narration settings
    (IMPLEMENTATION_PLAN.md 5.5). Reconstructed rather than read from
    ``event.speak``, because ``speak`` is ``None`` exactly when
    ``narrate_next_step``/``alert_on_repeat`` silence it -- but the log
    entry must record what happened regardless of whether it was spoken."""

    if event.kind == "step_confirmed":
        assert event.step_id is not None
        return f"Confirmed: {names[event.step_id]}"
    if event.kind == "deviation_detected":
        if event.deviation_type == "repeat":
            assert event.step_id is not None
            return f"Repeated: {names[event.step_id]}"
        assert event.speak is not None  # omission/out_of_order always speak
        return event.speak
    assert event.speak is not None  # run_completed always speaks
    return event.speak


class Router:
    """Synchronous per-frame, per-run routing: ``StateTracker.update`` ->
    ``Engine.on_state_event`` -> ``LogSink``/``Speaker``. Shared by the
    threaded ``RuntimeLoop`` and ``harness/replay.py``.

    ``tracker`` may be ``None`` for callers that only ever use
    ``process_state_event`` directly (``harness/replay.py``'s ``--script``
    mode, which bypasses ``StateTracker`` exactly like
    ``engine/reference.py``'s ``derive_expected_deviations`` does).

    Exceptions from ``StateTracker``/``Engine``/``LogSink`` are never
    caught here: they are contract violations, not per-frame perception
    noise, and AGENTS.md rule "ContractViolation is raised, never coerced"
    (essential-features.md section 0) applies. Only ``RuntimeLoop`` catches
    (around the ``Perception.process`` call specifically, plus a fatal-error
    backstop around everything else -- see its docstring).

    Thread-safe: every public method takes an internal ``RLock`` (reentrant
    so ``process_perception_frame`` can call ``process_state_event``
    without deadlocking itself). This matters once a run-control caller
    (``RuntimeLoop.start_run``/``reset_run``, and eventually P2.5's Flask
    ``POST /api/run/start``/``/reset`` handlers) runs on a different thread
    than the inference loop that calls ``process_perception_frame`` --
    essential-features.md section 0 explicitly puts run control and the
    inference loop on different threads."""

    def __init__(
        self,
        experiment: ExperimentDefinition,
        runtime_config: RuntimeConfig,
        tracker: StateTracker | None,
        engine: Engine,
        speaker: Speaker,
        log_dir: str | Path,
        wall_clock: Callable[[], datetime] | None = None,
        run_id_factory: Callable[[], str] = default_run_id,
        on_engine_event: Callable[[EngineEvent], None] | None = None,
    ) -> None:
        self._experiment = experiment
        self._runtime_config = runtime_config
        self._tracker = tracker
        self._engine = engine
        self._speaker = speaker
        self._log_dir = Path(log_dir)
        self._wall_clock = wall_clock or (lambda: datetime.now(UTC))
        self._run_id_factory = run_id_factory
        self._on_engine_event = on_engine_event
        self._names = _display_names(experiment)

        self._log_sink: JsonlLogger | None = None
        self._seq = 0
        self._run_id: str | None = None
        self._pending_run_id = run_id_factory()
        self._idle_armed = True
        self._lock = threading.RLock()

    @property
    def engine(self) -> Engine:
        """Read-only access for server/app.py's ``GET /api/status`` handler
        (F12), which needs ``Engine.snapshot()`` to render step statuses.
        Handlers only ever read this -- they never call a mutating method
        on it directly (run control goes through ``start``/``reset`` above)."""
        return self._engine

    @property
    def run_id(self) -> str | None:
        with self._lock:
            return self._run_id if not self._idle_armed else self._pending_run_id

    @property
    def run_state(self) -> RunState:
        with self._lock:
            if self._idle_armed:
                return "idle"
            return self._engine.run_state

    def start(self, t: float) -> str:
        with self._lock:
            if not self._idle_armed:
                raise ContractViolation(
                    "Router.start() called while a run is already active "
                    "(idle_armed=False); call reset() first"
                )
            run_id = self._pending_run_id
            self._run_id = run_id
            self._log_sink = JsonlLogger(self._log_dir / f"{run_id}.jsonl", self._wall_clock)
            self._seq = 0
            if self._tracker is not None:
                self._tracker.reset()
            self._engine.start(t, run_id=run_id)
            self._idle_armed = False
            self._write_runtime_event("run_started", t, f"Run started: {run_id}")
            if self._runtime_config.narrate_next_step and self._experiment.steps:
                self._speaker.say(self._experiment.steps[0].say, "info")
            return run_id

    def reset(self, t: float) -> str:
        with self._lock:
            if self._engine.run_state == "running":
                self._dispatch(self._engine.finish(t))
            self._pending_run_id = self._run_id_factory()
            self._idle_armed = True
            return self._pending_run_id

    def process_perception_frame(self, frame: PerceptionFrame) -> None:
        with self._lock:
            if self._tracker is None:
                raise ContractViolation("Router.process_perception_frame requires a tracker")
            if self.run_state != "running":
                return
            for state_event in self._tracker.update(frame):
                self.process_state_event(state_event)

    def process_state_event(self, event: StateEvent) -> None:
        with self._lock:
            if self.run_state != "running":
                return
            self._dispatch(self._engine.on_state_event(event))

    def write_feed_lost(self, t_video: float, detail: str) -> None:
        with self._lock:
            if self.run_state == "running":
                self._write_runtime_event("feed_lost", t_video, detail)

    def write_feed_restored(self, t_video: float, detail: str) -> None:
        with self._lock:
            if self.run_state == "running":
                self._write_runtime_event("feed_restored", t_video, detail)

    def _dispatch(self, events: list[EngineEvent]) -> None:
        for event in events:
            self._write_engine_event(event)

    def _write_engine_event(self, event: EngineEvent) -> None:
        entry = LogEntry(
            seq=self._next_seq(),
            run_id=self._run_id or "",
            t_video=event.t,
            event_type=event.kind,
            step_id=event.step_id,
            expected_step_id=event.expected_step_id,
            deviation_type=event.deviation_type,
            skipped_step_ids=event.skipped_step_ids,
            confidence_tag=event.confidence_tag,
            detail=_log_detail(event, self._names),
        )
        self._write(entry)
        if event.speak is not None:
            priority = "alert" if event.kind == "deviation_detected" else "info"
            self._speaker.say(event.speak, priority)
        if self._on_engine_event is not None:
            self._on_engine_event(event)

    def _write_runtime_event(self, event_type: LogEventType, t_video: float, detail: str) -> None:
        entry = LogEntry(
            seq=self._next_seq(),
            run_id=self._run_id or "",
            t_video=t_video,
            event_type=event_type,
            detail=detail,
        )
        self._write(entry)

    def _write(self, entry: LogEntry) -> None:
        if self._log_sink is None:
            return
        self._log_sink.write(entry)

    def _next_seq(self) -> int:
        seq = self._seq
        self._seq += 1
        return seq


def _default_recorder_factory(path: Path, fps: float) -> Recorder:
    return Recorder(path, fps)


class _FpsMeter:
    """Measured processing rate over the last ``window`` successfully
    processed frames (F12's ``StatusResponse.fps`` -- diagnostic only, not
    used by any perception/state/engine logic, so this is not a "clock
    read" under AGENTS.md rule 8). ``None`` until at least two frames have
    been processed."""

    def __init__(self, window: int = 30) -> None:
        self._times: deque[float] = deque(maxlen=window)

    def tick(self, now: float) -> None:
        self._times.append(now)

    def value(self) -> float | None:
        if len(self._times) < 2:
            return None
        span = self._times[-1] - self._times[0]
        if span <= 0.0:
            return None
        return (len(self._times) - 1) / span


class RuntimeLoop:
    """Threaded orchestration (essential-features.md section 0): a capture
    thread feeding ``LatestFrameStore`` (+ the recorder queue) and a
    throttled inference thread driving ``Perception`` -> ``Router``.
    Feed-loss handling and per-frame exception isolation live here; the
    frame-by-frame decision logic lives in ``Router``, shared with the
    no-thread ``harness/replay.py`` path."""

    def __init__(
        self,
        source: FrameSource,
        perception: Perception,
        router: Router,
        runtime_config: RuntimeConfig,
        video_dir: str | Path,
        recorder_factory: Callable[[Path, float], Recorder] = _default_recorder_factory,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._source = source
        self._perception = perception
        self._router = router
        self._runtime_config = runtime_config
        self._video_dir = Path(video_dir)
        self._recorder_factory = recorder_factory
        self._clock = clock

        self._store = LatestFrameStore()
        self._stop_event = threading.Event()
        self._capture_thread: threading.Thread | None = None
        self._inference_thread: threading.Thread | None = None
        self._recorder: Recorder | None = None
        self._control_lock = threading.Lock()

        self._last_captured_at: float | None = None  # monotonic
        self._last_frame_t: float = 0.0
        self._feed_ok = True
        self._fatal_error: BaseException | None = None
        self._fps_meter = _FpsMeter()

    @property
    def feed_ok(self) -> bool:
        return self._feed_ok

    @property
    def fps(self) -> float | None:
        """Measured processing rate (server/app.py's ``StatusResponse.fps``,
        F12) -- ``None`` until enough frames have been processed to measure
        a rate."""
        return self._fps_meter.value()

    @property
    def frame_store(self) -> LatestFrameStore:
        """The raw-frame store server/app.py's ``/video_feed`` reads from
        (F10) -- the same store the capture thread ``put()``s into, read-only
        from the handler's side."""
        return self._store

    @property
    def stopped(self) -> bool:
        return self._stop_event.is_set()

    @property
    def fatal_error(self) -> BaseException | None:
        """Set when the inference thread stopped itself because of an
        unexpected (non-perception) exception -- e.g. a ``ContractViolation``
        from ``StateTracker``/``Engine``/``LogSink`` -- rather than because
        the source was exhausted or ``stop()`` was called. ``stopped`` is
        ``True`` in both cases; this distinguishes them."""
        return self._fatal_error

    def start_run(self, t: float) -> str:
        with self._control_lock:
            run_id = self._router.start(t)
            recorder = self._recorder_factory(
                self._video_dir / f"{run_id}.avi", self._runtime_config.target_fps
            )
            recorder.open()
            self._recorder = recorder
            return run_id

    def reset_run(self, t: float) -> str:
        with self._control_lock:
            if self._recorder is not None:
                self._recorder.close()
                self._recorder = None
            return self._router.reset(t)

    def start_threads(self) -> None:
        if (self._capture_thread is not None and self._capture_thread.is_alive()) or (
            self._inference_thread is not None and self._inference_thread.is_alive()
        ):
            raise ContractViolation(
                "RuntimeLoop.start_threads() called while the previous threads are "
                "still running; call stop() first"
            )
        self._stop_event.clear()
        self._fatal_error = None
        self._last_captured_at = self._clock()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._capture_thread.start()
        self._inference_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=5.0)
        if self._inference_thread is not None:
            self._inference_thread.join(timeout=5.0)
        with self._control_lock:
            if self._recorder is not None:
                self._recorder.close()
                self._recorder = None
        try:
            self._source.close()
        except Exception:
            logger.warning("RuntimeLoop: error closing frame source", exc_info=True)

    def _capture_loop(self) -> None:
        # Ends the thread on exhaustion but deliberately does NOT set
        # ``_stop_event`` itself: doing so here raced the inference thread
        # for the very last frame (the capture thread could put() it and
        # terminate within the same GIL slice, before the inference thread
        # ever wakes to read it). The inference loop below decides when to
        # stop, only once it has confirmed (by seeing the same last
        # frame_id again) that it already processed the truly last frame.
        while not self._stop_event.is_set():
            try:
                frame = self._source.read()
            except Exception:
                logger.warning("RuntimeLoop: frame source read failed", exc_info=True)
                continue
            if frame is None:
                if self._source.exhausted:
                    return
                continue  # transient read failure, not end-of-stream
            self._store.put(frame)
            if self._recorder is not None:
                self._recorder.enqueue(frame)

    def _inference_loop(self) -> None:
        period = 1.0 / self._runtime_config.target_fps
        last_frame_id: int | None = None
        next_deadline = self._clock()
        while not self._stop_event.is_set():
            now = self._clock()
            if now < next_deadline:
                time.sleep(min(next_deadline - now, 0.05))
                continue
            next_deadline = now + period

            try:
                frame = self._store.get()
                if frame is None or frame.frame_id == last_frame_id:
                    self._check_feed_timeout(now)
                    if self._source.exhausted:
                        # The source has nothing left, and the frame current
                        # in the store (if any) was already processed on a
                        # prior iteration -- safe to stop.
                        self._stop_event.set()
                        return
                    continue
                last_frame_id = frame.frame_id
                self._on_frame_captured(now, frame.t)

                try:
                    pframe = self._perception.process(frame)
                except Exception:
                    # Routine, expected: a single bad frame in perception
                    # never crashes the loop (essential-features.md section
                    # 0, "Errors").
                    logger.warning(
                        "RuntimeLoop: perception failed on frame %d, skipping",
                        frame.frame_id,
                        exc_info=True,
                    )
                    continue
                self._router.process_perception_frame(pframe)
                self._fps_meter.tick(now)
            except Exception as exc:
                # Anything else (StateTracker/Engine/LogSink -- a genuine
                # ContractViolation, or the feed watchdog's own log write)
                # is NOT routine per-frame noise: it means tracking is now
                # in an unknown state, so continuing could silently miss a
                # real deviation. Stop observably (fatal_error/stopped)
                # rather than let the thread die silently as a zombie that
                # never processes another frame again.
                logger.error(
                    "RuntimeLoop: unexpected error in the inference loop, stopping",
                    exc_info=True,
                )
                self._fatal_error = exc
                self._stop_event.set()
                return

    def _on_frame_captured(self, now: float, frame_t: float) -> None:
        self._last_captured_at = now
        self._last_frame_t = frame_t
        if not self._feed_ok:
            self._feed_ok = True
            self._router.write_feed_restored(frame_t, "Feed restored")

    def _check_feed_timeout(self, now: float) -> None:
        if self._last_captured_at is None:
            return
        elapsed = now - self._last_captured_at
        if self._feed_ok and elapsed >= self._runtime_config.feed_timeout_s:
            self._feed_ok = False
            detail = f"No frame received for {elapsed:.1f}s"
            self._router.write_feed_lost(self._last_frame_t, detail)
