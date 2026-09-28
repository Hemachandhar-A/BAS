"""server/app.py -- Flask app factory (F10, F12; IMPLEMENTATION_PLAN.md 5.6;
essential-features.md sections 10 and 12).

``create_app`` takes the already-wired stores and ``RuntimeConfig``, not a
``RuntimeLoop`` instance directly -- handlers only ever *read* them
(essential-features.md #10 pitfall: "an inference call inside a request
handler is forbidden"), and tests can hand it lightweight fakes instead of a
threaded ``RuntimeLoop`` that needs a real ``FrameSource``/``Perception``.

Route count is deliberately exactly ``len(API_ROUTES)``: Flask's own
implicit ``static`` endpoint (serving ``server/static/`` by path) is
disabled (``static_folder=None``) and the dashboard's CSS/JS are inlined
into the single ``GET /`` response instead, so ``app.url_map`` equals
``API_ROUTES`` exactly, not "``API_ROUTES`` plus a framework extra" --
matching the contract's "the Flask URL map must equal it" wording literally
rather than by an implicit carve-out.
"""

from __future__ import annotations

import hmac
import json
import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import cv2
from flask import Flask, Response, jsonify, request

from contracts import (
    ContractViolation,
    EngineEvent,
    ExperimentDefinition,
    Frame,
    RunControlResponse,
    RuntimeConfig,
    StatusResponse,
    StepProgress,
)
from runtime.loop import LatestFrameStore, Router

STATIC_DIR = Path(__file__).parent / "static"


class RuntimeLoopLike(Protocol):
    """Structural subset of ``runtime.loop.RuntimeLoop`` that this module
    depends on. A real ``RuntimeLoop`` satisfies this by construction (see
    its ``feed_ok``/``fps``/``frame_store``/``start_run``/``reset_run``);
    tests substitute a lightweight fake so a Flask route test never needs a
    real ``FrameSource`` or camera."""

    @property
    def feed_ok(self) -> bool: ...

    @property
    def fps(self) -> float | None: ...

    @property
    def frame_store(self) -> LatestFrameStore: ...

    def start_run(self, t: float) -> str: ...

    def reset_run(self, t: float) -> str: ...


class RecentAlerts:
    """Thread-safe, bounded history of alert-worthy ``EngineEvent``s
    (``kind == "deviation_detected"`` -- the events the runtime speaks at
    ``alert`` priority, per ``runtime/loop.py``'s ``Router``), newest-last,
    capped at ``RECENT_ALERTS_CAP``. ``step_confirmed``/``run_completed``
    are routine progress, not alerts, so they are not collected here.

    Callable so it can be wired directly as ``Router(..., on_engine_event=
    recent_alerts)``; ``snapshot()`` is read by ``GET /api/status``. The
    lock matters because the callback fires from the inference thread while
    a request thread may be reading ``snapshot()`` concurrently."""

    def __init__(self, cap: int) -> None:
        self._lock = threading.Lock()
        self._events: deque[EngineEvent] = deque(maxlen=cap)

    def __call__(self, event: EngineEvent) -> None:
        if event.kind == "deviation_detected":
            with self._lock:
                self._events.append(event)

    def snapshot(self) -> list[EngineEvent]:
        with self._lock:
            return list(self._events)


class _JpegCache:
    """Encodes the newest ``Frame`` to JPEG at most once, sharing the same
    bytes across every concurrently-connected ``/video_feed`` client
    (essential-features.md #10: "encode once per new frame and share it
    across clients")."""

    def __init__(self, jpeg_quality: int) -> None:
        self._quality = jpeg_quality
        self._lock = threading.Lock()
        self._frame_id: int | None = None
        self._jpg: bytes | None = None

    def encode(self, frame: Frame) -> bytes:
        with self._lock:
            if frame.frame_id != self._frame_id or self._jpg is None:
                ok, buf = cv2.imencode(
                    ".jpg", frame.image, [cv2.IMWRITE_JPEG_QUALITY, self._quality]
                )
                if not ok:
                    raise RuntimeError("server/app.py: JPEG encode failed")
                self._jpg = buf.tobytes()
                self._frame_id = frame.frame_id
            return self._jpg


def resolve_bind_host(runtime_config: RuntimeConfig) -> str:
    """AGENTS.md rule 14 / essential-features.md #10: "without TLS the
    server binds 127.0.0.1 regardless of RuntimeConfig.host". TLS is
    considered configured only when both the cert and key are set."""
    if runtime_config.tls_cert and runtime_config.tls_key:
        return runtime_config.host
    return "127.0.0.1"


def _pending_step_ids(steps: list[StepProgress]) -> list[str]:
    return [s.step_id for s in steps if s.status == "pending"]


def _step_guidance(
    steps: list[StepProgress], say_by_id: dict[str, str]
) -> tuple[str | None, str | None, str | None]:
    """Derives ``StatusResponse``'s ``expected_step_id``/``next_step_id``/
    ``next_step_say`` from a plain snapshot poll (no in-flight
    ``EngineEvent`` to read them from). Mirrors ``EngineEvent``'s own
    naming: ``expected_step_id`` is the step now pending (what the operator
    should be doing), ``next_step_id`` is the step after it, and
    ``next_step_say`` is that next step's spoken guidance -- so the
    dashboard's single "current/next step" element (essential-features.md
    #12) can show both without a second lookup."""
    pending = _pending_step_ids(steps)
    expected_step_id = pending[0] if pending else None
    next_step_id = pending[1] if len(pending) > 1 else None
    next_step_say = say_by_id.get(next_step_id) if next_step_id is not None else None
    return expected_step_id, next_step_id, next_step_say


def _render_index() -> str:
    """Inlines ``style.css``/``app.js`` into ``index.html`` once (they are
    static files on disk, read a single time at app-creation, never per
    request)."""
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    css = (STATIC_DIR / "style.css").read_text(encoding="utf-8")
    js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    html = html.replace("/*__STYLE_CSS__*/", css)
    html = html.replace("/*__APP_JS__*/", js)
    return html


def create_app(
    *,
    experiment: ExperimentDefinition,
    runtime_config: RuntimeConfig,
    router: Router,
    loop: RuntimeLoopLike,
    log_dir: str | Path,
    username: str,
    password: str,
    recent_alerts: Callable[[], list[EngineEvent]],
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    run_clock: Callable[[], float] = time.monotonic,
) -> Flask:
    """Builds the Flask app. Every route in ``API_ROUTES`` requires the
    client's IP to be in ``runtime_config.allowed_client_ips`` *and* valid
    HTTP Basic auth (``username``/``password`` -- read from ``.env`` by the
    caller via ``server/env.py``, never hardcoded); this module never reads
    the environment itself, so tests can pass arbitrary test credentials
    without touching ``.env`` (IMPLEMENTATION_PLAN.md 5.6)."""

    app = Flask(__name__, static_folder=None)
    index_html = _render_index()
    say_by_id = {s.step_id: s.say for s in experiment.steps}
    log_dir = Path(log_dir)
    jpeg_cache = _JpegCache(runtime_config.jpeg_quality)
    allowed_ips = frozenset(runtime_config.allowed_client_ips)

    @app.before_request
    def _enforce_security() -> Response | None:
        # Order per essential-features.md #10: IP allowlist first, then
        # Basic auth -- an unlisted IP never even learns auth is required.
        if request.remote_addr not in allowed_ips:
            return Response(status=403)
        auth = request.authorization
        user_ok = auth is not None and hmac.compare_digest(auth.username or "", username)
        pass_ok = auth is not None and hmac.compare_digest(auth.password or "", password)
        if not (user_ok and pass_ok):
            return Response(
                status=401,
                headers={"WWW-Authenticate": 'Basic realm="BAS"'},
            )
        return None

    @app.get("/")
    def index() -> Response:
        return Response(index_html, mimetype="text/html")

    @app.get("/video_feed")
    def video_feed() -> Response:
        period = 1.0 / runtime_config.stream_fps

        def gen():
            while True:
                frame = loop.frame_store.get()
                if frame is not None:
                    jpg = jpeg_cache.encode(frame)
                    yield (
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                    )
                time.sleep(period)

        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/status")
    def get_status() -> Response:
        steps = router.engine.snapshot()
        expected_step_id, next_step_id, next_step_say = _step_guidance(steps, say_by_id)
        body = StatusResponse(
            run_id=router.run_id,
            run_state=router.run_state,
            feed_ok=loop.feed_ok,
            fps=loop.fps,
            steps=steps,
            expected_step_id=expected_step_id,
            next_step_id=next_step_id,
            next_step_say=next_step_say,
            recent_alerts=recent_alerts(),
            generated_at=clock(),
        )
        return Response(body.model_dump_json(), mimetype="application/json")

    @app.get("/api/experiment")
    def get_experiment() -> Response:
        return Response(experiment.model_dump_json(), mimetype="application/json")

    @app.get("/api/log")
    def get_log() -> Response:
        run_id = request.args.get("run_id") or router.run_id
        if not run_id:
            return jsonify({"run_id": None, "entries": []})
        path = log_dir / f"{run_id}.jsonl"
        if not path.exists():
            return jsonify({"error": f"no log for run_id {run_id!r}"}), 404
        entries: list[dict] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                # essential-features.md #9: "ignore an incomplete trailing
                # line" -- a run killed mid-write leaves every earlier line
                # valid and only the last line broken.
                continue
        return jsonify({"run_id": run_id, "entries": entries})

    @app.post("/api/run/start")
    def post_run_start() -> tuple[Response, int] | Response:
        try:
            run_id = loop.start_run(run_clock())
        except ContractViolation as exc:
            return jsonify({"ok": False, "error": str(exc)}), 409
        body = RunControlResponse(ok=True, run_id=run_id, run_state=router.run_state)
        return Response(body.model_dump_json(), mimetype="application/json")

    @app.post("/api/run/reset")
    def post_run_reset() -> Response:
        run_id = loop.reset_run(run_clock())
        body = RunControlResponse(ok=True, run_id=run_id, run_state=router.run_state)
        return Response(body.model_dump_json(), mimetype="application/json")

    return app


def run_app(app: Flask, runtime_config: RuntimeConfig) -> None:
    """Starts the Flask development server (adequate for one operator on
    one stream, essential-features.md #10 point 4 -- a production WSGI
    server is out of scope). Blocks until interrupted."""
    ssl_context = None
    if runtime_config.tls_cert and runtime_config.tls_key:
        ssl_context = (runtime_config.tls_cert, runtime_config.tls_key)
    app.run(
        host=resolve_bind_host(runtime_config),
        port=runtime_config.port,
        ssl_context=ssl_context,
        threaded=True,
    )
