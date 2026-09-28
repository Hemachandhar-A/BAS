"""tests/unit/server/test_app.py -- Flask test-client tests for
server/app.py (F10, F12; IMPLEMENTATION_PLAN.md 5.6, essential-features.md
sections 10 and 12). No camera, no threads: RuntimeLoop is stood in for by
a tiny fake (server/app.py's RuntimeLoopLike Protocol), and Router is
driven directly, exactly as harness/replay.py does."""

from __future__ import annotations

import base64
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from flask import Flask

from contracts import (
    API_ROUTES,
    EngineEvent,
    ExperimentDefinition,
    Frame,
    RunControlResponse,
    RuntimeConfig,
    StatusResponse,
)
from engine.sequence import SequenceEngine
from outputs.tts import FakeSpeaker
from runtime.loop import LatestFrameStore, Router
from server.app import RecentAlerts, _JpegCache, _render_index, create_app, resolve_bind_host

FIXTURE_PATH = Path(__file__).resolve().parents[3] / "fixtures" / "experiment_4step.json"
USERNAME = "op"
PASSWORD = "s3cret"
AUTH_HEADER = "Basic " + base64.b64encode(f"{USERNAME}:{PASSWORD}".encode()).decode()


@pytest.fixture()
def experiment() -> ExperimentDefinition:
    return ExperimentDefinition.from_json(FIXTURE_PATH)


class _FakeLoop:
    """A minimal ``RuntimeLoopLike`` double: no camera, no threads. Wraps
    the same ``Router.start``/``reset`` a real ``RuntimeLoop.start_run``/
    ``reset_run`` would call (real ``RuntimeLoop`` additionally opens/closes
    a ``Recorder``, out of scope for a route test -- F11 owns that)."""

    def __init__(self, router: Router, frame_store: LatestFrameStore) -> None:
        self._router = router
        self.frame_store = frame_store
        self.feed_ok = True
        self.fps: float | None = 12.5

    def start_run(self, t: float) -> str:
        return self._router.start(t)

    def reset_run(self, t: float) -> str:
        return self._router.reset(t)


def _make_app(
    tmp_path: Path,
    experiment: ExperimentDefinition,
    allowed_ips: list[str] | None = None,
) -> tuple[Flask, Router, _FakeLoop, LatestFrameStore]:
    runtime_config = RuntimeConfig(
        allowed_client_ips=allowed_ips if allowed_ips is not None else ["127.0.0.1"],
    )
    engine = SequenceEngine(experiment, runtime_config)
    recent_alerts = RecentAlerts(cap=20)
    router = Router(
        experiment,
        runtime_config,
        None,
        engine,
        FakeSpeaker(),
        tmp_path,
        run_id_factory=_counting_run_ids(),
        on_engine_event=recent_alerts,
    )
    frame_store = LatestFrameStore()
    loop = _FakeLoop(router, frame_store)
    app = create_app(
        experiment=experiment,
        runtime_config=runtime_config,
        router=router,
        loop=loop,
        log_dir=tmp_path,
        username=USERNAME,
        password=PASSWORD,
        recent_alerts=recent_alerts.snapshot,
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )
    return app, router, loop, frame_store


def _counting_run_ids():
    counter = iter(range(1000))

    def factory() -> str:
        return f"test-run-{next(counter)}"

    return factory


# ---------------------------------------------------------------------------
# F10 -- routes, auth, IP allowlist, URL map, TLS/bind policy
# ---------------------------------------------------------------------------


@pytest.mark.F10
def test_url_map_equals_api_routes_exactly(
    tmp_path: Path, experiment: ExperimentDefinition
) -> None:
    app, *_ = _make_app(tmp_path, experiment)
    live = {
        rule.rule: frozenset(m for m in rule.methods if m not in {"HEAD", "OPTIONS"})
        for rule in app.url_map.iter_rules()
    }
    assert live == API_ROUTES


@pytest.mark.F10
def test_missing_auth_returns_401_with_challenge(
    tmp_path: Path, experiment: ExperimentDefinition
) -> None:
    app, *_ = _make_app(tmp_path, experiment)
    resp = app.test_client().get("/api/status")
    assert resp.status_code == 401
    assert resp.headers["WWW-Authenticate"].startswith("Basic")


@pytest.mark.F10
def test_wrong_password_returns_401(tmp_path: Path, experiment: ExperimentDefinition) -> None:
    app, *_ = _make_app(tmp_path, experiment)
    bad = "Basic " + base64.b64encode(f"{USERNAME}:wrong".encode()).decode()
    resp = app.test_client().get("/api/status", headers={"Authorization": bad})
    assert resp.status_code == 401


@pytest.mark.F10
def test_correct_credentials_are_accepted(
    tmp_path: Path, experiment: ExperimentDefinition
) -> None:
    app, *_ = _make_app(tmp_path, experiment)
    resp = app.test_client().get("/api/status", headers={"Authorization": AUTH_HEADER})
    assert resp.status_code == 200


@pytest.mark.F10
def test_ip_not_in_allowlist_returns_403_even_with_valid_auth(
    tmp_path: Path, experiment: ExperimentDefinition
) -> None:
    app, *_ = _make_app(tmp_path, experiment, allowed_ips=["10.0.0.9"])
    resp = app.test_client().get("/api/status", headers={"Authorization": AUTH_HEADER})
    assert resp.status_code == 403


@pytest.mark.F10
def test_every_route_is_protected(tmp_path: Path, experiment: ExperimentDefinition) -> None:
    """Every route in API_ROUTES, hit with its declared method and no
    credentials, must be rejected -- not just the ones this file happens to
    exercise elsewhere (essential-features.md #10: "every route")."""
    app, *_ = _make_app(tmp_path, experiment)
    client = app.test_client()
    for path, methods in API_ROUTES.items():
        for method in methods:
            resp = client.open(path, method=method)
            assert resp.status_code in (401, 403), f"{method} {path} was not protected"


@pytest.mark.F10
def test_credentials_never_appear_in_a_response(
    tmp_path: Path, experiment: ExperimentDefinition
) -> None:
    app, *_ = _make_app(tmp_path, experiment)
    resp = app.test_client().get("/api/status", headers={"Authorization": AUTH_HEADER})
    assert PASSWORD not in resp.get_data(as_text=True)
    assert PASSWORD not in str(resp.headers)


@pytest.mark.F10
def test_resolve_bind_host_without_tls_is_loopback_regardless_of_host() -> None:
    cfg = RuntimeConfig(host="0.0.0.0")
    assert resolve_bind_host(cfg) == "127.0.0.1"


@pytest.mark.F10
def test_resolve_bind_host_with_both_tls_files_uses_configured_host() -> None:
    cfg = RuntimeConfig(host="0.0.0.0", tls_cert="certs/cert.pem", tls_key="certs/key.pem")
    assert resolve_bind_host(cfg) == "0.0.0.0"


@pytest.mark.F10
def test_resolve_bind_host_with_only_one_tls_file_is_still_loopback() -> None:
    cfg = RuntimeConfig(host="0.0.0.0", tls_cert="certs/cert.pem")
    assert resolve_bind_host(cfg) == "127.0.0.1"


@pytest.mark.F10
def test_video_feed_streams_mjpeg_of_the_newest_frame(
    tmp_path: Path, experiment: ExperimentDefinition
) -> None:
    app, _router, _loop, frame_store = _make_app(tmp_path, experiment)
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    frame_store.put(Frame(frame_id=1, t=0.0, image=image))

    resp = app.test_client().get("/video_feed", headers={"Authorization": AUTH_HEADER})
    assert resp.status_code == 200
    assert resp.mimetype == "multipart/x-mixed-replace"
    try:
        chunk = next(resp.response)
        assert chunk.startswith(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
        assert chunk.endswith(b"\r\n")
    finally:
        resp.response.close()


# ---------------------------------------------------------------------------
# F12 -- StatusResponse, run control, experiment/log views, alert history
# ---------------------------------------------------------------------------


@pytest.mark.F12
def test_status_matches_contract_and_reflects_idle_engine(
    tmp_path: Path, experiment: ExperimentDefinition
) -> None:
    app, router, _loop, _store = _make_app(tmp_path, experiment)
    resp = app.test_client().get("/api/status", headers={"Authorization": AUTH_HEADER})
    assert resp.status_code == 200
    status = StatusResponse.model_validate(resp.get_json())

    assert status.run_state == "idle"
    assert status.feed_ok is True
    assert status.fps == pytest.approx(12.5)
    assert status.expected_step_id == "s1"
    assert status.next_step_id == "s2"
    assert status.next_step_say == "Do step two"
    assert [s.step_id for s in status.steps] == ["s1", "s2", "s3", "s4"]
    assert status.recent_alerts == []
    assert status.generated_at == datetime(2026, 1, 1, tzinfo=UTC)


@pytest.mark.F12
def test_run_start_then_double_start_is_a_conflict_not_a_500(
    tmp_path: Path, experiment: ExperimentDefinition
) -> None:
    app, router, _loop, _store = _make_app(tmp_path, experiment)
    client = app.test_client()

    resp = client.post("/api/run/start", headers={"Authorization": AUTH_HEADER})
    assert resp.status_code == 200
    body = RunControlResponse.model_validate(resp.get_json())
    assert body.ok is True
    assert body.run_state == "running"
    assert router.run_state == "running"

    resp2 = client.post("/api/run/start", headers={"Authorization": AUTH_HEADER})
    assert resp2.status_code == 409
    assert router.run_state == "running"  # unchanged by the rejected call


@pytest.mark.F12
def test_run_reset_finishes_and_arms_a_fresh_idle_run(
    tmp_path: Path, experiment: ExperimentDefinition
) -> None:
    app, router, _loop, _store = _make_app(tmp_path, experiment)
    client = app.test_client()

    client.post("/api/run/start", headers={"Authorization": AUTH_HEADER})
    started_run_id = router.run_id

    resp = client.post("/api/run/reset", headers={"Authorization": AUTH_HEADER})
    assert resp.status_code == 200
    body = RunControlResponse.model_validate(resp.get_json())
    assert body.run_state == "idle"
    assert body.run_id != started_run_id
    assert router.run_state == "idle"


@pytest.mark.F12
def test_get_experiment_is_a_read_only_view_matching_the_source(
    tmp_path: Path, experiment: ExperimentDefinition
) -> None:
    app, *_ = _make_app(tmp_path, experiment)
    resp = app.test_client().get("/api/experiment", headers={"Authorization": AUTH_HEADER})
    assert resp.status_code == 200
    assert ExperimentDefinition.model_validate(resp.get_json()) == experiment


@pytest.mark.F12
def test_get_log_ignores_an_incomplete_trailing_line(
    tmp_path: Path, experiment: ExperimentDefinition
) -> None:
    app, *_ = _make_app(tmp_path, experiment)
    log_path = tmp_path / "run-abc.jsonl"
    valid_line = (
        '{"seq": 0, "run_id": "run-abc", "t_video": 0.0, '
        '"event_type": "run_started", "detail": "Run started"}'
    )
    log_path.write_text(valid_line + "\n" + '{"seq": 1, "run_id": "run-a', encoding="utf-8")

    resp = app.test_client().get(
        "/api/log?run_id=run-abc", headers={"Authorization": AUTH_HEADER}
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["run_id"] == "run-abc"
    assert len(body["entries"]) == 1
    assert body["entries"][0]["event_type"] == "run_started"


@pytest.mark.F12
def test_get_log_defaults_to_the_current_run_id(
    tmp_path: Path, experiment: ExperimentDefinition
) -> None:
    app, router, _loop, _store = _make_app(tmp_path, experiment)
    client = app.test_client()
    client.post("/api/run/start", headers={"Authorization": AUTH_HEADER})

    resp = client.get("/api/log", headers={"Authorization": AUTH_HEADER})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["run_id"] == router.run_id
    assert len(body["entries"]) >= 1
    assert body["entries"][0]["event_type"] == "run_started"


@pytest.mark.F12
def test_get_log_for_unknown_run_id_is_404(
    tmp_path: Path, experiment: ExperimentDefinition
) -> None:
    app, *_ = _make_app(tmp_path, experiment)
    resp = app.test_client().get(
        "/api/log?run_id=does-not-exist", headers={"Authorization": AUTH_HEADER}
    )
    assert resp.status_code == 404


@pytest.mark.F12
def test_recent_alerts_caps_and_orders_newest_last() -> None:
    history = RecentAlerts(cap=3)
    for i in range(5):
        history(
            EngineEvent(
                t=float(i),
                kind="deviation_detected",
                step_id="s1",
                deviation_type="repeat",
                confidence_tag="confirmed",
            )
        )
    snapshot = history.snapshot()
    assert [e.t for e in snapshot] == [2.0, 3.0, 4.0]


@pytest.mark.F12
def test_recent_alerts_ignores_non_deviation_events() -> None:
    history = RecentAlerts(cap=5)
    history(
        EngineEvent(t=0.0, kind="step_confirmed", step_id="s1", confidence_tag="confirmed")
    )
    assert history.snapshot() == []


# ---------------------------------------------------------------------------
# Edge cases found on review: path traversal via ?run_id, a degenerate frame
# crashing /video_feed, and a silently-broken index.html placeholder.
# ---------------------------------------------------------------------------


@pytest.mark.F12
@pytest.mark.parametrize(
    "malicious_run_id",
    [
        "../secret",
        "../../etc/passwd",
        "..%2f..%2fsecret",
        "a/b",
        "a\\b",
        "a b",  # rejected too -- no legitimately generated run_id has a space
        "",
    ],
)
def test_get_log_rejects_a_run_id_that_could_escape_log_dir(
    tmp_path: Path, experiment: ExperimentDefinition, malicious_run_id: str
) -> None:
    # A file a real run would never produce, placed just outside log_dir,
    # so a successful traversal would be observable as a 200 with content.
    outside = tmp_path.parent / "secret.jsonl"
    outside.write_text(
        '{"seq": 0, "run_id": "secret", "t_video": 0.0, '
        '"event_type": "run_started", "detail": "should never be readable"}\n',
        encoding="utf-8",
    )
    try:
        app, *_ = _make_app(tmp_path, experiment)
        resp = app.test_client().get(
            "/api/log",
            query_string={"run_id": malicious_run_id},
            headers={"Authorization": AUTH_HEADER},
        )
        # Either rejected outright (400) or treated as an ordinary, absent
        # run_id (404) -- never a 200 that leaked content from outside log_dir.
        assert resp.status_code in (400, 404)
        if resp.status_code == 400:
            assert resp.get_json()["error"]
    finally:
        outside.unlink(missing_ok=True)


@pytest.mark.F12
def test_get_log_accepts_every_run_id_format_this_codebase_actually_generates(
    tmp_path: Path, experiment: ExperimentDefinition
) -> None:
    for run_id in ["live-20260928T193538Z-7ff758", "train-01", "test-27", "concurrent-run-0"]:
        (tmp_path / f"{run_id}.jsonl").write_text(
            f'{{"seq": 0, "run_id": "{run_id}", "t_video": 0.0, '
            f'"event_type": "run_started", "detail": "Run started"}}\n',
            encoding="utf-8",
        )
    app, *_ = _make_app(tmp_path, experiment)
    client = app.test_client()
    for run_id in ["live-20260928T193538Z-7ff758", "train-01", "test-27", "concurrent-run-0"]:
        resp = client.get(f"/api/log?run_id={run_id}", headers={"Authorization": AUTH_HEADER})
        assert resp.status_code == 200, run_id
        assert resp.get_json()["run_id"] == run_id


@pytest.mark.F10
def test_jpeg_cache_encodes_and_caches_a_normal_frame() -> None:
    cache = _JpegCache(jpeg_quality=80)
    frame = Frame(frame_id=1, t=0.0, image=np.zeros((4, 4, 3), dtype=np.uint8))
    jpg1 = cache.encode(frame)
    assert jpg1 is not None
    assert jpg1.startswith(b"\xff\xd8")  # JPEG magic bytes
    jpg2 = cache.encode(frame)  # same frame_id -> served from cache
    assert jpg2 is jpg1


@pytest.mark.F10
def test_jpeg_cache_returns_none_instead_of_raising_for_a_degenerate_frame() -> None:
    # cv2.imencode raises cv2.error on a zero-height image (confirmed
    # empirically) -- contracts.Frame does not itself reject H=0, so
    # /video_feed must survive being handed one rather than crashing every
    # connected client's stream.
    cache = _JpegCache(jpeg_quality=80)
    degenerate = Frame(frame_id=1, t=0.0, image=np.zeros((0, 4, 3), dtype=np.uint8))
    assert cache.encode(degenerate) is None


@pytest.mark.F10
def test_video_feed_skips_a_degenerate_frame_and_recovers(
    tmp_path: Path, experiment: ExperimentDefinition
) -> None:
    # The store only ever holds one frame (latest-frame-wins), and the test
    # client's client.get() itself blocks internally until the generator
    # yields at least once -- so the good frame must already be racing in
    # from another thread *before* client.get() is called, not after (by
    # then it would be too late: client.get() would never have returned to
    # let this test start the injector in the first place).
    app, _router, _loop, frame_store = _make_app(tmp_path, experiment)
    frame_store.put(Frame(frame_id=1, t=0.0, image=np.zeros((0, 4, 3), dtype=np.uint8)))

    def _inject_good_frame_soon() -> None:
        time.sleep(0.05)
        frame_store.put(Frame(frame_id=2, t=0.1, image=np.zeros((4, 4, 3), dtype=np.uint8)))

    injector = threading.Thread(target=_inject_good_frame_soon, daemon=True)
    injector.start()
    try:
        # If the degenerate frame crashed the generator instead of being
        # skipped, this call would raise cv2.error (confirmed empirically)
        # instead of eventually returning once the injector's good frame
        # lands.
        resp = app.test_client().get("/video_feed", headers={"Authorization": AUTH_HEADER})
        assert resp.status_code == 200
        chunk = next(resp.response)
        assert chunk.startswith(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
        resp.response.close()
    finally:
        injector.join(timeout=2.0)


@pytest.mark.F12
def test_render_index_is_self_contained_html() -> None:
    html = _render_index()
    assert "/*__STYLE_CSS__*/" not in html
    assert "/*__APP_JS__*/" not in html
    assert "<style>" in html
    assert "<script>" in html
