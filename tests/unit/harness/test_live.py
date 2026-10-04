"""harness/live.py -- self-check items, argument handling, pacing, looping, metrics and the
graceful stop of the live wiring. Fakes and temporary files only; no model, camera or audio."""

from __future__ import annotations

import hashlib
import json
import socket
import threading
import time
from pathlib import Path

import pytest

from contracts import ExperimentDefinition, PerceptionConfig, RuntimeConfig
from harness import live
from harness.fakes import FakeFrameSource, FakePerception, make_blank_frame
from outputs.tts import FakeSpeaker, null_engine_factory

FIXTURE = Path(__file__).resolve().parents[3] / "fixtures" / "experiment_4step.json"


# --- weights ----------------------------------------------------------------------------------


def _manifest(tmp_path: Path, detector_bytes: bytes = b"det", hand_bytes: bytes = b"hand") -> Path:
    (tmp_path / "det.pt").write_bytes(detector_bytes)
    (tmp_path / "hand.task").write_bytes(hand_bytes)
    manifest = {
        "active_detector": "yolo11n",
        "detectors": [
            {"name": "yolo11n", "file": "det.pt", "sha256": hashlib.sha256(b"det").hexdigest()},
            {"name": "rfdetr_nano", "file": "other.pth", "sha256": "0" * 64},
        ],
        "hand": {"file": "hand.task", "sha256": hashlib.sha256(b"hand").hexdigest()},
    }
    path = tmp_path / "MANIFEST.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_weights_ok_when_hashes_match(tmp_path: Path) -> None:
    items = live.check_weights(_manifest(tmp_path), environ={})
    assert [i.ok for i in items] == [True, True]
    assert "yolo11n" in items[0].name


def test_weights_fail_on_a_changed_file(tmp_path: Path) -> None:
    items = live.check_weights(_manifest(tmp_path, detector_bytes=b"tampered"), environ={})
    assert not items[0].ok and "differs from the manifest" in items[0].detail
    assert items[1].ok


def test_weights_fail_on_a_missing_hand_model(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    (tmp_path / "hand.task").unlink()
    items = live.check_weights(manifest, environ={})
    assert items[0].ok and not items[1].ok and "not found" in items[1].detail


def test_weights_follow_the_detector_environment_variable(tmp_path: Path) -> None:
    items = live.check_weights(_manifest(tmp_path), environ={"SIH_DETECTOR": "rfdetr-nano"})
    assert "rfdetr_nano" in items[0].name and not items[0].ok  # its file is absent


def test_weights_fail_on_an_unreadable_manifest(tmp_path: Path) -> None:
    items = live.check_weights(tmp_path / "nope.json")
    assert len(items) == 1 and not items[0].ok


# --- config files -----------------------------------------------------------------------------


def test_configs_load_and_a_bad_key_fails_loudly(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime.yaml"
    perception = tmp_path / "perception.yaml"
    runtime.write_text("target_fps: 10\n", encoding="utf-8")
    perception.write_text("hysteresis_frames: 5\n", encoding="utf-8")
    item, settings = live.check_configs(runtime, perception, FIXTURE)
    assert item.ok and settings is not None and settings.runtime.target_fps == 10
    runtime.write_text("target_fsp: 10\n", encoding="utf-8")
    item, settings = live.check_configs(runtime, perception, FIXTURE)
    assert not item.ok and settings is None
    item, _ = live.check_configs(tmp_path / "missing.yaml", perception, FIXTURE)
    assert not item.ok


# --- .env -------------------------------------------------------------------------------------


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    monkeypatch.delenv("STREAM_USER", raising=False)
    monkeypatch.delenv("STREAM_PASSWORD", raising=False)
    return monkeypatch


def _env(tmp_path: Path, text: str) -> Path:
    path = tmp_path / ".env"
    path.write_text(text, encoding="utf-8")
    return path


def test_env_ok_never_shows_a_value(tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    item, creds = live.check_env(_env(tmp_path, "STREAM_USER=alice\nSTREAM_PASSWORD=s3cretvalue\n"))
    assert item.ok and creds == ("alice", "s3cretvalue")
    assert "s3cretvalue" not in item.line() and "alice" not in item.line()
    assert ".env file" in item.detail


@pytest.mark.parametrize(
    "text, fragment",
    [
        ("STREAM_USER=changeme\nSTREAM_PASSWORD=real-pass\n", "changeme"),
        ("STREAM_USER=alice\nSTREAM_PASSWORD=changeme\n", "changeme"),
        ("STREAM_USER=alice\nSTREAM_PASSWORD=\n", "non-empty"),
        ("STREAM_USER=\nSTREAM_PASSWORD=x\n", "non-empty"),
        ("STREAM_USER=alice\n", "non-empty"),
        ("", "non-empty"),
    ],
)
def test_env_fails(tmp_path: Path, clean_env: pytest.MonkeyPatch, text: str, fragment: str) -> None:
    item, creds = live.check_env(_env(tmp_path, text))
    assert not item.ok and creds is None and fragment in item.detail
    assert "real-pass" not in item.line()


def test_env_missing_file_fails_and_names_it(tmp_path: Path, clean_env: pytest.MonkeyPatch) -> None:
    item, creds = live.check_env(tmp_path / "absent.env")
    assert not item.ok and creds is None and "not found" in item.detail


def test_env_variables_alone_are_accepted_and_say_so(
    tmp_path: Path, clean_env: pytest.MonkeyPatch
) -> None:
    clean_env.setenv("STREAM_USER", "bob")
    clean_env.setenv("STREAM_PASSWORD", "pw-from-env")
    item, creds = live.check_env(tmp_path / "absent.env")
    assert item.ok and creds == ("bob", "pw-from-env")
    assert "process environment" in live.credentials_origin(item)


# --- TLS and port -----------------------------------------------------------------------------


def test_tls_items(tmp_path: Path) -> None:
    assert live.check_tls(RuntimeConfig()).ok
    assert not live.check_tls(RuntimeConfig(tls_cert="c.pem")).ok  # only one of the two
    assert not live.check_tls(RuntimeConfig(tls_cert="a.pem", tls_key="b.pem")).ok
    cert, key = tmp_path / "c.pem", tmp_path / "k.pem"
    cert.write_text("x")
    key.write_text("x")
    assert live.check_tls(RuntimeConfig(tls_cert=str(cert), tls_key=str(key))).ok


def test_tls_auto_uses_generated_certs_only_when_both_exist(tmp_path: Path) -> None:
    cfg = RuntimeConfig()
    assert live.tls_auto(cfg, tmp_path) is cfg
    (tmp_path / "cert.pem").write_text("x")
    assert not live.tls_is_on(live.tls_auto(cfg, tmp_path))
    (tmp_path / "key.pem").write_text("x")
    assert live.tls_is_on(live.tls_auto(cfg, tmp_path))
    explicit = RuntimeConfig(tls_cert="a", tls_key="b")
    assert live.tls_auto(explicit, tmp_path) is explicit


def test_effective_runtime_applies_port_tls_and_max_speed(tmp_path: Path) -> None:
    base = RuntimeConfig(target_fps=10, tls_cert="a", tls_key="b")
    opts = live.LiveOptions(port=9999, tls="off", max_speed=True)
    cfg = live.effective_runtime(opts, base)
    assert cfg.port == 9999 and not live.tls_is_on(cfg) and cfg.target_fps == live.UNTHROTTLED_FPS
    assert live.effective_runtime(live.LiveOptions(tls="off"), base).target_fps == 10


def test_public_url_follows_tls() -> None:
    assert live.public_url(RuntimeConfig(port=8443)) == "http://127.0.0.1:8443/"
    assert live.public_url(RuntimeConfig(port=8443, tls_cert="a", tls_key="b")).startswith(
        "https://"
    )


def test_port_free_and_in_use() -> None:
    with socket.socket() as holder:
        holder.bind(("127.0.0.1", 0))
        holder.listen()
        port = holder.getsockname()[1]
        busy = live.check_port("127.0.0.1", port)
        assert not busy.ok and "already in use" in busy.detail
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        free_port = probe.getsockname()[1]
    assert live.check_port("127.0.0.1", free_port).ok


# --- TTS --------------------------------------------------------------------------------------


@pytest.mark.slow
def test_tts_check_and_test_with_a_stub_engine() -> None:
    assert live.check_tts(null_engine_factory, timeout_s=30).ok
    item = live.tts_test("Audio check", engine_factory=null_engine_factory, timeout_s=30)
    assert item.ok and "Audio check" in item.detail


def _boom() -> object:
    raise RuntimeError("no audio device")


@pytest.mark.slow
def test_tts_check_reports_an_engine_that_cannot_start() -> None:
    item = live.check_tts(_boom, timeout_s=30)
    assert not item.ok and "no audio device" in item.detail
    item = live.tts_test(engine_factory=_boom, timeout_s=30)
    assert not item.ok and "no audio device" in item.detail


# --- source check and arguments ---------------------------------------------------------------


def test_source_check_reports_the_granted_mode_and_keeps_the_first_frame() -> None:
    frames = [make_blank_frame(i, i / 30, height=480, width=848) for i in range(3)]
    fake = FakeFrameSource([None, *frames], fps=30.0)
    item, src = live.check_source("0", opener=lambda s: fake)
    assert item.ok and "848x480" in item.detail and "requested 1280x720" in item.detail
    assert "different mode" in item.detail
    assert [src.read().frame_id for _ in range(3)] == [0, 1, 2]  # nothing was lost


def test_source_check_failures() -> None:
    def refuse(_: str) -> object:
        raise OSError("busy")

    item, src = live.check_source("0", opener=refuse)
    assert not item.ok and src is None and "busy" in item.detail
    item, src = live.check_source("clip.mp4", opener=lambda s: FakeFrameSource([]), tries=3)
    assert not item.ok and src is None and "no frame" in item.detail


def _runs(tmp_path: Path) -> Path:
    runs = tmp_path / "runs"
    for run_id, split in (("v1", "val"), ("t1", "test")):
        (runs / run_id).mkdir(parents=True)
        (runs / run_id / "video.mp4").write_bytes(b"x")
        (runs / run_id / "script.json").write_text(json.dumps({"split": split}), encoding="utf-8")
    (runs / "m1").mkdir()
    (runs / "m1" / "video.mp4").write_bytes(b"x")  # only in the manifest
    (runs / "manifest.csv").write_text("run_id,split\nm1,test\n", encoding="utf-8")
    return runs


def test_replay_run_resolves_a_val_run(tmp_path: Path) -> None:
    runs = _runs(tmp_path)
    assert live.resolve_source(None, "v1", runs_dir=runs) == str(runs / "v1" / "video.mp4")


def test_replay_run_refuses_the_test_split_unless_allowed(tmp_path: Path) -> None:
    runs = _runs(tmp_path)
    with pytest.raises(live.HeldoutRefused, match="--allow-heldout"):
        live.resolve_source(None, "t1", runs_dir=runs)
    with pytest.raises(live.HeldoutRefused):  # the manifest alone also says test
        live.resolve_source(None, "m1", runs_dir=runs)
    assert live.resolve_source(None, "t1", runs_dir=runs, allow_heldout=True).endswith("video.mp4")


def test_a_source_path_inside_a_test_run_folder_is_held_to_the_same_rule(tmp_path: Path) -> None:
    runs = _runs(tmp_path)
    with pytest.raises(live.HeldoutRefused):
        live.resolve_source(str(runs / "t1" / "video.mp4"), None, runs_dir=runs)
    assert live.resolve_source(str(runs / "v1" / "video.mp4"), None, runs_dir=runs)


def test_source_arguments(tmp_path: Path) -> None:
    runs = _runs(tmp_path)
    assert live.resolve_source(None, None, runs_dir=runs) == "0"  # the demo default
    assert live.resolve_source("2", None, runs_dir=runs) == "2"
    assert live.resolve_source("rtsp://host/stream", None, runs_dir=runs) == "rtsp://host/stream"
    with pytest.raises(live.StartupRefused, match="no recording"):
        live.resolve_source(None, "nope", runs_dir=runs)
    with pytest.raises(live.StartupRefused, match="alternatives"):
        live.resolve_source("0", "v1", runs_dir=runs)


def test_run_live_refuses_a_test_run_with_exit_code_2(tmp_path: Path) -> None:
    lines: list[str] = []
    opts = live.LiveOptions(replay_run="t1", runs_dir=_runs(tmp_path))
    assert live.run_live(opts, lines.append) == 2
    assert "test split" in lines[0]


def test_demo_cli_parses_the_documented_options() -> None:
    from scripts.demo import build_parser

    args = build_parser().parse_args(
        ["--replay-run", "x016", "--allow-heldout", "--check", "--no-auto-start", "--tls", "off"]
    )
    assert args.replay_run == "x016" and args.allow_heldout and args.check
    assert args.auto_start is False and args.tls == "off"
    assert build_parser().parse_args([]).auto_start is None
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--source", "0", "--replay-run", "x016"])
    assert "honest and allowed" in build_parser().format_help()


# --- pacing, looping ---------------------------------------------------------------------------


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def test_paced_source_delivers_frames_at_their_own_times() -> None:
    clock = FakeClock()
    frames = [make_blank_frame(i, i * 0.1) for i in range(5)]
    src = live.PacedSource(FakeFrameSource(frames), clock=clock, sleep=clock.sleep)
    delivered = []
    for _ in range(5):
        frame = src.read()
        delivered.append((frame.frame_id, round(clock.now - 100.0, 6)))
    assert delivered == [(i, round(i * 0.1, 6)) for i in range(5)]
    assert src.read() is None and src.exhausted


def test_paced_source_does_not_wait_for_a_late_reader_and_resyncs_after_a_stall() -> None:
    clock = FakeClock()
    frames = [make_blank_frame(0, 0.0), make_blank_frame(1, 0.1), make_blank_frame(2, 5.0)]
    src = live.PacedSource(
        FakeFrameSource(frames), clock=clock, sleep=clock.sleep, resync_after_s=1
    )
    src.read()
    clock.now += 0.5  # the consumer was slow: frame 1 is already due, no sleeping
    before = len(clock.slept)
    src.read()
    assert len(clock.slept) == before
    clock.now += 10.0  # a long stall: re-anchor, no catch-up burst afterwards
    src.read()
    assert len(clock.slept) == before


def test_looping_source_continues_ids_and_time_and_calls_back() -> None:
    made: list[FakeFrameSource] = []

    def factory() -> FakeFrameSource:
        src = FakeFrameSource([make_blank_frame(i, i / 10) for i in range(3)], fps=10.0)
        made.append(src)
        return src

    loops: list[int] = []
    src = live.LoopingSource(factory, on_loop=loops.append)
    got = [src.read() for _ in range(7)]
    assert [f.frame_id for f in got] == list(range(7))
    assert [round(f.t, 3) for f in got] == [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert loops == [1, 2] and src.loops == 2 and not src.exhausted
    assert made[0].closed and made[1].closed and not made[2].closed


def test_looping_source_refuses_an_empty_file() -> None:
    src = live.LoopingSource(lambda: FakeFrameSource([], fps=10.0))
    with pytest.raises(live.StartupRefused):
        src.read()


# --- metrics ----------------------------------------------------------------------------------


def test_metrics_report_throughput_drops_latency_and_samples() -> None:
    clock = FakeClock()
    metrics = live.LiveMetrics(
        clock=clock, cpu_clock=lambda: clock.now - 100.0, memory=lambda: 123.0
    )
    for frame_id in range(10):
        metrics.frame_arrived(frame_id)
        clock.now += 0.05
        if frame_id % 2 == 0:  # every other frame is dropped
            metrics.status_updated(frame_id)
        clock.now += 0.05
    metrics.sample_resources()
    report = metrics.report()
    assert report["frames_captured"] == 10 and report["frames_processed"] == 5
    assert report["frames_dropped"] == 5
    assert report["latency_ms_p95"] == pytest.approx(50.0, abs=1)
    assert report["achieved_fps"] == pytest.approx(5.0, rel=0.05)
    assert report["cpu_percent_of_one_core"] == pytest.approx(100.0)
    assert report["resources"][0]["rss_mb"] == 123.0


def test_metrics_pending_table_is_bounded() -> None:
    metrics = live.LiveMetrics()
    for frame_id in range(live.LiveMetrics._MAX_PENDING + 100):
        metrics.frame_arrived(frame_id)
    assert len(metrics._arrivals) == live.LiveMetrics._MAX_PENDING


# --- the wiring: a graceful stop with fakes ----------------------------------------------------


class _FakeServer:
    def __init__(self) -> None:
        self.stop = threading.Event()
        self.shut = False
        self.closed = False

    def serve_forever(self) -> None:
        self.stop.wait(5)

    def shutdown(self) -> None:
        self.shut = True
        self.stop.set()

    def server_close(self) -> None:
        self.closed = True


def _settings(tmp_path: Path) -> live.Settings:
    runtime = RuntimeConfig(
        target_fps=200.0, log_dir=str(tmp_path / "logs"), video_dir=str(tmp_path / "video")
    )
    return live.Settings(ExperimentDefinition.from_json(FIXTURE), runtime, PerceptionConfig())


def _system(tmp_path: Path, frames: int, metrics: live.LiveMetrics | None = None) -> tuple:
    source = FakeFrameSource(
        [make_blank_frame(i, i / 20, height=8, width=8) for i in range(frames)],
        fps=20.0,
        read_delay=0.01,
    )
    speaker = FakeSpeaker()
    server = _FakeServer()
    system = live.LiveSystem(
        settings=_settings(tmp_path),
        perception=FakePerception([]),
        source=source,
        speaker=speaker,
        username="u",
        password="p",
        metrics=metrics,
        server_factory=lambda app, cfg: server,
    )
    return system, source, speaker, server


@pytest.mark.F13
def test_graceful_stop_closes_everything_and_logs_the_aborted_run(tmp_path: Path) -> None:
    system, source, speaker, server = _system(tmp_path, frames=10_000)
    system.start(auto_start=True)
    time.sleep(0.4)
    stop = threading.Event()
    stop.set()
    assert system.wait(stop_event=stop) == "stopped"
    system.shutdown()
    system.shutdown()  # idempotent
    assert source.closed and speaker.closed and server.shut and server.closed
    log = next((tmp_path / "logs").glob("*.jsonl"))
    lines = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["event_type"] == "run_started" and lines[0]["t_video"] == 0.0
    assert lines[-1]["event_type"] == "run_completed"  # the aborted run, flushed to disk
    video = next((tmp_path / "video").glob("*.avi"))
    assert video.stat().st_size > 0


def test_wait_reasons(tmp_path: Path) -> None:
    system, *_ = _system(tmp_path, frames=3)
    system.start(auto_start=False)
    try:
        assert system.wait(exit_when_done=True, poll_s=0.02) == "source finished"
        assert system.wait(duration_s=0.05, poll_s=0.01) == "duration"
    finally:
        system.shutdown()


def test_run_start_uses_video_time_not_process_uptime(tmp_path: Path) -> None:
    system, *_ = _system(tmp_path, frames=10_000)
    system.start(auto_start=False)
    try:
        time.sleep(0.3)
        assert 0.0 <= system.run_clock() < 600.0  # frame t, never a monotonic uptime
        run_id = system.loop.start_run(system.run_clock())
        assert run_id
    finally:
        system.shutdown()


def test_metrics_are_fed_by_the_wiring(tmp_path: Path) -> None:
    metrics = live.LiveMetrics()
    system, *_ = _system(tmp_path, frames=60, metrics=metrics)
    wrapped = live.MeteredSource(system.source, metrics)
    system.loop._source = wrapped  # the loop was built on the bare source in this fixture
    system.start(auto_start=True)
    try:
        system.wait(exit_when_done=True, poll_s=0.02)
    finally:
        system.shutdown()
    report = metrics.report()
    assert report["frames_captured"] > 0 and report["frames_processed"] > 0
    assert report["latency_ms_p95"] is not None


def test_unthrottled_lifts_only_the_throttle() -> None:
    cfg = live.effective_runtime(live.LiveOptions(unthrottled=True), RuntimeConfig(target_fps=10))
    assert cfg.target_fps == live.UNTHROTTLED_FPS
