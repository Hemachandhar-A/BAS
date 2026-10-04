"""The demo flow: the start gate, the speech delay, the playlist (parser, order, clip boundary,
Enter to advance, refusals) and the dashboard's autostart. Fakes only; real threads with short
waits where a thread is the thing under test."""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from contracts import (
    Detection,
    ExperimentDefinition,
    Frame,
    PerceptionConfig,
    PerceptionFrame,
    RuntimeConfig,
)
from harness import live
from harness.fakes import FakeFrameSource, make_blank_frame
from outputs.tts import FakeSpeaker

FIXTURE = Path(__file__).resolve().parents[3] / "fixtures" / "experiment_4step.json"
AUTH = {"Authorization": "Basic " + base64.b64encode(b"u:p").decode()}


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


class _NoServer:
    def serve_forever(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def server_close(self) -> None:
        pass


def _until(predicate, timeout: float = 5.0) -> bool:  # noqa: ANN001
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


# --- the start gate --------------------------------------------------------------------------


def test_gate_holds_on_the_first_frame_without_advancing_the_clock_or_the_clip(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = FakeClock()
    inner = FakeFrameSource([make_blank_frame(i, i * 0.1) for i in range(5)], fps=10.0)
    gate = live.StartGate()
    src = live.PacedSource(inner, clock=clock, sleep=clock.sleep, gate=gate)
    with caplog.at_level(logging.INFO, logger="harness.live"):
        held = [src.read() for _ in range(6)]
    assert {(f.frame_id, f.t) for f in held} == {(0, 0.0)}  # the same frame every time
    assert all(f is held[0] for f in held)  # the very same object: nothing re-decoded
    assert inner._index == 1  # the clip did not advance
    assert not inner.exhausted


def test_gate_logs_waiting_once_per_ten_seconds(caplog: pytest.LogCaptureFixture) -> None:
    clock = FakeClock()
    inner = FakeFrameSource([make_blank_frame(i, i * 0.1) for i in range(5)])
    src = live.PacedSource(
        inner, clock=clock, sleep=clock.sleep, gate=live.StartGate(), hold_interval_s=1.0
    )
    with caplog.at_level(logging.INFO, logger="harness.live"):
        for _ in range(26):  # 25 s of fake time
            src.read()
    waits = [r for r in caplog.records if "waiting for dashboard" in r.getMessage()]
    assert len(waits) == 3  # at 0 s, 10 s and 20 s


def test_release_plays_the_clip_from_the_held_frame_without_a_burst() -> None:
    clock = FakeClock()
    frames = [make_blank_frame(i, i * 0.1) for i in range(5)]
    gate = live.StartGate()
    src = live.PacedSource(FakeFrameSource(frames), clock=clock, sleep=clock.sleep, gate=gate)
    for _ in range(40):  # held for a long time
        src.read()
    clock.slept.clear()
    gate.release()
    released_at = clock.now
    ids = []
    for _ in range(4):
        frame = src.read()
        ids.append((frame.frame_id, round(clock.now - released_at, 6)))
    assert ids == [(1, 0.1), (2, 0.2), (3, 0.3), (4, 0.4)]  # at the clip's own speed from release


def test_a_released_gate_costs_nothing() -> None:
    gate = live.StartGate()
    gate.release()
    clock = FakeClock()
    src = live.PacedSource(
        FakeFrameSource([make_blank_frame(i, i * 0.1) for i in range(3)]),
        clock=clock,
        sleep=clock.sleep,
        gate=gate,
    )
    assert [src.read().frame_id for _ in range(3)] == [0, 1, 2]


def _gated_system(tmp_path: Path, frames: int = 400):
    runtime = RuntimeConfig(
        target_fps=100.0, log_dir=str(tmp_path / "logs"), video_dir=str(tmp_path / "video")
    )
    settings = live.Settings(ExperimentDefinition.from_json(FIXTURE), runtime, PerceptionConfig())
    gate = live.StartGate()
    inner = FakeFrameSource(
        [make_blank_frame(i, i / 50, height=16, width=16) for i in range(frames)], fps=50.0
    )
    speaker = FakeSpeaker()
    system = live.LiveSystem(
        settings=settings,
        perception=_ClipPerception(),
        source=live.PacedSource(inner, gate=gate),
        speaker=speaker,
        username="u",
        password="p",
        server_factory=lambda app, cfg: _NoServer(),
    )
    return system, gate, speaker


def test_nothing_is_spoken_or_logged_before_the_run_starts_then_both_begin(tmp_path: Path) -> None:
    system, gate, speaker = _gated_system(tmp_path)
    system.start(auto_start=False)
    try:
        assert _until(lambda: system.loop.frame_store.get() is not None)
        time.sleep(0.5)  # the preview is on screen and the inference thread has seen it
        still = system.loop.frame_store.get()
        assert still.frame_id == 0 and still.t == 0.0  # the clip has not moved
        assert speaker.calls == []  # no step phrase
        assert system.router.run_state == "idle"
        assert not list((tmp_path / "logs").glob("*.jsonl"))  # no run_started, no step event
        assert system.recent_alerts.snapshot() == []

        run_id = system.loop.start_run(system.run_clock())
        assert speaker.calls == [("Do step one", "info")]  # the first phrase, at the start
        gate.release()
        assert _until(lambda: system.loop.frame_store.get().frame_id > 3)  # the frames advance
        assert system.loop.frame_store.get().t > 0.0
        log = tmp_path / "logs" / f"{run_id}.jsonl"
        first = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
        assert first["event_type"] == "run_started" and first["t_video"] == 0.0
    finally:
        system.shutdown()


def test_the_dashboard_start_releases_the_gate(tmp_path: Path) -> None:
    """POST /api/run/start (with credentials) flips run_state to running; the controller polls the
    status store and releases the gate. Without credentials nothing happens."""
    system, gate, speaker = _gated_system(tmp_path)
    system.start(auto_start=False)
    stop = threading.Event()
    watcher = threading.Thread(
        target=live.release_gate_when_running,
        args=(system.router, lambda: gate, stop),
        daemon=True,
    )
    watcher.start()
    try:
        assert _until(lambda: system.loop.frame_store.get() is not None)
        client = system.app.test_client()
        assert client.post("/api/run/start").status_code == 401
        time.sleep(0.2)
        assert not gate.released and system.router.run_state == "idle"
        assert client.post("/api/run/start", headers=AUTH).status_code == 200
        assert _until(lambda: gate.released)
        assert _until(lambda: system.loop.frame_store.get().frame_id > 3)
        assert speaker.calls[0] == ("Do step one", "info")
    finally:
        stop.set()
        watcher.join(timeout=2)
        system.shutdown()


def test_the_watcher_gives_up_when_stopped() -> None:
    class Idle:
        run_state = "idle"

    stop = threading.Event()
    stop.set()
    assert not live.release_gate_when_running(Idle(), live.StartGate, stop)


# --- the app: page, script, auth -------------------------------------------------------------


def test_the_autostart_page_is_served_behind_auth_and_the_script_guards_on_idle(
    tmp_path: Path,
) -> None:
    system, _gate, _speaker = _gated_system(tmp_path)
    client = system.app.test_client()
    assert client.get("/?autostart=1").status_code == 401
    page = client.get("/?autostart=1", headers=AUTH)
    assert page.status_code == 200
    html = page.get_data(as_text=True)
    assert "autostart=1" in html  # the script reads it from the page's own query string
    assert 'status.run_state === "idle"' in html  # the idle check, on a fresh status read
    assert "/api/run/start" in html and "AUTOSTART_DELAY_MS = 500" in html
    assert 'addEventListener("load", onVideoLoaded)' in html  # the first video frame
    assert "autostartFired" in html  # once per page load
    assert 'id="start-banner"' in html and "waiting to start" in html
    # the poll interval is the contract's 500 ms (the limit asked for)
    from contracts import DASHBOARD_POLL_MS

    assert DASHBOARD_POLL_MS <= 500 and f"POLL_MS = {DASHBOARD_POLL_MS}" in html
    routes = {r.rule for r in system.app.url_map.iter_rules()}
    from contracts import API_ROUTES

    assert len(routes) == len(API_ROUTES)  # no new route


# --- DelayedSpeaker --------------------------------------------------------------------------


def _delayed(delay: float = 0.2):
    clock = FakeClock()
    inner = FakeSpeaker()
    return live.DelayedSpeaker(inner, delay, clock=clock, autostart=False), inner, clock


def test_delayed_speaker_releases_in_fifo_order_after_the_delay() -> None:
    speaker, inner, clock = _delayed()
    speaker.say("one", "info")
    clock.now += 0.1
    speaker.say("two", "info")
    assert speaker.pump() == 0 and inner.calls == []
    clock.now += 0.15  # one is due (0.25 s old), two is not (0.15 s old)
    assert speaker.pump() == 1 and inner.calls == [("one", "info")]
    clock.now += 0.1
    assert speaker.pump() == 1
    assert inner.calls == [("one", "info"), ("two", "info")]


def test_an_alert_clears_older_waiting_speech_and_goes_out_as_an_alert() -> None:
    speaker, inner, clock = _delayed()
    speaker.say("old info", "info")
    speaker.say("another", "info")
    speaker.say("skipped step", "alert")
    speaker.say("later info", "info")
    clock.now += 1.0
    speaker.pump()
    assert inner.calls == [("skipped step", "alert"), ("later info", "info")]


def test_an_utterance_already_released_is_not_taken_back_by_a_later_alert() -> None:
    speaker, inner, clock = _delayed()
    speaker.say("spoken", "info")
    clock.now += 0.3
    speaker.pump()
    speaker.say("alert", "alert")
    clock.now += 0.3
    speaker.pump()
    assert inner.calls == [("spoken", "info"), ("alert", "alert")]


def test_delayed_speaker_never_blocks_or_raises_and_zero_passes_straight_through() -> None:
    class Broken:
        def say(self, text: str, priority: str) -> None:
            raise RuntimeError("audio gone")

        def close(self) -> None:
            raise RuntimeError("audio gone")

    clock = FakeClock()
    speaker = live.DelayedSpeaker(Broken(), 0.2, clock=clock, autostart=False)
    speaker.say("x", "info")
    clock.now += 1
    assert speaker.pump() == 1  # the failure is swallowed
    speaker.close()  # no raise
    speaker.say("after close", "info")  # dropped, no raise
    inner = FakeSpeaker()
    direct = live.DelayedSpeaker(inner, 0.0, clock=clock)
    direct.say("now", "info")
    assert inner.calls == [("now", "info")]


def test_close_hands_the_waiting_speech_to_the_inner_speaker_in_order_then_closes_it() -> None:
    speaker, inner, _ = _delayed()
    speaker.say("a", "info")
    speaker.say("b", "info")
    speaker.close()
    assert inner.calls == [("a", "info"), ("b", "info")] and inner.closed
    speaker.close()  # idempotent


def test_the_timer_thread_releases_after_the_real_delay_and_stops_on_close() -> None:
    inner = FakeSpeaker()
    speaker = live.DelayedSpeaker(inner, 0.15)
    started = time.monotonic()
    speaker.say("hello", "info")
    assert inner.calls == []  # not before the delay
    assert _until(lambda: inner.calls == [("hello", "info")])
    assert time.monotonic() - started >= 0.14
    speaker.close()
    assert not speaker._thread.is_alive() and inner.closed


# --- playlist: parser and order ---------------------------------------------------------------


def _touch(folder: Path, *names: str) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in names:
        (folder / name).write_bytes(b"x")
        paths.append(folder / name)
    return paths


def test_a_folder_gives_its_videos_in_name_order(tmp_path: Path) -> None:
    _touch(tmp_path / "clips", "b.mp4", "A.avi", "c.mov", "d.mkv", "notes.txt", "e.MP4")
    got = live.parse_playlist(tmp_path / "clips", runs_dir=tmp_path / "runs")
    assert [p.name for p in got] == ["A.avi", "b.mp4", "c.mov", "d.mkv", "e.MP4"]


def test_a_text_file_resolves_relative_paths_against_its_own_folder(tmp_path: Path) -> None:
    _touch(tmp_path / "sub", "one.mp4", "two.mp4")
    other = _touch(tmp_path / "elsewhere", "three.mp4")[0]
    listing = tmp_path / "sub" / "list.txt"
    listing.write_text(f"# the demo\n\ntwo.mp4\none.mp4\n{other}\nmissing.mp4\n", encoding="utf-8")
    warnings: list[str] = []
    got = live.parse_playlist(listing, runs_dir=tmp_path / "runs", warn=warnings.append)
    assert got == [tmp_path / "sub" / "two.mp4", tmp_path / "sub" / "one.mp4", other]
    assert len(warnings) == 1 and "missing.mp4" in warnings[0]


def test_shuffle_is_seeded_and_reproducible() -> None:
    clips = [Path(f"c{i}.mp4") for i in range(8)]
    a = live.order_clips(clips, shuffle=True, seed=7)
    assert a == live.order_clips(clips, shuffle=True, seed=7)
    assert a != live.order_clips(clips, shuffle=True, seed=8)
    assert sorted(a) == clips and a != clips
    assert live.order_clips(clips, shuffle=False, seed=7) == clips


@pytest.mark.parametrize("kind", ["missing", "empty folder", "no videos", "only missing paths"])
def test_an_unusable_playlist_is_refused(tmp_path: Path, kind: str) -> None:
    target = tmp_path / "p"
    if kind == "empty folder":
        target.mkdir()
    elif kind == "no videos":
        _touch(target, "readme.txt")
    elif kind == "only missing paths":
        target.write_text("nope.mp4\n", encoding="utf-8")
    with pytest.raises(live.StartupRefused):
        live.parse_playlist(target, runs_dir=tmp_path / "runs")


def test_a_clip_of_a_test_run_needs_allow_heldout(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    (runs / "t1").mkdir(parents=True)
    (runs / "t1" / "video.mp4").write_bytes(b"x")
    (runs / "t1" / "script.json").write_text(json.dumps({"split": "test"}), encoding="utf-8")
    (runs / "v1").mkdir()
    (runs / "v1" / "video.mp4").write_bytes(b"x")
    (runs / "v1" / "script.json").write_text(json.dumps({"split": "val"}), encoding="utf-8")
    listing = tmp_path / "list.txt"
    listing.write_text(f"{runs / 'v1' / 'video.mp4'}\n{runs / 't1' / 'video.mp4'}\n")
    with pytest.raises(live.HeldoutRefused):
        live.parse_playlist(listing, runs_dir=runs)
    assert len(live.parse_playlist(listing, runs_dir=runs, allow_heldout=True)) == 2
    listing.write_text(f"{runs / 'v1' / 'video.mp4'}\n")
    assert len(live.parse_playlist(listing, runs_dir=runs)) == 1


def test_unreadable_clips_are_skipped_with_a_warning_unless_none_is_left() -> None:
    def opener(path: str):
        if path.endswith("bad.mp4"):
            raise OSError("corrupt")
        if path.endswith("empty.mp4"):
            return FakeFrameSource([])
        return FakeFrameSource([make_blank_frame(0, 0.0)])

    warnings: list[str] = []
    clips = [Path("a.mp4"), Path("bad.mp4"), Path("empty.mp4"), Path("b.mp4")]
    assert live.usable_clips(clips, opener, warnings.append) == [Path("a.mp4"), Path("b.mp4")]
    assert len(warnings) == 2
    with pytest.raises(live.StartupRefused):
        live.usable_clips([Path("bad.mp4")], opener)


def test_run_live_refuses_with_exit_code_2(tmp_path: Path) -> None:
    lines: list[str] = []
    empty = tmp_path / "empty"
    empty.mkdir()
    for opts in (
        live.LiveOptions(mode="demo", playlist=empty, runs_dir=tmp_path),
        live.LiveOptions(mode="demo", playlist=tmp_path / "absent", runs_dir=tmp_path),
        live.LiveOptions(mode="demo", playlist=empty, source="clip.mp4", runs_dir=tmp_path),
    ):
        assert live.run_live(opts, lines.append) == 2
    assert all(line.startswith("refused:") for line in lines)


def test_the_demo_cli_parses_the_flow_options_and_documents_not_stitching() -> None:
    from scripts.demo import build_parser

    args = build_parser().parse_args(
        [
            "--playlist", "demo_videos", "--pause-between", "3", "--advance", "enter",
            "--shuffle", "--seed", "4", "--once", "--no-record", "--speech-delay", "0",
            "--no-wait-for-dashboard",
        ]
    )  # fmt: skip
    assert args.playlist == Path("demo_videos") and args.pause_between == 3
    assert args.advance == "enter" and args.shuffle and args.seed == 4 and args.once
    assert args.record is False and args.speech_delay == 0 and args.wait_for_dashboard is False
    defaults = build_parser().parse_args([])
    assert defaults.pause_between == 5 and defaults.advance == "auto" and defaults.record is True
    assert defaults.speech_delay == 0.2 and defaults.wait_for_dashboard is True
    assert "STITCH" in build_parser().format_help()
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--playlist", "x", "--replay-run", "x016"])


# --- playlist: the clip boundary -------------------------------------------------------------

CONF = 0.9


class _ClipPerception:
    """Four items appear one after another at 0.40, 0.55, 0.70 and 0.85 s of clip time and stay
    (the tracker takes its baseline from the first 10 frames). Stateless apart from the reset
    counter, so any leak between clips would come from the tracker or the engine, which is what
    the boundary test checks."""

    model_stamp = "fake:00000000|hand:00000000|pose:00000000"

    def __init__(self) -> None:
        self.resets = 0

    def process(self, frame: Frame) -> PerceptionFrame:
        appears = (("a", 0.40), ("b", 0.55), ("c", 0.70), ("d", 0.85))
        labels = [f"item_{c}" for c, at in appears if frame.t >= at]
        return PerceptionFrame(
            frame_id=frame.frame_id,
            t=frame.t,
            detections=[Detection(label=x, conf=CONF, box=(0.1, 0.1, 0.3, 0.3)) for x in labels],
        )

    def reset(self) -> None:
        self.resets += 1


def _clip_frames(count: int = 45, fps: float = 40.0) -> list[Frame]:
    return [make_blank_frame(i, i / fps, height=16, width=16) for i in range(count)]


def _playlist_rig(tmp_path: Path, names: list[str], **controller_kwargs):
    runtime = RuntimeConfig(
        target_fps=200.0, log_dir=str(tmp_path / "logs"), video_dir=str(tmp_path / "video")
    )
    settings = live.Settings(ExperimentDefinition.from_json(FIXTURE), runtime, PerceptionConfig())
    perception = _ClipPerception()
    opened: list[int] = []

    def opener(path: str) -> FakeFrameSource:
        return FakeFrameSource(_clip_frames(), fps=40.0)

    def on_new_clip(index: int, path: Path) -> None:
        if opened:  # the pipeline is fresh for the first clip (run_live does the same)
            perception.reset()
        opened.append(index)

    clips = [Path(n) for n in names]
    source = live.PlaylistSource(
        clips,
        opener=opener,
        pacer=lambda inner, gate: live.PacedSource(inner, gate=gate),
        on_new_clip=on_new_clip,
        loop=not controller_kwargs.pop("once", False),
    )
    label = live.RunLabel(clips[0].stem)
    speaker = FakeSpeaker()
    system = live.LiveSystem(
        settings=settings,
        perception=perception,
        source=source,
        speaker=speaker,
        username="u",
        password="p",
        server_factory=lambda app, cfg: _NoServer(),
        recorder_factory=live.NullRecorder,
        run_id_factory=label,
    )
    lines: list[str] = []
    controller = live.PlaylistController(
        system,
        source,
        label=label,
        out=lines.append,
        wait_for_dashboard=controller_kwargs.pop("wait_for_dashboard", False),
        **controller_kwargs,
    )
    return system, source, controller, speaker, perception, lines


def _read_logs(tmp_path: Path) -> dict[str, list[dict]]:
    out = {}
    for path in sorted((tmp_path / "logs").glob("*.jsonl")):
        out[path.stem] = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    return out


def test_two_clips_are_two_separate_runs_with_nothing_leaking_across(tmp_path: Path) -> None:
    system, source, controller, speaker, perception, lines = _playlist_rig(
        tmp_path, ["alpha.mp4", "beta.mp4"], once=True, pause_between=0.4
    )
    system.start(auto_start=False)
    stop = threading.Event()
    thread = controller.start_thread(stop)
    try:
        assert _until(lambda: controller.reason is not None, timeout=20)
    finally:
        stop.set()
        thread.join(timeout=5)
        system.shutdown()
    assert controller.reason == "playlist finished" and controller.clips_played == 2

    logs = _read_logs(tmp_path)
    assert len(logs) == 2
    (id1, log1), (id2, log2) = sorted(logs.items(), key=lambda kv: kv[1][0]["t_wall"])
    assert id1 != id2 and "alpha" in id1 and "beta" in id2  # the clip name is in the run id
    for run_id, log in ((id1, log1), (id2, log2)):
        kinds = [e["event_type"] for e in log]
        assert kinds[0] == "run_started" and kinds.count("run_completed") == 1
        assert kinds[-1] == "run_completed"
        assert [e["seq"] for e in log] == list(range(len(log)))
        assert {e["run_id"] for e in log} == {run_id}
        assert [e["step_id"] for e in log if e["event_type"] == "step_confirmed"] == [
            "s1", "s2", "s3", "s4",
        ]  # fmt: skip
        assert all(e["t_video"] <= 1.2 for e in log)  # clip time: t restarts at 0 for clip 2
    # no state leaked: both runs have the same events (kind, step) in the same order
    shape = lambda log: [  # noqa: E731
        (e["event_type"], e["step_id"], e["detail"]) for e in log[1:]  # after run_started
    ]
    assert shape(log1) == shape(log2)
    assert log1[-1]["detail"] == "Experiment complete"
    assert perception.resets == 1  # the pipeline was reset once, between the clips
    spoken = [t for t, _ in speaker.calls]
    assert spoken.count("Do step one") == 2  # the first step's phrase at the start of each clip
    summary = [x for x in lines if "finished:" in x]
    assert len(summary) == 2 and "4 step_confirmed" in summary[0]
    assert "Experiment complete" in summary[0] and "alpha.mp4" in summary[0]
    assert sum("started" in x for x in lines) == 2 and lines[-1] == "playlist finished"


def test_the_last_frame_is_held_during_the_pause(tmp_path: Path) -> None:
    system, source, controller, _speaker, _perception, _lines = _playlist_rig(
        tmp_path, ["alpha.mp4", "beta.mp4"], pause_between=1.0
    )
    system.start(auto_start=False)
    stop = threading.Event()
    thread = controller.start_thread(stop)
    try:
        assert _until(lambda: controller.clips_played == 1, timeout=20)
        t_pause = time.monotonic()
        held = system.loop.frame_store.get()
        assert held.t == pytest.approx(1.1)  # the last frame of alpha (45 frames at 40 fps)
        time.sleep(0.5)
        again = system.loop.frame_store.get()
        assert again.frame_id == held.frame_id and system.router.run_state == "idle"
        assert source.clip_name == "alpha.mp4"
        assert _until(lambda: source.clip_name == "beta.mp4", timeout=5)
        assert time.monotonic() - t_pause >= 0.9  # not before the pause is over
        assert _until(lambda: controller.clips_played >= 1)
    finally:
        stop.set()
        thread.join(timeout=5)
        system.shutdown()


def test_the_first_clip_waits_for_the_dashboard_and_later_clips_do_not(tmp_path: Path) -> None:
    system, source, controller, speaker, _p, lines = _playlist_rig(
        tmp_path, ["alpha.mp4", "beta.mp4"], once=True, pause_between=0.2, wait_for_dashboard=True
    )
    system.start(auto_start=False)
    stop = threading.Event()
    thread = controller.start_thread(stop)
    try:
        assert _until(lambda: system.loop.frame_store.get() is not None)
        time.sleep(0.4)
        assert system.router.run_state == "idle" and speaker.calls == []
        assert system.loop.frame_store.get().t == 0.0 and not source.gate.released
        client = system.app.test_client()
        assert client.post("/api/run/start", headers=AUTH).status_code == 200  # the page
        assert _until(lambda: controller.reason is not None, timeout=20)  # clip 2 needs no page
    finally:
        stop.set()
        thread.join(timeout=5)
        system.shutdown()
    assert controller.clips_played == 2
    assert sum("Do step one" == t for t, _ in speaker.calls) == 2


def test_a_run_started_during_the_pause_is_finished_before_the_next_clip(tmp_path: Path) -> None:
    """A page reloaded with ?autostart=1 between clips sees 'idle' and asks to start."""
    system, source, controller, speaker, _p, _lines = _playlist_rig(
        tmp_path, ["alpha.mp4", "beta.mp4"], once=True, pause_between=1.0
    )
    system.start(auto_start=False)
    stop = threading.Event()
    thread = controller.start_thread(stop)
    try:
        assert _until(lambda: controller.clips_played == 1, timeout=20)
        assert system.app.test_client().post("/api/run/start", headers=AUTH).status_code == 200
        assert _until(lambda: controller.reason is not None, timeout=20)
    finally:
        stop.set()
        thread.join(timeout=5)
        system.shutdown()
    assert controller.reason == "playlist finished" and controller.clips_played == 2
    logs = _read_logs(tmp_path)
    assert len(logs) == 3  # alpha, the aborted one from the pause, beta
    complete = [log[-1]["event_type"] for log in logs.values()]
    assert complete == ["run_completed"] * 3


def test_advance_enter_waits_for_the_presenter(tmp_path: Path) -> None:
    pressed = threading.Event()
    system, source, controller, _s, _p, lines = _playlist_rig(
        tmp_path,
        ["alpha.mp4", "beta.mp4"],
        once=True,
        advance="enter",
        pause_between=0.0,
        input_fn=lambda: pressed.wait(30) and "",
    )
    system.start(auto_start=False)
    stop = threading.Event()
    thread = controller.start_thread(stop)
    try:
        assert _until(lambda: any("press Enter" in x for x in lines), timeout=20)
        time.sleep(0.6)  # the pause is 0: only Enter can move on
        assert source.clip_name == "alpha.mp4" and controller.clips_played == 1
        pressed.set()
        assert _until(lambda: controller.reason is not None, timeout=20)
    finally:
        stop.set()
        pressed.set()
        thread.join(timeout=5)
        system.shutdown()
    assert controller.clips_played == 2


def test_the_controller_stops_promptly_when_asked(tmp_path: Path) -> None:
    system, _source, controller, _s, _p, _lines = _playlist_rig(
        tmp_path, ["alpha.mp4", "beta.mp4"], pause_between=30.0
    )
    system.start(auto_start=False)
    stop = threading.Event()
    thread = controller.start_thread(stop)
    try:
        assert _until(lambda: controller.clips_played == 1, timeout=20)
        stop.set()
        thread.join(timeout=3)
        assert not thread.is_alive()
    finally:
        system.shutdown()
    assert controller.reason == "stopped"


def test_frame_ids_continue_and_t_restarts_across_clips() -> None:
    def opener(path: str) -> FakeFrameSource:
        return FakeFrameSource(_clip_frames(3, 10.0), fps=10.0)

    src = live.PlaylistSource(
        [Path("a.mp4"), Path("b.mp4")],
        opener=opener,
        pacer=lambda inner, gate: inner,
        sleep=lambda s: None,
    )
    got = [src.read() for _ in range(3)]
    assert [(f.frame_id, round(f.t, 1)) for f in got] == [(0, 0.0), (1, 0.1), (2, 0.2)]
    assert src.read() is got[2] and src.ended.is_set() and src.last_frame_id == 2
    assert src.read() is got[2]  # held
    assert src.advance() and not src.ended.is_set()
    nxt = [src.read() for _ in range(3)]
    assert [(f.frame_id, round(f.t, 1)) for f in nxt] == [(3, 0.0), (4, 0.1), (5, 0.2)]
    assert src.first_frame_id == 3 and src.clip_name == "b.mp4"


def test_a_non_looping_playlist_ends_after_the_last_clip() -> None:
    src = live.PlaylistSource(
        [Path("a.mp4")],
        opener=lambda p: FakeFrameSource(_clip_frames(2, 10.0), fps=10.0),
        pacer=lambda inner, gate: inner,
        loop=False,
        sleep=lambda s: None,
    )
    src.read()
    assert src.next_index() is None
    assert not src.advance() and src.exhausted and src.read() is None


def test_run_labels_keep_run_ids_safe_for_the_log_route() -> None:
    import re

    label = live.RunLabel("x015 (final)!.mp4")
    run_id = label()
    assert re.fullmatch(r"[A-Za-z0-9_-]{1,128}", run_id) and "x015_final_mp4" in run_id
    assert run_id.startswith("live-")
    assert re.fullmatch(r"live-\d{8}T\d{6}Z-[0-9a-f]{6}", live.RunLabel(None)())


def test_the_event_tally_summarises_a_run() -> None:
    from contracts import EngineEvent

    tally = live.EventTally()
    tally(EngineEvent(kind="step_confirmed", t=1.0, step_id="s1", confidence_tag="confirmed"))
    tally(
        EngineEvent(
            kind="deviation_detected",
            t=2.0,
            deviation_type="omission",
            speak="Skipped step two",
            confidence_tag="confirmed",
        )
    )
    tally(  # a stand-in: a real run_completed event must carry a RunSummary
        SimpleNamespace(
            kind="run_completed", t=3.0, speak="Experiment ended with skipped steps"
        )
    )
    text = live.EventTally.summary(tally.take())
    assert text == (
        "1 step_confirmed; 1 deviation_detected (omission); "
        "run_completed: 'Experiment ended with skipped steps'"
    )
    assert tally.take() == []
