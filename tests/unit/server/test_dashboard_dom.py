"""The dashboard (server/static/app.js + index.html) rendered under node against a stub DOM and
stub API payloads built from the real contract models: idle, running, deviation, completed, feed
lost, fetch error, plus the data-access rules (what is fetched when) and the "never invent"
rules. Not a browser: layout, colour and real <img> stream behaviour are on the README's manual
checklist. Skipped when node is not installed."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.unit.server.dashboard_harness import (
    INDEX_HTML,
    NAME,
    RUN,
    SAY,
    STEP_IDS,
    alert,
    confirmed,
    entry,
    experiment_payload,
    log,
    needs_node,
    rows,
    run_page,
    seg_classes,
    status,
    step_rows,
    text,
)

pytestmark = [needs_node, pytest.mark.F12]

N = len(STEP_IDS)


def world(**kw):
    return {"experiment": experiment_payload(), **kw}


def page(tmp_path: Path, w: dict, steps: list | None = None, **extra) -> dict:
    return run_page(tmp_path, {"world": w, "steps": steps or [], **extra})


# --- idle ---------------------------------------------------------------------------------------


def test_idle_shows_step_one_as_the_cue_and_every_step_pending(tmp_path: Path) -> None:
    """After a Reset the server carries the old statuses over, keeps the old alert and the last
    fps, and (gated start) reports feed_ok False: none of that may read as current state."""
    stale = alert(12.4, "red_stowed", ["start_pressed"], "Step skipped: Start pressed")
    got = page(
        tmp_path,
        world(
            status=status("ccccscc", "idle", feed_ok=False, fps=9.1, alerts=[stale], expected=None),
            log=None,
        ),
    )
    snap = got["snapshots"][0]
    assert text(snap, "run-state") == "Idle" and text(snap, "run-id") == RUN
    assert text(snap, "feed-ok") == "—" and "is-bad" not in snap["feed-ok"]["cls"]
    assert text(snap, "fps") == "—"
    assert snap["btn-start"]["disabled"] is False and snap["btn-reset"]["disabled"] is True
    assert (
        text(snap, "start-banner") == "waiting to start" and snap["start-banner"]["hidden"] is False
    )
    assert snap["deviation-banner"]["hidden"] is True  # the stale alert belongs to another run
    assert snap["feed-lost"]["hidden"] is True and snap["video"]["hidden"] is False
    assert snap["guidance"]["hidden"] is False and snap["summary"]["hidden"] is True
    assert text(snap, "step-no") == f"Step 1 of {N}"
    assert text(snap, "cue") == SAY[STEP_IDS[0]]
    assert text(snap, "confirmed-a") == f"0 of {N} confirmed" == text(snap, "confirmed-b")
    assert snap["progress"]["attrs"]["aria-label"] == f"0 of {N} steps confirmed"
    listed = step_rows(snap)
    assert [r["status"] for r in listed] == ["Pending"] * N
    assert [r["time"] for r in listed] == ["—"] * N
    assert "is-current" in listed[0]["row_cls"] and "is-current" not in listed[1]["row_cls"]
    assert rows(snap, "event-log") == []
    # an idle run has no log yet: the page does not even ask for one
    assert got["fetches"][0] == ["/api/status", "/api/experiment"]
    assert text(snap, "experiment-name") == "Sample Transfer"


def test_idle_page_asks_for_the_experiment_once_and_never_for_a_log(tmp_path: Path) -> None:
    w = world(status=status("ppppppp", "idle", expected=STEP_IDS[0]), log=None)
    got = page(tmp_path, w, [{}, {}, {}])
    flat = [path for step in got["fetches"] for path in step]
    assert flat.count("/api/experiment") == 1
    assert not [p for p in flat if p.startswith("/api/log")]
    assert flat.count("/api/status") == 4  # the load poll and one per step


# --- running ------------------------------------------------------------------------------------


RUNNING_LOG = log(
    entry(0, "run_started", 0.0, f"Run started: {RUN}"),
    entry(1, "feed_restored", 0.0, "Feed restored"),
    confirmed(2, 3.5, "red_out"),
    confirmed(3, 3.6, "red_in_tray", tag="flagged_uncertain"),
)


def running_world(**kw):
    return world(
        status=status("ccpppp" + "p", "running", times={0: 3.5, 1: 3.6}),
        log=RUNNING_LOG,
        **kw,
    )


def test_running_shows_the_next_action_steps_and_event_log(tmp_path: Path) -> None:
    got = page(tmp_path, running_world(), [{}])
    snap = got["snapshots"][1]
    assert text(snap, "run-state") == "Running"
    assert text(snap, "feed-ok") == "OK" and text(snap, "fps") == "9.1 fps"
    assert snap["btn-start"]["disabled"] is True and snap["btn-reset"]["disabled"] is False
    assert text(snap, "start-banner") == "running"
    assert text(snap, "step-no") == f"Step 3 of {N}"
    assert text(snap, "cue") == SAY[STEP_IDS[2]]  # the say of the EXPECTED step, not the next
    assert seg_classes(snap) == [
        "seg seg-confirmed",
        "seg seg-confirmed",
        "seg seg-current",
        *["seg seg-pending"] * (N - 3),
    ]
    assert snap["progress"]["attrs"]["aria-label"] == f"2 of {N} steps confirmed"
    assert text(snap, "confirmed-a") == f"2 of {N} confirmed"
    listed = step_rows(snap)
    assert [r["status"] for r in listed][:4] == ["Confirmed", "Confirmed", "Pending", "Pending"]
    assert [r["time"] for r in listed][:3] == ["0:03.5", "0:03.6", "—"]
    assert listed[0]["title"] == NAME["red_out"]  # display_name from /api/experiment
    assert "is-current" in listed[2]["row_cls"]
    assert "is-pending" in listed[3]["row_cls"] and "is-pending" not in listed[2]["row_cls"]
    assert "is-bad" not in listed[0]["status_cls"]  # Confirmed is neutral, never coloured
    assert snap["deviation-banner"]["hidden"] is True
    assert snap["summary"]["hidden"] is True


def test_event_log_rows_use_the_log_detail_verbatim(tmp_path: Path) -> None:
    snap = page(tmp_path, running_world(), [{}])["snapshots"][1]
    listed = rows(snap, "event-log")
    assert [r["children"][0]["full"] for r in listed] == ["0:00.0", "0:00.0", "0:03.5", "0:03.6"]
    details = [r["children"][2]["children"][0]["full"] for r in listed]
    assert details == [
        f"Run started: {RUN}",
        "Feed restored",
        f"Confirmed: {NAME['red_out']}",
        f"Confirmed: {NAME['red_in_tray']}",
    ]
    # Never an invented spoken line: confirmations and run start have none.
    assert not any("Spoken" in r["full"] for r in listed)
    # The amber tag only where confidence_tag is flagged_uncertain (the 4th entry).
    assert ["Low confidence" in r["full"] for r in listed] == [False, False, False, True]


def test_log_is_fetched_by_run_id_on_each_poll_while_a_run_exists(tmp_path: Path) -> None:
    got = page(tmp_path, running_world(), [{}, {}])
    for step in got["fetches"]:
        assert f"/api/log?run_id={RUN}" in step
    experiment_calls = [p for step in got["fetches"] for p in step if p == "/api/experiment"]
    assert len(experiment_calls) == 1  # once per run, not per poll


def test_the_log_appends_new_rows_and_a_new_run_clears_it(tmp_path: Path) -> None:
    more = log(*RUNNING_LOG["entries"], confirmed(4, 8.0, "yellow_out"))
    other = "live-20261006T190000Z-aaaaaa"
    got = page(
        tmp_path,
        running_world(),
        [
            {"set": {"log": more}},
            {
                "set": {
                    "status": status("pppppp" + "p", "running", run_id=other),
                    "log": log(entry(0, "run_started", 0.0, f"Run started: {other}"), run_id=other),
                }
            },
        ],
    )
    counts = [len(rows(s, "event-log")) for s in got["snapshots"]]
    assert counts == [4, 5, 1]  # the log of the old run is gone after the run changes
    assert "/api/experiment" in got["fetches"][2]  # refetched for the new run


def test_log_autoscrolls_to_newest_unless_the_user_scrolled_up(tmp_path: Path) -> None:
    many = log(*[entry(i, "feed_restored", float(i), "Feed restored") for i in range(12)])
    more = log(*many["entries"], entry(12, "feed_restored", 12.0, "Feed restored"))
    more2 = log(*more["entries"], entry(13, "feed_restored", 13.0, "Feed restored"))
    w = world(status=status("pppppp" + "p"), log=many)
    got = page(
        tmp_path,
        w,
        [
            {"set": {"log": more}},  # follows: the user is at the bottom
            {"scroll": {"id": "event-log", "top": 0}, "set": {"log": more2}},  # scrolled up
        ],
    )
    s0, s1, s2 = got["snapshots"]
    assert s0["event-log"]["scrollTop"] == 12 * 60  # at the bottom after the first render
    assert s1["event-log"]["scrollTop"] == 13 * 60  # followed the new row
    assert s2["event-log"]["scrollTop"] == 0  # scrolled up: the user's position is left alone


# --- deviation ----------------------------------------------------------------------------------


DEVIATION_ALERT = alert(12.495, "red_stowed", ["start_pressed"], "Step skipped: Start pressed")
DEVIATION_LOG = log(
    entry(0, "run_started", 0.0, f"Run started: {RUN}"),
    confirmed(1, 3.5, "red_out"),
    entry(
        2,
        "deviation_detected",
        12.495,
        "Step skipped: Start pressed",
        step_id="red_stowed",
        deviation_type="omission",
        skipped=["start_pressed"],
        tag="confirmed",
    ),
)


def deviation_world():
    return world(
        status=status("ccccscp", "running", times={0: 3.5, 5: 12.5}, alerts=[DEVIATION_ALERT]),
        log=DEVIATION_LOG,
    )


def test_a_deviation_shows_the_alert_banner_a_red_row_and_a_skipped_step(tmp_path: Path) -> None:
    snap = page(tmp_path, deviation_world(), [{}])["snapshots"][1]
    banner = snap["deviation-banner"]
    assert banner["hidden"] is False
    assert text(snap, "deviation-label") == "Voice alert"
    assert text(snap, "deviation-msg") == "Step skipped: Start pressed"  # EngineEvent.speak
    assert text(snap, "deviation-time") == "Video time 0:12.4"
    last = rows(snap, "event-log")[-1]
    assert "is-bad" in last["cls"]
    assert "Spoken: “Step skipped: Start pressed”" in last["full"]
    listed = step_rows(snap)
    assert listed[4]["status"] == "Skipped" and "is-bad" in listed[4]["status_cls"]
    assert listed[4]["time"] == "—"
    assert seg_classes(snap)[4] == "seg seg-skipped"
    assert text(snap, "confirmed-a") == f"5 of {N} confirmed"


def test_an_alert_left_over_from_a_previous_run_does_not_raise_the_banner(tmp_path: Path) -> None:
    """The server never clears recent_alerts, so the banner follows THIS run's log."""
    w = world(
        status=status("ccpppp" + "p", "running", alerts=[DEVIATION_ALERT]),
        log=log(entry(0, "run_started", 0.0, f"Run started: {RUN}")),
    )
    snap = page(tmp_path, w, [{}])["snapshots"][1]
    assert snap["deviation-banner"]["hidden"] is True


def test_a_deviation_without_a_matching_alert_uses_the_log_detail(tmp_path: Path) -> None:
    w = deviation_world()
    w["status"] = status("ccccscp", "running", alerts=[])  # the alert fell out of the history
    snap = page(tmp_path, w, [{}])["snapshots"][1]
    assert text(snap, "deviation-msg") == "Step skipped: Start pressed"
    assert text(snap, "deviation-label") == "Voice alert"  # omission is always spoken


def test_a_repeat_is_never_labelled_spoken(tmp_path: Path) -> None:
    """A repeat may be silenced by configuration, so neither a Spoken line nor 'Voice alert'."""
    repeat = entry(
        2, "deviation_detected", 5.0, f"Repeated: {NAME['red_out']}", step_id="red_out",
        deviation_type="repeat", tag="confirmed",
    )  # fmt: skip
    w = world(
        status=status("ccpppp" + "p"), log=log(entry(0, "run_started", 0.0, "Run started"), repeat)
    )
    snap = page(tmp_path, w, [{}])["snapshots"][1]
    assert text(snap, "deviation-label") == "Deviation"
    assert text(snap, "deviation-msg") == f"Repeated: {NAME['red_out']}"
    assert "Spoken" not in text(snap, "event-log")


# --- completed ----------------------------------------------------------------------------------


COMPLETED_LOG = log(
    *DEVIATION_LOG["entries"],
    confirmed(3, 16.3, "yellow_stowed"),
    entry(4, "run_completed", 16.3, "Experiment ended with skipped steps", tag="confirmed"),
)


def test_completed_with_a_skipped_step_shows_the_summary(tmp_path: Path) -> None:
    w = world(
        status=status("ccccscc", "completed", times={0: 3.5, 6: 16.3}, alerts=[DEVIATION_ALERT]),
        log=COMPLETED_LOG,
    )
    got = page(tmp_path, w, [{}, {}])
    snap = got["snapshots"][1]
    assert snap["guidance"]["hidden"] is True and snap["summary"]["hidden"] is False
    assert text(snap, "summary-headline") == "Experiment ended with skipped steps"
    assert text(snap, "tile-skipped") == "1" and "is-bad" in snap["tile-skipped"]["cls"]
    assert text(snap, "tile-late") == "0" and "is-bad" not in snap["tile-late"]["cls"]
    assert text(snap, "summary-skipped") == f"Skipped: {NAME['start_pressed']}"
    assert text(snap, "summary-late") == "Completed late: none"
    assert "is-bad" in snap["summary-icon"]["cls"]
    assert text(snap, "run-state") == "Completed" and text(snap, "start-banner") == "run finished"
    assert snap["btn-start"]["disabled"] is False and snap["btn-reset"]["disabled"] is False
    assert "seg-current" not in " ".join(seg_classes(snap))
    assert not any("is-current" in r["row_cls"] for r in step_rows(snap))
    # The completion row repeats the spoken phrase (the engine always speaks it).
    assert "Spoken" in rows(snap, "event-log")[-1]["full"]
    # Order similarity is not exposed by any route, so the page must not show or compute it.
    everything = " ".join(node["full"] for node in snap.values())
    assert "similarity" not in everything.lower() and "%" not in everything
    # The run_completed line has been read: the log is not polled any more.
    assert not [p for p in got["fetches"][2] if p.startswith("/api/log")]


def test_completed_clean_is_neutral_and_late_steps_are_counted(tmp_path: Path) -> None:
    clean = world(
        status=status("c" * N, "completed"),
        log=log(entry(0, "run_completed", 19.7, "Experiment complete", tag="confirmed")),
    )
    snap = page(tmp_path, clean)["snapshots"][0]
    assert text(snap, "summary-headline") == "Experiment complete"
    assert "is-bad" not in snap["summary-icon"]["cls"]
    assert text(snap, "tile-skipped") == "0" and text(snap, "tile-late") == "0"

    late = world(
        status=status("cclcccc", "completed", times={2: 9.0}),
        log=log(entry(0, "run_completed", 19.7, "Experiment complete", tag="confirmed")),
    )
    snap = page(tmp_path, late)["snapshots"][0]
    assert text(snap, "tile-late") == "1" and "is-bad" in snap["tile-late"]["cls"]
    assert text(snap, "summary-late") == f"Completed late: {NAME[STEP_IDS[2]]}"
    row = step_rows(snap)[2]
    assert row["status"] == "Completed late" and "is-bad" in row["status_cls"]
    assert row["time"] == "0:09.0"


def test_completed_before_the_log_arrives_has_a_generic_headline(tmp_path: Path) -> None:
    w = world(status=status("c" * N, "completed"), log=None)
    snap = page(tmp_path, w)["snapshots"][0]
    assert text(snap, "summary-headline") == "Run finished"


# --- feed lost ----------------------------------------------------------------------------------


def test_feed_lost_replaces_the_video_and_retries_no_faster_than_every_3_s(tmp_path: Path) -> None:
    w = running_world()
    w["status"] = status("ccpppp" + "p", "running", feed_ok=False)
    got = page(
        tmp_path,
        w,
        [
            {"now": 100500},
            {"now": 102000},  # 2 s after the first retry: no new reconnect
            {"now": 103100},  # more than 3 s after it
            {"set": {"status": status("ccpppp" + "p", "running", feed_ok=True)}},
        ],
        now=100000,
    )
    first, s1, s2, s3, back = got["snapshots"]
    assert first["feed-lost"]["hidden"] is False and first["video"]["hidden"] is True
    markup = INDEX_HTML.read_text(encoding="utf-8")  # the lost state's own text is static markup
    assert '<div class="lost-title">Camera feed lost</div>' in markup
    assert '<div class="lost-sub">No frames received</div>' in markup
    assert text(first, "feed-ok") == "Lost" and "is-bad" in first["feed-ok"]["cls"]
    assert first["video"]["src"].startswith("/video_feed?t=")
    assert s1["video"]["src"] == first["video"]["src"]
    assert s2["video"]["src"] == first["video"]["src"]  # 2.0 s since the last retry
    assert s3["video"]["src"] != first["video"]["src"]  # 3.1 s: a new cache-busting URL
    assert back["feed-lost"]["hidden"] is True and back["video"]["hidden"] is False


def test_an_image_error_also_shows_the_feed_lost_state_until_it_loads_again(tmp_path: Path) -> None:
    got = page(
        tmp_path,
        running_world(),
        [{"events": ["video:error"]}, {"events": ["video:load"]}],
    )
    ok, err, loaded = got["snapshots"]
    assert ok["feed-lost"]["hidden"] is True
    assert err["feed-lost"]["hidden"] is False and err["video"]["hidden"] is True
    assert loaded["feed-lost"]["hidden"] is True and loaded["video"]["hidden"] is False


# --- fetch error --------------------------------------------------------------------------------


def test_a_failed_status_request_shows_the_error_banner_and_recovers(tmp_path: Path) -> None:
    got = page(
        tmp_path,
        running_world(),
        [{"set": {"fail_status": True}}, {"set": {"fail_status": False}}],
    )
    ok, failed, recovered = got["snapshots"]
    assert ok["error-banner"]["hidden"] is True
    assert failed["error-banner"]["hidden"] is False
    assert text(failed, "error-banner") == "Could not reach the server: status request failed: 500"
    assert text(failed, "run-state") == "Running"  # the last good state stays on screen
    assert recovered["error-banner"]["hidden"] is True


def test_a_failed_log_request_is_shown_too(tmp_path: Path) -> None:
    got = page(tmp_path, running_world(), [{"set": {"log": "error"}}])
    assert "log request failed: 500" in text(got["snapshots"][1], "error-banner")


def test_without_the_experiment_the_page_falls_back_to_step_ids(tmp_path: Path) -> None:
    w = running_world(fail_experiment=True)
    got = page(tmp_path, w, [{}])
    snap = got["snapshots"][1]
    assert text(snap, "cue") == STEP_IDS[2]
    assert step_rows(snap)[0]["title"] == STEP_IDS[0]
    assert text(snap, "experiment-name") == ""
    assert "experiment request failed" in text(snap, "error-banner")


# --- server strings are text, never markup ------------------------------------------------------


def test_server_strings_are_written_as_text(tmp_path: Path) -> None:
    """The stub DOM throws on any innerHTML/outerHTML/insertAdjacentHTML use."""
    evil = "<img src=x onerror=alert(1)>"
    definition = experiment_payload()
    definition["steps"][0]["display_name"] = evil
    definition["steps"][0]["say"] = "<b>say</b>"
    w = {
        "experiment": definition,
        "status": status("pppppp" + "p", "running"),
        "log": log(
            entry(0, "run_started", 0.0, "<script>alert(1)</script>"),
            entry(
                1,
                "deviation_detected",
                2.0,
                evil,
                step_id="red_out",
                deviation_type="omission",
                tag="flagged_uncertain",
            ),  # fmt: skip
        ),
    }
    snap = page(tmp_path, w, [{}])["snapshots"][1]
    assert text(snap, "cue") == "<b>say</b>"
    assert step_rows(snap)[0]["title"] == evil
    assert text(snap, "deviation-msg") == evil
    assert "<script>alert(1)</script>" in text(snap, "event-log")


# --- time format --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("t", "shown"),
    [(0.0, "0:00.0"), (12.495, "0:12.4"), (59.99, "0:59.9"), (60.0, "1:00.0"), (605.04, "10:05.0")],
)
def test_video_time_is_m_ss_s_truncated_to_the_tenth(tmp_path: Path, t: float, shown: str) -> None:
    w = world(status=status("c" + "p" * (N - 1), times={0: t}), log=None)
    snap = page(tmp_path, w)["snapshots"][0]
    assert step_rows(snap)[0]["time"] == shown
