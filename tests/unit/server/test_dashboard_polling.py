"""B1 (S-I1c review): the dashboard must notice a dead or hanging server.

Real browsers take about 21 s to give up a connect to a dead port on Windows. The old page started
a request every 500 ms and dropped every response that was not the newest, so with a hanging
server no request ever settled and neither a success nor a failure was applied: the page kept
showing Running / Feed OK for good. These tests run the page in the node harness on a VIRTUAL
clock (a hang costs no real time) with a fetch that never settles unless it is aborted.
Skipped when node is not installed."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.unit.server.dashboard_harness import (
    APP_JS,
    INDEX_HTML,
    STEP_IDS,
    confirmed,
    entry,
    experiment_payload,
    log,
    needs_node,
    rows,
    run_page,
    status,
    text,
)

pytestmark = [needs_node, pytest.mark.F12]

DASH = "—"
GOOD_LOG = log(
    entry(0, "run_started", 0.0, "Run started: r"),
    confirmed(1, 3.5, "red_out"),
)


def good_world(**kw):
    return {
        "experiment": experiment_payload(),
        "status": status("cppppp" + "p", "running", times={0: 3.5}),
        "log": GOOD_LOG,
        **kw,
    }


def page(tmp_path: Path, w: dict, steps: list | None = None) -> dict:
    return run_page(tmp_path, {"world": w, "steps": steps or []})


# --- the original failure ------------------------------------------------------------------------


def test_a_server_that_goes_silent_shows_the_error_banner_within_3_s(tmp_path: Path) -> None:
    got = page(tmp_path, good_world(), [{"set": {"hang_status": True}, "advance": 3000}])
    before, after = got["snapshots"]
    assert before["error-banner"]["hidden"] is True
    assert after["error-banner"]["hidden"] is False
    assert text(after, "error-banner") == "Could not reach the server: status request timed out"


def test_a_server_that_is_dead_from_the_start_shows_the_banner_within_3_s(tmp_path: Path) -> None:
    got = page(tmp_path, good_world(hang_status=True), [{"advance": 3000, "poll": True}])
    assert got["snapshots"][-1]["error-banner"]["hidden"] is False


def test_status_polls_never_overlap(tmp_path: Path) -> None:
    got = page(tmp_path, good_world(hang_status=True), [{"advance": 20000}])
    assert got["max_inflight_status"] == 1  # one request at a time, however long it hangs
    assert got["snapshots"][-1]["error-banner"]["hidden"] is False


def test_a_slow_server_never_gets_overlapping_polls_either(tmp_path: Path) -> None:
    got = page(tmp_path, good_world(status_delay_ms=1800), [{"advance": 20000}])
    assert got["max_inflight_status"] == 1
    assert got["snapshots"][-1]["error-banner"]["hidden"] is True  # 1.8 s is under the timeout


def test_a_slow_but_successful_response_is_applied_not_dropped(tmp_path: Path) -> None:
    got = page(tmp_path, good_world(status_delay_ms=1500), [{"advance": 1600}])
    snap = got["snapshots"][-1]
    assert text(snap, "run-state") == "Running" and snap["error-banner"]["hidden"] is True
    assert text(snap, "step-no") == f"Step 2 of {len(STEP_IDS)}"


# --- what the page shows while the server is unreachable ----------------------------------------


def test_after_a_failure_state_feed_and_processing_are_dashes(tmp_path: Path) -> None:
    got = page(tmp_path, good_world(), [{"set": {"hang_status": True}, "advance": 3000}])
    ok, failed = got["snapshots"]
    assert (text(ok, "run-state"), text(ok, "feed-ok"), text(ok, "fps")) == (
        "Running",
        "OK",
        "9.1 fps",
    )
    assert text(failed, "run-state") == DASH
    assert text(failed, "feed-ok") == DASH and failed["feed-ok"]["children"] == []  # no check icon
    assert text(failed, "fps") == DASH
    # The page cannot know the camera is lost: no lost panel, no red Feed.
    assert failed["feed-lost"]["hidden"] is True and "is-bad" not in failed["feed-ok"]["cls"]
    # The last good log stays on screen.
    assert len(rows(failed, "event-log")) == len(rows(ok, "event-log")) == 2


# A request already hanging when the server comes back is only given up at its 2 s timeout, so the
# first good answer can take up to about 2.5 s: the recovery steps below run 2.6 s.


def test_a_later_successful_poll_clears_the_banner_and_brings_the_values_back(
    tmp_path: Path,
) -> None:
    got = page(
        tmp_path,
        good_world(),
        [
            {"set": {"hang_status": True}, "advance": 3000},
            {"set": {"hang_status": False}, "advance": 2600},
        ],
    )
    failed, back = got["snapshots"][1], got["snapshots"][2]
    assert failed["error-banner"]["hidden"] is False
    assert back["error-banner"]["hidden"] is True
    assert (text(back, "run-state"), text(back, "feed-ok"), text(back, "fps")) == (
        "Running",
        "OK",
        "9.1 fps",
    )


def test_a_page_opened_while_the_server_is_down_recovers_when_it_comes_back(
    tmp_path: Path,
) -> None:
    got = page(
        tmp_path,
        good_world(hang_status=True),
        [{"advance": 3000, "poll": True}, {"set": {"hang_status": False}, "advance": 2600}],
    )
    assert got["snapshots"][-1]["error-banner"]["hidden"] is True
    assert text(got["snapshots"][-1], "run-state") == "Running"


# --- the other requests time out too, and never pile up -----------------------------------------


def test_a_hanging_log_request_times_out_without_stopping_the_status_poll(tmp_path: Path) -> None:
    got = page(tmp_path, good_world(hang_log=True), [{"advance": 3000}])
    snap = got["snapshots"][-1]
    assert text(snap, "error-banner") == "Could not reach the server: log request timed out"
    assert text(snap, "run-state") == "Running"  # status still arrives
    logs = [p for step in got["fetches"] for p in step if p.startswith("/api/log")]
    assert len(logs) <= 3  # one at a time, each given up after 2 s


def test_a_hanging_experiment_request_times_out(tmp_path: Path) -> None:
    got = page(tmp_path, good_world(hang_experiment=True), [{"advance": 3000}])
    snap = got["snapshots"][-1]
    assert text(snap, "error-banner") == "Could not reach the server: experiment request timed out"
    experiments = [p for step in got["fetches"] for p in step if p == "/api/experiment"]
    assert len(experiments) <= 3


def test_pressing_a_button_during_a_slow_poll_does_not_start_a_second_poll(tmp_path: Path) -> None:
    got = page(
        tmp_path,
        good_world(status_delay_ms=1000),
        [{"click": "btn-reset", "advance": 5000}],
    )
    assert got["max_inflight_status"] == 1
    assert got["posts"] == ["/api/run/reset"]


# --- source-level rules ---------------------------------------------------------------------------


def code() -> str:
    source = APP_JS.read_text(encoding="utf-8")
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    return re.sub(r"(?m)//.*$", "", source)


def test_one_request_helper_serves_every_call_and_the_timeout_is_one_named_constant() -> None:
    js = code()
    assert js.count("fetch(") == 1, "every request must go through the one helper"
    assert re.search(r"FETCH_TIMEOUT_MS\s*=\s*2000;", js)
    assert "AbortController" in js and "FETCH_TIMEOUT_MS" in js.split("function request")[1]
    assert "2000" not in js.replace("FETCH_TIMEOUT_MS = 2000", "")  # no second literal
    assert "POLL_MS = 500" in js  # the file still says it (test_static_offline reads it)
    assert "setInterval" not in js, "the next poll is scheduled after the current one settles"


# --- review wording (N1) and scrollbar (N3) ------------------------------------------------------


def test_voice_wording_does_not_claim_more_than_the_page_knows(tmp_path: Path) -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert "Voice cue text" in html and "Also spoken aloud" not in html
    assert "Spoken" not in code() and "Voice text: " in code()
    assert "Voice alert" in code()  # the banner label stays
