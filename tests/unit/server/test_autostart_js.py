"""The dashboard's autostart logic (server/static/app.js) run under node against the shared stub
DOM and a scripted fetch (see dashboard_harness.js). Not a browser: it checks the logic (once per
page load, about 500 ms after the video loaded, only when the run is idle, only with
?autostart=1, banner text, poll interval); whether a real browser fires the <img> load event on
the multipart stream is on the README's pre-demo checklist. Skipped when node is not installed."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.unit.server.dashboard_harness import (
    STEP_IDS,
    experiment_payload,
    needs_node,
    run_page,
    status,
    text,
)

pytestmark = needs_node

LOADED_TWICE = {"events": ["video:load", "video:load"], "fire_timers": True, "poll": False}


def run(tmp_path: Path, search: str, run_state: str, natural_width: int = 0, steps=None) -> dict:
    pending = "p" * len(STEP_IDS)
    return run_page(
        tmp_path,
        {
            "search": search,
            "natural_width": natural_width,
            "world": {
                "status": status(pending, run_state, expected=STEP_IDS[0]),
                "experiment": experiment_payload(),
                "log": None,
            },
            "steps": [LOADED_TWICE] if steps is None else steps,
        },
    )


def test_autostart_posts_once_after_500_ms_when_idle(tmp_path: Path) -> None:
    got = run(tmp_path, "?autostart=1", "idle")
    assert got["posts"] == ["/api/run/start"]  # the load event fired twice: still one POST
    assert 500 in got["delays"]
    snap = got["snapshots"][-1]
    assert (
        text(snap, "start-banner") == "waiting to start" and snap["start-banner"]["hidden"] is False
    )
    assert 500 in got["intervals"]  # the status poll interval


def test_autostart_does_not_start_a_run_that_is_already_going(tmp_path: Path) -> None:
    got = run(tmp_path, "?autostart=1", "running")
    assert got["posts"] == [] and text(got["snapshots"][-1], "start-banner") == "running"


def test_autostart_does_not_start_a_finished_run(tmp_path: Path) -> None:
    got = run(tmp_path, "?autostart=1", "completed")
    assert got["posts"] == [] and text(got["snapshots"][-1], "start-banner") == "run finished"


@pytest.mark.parametrize("search", ["", "?autostart=0", "?x=1", "?notautostart=1"])
def test_without_the_parameter_nothing_starts_by_itself(tmp_path: Path, search: str) -> None:
    got = run(tmp_path, search, "idle")
    assert got["posts"] == [] and got["delays"] == []


def test_a_decoded_first_frame_also_triggers_it(tmp_path: Path) -> None:
    """A browser that fires no load event on the stream: naturalWidth > 0 at the next poll."""
    got = run(
        tmp_path,
        "?autostart=1",
        "idle",
        natural_width=640,
        steps=[{"fire_timers": True, "poll": False}],
    )
    assert got["posts"] == ["/api/run/start"]
