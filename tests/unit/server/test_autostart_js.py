# ruff: noqa: E501  (the node harness below is JavaScript in a string)
"""The dashboard's autostart logic (server/static/app.js) run under node against a stub DOM and a
scripted fetch. Not a browser: it checks the logic (once per page load, about 500 ms after the
video loaded, only when the run is idle, only with ?autostart=1, banner text, poll interval);
whether a real browser fires the <img> load event on the multipart stream is on the README's
pre-demo checklist. Skipped when node is not installed."""

from __future__ import annotations

import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

APP_JS = Path(__file__).resolve().parents[3] / "server" / "static" / "app.js"

HARNESS = textwrap.dedent(
    """
    const fs = require("fs");
    const [search, runState, naturalWidth] = [process.argv[3], process.argv[4], Number(process.argv[5])];
    let now = 0; const timers = []; const posts = []; const log = [];
    function el(id) {
      return { id, textContent: "", hidden: false, disabled: false, className: "",
               firstChild: null, listeners: {}, children: [], naturalWidth: 0,
               addEventListener(n, f) { this.listeners[n] = f; },
               appendChild(c) { this.children.push(c); }, removeChild() {} };
    }
    const els = {};
    global.document = { getElementById(id) { return els[id] || (els[id] = el(id)); },
                        createElement() { return el("x"); } };
    els["video"] = el("video"); els["video"].naturalWidth = naturalWidth;
    global.window = { location: { search }, setTimeout(f, ms) { timers.push({ f, at: now + ms, ms }); } };
    global.setInterval = (f, ms) => { log.push("interval " + ms); };
    global.setTimeout = global.window.setTimeout;
    const status = () => ({ run_id: "r", run_state: runState, feed_ok: true, fps: null, steps: [],
                            expected_step_id: null, next_step_say: null, recent_alerts: [] });
    global.fetch = (path, opts) => {
      if (opts && opts.method === "POST") { posts.push(path); return Promise.resolve({ ok: true, status: 200, json: () => ({}) }); }
      return Promise.resolve({ ok: true, status: 200, json: () => status() });
    };
    eval(fs.readFileSync(process.argv[2], "utf8"));
    const settle = () => new Promise(r => setImmediate(r));
    (async () => {
      await settle(); await settle();
      const video = els["video"];
      if (video.listeners.load) { video.listeners.load(); video.listeners.load(); }  // twice: once only
      const delays = timers.map(t => t.ms);
      for (const t of timers.splice(0)) { t.f(); }
      await settle(); await settle(); await settle();
      const again = timers.length;
      console.log(JSON.stringify({ posts, delays, again, banner: els["start-banner"].textContent,
                                   hidden: els["start-banner"].hidden, log }));
    })();
    """
)


def run(tmp_path: Path, search: str, run_state: str, natural_width: int = 0) -> dict:
    script = tmp_path / "harness.js"
    script.write_text(HARNESS, encoding="utf-8")
    out = subprocess.run(
        ["node", str(script), str(APP_JS), search, run_state, str(natural_width)],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def test_autostart_posts_once_after_500_ms_when_idle(tmp_path: Path) -> None:
    got = run(tmp_path, "?autostart=1", "idle")
    assert got["posts"] == ["/api/run/start"]  # the load event fired twice: still one POST
    assert 500 in got["delays"]
    assert got["banner"] == "waiting to start" and got["hidden"] is False
    assert "interval 500" in got["log"]  # the status poll interval


def test_autostart_does_not_start_a_run_that_is_already_going(tmp_path: Path) -> None:
    got = run(tmp_path, "?autostart=1", "running")
    assert got["posts"] == [] and got["banner"] == "running"


def test_autostart_does_not_start_a_finished_run(tmp_path: Path) -> None:
    got = run(tmp_path, "?autostart=1", "completed")
    assert got["posts"] == [] and got["banner"] == "run finished"


@pytest.mark.parametrize("search", ["", "?autostart=0", "?x=1", "?notautostart=1"])
def test_without_the_parameter_nothing_starts_by_itself(tmp_path: Path, search: str) -> None:
    got = run(tmp_path, search, "idle")
    assert got["posts"] == [] and got["delays"] == []


def test_a_decoded_first_frame_also_triggers_it(tmp_path: Path) -> None:
    """A browser that fires no load event on the stream: naturalWidth > 0 at the next poll."""
    script = tmp_path / "harness.js"
    script.write_text(HARNESS.replace("video.listeners.load();", ""), encoding="utf-8")
    out = subprocess.run(
        ["node", str(script), str(APP_JS), "?autostart=1", "idle", "640"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got["posts"] == ["/api/run/start"]
