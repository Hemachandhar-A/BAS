// server/static/app.js -- vanilla JS, no framework, no build step
// (essential-features.md #12 point 2). Polls GET /api/status every
// POLL_MS and re-renders. Server-provided strings are always inserted
// with textContent, never innerHTML, so a step id or alert message can
// never be interpreted as markup. The browser reuses this page's own
// Basic-auth challenge for fetch() and <img src="/video_feed"> --
// no credential is read or stored by this script.
(function () {
  "use strict";

  // Must match contracts.DASHBOARD_POLL_MS.
  var POLL_MS = 500;

  var STATUS_CHIP_TEXT = {
    pending: "Pending",
    confirmed: "Confirmed",
    skipped: "Skipped",
    completed_late: "Completed late",
  };

  // ?autostart=1 (the demo URL): POST /api/run/start once, this long after the first video
  // frame has loaded, and only if the run is idle at that moment.
  var AUTOSTART_DELAY_MS = 500;
  var autostartRequested = /[?&]autostart=1(&|$)/.test(window.location.search);
  var autostartScheduled = false;
  var autostartFired = false;

  var els = {
    banner: document.getElementById("start-banner"),
    video: document.getElementById("video"),
    runId: document.getElementById("run-id"),
    runState: document.getElementById("run-state"),
    feedOk: document.getElementById("feed-ok"),
    fps: document.getElementById("fps"),
    expectedStep: document.getElementById("expected-step"),
    nextStepSay: document.getElementById("next-step-say"),
    stepList: document.getElementById("step-list"),
    alertList: document.getElementById("alert-list"),
    btnStart: document.getElementById("btn-start"),
    btnReset: document.getElementById("btn-reset"),
    errorBanner: document.getElementById("error-banner"),
  };

  function clearChildren(node) {
    while (node.firstChild) {
      node.removeChild(node.firstChild);
    }
  }

  function setText(node, text) {
    node.textContent = text;
  }

  function showError(message) {
    if (message) {
      els.errorBanner.textContent = message;
      els.errorBanner.hidden = false;
    } else {
      els.errorBanner.hidden = true;
    }
  }

  function renderSteps(steps) {
    clearChildren(els.stepList);
    steps.forEach(function (step) {
      var li = document.createElement("li");

      var label = document.createElement("span");
      label.textContent = step.step_id;

      var chip = document.createElement("span");
      chip.className = "chip chip-" + step.status;
      // Status is paired with visible text, never colour alone.
      chip.textContent = STATUS_CHIP_TEXT[step.status] || step.status;

      li.appendChild(label);
      li.appendChild(chip);
      els.stepList.appendChild(li);
    });
  }

  function describeAlert(event) {
    var parts = [];
    if (event.deviation_type) {
      parts.push(event.deviation_type);
    }
    if (event.step_id) {
      parts.push(event.step_id);
    }
    if (event.speak) {
      parts.push(event.speak);
    }
    return "t=" + event.t.toFixed(1) + "s -- " + parts.join(" -- ");
  }

  function renderAlerts(alerts) {
    clearChildren(els.alertList);
    alerts.forEach(function (event) {
      var li = document.createElement("li");
      li.textContent = describeAlert(event);
      els.alertList.appendChild(li);
    });
  }

  function renderStatus(status) {
    setText(els.runId, status.run_id || "--");
    setText(els.runState, status.run_state);
    setText(els.feedOk, status.feed_ok ? "OK" : "LOST");
    setText(els.fps, status.fps != null ? status.fps.toFixed(1) : "--");
    setText(els.expectedStep, status.expected_step_id || "--");
    setText(els.nextStepSay, status.next_step_say || "--");
    renderSteps(status.steps);
    renderAlerts(status.recent_alerts);

    els.btnStart.disabled = status.run_state === "running";
    els.btnReset.disabled = status.run_state === "idle";
    renderBanner(status.run_state);
    // Fallback for browsers that fire no load event on a multipart stream: the first decoded
    // frame gives the <img> a width.
    if (els.video.naturalWidth > 0) {
      onVideoLoaded();
    }
  }

  function renderBanner(runState) {
    if (runState === "running") {
      setText(els.banner, "running");
      els.banner.className = "banner banner-running";
    } else if (runState === "idle") {
      setText(els.banner, "waiting to start");
      els.banner.className = "banner banner-waiting";
    } else {
      setText(els.banner, "run finished");
      els.banner.className = "banner banner-finished";
    }
    els.banner.hidden = false;
  }

  function onVideoLoaded() {
    if (!autostartRequested || autostartScheduled) {
      return;
    }
    autostartScheduled = true;
    window.setTimeout(autostart, AUTOSTART_DELAY_MS);
  }

  function autostart() {
    // Once per page load, and only when the run is idle right now (a fresh status read, not the
    // last poll): a page reloaded mid-run, or opened by a second viewer, never restarts it.
    if (autostartFired) {
      return;
    }
    autostartFired = true;
    fetch("/api/status")
      .then(function (response) {
        return response.ok ? response.json() : null;
      })
      .then(function (status) {
        if (status && status.run_state === "idle") {
          postRunControl("/api/run/start", els.btnStart);
        }
      })
      .catch(function () {
        // the status poll shows the connection error; the page can still be started by hand
      });
  }

  // Guards against out-of-order responses: if a slow poll resolves after a
  // newer one already started (e.g. the server briefly lags), applying it
  // would flash stale state back onto the page. Only the response to the
  // most recently issued poll is ever rendered.
  var pollSeq = 0;

  function poll() {
    var seq = ++pollSeq;
    fetch("/api/status")
      .then(function (response) {
        if (!response.ok) {
          throw new Error("status request failed: " + response.status);
        }
        return response.json();
      })
      .then(function (status) {
        if (seq !== pollSeq) {
          return; // a newer poll already started; this response is stale
        }
        showError(null);
        renderStatus(status);
      })
      .catch(function (err) {
        if (seq !== pollSeq) {
          return;
        }
        showError("Could not reach the server: " + err.message);
      });
  }

  function postRunControl(path, button) {
    // Disabled immediately so a fast double-click can't fire the request
    // twice; renderStatus() (via the poll() below) sets the real
    // enabled/disabled state from the server's actual run_state once it
    // responds, on both the success and the "already in that state" paths.
    button.disabled = true;
    fetch(path, { method: "POST" })
      .then(function (response) {
        if (response.status === 409) {
          // Another click (or another client) already changed the run
          // state first -- not a real failure. The poll() below will show
          // the server's actual current state.
          return null;
        }
        if (!response.ok) {
          throw new Error("request failed: " + response.status);
        }
        return response.json();
      })
      .then(function () {
        poll();
      })
      .catch(function (err) {
        showError("Could not reach the server: " + err.message);
        button.disabled = false;
      });
  }

  els.btnStart.addEventListener("click", function () {
    postRunControl("/api/run/start", els.btnStart);
  });
  els.btnReset.addEventListener("click", function () {
    postRunControl("/api/run/reset", els.btnReset);
  });

  els.video.addEventListener("load", onVideoLoaded);

  poll();
  setInterval(poll, POLL_MS);
})();
