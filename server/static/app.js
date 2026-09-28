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

  var els = {
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
  }

  function poll() {
    fetch("/api/status")
      .then(function (response) {
        if (!response.ok) {
          throw new Error("status request failed: " + response.status);
        }
        return response.json();
      })
      .then(function (status) {
        showError(null);
        renderStatus(status);
      })
      .catch(function (err) {
        showError("Could not reach the server: " + err.message);
      });
  }

  function postRunControl(path) {
    fetch(path, { method: "POST" })
      .then(function (response) {
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
      });
  }

  els.btnStart.addEventListener("click", function () {
    postRunControl("/api/run/start");
  });
  els.btnReset.addEventListener("click", function () {
    postRunControl("/api/run/reset");
  });

  poll();
  setInterval(poll, POLL_MS);
})();
