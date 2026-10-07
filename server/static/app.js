// server/static/app.js -- vanilla JS, no framework, no build step
// (essential-features.md #12 point 2). Polls GET /api/status every POLL_MS and re-renders from
// one in-memory model. Server-provided strings are only ever written with textContent, never
// innerHTML, so a step name, a log detail or an alert can never be interpreted as markup. The
// browser reuses this page's own Basic-auth challenge for fetch() and <img src="/video_feed"> --
// no credential is read or stored by this script.
//
// Everything this page knows comes from the API: no step text, no order similarity (that is
// RunSummary.pos, which no route exposes) and no spoken text that the API cannot confirm.
(function () {
  "use strict";

  // ---- data access: the ONLY place that talks to the server ------------------------------------
  // A static replay can swap this object without touching any rendering code below. Each call
  // returns a promise; a failed request rejects with an Error whose message names the request.
  var api = {
    status: function () {
      return getJson("/api/status", "status");
    },
    // Fetched once per run (the definition never changes while a run is going).
    experiment: function () {
      return getJson("/api/experiment", "experiment");
    },
    // One run's log. 404 means "no log written yet" (an idle run has none): not an error.
    log: function (runId) {
      return getJson("/api/log?run_id=" + encodeURIComponent(runId), "log", true).then(function (body) {
        return body || { run_id: runId, entries: [] };
      });
    },
    startRun: function () {
      return post("/api/run/start");
    },
    resetRun: function () {
      return post("/api/run/reset");
    },
    // bust = a number makes the URL unique so the browser really reconnects.
    videoUrl: function (bust) {
      return bust ? "/video_feed?t=" + bust : "/video_feed";
    },
  };

  // The ONE place that calls fetch. Every request is given up after FETCH_TIMEOUT_MS (a connect to
  // a dead port otherwise hangs for about 21 s on Windows) and a timeout rejects like any other
  // failure. `accept` receives the Response and returns the value, or throws.
  function request(path, options, what, accept) {
    var controller = typeof AbortController === "function" ? new AbortController() : null;
    var timer = setTimeout(function () {
      if (controller) {
        controller.abort();
      }
    }, FETCH_TIMEOUT_MS);
    var init = options || {};
    if (controller) {
      init.signal = controller.signal;
    }
    return fetch(path, init)
      .then(accept)
      .then(
        function (value) {
          clearTimeout(timer);
          return value;
        },
        function (err) {
          clearTimeout(timer);
          if (err && err.name === "AbortError") {
            throw new Error(what + " request timed out");
          }
          throw err;
        }
      );
  }

  function getJson(path, what, emptyOn404) {
    return request(path, null, what, function (response) {
      if (emptyOn404 && response.status === 404) {
        return null;
      }
      if (!response.ok) {
        throw new Error(what + " request failed: " + response.status);
      }
      return response.json();
    });
  }

  function post(path) {
    return request(path, { method: "POST" }, "run control", function (response) {
      if (response.status === 409) {
        // Another click (or another client) already changed the run state first: not a failure.
        // The poll that follows shows the server's actual state.
        return null;
      }
      if (!response.ok) {
        throw new Error("request failed: " + response.status);
      }
      return response.json();
    });
  }

  // ---- constants --------------------------------------------------------------------------------

  // Must match contracts.DASHBOARD_POLL_MS.
  var POLL_MS = 500;
  // Design choice (S-I1c-fix): give up any request after this long. Four times the poll period,
  // far above a healthy loopback answer, far below the browser's own connect timeout.
  var FETCH_TIMEOUT_MS = 2000;
  // A lost feed reconnects the video stream no faster than this.
  var VIDEO_RETRY_MS = 3000;
  var DASH = "—";

  var STATUS_LABEL = {
    pending: "Pending",
    confirmed: "Confirmed",
    skipped: "Skipped",
    completed_late: "Completed late",
  };
  var STATUS_ICON = {
    pending: "circle",
    confirmed: "check",
    skipped: "skip",
    completed_late: "clock",
  };
  var STATE_LABEL = { idle: "Idle", running: "Running", completed: "Completed" };

  // ?autostart=1 (the demo URL): POST /api/run/start once, this long after the first video
  // frame has loaded, and only if the run is idle at that moment.
  var AUTOSTART_DELAY_MS = 500;
  var autostartRequested = /[?&]autostart=1(&|$)/.test(window.location.search);
  var autostartScheduled = false;
  var autostartFired = false;

  // ---- elements ---------------------------------------------------------------------------------

  function byId(id) {
    return document.getElementById(id);
  }

  var els = {
    banner: byId("start-banner"),
    video: byId("video"),
    feedLost: byId("feed-lost"),
    experimentName: byId("experiment-name"),
    runId: byId("run-id"),
    runState: byId("run-state"),
    feedOk: byId("feed-ok"),
    fps: byId("fps"),
    btnStart: byId("btn-start"),
    btnReset: byId("btn-reset"),
    errorBanner: byId("error-banner"),
    deviation: byId("deviation-banner"),
    deviationLabel: byId("deviation-label"),
    deviationMsg: byId("deviation-msg"),
    deviationTime: byId("deviation-time"),
    eventLog: byId("event-log"),
    guidance: byId("guidance"),
    stepNo: byId("step-no"),
    cue: byId("cue"),
    progress: byId("progress"),
    confirmedA: byId("confirmed-a"),
    summary: byId("summary"),
    summaryIcon: byId("summary-icon"),
    summaryHeadline: byId("summary-headline"),
    tileSkipped: byId("tile-skipped"),
    tileLate: byId("tile-late"),
    summarySkipped: byId("summary-skipped"),
    summaryLate: byId("summary-late"),
    stepList: byId("step-list"),
    confirmedB: byId("confirmed-b"),
    sprite: byId("icon-sprite"),
  };

  // ---- small DOM helpers (static strings and server strings alike go through textContent) -------

  function clearChildren(node) {
    while (node.firstChild) {
      node.removeChild(node.firstChild);
    }
  }

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) {
      node.className = className;
    }
    if (text !== undefined && text !== null) {
      node.textContent = text;
    }
    return node;
  }

  // An <svg><use href="#i-name"/></svg> that points into the sprite in index.html. The SVG
  // namespace is read off the sprite element, so no URL string is needed in this file.
  function icon(name, size) {
    var ns = els.sprite.namespaceURI;
    var svg = document.createElementNS(ns, "svg");
    svg.setAttribute("class", "icon");
    svg.setAttribute("width", String(size));
    svg.setAttribute("height", String(size));
    svg.setAttribute("aria-hidden", "true");
    var use = document.createElementNS(ns, "use");
    use.setAttribute("href", "#i-" + name);
    svg.appendChild(use);
    return svg;
  }

  // Video time as m:ss.s, truncated to the tenth (never rounds up to 0:60.0).
  function fmtTime(t) {
    if (typeof t !== "number" || !isFinite(t) || t < 0) {
      return DASH;
    }
    var tenths = Math.floor(t * 10 + 1e-6);
    var minutes = Math.floor(tenths / 600);
    var seconds = (tenths % 600) / 10;
    return minutes + ":" + (seconds < 10 ? "0" : "") + seconds.toFixed(1);
  }

  function humanize(id) {
    return String(id)
      .split("_")
      .filter(Boolean)
      .map(function (word) {
        return word.charAt(0).toUpperCase() + word.slice(1);
      })
      .join(" ");
  }

  // ---- model ------------------------------------------------------------------------------------

  var model = {
    status: null,
    runId: null,
    // /api/experiment, fetched once per run: {name, byId: {step_id: {display_name, say}}}
    experiment: null,
    experimentRunId: null,
    // this run's /api/log entries (empty while idle) and whether run_completed has been seen
    log: [],
    logComplete: false,
    // the last status request failed or timed out
    unreachable: false,
  };
  var inflight = { experiment: false, log: false };
  var errors = { status: null, experiment: null, log: null };
  var imgError = false;
  var lastVideoRetry = 0;
  var logRendered = { runId: null, count: 0 };
  var stepsSignature = "";

  function stepTitle(stepId) {
    var def = model.experiment && model.experiment.byId[stepId];
    return def && def.display_name ? def.display_name : stepId;
  }

  function stepCue(stepId) {
    var def = model.experiment && model.experiment.byId[stepId];
    return def && def.say ? def.say : stepId;
  }

  // An idle run has made no progress by definition. The server carries the previous run's step
  // statuses over after a Reset (as reference); showing them as the current state would read as
  // "everything confirmed" next to a Start button, so idle renders every step as pending.
  function effectiveSteps(status) {
    if (status.run_state !== "idle") {
      return status.steps;
    }
    return status.steps.map(function (step) {
      return { step_id: step.step_id, status: "pending", t_confirmed: null };
    });
  }

  function currentIndex(status, steps) {
    if (status.run_state === "completed") {
      return -1;
    }
    if (status.run_state === "idle") {
      return steps.length ? 0 : -1;
    }
    var i;
    for (i = 0; i < steps.length; i++) {
      if (steps[i].step_id === status.expected_step_id) {
        return i;
      }
    }
    for (i = 0; i < steps.length; i++) {
      if (steps[i].status === "pending") {
        return i;
      }
    }
    return -1;
  }

  function countStatus(steps, wanted) {
    return steps.filter(function (step) {
      return step.status === wanted;
    }).length;
  }

  function showErrors() {
    var message = errors.status || errors.experiment || errors.log;
    if (message) {
      els.errorBanner.textContent = message;
      els.errorBanner.hidden = false;
    } else {
      els.errorBanner.hidden = true;
    }
  }

  function unreachable(err) {
    return "Could not reach the server: " + err.message;
  }

  // ---- rendering --------------------------------------------------------------------------------

  function render() {
    var status = model.status;
    if (!status) {
      return;
    }
    var steps = effectiveSteps(status);
    var index = currentIndex(status, steps);
    renderTopBar(status);
    renderBanner(status.run_state);
    renderVideo(status);
    renderDeviation(status);
    renderLog();
    renderGuidance(status, steps, index);
    renderSummary(status, steps);
    renderSteps(steps, index);
    els.btnStart.disabled = status.run_state === "running";
    els.btnReset.disabled = status.run_state === "idle";
    // Fallback for browsers that fire no load event on a multipart stream: the first decoded
    // frame gives the <img> a width.
    if (els.video.naturalWidth > 0) {
      onVideoLoaded();
    }
  }

  function renderTopBar(status) {
    var idle = status.run_state === "idle";
    els.experimentName.textContent = model.experiment ? model.experiment.name : "";
    els.runId.textContent = status.run_id || DASH;
    var stale = model.unreachable;
    els.runState.textContent = stale ? DASH : STATE_LABEL[status.run_state] || status.run_state;
    // Processing rate: not shown while idle (nothing is being processed, and the server keeps the
    // last run's number there) nor while the server cannot be reached.
    els.fps.textContent =
      !stale && !idle && status.fps != null ? status.fps.toFixed(1) + " fps" : DASH;

    clearChildren(els.feedOk);
    els.feedOk.className = "v feed-val";
    if (stale) {
      els.feedOk.textContent = DASH; // unknown: no OK, no check icon, and no claim of a lost camera
    } else if (idle) {
      // The gated start serves a held first frame and captures nothing until Start, so the
      // server's feed watchdog reports feed_ok = false by design; that is not a lost camera.
      els.feedOk.textContent = DASH;
    } else if (status.feed_ok) {
      els.feedOk.appendChild(icon("check", 16));
      els.feedOk.appendChild(el("span", "", "OK"));
    } else {
      els.feedOk.className = "v feed-val is-bad";
      els.feedOk.appendChild(icon("camoff", 16));
      els.feedOk.appendChild(el("span", "", "Lost"));
    }
  }

  function renderBanner(runState) {
    if (runState === "running") {
      els.banner.textContent = "running";
    } else if (runState === "idle") {
      els.banner.textContent = "waiting to start";
    } else {
      els.banner.textContent = "run finished";
    }
    els.banner.hidden = false;
  }

  function renderVideo(status) {
    var lost = (status.run_state !== "idle" && status.feed_ok === false) || imgError;
    els.feedLost.hidden = !lost;
    els.video.hidden = lost;
    if (lost) {
      retryVideo();
    }
  }

  function retryVideo() {
    var now = Date.now();
    if (now - lastVideoRetry >= VIDEO_RETRY_MS) {
      lastVideoRetry = now;
      els.video.src = api.videoUrl(now);
    }
  }

  // Whether the server voices exactly the log entry's detail. True for omission and out-of-order
  // deviations (the engine always speaks them, and the log detail IS the spoken text) and for run
  // completion. False for confirmations (the voice says the NEXT step, which the log does not
  // carry), a repeat (it may be silenced by configuration) and runtime events.
  function spokenEqualsDetail(entry) {
    if (entry.event_type === "run_completed") {
      return true;
    }
    return entry.event_type === "deviation_detected" && entry.deviation_type !== "repeat";
  }

  function newestDeviation() {
    for (var i = model.log.length - 1; i >= 0; i--) {
      if (model.log[i].event_type === "deviation_detected") {
        return model.log[i];
      }
    }
    return null;
  }

  // recent_alerts is NOT cleared between runs by the server, so "this run has a deviation" comes
  // from this run's log; the alert (EngineEvent.speak) is looked up by its time only for the text.
  function alertFor(status, entry) {
    var alerts = status.recent_alerts || [];
    for (var i = alerts.length - 1; i >= 0; i--) {
      var alert = alerts[i];
      if (
        alert.kind === "deviation_detected" &&
        alert.deviation_type === entry.deviation_type &&
        Math.abs(alert.t - entry.t_video) < 1e-6
      ) {
        return alert;
      }
    }
    return null;
  }

  function renderDeviation(status) {
    var entry = status.run_state === "idle" ? null : newestDeviation();
    if (!entry) {
      els.deviation.hidden = true;
      return;
    }
    var alert = alertFor(status, entry);
    var spoken = alert && alert.speak ? alert.speak : null;
    els.deviationLabel.textContent = spoken || spokenEqualsDetail(entry) ? "Voice alert" : "Deviation";
    els.deviationMsg.textContent = spoken || entry.detail;
    els.deviationTime.textContent = "Video time " + fmtTime(entry.t_video);
    els.deviation.hidden = false;
  }

  function logIcon(entry) {
    var kind = entry.event_type;
    if (kind === "step_confirmed") {
      return icon("check", 18);
    }
    if (kind === "deviation_detected") {
      return icon("warn", 18);
    }
    if (kind === "feed_lost") {
      return icon("camoff", 18);
    }
    return icon("circle", 18);
  }

  function logRow(entry) {
    var bad = entry.event_type === "deviation_detected" || entry.event_type === "feed_lost";
    var row = el("li", bad ? "row is-bad" : "row");
    row.appendChild(el("span", "row-t mono", fmtTime(entry.t_video)));
    var mark = el("span", "row-i");
    mark.appendChild(logIcon(entry));
    row.appendChild(mark);

    var body = el("div", "row-m");
    body.appendChild(el("div", "row-e", entry.detail));
    if (spokenEqualsDetail(entry)) {
      var spoken = el("div", "row-sp");
      spoken.appendChild(icon("speaker-sm", 14));
      spoken.appendChild(el("span", "", "Voice text: “" + entry.detail + "”"));
      body.appendChild(spoken);
    }
    row.appendChild(body);

    if (entry.confidence_tag === "flagged_uncertain") {
      var tag = el("span", "tag");
      tag.appendChild(icon("info", 14));
      tag.appendChild(el("span", "", "Low confidence"));
      row.appendChild(tag);
    }
    return row;
  }

  // Appends only the new rows (so a user's scroll position is untouched); rebuilds on a new run.
  function renderLog() {
    var list = els.eventLog;
    var runKey = model.runId;
    var entries = model.log;
    var stick = list.scrollHeight - list.scrollTop - list.clientHeight < 8;
    if (logRendered.runId !== runKey || entries.length < logRendered.count) {
      clearChildren(list);
      logRendered.count = 0;
      logRendered.runId = runKey;
    }
    if (entries.length === logRendered.count) {
      return;
    }
    for (var i = logRendered.count; i < entries.length; i++) {
      list.appendChild(logRow(entries[i]));
    }
    logRendered.count = entries.length;
    if (stick) {
      list.scrollTop = list.scrollHeight;
    }
  }

  function renderGuidance(status, steps, index) {
    var completed = status.run_state === "completed";
    els.guidance.hidden = completed;
    if (completed) {
      return;
    }
    var confirmed = countStatus(steps, "confirmed");
    els.stepNo.textContent = index >= 0 ? "Step " + (index + 1) + " of " + steps.length : DASH;
    els.cue.textContent = index >= 0 ? stepCue(steps[index].step_id) : DASH;

    clearChildren(els.progress);
    steps.forEach(function (step, i) {
      els.progress.appendChild(el("div", "seg seg-" + (i === index ? "current" : step.status)));
    });
    els.progress.setAttribute("aria-label", confirmed + " of " + steps.length + " steps confirmed");
    els.confirmedA.textContent = confirmed + " of " + steps.length + " confirmed";
  }

  function completionDetail() {
    for (var i = model.log.length - 1; i >= 0; i--) {
      if (model.log[i].event_type === "run_completed") {
        return model.log[i].detail;
      }
    }
    return null;
  }

  function namesOf(steps, wanted) {
    var names = steps
      .filter(function (step) {
        return step.status === wanted;
      })
      .map(function (step) {
        return stepTitle(step.step_id);
      });
    return names.length ? names.join(", ") : "none";
  }

  // Order similarity (RunSummary.pos) is deliberately absent: no route exposes it and computing it
  // here would re-implement the engine. See the CONTRACT entry in ISSUES.md.
  function renderSummary(status, steps) {
    var completed = status.run_state === "completed";
    els.summary.hidden = !completed;
    if (!completed) {
      return;
    }
    var skipped = countStatus(steps, "skipped");
    var late = countStatus(steps, "completed_late");
    var bad = skipped > 0 || late > 0;
    clearChildren(els.summaryIcon);
    els.summaryIcon.className = bad ? "sum-icon is-bad" : "sum-icon";
    els.summaryIcon.appendChild(icon(bad ? "warn" : "check", 28));
    els.summaryHeadline.textContent = completionDetail() || "Run finished";
    els.tileSkipped.textContent = String(skipped);
    els.tileSkipped.className = skipped > 0 ? "tile-n mono is-bad" : "tile-n mono";
    els.tileLate.textContent = String(late);
    els.tileLate.className = late > 0 ? "tile-n mono is-bad" : "tile-n mono";
    els.summarySkipped.textContent = "Skipped: " + namesOf(steps, "skipped");
    els.summaryLate.textContent = "Completed late: " + namesOf(steps, "completed_late");
  }

  function renderSteps(steps, index) {
    var confirmed = countStatus(steps, "confirmed");
    els.confirmedB.textContent = confirmed + " of " + steps.length + " confirmed";
    // Rebuilt only when something shown changed (the poll runs twice a second).
    var signature = JSON.stringify([
      index,
      steps.map(function (step) {
        return [step.step_id, step.status, step.t_confirmed, stepTitle(step.step_id)];
      }),
    ]);
    if (signature === stepsSignature) {
      return;
    }
    stepsSignature = signature;
    clearChildren(els.stepList);
    steps.forEach(function (step, i) {
      var current = i === index;
      var pending = step.status === "pending";
      var row = el("li", "step" + (current ? " is-current" : "") + (pending && !current ? " is-pending" : ""));
      row.appendChild(el("span", "step-n mono", String(i + 1)));
      row.appendChild(el("span", "step-ti", stepTitle(step.step_id)));

      var tone = pending ? " is-pending" : step.status === "confirmed" ? "" : " is-bad";
      var state = el("span", "step-st" + tone);
      state.appendChild(icon(STATUS_ICON[step.status] || "circle", 16));
      state.appendChild(el("span", "", STATUS_LABEL[step.status] || step.status));
      row.appendChild(state);

      row.appendChild(el("span", "step-tm mono", step.t_confirmed != null ? fmtTime(step.t_confirmed) : DASH));
      els.stepList.appendChild(row);
    });
  }

  // ---- video events and autostart ---------------------------------------------------------------

  function onVideoLoaded() {
    imgError = false;
    if (!autostartRequested || autostartScheduled) {
      return;
    }
    autostartScheduled = true;
    window.setTimeout(autostart, AUTOSTART_DELAY_MS);
  }

  function onVideoError() {
    imgError = true;
    render();
  }

  function autostart() {
    // Once per page load, and only when the run is idle right now (a fresh status read, not the
    // last poll): a page reloaded mid-run, or opened by a second viewer, never restarts it.
    if (autostartFired) {
      return;
    }
    autostartFired = true;
    api
      .status()
      .then(function (status) {
        if (status && status.run_state === "idle") {
          postRunControl(api.startRun, els.btnStart);
        }
      })
      .catch(function () {
        // the status poll shows the connection error; the page can still be started by hand
      });
  }

  // ---- polling ----------------------------------------------------------------------------------

  // One status request at a time; the next poll is scheduled POLL_MS after the current one has
  // settled (a response, a failure or the FETCH_TIMEOUT_MS timeout), so a server that hangs can
  // neither pile up requests nor leave the page without an answer. Because requests no longer
  // overlap, a response can never arrive out of order, and the old sequence guard is gone: it was
  // what dropped every response while the server was slow (B1).
  var pollInFlight = false;
  var pollAgain = false;
  var pollTimer = null;

  function schedulePoll(delay) {
    if (pollTimer !== null) {
      clearTimeout(pollTimer);
    }
    pollTimer = setTimeout(poll, delay);
  }

  // A poll wanted now (after Start or Reset): runs at once, or right after the one in flight.
  function pollSoon() {
    if (pollInFlight) {
      pollAgain = true;
      return;
    }
    poll();
  }

  function pollSettled() {
    pollInFlight = false;
    var again = pollAgain;
    pollAgain = false;
    schedulePoll(again ? 0 : POLL_MS);
  }

  function poll() {
    if (pollTimer !== null) {
      clearTimeout(pollTimer);
      pollTimer = null;
    }
    if (pollInFlight) {
      return;
    }
    pollInFlight = true;
    api
      .status()
      .then(onStatus, onStatusError)
      .then(pollSettled, function (err) {
        pollSettled(); // a rendering bug must not stop the polling; it still surfaces in the console
        throw err;
      });
  }

  function onStatus(status) {
    errors.status = null;
    model.unreachable = false;
    if (status.run_id !== model.runId) {
      model.runId = status.run_id;
      model.log = [];
      model.logComplete = false;
    }
    model.status = status;
    showErrors();
    render();
    fetchRunData(status);
  }

  // The last good state stays on screen, except the three live values (State, Feed, Processing),
  // which become dashes: they would otherwise read as current. The page cannot tell a dead server
  // from a lost camera, so it does not claim the camera is lost.
  function onStatusError(err) {
    errors.status = unreachable(err);
    model.unreachable = true;
    showErrors();
    render();
  }

  // The experiment definition once per run; the run's log on each poll while a run exists
  // (running, or completed until its run_completed line has been read).
  function fetchRunData(status) {
    var runId = status.run_id;
    if (!runId) {
      return;
    }
    if (model.experimentRunId !== runId && !inflight.experiment) {
      inflight.experiment = true;
      api.experiment().then(
        function (definition) {
          inflight.experiment = false;
          errors.experiment = null;
          var byId = {};
          definition.steps.forEach(function (step) {
            byId[step.step_id] = { display_name: step.display_name, say: step.say };
          });
          model.experiment = { name: humanize(definition.experiment_id), byId: byId };
          model.experimentRunId = runId;
          showErrors();
          render();
        },
        function (err) {
          inflight.experiment = false;
          errors.experiment = unreachable(err);
          showErrors();
        }
      );
    }
    if (status.run_state !== "idle" && !model.logComplete && !inflight.log) {
      inflight.log = true;
      api.log(runId).then(
        function (body) {
          inflight.log = false;
          errors.log = null;
          if (model.runId !== runId) {
            return; // the run changed while this request was in flight
          }
          model.log = body.entries || [];
          model.logComplete =
            status.run_state === "completed" && completionDetail() !== null;
          showErrors();
          render();
        },
        function (err) {
          inflight.log = false;
          errors.log = unreachable(err);
          showErrors();
        }
      );
    }
  }

  function postRunControl(call, button) {
    // Disabled immediately so a fast double-click can't fire the request twice; render() (via the
    // pollSoon() below) sets the real enabled/disabled state from the server's actual run_state once
    // it responds, on both the success and the "already in that state" paths.
    button.disabled = true;
    call()
      .then(function () {
        pollSoon();
      })
      .catch(function (err) {
        errors.status = unreachable(err);
        showErrors();
        button.disabled = false;
      });
  }

  els.btnStart.addEventListener("click", function () {
    postRunControl(api.startRun, els.btnStart);
  });
  els.btnReset.addEventListener("click", function () {
    postRunControl(api.resetRun, els.btnReset);
  });

  els.video.addEventListener("load", onVideoLoaded);
  els.video.addEventListener("error", onVideoError);

  poll();
})();
