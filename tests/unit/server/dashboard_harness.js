// Node harness for server/static/app.js: a stub DOM built from the ids in the real index.html, a
// scripted fetch, a fake clock and timers. Not a browser (no layout, no CSS): it renders the page
// logic from stub API payloads and dumps what the page wrote.
//
//   node dashboard_harness.js <app.js> <index.html> <scenario.json>
//
// Scenario: {"search": "?autostart=1", "natural_width": 0, "now": 100000,
//            "steps": [{"set": {...world fields...}, "now": ms, "events": ["video:load"],
//                       "scroll": {"id": "event-log", "top": 0}, "click": "btn-start",
//                       "fire_timers": true, "poll": true}, ...]}
// World fields: status, experiment, log (an object, null = 404, "error" = 500), fail_status,
// fail_experiment, post_status (HTTP status for POSTs), hang_status / hang_experiment / hang_log
// (the request never settles unless aborted), status_delay_ms (virtual response time).
// A step's "advance": ms runs that much virtual time (default 500, "poll": false = none). The initial poll happens at load; each
// step then optionally mutates the world, acts, polls (default) and snapshots. Output: JSON
// {snapshots: [...], fetches: [[path, ...] per step], posts: [...], timers: [ms ...], intervals}.
"use strict";
const fs = require("fs");

const [appPath, indexPath, scenarioPath] = process.argv.slice(2);
const scenario = JSON.parse(fs.readFileSync(scenarioPath, "utf8"));
const html = fs.readFileSync(indexPath, "utf8");

const SVG_NS = "ns:svg";

class El {
  constructor(tag, id) {
    this.tag = tag;
    this.id = id || null;
    this.children = [];
    this.attrs = {};
    this.className = "";
    this.hidden = false;
    this.disabled = false;
    this._text = "";
    this.listeners = {};
    this.scrollTop = 0;
    this.clientHeight = 300;
    this.naturalWidth = 0;
    this.namespaceURI = null;
    this.src = "";
  }
  get firstChild() {
    return this.children[0] || null;
  }
  get scrollHeight() {
    return this.children.length * 60;
  }
  get textContent() {
    return this.children.length ? this.children.map((c) => c.textContent).join("") : this._text;
  }
  set textContent(value) {
    this.children = [];
    this._text = String(value);
  }
  get innerHTML() {
    throw new Error("innerHTML is not allowed");
  }
  set innerHTML(_value) {
    throw new Error("innerHTML is not allowed");
  }
  get outerHTML() {
    throw new Error("outerHTML is not allowed");
  }
  set outerHTML(_value) {
    throw new Error("outerHTML is not allowed");
  }
  insertAdjacentHTML() {
    throw new Error("insertAdjacentHTML is not allowed");
  }
  appendChild(child) {
    this.children.push(child);
    return child;
  }
  removeChild(child) {
    this.children = this.children.filter((c) => c !== child);
    return child;
  }
  setAttribute(name, value) {
    this.attrs[name] = String(value);
  }
  getAttribute(name) {
    return name in this.attrs ? this.attrs[name] : null;
  }
  addEventListener(name, fn) {
    this.listeners[name] = fn;
  }
}

const byId = {};
for (const match of html.matchAll(/<[^>]*\sid="([^"]+)"[^>]*>/g)) {
  byId[match[1]] = new El("x", match[1]);
  // the markup's own `hidden` attribute is the element's state until the page changes it
  byId[match[1]].hidden = /\shidden(\s|>|=)/.test(match[0]);
}
byId["icon-sprite"].namespaceURI = SVG_NS;
byId["video"].naturalWidth = scenario.natural_width || 0;
byId["video"].src = "/video_feed";
byId["btn-start"].disabled = false;

global.document = {
  getElementById(id) {
    return byId[id] || null;
  },
  createElement(tag) {
    return new El(tag);
  },
  createElementNS(ns, tag) {
    const node = new El(tag);
    node.namespaceURI = ns;
    return node;
  },
};

// Two kinds of timer. window.setTimeout is the page's autostart timer: recorded in `timers` /
// `delays` and fired only by a step's fire_timers. The global setTimeout / setInterval (poll
// scheduling, request timeouts) run on a VIRTUAL clock: a step advances it, so a 2 s timeout or a
// 21 s hang costs no real time. Date.now() is the same virtual clock.
const timers = [];
const delays = [];
const intervals = []; // ms of every setInterval the page registered
const pollDelays = []; // ms of every global setTimeout the page registered
const sched = []; // {fn, at, every}
const world = { now: scenario.now || 100000, post_status: 200 };
let stepFetches = [];
const posts = [];
let inflightStatus = 0;
let maxInflightStatus = 0;
let schedSeq = 0;
Date.now = () => world.now;

global.window = {
  location: { search: scenario.search || "" },
  setTimeout(fn, ms) {
    timers.push({ fn, ms });
    delays.push(ms);
  },
};
global.setTimeout = (fn, ms) => {
  pollDelays.push(ms);
  const id = ++schedSeq;
  sched.push({ id, fn, at: world.now + (ms || 0), every: 0 });
  return id;
};
global.clearTimeout = (id) => {
  const i = sched.findIndex((t) => t.id === id);
  if (i >= 0) sched.splice(i, 1);
};
global.setInterval = (fn, ms) => {
  intervals.push(ms);
  const id = ++schedSeq;
  sched.push({ id, fn, at: world.now + ms, every: ms });
  return id;
};
global.clearInterval = global.clearTimeout;

const settle = async () => {
  for (let i = 0; i < 6; i++) await new Promise((r) => setImmediate(r));
};

// Runs every virtual timer due within the next `ms`, in order, settling promises after each.
async function advance(ms) {
  const target = world.now + ms;
  for (;;) {
    const due = sched.filter((t) => t.at <= target).sort((a, b) => a.at - b.at || a.id - b.id)[0];
    if (!due) break;
    world.now = Math.max(world.now, due.at);
    if (due.every) due.at += due.every;
    else sched.splice(sched.indexOf(due), 1);
    due.fn();
    await settle();
  }
  world.now = Math.max(world.now, target);
}

function reply(status, body) {
  return Promise.resolve({ ok: status >= 200 && status < 300, status, json: () => Promise.resolve(body) });
}

// A response that arrives after `ms` of virtual time (and is dropped if the request is aborted).
function later(ms, make, signal) {
  return new Promise((resolve, reject) => {
    const abort = () => reject(Object.assign(new Error("aborted"), { name: "AbortError" }));
    if (signal && signal.aborted) return abort();
    if (signal) signal.addEventListener("abort", abort);
    if (ms === Infinity) return; // never settles on its own (a connect to a dead port)
    const id = ++schedSeq;
    sched.push({ id, fn: () => resolve(make()), at: world.now + ms, every: 0 });
  });
}

global.fetch = (url, opts) => {
  const path = String(url).split("?")[0];
  stepFetches.push(String(url));
  if (opts && opts.method === "POST") {
    posts.push(path);
    return reply(world.post_status, { ok: true, run_id: "r", run_state: "running" });
  }
  if (path === "/api/status") {
    let pending;
    if (world.hang_status) {
      pending = later(Infinity, null, opts && opts.signal);
    } else if (world.status_delay_ms) {
      pending = later(world.status_delay_ms, () => ({ ok: true, status: 200, json: () => Promise.resolve(world.status) }), opts && opts.signal);
    } else {
      pending = world.fail_status ? reply(500, {}) : reply(200, world.status);
    }
    inflightStatus += 1;
    maxInflightStatus = Math.max(maxInflightStatus, inflightStatus);
    const done = () => {
      inflightStatus -= 1;
    };
    pending.then(done, done);
    return pending;
  }
  if (path === "/api/experiment") {
    if (world.hang_experiment) return later(Infinity, null, opts && opts.signal);
    return world.fail_experiment ? reply(500, {}) : reply(200, world.experiment);
  }
  if (path === "/api/log") {
    if (world.hang_log) return later(Infinity, null, opts && opts.signal);
    if (world.log === null || world.log === undefined) return reply(404, { error: "no log" });
    if (world.log === "error") return reply(500, {});
    return reply(200, world.log);
  }
  return reply(404, {});
};

function dump(node) {
  return {
    tag: node.tag,
    cls: node.className,
    text: node.children.length ? undefined : node._text,
    hidden: node.hidden,
    disabled: node.disabled,
    attrs: node.attrs,
    scrollTop: node.scrollTop,
    children: node.children.map(dump),
    full: node.textContent,
  };
}

function snapshot() {
  const out = {};
  for (const id of Object.keys(byId)) {
    out[id] = dump(byId[id]);
  }
  out["video"].src = byId["video"].src;
  return out;
}

(async () => {
  Object.assign(world, scenario.world || {});
  const appSource = fs.readFileSync(appPath, "utf8");
  eval(appSource);
  await settle();
  const snapshots = [];
  const fetches = [];
  const maxPerStep = [];
  fetches.push(stepFetches);
  snapshots.push(snapshot());
  for (const step of scenario.steps || []) {
    stepFetches = [];
    Object.assign(world, step.set || {});
    if (step.now !== undefined) world.now = step.now;
    if (step.scroll) byId[step.scroll.id].scrollTop = step.scroll.top;
    for (const ev of step.events || []) {
      const [id, name] = ev.split(":");
      if (id === "video" && name === "load") byId["video"].naturalWidth = 640;
      if (byId[id].listeners[name]) byId[id].listeners[name]();
    }
    if (step.click) byId[step.click].listeners.click();
    if (step.fire_timers) {
      for (const t of timers.splice(0)) t.fn();
    }
    // Default: one poll period of virtual time (what the old interval-based page needed to poll
    // once); "advance" runs any other span, "poll": false none at all.
    if (step.advance !== undefined) await advance(step.advance);
    else if (step.poll !== false) await advance(500);
    await settle();
    fetches.push(stepFetches);
    snapshots.push(snapshot());
    maxPerStep.push(maxInflightStatus);
  }
  console.log(
    JSON.stringify({
      snapshots,
      fetches,
      posts,
      delays,
      intervals,
      poll_delays: pollDelays,
      max_inflight_status: maxInflightStatus,
      max_per_step: maxPerStep,
    })
  );
})().catch((err) => {
  console.error(err && err.stack ? err.stack : String(err));
  process.exit(1);
});
