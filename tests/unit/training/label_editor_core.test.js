/* Unit tests for training/label_editor/editor_core.js (run by test_label_editor_js.py with node). */
"use strict";
const assert = require("node:assert/strict");
const path = require("node:path");
const LE = require(path.join(__dirname, "..", "..", "..", "training", "label_editor", "editor_core.js"));

const CLASSES = ["outer_box", "tray", "red_box", "yellow_box", "start_button"];
const STATIC = ["outer_box", "tray", "start_button"];
const W = 800, H = 400;
const AUTO = {
  outer_box: [10, 10, 400, 390], tray: [500, 20, 700, 200], yellow_box: [100, 200, 180, 260],
  start_button: [520, 300, 640, 380],
};
const FULL = Object.assign({ red_box: [100, 100, 180, 160] }, AUTO);

function frame(key, kind, run, fid, boxes, proposals) {
  const miss = CLASSES.filter((c) => !boxes[c]);
  return { key: key, kind: kind, run: run, frame: fid, file: run + "_" + fid + ".jpg", width: W, height: H,
    auto_boxes: boxes, missing: miss, proposals: proposals || {} };
}
const DATA = {
  version: 1, split: "train", classes: CLASSES, static_classes: STATIC,
  frames: [
    frame("excluded|a_15.jpg", "excluded", "a", 15, AUTO, { red_box: [{ n: 1, box: [100, 100, 130, 125], score: 0.5 }, { n: 2, box: [90, 90, 170, 150], score: 0.3 }] }),
    frame("gold|a_30.jpg", "gold", "a", 30, FULL),
    frame("static|a_0.jpg", "static", "a", 0, { outer_box: AUTO.outer_box, tray: AUTO.tray, start_button: AUTO.start_button }),
  ],
};

let n = 0;
function test(name, fn) { fn(); n++; }

test("clampBox normalises, clamps to the image and keeps a minimum size", () => {
  assert.deepEqual(LE.clampBox([50, 60, 10, 20], W, H), [10, 20, 50, 60]);
  assert.deepEqual(LE.clampBox([-5, -5, 900, 500], W, H), [0, 0, W, H]);
  const t = LE.clampBox([100, 100, 101, 100], W, H);
  assert.ok(t[2] - t[0] >= 4 && t[3] - t[1] >= 4);
  const edge = LE.clampBox([799, 399, 800, 400], W, H);
  assert.ok(edge[0] >= 0 && edge[2] <= W && edge[2] - edge[0] >= 4);
});

test("moveBox keeps the size and stays inside the image", () => {
  assert.deepEqual(LE.moveBox([10, 10, 60, 40], 5, 7, W, H), [15, 17, 65, 47]);
  assert.deepEqual(LE.moveBox([10, 10, 60, 40], -50, -50, W, H), [0, 0, 50, 30]);
  assert.deepEqual(LE.moveBox([10, 10, 60, 40], 5000, 5000, W, H), [750, 370, 800, 400]);
});

test("resizeBox moves only the dragged edges", () => {
  assert.deepEqual(LE.resizeBox([100, 100, 200, 200], "se", 10, 20, W, H), [100, 100, 210, 220]);
  assert.deepEqual(LE.resizeBox([100, 100, 200, 200], "w", -30, 99, W, H), [70, 100, 200, 200]);
  const flipped = LE.resizeBox([100, 100, 200, 200], "e", -150, 0, W, H); // dragged past the other edge
  assert.ok(flipped[0] < flipped[2]);
});

test("hitTest picks the smallest box under the point", () => {
  assert.equal(LE.hitTest(FULL, 120, 120), "red_box"); // inside outer_box too
  assert.equal(LE.hitTest(FULL, 50, 300), "outer_box");
  assert.equal(LE.hitTest(FULL, 790, 5), null);
});

test("handleAt finds corners and edges", () => {
  assert.equal(LE.handleAt([100, 100, 200, 200], 101, 99, 6), "nw");
  assert.equal(LE.handleAt([100, 100, 200, 200], 200, 150, 6), "e");
  assert.equal(LE.handleAt([100, 100, 200, 200], 150, 150, 6), null);
});

test("class keys: 1..n select classes in order, others are not classes", () => {
  assert.equal(LE.classIndexForKey("1", 5), 0);
  assert.equal(LE.classIndexForKey("5", 5), 4);
  assert.equal(LE.classIndexForKey("6", 5), -1);
  assert.equal(LE.classIndexForKey("0", 5), -1);
  assert.equal(LE.classIndexForKey("a", 5), -1);
  assert.equal(LE.proposalIndexForKey("1"), 0);
  assert.equal(LE.proposalIndexForKey("0"), 9);
  assert.equal(LE.proposalIndexForKey("x"), -1);
});

test("acceptProposal sets the missing class box, marks touched, drops verified", () => {
  const s0 = LE.freshState(DATA.frames[0]);
  assert.deepEqual(LE.missingClasses(CLASSES, s0), ["red_box"]);
  const s1 = LE.acceptProposal(s0, "red_box", DATA.frames[0].proposals.red_box[1], W, H);
  assert.deepEqual(s1.boxes.red_box, [90, 90, 170, 150]);
  assert.equal(s1.touched, true);
  assert.ok(LE.isComplete(CLASSES, s1));
  assert.equal(s0.boxes.red_box, undefined); // the old state is untouched (undo relies on it)
  const v = LE.setVerified(CLASSES, s1);
  assert.equal(v.verified, true);
  const s2 = LE.deleteBox(v, "yellow_box");
  assert.equal(s2.verified, false);
  assert.equal(LE.setVerified(CLASSES, s2), null); // incomplete cannot be verified
  assert.equal(LE.setVerified(CLASSES, LE.setExcluded(s1, true)), null);
});

test("acceptProposal clamps a box that leaves the image", () => {
  const s = LE.acceptProposal(LE.freshState(DATA.frames[0]), "red_box", { box: [-10, -10, 50, 900] }, W, H);
  assert.deepEqual(s.boxes.red_box, [0, 0, 50, H]);
});

test("setClass moves a box to another class and swaps when the class is taken", () => {
  const s0 = LE.freshState(DATA.frames[1]);
  const s1 = LE.setClass(s0, "red_box", "yellow_box");
  assert.deepEqual(s1.boxes.yellow_box, FULL.red_box);
  assert.deepEqual(s1.boxes.red_box, FULL.yellow_box); // swapped, nothing lost
  assert.equal(Object.keys(s1.boxes).length, 5);
  const s2 = LE.setClass(LE.freshState(DATA.frames[0]), "yellow_box", "red_box"); // red was missing
  assert.deepEqual(s2.boxes.red_box, AUTO.yellow_box);
  assert.equal(s2.boxes.yellow_box, undefined);
  assert.equal(LE.setClass(s0, "red_box", "red_box"), s0);
});

test("frameStatus", () => {
  const f = DATA.frames[0];
  const s0 = LE.freshState(f);
  assert.equal(LE.frameStatus(CLASSES, s0), "todo");
  assert.equal(LE.frameStatus(CLASSES, LE.deleteBox(s0, "tray")), "incomplete");
  const done = LE.acceptProposal(s0, "red_box", f.proposals.red_box[0], W, H);
  assert.equal(LE.frameStatus(CLASSES, done), "edited");
  assert.equal(LE.frameStatus(CLASSES, LE.setVerified(CLASSES, done)), "verified");
  assert.equal(LE.frameStatus(CLASSES, LE.setExcluded(s0, true)), "excluded");
});

test("assembleOverlay: an accepted frame carries ALL five classes (auto plus accepted), in class order", () => {
  const states = {};
  states[DATA.frames[0].key] = LE.setVerified(CLASSES, LE.acceptProposal(LE.freshState(DATA.frames[0]), "red_box", DATA.frames[0].proposals.red_box[0], W, H));
  const { overlay, warnings } = LE.assembleOverlay(DATA, states);
  assert.deepEqual(warnings, []);
  assert.equal(overlay.version, 1);
  assert.equal(overlay.split, "train");
  assert.deepEqual(Object.keys(overlay.frames), ["a_15.jpg"]);
  const e = overlay.frames["a_15.jpg"];
  assert.equal(e.excluded, false);
  assert.equal(e.verified, true);
  assert.deepEqual(e.boxes.map((b) => b["class"]), CLASSES);
  assert.deepEqual(e.boxes[2].xyxy, [100, 100, 130, 125]);
  assert.deepEqual(e.boxes[0].xyxy, AUTO.outer_box); // the auto box is carried unchanged
  assert.deepEqual(overlay.static_overrides, {});
});

test("assembleOverlay: untouched frames are not written; excluded frames are; incomplete ones are refused", () => {
  const states = {};
  states[DATA.frames[0].key] = LE.deleteBox(LE.freshState(DATA.frames[0]), "tray"); // still lacks red and tray
  states[DATA.frames[1].key] = LE.setExcluded(LE.freshState(DATA.frames[1]), true);
  const { overlay, warnings } = LE.assembleOverlay(DATA, states);
  assert.deepEqual(overlay.frames, { "a_30.jpg": { excluded: true, verified: false, boxes: [] } });
  assert.equal(warnings.length, 1);
  assert.match(warnings[0], /a#15/);
  assert.match(warnings[0], /red_box/);
  assert.match(warnings[0], /tray/);
});

test("assembleOverlay: a confirmed unchanged gold frame still gets an entry (verified)", () => {
  const states = {};
  states[DATA.frames[1].key] = LE.setVerified(CLASSES, LE.freshState(DATA.frames[1]));
  const { overlay } = LE.assembleOverlay(DATA, states);
  assert.equal(overlay.frames["a_30.jpg"].verified, true);
  assert.equal(overlay.frames["a_30.jpg"].boxes.length, 5);
});

test("assembleOverlay: a static edit becomes a static override only for the changed class", () => {
  const sf = DATA.frames[2];
  let st = LE.freshState(sf);
  st = LE.setBox(st, "tray", [505, 25, 705, 205], W, H);
  const states = {}; states[sf.key] = st;
  const { overlay } = LE.assembleOverlay(DATA, states);
  assert.deepEqual(overlay.static_overrides, { a: { tray: [505, 25, 705, 205] } });
  assert.deepEqual(overlay.frames, {});
  // a static view that was only looked at (not changed) writes nothing
  const same = LE.setVerified(CLASSES.filter((c) => STATIC.includes(c)), LE.freshState(sf));
  const { overlay: o2 } = LE.assembleOverlay(DATA, { [sf.key]: same });
  assert.deepEqual(o2.static_overrides, {});
});

test("assembleOverlay: a frame entry of a run with a static fix carries the fix unless that frame changed the box", () => {
  const sf = DATA.frames[2];
  const states = {};
  states[sf.key] = LE.setBox(LE.freshState(sf), "tray", [505, 25, 705, 205], W, H);
  states[DATA.frames[1].key] = LE.setVerified(CLASSES, LE.freshState(DATA.frames[1])); // a#30, unchanged
  const { overlay } = LE.assembleOverlay(DATA, states);
  const tray = overlay.frames["a_30.jpg"].boxes.find((b) => b["class"] === "tray");
  assert.deepEqual(tray.xyxy, [505, 25, 705, 205]);
  // if the user moved the tray on that frame itself, their own box wins
  const s2 = {}; s2[sf.key] = states[sf.key];
  s2[DATA.frames[1].key] = LE.setBox(LE.freshState(DATA.frames[1]), "tray", [510, 30, 710, 210], W, H);
  const o2 = LE.assembleOverlay(DATA, s2).overlay;
  assert.deepEqual(o2.frames["a_30.jpg"].boxes.find((b) => b["class"] === "tray").xyxy, [510, 30, 710, 210]);
});

test("assembleOverlay: a frame entry replaces wholesale (a deleted-then-redrawn box is the new one)", () => {
  const f = DATA.frames[1];
  let s = LE.deleteBox(LE.freshState(f), "red_box");
  s = LE.drawBox(s, "red_box", [300, 100, 380, 160], W, H);
  const e = LE.assembleOverlay(DATA, { [f.key]: s }).overlay.frames["a_30.jpg"];
  assert.deepEqual(e.boxes.find((b) => b["class"] === "red_box").xyxy, [300, 100, 380, 160]);
  assert.equal(e.boxes.length, 5);
});

test("applyOverlay round trip: assemble -> apply gives the same overlay again", () => {
  const states = {};
  states[DATA.frames[0].key] = LE.setVerified(CLASSES, LE.acceptProposal(LE.freshState(DATA.frames[0]), "red_box", DATA.frames[0].proposals.red_box[0], W, H));
  states[DATA.frames[1].key] = LE.setExcluded(LE.freshState(DATA.frames[1]), true);
  states[DATA.frames[2].key] = LE.setBox(LE.freshState(DATA.frames[2]), "tray", [505, 25, 705, 205], W, H);
  const first = LE.assembleOverlay(DATA, states).overlay;
  const back = LE.applyOverlay(DATA, JSON.parse(JSON.stringify(first)), {});
  assert.deepEqual(back.problems, []);
  assert.equal(back.applied, 3);
  const again = LE.assembleOverlay(DATA, back.states, back.foreign).overlay;
  assert.deepEqual(again, first);
});

test("applyOverlay keeps entries for frames not on this page and writes them out again", () => {
  const overlay = { version: 1, split: "train", static_overrides: { z: { tray: [1, 2, 30, 40] } },
    frames: { "z_5.jpg": { excluded: true, verified: false, boxes: [] } } };
  const back = LE.applyOverlay(DATA, overlay, {});
  assert.deepEqual(Object.keys(back.foreign.frames), ["z_5.jpg"]);
  assert.deepEqual(back.foreign.static, { z: { tray: [1, 2, 30, 40] } });
  const out = LE.assembleOverlay(DATA, back.states, back.foreign).overlay;
  assert.deepEqual(out.frames, overlay.frames);
  assert.deepEqual(out.static_overrides, overlay.static_overrides);
});

test("applyOverlay refuses a wrong split, a wrong version and unknown classes", () => {
  assert.match(LE.applyOverlay(DATA, { version: 1, split: "val", frames: {} }, {}).problems[0], /split/);
  assert.match(LE.applyOverlay(DATA, { version: 2, split: "train" }, {}).problems[0], /version/);
  const bad = { version: 1, split: "train", frames: { "a_15.jpg": { excluded: false, verified: false, boxes: [{ "class": "blue_box", xyxy: [1, 2, 3, 4] }] } } };
  const r = LE.applyOverlay(DATA, bad, {});
  assert.match(r.problems[0], /unknown class blue_box/);
  assert.equal(Object.keys(r.states).length, 0);
});

console.log("ok " + n);
