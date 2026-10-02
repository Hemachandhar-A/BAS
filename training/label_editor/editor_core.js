/* Label editor: pure helpers (no DOM, no network). Tested with node. */
(function (root) {
  "use strict";

  var MIN_SIZE = 4;
  var SAME_PX = 0.5;

  function round2(v) { return Math.round(v * 100) / 100; }

  /* Normalise (x1 < x2, y1 < y2), clamp into the image, keep at least MIN_SIZE px. */
  function clampBox(box, w, h) {
    var x1 = Math.min(box[0], box[2]), x2 = Math.max(box[0], box[2]);
    var y1 = Math.min(box[1], box[3]), y2 = Math.max(box[1], box[3]);
    x1 = Math.max(0, Math.min(w, x1)); x2 = Math.max(0, Math.min(w, x2));
    y1 = Math.max(0, Math.min(h, y1)); y2 = Math.max(0, Math.min(h, y2));
    if (x2 - x1 < MIN_SIZE) { if (x1 + MIN_SIZE <= w) x2 = x1 + MIN_SIZE; else x1 = x2 - MIN_SIZE; }
    if (y2 - y1 < MIN_SIZE) { if (y1 + MIN_SIZE <= h) y2 = y1 + MIN_SIZE; else y1 = y2 - MIN_SIZE; }
    return [x1, y1, x2, y2];
  }

  /* Move a box by (dx, dy) keeping its size and staying inside the image. */
  function moveBox(box, dx, dy, w, h) {
    var bw = box[2] - box[0], bh = box[3] - box[1];
    var x1 = Math.max(0, Math.min(w - bw, box[0] + dx));
    var y1 = Math.max(0, Math.min(h - bh, box[1] + dy));
    return [x1, y1, x1 + bw, y1 + bh];
  }

  /* Resize by dragging a handle: n, s, e, w, ne, nw, se, sw. */
  function resizeBox(box, handle, dx, dy, w, h) {
    var b = box.slice();
    if (handle.indexOf("w") >= 0) b[0] += dx;
    if (handle.indexOf("e") >= 0) b[2] += dx;
    if (handle.indexOf("n") >= 0) b[1] += dy;
    if (handle.indexOf("s") >= 0) b[3] += dy;
    return clampBox(b, w, h);
  }

  function area(b) { return Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]); }

  /* Class of the smallest box containing the point, or null. */
  function hitTest(boxes, x, y) {
    var best = null, bestArea = Infinity;
    Object.keys(boxes).forEach(function (cls) {
      var b = boxes[cls];
      if (x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3] && area(b) < bestArea) {
        best = cls; bestArea = area(b);
      }
    });
    return best;
  }

  /* Which handle (if any) of ``box`` is within ``tol`` image px of the point. */
  function handleAt(box, x, y, tol) {
    var near = function (a, v) { return Math.abs(a - v) <= tol; };
    var inX = x >= box[0] - tol && x <= box[2] + tol, inY = y >= box[1] - tol && y <= box[3] + tol;
    if (!inX || !inY) return null;
    var w = near(x, box[0]), e = near(x, box[2]), n = near(y, box[1]), s = near(y, box[3]);
    if (n && w) return "nw"; if (n && e) return "ne"; if (s && w) return "sw"; if (s && e) return "se";
    if (n) return "n"; if (s) return "s"; if (w) return "w"; if (e) return "e";
    return null;
  }

  /* --- frame state: {boxes: {class: xyxy}, excluded, verified, touched} ---------------- */

  function freshState(frame) {
    var boxes = {};
    Object.keys(frame.auto_boxes).forEach(function (c) { boxes[c] = frame.auto_boxes[c].slice(); });
    return { boxes: boxes, excluded: false, verified: false, touched: false };
  }

  function cloneState(s) {
    var boxes = {};
    Object.keys(s.boxes).forEach(function (c) { boxes[c] = s.boxes[c].slice(); });
    return { boxes: boxes, excluded: s.excluded, verified: s.verified, touched: s.touched };
  }

  /* Every mutation returns a new state, marks it touched and drops "verified". */
  function mutated(s, mutate) {
    var n = cloneState(s);
    mutate(n);
    n.touched = true; n.verified = false;
    return n;
  }

  function missingClasses(classes, state) {
    return classes.filter(function (c) { return !state.boxes[c]; });
  }

  function isComplete(classes, state) { return missingClasses(classes, state).length === 0; }

  /* Accept a cached candidate as the box of ``cls``. */
  function acceptProposal(state, cls, proposal, w, h) {
    return mutated(state, function (n) { n.boxes[cls] = clampBox(proposal.box, w, h); });
  }

  function drawBox(state, cls, box, w, h) {
    return mutated(state, function (n) { n.boxes[cls] = clampBox(box, w, h); });
  }

  function setBox(state, cls, box, w, h) { return drawBox(state, cls, box, w, h); }

  function deleteBox(state, cls) {
    return mutated(state, function (n) { delete n.boxes[cls]; });
  }

  /* Give the box of ``from`` the class ``to``; a box already there takes ``from`` (swap). */
  function setClass(state, from, to) {
    if (from === to || !state.boxes[from]) return state;
    return mutated(state, function (n) {
      var moving = n.boxes[from], other = n.boxes[to];
      delete n.boxes[from];
      n.boxes[to] = moving;
      if (other) n.boxes[from] = other;
    });
  }

  function setExcluded(state, value) {
    return mutated(state, function (n) { n.excluded = value; });
  }

  /* Verify a complete, not-excluded frame as it is now (the only mutation that keeps verified). */
  function setVerified(classes, state) {
    if (state.excluded || !isComplete(classes, state)) return null;
    var n = cloneState(state);
    n.verified = true; n.touched = true;
    return n;
  }

  function frameStatus(classes, state, kind) {
    if (state.excluded) return "excluded";
    if (state.verified) return "verified";
    if (!state.touched) return "todo";
    return isComplete(classes, state) ? "edited" : "incomplete";
  }

  /* digit key -> class index (keys 1..n) or -1; digit key -> proposal index (1..9, 0 = tenth) */
  function classIndexForKey(key, n) {
    var d = parseInt(key, 10);
    return (key.length === 1 && d >= 1 && d <= n) ? d - 1 : -1;
  }
  function proposalIndexForKey(key) {
    if (key.length !== 1 || key < "0" || key > "9") return -1;
    return key === "0" ? 9 : parseInt(key, 10) - 1;
  }

  function sameBox(a, b) {
    for (var i = 0; i < 4; i++) if (Math.abs(a[i] - b[i]) > SAME_PX) return false;
    return true;
  }

  /* --- the overlay (Plan 5.8) ---------------------------------------------------------- */

  /* data = EDITOR_DATA; states: {frameKey: state}; foreign: {frames:{file: entry}, static:{run:{cls:xyxy}}}.
     Returns {overlay, warnings}. A frame entry carries ALL class boxes (it replaces the frame
     wholesale); an incomplete, not-excluded frame is NOT written (a missing class would train as
     a negative) and is listed in warnings. */
  function assembleOverlay(data, states, foreign) {
    foreign = foreign || { frames: {}, static: {} };
    var classes = data.classes, warnings = [];
    var frames = {}, statics = {};
    Object.keys(foreign.frames || {}).forEach(function (f) { frames[f] = foreign.frames[f]; });
    Object.keys(foreign.static || {}).forEach(function (r) {
      statics[r] = {}; Object.keys(foreign.static[r]).forEach(function (c) { statics[r][c] = foreign.static[r][c]; });
    });
    data.frames.forEach(function (fr) {
      var st = states[fr.key];
      if (!st || !st.touched) return;
      if (fr.kind === "static") {
        classes.forEach(function (c) {
          if (data.static_classes.indexOf(c) < 0 || !st.boxes[c]) return;
          var auto = fr.auto_boxes[c];
          if (!auto || !sameBox(auto, st.boxes[c])) {
            statics[fr.run] = statics[fr.run] || {};
            statics[fr.run][c] = st.boxes[c].map(round2);
          }
        });
        return;
      }
      if (st.excluded) { frames[fr.file] = { excluded: true, verified: false, boxes: [] }; return; }
      var miss = missingClasses(classes, st);
      if (miss.length) {
        delete frames[fr.file];
        warnings.push(fr.run + "#" + fr.frame + " not written: no box for " + miss.join(", "));
        return;
      }
      frames[fr.file] = {
        excluded: false,
        verified: !!st.verified,
        boxes: classes.map(function (c) { return { "class": c, xyxy: st.boxes[c].map(round2) }; })
      };
    });
    /* a frame entry replaces wholesale, so it must also carry its run's static fix unless the
       user changed that static box on that very frame */
    data.frames.forEach(function (fr) {
      if (fr.kind === "static") return;
      var entry = frames[fr.file], over = statics[fr.run];
      if (!entry || entry.excluded || !over) return;
      entry.boxes.forEach(function (b) {
        var auto = fr.auto_boxes[b["class"]];
        if (over[b["class"]] && auto && sameBox(auto, b.xyxy)) b.xyxy = over[b["class"]].slice();
      });
    });
    var sortedFrames = {};
    Object.keys(frames).sort().forEach(function (f) { sortedFrames[f] = frames[f]; });
    var sortedStatic = {};
    Object.keys(statics).sort().forEach(function (r) { sortedStatic[r] = statics[r]; });
    return {
      overlay: { version: 1, split: data.split, static_overrides: sortedStatic, frames: sortedFrames },
      warnings: warnings
    };
  }

  /* Read a previous overlay back into editor state. Returns {states, foreign, problems, applied};
     entries for frames not on this page are kept in ``foreign`` and written out again. */
  function applyOverlay(data, overlay, states) {
    var problems = [], applied = 0;
    var foreign = { frames: {}, static: {} };
    if (!overlay || overlay.version !== 1) { problems.push("not a version 1 overlay"); return { states: states, foreign: foreign, problems: problems, applied: 0 }; }
    if (overlay.split !== data.split) { problems.push("overlay is for split " + overlay.split + ", this page is " + data.split); return { states: states, foreign: foreign, problems: problems, applied: 0 }; }
    var out = {}; Object.keys(states).forEach(function (k) { out[k] = states[k]; });
    var byFile = {}, staticByRun = {};
    data.frames.forEach(function (fr) {
      if (fr.kind === "static") staticByRun[fr.run] = fr; else byFile[fr.file] = fr;
    });
    Object.keys(overlay.frames || {}).forEach(function (file) {
      var e = overlay.frames[file], fr = byFile[file];
      var bad = (e.boxes || []).filter(function (b) { return data.classes.indexOf(b["class"]) < 0; });
      if (bad.length) { problems.push(file + ": unknown class " + bad[0]["class"]); return; }
      if (!fr) { foreign.frames[file] = e; return; }
      var boxes = {};
      (e.boxes || []).forEach(function (b) { boxes[b["class"]] = b.xyxy.slice(); });
      out[fr.key] = { boxes: boxes, excluded: !!e.excluded, verified: !!e.verified, touched: true };
      applied++;
    });
    Object.keys(overlay.static_overrides || {}).forEach(function (run) {
      var per = overlay.static_overrides[run];
      var bad = Object.keys(per).filter(function (c) { return data.classes.indexOf(c) < 0; });
      if (bad.length) { problems.push("static override " + run + ": unknown class " + bad[0]); return; }
      var fr = staticByRun[run];
      if (!fr) { foreign.static[run] = per; return; }
      var st = out[fr.key] ? cloneState(out[fr.key]) : freshState(fr);
      Object.keys(per).forEach(function (c) { st.boxes[c] = per[c].slice(); });
      st.touched = true;
      out[fr.key] = st;
      applied++;
    });
    return { states: out, foreign: foreign, problems: problems, applied: applied };
  }

  var api = {
    clampBox: clampBox, moveBox: moveBox, resizeBox: resizeBox, hitTest: hitTest, handleAt: handleAt,
    freshState: freshState, cloneState: cloneState, missingClasses: missingClasses,
    isComplete: isComplete, acceptProposal: acceptProposal, drawBox: drawBox, setBox: setBox,
    deleteBox: deleteBox, setClass: setClass, setExcluded: setExcluded, setVerified: setVerified,
    frameStatus: frameStatus, classIndexForKey: classIndexForKey,
    proposalIndexForKey: proposalIndexForKey, sameBox: sameBox,
    assembleOverlay: assembleOverlay, applyOverlay: applyOverlay
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.LE = api;
})(typeof window !== "undefined" ? window : this);
