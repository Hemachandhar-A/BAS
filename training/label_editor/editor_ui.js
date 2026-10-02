/* Label editor: the page (DOM, canvas, keys, autosave). Offline: no fetch, no network. */
(function () {
  "use strict";
  var D = window.EDITOR_DATA, LE = window.LE;
  var STORE_KEY = "label_editor_v1_" + D.split;
  var $ = function (id) { return document.getElementById(id); };

  var idx = 0;
  var states = {};
  var foreign = { frames: {}, static: {} };
  var undoStack = [];
  var selected = null;     // class of the selected box
  var active = null;       // missing class the number keys act on
  var drag = null;         // {mode, ...}
  var hoverProp = null;
  var imgs = {};
  var scale = 1;
  var cv = $("cv"), ctx = cv.getContext("2d");

  function frame() { return D.frames[idx]; }
  function classesOf(fr) { return fr.kind === "static" ? D.static_classes : D.classes; }
  function stateOf(fr) {
    if (!states[fr.key]) states[fr.key] = LE.freshState(fr);
    return states[fr.key];
  }
  function statusOf(fr) { return LE.frameStatus(classesOf(fr), stateOf(fr)); }

  /* --- autosave -------------------------------------------------------------------- */
  function save() {
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify({ states: states, foreign: foreign, at: Date.now() }));
    } catch (e) { /* private window or blocked storage: the download still works */ }
  }
  function restore() {
    try {
      var raw = localStorage.getItem(STORE_KEY);
      if (!raw) return 0;
      var s = JSON.parse(raw), n = 0, keys = {};
      D.frames.forEach(function (f) { keys[f.key] = true; });
      Object.keys(s.states || {}).forEach(function (k) { if (keys[k]) { states[k] = s.states[k]; n++; } });
      foreign = s.foreign || foreign;
      return n;
    } catch (e) { return 0; }
  }

  function msg(text) { $("msg").textContent = text || ""; }

  /* --- mutation with undo ---------------------------------------------------------------- */
  function apply(newState) {
    var fr = frame();
    undoStack.push({ key: fr.key, state: LE.cloneState(stateOf(fr)), sel: selected });
    if (undoStack.length > 300) undoStack.shift();
    states[fr.key] = newState;
    save();
    refresh();
  }
  function undo() {
    var u = undoStack.pop();
    if (!u) { msg("nothing to undo"); return; }
    states[u.key] = u.state;
    var i = D.frames.findIndex(function (f) { return f.key === u.key; });
    if (i >= 0) idx = i;
    selected = u.sel && u.state.boxes[u.sel] ? u.sel : null;
    save(); refresh(); msg("undone");
  }

  /* --- navigation ----------------------------------------------------------------------- */
  function go(i) {
    if (i < 0 || i >= D.frames.length) return;
    idx = i; selected = null; active = null; hoverProp = null; drag = null;
    refresh();
    [i + 1, i + 2].forEach(function (j) { if (D.frames[j]) loadImg(D.frames[j]); });
  }
  function nextTodo() {
    for (var k = 1; k <= D.frames.length; k++) {
      var j = (idx + k) % D.frames.length, st = statusOf(D.frames[j]);
      if (st === "todo" || st === "incomplete" || st === "edited") { go(j); return; }
    }
    msg("every frame is verified or excluded");
  }

  /* --- images ---------------------------------------------------------------------------- */
  function loadImg(fr) {
    var e = imgs[fr.img];
    if (e) return e;
    e = imgs[fr.img] = { el: new Image(), ok: false, err: false };
    e.el.onload = function () { e.ok = true; if (frame().img === fr.img) refresh(); };
    e.el.onerror = function () { e.err = true; if (frame().img === fr.img) refresh(); };
    e.el.src = fr.img;
    return e;
  }

  /* --- drawing --------------------------------------------------------------------------- */
  function fit() {
    var fr = frame(), stage = $("stage");
    var aw = stage.clientWidth, ah = stage.clientHeight;
    scale = Math.max(0.1, Math.min(aw / fr.width, ah / fr.height));
    var dpr = window.devicePixelRatio || 1;
    cv.style.width = Math.floor(fr.width * scale) + "px";
    cv.style.height = Math.floor(fr.height * scale) + "px";
    cv.width = Math.floor(fr.width * scale * dpr);
    cv.height = Math.floor(fr.height * scale * dpr);
    ctx.setTransform(dpr * scale, 0, 0, dpr * scale, 0, 0);   // draw in image pixels
  }

  function label(text, x, y, color, size) {
    ctx.font = "bold " + Math.round(size / scale) + "px system-ui, sans-serif";
    ctx.lineWidth = 4 / scale; ctx.strokeStyle = "#000"; ctx.strokeText(text, x, y);
    ctx.fillStyle = color; ctx.fillText(text, x, y);
  }

  function drawBoxes() {
    var fr = frame(), st = stateOf(fr), px = 1 / scale;
    classesOf(fr).forEach(function (c) {
      var b = st.boxes[c];
      if (!b) return;
      ctx.lineWidth = (c === selected ? 5 : 3) * px;
      ctx.strokeStyle = D.colors[c];
      ctx.setLineDash([]);
      ctx.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1]);
      label(c, b[0] + 4 * px, Math.max(b[1] + 20 * px, 20 * px), D.colors[c], 18);
      if (c === selected) {
        ctx.fillStyle = "#fff";
        [[b[0], b[1]], [b[2], b[1]], [b[0], b[3]], [b[2], b[3]], [(b[0] + b[2]) / 2, b[1]],
         [(b[0] + b[2]) / 2, b[3]], [b[0], (b[1] + b[3]) / 2], [b[2], (b[1] + b[3]) / 2]].forEach(function (p) {
          ctx.fillRect(p[0] - 4 * px, p[1] - 4 * px, 8 * px, 8 * px);
        });
      }
    });
    // faint candidates for the highlighted missing class
    if (active && fr.proposals[active]) {
      fr.proposals[active].forEach(function (p) {
        var b = p.box, hot = hoverProp === p.n;
        ctx.globalAlpha = hot ? 0.95 : 0.45;
        ctx.lineWidth = (hot ? 4 : 2) * px;
        ctx.setLineDash([8 * px, 6 * px]);
        ctx.strokeStyle = D.colors[active];
        ctx.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1]);
        ctx.setLineDash([]);
        ctx.globalAlpha = hot ? 1 : 0.8;
        label(p.n + "  " + p.score.toFixed(2), b[0] + 4 * px, b[3] - 6 * px, D.colors[active], 16);
        ctx.globalAlpha = 1;
      });
    }
    if (drag && drag.mode === "draw" && drag.moved) {
      var r = drag.rect;
      ctx.lineWidth = 3 * px; ctx.setLineDash([4 * px, 4 * px]);
      ctx.strokeStyle = D.colors[active] || "#fff";
      ctx.strokeRect(r[0], r[1], r[2] - r[0], r[3] - r[1]);
      ctx.setLineDash([]);
    }
  }

  function draw() {
    var fr = frame(), im = loadImg(fr);
    fit();
    ctx.fillStyle = "#000"; ctx.fillRect(0, 0, fr.width, fr.height);
    $("imgerr").hidden = !im.err;
    if (im.err) $("imgerr").textContent = "image not found: " + fr.img + "  (open this page from data/review/ so the relative path works)";
    if (im.ok) ctx.drawImage(im.el, 0, 0, fr.width, fr.height);
    drawBoxes();
  }

  /* --- panels ---------------------------------------------------------------------------- */
  function pickActive() {
    var fr = frame(), miss = LE.missingClasses(classesOf(fr), stateOf(fr));
    if (!active || miss.indexOf(active) < 0) active = miss.length ? miss[0] : null;
  }

  function renderClasses() {
    var fr = frame(), st = stateOf(fr), ul = $("classes");
    ul.innerHTML = "";
    D.classes.forEach(function (c, i) {
      var li = document.createElement("li");
      var inView = classesOf(fr).indexOf(c) >= 0;
      li.className = (c === active ? "active " : "") + (c === selected ? "sel" : "");
      var has = !!st.boxes[c];
      li.innerHTML = '<span class="key">' + (i + 1) + '</span><span class="sw" style="background:' + D.colors[c] +
        '"></span><span>' + c + '</span><span class="st ' + (has ? "has" : "none") + '">' +
        (!inView ? "" : has ? "box" : "MISSING") + "</span>";
      if (inView) li.onclick = function () { if (st.boxes[c]) { selected = c; } else { active = c; selected = null; } refresh(); };
      ul.appendChild(li);
    });
  }

  function renderProps() {
    var fr = frame(), ol = $("props");
    ol.innerHTML = "";
    var props = active ? (fr.proposals[active] || []) : [];
    $("proph").textContent = active ? "Candidates for " + active : "Candidates";
    var hint;
    if (selected) hint = "Box " + selected + " selected: keys 1-" + D.classes.length + " set its class, Delete removes it, Esc deselects.";
    else if (active && props.length) hint = "Press 1-" + Math.min(props.length, 10) + " (0 = tenth) to accept a candidate as " + active + ", or drag a box on the image. Tab = next missing class.";
    else if (active) hint = "No cached candidate for " + active + ": drag a box on the image, or exclude the frame (X).";
    else hint = "All classes have a box. V = verified and next, X = exclude, arrows = move on.";
    $("modehint").textContent = hint;
    props.forEach(function (p) {
      var li = document.createElement("li");
      li.innerHTML = '<span class="n" style="background:' + D.colors[active] + '">' + (p.n === 10 ? "0" : p.n) + "</span>" +
        "score " + p.score.toFixed(2) + " &middot; area " + (p.area_frac * 100).toFixed(2) + "%" +
        '<span class="why"></span>';
      li.querySelector(".why").textContent = "rejected: " + p.reason;
      li.onmouseenter = function () { hoverProp = p.n; draw(); };
      li.onmouseleave = function () { hoverProp = null; draw(); };
      li.onclick = function () { acceptNth(p.n - 1); };
      ol.appendChild(li);
    });
  }

  function renderTop() {
    var fr = frame(), st = statusOf(fr);
    $("tag").textContent = fr.run + "#" + fr.frame;
    var kindText = { excluded: "labeler excluded", gold: "gold check", static: "static boxes (whole run)" }[fr.kind];
    $("kind").textContent = kindText + (fr.gold && fr.kind !== "gold" ? " + gold" : "") + "  |  " + st +
      (fr.reason ? "  |  " + fr.reason : "");
    var count = { verified: 0, excluded: 0, edited: 0, incomplete: 0, todo: 0 };
    D.frames.forEach(function (f) { count[statusOf(f)]++; });
    $("progress").innerHTML = "frame <b>" + (idx + 1) + "/" + D.frames.length + "</b> &nbsp; verified <b>" + count.verified +
      "</b> &nbsp; excluded <b>" + count.excluded + "</b> &nbsp; edited <b>" + count.edited + "</b> &nbsp; incomplete <b>" + count.incomplete +
      "</b> &nbsp; todo <b>" + count.todo + "</b>";
    var strip = $("strip");
    strip.innerHTML = "";
    D.frames.forEach(function (f, i) {
      var c = document.createElement("div");
      c.className = "cell " + statusOf(f) + (i === idx ? " current" : "") +
        (i > 0 && D.frames[i - 1].run !== f.run ? " gap" : "");
      c.title = f.run + "#" + f.frame + " (" + f.kind + ")";
      c.onclick = function () { go(i); };
      strip.appendChild(c);
    });
  }

  function refresh() { pickActive(); renderTop(); renderClasses(); renderProps(); draw(); }

  /* --- actions --------------------------------------------------------------------------- */
  function acceptNth(i) {
    var fr = frame(), props = active ? (fr.proposals[active] || []) : [];
    if (i < 0 || i >= props.length) { msg("no candidate " + (i + 1)); return; }
    var cls = active;
    apply(LE.acceptProposal(stateOf(fr), cls, props[i], fr.width, fr.height));
    msg(cls + " := candidate " + (i + 1));
  }

  function verify() {
    var fr = frame(), v = LE.setVerified(classesOf(fr), stateOf(fr));
    if (!v) {
      var miss = LE.missingClasses(classesOf(fr), stateOf(fr));
      msg(stateOf(fr).excluded ? "frame is excluded; press X to un-exclude first" :
        "cannot verify: no box for " + miss.join(", ") + ". Accept a candidate, draw one, or exclude (X).");
      return;
    }
    apply(v); msg("");
    if (idx < D.frames.length - 1) go(idx + 1);
  }

  function toggleExclude() {
    var fr = frame();
    if (fr.kind === "static") { msg("a static check frame cannot be excluded"); return; }
    var st = stateOf(fr);
    var wasExcluded = st.excluded;
    apply(LE.setExcluded(st, !wasExcluded));
    if (!wasExcluded && idx < D.frames.length - 1) go(idx + 1);   // excluding moves on; un-excluding stays
  }

  function download() {
    var res = LE.assembleOverlay(D, states, foreign);
    var blob = new Blob([JSON.stringify(res.overlay, null, 1)], { type: "application/json" });
    var a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = D.split + ".json";
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(function () { URL.revokeObjectURL(a.href); }, 2000);
    var n = Object.keys(res.overlay.frames).length, s = Object.keys(res.overlay.static_overrides).length;
    msg("downloaded " + D.split + ".json: " + n + " frame entries, " + s + " static override run(s). Save it as data/corrections/" + D.split + ".json" +
      (res.warnings.length ? "\nNOT written:\n" + res.warnings.join("\n") : ""));
  }

  function loadPrevious(file) {
    var rd = new FileReader();
    rd.onload = function () {
      var overlay;
      try { overlay = JSON.parse(rd.result); } catch (e) { msg("not valid JSON: " + e.message); return; }
      var r = LE.applyOverlay(D, overlay, states);
      if (r.problems.length && !r.applied) { msg("overlay not loaded: " + r.problems.join("; ")); return; }
      states = r.states;
      Object.keys(r.foreign.frames).forEach(function (f) { foreign.frames[f] = r.foreign.frames[f]; });
      Object.keys(r.foreign.static).forEach(function (f) { foreign.static[f] = r.foreign.static[f]; });
      undoStack = []; save(); refresh();
      msg("loaded " + r.applied + " entries; " + Object.keys(r.foreign.frames).length + " entries for frames not on this page are kept" +
        (r.problems.length ? "\nskipped: " + r.problems.join("; ") : ""));
    };
    rd.readAsText(file);
  }

  /* --- pointer --------------------------------------------------------------------------- */
  function pos(ev) {
    var r = cv.getBoundingClientRect();
    var fr = frame();
    return [(ev.clientX - r.left) / r.width * fr.width, (ev.clientY - r.top) / r.height * fr.height];
  }

  cv.addEventListener("pointerdown", function (ev) {
    var fr = frame(), st = stateOf(fr), p = pos(ev), tol = 9 / scale;
    cv.setPointerCapture(ev.pointerId);
    if (selected && st.boxes[selected]) {
      var h = LE.handleAt(st.boxes[selected], p[0], p[1], tol);
      if (h) { drag = { mode: "resize", handle: h, start: p, box: st.boxes[selected].slice(), cls: selected, moved: false, before: LE.cloneState(st) }; return; }
      var b = st.boxes[selected];
      if (p[0] >= b[0] && p[0] <= b[2] && p[1] >= b[1] && p[1] <= b[3]) {
        drag = { mode: "move", start: p, box: b.slice(), cls: selected, moved: false, before: LE.cloneState(st) }; return;
      }
    }
    drag = { mode: "draw", start: p, rect: [p[0], p[1], p[0], p[1]], moved: false };
  });

  cv.addEventListener("pointermove", function (ev) {
    var fr = frame(), p = pos(ev);
    if (!drag) {
      var st = stateOf(fr);
      var h = selected && st.boxes[selected] ? LE.handleAt(st.boxes[selected], p[0], p[1], 9 / scale) : null;
      cv.style.cursor = h ? (h.length === 2 ? (h === "nw" || h === "se" ? "nwse-resize" : "nesw-resize") : (h === "n" || h === "s" ? "ns-resize" : "ew-resize")) : "crosshair";
      return;
    }
    var dx = p[0] - drag.start[0], dy = p[1] - drag.start[1];
    if (!drag.moved && Math.hypot(dx, dy) * scale < 4) return;
    drag.moved = true;
    if (drag.mode === "draw") { drag.rect = [drag.start[0], drag.start[1], p[0], p[1]]; }
    else {
      var nb = drag.mode === "move" ? LE.moveBox(drag.box, dx, dy, fr.width, fr.height)
                                    : LE.resizeBox(drag.box, drag.handle, dx, dy, fr.width, fr.height);
      stateOf(fr).boxes[drag.cls] = nb;     // live preview; committed with undo on release
    }
    draw();
  });

  cv.addEventListener("pointerup", function (ev) {
    if (!drag) return;
    var fr = frame(), d = drag; drag = null;
    if (d.mode === "draw") {
      if (!d.moved) {                              // a click: select the smallest box under it
        var hit = LE.hitTest(stateOf(fr).boxes, d.start[0], d.start[1]);
        // a click on a candidate's number badge accepts it
        var props = active ? (fr.proposals[active] || []) : [];
        var badge = props.filter(function (q) {
          return Math.abs(d.start[0] - q.box[0]) < 16 / scale && Math.abs(d.start[1] - q.box[1]) < 16 / scale;
        })[0];
        if (badge) { acceptNth(badge.n - 1); return; }
        selected = hit; refresh(); return;
      }
      if (!active) { msg("every class has a box: delete one (select it, Delete) before drawing a new one"); refresh(); return; }
      var r = d.rect;
      if (Math.abs(r[2] - r[0]) * scale < 6 || Math.abs(r[3] - r[1]) * scale < 6) { refresh(); return; }
      var cls = active;
      apply(LE.drawBox(stateOf(fr), cls, r, fr.width, fr.height));
      msg("drew " + cls);
      return;
    }
    if (!d.moved) return;
    var finalBox = stateOf(fr).boxes[d.cls];
    states[fr.key] = d.before;                    // restore, then commit as one undoable step
    apply(LE.setBox(d.before, d.cls, finalBox, fr.width, fr.height));
  });

  /* --- keyboard -------------------------------------------------------------------------- */
  document.addEventListener("keydown", function (ev) {
    if (ev.ctrlKey || ev.metaKey) {
      if (ev.key.toLowerCase() === "z") { ev.preventDefault(); undo(); }
      return;
    }
    var k = ev.key, fr = frame(), st = stateOf(fr);
    if (k === "ArrowRight") { ev.preventDefault(); go(idx + 1); return; }
    if (k === "ArrowLeft") { ev.preventDefault(); go(idx - 1); return; }
    if (k === "Escape") { selected = null; refresh(); return; }
    if (k === "Tab") {
      ev.preventDefault();
      var miss = LE.missingClasses(classesOf(fr), st);
      if (miss.length) { active = miss[(miss.indexOf(active) + 1) % miss.length]; selected = null; refresh(); }
      return;
    }
    if (k === "Delete" || k === "Backspace") {
      ev.preventDefault();
      if (selected && st.boxes[selected]) { var c = selected; selected = null; apply(LE.deleteBox(st, c)); msg("deleted " + c); }
      return;
    }
    var lower = k.toLowerCase();
    if (lower === "x") { toggleExclude(); return; }
    if (lower === "v") { verify(); return; }
    if (lower === "z" || lower === "u") { undo(); return; }
    if (lower === "n") { nextTodo(); return; }
    if (lower === "s") { download(); return; }
    if (lower === "b") {
      var have = classesOf(fr).filter(function (c) { return st.boxes[c]; });
      if (have.length) selected = have[(have.indexOf(selected) + 1) % have.length];
      refresh(); return;
    }
    if (k === "?") { $("help").hidden = !$("help").hidden; return; }
    if (k.length === 1 && k >= "0" && k <= "9") {
      if (selected && st.boxes[selected]) {
        var ci = LE.classIndexForKey(k, D.classes.length);
        if (ci < 0) return;
        var to = D.classes[ci];
        if (classesOf(fr).indexOf(to) < 0) { msg(to + " is not part of this view"); return; }
        apply(LE.setClass(st, selected, to));
        selected = to;
        msg("class := " + to);
      } else if (active && (fr.proposals[active] || []).length) {
        acceptNth(LE.proposalIndexForKey(k));
      } else {
        msg("select a box first (click it) to set its class with 1-" + D.classes.length);
      }
    }
  });

  window.addEventListener("resize", draw);
  $("bPrev").onclick = function () { go(idx - 1); };
  $("bNext").onclick = function () { go(idx + 1); };
  $("bVerify").onclick = verify;
  $("bExclude").onclick = toggleExclude;
  $("bUndo").onclick = undo;
  $("bSave").onclick = download;
  $("bHelp").onclick = function () { $("help").hidden = !$("help").hidden; };
  $("bLoad").onclick = function () { $("fLoad").click(); };
  $("fLoad").onchange = function () { if (this.files[0]) loadPrevious(this.files[0]); this.value = ""; };
  $("bClear").onclick = function () {
    if (!window.confirm("Forget the autosaved progress of this page in this browser?")) return;
    try { localStorage.removeItem(STORE_KEY); } catch (e) { /* ignore */ }
    states = {}; foreign = { frames: {}, static: {} }; undoStack = []; refresh(); msg("autosave cleared");
  };

  if (!D.frames.length) {
    $("tag").textContent = "no frames";
    $("kind").textContent = "nothing to review for this split (see the generator output)";
  } else {
    var restored = restore();
    var first = D.frames.findIndex(function (f) { var s = statusOf(f); return s === "todo" || s === "incomplete"; });
    idx = first >= 0 ? first : 0;
    go(idx);
    if (restored) msg("restored " + restored + " frame(s) from this browser's autosave");
  }
})();
