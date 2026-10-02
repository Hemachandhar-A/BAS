# Essential features (Tier 1 — must be built for the SIH submission)

This document is the **implementation spec**: for every must-build feature it gives the contract it must satisfy, how to build it (libraries and calls), its parameters, its known pitfalls, and where its acceptance test is defined. For **why** a choice was made — evidence, alternatives, rejections — follow the links into [`context.md`](./context.md). For **who builds it, in what order, and how it merges**, see [`IMPLEMENTATION_PLAN.md`](./IMPLEMENTATION_PLAN.md). The exact shapes and signatures are in [`contracts.py`](./contracts.py), which wins if anything here disagrees.

Every feature is required either because the problem statement asks for it directly or because the architecture cannot function without it. Optional and excluded items: [`non-essential-features.md`](./non-essential-features.md). **F\<n\>** everywhere = Feature n below.

| F | Feature | Owner · session | Code lives in |
|---|---|---|---|
| 1 | Video ingestion | P1 · P1.1 | `perception/camera.py` |
| 2 | Object detection | P1 · P1.5–P1.6 | `perception/detector.py`, `training/` |
| 3 | Hand/pose tracking | P1 · P1.3 | `perception/hands.py` |
| 4 | Hand–object interaction | P2 · P2.1 | `state/tracker.py` |
| 5 | Sequence engine | P2 · P2.2 | `engine/sequence.py` |
| 6 | Skip / out-of-order detection | P2 · P2.2 | `engine/sequence.py` |
| 7 | Voice alerts | P2 · P2.3 | `outputs/tts.py` |
| 8 | Next-step suggestion | P2 · P2.2, P2.4 | `engine/sequence.py`, `runtime/loop.py` |
| 9 | Structured log | P2 · P2.3, P2.4 | `outputs/logger.py` |
| 10 | Video streaming | P2 · P2.5 | `server/app.py` |
| 11 | Local video storage | P2 · P2.4 | `runtime/recorder.py` |
| 12 | GUI dashboard | P2 · P2.5 | `server/static/` |
| 13 | Reliability layer | P2 · P2.1 (+ P2.4, P2.6) | `state/tracker.py`, `runtime/loop.py` |
| 14 | Dataset generation pipeline | P1 · P1.1, P1.4–P1.7 | `perception/record.py`, `training/` |

---

## 0. Shared implementation rules (apply to every feature)

**Conventions** (from `contracts.py`, restated because they cause the most bugs): time `t` is float **seconds since the source started**, never wall-clock; pixels are **original-frame pixels**, origin top-left, boxes `(x1, y1, x2, y2)`; images are `np.ndarray (H, W, 3) uint8` in **BGR** (OpenCV native). **RF-DETR and MediaPipe want RGB** — convert *at the model boundary only* (`image[..., ::-1]`), never earlier. IDs are `lower_snake_case`. Confidence is `[0, 1]`.

**Threading model** (runtime, F1/F10/F11/F13):

```
capture thread   : source.read() -> LatestFrameStore (holds ONLY the newest Frame; overwrites; never blocks)
                   \-> recorder queue (every captured frame; F11)
inference thread : take newest frame (skip if same frame_id) -> throttle to target_fps
                   -> Perception.process -> StateTracker.update -> Engine.on_state_event -> router
router           : LogEntry -> JsonlLogger ; EngineEvent.speak -> Speaker ; status store update
Flask threads    : READ-ONLY from the latest-frame store (JPEG-encoded once per frame) and the status store.
TTS              : a separate process (F7)
```

**Latest-frame-wins:** if inference is slower than capture, intermediate frames are dropped, never queued, so latency does not grow. File replay (harness) has no threads: it reads sequentially and never sleeps.

**Processing rate is a first-class parameter.** Hysteresis is counted in *processed frames* (F13). A simulation of the reference tracker on the sample experiment fired all 7 steps at 15, 10 and 6 fps but only **5 of 7 at 4 fps**. Therefore: the live pipeline must sustain **≥ 8 fps** (`min_pipeline_fps` in `config/acceptance.yaml`); `RuntimeConfig.target_fps` is set from the P1.5 benchmark (default 15, lower it if the CPU cannot keep up); perception caches are built at the same rate (F14, stage 9); and `hysteresis_frames` is tuned at that rate (P2.6).

**Project-wide constants (all disclosed judgment calls, defined once, never re-invented):**

| Constant | Value | Where |
|---|---|---|
| `DETECTOR_MIN_CONF` | 0.10 | `perception/detector.py` — deliberately far below `detector_conf_floor` (0.30) so the tracker's floor is what decides, and tuning has room |
| Capture mode | 1280 × 720 @ 30 fps | `perception/camera.py` |
| `MAX_SPOKEN_WORDS` | 8 | experiment lint (a `say` is a phrase, not a sentence) |
| `RECENT_ALERTS_CAP` / `DASHBOARD_POLL_MS` | 20 / 500 | `server/` |
| `min_pipeline_fps` | 8 | `config/acceptance.yaml` |

**Errors:** frame-level exceptions in perception are caught, logged with the frame id, and the frame is skipped (the loop never crashes on one bad frame). `ContractViolation` is raised, never coerced. No `print` in library code — use `logging`.

**Cross-platform:** the team's laptop OS is not fixed (see `ISSUES.md`), so every tool is a Python entry point (`python -m ...` / `python scripts/x.py`), never a bare shell script.

---

## 1. Video ingestion

**What:** reads frames from the fixed payload camera (or a recorded file) in real time.
**Contract:** the `FrameSource` protocol (the factory `open_source(source: str) -> FrameSource` is named in the comment on `FrameSource` in `contracts.py` and implemented in `perception/camera.py`) has `fps: float | None`, `exhausted: bool`, `read() -> Frame | None`, `close()`; `Frame(frame_id, t, image)`; `SourceError` raised **at construction**.

**Implementation** (`perception/camera.py`)
1. **Parse `source`:** all digits → camera index `int(source)`; anything else → file path or stream URL, passed straight to `cv2.VideoCapture(source)`. One module, three inputs, so an IP camera later is a one-line swap.
2. **Open and validate at construction:** if `not cap.isOpened()` or the first `read()` fails → `SourceError`. On Windows, if opening a webcam is slow or fails with the default backend, retry with `cv2.CAP_DSHOW`.
3. **Request the capture mode** (`cap.set` for width, height, fps) and **read back what the driver actually granted**; log it. Many USB webcams reach 720p30 only in MJPG mode — if not reached, set the FOURCC to `MJPG` before requesting size.
4. **`fps` attribute:** `cap.get(cv2.CAP_PROP_FPS)`; if it is 0 or NaN (common for webcams), **measure** the median inter-frame interval over the first 30 frames. This value is what `RunScript.fps` records.
5. **`Frame.t`:** file → `frame_id / fps`; camera → `time.monotonic()` at read minus the start value. **This is the only place in the perception layer allowed to read a clock.** `frame_id` starts at 0 and is strictly increasing.
6. **`read()`:** file → on failure set `exhausted = True`, return `None`. Camera → on failure return `None` with `exhausted = False` (transient); the runtime turns a long gap into `feed_lost` (F13).
7. **Decimation wrapper** `DecimatedSource(src, every=k)` yields every k-th frame **keeping the original `frame_id`** (so `t` stays correct); used by the cache builder (F14, stage 9).
8. Try `CAP_PROP_BUFFERSIZE = 1` where the backend honors it, to avoid stale buffered frames.

**Parameters:** `RuntimeConfig.source` (`"0"` by default), `target_fps`; the capture-mode constants above.
**Pitfalls:** camera index changes with the USB port; a URL source can block on open; replaying a file must not sleep.
**Done when:** [Plan §7.4, F1]. **Why this shape:** [context.md, camera hardware](./context.md#camera-hardware).

---

## 2. Object detection

**What:** locates the experiment's objects (`ExperimentDefinition.classes`; the draft sample experiment has `outer_box`, `tray`, `red_box`, `yellow_box`, `start_button`) in every frame.
**Contract:** produces `Detection(label ∈ experiment.classes, conf ∈ [0,1], box in original pixels)` inside a `PerceptionFrame`. **No experiment logic here**; the tracker applies the confidence floor.

**Implementation** (`perception/detector.py`; training in `training/`, feature 14)
1. **Model:** `RFDETRNano` from the `rfdetr` package (Apache-2.0; native resolution 384 × 384; ~30 M parameters). Fine-tuned from COCO-pretrained weights (F14, stage 6).
2. **Load:** `model = RFDETRNano(pretrain_weights="weights/detector.pth")`, then `model.optimize_for_inference()` once (the RF-DETR docs report up to ~2× speedup; it fixes image and batch size, which is fine here). **Warm up** with 3 dummy frames so the first live frame is not slow.
3. **Detect:** convert BGR → **RGB**; `dets = model.predict(rgb, threshold=DETECTOR_MIN_CONF)` returns a `supervision.Detections` with `.xyxy`, `.confidence`, `.class_id`.
4. **Map ids to labels** through the class list stored in `weights/MANIFEST.json` and **assert it matches the checkpoint's own class names and `experiment.classes` at load** (RF-DETR uses 0-based class indices; a silent off-by-one here would swap red and yellow). Clip boxes to the image and enforce `x1 ≤ x2, y1 ≤ y2` (the contract rejects otherwise).
5. **Threads:** `torch.set_num_threads(cores − 2)` so the tracker, server and TTS keep a core (a judgment call; measured in the benchmark).
6. **ONNX/CPU alternative:** `model.export()` (needs `pip install "rfdetr[onnx]"`) → run with `onnxruntime` on CPU; input resolution is frozen at export. Third-party model cards claim 2–4× over PyTorch on CPU — **treat as an unverified hypothesis**. The P1.5 benchmark compares **PyTorch, ONNX Runtime and YOLO11n on this laptop's CPU** and records the winner in `ISSUES.md`. If ONNX wins, `rfdetr`/`torch` become training-only and the runtime needs only `onnxruntime`; the wrapper interface is unchanged.
7. **Fallback:** YOLO11n (Ultralytics, AGPL-3.0) only if the benchmark or the evaluation forces it, with the license decision logged. It is a **build-time decision made once at P1.5**, not a runtime switch: the runtime loads exactly one detector, and a different detector means a new `model_stamp` and rebuilt caches. Triggers (proposed 2026-09-28; both people ack at G0): after one fix cycle RF-DETR-Nano cannot reach `min_pipeline_fps` under PyTorch or ONNX Runtime (try ONNX Runtime first), misses the `acceptance.yaml` recall on `val`, or cannot be fine-tuned on any available GPU. See `context.md` §6.
8. **Small objects:** the START button is the smallest object (~6 cm). If its recall on `val` is weak, first enlarge the physical button; only then consider a larger `resolution` (follow the constraint stated in the RF-DETR docs for the installed version).

**Determinism:** eval mode, no augmentation at inference; same frame → identical detections (tested).
**Pitfalls:** RGB vs BGR; class-id mapping; the first call is slow; `optimize_for_inference` fixes the batch size.
**Done when:** [Plan §7.4, F2]. **Why RF-DETR over YOLO, and the open CPU question:** [context.md §6](./context.md#6-perception-model-choices).

---

## 3. Hand/pose tracking

**What:** locates hands (and optionally body pose) per frame.
**Contract:** `Hand(handedness, score, landmarks_px[21])`; `Pose(score, landmarks_px[33])`; `hands = []` when no hand is visible — **never an error**; `pose = None` when disabled.

**Implementation** (`perception/hands.py`)
1. **Use the MediaPipe *Tasks* API**, not the legacy `mp.solutions`: `HandLandmarker.create_from_options(HandLandmarkerOptions(base_options=BaseOptions(model_asset_path="weights/hand_landmarker.task"), running_mode=VisionRunningMode.VIDEO, num_hands=2, …))`. Set the detection / presence / tracking confidence options to 0.5 (check the option names against the installed version's docs). **The `.task` file is vendored in `weights/`** — the runtime must download nothing.
2. **Call** `landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)`. In VIDEO mode the timestamp must be **monotonically increasing** — MediaPipe raises `ValueError` for a smaller value. Use `int(round(frame.t * 1000))` and bump by +1 ms if two frames round to the same value. **`reset()` recreates the landmarker** (timestamps restart at each run).
3. **Convert to pixels:** landmarks come back normalized `[0,1]`; multiply x by width and y by height. Nobody downstream ever sees normalized coordinates (contract rule).
4. **Handedness** is passed through as reported and is **informational — no rule consumes it.** MediaPipe assumes a mirrored (selfie) image, so labels may look swapped on a non-mirrored camera; do not depend on it.
5. **Pose (`PoseLandmarker`, `pose_landmarker_lite.task`)** is **off by default** (`enable_pose=False` on the pipeline constructor): no Tier-1 rule consumes it and it costs CPU. Enable it only if the P1.6 benchmark shows headroom above `min_pipeline_fps`; record the decision in `ISSUES.md`.
6. Hand and pose model sha256 prefixes go into `model_stamp`.

**Pitfalls:** non-monotonic timestamps; hands in **gloves** may drop out (the robustness runs measure this); input must be contiguous `uint8` RGB.
**Done when:** [Plan §7.4, F3]. **Why MediaPipe:** [context.md §6](./context.md#6-perception-model-choices).

---

## 4. Hand–object interaction

**What:** decides when a hand is touching a specific object (the `hand_touching` rule).
**Contract:** rule `HandTouching(label)`; `PerceptionConfig.touch_margin_frac`.

**Implementation** (a pure function in `state/tracker.py`)
1. Take the **best** (highest-confidence, ties first-in-list) detection of `label`; if none, the rule is false.
2. **Grow the box** by `touch_margin_frac` × its width on left/right and × its height on top/bottom.
3. The rule is **true when at least one fingertip landmark (MediaPipe indices 4, 8, 12, 16, 20) of any hand lies inside the grown box.** Wrist, palm and the other landmarks do not count. For the step's confidence, include the touching hand's `score` (with the detection `conf`).
4. Not a learned model: no interaction classifier.

**Parameters:** `touch_margin_frac = 0.10` (a disclosed default, tuned in P2.6).
**Pitfalls:** with an **overhead** camera a hand passing *above* an object overlaps it in 2-D without touching. The defenses are the fingertip-only rule (an arm or palm crossing no longer counts, but the fingertip must be visible near the object), hysteresis (a real touch is held) and a physical layout that keeps START away from the boxes' path; measured limitation: on the train clips the best fingertip variant recognised the START press in 9 of 14 correct clips with 0 extras (ISSUES.md, 2026-10-02 P2 CONTRACT); the `idle-b` runs (hands wandering) measure the false-positive rate.
**Done when:** [Plan §7.4, F4]. **Why a heuristic:** [context.md §6](./context.md#6-perception-model-choices).

---

## 5. Sequence engine

**What:** tracks which step is happening and validates order against the canonical list.
**Contract:** `SequenceEngine(experiment: ExperimentDefinition, config: RuntimeConfig)` implements `Engine` (`run_state`, `start(t)`, `on_state_event(StateEvent) -> list[EngineEvent]`, `finish(t)`, `snapshot() -> list[StepProgress]`). Emits `EngineEvent` and `RunSummary`.

**Implementation** (`engine/sequence.py`) — the decision table is in [Plan §5.4](./IMPLEMENTATION_PLAN.md); this is the verified reference logic:

```python
# state: status[step_id] in {"pending","confirmed","skipped","completed_late"}; observed: list[step_id]
def on_state_event(ev):
    if state != "running": return []
    o = ev.step_id;  if o not in status: raise ContractViolation(o)
    observed.append(o); e = first_pending()                       # expected BEFORE this event
    tag = "flagged_uncertain" if ev.uncertain else "confirmed"
    if   status[o] == "pending" and o == e:   -> step_confirmed;   status[o] = "confirmed"
    elif status[o] == "pending":              -> deviation omission; skipped = pending steps before o;
                                                 those -> "skipped"; status[o] = "confirmed"
    elif status[o] == "skipped":              -> deviation out_of_order; status[o] = "completed_late"
    else:                                     -> deviation repeat (status unchanged)
    if o == last_step and it was newly confirmed:  also emit run_completed(summary); state = "completed"
```

Every `EngineEvent` also carries `expected_step_id`, `next_step_id` (first pending *after* the event), `confidence_tag`, and `speak` (exact TTS text or `None`, per the plan's templates so goldens can pin it). **`RunSummary.pos`** = `1 − min(d / n, 1)` where `d = DamerauLevenshtein.distance(canonical_ids, observed_sequence)` from **RapidFuzz** (`from rapidfuzz.distance import DamerauLevenshtein`; verified to accept lists of step ids) and the observed sequence is every step id processed while running, repeats included. `snapshot()` returns `StepProgress` for every step with `t_confirmed` set from the confirming event.

**Reference derivation** (`engine/reference.py`, same session): `derive_expected_deviations(experiment, performed_steps) -> list[ExpectedDeviation]` (omission → the skipped ids; out_of_order / repeat → `[the observed step]`) and a CLI `python -m engine.reference --write runs/` that fills each `script.json`'s `expected_deviations` (a **derived** field, never hand-edited). A contract test fails if any file is stale.

**Pitfalls:** swapping the **last** pair reports the late step as `skipped` (run completes at the last step; later events are ignored) — a contract consequence, documented in Plan §5.4. The engine never reads a clock; time arrives on `StateEvent.t`.
**Done when:** [Plan §7.4, F5]. **Why this design:** [context.md §7](./context.md#7-the-sequence-engine-the-projects-core-logic).

---

## 6. Skip / out-of-order detection

**What:** flags deviations from the canonical order. **It is not a separate module:** it *is* rows 2–4 of the engine's decision table (F5).
**Implementation:** omission (later step first, earlier ones reported skipped) · out_of_order (a skipped step done late) · repeat (an already-done step again). One `deviation_detected` `EngineEvent` per deviating `StateEvent`; alerts are spoken by F7 with the exact `speak` text.
**Verification:** GOLD-1..5 (Plan §7.2) and a **property test**: for random permutations / omissions / duplications of the canonical list that end *before* the last step completes, the engine flags ≥ 1 deviation **iff** the observed sequence differs from the canonical prefix of equal length (Damerau-Levenshtein > 0).
**Done when:** [Plan §7.4, F6]. **Why:** same as F5.

---

## 7. Voice alerts

**What:** speaks a warning on every `deviation_detected`; also speaks F8's cues.
**Contract:** `Speaker.say(text: str, priority: "alert" | "info") -> None` (non-blocking, **never raises**, an `alert` pre-empts current speech) and `close()`. Implemented by `TTSWorker` in `outputs/tts.py`.

**Implementation**
1. **Engine: `pyttsx3`** — fully offline; it drives the operating system's own voices (SAPI5 on Windows, NSSpeechSynthesizer on macOS, eSpeak on Linux). There is **no model to download or vendor**. Linux needs `espeak-ng` and `libespeak1`; on Windows, if `win32com` is missing, install `pypiwin32`.
2. **A separate process** (`multiprocessing`, start method set explicitly to `spawn`). pyttsx3's `runAndWait()` has long-standing hang reports on some macOS setups, and the workaround users report is to run it in a child process and **kill the process to interrupt speech**. So: one persistent worker process reads `(text, priority)` from a `multiprocessing.Queue`. An **`alert` clears the queue and interrupts the current utterance by terminating the worker and immediately starting a fresh, already-initialized one**; `info` is queued.
3. **Voice setup in the worker:** `engine = pyttsx3.init()`; `engine.setProperty("rate", 170)` (a judgment call; intelligibility over speed), `("volume", 1.0)`; choose the first installed voice whose name or languages indicate English, else the default; log the chosen voice.
4. **Cooldown:** an identical `alert` text inside `RuntimeConfig.alert_cooldown_s` is not re-spoken.
5. **Degrade quietly:** if no audio device or engine init fails, become a silent no-op and log **one** warning. `say()` still never raises.
6. **Test doubles:** `FakeSpeaker` records `(text, priority)` in order (used by every golden). One integration test uses a stub backend, because CI has no audio device. **Every dev machine runs an audio self-test at G0** (speak a sentence) so a silent-voice surprise never reaches the demo.

**Quality upgrade (Tier 2):** pre-render the *closed, finite* set of utterances (the step `say` phrases, the alert templates × display names, the completion phrases) with a higher-quality offline voice and play the WAVs, falling back to pyttsx3 for anything not cached — see [non-essential-features.md](./non-essential-features.md).
**Pitfalls:** the OS voice quality varies by machine (test the demo laptop); non-English voices mangle the phrases.
**Done when:** [Plan §7.4, F7]. **Why offline TTS, and the voice decision:** [context.md, voice output (TTS)](./context.md#voice-output-tts).

---

## 8. Next-step suggestion

**What:** tells the operator what to do next, at the start and after each step.
**Implementation:** it is a by-product of F5. On `step_confirmed`, the engine sets `speak = next_step.say` (if `RuntimeConfig.narrate_next_step` and a next step exists). **At run start** the runtime itself speaks `experiment.steps[0].say` and writes `run_started` (the engine's `start()` returns nothing). The dashboard shows the same text from `StatusResponse.next_step_say`. Phrases come from `StepDef.say` — a short phrase, not a sentence (lint: ≤ `MAX_SPOKEN_WORDS`).
**Done when:** [Plan §7.4, F8]. **Why voice-primary:** [context.md, voice vs. on-screen](./context.md#voice-vs-on-screen-instruction-delivery).

---

## 9. Structured log

**What:** a timestamped, structured, lightweight record of steps and outcomes — the problem statement's explicit requirement.
**Contract:** `LogSink.write(LogEntry)`; `LogEntry(seq, run_id, t_wall, t_video, event_type, step_id, expected_step_id, deviation_type, skipped_step_ids, confidence_tag, detail)`. Implemented by `JsonlLogger(path, wall_clock=...)` in `outputs/logger.py`.

**Implementation**
1. **JSON Lines**, one entry per **event, never per frame**; UTF-8; file `{log_dir}/{run_id}.jsonl`.
2. `write()` appends `entry.model_dump_json() + "\n"`, then `flush()` and `os.fsync()` (events are rare, so fsync is cheap and survives a power cut).
3. **`t_wall` is stamped by the logger** from the injected `wall_clock` — the only wall-clock read in the system. Because `LogEntry` requires the field at construction, the router builds entries with the placeholder `datetime(1970, 1, 1, tzinfo=timezone.utc)` and the logger overwrites it via `model_copy(update=...)`. *(Awkward but contract-legal; `ISSUES.md` proposes making it optional at G0.)*
4. **`seq`** comes from a per-run counter in the router (starts at 0); the logger **asserts `seq == last + 1`** (`ContractViolation` otherwise).
5. **Event mapping:** engine events keep their `kind` as `event_type` (`step_confirmed`, `deviation_detected`, `run_completed`); the runtime writes `run_started`, `feed_lost` (`detail` = the reason), `feed_restored`. `detail` is a fixed one-liner **independent of narration settings** (Plan §5.5). `confidence_tag` is carried from the event, giving an audit trail without manual sign-off.
6. **Readers** (`/api/log`, post-hoc review) parse line by line and **ignore an incomplete trailing line**; a run killed mid-write must leave every earlier line valid.

**Done when:** [Plan §7.4, F9]. **Why confidence-tagging instead of manual sign-off:** [non-essential-features.md](./non-essential-features.md) and [context.md §8](./context.md#8-reliability-engineering).

---

## 10. Video streaming to a specified IP

**What:** fulfils "stream video to specific IP".
**Contract:** `API_ROUTES` in `contracts.py` (7 routes) — the Flask URL map must equal it (tested). `/video_feed` serves the **raw** camera feed as `multipart/x-mixed-replace` MJPEG.

**Implementation** (`server/app.py`, an app factory taking the stores and `RuntimeConfig`)
1. **MJPEG generator:** loop at `stream_fps` (default 10): read the newest frame, JPEG-encode with `cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])` (encode **once per new frame** and share it across clients), `yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"`; return `Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")`. Run with `threaded=True`.
2. **Access control on every route** (a `before_request` hook): first the **client-IP allowlist** (`request.remote_addr in RuntimeConfig.allowed_client_ips` → else `403`), then **HTTP Basic auth** (`request.authorization`; credentials from the environment variables `STREAM_USER` / `STREAM_PASSWORD` loaded from a git-ignored `.env`; compare with `hmac.compare_digest`; else `401` with `WWW-Authenticate: Basic`). Credentials never appear in a URL or a log line.
3. **TLS:** when `tls_cert` and `tls_key` are set, `app.run(host, port, ssl_context=(cert, key), threaded=True)`. **Without TLS the server binds `127.0.0.1` regardless of `host`.** `python scripts/gen_cert.py` makes a self-signed cert with the `cryptography` package (works on every OS; browsers will warn once).
4. The Flask development server is adequate for one operator on one stream; a production WSGI server is out of scope.

**Pitfalls:** an inference call inside a request handler (forbidden — handlers only read stores); many clients multiplying JPEG encodes (hence encode-once).
**Done when:** [Plan §7.4, F10]. **Why Flask, and why these security measures:** [context.md §12](./context.md#12-backend-and-frontend-framework-decisions) and [§9](./context.md#9-security-considerations).

---

## 11. Local video storage

**What:** saves the raw feed locally.
**Implementation** (`runtime/recorder.py`)
1. A **writer thread** with a bounded queue fed by the capture thread (every captured frame, at capture rate — not the processing rate). If the queue is full, log a warning and drop that frame rather than stall capture.
2. **`cv2.VideoWriter`** to `{video_dir}/{run_id}.avi` with the **MJPG** fourcc: unlike an unfinalized MP4, a truncated AVI/MJPG file generally remains playable after a crash. *(The dataset recorder, F14, writes MP4 because a human stops it cleanly.)*
3. Open at run start, release at run end (`finish`) and on shutdown. Check free disk at start and warn.
4. **A recorder failure is logged and never stops the run.**
**Done when:** [Plan §7.4, F11]. **Why:** straightforward implementation of an explicit requirement.

---

## 12. GUI dashboard

**What:** one operator-facing page: live feed, current step, next-step suggestion, step list with statuses, alert history, run start/reset. It is the persistent/reference display, **not** the primary guidance channel (voice is — F8).
**Contract:** `StatusResponse` / `RunControlResponse` from `/api/status`, `/api/run/start`, `/api/run/reset`, `/api/log`, `/api/experiment` (a read-only view).

**Implementation** (`server/static/`: `index.html`, `app.js`, `style.css`, `vendor/`)
1. **Layout:** left, `<img src="/video_feed">`; right, a header with `run_state`, `feed_ok` and `fps`; the **current/next step** in large text (from `expected_step_id` / `next_step_say`); the step list with status chips (`pending`, `confirmed`, `skipped`, `completed_late`); the alert list (newest last, capped at `RECENT_ALERTS_CAP`); **Start** and **Reset** buttons (`fetch` POST). Status must not rely on colour alone — pair it with text.
2. **Vanilla JS:** `setInterval` every `DASHBOARD_POLL_MS` → `fetch("/api/status")` → render. Insert text with `textContent` (never `innerHTML` with server strings). The browser reuses the Basic-auth credentials for `fetch` and `<img>`.
3. **Offline:** CSS is a small framework (or hand-rolled) **vendored under `server/static/vendor/`**; no CDN, no web fonts, no external URL. A test scans the served HTML/CSS/JS for non-local `http(s)://`.
**Done when:** [Plan §7.4, F12]. **Why one page and vanilla JS:** [context.md §12](./context.md#12-backend-and-frontend-framework-decisions) and [§13](./context.md#13-gui--page-structure).

---

## 13. Reliability layer

**What:** stops single-frame glitches and weak detections from causing false transitions or false alerts, and keeps the system alive through faults.
**Contract:** `PerceptionConfig` (`detector_conf_floor`, `confirm_conf`, `hysteresis_frames`, `release_frames`, `baseline_frames`, `touch_margin_frac`); `StateTracker.update(PerceptionFrame) -> list[StateEvent]`.

**Implementation** — the semantics are exactly [Plan §5.3](./IMPLEMENTATION_PLAN.md); in short:
1. **Floor:** detections below `detector_conf_floor` (0.30) are ignored ("unknown", never acted on).
2. **Confirm:** events whose supporting confidence is between the floor and `confirm_conf` (0.60) are emitted with `uncertain = True` and carried into the log as `flagged_uncertain`.
3. **Hysteresis:** a step's rules must hold `hysteresis_frames` (5) consecutive processed frames.
4. **Baseline latch:** for the first `baseline_frames` (10) frames after `reset()`, already-true rules are latched and cannot fire.
5. **Release:** a fired or latched step re-arms only after `release_frames` (5) consecutive false frames.
6. **Runtime faults:** a lost feed writes `feed_lost` after `feed_timeout_s` and `feed_restored` on recovery; a perception exception skips that frame; TTS and the recorder can fail without stopping the run.

**Verified by simulation** (reference tracker, synthetic 15 fps world, sample experiment): a one-frame false detection fires nothing; confidence ≈ 0.45 fires all 7 steps flagged `uncertain`; confidence ≈ 0.15 fires nothing; and **a step whose rule is already true at the start of the run cannot fire until it has gone false and re-armed** (a first step with such a rule fired 5th instead of 1st) — hence the authoring rule that *every step's rule must be able to go false before its turn*, enforced by an experiment lint (Sample Transfer's two stow steps start true and are released when the boxes leave; revised 2026-09-28, see `ISSUES.md`).
**Parameters:** the six `PerceptionConfig` values are **disclosed judgment-call defaults**, tuned on `val` only in P2.6 at the real processing rate.
**Done when:** [Plan §7.4, F13]. **Why these mechanisms, and the calibration caveat:** [context.md §8](./context.md#8-reliability-engineering).

---

## 14. Dataset generation pipeline

**What:** turns the crew's recordings into a trained detector, per-run perception caches, and honest test numbers. Crew-facing instructions: [`DATA_COLLECTION.md`](./DATA_COLLECTION.md). Data lives in `runs/` (recordings), `data/` (everything derived; git-ignored).

**Stage 0 — Record** (`python -m perception.record --plan runs/run_plan.csv`, P1.1). Opens `open_source`, shows a live preview, runs a 3-2-1 countdown, writes `runs/<run_id>/video.mp4` (`mp4v`, capture fps) and a `script.json` (`RunScript`): `run_id`, `experiment_id`, `script_type`, `split`, `fps` (the **measured** capture fps), `camera_setup_id`, `operator` come from the plan row; the operator either confirms the planned `performed_steps` or types the actual list. `expected_deviations` is left empty here and filled by `engine.reference` (F5).

**Stage 1 — Validate** (`python -m perception.record --validate runs/`). Per run: schema-valid `script.json`; every `performed_steps` id ∈ `experiment.step_ids`; the video opens; measured fps within ±5 % of the declared `fps`; resolution equals the session's; duration 20–150 s (judgment); mean luminance in a sane band and not a frozen video; `expected_deviations` current (calls `engine.reference` when present). Writes the **derived** `runs/manifest.csv` (`run_id, split, script_type, fps, frames, duration_s, width, height, operator, camera_setup_id, video_sha256`).

**Stage 2 — Sample frames** (`training/sample_frames.py`; **train, val and test runs**, so the detector's `valid`/`test` folders exist; only `train` frames are ever trained on). Per run: stride ≈ `round(src_fps / 2)` (≈ 2 fps), then **de-duplicate**: keep a frame only if the mean absolute difference of its 32 × 32 grayscale thumbnail from the last kept frame exceeds ~6/255 **or** ≥ 1 s has passed; cap at 120 frames per run. Deterministic (no randomness). Output: JPEGs at native resolution named `<run_id>_<frame_id>.jpg` (labels are in original pixels; the trainer resizes). Expect ≈ 2,000–3,000 train images, inside the range RF-DETR's docs cover for fine-tuning.

**Stage 3 — Auto-label** (`training/autolabel.py`, prompts in `training/prompts.yaml`). Run an open-vocabulary detector offline: **YOLO-World** (through Ultralytics — AGPL, so kept in a dev-tools dependency group and never in the runtime environment) **or Grounding DINO** (`transformers`, Apache-2.0); P1.2 chooses by license, speed and accuracy on real frames. One text prompt list per class (e.g. `red_box: ["red box"]`, `start_button: ["green button","green coaster"]`). Then **post-process**: keep the single highest-scoring box per class (each object appears once), enforce an area band per class, and check the mean hue inside the box against the class colour for red/yellow/green. **Static-object smoothing:** the outer box, tray and START button never move, so within a run replace each such box with the run's per-class median box when its IoU with that median is ≥ 0.5, and flag it otherwise. Frames missing a static object are excluded and counted; movable-object misses are kept only if flagged for review. Output: COCO-style JSON per split.

**Stage 4 — Human review** (`python -m training.review_sheet` → `data/review/index.html`; the crew fills `data/label_review.csv`: `run_id, frame_id, class, verdict(ok|bad), reason(wrong_box|missing|duplicate|wrong_class)`). One page per class, ~40 frames drawn at random (seeded) and **stratified across runs and setups**, plus all flagged frames up to 20. **Gate: if any class exceeds 10 % `bad` (a judgment threshold), fix prompts/thresholds (or correct the flagged frames in the label editor, Stage 4b), rebuild, and review a *fresh* sample.** The review outcome is recorded in `reports/dataset.json`. **Also (added 2026-09-28):** (a) **pilot gate:** P1.2 runs Stages 2–4 on the pilot frames (all `train`) and records the per-class `bad` fraction in its `DECISION` entry, so a failing class is known before rows 13–77 are recorded; (b) **static boxes are reviewed per run:** `review_sheet --static` renders one frame per run with its three static-object boxes (`outer_box`, `tray`, `start_button`), and every run must be `ok` or fixed (Stage 4b) before Stage 5, because one fix covers every frame in that run.

**Stage 4b — Correct** (`training/label_editor/`; a dev-time **static, offline HTML page**: no server, no CDN, vanilla JS; not part of the product GUI, see `non-essential-features.md` excluded §3). Generated by `python -m training.review_sheet --edit --split <s> [--gold N]`, with the frame list and auto-boxes **inlined** in the page (browsers typically block `fetch()` of local files, so nothing is fetched; verify on the crew's browsers, `ISSUES.md`). Verbs: **move/resize** a box (`wrong_box`), **set class** with keys 1–5 in `experiment.classes` order (`wrong_class`), **draw** a box (`missing`), **delete** a box (`duplicate` or a false positive), **exclude** a frame, **confirm** a frame as verified. It writes an **overlay** (`data/corrections/<split>.json`, format in Plan §5.8) through a browser download; the auto-labels are never overwritten. Uses, in priority order: (1) per-run static-box fixes; (2) the **gold subset:** `--gold N` picks N frames per `val` and `test` run (default 6, about 250 frames; a judgment call), seeded and stratified across runs and setups, which the crew verifies or fixes; (3) hard `train` frames, where **excluding** a frame is preferred to correcting it (my reasoning, not sourced). **Test hygiene:** `test` frames are corrected from the image and the auto box only, before any detector prediction on `test` has been seen. **Why a gold subset:** without it Stage 7 scores the detector against the auto-labeler's own labels, and label errors in evaluation sets are known to distort conclusions ([Northcutt et al., NeurIPS 2021](https://arxiv.org/abs/2103.14749)); the headline step-detection numbers are unaffected because they are scored against the crew's declared steps. **Existing tools were considered and not adopted:** Label Studio ([Apache-2.0](https://github.com/HumanSignal/label-studio), pip-installable, but pre-labels need a converter, local-file serving and a database, and analytics must be switched off with `COLLECT_ANALYTICS=False`) and CVAT ([MIT](https://github.com/cvat-ai/cvat); the install guides seen use a Docker Compose stack with Postgres and Redis, current guide unverified). Both are heavy for a few hundred edits and add a class-id conversion step, where an off-by-one would swap red and yellow (F2, step 4).

**Stage 5 — Build the dataset** (`python -m training.build_dataset`). RF-DETR's expected layout: `data/dataset/{train,valid,test}/` each with the images and an `_annotations.coco.json` (RF-DETR auto-detects COCO). **Our `val` split → `valid`**. **Assertions:** no run appears in two splits; `train` contains only `train` runs; class names equal `experiment.classes` in order; ids contiguous. Verify the id ↔ name mapping by a round-trip test (load the JSON, look up a known box). Writes `reports/dataset.json` (counts per split, class and run; a stamp = sha256 prefix of the annotation files **and the overlay files**). **Apply `data/corrections/*.json` on top of the auto-labels** (a frame overlay replaces that frame's boxes; run-level static overrides apply to every frame of the run; excluded frames are dropped) and assert: every overlay class ∈ `experiment.classes`, every overlay frame exists, and each overlay file only touches frames of its own split. `reports/dataset.json` also records frames corrected, excluded and verified per split. **Split by run, never by frame.**

**Stage 6 — Train** (`training/finetune.py`, on a **GPU**: the team's laptops are CPU-only, so this runs on Colab/Kaggle — see [context.md, compute environments](./context.md#compute-environments-where-each-step-runs)). `RFDETRNano().train(dataset_dir=..., epochs=E, batch_size=4, grad_accum_steps=4, lr=1e-4, output_dir=..., early_stopping=True)` — effective batch 16 as the RF-DETR docs recommend for a T4-class GPU. **Epochs `E` by dataset size, per the RF-DETR docs:** < 500 images 100–200; 500–2,000 → 50–100; 2,000–10,000 → 30–50; use `lr = 5e-5` below ~1,000 images; early stopping on validation mAP. Seed everything. Take the best-EMA checkpoint. **Environment (added 2026-09-28):** the official RF-DETR Colab notebook trains Nano on a T4 with batch 8 × accumulation 2, which is the same effective batch of 16 as 4 × 4. Before launching, run a 5-minute smoke test (forward and backward pass) on the assigned accelerator: a March 2026 report says Kaggle's default PyTorch build lacked P100 kernels, with conflicting comments on T4 ([issue](https://github.com/Kaggle/docker-python/issues/1546)); its status today is unverified. Pin the `rfdetr` version in the `train` group and note it in the P1.5 `DECISION` entry (release 1.10.0 changed validation to report the EMA model, so numbers are not comparable across versions). Save a checkpoint every epoch to persistent storage, because a free session can end early. A run of the size expected here is roughly tens of minutes on a T4 (an estimate extrapolated from the RF-DETR cookbook's 50 epochs on ~1,500 images in 10–25 minutes for a different model size). Keep the RF-DETR default augmentations first; add extras only if `val` shows overfitting. At 2,000+ images the RF-DETR docs recommend their aggressive preset; treat it as an experiment against the default on `val`, and check that any colour or hue jitter in it does not blur red versus yellow (my reasoning; verify on `val`). *(Domain note: in microgravity there is no fixed "up", so orientation-augmentation is defensible if the detector fails on rotated objects.)* Copy the checkpoint to `weights/detector.pth` and write `weights/MANIFEST.json`.

**Stage 7 — Evaluate** (`training/eval_detector.py` → `reports/detector_eval.json`). Per-class precision/recall at `detector_conf_floor` and COCO mAP on `val` and `test`. Acceptance recall thresholds live in `config/acceptance.yaml`. **Report every metric twice:** on all labelled frames (agreement with the auto-labeler) and on the **verified gold subset** (Stage 4b). The gold-subset numbers are the ones checked against `acceptance.yaml` and quoted in the PPT.

**Stage 8 — CPU benchmark** (`training/benchmark_cpu.py` → `reports/benchmark_cpu.json`). On the team laptop: 20 warm-up + 200 timed frames of the **full per-frame path** (BGR→RGB, predict, hands), for PyTorch(+`optimize_for_inference`), ONNX Runtime, and YOLO11n; report mean and p95 latency, fps, thread count and CPU model. Decides the detector (`ISSUES.md`), the achievable `target_fps`, and whether pose can be enabled.

**Stage 9 — Cache** (`python -m perception.cache build --runs runs --out data/cache`, P1.7). Run the **real** `PerceptionPipeline` over each run video **decimated to `RuntimeConfig.target_fps`** (`DecimatedSource`, original `frame_id` kept), for **all splits**. Line 1 is a `PerceptionCacheHeader` (`fps` = the cache rate, `model_stamp`); every later line a `PerceptionFrame`. Rebuild whenever the `model_stamp` changes. Size is small (~1.7 MB per 90 s run in the worst case).

**Stage 10 — Tune** (P2.6, `val` caches only) and **Stage 11 — Test** (P2.7, `test` caches, exactly once per model version). The numbers in `reports/` feed the PPT.

**Reproducibility:** every stage takes explicit input/output paths, fixed seeds and writes a JSON report; rerunning a stage on the same inputs gives the same outputs.
**Done when:** [Plan §7.4, F14]. **Why auto-labeling instead of manual annotation:** [context.md §5](./context.md#5-dataset-strategy) and [§7](./context.md#7-the-sequence-engine-the-projects-core-logic).
