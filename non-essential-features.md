# Non-essential features (Tier 2 — deferred, and explicitly excluded)

This document covers two different categories, kept distinct:

- **Deferred (Tier 2):** legitimate features, worth attempting only if time remains after every [essential feature](./essential-features.md) is working, or worth naming as roadmap for a future, higher-TRL version of the system.
- **Explicitly excluded:** things considered and consciously left out of scope entirely — not "if time allows," but "not planned, and here's why."

For the full evidence/reasoning behind each decision, see [`context.md`](./context.md).

---

## Deferred (Tier 2) — attempt only after every essential feature works

### 1. Orientation-agnostic 3D Human Mesh Recovery (HMR)

**What:** the problem statement's own optional stretch requirement — pose estimation relative to the payload rack rather than a floor plane, since microgravity removes any fixed "up."

**If attempted:** render a small synthetic set of SMPL-X bodies at randomized orientations relative to a virtual rack plane, and fine-tune a lightweight HMR head on that synthetic set — adapting the same methodology the [BEDLAM dataset](https://arxiv.org/html/2306.16940v1) used to prove that networks trained purely on synthetic data can achieve state-of-the-art 3D pose accuracy on real images.

**Why deferred, not core:** explicitly marked optional in the problem statement itself; no public microgravity RGB+SMPL dataset exists to shortcut this (real research in the space uses IMU suits or Kinect motion capture during parabolic flights, not usable for this purpose). Full reasoning: [context.md, optional-task discussion](./context.md#6-perception-model-choices).

### 2. Full confidence calibration

**What:** formal calibration curves for the object detector and hand tracker's confidence outputs, beyond the basic threshold in the [essential reliability layer](./essential-features.md#13-reliability-layer).

**Why deferred:** the basic threshold already prevents the worst failure mode (acting on a low-confidence detection); calibration is a refinement, not a requirement, given the timeline. See [context.md §8](./context.md#8-reliability-engineering).

### 3. Cross-modal redundancy voting (formalized)

**What:** requiring explicit agreement between the object detector and hand tracker as a distinct, separately-implemented voting step, beyond what temporal hysteresis already provides.

**Why deferred:** a hardening refinement on top of the essential reliability layer, not a functional requirement for the demo.

### 4. Process watchdog / supervision

**What:** automatic restart-on-crash for the runtime pipeline, rather than a manual restart during the demo.

**Why deferred:** manual restart is acceptable for a scripted, team-run demo; automated supervision matters more for a genuinely unattended, real deployment.

### 5. Model integrity checksum

**What:** *verifying* a `hashlib` sha256 at model load time — refusing to run if the loaded weights file does not match the value recorded in `weights/MANIFEST.json`.

**Already done in Tier 1:** the sha256 prefixes of the detector and MediaPipe files are computed anyway, because they form `model_stamp`, which decides whether a perception cache is still valid (Plan §5.2). Only the *refuse-to-run-on-mismatch check at load* is deferred.

**Why deferred:** zero functional impact on the demo itself; good practice to mention as designed-for, but not worth build time this week.

### 6. Multi-experiment generalization

**What:** supporting more than one experiment definition (i.e., a GUI or config system for authoring and switching between multiple canonical step-lists), rather than the single, self-authored experiment this round targets.

**Why deferred:** the problem statement only requires demonstrating one defined experiment; generalizing the sequence engine's template format to arbitrary experiments is real, legitimate future work, worth naming explicitly as a roadmap item rather than attempting now.

**Round scope (locked 2026-09-28):** Sample Transfer only. The first candidate after G5 is a second experiment, "photograph and cold-stow" (a sample dish leaves stowage, sits in an observation tray, a hand touches a CAPTURE control, the dish goes into a cold-stow box), which mirrors the Ax-4 photograph-then-freezer task described in `context.md` §5. It needs **one new container class**, so a short top-up recording and a refit, plus its own golden set. A variant that only reorders steps with the same five classes needs no new detector data but still needs its own golden set. The engine stays linear, single-valid-order. Prefer generic class names so later experiments can reuse the detector; zero-shot detection is not a runtime substitute, because it works well only when class names are semantically meaningful ([RF100-VL](https://arxiv.org/pdf/2505.20612)).

### 7. Report generation

**What:** a formatted, human-readable summary (potentially PDF) generated from the structured log, distinct from the raw log itself.

**Why deferred but cheap to plan for:** this is pure post-processing — it reads the already-logged data after the fact and touches none of the real-time, reliability-critical pipeline, so it carries none of the risk a Tier 1 feature would. Recommendation: design the log schema well now (already done as part of [Feature 9](./essential-features.md#9-structured-log)) so that building the report generator later, if time allows, is a same-day task.

---

### 8. TTS quality upgrade (pre-rendered phrase cache)

**What:** replace the operating system's robotic voice with a higher-quality offline voice **without adding a runtime model**. Every utterance the system speaks comes from a closed, finite set — each step's `say` phrase, the alert templates × the step display names, and the completion phrases — so the set can be synthesized once, at build time, with a better offline voice (Kokoro-82M, Apache-2.0, is the cleaner-licensed candidate; Piper's maintained `piper1-gpl` is GPL-3.0) and saved as WAV files. At runtime `TTSWorker` plays a cached WAV keyed by a hash of the text, and falls back to live `pyttsx3` for anything not cached. This also removes runtime synthesis latency and the pyttsx3 hang risk for all cached phrases.

**Cost:** a small playback dependency (chosen and license-checked under R7), a build script that walks `config/experiment.json` and the alert templates, and the check that transitive licenses (e.g. phonemizers) are acceptable.

**Why deferred:** the OS voice already meets the requirement (a voice-based alert); this is polish for the demo, worth doing only after G5.

---

## Explicitly excluded — not planned at all, with reasoning

### 1. Manual sign-off / attestation

**What was considered:** requiring the astronaut to explicitly confirm (button press, verbal confirmation, or literal signature) that each step was completed, as a stronger audit-trail mechanism.

**Why excluded:** this would directly contradict the point of the problem statement, which is that the vision system observes and validates *automatically* so the astronaut does not have to manually track and report progress. A confirmation requirement after every step reintroduces exactly the manual overhead the system exists to remove. The `confirmed` / `flagged_uncertain` tagging already built into [the structured log](./essential-features.md#9-structured-log) achieves the same audit-trail purpose — routing what the system itself isn't sure about to human attention — without the contradiction. See [context.md §8](./context.md#8-reliability-engineering) for the underlying "outer loop" reasoning this is based on.

### 2. Person / crew identification

**What was considered:** identifying which specific crew member is performing the procedure (e.g., via face recognition), logged alongside the procedure steps.

**Why excluded, not even as a stretch goal:**
- Not asked for — the problem statement is scoped entirely to the procedure (steps, objects, sequence), never to who performs it.
- A materially larger scope addition than it appears: person recognition is an entirely separate computer-vision subsystem, with its own failure modes, made worse in microgravity where bodies aren't reliably upright or facing the camera.
- A different, more sensitive category of concern than anything else in this system: biometric identification of a specific named individual would require data governance, consent, and security handling this project has not scoped for at all.
- Very likely redundant: mission control already knows who is in the module at any given time through crew scheduling, so this would not fill an actual unmet need for this system to solve.

### 3. Dedicated GUI pages for dataset preparation, training, and experiment authoring

**What was considered:** separate frontend pages for recording/labeling data, running training, and authoring new experiment templates through the GUI, alongside the live monitoring dashboard.

**Why excluded:** these are one-time or rare workflows the team runs on itself, not something demoed live to judges or used repeatedly by an end operator — command-line scripts and a README serve this better and faster than a GUI would. Building GUI pages here is effort spent on something nobody scoring the submission will interact with, at the direct expense of hardening the actual judged, live-demo-facing dashboard.

**Clarification (2026-09-28):** the label editor (`training/label_editor/`) is a dev-time, static, offline HTML page the crew uses while preparing data. It is not a page of the product GUI, is not served by `server/`, is not part of the demo, and does not reopen this exclusion. This follows the "one screen, one decision" HMI design principle applied to the dashboard itself — see [context.md §13](./context.md#13-gui--page-structure).

### 4. FastAPI (as the backend framework)

**What was considered:** using FastAPI instead of Flask for the backend, given its documented throughput advantage in benchmarks.

**Why excluded:** that throughput advantage is specific to high-concurrency, I/O-bound workloads (many simultaneous clients), which this single-camera, single-operator system does not have. More importantly, naively running this project's CPU-bound inference inside a FastAPI async route without correctly offloading it would block the entire event loop — a real, documented pitfall, and added risk for a team without deep async experience under a one-week deadline. Full reasoning: [context.md §12](./context.md#12-backend-and-frontend-framework-decisions).

### 5. React (as the frontend framework)

**What was considered:** building the GUI dashboard in React instead of plain HTML/CSS/JS.

**Why excluded:** React's value proposition — efficient management of complex, deeply-nested, frequently-changing component trees — does not apply to a dashboard with four pieces of state, one of which (the video feed) isn't even React-rendered. Vanilla JS is the evidence-backed recommendation specifically for simple projects and low-dependency, embedded/low-bandwidth contexts, which matches both this project's scope and its own deployment description. Full reasoning: [context.md §12](./context.md#12-backend-and-frontend-framework-decisions).
