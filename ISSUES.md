# ISSUES.md

Append-only. One log for everything that needs the other person's attention. Read it at the start and the end of every session (IMPLEMENTATION_PLAN.md Part 11). Never edit or delete an existing entry; resolve by appending a new line under it.

Three entry types:

- `CONTRACT` — a gap in `contracts.py` or `config/experiment.json` (Part 6). Agents never edit those files; log here, use a `TEMP_<n>` stub, keep working. (`contracts.py`'s header says `CONTRACT_CHANGES.md`; read that as this file.)
- `BLOCKER` — genuinely stuck, beyond a contract gap.
- `DECISION` — a choice that must be recorded (e.g. which detector).

Entry format:

```
## [date] [P1|P2] [TYPE] - short title
- Missing/blocked/decision: ...
- Why it matters: ...
- Proposed fix / options / what was tried: ...
- Status: OPEN
```

---

## 2026-09-28 setup DECISION - detector: RF-DETR-Nano vs YOLO11n (open until P1.5)
- Decision needed: which detector ships. RF-DETR-Nano is the primary (Apache-2.0, evidenced for limited custom data, context.md section 6); YOLO11n is the benchmarked fallback.
- Why it matters: every published RF-DETR latency figure is GPU/TensorRT. No CPU-only number was found, and the team has no GPU. YOLO (Ultralytics) is AGPL-3.0, which matters if this ever leaves the hackathon.
- Proposed: P1.5 runs `training/benchmark_cpu.py` on the team laptop's CPU, chooses, and records the result here. If YOLO wins, also record the license implication.
- Status: OPEN
- Update 2026-09-28: proposed decision rule (both people ack at G0). The detector is chosen once at P1.5 and stamped; the runtime never switches detectors automatically. YOLO11n is used only if, after one fix cycle, RF-DETR-Nano (a) cannot reach `min_pipeline_fps` on the demo laptop under PyTorch or ONNX Runtime (ONNX Runtime first), (b) misses the `acceptance.yaml` recall on `val`, or (c) cannot be fine-tuned on any available GPU. See context.md section 6.

## 2026-09-28 setup DECISION - experiment is a DRAFT until P1.2 closes (G2)
- Decision needed: the final `config/experiment.json` (classes and `when` rules).
- Why it matters: contracts.py prefers position rules over fine appearance classes; the feasibility spike (P1.2) decides what is separable. Every step must begin false or it can never fire (plan 5.3).
- Proposed: draft at G0; P1.2 proposes the final version through a contract PR.
- Status: OPEN
- Update 2026-09-28 (rule wording, see the CONTRACT/DECISION entry at the end of this file): "every step must begin false" above is superseded. The two stow steps start true and are latched, which the tracker handles; the requirement is that each step's rule can go false before its turn.
- Update 2026-09-28: a concrete draft now exists at `config/experiment.json` ("Sample Transfer": 5 classes, 7 position-rule steps) and validates against `contracts.py`. A reference tracker + engine run on a synthetic world produced the intended event sequence for all 10 planned variants. Rows 1-12 of `runs/run_plan.csv` are the pilot; the crew waits for G2 before recording rows 13-77.

## 2026-09-28 setup DECISION - licenses of auto-labeling tools not yet verified
- Decision needed: confirm the license of YOLO-World / Grounding DINO before either is added to the lockfile (R7).
- Why it matters: they run offline and only their outputs (labels) are data, but this is unchecked.
- Proposed: P1.2 records each tool's license here when first used.
- Status: OPEN

## 2026-09-28 setup DECISION - MediaPipe model files must be vendored
- Decision needed: place the hand and pose model files under `weights/` (sha256 in `weights/MANIFEST.json`).
- Why it matters: the runtime must download nothing (offline requirement); the sha256 prefixes are also part of `model_stamp`.
- Proposed: P1.3.
- Status: OPEN

---

## 2026-09-28 setup CONTRACT - non-finite numbers are accepted in four fields
- Missing: `_finite` validation on `StateEvent.t`, `EngineEvent.t`, `LogEntry.t_video` and `RunScript.fps`. Probe result: `+inf` / `NaN` are accepted there, while `PerceptionFrame.t`, `Detection.box` already reject them.
- Why it matters: contracts.py's own convention is "reject a literal NaN/inf once, at the boundary".
- Proposed fix at G0: apply the existing `_finite` helper to those four fields (additive, no field changes).
- Status: RESOLVED at G0 -- `contracts.py` applies `FiniteFloat` (the existing `_finite` validator) to `StateEvent.t`, `EngineEvent.t`, `LogEntry.t_video` and `RunScript.fps`. Regression tests: `tests/unit/contracts/test_config_shapes.py`.

## 2026-09-28 setup CONTRACT - a typo in a YAML config key is silently ignored
- Missing: `extra="forbid"` on `PerceptionConfig` and `RuntimeConfig`. Probe result: `hysteresis_frmes: 7` loads without error and the default (5) is used.
- Why it matters: the whole tuning step (P2.6) writes `config/perception.yaml`; a misspelled key would quietly revert a tuned threshold to its default and nobody would notice.
- Proposed fix at G0: `ConfigDict(frozen=True, extra="forbid")` on both models.
- Status: RESOLVED at G0 -- both `PerceptionConfig` and `RuntimeConfig` in `contracts.py` use `ConfigDict(frozen=True, extra="forbid")`. Regression tests: `tests/unit/contracts/test_config_shapes.py` (`hysteresis_frmes` / `target_fp` typos are rejected).

## 2026-09-28 setup CONTRACT - `LogEntry.t_wall` is required but is stamped by the logger
- Missing: a way to construct a `LogEntry` before the logger stamps `t_wall`.
- Why it matters: contracts.py says wall time exists only in the logger, yet the field is mandatory at construction. Current workaround (documented in Plan 5.5): the router builds entries with `datetime(1970,1,1,tzinfo=utc)` and the logger overwrites it.
- Proposed fix at G0: `t_wall: datetime | None = None`, keeping the timezone-aware check when set; the logger always sets it.
- Status: RESOLVED at G0 -- `LogEntry.t_wall: datetime | None = None` in `contracts.py`, with a validator that still rejects a naive (non-tz-aware) value when set. The router builds entries with `t_wall=None`; `JsonlLogger` sets it via `model_copy(update=...)`. Regression tests: `tests/unit/contracts/test_config_shapes.py`.

## 2026-09-28 setup CONTRACT - the small file seams are not typed
- Missing: Pydantic models for `config/acceptance.yaml` and the `reports/*.json` shapes (currently specified only as a table in Plan 5.8).
- Why it matters: they are written by one person and read by another (P1 writes acceptance thresholds; P2 reads them; reports feed the PPT).
- Proposed fix at G0: `AcceptanceConfig` (frozen, `extra="forbid"`) and a small `ReportHeader` (`generated_at`, `model_stamp`, input stamps). Report bodies may stay dicts.
- Status: RESOLVED at G0 -- `contracts.py` adds `AcceptanceConfig` (with `DetectorAcceptance`/`PipelineAcceptance`/`ReplayAcceptance`/`LabelReviewAcceptance`, all frozen + `extra="forbid"`) and `ReportHeader` (`generated_at` tz-aware, `model_stamp`, `input_stamps`). Report bodies stay dicts as proposed. Regression tests: `tests/unit/contracts/test_config_shapes.py`.

## 2026-09-28 setup DECISION - operating system of the two dev laptops and the demo laptop
- Decision needed: Windows / macOS / Linux for each machine.
- Why it matters: TTS backend (SAPI5 / NSSpeechSynthesizer / eSpeak), webcam backend (`CAP_DSHOW` on Windows), `multiprocessing` start method, and whether shell scripts would even run. The plan already avoids bare shell scripts, but the demo machine's voice and camera must be tested on that exact machine.
- Proposed: record the answer at G0; run the audio self-test and camera check on every machine.
- Status: OPEN
- Update 2026-09-28 (G0): both dev laptops (P1, P2) and the demo laptop are Windows. Webcam backend: `CAP_DSHOW`. TTS backend: SAPI5.
- Update 2026-09-28 (P2's machine): audio self-test passed -- `pyttsx3.init()` found 3 SAPI5 voices (Microsoft David/Hazel/Zira, en-US/en-GB), `say()` + `runAndWait()` returned without raising. Camera check passed -- `cv2.VideoCapture(0, cv2.CAP_DSHOW)` opened, `read()` returned a `(480, 640, 3)` frame, but `CAP_PROP_FPS` reported `0.0` -- this confirms F1's documented fallback (measure the median inter-frame interval over the first 30 frames) is required on this hardware, not optional. Both checks were run as raw capability probes (`cv2`/`pyttsx3` directly), not through `perception.camera.open_source`, which does not exist until P1.1. P1's machine still needs to run both checks and add its own line here before G0 can close (see the sign-off entry at the end of this file).

## 2026-09-28 setup DECISION - where the detector is fine-tuned (GPU access)
- Decision needed: who has a GPU (Colab or Kaggle notebook, a lab machine) and who runs stage 6 of the dataset pipeline.
- Why it matters: "CPU laptop only" is a deployment constraint; fine-tuning RF-DETR is a GPU job (its docs are written for T4/A100-class GPUs). No GPU means the fallback is a slow CPU fine-tune of YOLO11n, which is AGPL.
- Proposed: settle at G0; keep the fine-tune as a background script, not an open agent session.
- Status: OPEN
- Update 2026-09-28 (G0): P1 has Colab/Kaggle access and runs the fine-tune (P1.5, background script). P2 does not currently have GPU access; if P1 is blocked at G3, P2's fallback duties per IMPLEMENTATION_PLAN.md Part 2 do not include training code, so a GPU-access gap would need re-discussing then.
- Update 2026-09-28: plan is Kaggle first, Colab as backup. A March 2026 GitHub issue (Kaggle/docker-python #1546) reports Kaggle's default PyTorch build lacks P100 (sm_60) kernels; comments disagree on whether T4 is also affected, and today's status is unverified. Action: whoever trains runs a 5-minute forward-and-backward smoke test on the assigned accelerator at G0 and records the result here. Also pin the `rfdetr` version and save checkpoints every epoch to persistent storage.

## 2026-09-28 setup DECISION - TTS: pyttsx3 in a worker process; measure interruption at P2.3
- Decision: Tier 1 uses `pyttsx3` (OS voice, offline, nothing to vendor) in a separate process; an `alert` interrupts by restarting the worker. Piper's maintained `piper1-gpl` is GPL-3.0; Kokoro-82M is Apache-2.0. A pre-rendered phrase cache is Tier 2 (non-essential-features #8).
- Open: P2.3 measures worker-restart time and interruption reliability on the real machine and records the numbers here.
- Status: OPEN (decision made, measurement pending)
- Update 2026-09-28 (P2.3, P2's machine, Windows/SAPI5): built `outputs/tts.py` (`TTSWorker`, `FakeSpeaker`) and `outputs/logger.py` (`JsonlLogger`) per 5.5. Measured with `scripts/measure_tts_restart.py` (real `pyttsx3` backend, not the null stub): **cold start** (construction -> first worker ready, i.e. `multiprocessing` spawn + `pyttsx3.init()` on this machine) is 0.58-0.70s over 3 trials. **Warm restart** (an `alert` `say()` call -> the *new* worker signals ready, while the old one is mid-utterance) is 0.49-0.68s over 5 trials -- `say()` itself does not block on this (it only blocks on `terminate()`+bounded `join(timeout=1.0)`+`Process()`+`start()`, none of which wait for the child's `pyttsx3.init()`); the ready-wait is only used by the measurement script, never by `say()`. **Interruption reliability**: in all 5 trials the pre-empted process was confirmed dead (`is_alive() is False`) within 100ms of the restart call -- `terminate()` on this machine's SAPI5 backend reliably kills a mid-`runAndWait()` process. Numbers are per-machine (spawn + COM init cost varies); re-run the script on the demo laptop before the final gate and append a line here. `t_wall`'s optional-at-construction / placeholder-overwrite convention (the earlier CONTRACT resolution in this file) is implemented exactly as specified in `JsonlLogger.write`.
- Status: DECIDED and measured on P2's machine; demo-laptop re-measurement still pending.

## 2026-09-28 setup DECISION - processing rate: target_fps and the 8 fps floor
- Decision needed: the achievable `RuntimeConfig.target_fps` (default 15) on the demo laptop.
- Why it matters: hysteresis is counted in frames. Simulation of the sample experiment: 7/7 steps fired at 15, 10 and 6 fps, but only 5/7 at 4 fps. `min_pipeline_fps = 8` is a judgment call in `config/acceptance.yaml`.
- Proposed: P1.5 benchmark sets `target_fps`; caches are built at that rate; P2.6 tunes `hysteresis_frames` at that rate.
- Status: OPEN

## 2026-09-28 setup DECISION - pose tracking on or off
- Decision needed: whether `enable_pose` is on in the demo. No Tier-1 rule consumes pose, and it costs CPU.
- Proposed: default off; enable only if the P1.6 benchmark shows headroom above `min_pipeline_fps`.
- Status: OPEN

## 2026-09-28 setup DECISION - label-correction tool
- Decision needed: whether hand-correcting bad labels needs a tool (CVAT, Label Studio) or re-running the auto-labeler with better prompts is enough.
- Why it matters: license and setup cost (R7). The review gate (at most 10 percent bad per class) decides.
- Proposed: decide at P1.4 only if a class fails the gate once.
- Status: OPEN
- Update 2026-09-28: decided; see the DECISION entry at the end of this file (a minimal static HTML editor, not Label Studio or CVAT).

## 2026-09-28 setup DECISION - run-plan sizes are judgment calls
- Decision: 77 runs (36 train / 14 val / 27 test), operators O1-O4 with O4 in test only, setups S1/S2. No source gives the right number; the sizing targets are in context.md section 5.
- Proposed: if the detector or the replay goldens under-perform, add `correct` runs first.
- Status: OPEN (revisit after P1.5)


## 2026-09-28 setup DECISION - scope: Sample Transfer is the only experiment this round
- Decision: one experiment (E1, Sample Transfer) for the SIH prototype and PPT. A second experiment is Tier 2 after G5 (non-essential-features.md section 6).
- Why it matters: the pilot gates 65 of the 77 recordings (G2), and the schedule is one week.
- Proposed: candidate E2 is "photograph and cold-stow" (one new container class, top-up recording, refit, own golden set).
- Status: DECIDED

## 2026-09-28 setup CONTRACT - experiment lint and the "begin false" rule need rewording
- Missing: `IMPLEMENTATION_PLAN.md` 5.3(6b) and the lint say every step must begin false / no step's `when` may be true in the baseline window. Sample Transfer's `red_stowed` and `yellow_stowed` ("inside outer_box") are true at baseline because both boxes start in the outer box, so the lint as written would reject the draft. The tracker already latches such steps and re-arms them after `release_frames` false frames.
- Why it matters: without a fix, either the lint fails on the draft or someone changes the experiment to satisfy a rule that is stricter than the mechanism needs.
- Proposed fix at G0: a step must be able to go false before its turn. Lint: a step true in a baseline window must go false for at least `release_frames` consecutive frames before its turn in every `correct` run. Add a P2.1 test that a true-at-baseline step fires after a leave-and-return. Confirm against contracts.py and the tracker at G0 (contracts.py was not available when this was written).
- Status: OPEN (wording confirmed at G0, dynamic check deferred) -- `contracts.py` now exists; `red_stowed`/`yellow_stowed` are ordinary `inside(label, outer_box)` rules with no special-casing needed in the schema (the "begin false" property is a runtime/tracker behavior, not a shape constraint). `tests/unit/contracts/test_experiment_lint.py` covers the static half now checkable from `config/experiment.json` alone (snake_case + uniqueness, every rule's `label`/`container` in `classes`, every `say` <= `MAX_SPOKEN_WORDS`). The dynamic half (a step true at baseline must go false for `release_frames` before its turn, checked against recorded/cached runs) stays deferred to P2.6 as this entry proposed, since it needs perception caches that do not exist yet. P2.1 should still add the leave-and-return unit test on hand-built `PerceptionFrame`s per the proposal above.

## 2026-09-28 setup DECISION - run count: extra-run stop rule
- Decision: 77 runs stay the baseline (36 train / 14 val / 27 test). No source gives the right number (context.md section 5). After the first training pass, fine-tune on about 50% of the train runs; if val mAP and the replay-golden pass rate barely move, do not add extra `correct` runs.
- Why it matters: it turns "add more runs if it under-performs" into a measurable check. It cannot shrink val or test, which are set by what evaluation needs.
- Proposed: P1.5 runs the check and records the numbers here.
- Status: OPEN

## 2026-09-28 setup DECISION - label correction: minimal static HTML editor, overlay file, gold subset
- Decision: (1) build a small static, offline HTML label editor (`training/label_editor/`); Label Studio and CVAT are not adopted. (2) Corrections live in an overlay (`data/corrections/<split>.json`, Plan 5.8), applied by `build_dataset`; auto-labels are never edited. (3) Static-object boxes are checked and fixed once per run. (4) A gold subset of val and test frames (default 6 per run, a judgment call) is hand-verified; detector metrics for `acceptance.yaml` and the PPT are reported on it and also on all frames. (5) Uncertain train frames are excluded, not corrected. (6) P1.2 runs Stages 2-4 on the pilot frames and records the per-class bad fraction.
- Why it matters: about 10,000-15,000 train boxes cannot be hand-fixed and the review only samples about 40 frames per class; without a gold subset, detector metrics measure agreement with the auto-labeler, and label errors in evaluation sets distort conclusions (Northcutt et al., NeurIPS 2021). The headline step-detection numbers are scored against the crew's declarations and are unaffected. Label Studio (Apache-2.0) and CVAT (MIT) are heavy for a few hundred edits and add a class-id conversion step.
- Proposed: owner is one Comms-crew teammate with an agent, reviewed by P1 (C5 in Part 10), needed before C3; confirm at G0 and add `training/label_editor/` to Part 4 ownership (already drawn in the tree as proposed). The overlay format is a proposal that P1 may refine via an ISSUES entry before P1.4. Timebox one agent session (an estimate, not sourced).
- Status: DECIDED (owner and overlay format to confirm at G0)

## 2026-09-28 setup DECISION - label editor: browser file access to verify
- Decision needed: confirm on the crew's actual browsers that the inlined static page loads its local images and that the Download button works. Browsers typically block fetch() of local files, which is why the data is inlined (unverified here).
- Why it matters: if it fails, fall back to serving the page from a loopback-only Python server with a one-time token (which then needs a decision under AGENTS rule 14).
- Proposed: check in C5 on each crew member's browser.
- Status: OPEN

## 2026-09-28 setup DECISION - the closed WhenRule vocabulary (five kinds)
- Decision needed: `contracts.py` did not exist before this G0 session, so the exact set of `WhenRule` kinds referenced by IMPLEMENTATION_PLAN.md 7.4 ("all five rule kinds") had to be designed, not just read, at G0.
- Why it matters: AGENTS.md rule 10 closes the vocabulary once set -- nobody may invent a sixth kind later without a contract PR.
- Decided: five kinds, all pure functions of a floor-filtered `PerceptionFrame` (no learned interaction model, per F4): `inside(label, container)` (both must be detected; true iff the label's box centroid lies inside the container's box), `outside(label, container)` (label must be detected; true if not inside container, and vacuously true if container is not detected at all -- the "missing container" case), `hand_touching(label)` (label's box grown by `touch_margin_frac`; true if any of the 21 landmarks of any hand lies inside it), `absent(label)` (no detection of label at or above `detector_conf_floor` this frame), `present(label)` (the complement of `absent`). The draft `config/experiment.json` only exercises `inside`/`outside`/`hand_touching`; `absent`/`present` are available for P1.2's final experiment revision.
- Status: DECIDED at G0. Evaluation logic (not the shape) lives in `state/tracker.py`, built at P2.1.

## 2026-09-28 setup CONTRACT - CONTRACT_CHANGES.md docstring reference
- Missing: `contracts.py`'s header pointed at a `CONTRACT_CHANGES.md` file that was never created; IMPLEMENTATION_PLAN.md Part 6 says to read that as this file (`ISSUES.md`) and "fix that docstring in the G0 contract PR."
- Why it matters: a stale pointer sends the next reader looking for a file that doesn't exist.
- Status: RESOLVED at G0 -- `contracts.py`'s module docstring now names `ISSUES.md` directly.

---

## 2026-09-28 P2.1 DECISION - StateEvent.confidence default when a step's rules reference no detection
- Decision needed: IMPLEMENTATION_PLAN.md 5.3.4 defines `confidence` as "the minimum over the hysteresis window of the conf of the best detections the step's rules reference." A step built only from `absent(label)` rules never references a detection at all (by definition, `absent` is true exactly when there is none) -- essential-features.md and the plan do not say what the value should be in that case.
- Why it matters: `StateEvent.confidence` is a required `Unit` field (contracts.py); `state/tracker.py` (P2.1) must return something, not raise.
- Decided: `state/tracker.py` defaults `confidence = 1.0` (and therefore `uncertain = False`) when the hysteresis window collects zero referenced confidences. Rationale: nothing observed contradicts the step firing, so there is nothing to be uncertain about. Regression test: `tests/unit/state/test_tracker.py::test_confidence_defaults_to_one_when_no_detection_referenced`. `config/experiment.json`'s Sample Transfer steps are all `inside`/`outside`/`hand_touching`, so this default is not exercised by the current experiment -- flagging for P1's ack in case a future revision adds an `absent`-only step.
- Status: DECIDED (P2, open for P1 ack)

## 2026-09-28 G0 sign-off
- P2 (this machine, Windows): `uv sync --group dev` installs cleanly (271 packages resolved, `uv.lock` committed); `python scripts/check.py --quick` is green (ruff clean, 31 passed / 1 skipped -- the one skip is the RunScript-integrity test, which has nothing to check until runs exist); git hooks installed and verified live (pre-commit and pre-push both fired and passed on the scaffold commit); audio self-test and camera check both passed (see the OS DECISION entry above for details). `contracts.py` authored (it did not exist before this session) and reviewed against every reference to it in `IMPLEMENTATION_PLAN.md` and `essential-features.md`; the four open CONTRACT entries above are applied and resolved; the WhenRule vocabulary is decided. `develop` and `main` created and pushed; the G0 scaffold is on `develop` (commit `d8d4a66`).
- P1 sign-off: PENDING. Per AGENTS.md, "nothing closes until BOTH people say yes in this sitting" -- P1 still needs to: read `contracts.py` line by line and ack (or raise) any changes; review the draft `config/experiment.json`; run `uv sync --group dev` and `python scripts/check.py --quick` green on their own machine; run their own audio self-test and camera check; add their own sign-off line here.
- Status: OPEN -- do not tag `g0` until P1 adds their sign-off line below this one.
- P1 sign-off (2026-09-28, Windows): Read `contracts.py` line by line (812 lines). All four open CONTRACT entries confirmed applied and correct: `_finite` on `StateEvent.t`, `EngineEvent.t`, `LogEntry.t_video`, `RunScript.fps`, `PerceptionCacheHeader.fps`; `extra="forbid"` on `PerceptionConfig` and `RuntimeConfig`; `LogEntry.t_wall: datetime | None = None` (tz-aware validator only fires when set); `AcceptanceConfig` and `ReportHeader` added in Section 10. Module docstring now names `ISSUES.md` (not `CONTRACT_CHANGES.md`) -- confirmed resolved. WhenRule five-kind vocabulary (`inside`, `outside`, `hand_touching`, `absent`, `present`) confirmed decided and frozen. Reviewed `config/experiment.json` DRAFT (5 classes, 7 steps, all position rules): confirmed that `red_stowed` and `yellow_stowed` start true at baseline, are latched by the tracker, and each goes false before its canonical turn -- `red_stowed`'s rule (`inside(red_box, outer_box)`) goes false when the operator removes `red_box` at step 1 (`red_out`); `yellow_stowed`'s rule goes false at step 3 (`yellow_out`) -- both are therefore re-armed well before their turn at steps 6 and 7. All five initially-false steps confirmed false at baseline. Reviewed and ack'd all open DECISION entries (detector, processing rate, TTS, pose, OS, GPU access, label editor, run count, label editor browser check, WhenRule vocab). `uv sync --group dev` ran (installing lockfile); `python scripts/check.py --quick` green (31 passed, 1 skipped -- RunScript-integrity skip is expected with no recorded runs). Audio self-test and camera check confirmed passing on this machine.

---

## 2026-09-29 P1 DECISION - defer P1.2's real-footage check while building content-independent plumbing ahead of schedule
- Decision needed: P1 (this track) proposed not waiting for real crew recordings (pilot rows 1-12, or the 77-run set) before continuing to build the rest of the pipeline, since G1's "pilot recorded" condition and G2's "experiment frozen" condition currently sit in front of a lot of work that doesn't actually depend on footage content.
- Agreed, with one specific disagreement on scope, not principle:
  1. **Agree in general.** The plan already treats this as safe practice for P2: G1 explicitly lets P2.1-P2.5 be "done against fakes" with no real perception at all, and `state/tracker.py` + `engine/sequence.py`'s correctness is *already* verified only symbolically ("Verified in simulation, not just argued", context.md section 5) -- real footage was never going to add anything to that verification, only to the *content* decisions (which classes, which rules, which detector). Deferring content decisions while building shape/plumbing is exactly the R6 precedent (build against your own fake when two things would otherwise wait on each other), applied to P1 instead of just P2.
  2. **Disagree on one point: don't defer recording the pilot itself.** DATA_COLLECTION.md section 4 already sequences the pilot (rows 1-12, about 25 minutes) before the expensive 77-run recording specifically so a wrong experiment design is caught cheaply. The pilot rows already include the exact risk cases P1.2 exists to catch -- `011-robustness-gloves` (does MediaPipe survive gloves, essential-features.md section 3's own named pitfall) and the START button close-up cases. Recording it is Data-crew time, not P1 engineering time, and does not block anything P1 is doing -- there is no real either/or here. Recommendation: **record the pilot now, in parallel; defer only P1's own spike-analysis session (the full P1.2: stages 2-4 on pilot frames, the auto-labeler license/accuracy comparison, the formal `config/experiment.json` contract PR).**
  3. **One item on the "can proceed" list needs its wording fixed, not its scope:** "P1.6 pipeline and cache writer/reader ... using a stub or COCO-pretrained detector" -- essential-features.md F2 step 4 requires the id-to-label mapping be asserted against `weights/MANIFEST.json` and `experiment.classes` *at load*; a COCO-pretrained RF-DETR-Nano's class list contains none of `outer_box/tray/red_box/yellow_box/start_button`, so that assertion would correctly fail. COCO-pretrained weights are right only for the CPU latency benchmark (already its own separate "can proceed" item, since raw forward-pass speed doesn't depend on fine-tuning). P1.6 plumbing needs a **stub only** (a `contracts.Perception` implementation returning fixture `PerceptionFrame`s with labels that already match `experiment.classes`), not a real COCO-pretrained model.
  4. **Everything else on the "can proceed" list is genuinely safe**, including the GPU forward/backward smoke test -- which is unrelated to this decision and already overdue: the 2026-09-28 GPU-access DECISION entry above asked for it "at G0" and no result was ever recorded here. It should happen now regardless of this deferral.
  5. **Tripwire (the plan's existing gates already enforce most of this; restating the two that matter):** the minimal real-footage check -- 3-4 of the already-recorded pilot clips (a `correct` run, `011-robustness-gloves`, one close-up on `start_pressed`), zero-shot detect + MediaPipe, eyeball whether the five classes separate, whether `inside`/`outside` behaves from the real camera angle, whether the tray and outer_box stay non-overlapping in frame (DATA_COLLECTION.md section 2's layout instruction, worth checking on one real frame rather than assumed), whether gloves keep hand landmarks, and whether the ~6cm START button is detectable -- must run and its `DECISION` entry land here **before P1.4 starts** (already true: P1.4's prereq is G2 + all 77 runs validated) and, just as importantly, **before the Data crew is asked to record rows 13-77** (already true per DATA_COLLECTION.md section 4's "record only rows 1-12, validate, tell P1, and wait"). Nothing new needs enforcing here beyond making sure nobody skips straight to recording 13-77 without this check landing first.
  6. **Nothing else in P1's track needs a stronger real-footage gate than the plan already has.** Detector choice, stage 2-4 on real data, `acceptance.yaml` thresholds, hysteresis/`target_fps` tuning and freezing `config/experiment.json` are already correctly on the "must NOT be done yet" list and are already prerequisite-gated (P1.4 through P1.7, and P2.6) behind G2/G3.
- Can proceed now (unchanged from the proposal, with 3 corrected to "stub only"): P1.3 hand-tracking plumbing (test only, not a gloves verdict); the CPU latency benchmark of COCO-pretrained RF-DETR-Nano; F14 stage code (`sample_frames`, `autolabel`, `review_sheet`, `build_dataset`, `finetune.py`) written and tested on fixtures, with the actual YOLO-World-vs-Grounding-DINO choice left open; P1.6 pipeline/cache behind a **stub** `Perception`; the GPU smoke test (overdue independent of this decision).
- Must NOT be done yet (unchanged from the proposal): choosing the auto-labeling tool, running stages 2-4 on real data, writing `acceptance.yaml` thresholds, tuning `hysteresis_frames`/`target_fps` on real output, freezing `config/experiment.json`.
- Status: OPEN -- awaiting P2's ack (this changes the practical order in which G1/G2's conditions are satisfied, even though no individual session's documented prerequisite is violated).

---

## 2026-09-29 P2 review (PR #1, p1-perception -> develop, P1.1) - non-blocking: a wall-clock read in record.py's plan bookkeeping
- Missing/blocked: `perception/record.py::cmd_record` sets `row.recorded = time.strftime("%Y-%m-%d")` when it ticks off a finished run in `runs/run_plan.csv`. `perception/record.py` is not `camera.py`, so this is a literal clock read inside `perception/` outside the one file AGENTS.md rule 8 exempts.
- Why it matters: rule 8 ("no clock reads in perception/ except camera.py ... wall-clock time exists only inside JsonlLogger") exists to keep Frame/StateEvent/EngineEvent/LogEntry timing deterministic and clock-free. This value never reaches that pipeline -- it only stamps a crew-bookkeeping column in `runs/run_plan.csv` (DATA_COLLECTION.md section 7.3's "tick the run"), so it looks like it's outside the rule's intent, but it is inside the rule's literal words, and there's no automated lint catching it (checked: no hook or `scripts/check.py` AST rule for this).
- Proposed fix / what was tried: not fixed by the reviewer (P1's files, not mine to edit -- AGENTS.md rule 2). Either P1 should get an explicit ack that this is an intentional, narrow exception to rule 8 (CSV bookkeeping metadata, not pipeline data), or move the date-stamp responsibility out of `record.py` (e.g. the crew fills `recorded` by hand per DATA_COLLECTION.md 7.3, and the tool only fills `recorded_by`). `time.sleep(1)` in `_countdown()` is not flagged -- it blocks, it doesn't read or record a clock value.
- Status: OPEN (non-blocking; PR #1 merged as-is -- see the sign-off/verdict entry below).

## 2026-09-29 P2 review verdict - PR #1 (p1-perception -> develop, P1.1) merged
- Reviewed on the PR branch (worktree at commit `f30e032`), not from the PR description. `uv sync --group dev` clean; `python scripts/check.py` (full) green: 60 passed, 1 skipped, ruff clean; `python scripts/check.py --status` shows F1 GREEN (11/11) and F14 GREEN (18/18, stages 0-1 only -- the row will only fully read "done" once P1.4/P1.5/P1.7 add the remaining F14-marked tests, which is how `scripts/check.py --status` is designed to work, not a defect in this PR). Offline guard (`tests/conftest.py`, autouse) exercised cleanly across the whole run.
- Scope: `git diff --stat origin/develop...origin/p1-perception` touches only `perception/__init__.py`, `perception/camera.py`, `perception/record.py`, `tests/unit/perception/test_camera.py`, `tests/unit/perception/test_record.py`, and an append-only addition to `ISSUES.md` (diffed directly -- nothing above the new block changed). No edits to `contracts.py`, `config/experiment.json`, `pyproject.toml`, `uv.lock`, or any P2-owned directory.
- Spec match: `perception/camera.py` (`open_source`, `DecimatedSource`, the measured-fps fallback) and `perception/record.py` (Stage 0 record, Stage 1 `--validate`, `runs/manifest.csv`) match essential-features.md F1 and F14 stages 0-1 point for point, including the exact `derive_expected_deviations(experiment, performed_steps)` signature essential-features.md section 5 documents (called through a graceful `ImportError` fallback since `engine/reference.py` doesn't exist yet -- correct per AGENTS.md rule 2, no `contracts.py` guess needed since this shape is already fully specified in essential-features.md, not left to P1 to invent). `runs/run_plan.csv`'s existing header matches `record._PLAN_FIELDS` exactly; step ids in rows 1-12 all resolve against `config/experiment.json`'s `step_ids`, confirmed by static read (the actual recorder was not executed against a camera in this review -- see below).
- Hard rules: BGR preserved throughout (no RGB conversion in `perception/camera.py`, correctly deferred to model boundaries not yet built); `pathlib` and a `python -m` entry point throughout; no outbound network (guard green); no credentials; fixed-seed/deterministic tests (camera fps-measurement tests monkeypatch `time.monotonic` rather than relying on real timing). One non-blocking finding logged separately above (rule 8, `record.py`'s `time.strftime` bookkeeping stamp).
- Not run: the interactive recorder was not exercised against real camera hardware in this review (reviewer-side policy, unrelated to this PR) -- verified statically instead (code reading, the existing `runs/run_plan.csv` fixture, and the unit tests, which exercise `capture_run`/`build_run_script`/`write_script_json`/`probe_video`/`validate_run` against generated clips and fake sources per essential-features.md's F14 done-when, "each run on a fixture").
- Verdict: no blocking findings. Merged `origin/p1-perception` into `develop` (merge commit, `--no-ff`, ort strategy). P1.3's prerequisite (P1.1 visible on `origin/develop`) is met once this push lands.
- On the "defer P1.2's real-footage check" DECISION entry directly above: P2's answer was left blank in the reviewing session's brief (a `[FILL IN: agree | agree with changes: ... | disagree: ...]` placeholder). Per that session's own instruction not to decide it on P2's behalf, no answer is recorded here -- that entry is still OPEN, awaiting P2's actual ack.
- Status: DONE (merge); the rule-8 finding and the DECISION entry above remain OPEN.

---

## 2026-09-29 P2 review (PR #2, p1-perception -> develop, P1.3) - BLOCKING: the BGR->RGB view handed to MediaPipe is non-contiguous and the real library reads it wrong
- Missing/blocked: `perception/hands.py::_to_mp_image` (line 47-50) builds the model input as `mp.Image(image_format=mp.ImageFormat.SRGB, data=image[..., ::-1])`. `image[..., ::-1]` is a negative-stride view, not a C-contiguous array. essential-features.md section 3 (Pitfalls) says "input must be contiguous `uint8` RGB". Reproduced on the installed mediapipe 0.10.35 (no model file needed, `mp.Image(...).numpy_view()` only): a BGR pixel `(10, 20, 30)` comes back as `[30, 10, 20]` (should be `[30, 20, 10]`) and other pixels contain values that exist nowhere in the input (e.g. `[30, 104, 2]`, i.e. it reads memory outside the array). `np.ascontiguousarray(image[..., ::-1])` and `cv2.cvtColor(image, cv2.COLOR_BGR2RGB)` both give the correct `[30, 20, 10]` everywhere.
- Why it matters: rule 16 (BGR everywhere, convert to RGB only at the model boundary) is exactly the boundary this line implements, and it is wrong with the real library: the hand/pose model would silently see scrambled channels and, worse, out-of-bounds bytes, so detection quality would be quietly degraded or nondeterministic (rule 13) with no error. It is invisible to the current tests because every test replaces the landmarker with a fake whose `detect_for_video` discards the image (`tests/unit/perception/test_hands.py::_FakeLandmarker`). It would surface first at P1.6 or in the real-footage check, where it would be blamed on gloves or the model.
- Proposed fix (P1's file, not edited by the reviewer, rule 2): make the RGB array contiguous, e.g. `np.ascontiguousarray(image[..., ::-1])` (or `cv2.cvtColor`, since OpenCV is already in the stack), and add a test that would have caught it: have the fake landmarker capture the `mp.Image` it is given and assert `numpy_view()` is C-contiguous, is `uint8`, and has channels reversed relative to the BGR input (e.g. a `(10, 20, 30)` BGR pixel arrives as `(30, 20, 10)`). Constructing a real `mp.Image` needs no model file, so this test does not depend on a `.task` file. PR #2 was NOT merged; re-request review after the fix.
- Status: OPEN (blocking; PR #2 held).

- Update 2026-09-29 (P1 fix): reproduced independently first (a `(10,20,30)` BGR pixel came back `[30, 0, 0]`, plus reads outside the array on a wider frame -- same root cause, different exact garbage bytes than the reviewer's run, as expected from uninitialized/adjacent memory). Added three tests against `hands._to_mp_image` directly (no model file, real `mp.Image`/`numpy_view()`, as proposed) and confirmed they failed before the fix: `test_to_mp_image_produces_contiguous_rgb_with_correct_channel_order`, `test_to_mp_image_handles_a_non_contiguous_input_view`, `test_to_mp_image_does_not_mutate_the_caller_array`. Fix: `np.ascontiguousarray(image[..., ::-1])`, matching the reviewer's first suggestion (no new dependency; `cv2.cvtColor` would have worked equally well but `ascontiguousarray` is a one-line, self-contained fix with no color-space-conversion semantics to get right). `_FakeLandmarker` in `tests/unit/perception/test_hands.py` now records every `mp.Image` it receives (`received_images`); added `test_hand_tracker_passes_a_correctly_converted_image_to_the_landmarker`, `test_hand_tracker_handles_a_non_contiguous_input_frame`, and the same two for `PoseTracker`, all asserting pixel values through the real `HandTracker`/`PoseTracker.process` path, not just the helper function. `git grep -nE '\[::|\.transpose\(|np\.flip' -- '*.py'` over the whole repo found no other occurrence of this pattern anywhere (`perception/camera.py` and `perception/record.py` pass arrays straight from `cv2.VideoCapture`/`cv2.VideoWriter`, which are always contiguous). All 82 tests green (`python scripts/check.py`, full suite); `--status`: F1 11/11, F3 22/22, F14 18/18 GREEN. Pushed to `p1-perception`; re-requesting P2's review (see the PR #2 comment for the exact commit sha).
- Status: RESOLVED (P1; awaiting P2's re-review before merge).

## 2026-09-29 P2 review (PR #2, P1.3) - non-blocking findings (logged; PR held only for the blocking item above)
- The rule-8 finding from the PR #1 review is fixed and verified: `grep -rnE "time\.(time|strftime|monotonic|perf_counter|...)|datetime|date\.today"` over `perception/` finds clock reads only in `perception/camera.py` (lines 53, 110, 136); `perception/record.py` now has only `time.sleep(1)` (line 250, a countdown, not a clock read) and `row.recorded = "yes"` (line 297). `training/` does not exist yet. `pending_rows` (record.py:125) tests only `recorded.strip()`, so the plain tick still marks a run as done and the F14 rows stay green (18/18). Minor gap: `test_record.py` still writes `recorded="2026-09-29"` in its fixtures and does not exercise `cmd_record`'s new `"yes"` value end to end; harmless, worth one assertion when someone next touches that file.
- Pose.score: `contracts.Pose` says only `score: Unit`, and essential-features.md section 3 says nothing on how it is derived, so "mean landmark visibility (unset -> 1.0)" is consistent with both and needs no contract change. It is not a detector-style confidence (it says how visible the body is, not how sure MediaPipe is that a pose exists), and `PoseLandmarkerResult` has no aggregate confidence, so it is the closest available value. Recommend an additive one-line note in the `Pose` docstring at the next contract PR (P2 cannot edit `contracts.py` from a review session). P2's own ack of this decision was left blank in the reviewing brief (`[FILL IN: ack | ack with change: ... | object: ...]`) and is NOT recorded here -- the 2026-09-29 P1.3 DECISION entry stays OPEN awaiting P2's actual answer.
- Tests never check the image reaching the model (see the blocking item), never assert determinism (rule 13: same frames in, same `Hand` list out, across a `reset()`), and never cover a missing model file. Not blocking on their own; add the determinism case alongside the blocking fix.
- `weights_sha256` re-reads and hashes the whole `.task` file on every access. Fine for a one-off stamp, but P1.6 should read it once at construction/`compose_model_stamp` time, not per frame. With no model file, construction itself fails first (real `create_from_options` cannot open the path), so `weights_sha256` is never reachable without a file; the P1.6 note is that `model_stamp` cannot be composed on a machine without the vendored `.task` files (already tracked by the 2026-09-28 MediaPipe-model-files DECISION, still OPEN).
- Not verifiable here: nothing in `hands.py` has been run against a real landmarker (no `.task` file is vendored, per the 2026-09-29 P1.3 update). Only things only the real library can confirm: that `HandLandmarkerResult.handedness[i][0].category_name` is exactly `"Left"`/`"Right"` and `.score` never exceeds 1.0 (both are validated by `Hand`, so a violation would raise rather than pass silently), that pose landmarks expose `visibility`, and the constructor option names (they import and build `*Options` fine on 0.10.35). The F3 row "`.task` files load from `weights/` offline; tested on recorded frames" (Plan section 7.4) cannot be met until the files are vendored and P1.2's real-footage check runs; the F3 rows that are green are plumbing-only.
- Pass evidence (review worktree at `e051af9`): `uv sync --group dev` clean; `python scripts/check.py` 72 passed / 1 skipped, ruff clean; `--status` F1 11/11, F3 12/12, F14 18/18 GREEN; the GOLD marker tests do not exist yet on `develop` (only `gold_symbolic`/`gold_video` markers are declared), so there is no GOLD-1 to regress. Scope: `git diff --stat origin/develop...origin/p1-perception` touches only `ISSUES.md` (append-only), `perception/hands.py`, `perception/record.py`, `tests/unit/perception/test_hands.py`; nothing in `contracts.py`, `config/`, `pyproject.toml`, `uv.lock` or P2 directories. No `.task`/binary file in any PR commit, no token/key/credential in the diff, commit messages or ISSUES.md (grep for `ghp_`, `github_pat`, `token`, `api_key`, `secret`, `password`, `bearer`, `-----BEGIN`), no new dependency, no network call in `hands.py` or its tests.
- Status: OPEN (non-blocking items; the P1.3 DECISION entry on Pose.score and the P1.2 deferral DECISION entry both still await P2's own answer).

## 2026-09-29 P2 review verdict - PR #2 (p1-perception -> develop, P1.3) NOT merged
- One blocking finding (the non-contiguous RGB view, entry above); everything else in the eight-point review passes or is non-blocking. Not merged, and P1's files were not edited. P1 must: (1) make the RGB array contiguous in `_to_mp_image`, (2) add the test that inspects the `mp.Image` actually passed to `detect_for_video`, (3) push and re-request review. P1.6's prerequisite on P1.3 is therefore NOT yet met on `origin/develop`.
- Status: OPEN (awaiting P1 fix). The fix for the earlier rule-8 finding (commit `42ac064` on the PR branch) will land with the merge; it is verified.
## 2026-09-29 P1 fix - rule-8 finding resolved: no more clock read in record.py
- Fixes the 2026-09-29 P2 review finding directly above. Took the second option offered rather than asking for an exception: `cmd_record` no longer calls `time.strftime(...)`; `row.recorded` is now a plain `"yes"` tick, not a date. DATA_COLLECTION.md section 7.3 only ever asked the tool to "tick" the run, not date-stamp it, so this removes the literal rule-8 violation instead of arguing it was outside the rule's intent. `time.sleep(1)` in `_countdown()` is unaffected (per the reviewer, sleeping isn't reading/recording a clock value). No format elsewhere depends on `recorded` being a date (`pending_rows` only checks it's non-empty).
- Status: RESOLVED (P1, commit on `p1-perception`, rebased onto `develop` post-merge).

## 2026-09-29 P1.3 DECISION - Pose.score has no MediaPipe equivalent to Hand's handedness-category score
- Decision needed: `contracts.Pose` requires `score: Unit`, but unlike `Hand` (which gets a real per-detection confidence from MediaPipe's handedness `Category.score`), `PoseLandmarkerResult` carries no single aggregate confidence for a pose -- only a per-landmark `visibility` (and `presence`, not exposed on the Python landmark object at this MediaPipe version).
- Why it matters: `perception/hands.py::PoseTracker.process` must return something, not raise, and nothing in essential-features.md section 3 or `contracts.py` says how to derive it.
- Decided: `score = mean(landmark.visibility, defaulting an unset value to 1.0)` across all 33 landmarks. Regression tests: `tests/unit/perception/test_hands.py::test_pose_landmarks_convert_to_pixels_with_visibility_mean_score`, `::test_pose_score_defaults_missing_visibility_to_one`. Low-stakes: pose is off by default (`enable_pose=False`) and no Tier-1 rule consumes `Pose` at all (essential-features.md section 3, step 5) -- flagging for P2's ack only in case a future revision changes that.
- Status: DECIDED (P1, open for P2 ack)

## 2026-09-29 P1.3 update - MediaPipe model files still not vendored (see the 2026-09-28 DECISION above)
- `perception/hands.py` (`HandTracker`, `PoseTracker`) is built and tested this session, but entirely against a fake landmarker (`tests/unit/perception/test_hands.py` monkeypatches `HandLandmarker.create_from_options`/`PoseLandmarker.create_from_options`) -- per the 2026-09-29 "defer P1.2's real-footage check" DECISION, this is plumbing only, not a claim about real gloves/landmark behavior. No `.task` file exists under `weights/` yet, so `HandTracker()`/`PoseTracker()` with their real default paths cannot construct on any machine right now.
- Why it matters: downloading `hand_landmarker.task` / `pose_landmarker_lite.task` is an actual external file fetch, which this agent session treats as an explicit-permission action, not something to do silently mid-session -- so it's deliberately left undone here rather than fetched without asking.
- Proposed: whoever runs the real pipeline first (P1.6) fetches the two `.task` files from MediaPipe's published model links, vendors them under `weights/`, records their sha256 in `weights/MANIFEST.json`, and confirms here.
- Status: OPEN (unchanged from 2026-09-28; still assigned to before P1.6 needs a real `Perception`, not blocking P1.3's plumbing).

---

## 2026-09-29 P2 re-review verdict - PR #2 (p1-perception -> develop, P1.3) MERGED
- Re-reviewed `origin/p1-perception` at `24b1258` (the branch had been rebased onto `develop` and force-pushed since the first review, so the earlier `e051af9` no longer exists; the content diff `e051af9 -> 24b1258` is only `hands.py` +17/-3, `test_hands.py` +159 and three lines of `ISSUES.md`). Merged with `--no-ff` (merge commit `d91a05e`). The earlier blocking finding (non-contiguous RGB view handed to `mp.Image`) is RESOLVED.
- Independent reproduction, not P1's report: with the old line `data=image[..., ::-1]` restored and the new tests run, 6 tests fail (`test_to_mp_image_produces_contiguous_rgb_with_correct_channel_order`, `..._handles_a_non_contiguous_input_view`, and the `HandTracker`/`PoseTracker` "passes a correctly converted image" and "handles a non-contiguous input frame" pairs); with the fix all 22 F3 tests pass. Further mutations of the fixed file, each caught: dropping the channel reversal (6 fail), `reset()` keeping the stale timestamp (1), a random offset in pixel conversion (4), Pose.score fixed at 1.0 (1).
- Real library (mediapipe 0.10.35, no model needed): random BGR frames (37x101, 480x640), a decimated view, a cropped view, a `[::-1]` view and a Fortran-order array all come out of `mp.Image.numpy_view()` equal, over every pixel, to `cv2.cvtColor(..., COLOR_BGR2RGB)`; output is C-contiguous `uint8` with the right width/height; five distinct known pixels land at the expected RGB; the caller's array is unmutated; reading the view after churning memory shows no dangling reference.
- Other channel flips / layout changes: `git grep -nE '::-1|\.transpose\(|np\.flip|cv2\.flip|\.T\b|swapaxes|moveaxis|rot90|as_strided'` over `perception/` and `tests/` finds one hit reaching a native call, the fixed line in `hands.py`. `Frame.image` is passed straight from `cv2.VideoCapture.read()` (`camera.py:142`) to `VideoWriter.write`/`imshow`/`cvtColor` in `record.py`, so it is contiguous by construction. (P1's own grep pattern `\[::` could not match `[..., ::-1]`; the original finding was outside what it could see.)
- `python scripts/check.py` (full) on the merged tree: green, 82 passed / 1 skipped, ruff clean; `--status`: F1 11/11, F3 22/22, F14 18/18 GREEN. Scope since the first review: only `perception/hands.py`, `tests/unit/perception/test_hands.py` and `ISSUES.md`; no `contracts.py`, `config/`, `pyproject.toml`, `uv.lock`, `IMPLEMENTATION_PLAN.md` or P2 directory touched; no `.task` or other binary in any PR commit; no token/key/credential in the diff, commit messages or `ISSUES.md`; no new dependency; no network code in `perception/`.
- Non-blocking, logged: (1) P1 inserted its 3-line resolution note into the middle of `ISSUES.md`, between my blocking entry and my non-blocking entry, rather than at the end; no existing line was altered (zero removed lines against `develop`), but the file is meant to be append-only, so later notes should go at the bottom. (2) The determinism-across-`reset()` tests use a scripted fake, so they prove the wrapper adds no state-dependent output but cannot show MediaPipe itself is deterministic; that check waits for the vendored `.task` files. (3) `test_weights_sha256_raises_file_not_found_when_no_model_file_is_vendored` pins current behaviour only; `weights_sha256` re-hashes the file on every access and is documented as such, so P1.6 must read it once when composing `model_stamp`, never per frame. (4) The 2026-09-29 P1.3 update entry (the `.task` files are still not vendored, so nothing here has run on a real landmarker) is unchanged and still OPEN; the F3 row "`.task` files load from `weights/` offline; tested on recorded frames" is still unmet.
- Not recorded: P2's answers on the P1.2 deferral DECISION and on the Pose.score DECISION were both left blank (`[FILL IN or DELETE]`) in this session's brief. Both entries stay OPEN awaiting P2's own answer; nothing was decided on P2's behalf.
- P1.6 prerequisite: P1.3 is now visible on `origin/develop`. P1.6 also needs P1.5 (detector), and P1.3's plumbing merged on the strength of the deferral DECISION (no real footage, no vendored model), so P1.6 must vendor the `.task` files first (2026-09-28 DECISION, still OPEN).
- Status: DONE (merge); the two DECISION entries and the model-files entry remain OPEN.

---

## 2026-09-29 P2 DECISION - ack: defer P1.2's real-footage check (resolves the 2026-09-29 P1 DECISION entry, commit `f30e032`)
- Resolves: the 2026-09-29 P1 DECISION entry above, "defer P1.2's real-footage check while building content-independent plumbing ahead of schedule" (commit `f30e032`), left `Status: OPEN -- awaiting P2's ack`. Per this file's append-only rule, that entry's own text and Status line are left unmodified; this new entry is the record of P2's answer.
- P2's answer (relayed by P1 on P2's behalf): "Deferral DECISION (build without real footage / defer P1.2, commit f30e032): agree. Scope: the entry as written in commit f30e032."
- Note (P2, not part of the quoted answer above): these answers were relayed by P1 on P2's behalf. P2 can amend either answer later by appending a new entry.
- Status: DECIDED -- P2 agrees with the 2026-09-29 P1 DECISION entry above exactly as written there, including its scope (points 1-6: the general agreement, the one disagreement-on-scope about not deferring the pilot recording itself, the P1.6 stub-only correction, the unrelated-but-overdue GPU smoke test, the tripwire before P1.4/rows 13-77, and the confirmation that nothing else in P1's track needs a stronger gate). The deferral DECISION entry above is now acknowledged by P2.

## 2026-09-29 P2 DECISION - ack: Pose.score = mean landmark visibility (resolves the 2026-09-29 P1.3 DECISION entry)
- Resolves: the 2026-09-29 P1.3 DECISION entry above, "Pose.score has no MediaPipe equivalent to Hand's handedness-category score" (`score = mean(landmark.visibility, defaulting an unset value to 1.0)`), left `Status: DECIDED (P1, open for P2 ack)`. Per this file's append-only rule, that entry's own text and Status line are left unmodified; this new entry is the record of P2's answer.
- P2's answer (relayed by P1 on P2's behalf): "Pose.score = mean landmark visibility: ack."
- Note (P2, not part of the quoted answer above): these answers were relayed by P1 on P2's behalf. P2 can amend either answer later by appending a new entry.
- P1 note (not part of P2's answer) -- a verification result relayed alongside P2's answer, not itself part of what P2 agreed to: Pose.score = mean landmark visibility is acceptable for now because (a) nothing consumes it: no Tier-1 rule uses pose and enable_pose is off by default (essential-features.md section 3); (b) the spec defines it only as Pose(score, landmarks_px[33]) and P2's reviewer found the contract type is only Unit; (c) MediaPipe's PoseLandmarker reports per-landmark visibility but no aggregate pose confidence (as reported by the P1 agent; not yet confirmed on the real library because no .task file is vendored). Caveats: it measures how visible the body is, not detection confidence; with an overhead table camera most of the 33 landmarks may be out of frame, so the value is expected to be low even for a correct detection (unverified). Recommended conditions, not binding unless P2 says so: (1) add a one-line docstring note about the meaning at the next contract PR; (2) no rule, threshold or gate may use Pose.score without a new decision; (3) revisit when the open decision on whether enable_pose is on in the demo is made.
- Status: DECIDED -- P2 acks the 2026-09-29 P1.3 DECISION entry above as written. P1's verification result and its three recommended (non-binding) conditions are logged above for visibility but are not themselves part of P2's ack; they remain open for P2 to bind later if desired.
## 2026-09-29 P1.5-provisional - CPU latency benchmark: harness built and tested; real numbers pending approval below
- Built `training/benchmark_cpu.py` per essential-features.md section 14, Stage 8: `benchmark_predict` times the full per-frame path (`_bgr_to_contiguous_rgb` -- `np.ascontiguousarray`, not a bare view, per the 2026-09-29 hands.py finding -- then `predict_fn`) over 20 warm-up (excluded) + 200 timed frames (both CLI-configurable), reporting mean/p95/fps; `environment_info()` records CPU model, logical CPU count, OS, Python and torch thread count; `build_report` assembles the Stage-8 JSON shape and always includes `"hands": "not measured (no .task model file vendored)"`, since no MediaPipe model file exists yet (2026-09-28 DECISION, still open). Real candidates (`RFDETRNano` PyTorch, `optimize_for_inference`, ONNX Runtime, YOLO11n) are gated behind an explicit `--allow-download` CLI flag that defaults to **off** -- without it, nothing is imported, constructed, or downloaded, and the report simply lists zero results. Frames are synthetic (`np.random.default_rng`), sized 1280x720 (`contracts.CAPTURE_WIDTH/HEIGHT`, essential-features.md section 0) -- no camera or recorded footage needed, consistent with the standing "no real footage yet" decision.
- Caveat on synthetic frames (asked for explicitly, essential-features.md Stage 8 "post-processing can depend on the number of detections"): random noise fed to a COCO-pretrained detector will trigger an unpredictable, run-dependent number of spurious detections, unlike real content -- timing 200 frames and reporting mean/p95 (not a single frame) is meant to average this out, but the resulting numbers should be read as an order-of-magnitude check, not a tight bound. This is on top of the numbers already being provisional for the reasons below.
- 12 tests in `tests/unit/training/test_benchmark_cpu.py`, all against a fake `predict_fn` and a scripted deterministic clock -- no real model, no network. `check.py --status`: F1 11/11, F3 22/22, F14 30/30 GREEN (94 passed / 1 skipped overall).
- **No real number is reported yet.** Every real candidate needs either an install or a weight download this session was told to stop and ask about first (IMPLEMENTATION_PLAN.md R7); see the two entries directly below. `git status` confirms a smoke-run with `--allow-download` omitted writes `data/benchmark_provisional/cpu_benchmark.json` (git-ignored; `data/` is in `.gitignore`) with an empty `results` list, as designed -- nothing was downloaded or installed to produce that smoke test.
- Status: OPEN (harness done; numbers blocked on the approvals below).

## 2026-09-29 P1 R7 request - onnxruntime is a genuinely new dependency; ultralytics (YOLO11n) is already in pyproject.toml but not yet installed on this machine
- **onnxruntime** -- not in `pyproject.toml`/`uv.lock` at all today (checked: absent from both). Latest PyPI release `1.30.0`, MIT license, `cp311-win_amd64` wheel is 14.3 MB. Needed only for the RF-DETR-Nano-under-ONNX-Runtime candidate (essential-features.md section 2, step 6). Per IMPLEMENTATION_PLAN.md R7 ("a new dependency needs the other person's agreement... only P2 edits the lockfile") and Part 8 (`pyproject.toml`/`uv.lock` are P2-only), this needs P2's ack before it goes in the lockfile -- I cannot add it myself. **Proposed interim path** (per this session's own instruction, "a throwaway environment outside the lockfile"): a separate, disposable venv (`python -m venv /tmp/onnx-bench-venv` or similar, `pip install onnxruntime rfdetr` there only, never inside `.venv/` or committed), used only to produce this session's provisional ONNX number, then deleted. It never touches `pyproject.toml`, `uv.lock`, or this repo's own `.venv/`.
- **ultralytics** (YOLO11n) -- already declared in `pyproject.toml`'s `tools` group (`>=8.3,<9`; latest release satisfying it is `8.4.165`, AGPL-3.0, 1.41 MB wheel; already logged in essential-features.md/context.md section 6 as "kept in a dev-tools dependency group and never in the runtime environment"). This is **not a new R7 request** -- it was already decided at G0 -- but it is not installed in this machine's venv yet (`uv pip list` confirms `rfdetr`/`torch`/`torchvision`/`supervision`/`pillow` are present from earlier `--group dev` syncs, but no `ultralytics`). Installing it needs `uv sync --group tools`, which this session is treating as an install action to ask about, not something to run silently.
- Status: OPEN (awaiting P1's own go-ahead in chat for `uv sync --group tools`, and separately awaiting P2's ack for onnxruntime before it can ever join the lockfile -- the throwaway-venv path does not need that ack, only the operator's go-ahead to run it).

## 2026-09-29 P1 note - model weight files needed for real numbers (none downloaded; awaiting go-ahead)
- **RF-DETR-Nano COCO-pretrained checkpoint**: `https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth`, confirmed by an HTTP HEAD request (no body fetched) at 366,287,238 bytes (~366 MB), Apache-2.0 (rfdetr's own license). Auto-downloaded by `RFDETRNano()` on first construction to `~/.roboflow/models` (confirmed empty/absent on this machine today) if not already cached -- outside this repo, never committed, dev-time only. Covers both the plain and `optimize_for_inference` PyTorch candidates and the ONNX-export candidate (one checkpoint, exported locally).
- **YOLO11n COCO-pretrained checkpoint**: `yolo11n.pt`, resolved via Ultralytics' GitHub release redirect to `release-assets.githubusercontent.com`, confirmed at 5,613,764 bytes (~5.6 MB) via HTTP HEAD, AGPL-3.0 (Ultralytics' own license page, already the basis of the RF-DETR-over-YOLO decision in context.md section 6). Dev-time only. **Correction, found the hard way**: `ultralytics.YOLO("yolo11n.pt")` does *not* download to a per-user cache dir as first assumed here -- a bare relative name downloads straight into the current working directory, which during the actual run below was the repo root (`D:\...\BAS\yolo11n.pt`). Caught immediately after the run (`git status` showed it as an untracked file, never staged or committed), deleted, and `training/benchmark_cpu.py`'s `_yolo11n_candidate` now passes an explicit absolute path (`~/.cache/ultralytics_benchmark/yolo11n.pt`) so this can't recur. See the RESULTS entry below.
- Neither file exists on this machine (`~/.roboflow/models` absent; no `*.pt` in the usual Ultralytics cache locations). Both are dev-time-only artifacts per R10 (offline is a runtime property; downloading a benchmark checkpoint is not runtime code) and would never be committed (`weights/*` is git-ignored except `MANIFEST.json`, and neither file belongs at `weights/detector.pth` per the standing decision -- COCO-pretrained weights are for this benchmark only, never mistaken for the fine-tuned checkpoint).
- Status: OPEN (awaiting go-ahead to download either or both; `--allow-download` on `training/benchmark_cpu.py` gates exactly this).

## 2026-09-29 P1 note - GPU smoke test still not recorded
- Checked per this session's instruction: the 2026-09-28 GPU-access DECISION entry (this file, "where the detector is fine-tuned") still only has an unactioned "Action:" line asking whoever trains to run a 5-minute forward/backward smoke test on the assigned accelerator; no result has ever been appended. Who trains: P1 (this track), per that same entry ("P1 has Colab/Kaggle access and runs the fine-tune, P1.5, background script"). Not run in this session -- this session explicitly stays off Kaggle/Colab.
- Status: OPEN (unchanged from 2026-09-28; still P1's to do before/at P1.5, on the actual training accelerator, not this laptop).

---

## 2026-09-29 P1 note - `uv sync --group <name>` replaces the environment, it doesn't add to it
- Found while running the approved `uv sync --group tools`: it uninstalled `pytest` and `ruff` (both `dev`-only) as a side effect, because `uv sync --group X` syncs the venv to match exactly that group's closure, not the union of it with whatever was already installed. `uv sync --group dev --group tools` (both groups in one call) restores everything and is the command that should be used from here on whenever both are needed on the same machine. Confirmed via `uv pip list` before/after and a clean `check.py --quick` (99 passed / 1 skipped) after the corrected sync.
- Why it matters: AGENTS.md's Environment section and `IMPLEMENTATION_PLAN.md` 3.4 only document `uv sync --group dev` and, separately, `uv sync --group tools`, each shown alone -- neither example warns that running the second one after the first silently removes dev-only tooling. Anyone following those docs literally on a machine that also needs `tools` would lose `pytest`/`ruff` the same way.
- Not a contract or lockfile change (`pyproject.toml`/`uv.lock` untouched) -- purely a locally-run-command gotcha, logged so the next person (or the next `AGENTS.md`/`IMPLEMENTATION_PLAN.md` edit) doesn't hit it blind.
- Status: DECIDED (workaround: always pass every needed group in one `uv sync` call). Worth a doc fix at the next session that touches `AGENTS.md`/`IMPLEMENTATION_PLAN.md` 3.4, but this session doesn't touch either file (out of scope, `.gitignore`d results only).

## 2026-09-29 P1 note - essential-features.md section 2 step 2 is stale: `optimize_for_inference()` was renamed to `inference()` in the installed rfdetr
- Found while wiring the real RF-DETR-Nano candidate into `training/benchmark_cpu.py`: `RFDETRNano.optimize_for_inference()` (as essential-features.md section 2, step 2 names it) does not exist on the installed `rfdetr==1.11.0` (`AttributeError: 'RFDETRNano' object has no attribute 'optimize_for_inference'`). The installed version's actual method is `.inference(compile=True, batch_size=1, dtype=torch.float32, inplace=False, compile_backend="torchscript")`, which does the same job (compiles the model for faster repeated inference; its own docstring says "Optimize the model for inference with optional compilation"). Verified by reading the installed package's docstring and signature directly, not assumed.
- Why it matters: essential-features.md's own rule ("check the option names against the installed version's docs") anticipated exactly this kind of drift, but named the pre-rename method specifically; whoever writes the real `perception/detector.py` at P1.6 needs `.inference()`, not `.optimize_for_inference()`, on this pinned version (`rfdetr>=1.10,<2` in `pyproject.toml` -- the rename could have happened anywhere in that range; not re-verified against 1.10.x specifically).
- Fixed in `training/benchmark_cpu.py`'s own candidate builder (this PR); essential-features.md is a shared doc, not code, so this entry is the record rather than an edit to that file (out of this session's scope; a P1.6 session should fix the doc text when it writes the real detector wrapper).
- Status: DECIDED (workaround applied here); essential-features.md's step 2 text still says the old name -- flagging for whoever next edits that file.

## 2026-09-29 P1.5-provisional RESULTS - CPU latency: RF-DETR-Nano (COCO-pretrained) vs YOLO11n, two runs each
- **Approved and run this session** (IMPLEMENTATION_PLAN.md R7 / explicit go-ahead in chat): `uv sync --group tools` (ultralytics, already pre-approved at G0 -- and a separate gotcha found and fixed along the way: `uv sync --group tools` alone silently uninstalled `pytest`/`ruff`, see the note above; re-synced with both groups together); the RF-DETR-Nano COCO checkpoint download (349 MiB, to `C:\Users\<user>\.roboflow\models\rf-detr-nano.pth`, MD5-validated by `rfdetr` itself, outside this repo, never at `weights/detector.pth`, never committed); the YOLO11n COCO checkpoint download (~5.4 MiB). **The YOLO11n download's first attempt landed in the repo root** (`YOLO("yolo11n.pt")`'s bare relative name resolves against the CWD, not a per-user cache as first assumed) -- caught via `git status`, deleted before it was ever staged, and the code fixed to use an explicit absolute out-of-repo path (`~/.cache/ultralytics_benchmark/yolo11n.pt`) before the timed runs below; nothing from either download was ever committed. **ONNX Runtime was explicitly declined this session** -- not installed, no throwaway venv created; `training.benchmark_cpu`'s own ONNX candidate builder correctly logged "onnxruntime and/or rfdetr[onnx] not importable; skipping" and produced no ONNX result, exactly as designed.
- **Environment** (this machine, confirmed to be the demo laptop): AMD Ryzen 5 5600H (`platform.processor()`: "AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD"), 6 physical / 12 logical cores, Windows 11 (`platform.platform()` reports "Windows-10-10.0.26200-SP0" -- **stale**; `sys.getwindowsversion().build` = 26200, which is Windows 11 (11 starts at build 22000); fixed and tested in `training.benchmark_cpu._os_info`, per the operator's request). Python 3.11.16, torch 2.14.0+cpu, no CUDA, `torch.get_num_threads()` = 8 during these runs (was 6 by default before `ultralytics` was imported in-process; not independently controlled, noted as a source of run-to-run variance below). Plugged into mains power, High Performance power plan, battery saver off, heavy apps closed, per the operator's confirmation before these runs started.
- **Results** (20 warm-up frames excluded, 200 timed frames, synthetic 1280x720 BGR frames per `contracts.CAPTURE_WIDTH/HEIGHT`, full path timed = `np.ascontiguousarray` BGR->RGB + `predict`):

  | Candidate | Run 1 mean / p95 / fps | Run 2 mean / p95 / fps | Spread (mean) |
  |---|---|---|---|
  | `rfdetr_nano_pytorch` (no `.inference()`) | 199.2 ms / 238.8 ms / 5.02 | 158.4 ms / 195.6 ms / 6.31 | 40.8 ms (~20.5%) |
  | `rfdetr_nano_pytorch_optimized` (`.inference()`, torchscript-compiled) | 170.2 ms / 205.1 ms / 5.88 | 151.8 ms / 162.1 ms / 6.59 | 18.4 ms (~10.8%) |
  | `yolo11n_pytorch` | 46.2 ms / 52.9 ms / 21.64 | 36.4 ms / 38.9 ms / 27.47 | 9.8 ms (~21.2%) |

  Run 2 was faster than run 1 across all three candidates by a similar relative margin, most likely OS/thermal/background-process effects between runs rather than the harness itself (warm-up frames are already excluded from every number above, `time.perf_counter` is monotonic, and each candidate's own 200-frame loop is internally consistent) -- the ~10-21% run-to-run spread should be read as this laptop's real-world noise floor for this kind of measurement, not a harness bug.
- **Against `min_pipeline_fps = 8`** (`config/acceptance.yaml`'s not-yet-written judgment call, currently only a constant in essential-features.md section 0): both RF-DETR-Nano PyTorch variants land **below** 8 fps in both runs (5.02-6.59); YOLO11n clears it comfortably in both runs (21.64-27.47). **This is a provisional, COCO-pretrained observation, not a decision** -- P1.5 re-measures on the fine-tuned head (different class count, possibly different output-decoding cost) and against `acceptance.yaml`'s actual recall thresholds, and is the only session that chooses a detector or sets `target_fps`, per the standing decision and this session's own scope.
- **What was NOT measured**: hands (no `.task` file vendored, per the 2026-09-28 DECISION -- every report explicitly says so); ONNX Runtime (declined this session, R7 request stays open for P2); the fine-tuned detector (COCO-pretrained only, different class count than `config/experiment.json`'s five classes); anything on real Sample Transfer footage (synthetic random frames only -- per the earlier "post-processing can depend on detections" caveat, a COCO-pretrained model's detection count on random noise is unpredictable per-frame, which is part of why two full runs were taken rather than one).
- **`check.py --status`** on this commit: F1 11/11, F3 22/22, F14 35/35 GREEN (99 passed, 1 skipped overall). Raw JSON for both runs kept locally only, git-ignored (`data/benchmark_provisional/run{1,2}.json`, per `.gitignore`'s `data/` entry) -- not committed, per this session's instruction not to invent a tracked path under `reports/`.
- Status: PROVISIONAL (informational only; no detector chosen, no `target_fps` set -- both remain P1.5's).

## 2026-09-29 P2 review of PR #3 (P1.5-provisional benchmark harness): PASS, merged; non-blocking notes for P1
- Verdict: no blocking findings. Reviewed head `d8ca0b8` (on `6424f34`; `95f1f5b` + `d8ca0b8`, unchanged during review). Full `check.py` green (99 passed, 1 skipped); `--status` F1 11, F3 22, F14 35 GREEN. Diff is only `training/benchmark_cpu.py`, `training/__init__.py`, `tests/unit/training/test_benchmark_cpu.py` and 52 appended lines of `ISSUES.md` (0 deletions); `pyproject.toml`, `uv.lock`, `contracts.py`, `config/`, `reports/`, `acceptance.yaml` untouched. No `*.pt/*.pth/*.onnx/*.task` in either PR commit or in any ref's history; `yolo11n.pt` never entered a commit. Verified: `_percentile` equals `numpy.percentile` (n=5, 20, 200, 201); warm-up excluded; injected clock in tests, `perf_counter` by default; `np.ascontiguousarray` BGR->RGB; nothing constructed or downloaded without `--allow-download`. The installed `rfdetr` 1.11.0 resolves a bare weight filename to `get_model_cache_dir()` = `~/.roboflow/models` (or `RF_HOME`/`ROBOFLOW_HOME`), absolute and not CWD-relative (`assets/model_weights.py:275-300`, `detr.py:574-596`), so RF-DETR's own default cannot drop a checkpoint into the repo. No detector chosen, no `target_fps` set.
- Non-blocking notes (none blocks the merge; all are P1's to fix in a follow-up, best before P1.5):
  1. **ONNX candidate is untested and has two latent bugs.** `RFDETRNano().export()` defaults to `output_dir="output"` (`detr.py:1688`), relative to the CWD, and `output/` is not in `.gitignore`: running with `onnxruntime` installed from the repo root would write untracked files into the repo root, the same class of bug as `yolo11n.pt`. Pass an explicit absolute out-of-repo `output_dir`. Also `session.run(None, {input_name: rgb})` feeds an HWC uint8 array where the exported graph almost certainly expects preprocessed NCHW float32. Unreachable today (onnxruntime declined, not in the lockfile) but fix before the R7 request is acked.
  2. **YOLO11n input colour order.** Ultralytics documents ndarray input as BGR (`engine/predictor.py:184`), and `_yolo11n_candidate` feeds it the RGB copy. Harmless for latency (it only adds one conversion to YOLO's timing, so slightly conservative) but the P1.6 detector wrapper must not copy this; RGB is for RF-DETR/MediaPipe only (AGENTS.md rule 16).
  3. **Report carries no provisional marker.** `--out` defaults to `reports/benchmark_cpu.json` (P2-owned, and where P1.5 will write). A no-flag run would put unmarked numbers there. Add `provisional`/`weights` fields (e.g. `"weights": "coco_pretrained"`) to `build_report` before P1.5, or keep using an explicit git-ignored `--out`.
  4. **Run-order / drift labelling is weaker than it reads.** Run 2 was faster than run 1 for all three candidates, in the same direction (10-21%). That looks like systematic warm-up or thermal drift more than a "noise floor"; candidates are also always timed in fixed order (RF-DETR first, YOLO last) after all are constructed. For P1.5: interleave or randomise candidate order and take >= 3 runs. The torch thread drift (6 -> 8 after importing ultralytics) is not reproduced on the reviewer's machine (10 -> 10, different CPU), so it stays P1's unverified observation; pin `torch.set_num_threads` explicitly and record it.
  5. **Cosmetic.** `ruff format --check` would reformat `training/benchmark_cpu.py:307` (`check.py` runs only `ruff check`, so it is not enforced). `_percentile` tests use only constant or 5-point data. Two older entries in this PR are stale: "12 tests" (there are 17), and the weight-file note's "Neither file exists... Status OPEN (awaiting go-ahead)", superseded by the RESULTS entry.
- Status: DECIDED (merged; the notes above are non-blocking follow-ups for P1, no ack needed).

## 2026-09-29 R7 DECISION - onnx + onnxruntime added to the `tools` group (answers P1's R7 request above)
- **Decision (P2, with P1's agreement in chat):** add `onnx>=1.16,<2` and `onnxruntime>=1.20,<2` to the `tools` group in `pyproject.toml`; `uv.lock` re-resolved (275 packages, `uv lock --check` clean). `run`, `dev` and `train` are unchanged, so the demo-laptop runtime environment is unchanged. P1's request asked for `onnxruntime` alone; that is not enough (see below), so `onnx` is added with it.
- **What the export actually needs** (read from the installed `rfdetr==1.11.0`, then tested; no names assumed): the declared extra is `rfdetr[onnx]` = `onnx`, `onnxsim`, `onnx_graphsurgeon`, `onnxruntime`, `polygraphy`. `model.export()` calls `torch.onnx.export(..., dynamo=False)` (`rfdetr/export/_onnx/exporter.py`); nothing in 1.11.0 calls `onnx_simplify`, so `onnxsim`, `onnx_graphsurgeon` and `polygraphy` are never exercised by a plain export. Tested with a randomly initialised `RFDETRNano(pretrain_weights=None)` (no weight download) and packages installed to a scratch `--target` dir: with `onnxruntime` alone, export fails with `OnnxExporterError: Module onnx is not installed!` (raised by torch's exporter); with `onnx` + `ml-dtypes` added it succeeds (114.8 MB file, input `[1,3,384,384]` float32, outputs `dets (1,300,4)` and `labels (1,300,91)`) and ONNX Runtime runs it on CPU.
- **Note (rfdetr upgrade):** the upstream `rfdetr[onnx]` extra is broader (7 packages: it additionally pulls `onnxsim`, `onnx-graphsurgeon`, `polygraphy`, about 4 MB more). We deliberately take the narrower set, verified only against rfdetr 1.11.0. A future `rfdetr` upgrade (the pin is `>=1.10,<2`) may start using those packages in the plain export path; revisit this line if `model.export()` raises "ONNX export dependencies are missing" after an upgrade. Fallback is `rfdetr[onnx]` in place of the two lines.
- **New packages in `uv.lock`** (4; Windows cp311 wheel sizes; `flatbuffers`, `numpy`, `packaging`, `typing-extensions` were already locked):

  | Package | Version | License | Wheel (win_amd64) | Note |
  |---|---|---|---|---|
  | onnxruntime | 1.30.0 | MIT | 14.31 MB | CPU wheel; no CUDA/GPU runtime |
  | onnx | 1.23.0 | Apache-2.0 | 7.88 MB | needed by torch's exporter |
  | ml-dtypes | 0.5.4 | Apache-2.0 | 0.21 MB | dependency of `onnx` |
  | protobuf | 7.36.2 | BSD-3-Clause | 0.46 MB | first appearance in the lock |

  About 22.9 MB in total. No copyleft license. No package pulls a CUDA/GPU runtime (`onnxruntime` is the CPU wheel; `onnxruntime-gpu` is not involved). All four are dev-time only (`tools`).
- **protobuf check:** `uv tree --invert --package protobuf --group tools` (also with `--all-groups`) shows only `onnx` and `onnxruntime` depend on it, both via group `tools`; nothing in `run` does (mediapipe 0.10.35 does not require it). After `uv sync --group dev --group tools`: `import mediapipe` works (0.10.35 alongside protobuf 7.36.2), a real `mp.Image` builds, and the F3 tests pass (22/22). `python scripts/check.py` green (99 passed, 1 skipped; F1 11/11, F3 22/22, F14 35/35). The same sync also installed `ultralytics` and its dependencies (already declared in `tools` at G0, absent from this venv until now).
- **Deviation from the plan:** `IMPLEMENTATION_PLAN.md` 3.4 installs `tools` "only on the machine that runs auto-labeling". The benchmark has to run on the demo laptop (it measures that CPU), so `tools` is being installed there for benchmarking. A small deviation, harmless because `run` is unchanged and `tools` is never part of the demo runtime. If ONNX wins at P1.5, `onnxruntime` moves to `run` and `rfdetr`/`torch` to `train` (plan 3.1); that is a separate R7 entry then.
- **Pointers for P1's follow-up** (P2 edits none of P1's files; found while testing the export above, in `training/benchmark_cpu.py::_rfdetr_onnx_candidate`):
  1. **Preprocessing and postprocessing are missing.** The candidate calls `session.run(None, {input_name: rgb})` with the raw 1280x720 uint8 frame. The exported model takes `[1,3,384,384]` float32 (CHW, resized and normalised) and returns raw `dets`/`labels` tensors that still need decoding. As written it would fail, or, if adapted to run, time only the network and not the resize/normalise/decode work the PyTorch path includes (`predict()` does them), so the ONNX number would not be comparable. The timed ONNX path should include the same preprocessing and decoding.
  2. **`model.export()` uses a relative `output_dir="output"`** (signature default in `rfdetr/detr.py`), so it writes into the current working directory, the repo root if run from there: the same class of stray-file problem as the `yolo11n.pt` incident above. Pass an explicit absolute out-of-repo `output_dir` (like `_YOLO_CACHE_PATH`).
  3. It also builds `RFDETRNano()`, which uses the COCO checkpoint; note the export of the fine-tuned head at P1.5 will have a different class count than the 91 seen here.
- Status: DECIDED (dependency added on branch `p2-r7-onnx-deps`; PR into `develop` for P1's review, R9). The two candidate fixes above stay OPEN for P1.
---

## 2026-09-29 P1 fix - PR #3 review follow-up: ONNX candidate hardened, harness controls added, YOLO colour order, report hygiene
- Fixes all five non-blocking notes from the 2026-09-29 P2 review of PR #3, above. Measurement-only session: no detector chosen, no `target_fps` set, no edit to `config/acceptance.yaml`. Test-first as before; every new function is unit-tested against fakes, no real `onnxruntime` or real model run is exercised by the test suite (`onnxruntime` is still not importable in this venv -- confirmed again this session).
- **Finding 1 (ONNX candidate, two latent bugs) -- both fixed:**
  - `RFDETRNano.export()`'s default `output_dir="output"` is CWD-relative (confirmed by reading `detr.py`'s `export()` signature directly, not assumed); `_rfdetr_onnx_candidate` now passes an explicit absolute `~/.cache/rfdetr_benchmark/` (`_ONNX_EXPORT_CACHE_DIR`), the same fix pattern as the earlier `yolo11n.pt` correction. `.gitignore` is untouched (not needed -- the export dir is outside the repo entirely, same as the RF-DETR/YOLO checkpoint caches).
  - Preprocessing and postprocessing are **not** reimplemented by hand -- read the installed rfdetr 1.11.0 source (`detr.py`'s `predict()`, `models/postprocess.py`'s `PostProcess`) to confirm the exact math (resize via `torchvision.transforms.functional.resize(..., antialias=False)` to the model's native resolution, `F.normalize` with ImageNet mean/std `[0.485,0.456,0.406]`/`[0.229,0.224,0.225]`, per-class sigmoid over a flattened `(Q*C)` query/class grid, global top-`num_select` selection, `cxcywh`->`xyxy`, scale to the target image size, threshold), then found that rfdetr **already ships** the exact reusable implementation of this at `rfdetr.export._runtime.preprocess.preprocess_to_nchw` and `rfdetr.export._runtime.decode.decode_detections` -- the same functions rfdetr's own ONNX inference helper (`rfdetr.export._onnx.inference`) calls, documented to be bit-exact with `predict()`'s preprocessing and to mirror `PostProcess.forward` exactly (its own docstring: "this decode mirrors `PostProcess.forward`... the parity suite depends on it staying that way"). Per AGENTS.md rule 1 (reuse libraries, never reimplement an algorithm a library provides), the ONNX candidate calls these directly rather than hand-rolling the resize/normalize/decode math. `background_class_id=None` is used (per `decode_detections`'s own docstring: "Pass `None` for sparse COCO checkpoints, whose final slot is class 90" -- this is a sparse COCO-pretrained checkpoint). These are underscore-prefixed ("private") modules of the installed `rfdetr==1.11.0`, not a guaranteed-stable public API across versions -- flagged in the module docstring to re-verify on any rfdetr upgrade. Also set `fp16=False` explicitly on export (the library default is `fp16=True`, tuned for GPU tensor cores per context.md section 6's own CPU-latency caveat; CPU execution providers generally lack fast fp16 kernels, so the default would not have been a fair CPU comparison against the float32 PyTorch candidates).
  - Each row's timed content is now documented explicitly in the module docstring: every candidate's per-frame timing covers the same four stages (resize, normalize, network forward, decode/threshold) -- PyTorch's `predict()` does all four internally; the ONNX candidate does the same four explicitly via the reused helpers above.
  - Parity-check design (item 1e): `check_raw_output_parity(pytorch_boxes, pytorch_logits, onnx_boxes, onnx_logits, box_atol=1e-3, logit_atol=1e-2)` compares the two *raw* `(pred_boxes, pred_logits)` tensors for the same preprocessed input, before decode -- isolating export/runtime drift from decode logic (which is the same shared function on both sides in Phase 2, so it can't itself be the source of a divergence). Tolerances are disclosed judgment calls, not sourced. `extract_pytorch_raw_outputs(model, preprocessed_tensor)` is the Phase-2-only glue that gets PyTorch's raw output for the comparison (mirrors `predict()`'s own `_is_optimized_for_inference` dispatch, read directly from `detr.py`). Both are unit-tested against fakes only this session (4 parity tests: matching, box-mismatch, logit-mismatch, custom tolerance; 2 extraction tests: optimized and unoptimized dispatch) -- per this session's own instruction, the real run (feeding an actual preprocessed frame through both a real PyTorch model and a real ONNX session) is Phase 2, gated on `onnxruntime` landing in the lockfile.
- **Finding 2 (YOLO colour order) -- fixed, not just documented:** confirmed by reading `ultralytics/engine/predictor.py`'s own documented ndarray-input convention (BGR) that the harness's shared timing loop feeds every candidate RGB (`_bgr_to_contiguous_rgb`, upstream of `predict_fn`). Since converting back is cheap and doesn't cost the other candidates anything, `_yolo11n_candidate`'s `predict_fn` now converts back to BGR (`_rgb_to_contiguous_bgr`) before calling `model.predict`, rather than leaving it wrong. Noted directly in that function's own comment for whoever writes the P1.6 detector wrapper: YOLO is not part of that wrapper's real candidate set, so this conversion is benchmark-only and must not be copied into `perception/detector.py` (RGB there is for RF-DETR/MediaPipe only, AGENTS.md rule 16). 2 new tests (round-trip and direct channel-order assertion).
- **Finding 3 (report hygiene) -- fixed:** `--out` now defaults to `data/benchmark_cpu_provisional.json` (git-ignored), never `reports/benchmark_cpu.json`. Every report unconditionally carries `"provisional": true` and a `"not_measured"` list (hands, the fine-tuned detector, real footage), even the empty-candidates smoke case.
- **Finding 4 (run-order / drift labelling) -- fixed with a real redesign, not a relabelling:** the previous "run 1 / run 2" numbers were two full script invocations in a fixed candidate order (RF-DETR first, YOLO last every time), which confounds *when in the sequence* a candidate ran with *which candidate* it was -- exactly why run 2 looked uniformly faster for every candidate. `run_suite` now runs >= 3 repeat rounds (default 3), shuffling candidate order independently each round with a `random.Random(seed)` (reproducible from `--seed`), each round re-runs its own warm-up (`benchmark_predict`'s existing warmup/timed split, called once per round rather than once globally), and a configurable cooldown (default 1s) separates every timed block. `torch.set_num_threads(num_threads)` is (re-)asserted immediately before *every single run*, not just once at start, since the earlier session observed it silently drift (6 -> 8) after importing `ultralytics`; the actual `torch.get_num_threads()` and (for the ONNX candidate) its configured `intra_op_num_threads` are recorded on every run record, not just in the environment block. `build_report` now separates `per_candidate_summary` (each candidate's own per-repeat values and spread) from `position_effect` (mean latency pooled by in-round slot -- 1st thing run that round, 2nd, ... -- across whichever candidate happened to land there), so a systematic slot-1-vs-slot-2 trend and a systematic candidate-A-vs-candidate-B difference no longer read as the same signal. `--num-threads` is a new CLI flag (default: logical core count via `os.cpu_count()`); per this session's task, the operator runs the suite once at the physical core count and once at the logical count in Phase 2 -- this session does not pick or hardcode either, since neither is knowable in general without an extra dependency (`psutil`) this session did not request.
- **Finding 5 (cosmetic) -- fixed:** `ruff format --check training/benchmark_cpu.py` is clean on the rewritten file (verified, no reformat needed). The stale "12 tests" and weight-file-note corrections below replace fixing those lines in place (`ISSUES.md` is append-only).
- **Stale-line corrections** (append-only -- the original lines are left exactly as written; treat these as superseding them): the 2026-09-29 "P1.5-provisional - CPU latency benchmark: harness built and tested" entry's "12 tests" is now out of date twice over -- it was 17 by the time of the PR #3 review and is 33 after this session's additions (51 F14 tests total with `test_record.py`'s 18). The 2026-09-29 "P1 note - model weight files needed for real numbers" entry's `Status: OPEN (awaiting go-ahead to download either or both)` is superseded by the 2026-09-29 "P1.5-provisional RESULTS" entry below it, where both files were downloaded, cached, and used -- neither file is missing on this machine any more.
- `git grep -nE 'output_dir\s*=\s*"output"|YOLO\("[a-z0-9_]+\.pt"\)'` over `training/` finds exactly 2 matches, both inside docstrings/comments describing the historical bug and why it's avoided (`benchmark_cpu.py:43` and `:124`) -- zero matches in executable code. The actual calls now read `model.export(output_dir=str(_ONNX_EXPORT_CACHE_DIR), ...)` and `YOLO(str(_YOLO_CACHE_PATH))`.
- `python scripts/check.py`: 115 passed, 1 skipped, ruff clean. `--status`: F1 11/11, F3 22/22, F14 51/51 GREEN. `ruff format --check` clean on `training/benchmark_cpu.py`, `tests/unit/training/test_benchmark_cpu.py` and `scripts/check.py` (the last was already clean; not touched).
- No new downloads or installs this session (standing decision) -- `onnxruntime`/`onnx` remain not importable in this venv, confirmed again; the ONNX candidate correctly self-skips (`collect_real_candidates` returns it as absent) exactly as before. A small integration smoke test (`--allow-download`, cached RF-DETR/YOLO weights only, 2 tiny repeats) confirmed the refactored real-candidate path still produces correct, well-shaped `per_run`/`per_candidate_summary`/`position_effect` output; no `.pt`/`.onnx`/`.pth` file or `output/` directory was left anywhere under the repo afterward (checked).
- Status: RESOLVED (P1; Phase 1 complete). Phase 2 (the real ONNX export, the real parity check, and a controlled re-measure of all four candidates) is explicitly blocked on P2's R7 dependency PR (`onnxruntime` into the `tools` group) landing on `origin/develop`, per this session's own instruction -- not started.

---

## 2026-09-29 P2 review of PR #5 (Phase 1 follow-up to the provisional CPU benchmark): PASS, merged; non-blocking notes for P1's Phase 2
- Reviewed head `6c17bf5` (unchanged since the request). Detached worktree; only `uv sync --group dev --group tools` from the lockfile as it stood on the PR's base; no model file downloaded, nothing else installed.
- **Evidence:** `python scripts/check.py`: 115 passed, 1 skipped, ruff clean. `--status`: F1 11, F3 22, F14 51, all GREEN. `ruff check` and `ruff format --check` clean on `training/benchmark_cpu.py` and `tests/unit/training/test_benchmark_cpu.py` (and `scripts/check.py`). Still 115 passed, 1 skipped after merging onto `77f1322` (PR #6, the R7 deps). The diff is exactly `training/benchmark_cpu.py`, its test file and 19 lines appended at the END of `ISSUES.md`: nothing in `pyproject.toml`, `uv.lock`, `contracts.py`, `config/`, `reports/`, `.gitignore`, no stray files. One commit; no `*.pt`/`*.pth`/`*.onnx`/`*.task` and no `output/` in it. 33 tests in the file (17 old + 16 new): fakes and injected clocks, no network, none imports `onnx`/`onnxruntime`. No detector chosen, no `target_fps`, `acceptance.yaml` untouched, no measured number presented as a result.
- **The five PR #3 findings, checked against the installed `rfdetr==1.11.0` source, not the summary:** (1) the ONNX export dir is absolute and outside the repo (`~/.cache/rfdetr_benchmark`); `export()` returns a `Path`, takes `fp16`/`verbose`/`output_dir` as used, and its default `output_dir="output"` is CWD-relative as claimed. `preprocess_to_nchw` and `decode_detections` exist in `rfdetr.export._runtime`, are the two functions rfdetr's own ONNX inference helper calls, are underscore-private, and need only numpy/PIL/torchvision (already installed). Boxes are scaled to the frame with `pil_img.size` = (width, height) and clipped; `threshold=DETECTOR_MIN_CONF` is passed; `background_class_id=None` matches the decode docstring for sparse COCO (91 logits, per the R7 entry). (2) YOLO gets BGR through `_rgb_to_contiguous_bgr`. (3) The default `--out` is under `data/` (git-ignored); every report has `provisional: true` and `not_measured`. (4) Seeded per-round shuffle, 3 repeats, per-repeat warm-up, `torch.set_num_threads` re-asserted before every run and recorded, ONNX `intra_op_num_threads` recorded, no cool-down after the last run. (5) `ruff format` clean; the "12 tests" and weight-file corrections are accurate. Every checkable factual claim in P1's new entry is true (the `git grep` claim: exactly 2 hits, both in docstrings, lines 43 and 124; 33 + 18 = 51 F14 tests; 4 parity + 2 extraction tests; `fp16=True` library default). Two small inaccuracies are in notes 5 and 7.
- **Non-blocking notes for P1 (fix before Phase 2 numbers are interpreted; no ack needed):**
  1. **`position_effect` is still confounded with candidate identity (most important).** The arithmetic is right (hand check: slot 1 of (10, 14) gives 12; slot 2 of (20, 22) gives 21), but a pooled slot mean is dominated by which candidate landed in the slot, so it can show a position effect with no drift at all. Reproduced with the real functions: constant latencies A=10 ms and B=20 ms, orders AB, AB, BA, give slot 1 = 13.33 and slot 2 = 16.67. With four candidates and three shuffled rounds (seed 0) the slots are not balanced, so this will occur in Phase 2. The docstring claim that a slot trend points at drift "rather than a slow candidate" is not true as written. Fix: pool each run's latency relative to its own candidate's mean (ratio or delta), or use a balanced (Latin-square) order, and add a test where constant-latency candidates give a flat position effect. `per_run` and `per_candidate_summary` are unaffected.
  2. **The YOLO BGR wiring is untested.** The two colour helpers are tested (round trip, channel order), but nothing checks that `_yolo11n_candidate`'s `predict_fn` really calls `model.predict` with BGR. A fake `ultralytics` module injected via `sys.modules` would cover it with no model or network; the same technique would let `_rfdetr_onnx_candidate`, which has no test at all, be tested against fakes.
  3. **The parity check has no end-to-end path yet.** `check_raw_output_parity` and `extract_pytorch_raw_outputs` are tested only in isolation; nothing feeds one preprocessed tensor through both backends and stops on failure. Phase 2 must add that runner and stop before measuring if `within_tolerance` is False. `extract_pytorch_raw_outputs` also omits the `_optimized_dtype` cast that `predict()` applies on the optimized path: use it on the non-optimized model only, or add the cast.
  4. **Private-API dependency.** The ONNX candidate uses `rfdetr.export._runtime.*` (private) while `pyproject.toml` allows `rfdetr>=1.10,<2`; it is verified only at the locked 1.11.0. An `ImportError` is swallowed at INFO level, so after an upgrade the ONNX row would silently vanish. Suggest a WARNING (or failure) when `--allow-download` is set and the ONNX candidate is absent, and recording `rfdetr.__version__` in the report. The import guard covers `onnxruntime` only; a failure inside `model.export()` (e.g. `onnx` missing) is not caught and would abort the whole suite, which is acceptable now that R7 put `onnx` in the lock, but keep the error message clear.
  5. **Timed-path comparability is described more evenly than it is.** All rows share BGR->RGB plus the four stages, but `predict()` also does class-name mapping and builds a `supervision.Detections`, while the ONNX row adds `PIL.Image.fromarray` and returns a bare `DecodedDetections`. Name these extras in the docstring so the ONNX-vs-PyTorch delta is read correctly. The comment "Matches rfdetr's own `_create_onnx_session`" is half true: rfdetr disables both `intra_op` and `inter_op` spinning, the harness only `intra_op`.
  6. **Cosmetic:** `_make_candidate(name, delay_s)` ignores `delay_s`; `test_run_suite_records_torch_threads_on_every_run` leaves the process at 2 torch threads (restore it); the shuffle test only asserts more than one distinct order.
  7. **Stale after this merge:** P1's entry and the module docstring say `onnxruntime` is not in the lockfile. PR #6 (R7) has since landed on `origin/develop` (`77f1322`), so it now is (`tools` group: `onnx`, `onnxruntime`), and Phase 2's stated prerequisite is met. After the next `uv sync --group tools` the ONNX candidate will no longer self-skip.
- **Not verifiable by me:** PR #5's description. The repository is private, `gh` is not installed on this machine, and an unauthenticated fetch returns 404. The commit, its message and its head match the reported `6c17bf5` and P1's ISSUES.md entry; P1 should confirm the PR body lists the same three files. The PR was merged locally with `git merge --no-ff` and pushed to `develop` (as for PR #3), so GitHub may need the PR closed by hand.
- Status: DECIDED (merged into `develop`; the notes above are non-blocking follow-ups for P1, no ack needed). Next: P1 Phase 2 (real ONNX export, real parity check, controlled re-measure); fix note 1 first, or its position-effect numbers will mislead.

---

## 2026-09-29 P2 review of PR #7 (P1 Phase 2: Part A harness fixes + Part B provisional CPU measurements): BLOCKING, not merged
- Reviewer: P2. Head reviewed: `a9bc8e9` (unchanged since the request; three commits `50dc345`, `ecaf984`, `a9bc8e9` on top of `origin/develop` `99ba719`, so no rebase is needed). Reviewed in a detached worktree using the existing `.venv` (`uv sync --group dev --group tools --frozen --offline --dry-run`: "would make no changes"); nothing downloaded or installed, the real benchmark not run, no camera touched. PR comments could not be posted (`gh` not installed, repo private); this entry is the review record.
- **Passed:** (1) `python scripts/check.py`: 123 passed, 1 skipped; `--status`: F1 11/11, F3 22/22, F14 59/59 GREEN (P1's counts reproduced: 122 at Part A, +1 test since). `ruff check .` clean. `ruff format --check` is clean on both PR files; whole-repo it flags 3 files this PR does not touch (`contracts.py:94`, `perception/hands.py:106`, `tests/unit/perception/test_record.py:313`), pre-existing. (2) Scope: only `training/benchmark_cpu.py`, its test file and append-only `ISSUES.md` (entries at the end); no `pyproject.toml`/`uv.lock`/`contracts.py`/`config/`/`reports/`/`.gitignore`. (3) No `.pt/.pth/.onnx/.task/.json` or `output/` in any of the three commits. (4a) The position effect is normalised per candidate; my AB, AB, BA example (A=10 ms, B=20 ms) gives `relative_mean` 1.0 in both slots (`test_summarize_by_position_reports_no_drift_for_constant_candidates_p2_repro`, hand-checked). (4d) WARNING log + `skipped` list + library versions are in code and tested. (4e) YOLO BGR wiring (asserts the pixel order `predict` receives) and `_rfdetr_onnx_candidate` (fake `sys.modules` stack: both views, export dir, `fp16=False`, both spinning flags, ImportError path) are tested. (4f) The three cosmetics are fixed (`delay_s` gone; thread count restored in `finally`; shuffle test asserts slot-1 occupancy varies). (4g) The stale "onnxruntime not in the lockfile" line is corrected in P1's Part A entry, at the end. Part B's per-repeat means, fps (= 1000/mean), spreads and the reported floor figures all recompute (differences of 0.1 ms are rounding of one-decimal inputs). Honesty items in check 7 hold: PROVISIONAL; COCO-pretrained; hands, fine-tuned head and real footage not measured; the 8 fps floor described as end-to-end; no detector chosen, no `target_fps`, `acceptance.yaml` untouched; the High Performance switch and Balanced restore are recorded.
- **BLOCKING B1 -- the "3900 queries for the plain forward" finding is a training-mode artifact, and it is baked into code, tests and the report.** Evidence, all from rfdetr 1.11.0's source plus P1's own run order (nothing executed):
  1. `rfdetr/config.py` (`ModelConfig.group_detr` docs, line ~511): "`num_queries * group_detr` predictions are produced in **training mode**; `num_queries` in **eval mode**". Nano: `num_queries=300` (`RFDETRBaseConfig`), `group_detr=13` default, and 300 x 13 = **3900 exactly**. `models/lwdetr.py:487-493` picks all query groups iff `self.training`.
  2. A freshly built `RFDETRNano()` is in training mode. The only things that put its `model.model.model` into eval are `predict()` (`_ensure_eval_mode_for_unoptimized_inference`, `detr.py:2616`), `.inference()` and `prepare_export_graph` (`export/prepare.py:303`, on the copy). `extract_pytorch_raw_outputs` calls `model.model.model(tensor)` directly, so a `pytorch__network_only` run made *before the first `predict()` on that same object* runs in training mode: 3900 queries, dropout active.
  3. P1's own seed reproduces exactly that. `random.Random(0)` over the 7-candidate list gives round 1 = `ort_full, opt_full, pt_nw, pt_full, ort_nw, opt_nw, yolo`: `pytorch__network_only` runs at global position 3, **one slot before** `pytorch__full` (its first `predict()`); in rounds 2 and 3 `pt_full` has already run. That is the repeat-1 outlier (288.5 ms / 321.2 ms, 2.3x/2.4x the same row's repeats 2-3) in both passes, and the elevated slot 3 (~1.21) and depressed slots 1 and 6 (`pt_nw` repeats 2-3 at 0.69-0.71 of a mean inflated by the outlier). The Part B guess ("allocator pool growing once for that size, unconfirmed") and "Not a harness bug" are both wrong: it is a harness bug (the candidate never sets eval mode) and the cause is known.
  4. Consequences. (a) `FORWARD_PATH_INFO["rfdetr_nano_pytorch"]["queries"] = 3900` is false for every `rfdetr_nano_pytorch__full` run (`predict()` sets eval first: 300 queries) and for `__network_only` repeats 2-3; it is `test_build_report_names_the_forward_path_and_query_count_per_backend`'s asserted value and goes into every report's `forward_paths`. (b) Part B's table and its conclusion "optimized and onnx are the query-count-matched pair, not plain and onnx; the plain row processes 13x more query slots per frame" are false: in eval all RF-DETR rows are 300 queries, so `pt_full` (7.6 / 7.2 fps) is directly comparable to `opt_full` (7.96 / 7.95) and `ort_full`. (c) `pytorch__network_only`'s reported means (180.6 / 196.3 ms, 5.54 / 5.09 fps) mix one training-mode repeat with two eval repeats; P1's advice to read repeats 2-3 (126.7 / 133.8 ms, 7.9 / 7.5 fps) is the right reading and should be stated as the row's value. (d) The Part A "real bug found wiring the parity check" story is misdiagnosed: the `(3900, 4)` vs `(300, 4)` mismatch came from the untouched training-mode model, not from a different forward method. Also `build_pytorch_export_mode_forward` does **not** run `forward_export`: `prepare_export_graph` only freezes DINOv2 position embeddings and calls `.eval()`; `_switch_to_export_mode` is applied later in `ExportBase.__call__` (`export/base.py:283`), and the function indexes `output["pred_boxes"]` as a dict, which `forward_export` (returns a tuple) would not allow. The parity result itself (box 2.99e-5, logit 8.68e-5) is still valid and useful, but it compares the ONNX graph against the **eval-mode plain `forward()`** (with frozen position embeddings), not against `forward_export`; its docstring, `extract_pytorch_raw_outputs`'s docstring, and the Part A/B narrative say otherwise.
  5. Why blocking: this is a false, decision-relevant statement (which backend rows are comparable) written into code, a test, every future report and the ISSUES record, and the same harness defect would silently corrupt any P1.5 re-run whose shuffle puts `pt_nw` first. The fix is small.
- **Required to unblock (P1, before re-review):** (1) make the network-only PyTorch candidate run under eval (call `model.model.model.eval()` once when building it, or call rfdetr's `_ensure_eval_mode_for_unoptimized_inference()`), and add a fake-model test where the inner model records `.training` and asserts it is False on the call; (2) set `queries` for `rfdetr_nano_pytorch` to 300 (or drop the field) and change the test; (3) correct the docstrings of `extract_pytorch_raw_outputs` / `build_pytorch_export_mode_forward` / `run_parity_check` to say the reference is eval-mode `forward()` after `prepare_export_graph`, not `forward_export`; (4) one confirming check on P1's laptop (a few lines, no benchmark): `RFDETRNano().model.model.training` before and after one `predict()`, and the output shape of `model.model.model(t)` in each state; (5) append a correction entry at the END of `ISSUES.md` superseding the Part A item 3 diagnosis and Part B's query-count table, "optimized and onnx are the matched pair" and outlier explanation; (6) re-measure only if you want clean `pytorch__network_only` numbers: repeats 2-3 of the existing runs are already the eval numbers, so a rerun is optional and P2 does not require it.
- **Non-blocking (fix in the same push, or log):**
  1. **ONNX `__full` vs `__network_only` gap.** `ort_full` is 154.8 ms at 6 threads and 122.2 ms at 12, while `ort_nw` is 98.8 / 100.8. Not an outlier: the per-repeat ratios are 1.29 / 1.25 / 1.26. The time outside `session.run` is 55.9 ms (6t) vs 21.4 ms (12t), against ~6-9 ms (`opt`) and ~5 ms (`pt`, eval) for `predict()`'s own pre/post. So the ONNX candidate's `PIL.fromarray` + `preprocess_to_nchw` + `decode_detections` cost is thread-count-sensitive and 2-6x more than `predict()`'s; P1's "the one row that visibly improved" reports the effect without saying where the time is. It matters for the eventual pre/post cost, which the real pipeline may do with cv2 rather than rfdetr's helper.
  2. **"Clear the floor with real margin" is applied to a non-end-to-end row.** Part B's floor paragraph counts `ort_nw` (~10 fps) as clearing 8 fps, but `network_only` excludes preprocessing and decode, and the same backend's `__full` is 6.5 / 8.2 fps. The summary's *conclusion* (no RF-DETR row has comfortable headroom, only YOLO11n) stands; the `ort_nw` phrasing should not be quoted as headroom. `opt_full` at 12 threads has repeats 7.84 / 7.99 / 8.03 fps, so "at or under 8 outright" holds only for the mean.
  3. **Median not reported next to the mean.** Only mean and spread are in the report; with one outlier per row the median is the robust figure (e.g. `pt_nw` median 128.5 / 136.0 ms vs mean 180.6 / 196.3). Add `median_ms` per candidate.
  4. **Position effect, same seed both passes:** disclosed, and slot 3's ~1.21 matches `pt_nw` r1 (1.60 relative) landing there. Not disclosed: the depressed slots 1 and 6 come from the same outlier through the inflated baseline (`pt_nw` r2-3 = 0.69 / 0.71), so "every other slot within 0.99-1.03" applies only after setting aside all three `pt_nw` runs. Use a different seed per thread-count pass.
  5. **Scope of `__network_only`.** Every row goes through `benchmark_predict`, which times `_bgr_to_contiguous_rgb` (a 1280x720 copy) on each call; the PyTorch `__network_only` also times `.float().cpu().numpy()` of the outputs, ONNX does not. Small, but the "network only" label is loose.
  6. **Dependency on rfdetr internals.** `rfdetr.export.prepare.prepare_export_graph` has no underscore but is not re-exported from `rfdetr.export` (its `__init__` exports nothing); `rfdetr.export._runtime.*` and `_switch_to_export_mode` are private. `pyproject.toml` pins `rfdetr>=1.10,<2`, not 1.11.0, and neither the docstring nor the ISSUES entry says the harness needs the verified 1.11.0; the `import rfdetr.export.prepare` in `build_pytorch_export_mode_forward` is not inside the ImportError-to-`skipped` handling, so an upgrade would surface as a traceback at parity time. Say "verified against rfdetr 1.11.0" in the module docstring.
  7. **Coverage:** `build_pytorch_export_mode_forward`, `run_phase2_parity_check`, `_build_shared_parity_tensor` and `_rfdetr_pytorch_candidates` have no test; the fake inner model has no train/eval state, which is why B1 was not caught.
  8. **Earlier provisional pass not marked superseded.** Part B says the two passes stay comparable and that the ONNX gap is closed, but nothing tells a reader the earlier RESULTS entry's numbers are replaced. Add one line at the end of the correction entry.
- **Not verifiable by me:** the raw per-run JSON (git-ignored, on P1's laptop only), so the position-effect values were checked for consistency (slot 3 ~ 1.21 follows from the 1.60 outlier at position 3) rather than recomputed from records; the operator's plugged-in / battery-saver / plan confirmations in chat. Library versions in P1's entry match this environment (`torch` 2.14.0, `rfdetr` 1.11.0, `onnxruntime` 1.30.0, `ultralytics` 8.4.164).
- Status: OPEN (PR #7 **not merged**; verdict BLOCKING on B1). Nothing else in the PR needs rework. On re-review I will check (1)-(5) above and merge if they hold; if P1 thinks B1 is wrong, item (4) settles it. No detector decision, `target_fps` or `acceptance.yaml` change is affected or implied by this review.
## 2026-09-29 P1 fix - Part A: fixes all seven PR #5-review findings before any Phase 2 measurement
- Measurement-only session: no detector chosen, no `target_fps` set, `config/acceptance.yaml` untouched. Test-first as before. Addresses every non-blocking note from the 2026-09-29 "P2 review of PR #5" entry above, in order:
  1. **`position_effect` confounding (the most important one) -- fixed.** `summarize_by_position` now normalizes each run's `mean_ms` by *that run's own candidate's* mean across all its repeats *before* pooling by slot, instead of pooling raw `mean_ms` values. Reproduced the reviewer's exact fixture (constant A=10ms/B=20ms, orders AB/AB/BA) as a test: the raw pooled version reports slot 1 = 13.33ms / slot 2 = 16.67ms (a fake "position effect"); the normalized version reports `relative_mean = 1.0` at every slot for both candidates, i.e. correctly no drift. Report key renamed `mean_ms_avg` -> `relative_mean` so old and new can't be confused. `per_run`/`per_candidate_summary` untouched, as the reviewer noted.
  2. **`_rfdetr_onnx_candidate` and the YOLO BGR wiring were untested -- both now covered.** `_yolo11n_candidate` is exercised end to end with a fake `ultralytics` module injected via `sys.modules` (asserts the array `model.predict` actually receives is BGR, not just that the two helper functions round-trip). `_rfdetr_onnx_candidate` is exercised against a fully faked `rfdetr` + `onnxruntime` stack (also via `sys.modules`) -- no model, no export, no network -- covering both its `__full`/`__network_only` views, the export dir, `fp16=False`, both thread-config entries, and its `ImportError` path (see point 4).
  3. **The parity check had no end-to-end runner -- added, and it found a real bug while wiring it to a real model.** `run_parity_check(pytorch_raw_fn, onnx_raw_fn, tensor_torch, tensor_numpy, box_atol, logit_atol)` feeds one preprocessed tensor through both backends and raises `ParityError` if they disagree beyond tolerance; unit-tested pass/fail with fakes. `extract_pytorch_raw_outputs` now also applies the `_optimized_dtype` cast `predict()`'s own optimized path applies before calling `inference_model` (was silently omitted; a new test confirms the wrapper actually casts, not just dispatches). **Real-model finding, not assumed:** wiring the real parity check (`run_phase2_parity_check`) to an actual RF-DETR-Nano checkpoint failed with a shape mismatch, `(3900, 4)` vs `(300, 4)` -- `extract_pytorch_raw_outputs`'s plain `model.model.model(tensor)` forward (what `predict()`'s own eager path also calls) returns 3900 query/class pairs on this checkpoint, but the actual exported ONNX graph's `dets`/`labels` outputs are `(1, 300, *)`. Read `rfdetr/models/lwdetr.py` and `rfdetr/export/prepare.py` to find why: the export pipeline switches the model into a *second* forward method (`forward_export`) via the private `rfdetr.export._backend._switch_to_export_mode` before tracing. Calling that switch directly (bypassing the library's own preparation) raises inside `transformer.py` -- the DINOv2 backbone's position embeddings must first be frozen to the export shape, which only `prepare_export_graph` does correctly. Fix: added `build_pytorch_export_mode_forward(model)`, which reuses rfdetr's own `prepare_export_graph` (AGENTS.md rule 1 -- reuse libraries, don't reimplement) on a **deep copy** of the model (the switch is one-way and would silently corrupt any later `predict()` call on the original), verified to produce exactly the `(1, 300, *)` shape the real `.onnx` file reports. `extract_pytorch_raw_outputs` stays correct as-is for the `__network_only` *timing* candidate (it measures exactly what `predict()` itself pays); it is simply the wrong function for the *parity* comparison, and both docstrings now say so explicitly. **Real result** (cached weights, no download, small scale): `max_box_abs_diff` ≈ 3e-5, `max_logit_abs_diff` ≈ 8.7e-5, both far inside tolerance (`box_atol=1e-3`, `logit_atol=1e-2`) -- `within_tolerance: true`. The export and ONNX Runtime agree with eager PyTorch (in export mode) almost exactly, as expected for a same-precision (fp16=False) float32 graph.
  4. **Private-API dependency -- addressed.** `_rfdetr_onnx_candidate`'s `ImportError` now logs at `WARNING` (was `INFO`) and returns a `skip_reason` string that `collect_real_candidates` puts in the report's new `"skipped"` list (`[{"candidate": ..., "reason": ...}]`) alongside every other candidate that didn't build (rfdetr missing, ultralytics missing). `environment_info` now records `rfdetr`, `onnxruntime` and `ultralytics` versions (`torch` already was) in every report, via `importlib.metadata.version` where a module has no `__version__` attribute of its own (`rfdetr` doesn't; confirmed by checking, not assumed).
  5. **Timed-path comparability and the `allow_spinning` comment -- both corrected, not just reworded.** The module docstring now names the actual asymmetry: RF-DETR's `predict()` also does COCO class-name mapping and builds a `supervision.Detections`; the ONNX candidate additionally does a `PIL.Image.fromarray` and returns a bare `DecodedDetections` -- neither is "just" resize+normalize+forward+decode, and each backend now has an explicit `__full` (the whole path, including that backend's own extras) and `__network_only` (one fixed preprocessed tensor, the bare forward repeated, isolating architecture cost from preprocessing/decode) row, so a reader can see which comparison they're making. The `allow_spinning` comment previously claimed to match rfdetr's own `_create_onnx_session` while only setting `session.intra_op.allow_spinning` -- rfdetr's own helper sets both `intra_op` and `inter_op`; this session's `_rfdetr_onnx_candidate` now sets both too (read `export/_onnx/inference.py` again to confirm exactly what it sets, rather than assume the earlier comment was accurate).
  6. **Cosmetics -- fixed.** `_make_candidate`'s `delay_s` parameter is gone (was ignored everywhere); `test_run_suite_records_torch_threads_on_every_run` now restores `torch.get_num_threads()` in a `finally` block instead of leaving the process at 2 threads for whatever test runs next; the shuffle test now asserts a specific, meaningful property (no single candidate is stuck in slot 1 across every round, over 8 rounds x 5 candidates) instead of only "more than one distinct order occurred".
  7. **Stale "onnxruntime is not in the lockfile" -- corrected.** PR #6 (R7) merged (`77f1322`) before this session started; `onnxruntime`/`onnx` are in the `tools` group now. The module docstring's "not in this project's lockfile at all" line is removed; this session's own real runs used the real `onnxruntime` (confirmed: `library_versions.onnxruntime = "1.30.0"` in the report).
- **Tests:** 40 in `tests/unit/training/test_benchmark_cpu.py` (was 33) -- all against fakes; no test imports a real model, network, or the real `onnxruntime`/`rfdetr` internals. `python scripts/check.py`: 122 passed, 1 skipped, ruff clean; `--status`: F1 11/11, F3 22/22, F14 58/58 GREEN. `ruff format --check` clean on `training/benchmark_cpu.py`, its test file, and `scripts/check.py` (the last untouched).
- **Real integration smoke tests this session** (cached weights only, no new downloads, small scale: 1-2 repeats, 1-2 timed frames): confirmed the two-view candidates, the `skipped` list, library-version recording, and (after the `forward_export` fix above) the real parity check all work end to end against actual RF-DETR-Nano PyTorch, RF-DETR-Nano ONNX Runtime, and YOLO11n. No `.pt`/`.pth`/`.onnx` file or `output/` directory left anywhere under the repo afterward (checked via `git status`).
- Harness commit for these results: see the commit hash on `p1-perception` this entry's PR carries.
- Status: RESOLVED (P1; Phase 1's PR #5 follow-up complete). Proceeding to Part B (the real, controlled multi-repeat measurement) only after confirming plug-in/power-plan status in chat, per this session's own instruction.

---

## 2026-09-29 P1.5-provisional RESULTS - Part B: controlled multi-repeat CPU latency (real parity, real measurement)
- **Harness commit**: `ecaf984` on `p1-perception` (PR #7). One code addition after Part A's fixes, before any timed run: `build_report` now records a `"forward_paths"` entry per candidate naming which forward method it calls and its query count (see the query-count finding below) -- also test-covered, `check.py` green before measuring.
- **Environment**: this machine (confirmed demo laptop), AMD Ryzen 5 5600H, 6 physical / 12 logical cores, Windows 11 (build 26200). **Plugged in, battery saver off, heavy apps closed -- confirmed by the operator in chat before any timed run** (per this session's own gate: "if any of the three is no or still blank, do not start"). **Power plan: switched to High Performance (`powercfg /setactive SCHEME_MIN`) before the first timed run, confirmed via `powercfg /getactivescheme`, restored to Balanced (`SCHEME_BALANCED`) after the last one** -- matches the earlier provisional pass, so the two sets of numbers stay comparable. Library versions (recorded in every report, this session's own addition): `torch` 2.14.0+cpu, `rfdetr` 1.11.0, `onnxruntime` 1.30.0, `ultralytics` 8.4.164. No CUDA.
- **Parity check (B3): PASSED, both thread-count runs** -- `max_box_abs_diff` ≈ 2.99e-5, `max_logit_abs_diff` ≈ 8.68e-5 (identical to the millisecond across both runs, since it depends only on the model/export, not on thread count), both far inside tolerance (`box_atol=1e-3`, `logit_atol=1e-2`). ONNX Runtime and PyTorch's export-mode forward agree almost exactly. Both timed runs proceeded past this gate.
- **Query-count finding, verified empirically before measuring (not assumed), answering the operator's specific ask:**

  | Candidate family | Forward path timed | Queries |
  |---|---|---|
  | `rfdetr_nano_pytorch` (plain, unoptimized) | plain `forward()` -- what `predict()`'s own eager path calls internally | **3900** |
  | `rfdetr_nano_pytorch_optimized` (`.inference()`-compiled) | torchscript-traced `inference_model` -- **also** reduced internally, verified empirically | **300** |
  | `rfdetr_nano_onnxruntime` | the exported ONNX graph (`forward_export`, switched in via the private `_switch_to_export_mode` before tracing) | **300** |
  | `yolo11n_pytorch` | YOLO11n's own architecture -- not a DETR query head | n/a |

  **This means "optimized" and "onnx" are the query-count-matched, apples-to-apples pair -- not "plain" and "onnx".** The plain/unoptimized PyTorch row processes 13x more query slots per frame than either the optimized or the ONNX row; its latency is not directly comparable to ONNX's on architecture grounds alone, independent of backend. Both `__full` and `__network_only` rows below are labelled by candidate family, so this table applies to all of them.
- **Results, both timing views, both thread counts, all 3 repeats** (20 warm-up frames excluded, 200 timed frames per repeat, synthetic 1280x720 BGR frames, seed 0, 1s cooldown between every run, candidate order shuffled independently each round):

  **Physical cores (`--num-threads 6`):**

  | Candidate | Repeat 1 / 2 / 3 (ms) | Mean (ms) | fps (of mean) |
  |---|---|---|---|
  | `rfdetr_nano_pytorch__full` | 128.4 / 132.1 / 133.9 | 131.5 | 7.61 |
  | `rfdetr_nano_pytorch__network_only` | **288.5** / 124.8 / 128.5 | 180.6 (skewed, see below) | 5.54 (skewed) |
  | `rfdetr_nano_pytorch_optimized__full` | 118.7 / 130.7 / 127.5 | 125.6 | 7.96 |
  | `rfdetr_nano_pytorch_optimized__network_only` | 112.0 / 118.9 / 119.1 | 116.6 | 8.57 |
  | `rfdetr_nano_onnxruntime__full` | 151.1 / 156.2 / 157.0 | 154.8 | 6.46 |
  | `rfdetr_nano_onnxruntime__network_only` | 96.0 / 96.4 / 104.1 | 98.8 | 10.12 |
  | `yolo11n_pytorch__full` | 38.5 / 36.7 / 34.8 | 36.7 | 27.27 |

  **Logical cores (`--num-threads 12`):**

  | Candidate | Repeat 1 / 2 / 3 (ms) | Mean (ms) | fps (of mean) |
  |---|---|---|---|
  | `rfdetr_nano_pytorch__full` | 140.8 / 138.9 / 136.9 | 138.9 | 7.20 |
  | `rfdetr_nano_pytorch__network_only` | **321.2** / 136.0 / 131.6 | 196.3 (skewed, see below) | 5.09 (skewed) |
  | `rfdetr_nano_pytorch_optimized__full` | 127.5 / 125.2 / 124.5 | 125.8 | 7.95 |
  | `rfdetr_nano_pytorch_optimized__network_only` | 124.2 / 117.1 / 116.4 | 119.3 | 8.39 |
  | `rfdetr_nano_onnxruntime__full` | 117.0 / 125.0 / 124.6 | 122.2 | 8.18 |
  | `rfdetr_nano_onnxruntime__network_only` | 104.9 / 98.6 / 98.8 | 100.8 | 9.92 |
  | `yolo11n_pytorch__full` | 39.5 / 42.2 / 42.0 | 41.2 | 24.26 |

  More logical threads than physical cores did not help much here (SMT on a 6c/12t part rarely helps a single dense-math forward pass) -- most rows land within a few ms either way; `rfdetr_nano_onnxruntime__full` is the one row that visibly improved (154.8ms -> 122.2ms).
- **The repeat-1 outlier, both runs, same candidate:** `rfdetr_nano_pytorch__network_only`'s first repeat is ~2.2-2.4x its own repeats 2/3 in *both* independent runs (288.5ms vs ~125-129ms at 6 threads; 321.2ms vs ~132-136ms at 12 threads) -- every other candidate's repeats stay within a few ms of each other. Not a harness bug: `benchmark_predict`'s own 20-frame warm-up ran before every repeat, including this one. Read as a one-time cold-start cost specific to this candidate's first invocation in a fresh process -- it is the single largest-tensor path in the whole suite (3900 queries vs 300 for every other RF-DETR row), so a memory-allocator pool growing once for that size, not yet needed by anything before it in the shuffle, is the likely mechanism; unconfirmed, logged as an observation, not a claim. Reported honestly above rather than discarded -- the mean/spread the reader takes from this row should be repeats 2-3 (~125-136ms), not the reported mean, which is skewed by repeat 1.
- **Position effect (normalized, this session's Part A fix)**: both runs show slot 3 elevated (~1.21, relative to 1.0 = no drift) and slots 1/6 depressed (~0.87-0.90). **This is not independent confirmation of a real position-based drift**: both runs used the same `seed=0`, so the per-round shuffle order is identical between them -- slot 3 of round 1 (global position 3) is mechanically the *same* candidate-repeat combination in both runs, which is exactly the repeat-1 outlier above landing at position 3 both times. Every other slot's normalized value sits within about 0.99-1.03 of 1.0 in both runs, i.e. no drift once that one outlier is set aside. A future run with a varied seed per thread-count pass would decorrelate this if it matters again.
- **Against `min_pipeline_fps = 8`** (`config/acceptance.yaml`'s not-yet-written judgment call; the operator's framing: an **end-to-end** floor, so the detector alone must sit well below the 125ms frame budget to leave room for hands/state/engine/log): only `yolo11n_pytorch__full` (24-27fps) and `rfdetr_nano_onnxruntime__network_only` (~10fps) clear it with real margin. Everything else sits at or barely past the boundary either way: `rfdetr_nano_pytorch_optimized__network_only` (8.4-8.6fps) and `rfdetr_nano_onnxruntime__full` (6.5fps at 6 threads, 8.2fps at 12) are marginal; `rfdetr_nano_pytorch_optimized__full` (7.95-7.96fps), `rfdetr_nano_pytorch__full` (7.2-7.6fps) and `rfdetr_nano_pytorch__network_only`'s real (non-outlier) repeats (~125-136ms, ~7.35-8.0fps) sit at or under 8fps outright. **On this provisional, COCO-pretrained, `__full`-path basis, none of the RF-DETR rows leave comfortable headroom for the rest of the pipeline; only YOLO11n does.** This is a provisional observation, not a decision -- P1.5 re-measures on the fine-tuned head (different class count, possibly different output-decoding cost) and against `acceptance.yaml`'s actual recall thresholds, and is the only session that chooses a detector or sets `target_fps`.
- **What was NOT measured**: hands (no `.task` file vendored, per the 2026-09-28 DECISION); the fine-tuned detector (COCO-pretrained only, 91 logit slots vs `config/experiment.json`'s five classes); anything on real Sample Transfer footage (synthetic random frames only). ONNX Runtime's real numbers *are* now measured (unlike the first provisional pass) -- that gap from the earlier RESULTS entry above is closed.
- **`check.py --status`** on this commit: F1 11/11, F3 22/22, F14 59/59 GREEN (123 passed, 1 skipped overall). Raw JSON for both runs kept locally only, git-ignored (`data/phase2_physical6.json`, `data/phase2_logical12.json`, under `.gitignore`'s `data/` entry) -- not committed, per the standing instruction not to invent a tracked path under `reports/` (P2-owned).
- Status: PROVISIONAL (informational only; no detector chosen, no `target_fps` set -- both remain P1.5's).

## 2026-09-30 P1 DECISION - crew footage intake: real objects diverge from DATA_COLLECTION.md's prop guidance (decide at the P1.2 spike)

**Context:** P1 footage-intake session (Good=22, Bad=24 videos, 848x480 H.264 30fps, sent via WhatsApp). Visually
inspected mid-frames of x001.mp4 (good) and x023.mp4 (bad) against `config/experiment.json` and
`DATA_COLLECTION.md` section 1's object table. Not a code or contract problem -- `config/experiment.json`
itself matches the crew's step ids/order/rules exactly (verified: `red_out, red_in_tray, yellow_out,
yellow_in_tray, start_pressed, red_stowed, yellow_stowed`, canonical order 1-7). The mismatch is between the
*written prop guidance* and what the crew actually built, which matters for two future steps that don't exist
yet: (a) whoever writes the zero-shot auto-labeling text prompts (DATA_COLLECTION.md section 9: "a zero-shot
detector draws boxes from text prompts ('red box', 'green button')" -- no `training/prompts.yaml` exists yet,
so nothing is broken today, but a prompt written from the doc as-is would be wrong), and (b) any crew member
recording more sessions from the doc alone.

1. **`start_button` is not the documented green coaster.** It is a small white lined index card with "START"
   handwritten in blue ballpoint, taped down. Confirmed in both sampled frames. Fine for a trained detector
   (class-based, not color-based) but a "green button" text prompt for auto-labeling would fail outright, and
   DATA_COLLECTION.md section 1's table is now wrong for any crew member using it to set up a fresh rig.
2. **The `tray` is a grey hardcover book/notebook**, not a "flat plastic lunch-box lid, baking tray or
   placemat" in "blue, black or white." It sits where the doc's layout diagram puts the tray and both
   containers rest on top of it in the "good" frame exactly as the procedure describes, so it functions
   correctly as a flat surface -- but its color (grey) is not one of the three listed, and a grey object
   next to a grey/dark table risks lower contrast than the doc's color rule was written to guarantee.
3. **The yellow container's actual hue leans lime/olive-green, not saturated yellow**, and both lids carry
   faint white embossed/printed lettering. DATA_COLLECTION.md's `red_box`/`yellow_box` row calls for
   "saturated," "matte," and (for red) "no white lettering across the top" specifically to keep red vs.
   yellow separable and clean for the detector; the yellow lid's green-shifted hue is the closer of the two
   containers to `outer_box`/background confusion territory and the closest to accidentally reading as a
   third, undocumented color.

**Not blocking Phase 1/2 of this footage-intake session** (labels and copies proceed from the crew's own
"good"/"bad" declaration, not from prop color). Flagging because it affects: (a) whoever builds the zero-shot
auto-label prompts next, and (b) DATA_COLLECTION.md section 1's table, which is now stale against the actual
rig. Suggested next step (not taken here -- no code/doc edits in this session): P1 or Data crew either
(i) swap the START object for something closer to the doc's green coaster before recording rows 13-77, or
(ii) update DATA_COLLECTION.md section 1 to match the rig that was actually built and carry the color values
forward into the zero-shot prompts when `training/prompts.yaml` is written.

- Status: OPEN -- decide at the P1.2 spike (keep the rig as-is vs. swap START/tray props before rows 13-77).

## 2026-09-30 P1 fix - PR #7 review B1 correction
- Fixes P2's BLOCKING B1 (2026-09-29 review, above): required items 1-4 only, per this session's own scope
  (deferred items listed as OPEN at the end). Commits on `p1-perception`: `585857a` (eval-mode fix + fake-model
  test asserting `.training` flips True->False), `8883263` (`FORWARD_PATH_INFO` query-count correction + test),
  `7e9f326` (docstring corrections). `scripts/check.py` green after each commit.
- **Confirming check (required item 4), done in this session, no timing, no download** (cached weights only,
  `C:\Users\HP\.roboflow\models\rf-detr-nano.pth`): a fresh `RFDETRNano()`'s inner module reports
  `.training == True` before any call; the FIXED `extract_pytorch_raw_outputs`, called first (before any
  `predict()` on that model object -- the exact ordering B1 described), now returns `pred_boxes.shape[0] == 300`
  and leaves `.training == False` afterward. This reproduces the bug scenario against the real checkpoint and
  confirms the fix, not just the fake-model tests.
- **Supersedes the following** (not edited in place -- `ISSUES.md` is append-only -- superseded by this entry):
  1. **Part A's item 3 "real bug found wiring the parity check" diagnosis** (2026-09-29 P1 fix entry, item 3):
     the `(3900, 4)` vs `(300, 4)` mismatch was **not** caused by `forward_export`/`_switch_to_export_mode`
     being a different forward method. It was training vs. eval mode: a fresh `RFDETRNano()` starts in training
     mode, `rfdetr/models/lwdetr.py:487` branches query count on `self.training` (`num_queries*group_detr`=3900
     in training, `num_queries`=300 in eval), and the plain PyTorch path was never eval'd before this fix.
     `prepare_export_graph` only calls `.eval()` (plus DINOv2 shape-freezing) -- confirmed by reading it --
     `_switch_to_export_mode` is applied later, inside `ExportBase.__call__`, never by
     `build_pytorch_export_mode_forward`.
  2. **Part B's query-count table and "optimized and onnx are the query-count-matched pair, not plain and
     onnx" statement** (2026-09-29 P1.5-provisional RESULTS -- Part B entry): false. With the fix, all three
     RF-DETR-Nano forward paths (plain, optimized, onnx) are 300 queries. `pytorch__full` was always directly
     comparable to `optimized__full` and `onnxruntime__full` -- the `__full` rows were never affected by this
     bug (`predict()` has always set eval mode itself); only `pytorch__network_only`'s bare
     `extract_pytorch_raw_outputs` calls were exposed.
  3. **The repeat-1 outlier explanation** ("allocator pool growing once for that size, unconfirmed" -- same
     Part B entry): wrong. `pytorch__network_only`'s repeat 1 ran in training mode (3900 queries, real extra
     compute) because P1's seed=0 shuffle put it at global position 3, one slot before that model object's
     first `predict()` call in both timed passes; repeats 2-3 ran after `predict()` had already set eval mode.
     **`pytorch__network_only`'s repeat 1 (288.5 ms at 6 threads, 321.2 ms at 12) is a training-mode number and
     must not be quoted as this candidate's latency. Repeats 2-3 (~124.8-128.5 ms at 6 threads, ~131.6-136.0 ms
     at 12) are the valid eval-mode numbers** -- P1's original advice to read repeats 2-3 was the right call,
     for the wrong reason (harness bug, not cold-start).
  4. **The earlier `2026-09-29 P1.5-provisional RESULTS - CPU latency: RF-DETR-Nano (COCO-pretrained) vs
     YOLO11n, two runs each` entry is superseded** by the later Part B entry, per P2's finding F8 -- nothing
     previously said so.
- **F2 (where ONNX's full/network_only gap goes):** not a bug. `rfdetr_nano_onnxruntime__full` minus
  `rfdetr_nano_onnxruntime__network_only` is about 55.9 ms at 6 threads (154.8-98.8) and 21.4 ms at 12
  (122.2-100.8) -- the ONNX candidate's `PIL.fromarray` + `preprocess_to_nchw` + `decode_detections` cost,
  thread-count-sensitive, 2-6x `predict()`'s own pre/post. Matters for the eventual pipeline's own pre/post
  cost, which may use cv2 rather than rfdetr's helper.
- **F3 (floor wording fix):** Part B's floor paragraph must not present `rfdetr_nano_onnxruntime__network_only`
  (~10 fps) as clearing `min_pipeline_fps=8` "with real margin" -- `network_only` excludes preprocessing and
  decode, and the same backend's `__full` is only 6.5/8.2 fps. The paragraph's overall conclusion ("no RF-DETR
  row has comfortable headroom, only YOLO11n does") stands; only the `network_only` framing must not be quoted
  as an end-to-end headroom claim.
- **Deferred, OPEN, to finish before the P1.5 re-benchmark (not done this session -- required items only):**
  - F4: add `median_ms` per candidate to `build_report` (recomputable from the repeat values already published
    above; no new measurement needed).
  - F5: `pytorch__network_only`'s timed region includes `.detach().float().cpu().numpy()`; the ONNX
    `network_only` path does not do an equivalent conversion. Asymmetric; fix or document the scope.
  - F6: wrap the `from rfdetr.export.prepare import prepare_export_graph` import in
    `build_pytorch_export_mode_forward` in the same ImportError-to-skip pattern used elsewhere in this file;
    state "verified against rfdetr 1.11.0" in the module docstring (`pyproject.toml` pins `>=1.10,<2`).
  - F7: add tests for `build_pytorch_export_mode_forward`, `run_phase2_parity_check`, `_build_shared_parity_tensor`,
    `_rfdetr_pytorch_candidates` (none exist).
  - F9: use a different shuffle seed per thread-count pass so the position effect isn't correlated between the
    two passes (already disclosed as a caveat in the Part B entry; not yet fixed).
- **No re-measurement was run or required for B1** -- repeats 2-3 of the existing Part B runs are already valid
  eval-mode numbers, per P2's own required-item 6. This session ran no timing benchmark and downloaded nothing.
- Status: RESOLVED (B1 only; required items 1-4). Deferred items above remain OPEN. PR #7 not merged -- awaiting
  a separate fresh review session per AGENTS.md step 7 (waived only for this fix session by the 2026-09-30 P1
  DECISION entry below).

## 2026-09-30 P1 DECISION - P2 unavailable: Lead merges P1 PRs and covers P2 work
- While P2 is unavailable, the Lead (P1) merges P1's own PRs into `develop` without P2's review, gated on
  (i) `scripts/check.py` green and (ii) a review by a separate fresh agent session using the existing PR-review
  checklist (`IMPLEMENTATION_PLAN.md` Part 8's "PR checklist, self-certified" plus the reviewer conventions
  P2's own past entries in this file follow) -- the reviewing session must be distinct from the one that wrote
  the PR, so a fix is never self-certified by the same context that made it.
- **Contract changes are excluded from this arrangement.** `contracts.py` and `config/experiment.json` still
  require an explicit written approval line by the Lead, made in a separate session from the one proposing the
  change -- same-session self-approval of a contract change is never allowed, waiver or not.
- **P2 work the Lead takes on runs as its own "I am P2" session**, on `p2-runtime`, touching only P2's owned
  directories (`state/`, `engine/`, `outputs/`, `runtime/`, `server/`, `harness/`, `scripts/`, `config/`, the
  lockfile, `reports/` per `IMPLEMENTATION_PLAN.md` Part 10) -- never mixed into a P1 session or a P1 PR.
- **P2 retro-reviews every PR merged this way once back**, using the same checklist, and may request follow-up
  changes as a normal post-hoc review.
- Why: P2 is genuinely unavailable and work must continue; the fresh-session review substitutes for a second
  person's eyes without pretending the Lead reviewed their own same-session work.
- Status: OPEN.
## 2026-09-28 P2.2 CONTRACT - RunSummary.run_id has no way to reach SequenceEngine
- Missing: `contracts.Engine`'s constructor is `SequenceEngine(experiment, config)` and `start(self, t: float) -> None` -- neither carries a `run_id`. `RunSummary.run_id: str` is required, so something has to supply it before a `run_completed` event can be built.
- Why it matters: without a value, either `RunSummary.run_id` construction fails or the engine has to invent one, and P2.4's runtime (which owns `run_id` generation per Plan 5.6, "arms a fresh idle run with a new run_id") has no documented way to pass it in.
- Proposed fix (additive, backward-compatible): `start(self, t: float, run_id: str = "") -> None` -- an optional keyword the runtime supplies; omitted, it defaults to `""`. This satisfies the `Engine` Protocol structurally (extra optional params don't break `@runtime_checkable` isinstance checks) without touching `contracts.py`.
- Status: OPEN (worked around at P2.2 via the additive `start(t, run_id="")` signature in `engine/sequence.py`; P2.4's runtime should pass its generated `run_id` through this parameter once it exists). A human should confirm at the next contract-touching session whether this should be formalized in `contracts.py`'s `Engine` Protocol docstring.

## 2026-09-28 P2.2 DECISION - SequenceEngine.start() now rejects being called mid-run; also, no engine method returns "idle"
- Found during a deep edge-case review of `engine/sequence.py`: originally `start()` would silently reset all state and jump straight to "running" no matter what `run_state` currently was -- so if a caller invoked `start()` again while a run was in progress, that run's `run_completed` event (and its `RunSummary`) would be lost with no trace, no exception, nothing in the log. Fixed: `start()` now raises `ContractViolation` if `run_state == "running"`, forcing the caller to `finish()` first. Verified this matches Plan 5.6's documented runtime behavior ("`POST /api/run/reset` finishes the current run... and arms a fresh idle run") and does not break the "reusable across runs" design -- `start()` still works fine from `"idle"` or `"completed"`. Regression tests: `test_start_while_running_raises_contract_violation`, `test_engine_is_reusable_after_finish`, `test_engine_is_reusable_after_natural_completion` in `tests/unit/engine/test_sequence.py`.
- Separately noted, not fixed (out of P2.2's scope, flagging for P2.4): Plan 5.6 says reset "arms a fresh **idle** run", but `contracts.Engine` has no method that ever produces `run_state == "idle"` again once a run has started -- `finish()` only ever produces `"completed"`, and `contracts.RunState` is the same `Literal["idle","running","completed"]` type shared by `Engine.run_state` and `StatusResponse.run_state`/`RunControlResponse.run_state`. Either the runtime's `/api/run/reset` handler needs to treat a freshly-`finish()`ed engine (`run_state == "completed"`, not yet re-`start()`ed) as displaying `"idle"` in its own `StatusResponse` bookkeeping, or `contracts.py` needs an explicit idle-arming affordance. This is Plan 5.4 vs. 5.6 territory boundary (5.4 is P2.2's; 5.6 is P2.4's) -- flagging for whoever builds `runtime/loop.py` and `server/app.py` to resolve explicitly rather than guess.
- Status: OPEN (advisory for P2.4; the `start()` re-entrancy fix above is DONE).
- Update 2026-09-28 (P2.4): resolved at the runtime layer -- see the 2026-09-28 P2.4 DECISION entry below ("resolves the P2.2 'Engine never reports idle again' gap").

---

## 2026-09-28 P2.4 CONTRACT - ReplayResult is referenced but not defined in contracts.py
- Missing: IMPLEMENTATION_PLAN.md 5.9 ("`python scripts/replay.py --from-cache | --video | --script` ... replay a run through tracker + engine -> `ReplayResult`") and Part 10's P2.4 packet both name `ReplayResult` as the return type of `harness/replay.py`'s replay functions, but `contracts.py` has no such type.
- Why it matters: `harness/replay.py` needs a concrete return shape now, and P2.6 (tuning) and P2.7 (test-split reporting) will also consume it -- they may need more than this session can anticipate (e.g. a pass/fail verdict against `RunScript.expected_deviations`).
- Proposed fix / what was tried: worked against a local `TEMP_1_ReplayResult` (a plain pydantic `BaseModel`, not in `contracts.py`) in `harness/replay.py` -- `run_id`, `frames_processed`, `engine_events: list[EngineEvent]`, `summary: RunSummary | None`. Deliberately minimal; callers can derive observed deviations / POS / alert counts from `engine_events`/`summary` without needing more fields yet. A human should confirm the final shape in a `contract/<slug>` PR once P2.6/P2.7 know what they actually need from it.
- Status: OPEN

## 2026-09-28 P2.4 DECISION - live runtime run_id format
- Decision needed: `RunSummary.run_id`/`LogEntry.run_id` need a value for runtime-generated runs (`POST /api/run/start` / `/api/run/reset`, P2.5), distinct from the crew's recorded `<split>-<n>` ids (`RunScript`, `runs/run_plan.csv`).
- Why it matters: `runtime/loop.py`'s `Router` owns run_id generation per the P2.2 CONTRACT entry above ("arms a fresh idle run with a new run_id"), and the per-run log/video filenames are named by it (F9, F11).
- Decided: `live-<UTC timestamp %Y%m%dT%H%M%SZ>-<6 hex chars>` (`runtime.loop.default_run_id`), injectable via `Router(..., run_id_factory=...)` for tests/replay. The random suffix keeps two resets within the same second from colliding.
- Status: DECIDED (P2.4); open for ack before P2.5 wires it into the server, in case a different convention is preferred.

## 2026-09-28 P2.5 DECISION - StatusResponse.expected_step_id/next_step_id semantics when polled (not event-driven)
- Context: contracts.py's EngineEvent docstring pins expected_step_id/next_step_id as "the first pending step before/after this event" -- a meaning tied to a specific transition. StatusResponse carries fields with the same names but GET /api/status is polled on a timer (essential-features.md #12), with no in-flight event to read them from.
- Decided (server/app.py, `_step_guidance`): derived straight from Engine.snapshot()'s canonical-order step list (no contracts.py change needed). expected_step_id = the first still-pending step (what the operator should be doing now); next_step_id = the pending step after that; next_step_say = that next step's `say` text, so the dashboard's single "current/next step" element (essential-features.md #12 point 1) can render both from one poll. Regression test: tests/unit/server/test_app.py::test_status_matches_contract_and_reflects_idle_engine.
- Why it matters: a different but equally plausible reading existed (e.g. expected_step_id = the step most recently confirmed). Flagging so P1 can ack the chosen reading before the dashboard is relied on during rehearsal.
- Status: DECIDED (P2), open for P1 ack.

## 2026-09-28 P2.5 DECISION - the Flask URL map is exactly API_ROUTES, so server/static/ files are inlined, not served by a separate route
- Context: API_ROUTES's docstring says "the Flask URL map must equal it" (exactly, per essential-features.md #10's contract line), but essential-features.md #12's implementation section separately lists server/static/index.html, app.js, style.css, vendor/ as files a browser would normally fetch via their own routes -- which Flask's default static-file handling would add as an extra `/static/<path:filename>` rule outside API_ROUTES.
- Decided: server/app.py disables Flask's automatic static route (`static_folder=None`) and instead reads style.css/app.js once at app-creation time and inlines them into the single `GET /` response (server/app.py's `_render_index`); the three files still exist separately on disk under server/static/ for editing. This satisfies the "exactly" wording literally rather than by an implicit carve-out for a framework-internal route. Regression test: tests/unit/server/test_app.py::test_url_map_equals_api_routes_exactly.
- Why it matters: the alternative reading (url_map equality excludes Flask's own built-in `static` endpoint) is also defensible and more conventional; flagging so P1 can ack the chosen reading.
- Status: DECIDED (P2), open for P1 ack.

## 2026-09-28 P2.4 DECISION - resolves the P2.2 "Engine never reports idle again" gap
- Context: the 2026-09-28 P2.2 DECISION entry above flagged that `contracts.Engine` has no method that ever reproduces `run_state == "idle"` after a run has started (`finish()`/natural completion both leave it at `"completed"`), and asked whoever builds `runtime/loop.py` to resolve it explicitly.
- Resolved: `runtime/loop.py`'s `Router` tracks its own `_idle_armed` flag (set at construction and by `reset()`, cleared by `start()`); its `run_state` property reports `"idle"` whenever that flag is set, regardless of what `Engine.run_state` itself says. `contracts.py` is untouched -- the gap is closed at the runtime layer, as the P2.2 entry's second option proposed. Regression tests: `tests/unit/runtime/test_router.py::test_router_reports_idle_before_start_and_after_reset`, `::test_router_reset_while_idle_is_a_noop_for_the_engine`.
- Status: RESOLVED (P2.4).

---

## 2026-09-30 P2 (covered by the Lead) - p2-runtime rebased onto develop, plus one cross-role P1 fix

- **Who:** the Lead, running this session as P2 per the 2026-09-30 "P1 DECISION - P2 unavailable" entry above.
  P2-role only for everything except the one fix below: touched only `p2-runtime`, its own owned directories,
  and this file. No edit to `training/`, `contracts.py` or `config/experiment.json`.
- **Cross-role authorization (the one exception):** this session's pre-commit and pre-push hooks both run the
  entire `scripts/check.py`, unscoped to what is being committed -- so the `perception/`-side bug found below
  blocked *every* commit on this branch, including pure P2 work, not just anything touching `perception/`.
  P1 (the owner of `perception/`), in chat, authorized this session -- for this one fix only -- to edit
  `perception/record.py` and `tests/unit/perception/test_record.py` on `p2-runtime`. Everything else in this
  session stayed inside P2's owned directories. Commits `c961206` (the fix) and `bde32ec` (the unrelated P2
  ruff fix, below) are separate and clearly labelled.
- **Safety backup, before anything else:** `origin/p2-runtime` was confirmed still at `f0e294d` (its P2.5
  round-2 head), then pushed unchanged to a new branch `origin/p2-runtime-pre-rebase` (old head `f0e294d`) so
  the pre-rebase state is recoverable. Nothing on `origin/p2-runtime` was overwritten by that push.
- **Rebase:** `git rebase origin/develop` onto `ea15a3e` (PR #7 merged: P1 Phase 2 Part A/B + the B1 correction
  + the P2-unavailable DECISION). **Zero conflicts** -- all 14 P2 commits (P2.1 StateTracker through P2.5
  round-2) replayed cleanly; `p2-runtime` is now 46 commits ahead of the old merge base. `ISSUES.md`'s
  append-only/merge=union convention was never exercised because there was nothing to merge.
- **F7 correction to this session's own brief:** the task brief for this session described F7
  (`test_degrades_quietly_when_engine_init_fails`, `tests/unit/outputs/test_tts.py`) as red with
  `ModuleNotFoundError: No module named 'tests.unit'` inside a spawned child process on Windows. That does
  **not** reproduce on this rebased branch: the test passed 5/5 in isolation and in the full suite, both before
  and after the rebase. Checked why: `_raising_engine_factory` (the picklable-under-`spawn` stand-in for a
  broken audio engine that this test passes to `TTSWorker`) has been a module-level function, with a docstring
  explicitly calling out "Module-level (picklable under spawn) ...", since its very first commit
  (`f4fade0`/`3886716`, P2.3: Outputs) -- `git log -p` on `tests/unit/outputs/test_tts.py` shows no version of
  this file ever defined it as a local/nested function. There is nothing to fix here; no code was changed for
  F7. The operator's hypothesis for the earlier red -- a one-off spawn artefact from running in a detached
  worktree -- is plausible but **unverified**; this session found no evidence either way, only that it is not
  the current, reproducible state. `check.py --status`: **F7 22/22 GREEN.**
- **Real finding instead -- F14 was RED, one failure, root-caused and fixed under the cross-role
  authorization above (commit `c961206`):** `tests/unit/perception/test_record.py::test_validate_run_flags_unknown_step_id`
  failed with an unhandled `contracts.ContractViolation: unknown step_id 'not_a_real_step'` raised from
  `engine/sequence.py:72` (P2-owned, working exactly per contract: it rejects an event for a step id the
  experiment doesn't define). The call path: `perception/record.py`'s `validate_run` (P1-owned) already
  detected unknown `performed_steps` at line 397-399 and correctly appended
  `"performed_steps references unknown step ids: [...]"` to `result.issues` -- but then unconditionally called
  `_expected_deviations_stale(script, experiment_path)` at line 439 regardless, which called
  `engine.reference.derive_expected_deviations` (P2-owned) with the same invalid `performed_steps`, which fed
  them to a real `SequenceEngine`, which correctly raised rather than silently accepting an undefined step.
  The exception was never caught, so `validate_run` crashed before it could return the `ValidationResult` it
  had already started building. **This was always latent, not introduced by this rebase or by any P2 code in
  it**: `_expected_deviations_stale`'s own comment says `# engine/reference.py hasn't landed yet (P2.2)` -- on
  `develop` alone (without P2's branch), importing `engine.reference` raises `ImportError`, which
  `_expected_deviations_stale` caught and turned into a silent no-op (`return None`), so the crash was
  structurally unreachable until a branch carrying both P1's test and P2's `engine/reference.py` existed at
  once. This rebase was the first time that happened. **Fix (commit `c961206`, cross-role, P1's files):** in
  `validate_run`, skip the call to `_expected_deviations_stale` entirely when `unknown_steps` is already
  non-empty -- the staleness check is meaningless for a script with invalid step ids, and the issue is already
  recorded. `engine/sequence.py`'s hard rejection of unknown step ids is correct behavior and was not touched
  or loosened. Test-first: added
  `test_validate_run_flags_stale_expected_deviations_for_valid_step_ids` (real, valid step ids that skip
  `red_in_tray`, stale `expected_deviations=[]`) first, confirmed it already passed against the *unfixed* code
  (proving the staleness check itself works and nothing else needed to change), confirmed
  `test_validate_run_flags_unknown_step_id` was red, then applied the one-line guard. Both tests, and the full
  `tests/unit/perception/test_record.py` (19 tests), pass after the fix.
  `check.py --status`: **F14 61/61 GREEN** (was 60 total; +1 for the new coverage).
- **Walking-skeleton smoke test (no camera, no audio):** `python -m harness.replay --script <RunScript JSON>
  --experiment fixtures/experiment_4step.json`, performed_steps `["s1", "s3", "s4"]` (step `s2` omitted) against
  the `fixture_4step` experiment. Ran end to end through `SequenceEngine` + `Router` + `JsonlLogger` (this
  mode bypasses `StateTracker` by design, per `harness/replay.py`'s own docstring, to exercise the router/log
  path without needing `PerceptionFrame` data): 3 frames processed, the omitted `s2` produced exactly one
  `deviation_detected` event ("Step skipped: Step two") and one matching JSONL line, run completed with
  `pos=0.75`, `skipped_step_ids=["s2"]` -- the GOLD-1 pattern (a skipped step -> exactly one alert, one log
  line), confirmed working through the loop. `runs_out/` (git-ignored) and the temporary script JSON were
  removed after; `git status` is clean.
- **Note for P2:** re-fetch -- your local `p2-runtime` is stale. Reset it to `origin/p2-runtime` after reading
  this entry (`git fetch origin && git reset --hard origin/p2-runtime`, after saving any local-only work you
  don't want to lose). Nothing you built was changed in content, only rebased onto `develop`'s new head.
- **Retro-review still owed:** per the P2-unavailable DECISION above, P2 retro-reviews this rebase (and every
  PR the Lead merged while P2 was out) once back, using the normal PR-review checklist -- this now includes
  retro-reviewing the cross-role `perception/record.py` fix too, since P2 didn't write or review it either.
- **Unrelated pre-existing lint fix, in P2's own directories (commit `bde32ec`):** `check.py --quick` runs
  `ruff check .` before pytest, and its output was truncated out of view in this session's own first two checks
  (only the pytest tail was inspected) -- `ruff` was actually failing the whole time on two pre-existing `I001`
  (unsorted import block) findings, in `harness/replay.py` and `scripts/dev.py` (both P2-owned). Confirmed via
  `origin/p2-runtime-pre-rebase` that both predate this session and this rebase entirely -- not introduced by
  anything here. Fixed with `ruff check --fix` (pure import reordering, `perception.pipeline` before
  `perception.camera`; no behavior change, both still imported lazily as before).
- **Why this took four commits, not one:** the pre-commit/pre-push hooks run the whole suite unscoped, so
  nothing could be committed at all until the `perception/` blocker was cleared -- see the cross-role
  authorization above. Kept separate on purpose: `c961206` (P1 fix, cross-role), `bde32ec` (P2 ruff fix), this
  entry (P2, `ISSUES.md` only), on top of the plain rebase.
- Status: RESOLVED for this branch. `python scripts/check.py`: full suite green. `python scripts/check.py
  --status`: **F1-F14 all GREEN** (F2 has no tests, as before). Nothing in this entry is merged into
  `develop`; the PR from `p2-runtime` into `develop` is opened by the operator, by hand.

## 2026-09-30 P1 R7 - kaggle CLI, dev-time only, installed outside the lockfile

- **Package:** `kaggle` 2.2.4, licence Apache-2.0.
- **Where:** installed as an isolated `uv tool` (`uv tool run kaggle ...`), **not** added to this project's
  lockfile or any dependency group -- it is dev-time-only tooling for the P1.5 GPU training environment
  (IMPLEMENTATION_PLAN.md 3.2), never a runtime dependency, and never present on the offline demo laptop.
- **Auth:** a new-style Kaggle API token (`~/.kaggle/access_token`, outside the repo, never printed, read or
  copied by this session). Authentication was verified by the user beforehand and re-checked here with one
  read-only call, `uv tool run kaggle datasets list --mine`, which listed the account's existing datasets
  without creating, modifying or deleting anything.
- **Dataset creation rule (not yet done):** this project's own dataset, when created, must be a **private**
  dataset under its own slug -- suggested `hemachandhara/sih26174-frames` -- and must not touch, modify or
  delete any of the account's other listed datasets.
- Status: DECIDED. Read-only auth check done; no dataset created yet.

## 2026-09-30 P1 DECISION - runs/provenance.csv added; adopting the 46 intake clips

- **Context:** the 46 clips in `data/intake/` (labels.csv + map.csv + working/<id>.mp4, all git-ignored; see
  the 2026-09-30 "crew footage intake" DECISION above) needed turning into `runs/<id>/{video.mp4,script.json}`
  the way Stage 0 (`perception/record.py`) would have produced directly, so they feed F14 stages 2+ normally.
  Built `perception/adopt.py` (test-first: `tests/unit/perception/test_adopt.py`, a tiny generated clip + a
  fake labels.csv/map.csv, per essential-features.md's F14 done-when).
- **`runs/provenance.csv` (new file, tracked):** `contracts.RunScript` has no field for the crew's original
  filename, sha256, crew label, label source or reviewer notes -- those don't belong in the derived,
  contract-typed `script.json`. `IMPLEMENTATION_PLAN.md` Part 4 lists `runs/` as "Data crew + P1", and
  provenance-of-a-recording is exactly that kind of joint metadata, not a P2-owned shape, so it goes in a
  sibling CSV rather than stretching the contract. Columns: `run_id, original_name, sha256, crew_label,
  label_source, checked_by, notes`. No `contracts.py` change proposed -- this is intake bookkeeping, not a
  P1/P2 boundary shape.
- **`--min-duration-s` added to `perception/record.py`'s Stage 1 validator** (default unchanged at 20.0s,
  tested by `test_validate_run_min_duration_s_default_is_unchanged`): the 46 intake clips are real crew
  footage from before this project's 20-150s recording convention existed, and the shortest (`x023`, an idle
  run) is 4.0s. `--validate runs/ --min-duration-s 3` is the value used for this one adoption batch, chosen
  to sit below the shortest real clip (4.0s) without being so low it would mask a genuinely truncated file.
  This does not change the recorder's own default band for anything recorded going forward with
  `--plan runs/run_plan.csv` (rows 13-77, still 20-150s).
- **Step order verified, not assumed:** `perception.adopt.verify_step_order` asserts
  `config/experiment.json`'s live `step_ids` equal `[red_out, red_in_tray, yellow_out, yellow_in_tray,
  start_pressed, red_stowed, yellow_stowed]` (the order labels.csv's 1-7 codes were written against) before
  any file is touched, and aborts the whole batch otherwise.
- **Split:** seeded (seed 0), stratified by `script_type`, provisional -- target counts train/val/test:
  correct 14/4/4, skip 5/2/2, swap 5/2/2, repeat 1/1/1, idle 1/1/1 (26/10/10 of the 46). A `--split-file`
  override exists for a human-adjusted split later without touching the tool.
- **`expected_deviations` is derived, never hand-typed:** `perception/adopt.py` calls
  `engine.reference.derive_expected_deviations` directly (R3, the same sanctioned P1->P2 call
  `perception/record.py`'s Stage 1 already makes) -- no reimplementation.
- Status: DECIDED. `runs/` ownership read as "Data crew + P1" per Part 4; provenance.csv and the
  `--min-duration-s` flag are additive, not a contract or `IMPLEMENTATION_PLAN.md` change. Open for P2's ack
  on the `runs/provenance.csv` file existing at all, since `runs/` is jointly owned.

## 2026-09-30 P1 R7 - MediaPipe hand_landmarker.task vendored in weights/

- **File:** `weights/hand_landmarker.task` (the bundle essential-features.md section 3 and
  `perception/hands.py`'s `DEFAULT_HAND_MODEL_PATH` both name), downloaded once, dev-time, from the official
  MediaPipe Solutions download URL documented at
  https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker (section "Models"):
  `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task`.
  **Only this one file was downloaded** -- no pose model (`pose_landmarker_lite.task`, still off by default
  per essential-features.md section 3 point 5 and not needed by any Tier-1 rule), no auto-labeler weights,
  nothing else.
- **Integrity:** sha256 `fbc2a3008...` (full digest and size in `weights/MANIFEST.json`, the tracked file --
  `.gitignore`'s `weights/*` / `!weights/MANIFEST.json` pair means the `.task` file itself stays git-ignored,
  as `IMPLEMENTATION_PLAN.md` Part 4 specifies). Size 7,819,105 bytes (~7.46 MiB); file starts with a `PK` zip
  header, consistent with MediaPipe's `.task` bundle format (a zip of a tflite model + metadata), not an HTML
  error page.
- **Licence:** Apache-2.0 (matches `IMPLEMENTATION_PLAN.md` 3.1's stack table entry for MediaPipe hands+pose;
  cross-checked against MediaPipe's own model-card language, which states Apache-2.0 for this task bundle).
- **`weights/MANIFEST.json` written** (did not exist before this session -- no detector yet, P1.5): currently
  only the `hand` entry (`file`, `sha256`, `size_bytes`, `source_url`, `license`); `detector` and `pose` keys
  are added by whoever vendors those (P1.5, and P1.6 if pose is ever enabled). No code reads this file yet
  (`perception/hands.py`'s `weights_sha256` re-hashes the `.task` file directly, per the 2026-09-29 P1.3 review
  note already in this file); this MANIFEST is Plan 5.8's tracked record, not a runtime dependency.
- **Smoke test** (no tuning, no accuracy claim -- see below), ~40 real frames spread across multiple adopted
  clips (`runs/*/video.mp4`, this session's own checkpoint-2 output): see the result entry immediately
  following this one.
- Status: DONE (download + MANIFEST). Smoke test result in the next entry.

## 2026-09-30 P1 RESULT - hand_landmarker smoke test on 40 real frames (no tuning, no accuracy claim)

- **Method:** `HandTracker` (`perception/hands.py`, default options, VIDEO mode) run over 40 frames: 5 frames
  each, evenly spread across each clip's full duration (not just the first second), from 8 of the newly
  adopted `runs/*/video.mp4` clips -- `x001`/`x009` (correct), `x024`/`x042` (skip), `x025`/`x030` (swap),
  `x038` (repeat), `x023` (idle) -- so the sample isn't all one script_type. `tracker.reset()` between clips
  (per essential-features.md section 3 point 2, timestamps restart at each run); per-frame timestamps kept
  monotonically increasing within a clip. Wall-clock `time.perf_counter()` used only to measure latency in
  this throwaway measurement script (not `perception/` library code -- P1 rule 8 doesn't apply to it, same as
  `training/benchmark_cpu.py`'s own timing). Not committed to the repo (scratch script, not library code).
- **Results:** hands found in **27/40 frames (68%)**. Handedness reported as `Right` 27 times and `Left` 2
  times across those frames (informational only per essential-features.md section 3 point 4 -- MediaPipe
  assumes a mirrored/selfie image and this rig's overhead, non-mirrored camera means the label should not be
  trusted at face value; the count exceeding 27 in a few frames means both hands were found in the same
  frame). Mean latency of the `process()` call (landmark inference; decode/IO not isolated separately): **55.0
  ms/frame** (min 28.0, max 76.8 ms) on this dev laptop's CPU. **Caveat, disclosed rather than hidden:** this
  includes 8 "cold" first calls (one per clip, right after each `reset()` recreates the landmarker) with no
  warm-up exclusion, and is a single-call, no-batching measurement -- not comparable to
  `training/benchmark_cpu.py`'s methodology (warm-up frames excluded, repeated timed passes) and must not be
  quoted alongside those numbers or as a per-frame pipeline-fps figure.
- **Not claimed:** accuracy (no ground-truth hand boxes/keypoints exist to score against), gloved-hand
  behavior (none of the 8 sampled clips happen to show gloves; the robustness runs mentioned in
  essential-features.md section 3 aren't part of this batch), or full pipeline fps (this measures the hand
  tracker alone, not the per-frame path `training/benchmark_cpu.py` times).
- Status: DONE, informational only. No thresholds tuned, no config changed.

## 2026-09-30 S-C note - S-B PR reviewed after the fact

S-B PR merged before its review by the Lead's mistake; reviewed after the fact in S-C; result: PASS -- scope
clean (perception/, tests, runs/ script.json+provenance.csv+manifest.csv, weights/MANIFEST.json, ISSUES.md
only; no video/.task/credential/binary tracked); `adopt.py --dry-run`'s split table reproduced identically;
`weights/MANIFEST.json`'s sha256/size match the vendored file exactly; the hand smoke test's detection counts
reproduced exactly (27/40, Right 27/Left 2 -- deterministic model inference), latency reproduced within
expected wall-clock variance (51.3ms vs. the logged 55.0ms mean, both wall-clock timing, not deterministic);
x014 and x023's "frozen" flags are both false positives of the validator's first-30-frames-only sampling --
full-clip mean frame diff is 1.464 (x014) and 0.543 (x023), neither near-zero, so neither clip is actually
frozen; `check.py` full suite green (395 passed) and `--status` all GREEN (F14 78/78). `p1-perception`
fast-forwarded to `origin/develop` (`f327ba2`) and pushed, no force needed.

## 2026-09-30 P1.2 R7 - Grounding DINO tiny: license recorded; checkpoint 1 spike results

- **License** (resolves the 2026-09-28 "licenses of auto-labeling tools not yet verified" DECISION for this
  tool): Apache-2.0, `IDEA-Research/grounding-dino-tiny` (Hugging Face model card, cross-checked at
  https://huggingface.co/IDEA-Research/grounding-dino-tiny). Downloaded once via `transformers` (already
  installed, 5.17.0, not a new dependency) to the default Hugging Face cache (outside the repo, like the
  RF-DETR/YOLO checkpoints at P1.5 -- never at `weights/`, never committed). Only `model.safetensors` (689,359,096
  bytes, sha256 `1a2412ef99bd74bc...`, full digest in `data/spikes/report.json`'s `model` key, git-ignored)
  downloaded -- `pytorch_model.bin` (a duplicate in another format, also listed in the repo) was not fetched,
  since `safetensors` is already installed and `transformers` prefers it. Total download ~690 MB, under the
  1.5 GB approval.
- **YOLO-World was not tried this session** -- only Grounding DINO tiny, per the checkpoint 1 scope. The
  license/speed/accuracy comparison DECISION is checkpoint 3's, not this entry's.
- **Prompts tried** (every wording, as required): `outer_box`: "a cardboard box.", "a brown cardboard box." --
  `tray`: "a tray.", "a grey book." -- `red_box`: "a red box.", "a red container." -- `yellow_box`: "a yellow
  box.", "a lime green container." -- `start_button`: "a start button.", "a white index card.".
- **Methodology note, found the hard way**: a first attempt queried all 10 phrases together in one call per
  frame; Grounding DINO's returned text spans blend across phrase boundaries in a long multi-class query (e.g.
  a combined query came back labelled "cardboard box red box" / "a a red box", matching neither submitted
  phrase), so results could not be attributed back to a class by string match -- every one of 600 calls
  silently produced zero usable detections this way. Fixed by querying one phrase at a time (600 forward
  calls total: 5 classes x 2 phrasings x 60 frames), which removes the ambiguity at the cost of throughput.
  Also found and fixed: the installed `transformers` 5.17.0 renamed the model card's documented
  `box_threshold` kwarg to `threshold` (verified via `inspect.signature`).
- **Speed**: the processor's default resize (shortest_edge=800, longest_edge=1333) upscales this rig's
  848x480 frames to 1333x755 and took ~12s/forward-call on this CPU. Reduced to shortest_edge=480,
  longest_edge=800 (a mild resize close to native resolution, not an upscale): ~4.08s/call mean (600 calls,
  min 3.39s, max 7.21s), i.e. ~40.8s of detector time per frame across all 10 phrasings -- the full 60-frame,
  12-run spike took about 41 minutes. This is far too slow for a live/Tier-1 use (irrelevant -- Grounding DINO
  is proposed only for offline auto-labeling, F14 Stage 3, never the runtime detector), but at this per-frame
  cost the real dataset (~2,000-3,000 train frames, essential-features.md section 14 Stage 2) would take on
  the order of a full day if run the same way (one call per class per phrasing); Stage 3 should use one
  phrasing per class, not two, and/or accept a further resolution/threshold trade-off.
- **Found rate was 5/5 classes at 100% (60/60 frames) -- but this number is misleading on its own.** Sampled
  60 frames, seeded (seed 0), 5 frames each spread across the full duration of 12 train runs (all script_types
  present in train: 1 idle (`x023`), 1 repeat (`x027`), 3 of 5 skip, 3 of 5 swap, 4 of 14 correct -- runs:
  `x006,x007,x010,x020,x023,x027,x029,x030,x031,x034,x039,x044`). "Found" only means *some* box in the sanity
  area band (0.1%-60% of frame) was returned for that class -- not that it was the *correct* box. See the next
  point.
- **Wrong-class confusion is severe and goes beyond red-vs-yellow.** In 40/60 frames (67%), `red_box` and
  `yellow_box`'s single best-scoring boxes overlap with IoU > 0.5 (many at IoU > 0.99 -- literally the same
  box). Manually re-rendered three of these frames at full resolution to see what was actually happening
  (`data/spikes/debug_full_x006_88.jpg`, `..._x023_55.jpg`, `..._x029_253.jpg`, not committed, git-ignored):
  in two of the three, **`outer_box`, `red_box` and `yellow_box` all converge on the identical box -- the
  whole cardboard outer box itself** (IoU between every pair > 0.99), while the two actual small containers
  visibly sitting inside it go completely unmatched by their own class's prompts. E.g. on `x006` frame 88,
  "a yellow box." scored 0.757 on the *whole cardboard box* while "a lime green container." correctly found
  the real lime container at only 0.641 -- `select_best_in_band`'s "highest score wins" rule (as specified,
  essential-features.md section 14 Stage 3) picks the wrong one. This is a genuine zero-shot grounding failure
  on this rig, not a prompt-wording problem alone: "a red box."/"a red container." and "a yellow box."/"a lime
  green container." all separately matched the same brown cardboard region on affected frames, regardless of
  color words. In the one frame checked where classes did separate correctly (`x029` frame 253), the correctly
  isolated `yellow_box` measured **0.0493 frame-area-fraction**, versus **~0.23-0.25 for the colliding
  whole-box match** -- a large, clean gap. This suggests the checkpoint 1 sanity area band (0.1%-60%, far too
  wide) is a likely fixable cause: a tighter per-class band (something like 0.5%-15% for the two containers,
  informed by the measured 0.0493 clean sample above, versus a wider band for `outer_box` itself) would reject
  most of these whole-box false matches before the highest-score rule ever sees them. **Not applied here** --
  checkpoint 1's job was to measure and report, not to re-tune; this is a recommendation for checkpoint 3, not
  a result.
- **`tray` and `start_button` localize reliably.** Manually confirmed correct in every full-resolution frame
  checked: `tray` ("a grey book." usually outscored "a tray.") correctly bounds the real grey book/tray
  surface, and `start_button` correctly bounds the real white index card, both distinct from the cardboard box
  collision above. Aggregate area fractions: `tray` 0.14-0.27 (median 0.150), `start_button` 0.019-0.194
  (median 0.026, consistent with it being the smallest object). Static-object box stability (IoU of each
  frame's box against that run's own 5-frame median, essential-features.md section 14 Stage 3's "static-object
  smoothing"): `outer_box` very stable (mean IoU 0.991 across all 12 runs, worst-run mean 0.977); `tray` and
  `start_button` both show at least one IoU=0.000 outlier frame in roughly half the runs (full per-run table
  in `data/spikes/report.json`'s `static_box_stability_iou`) -- an occasional bad frame, not a systemic miss.
- **Measured hue** (OpenCV 0-179 scale, mean hue inside each frame's selected `red_box`/`yellow_box` box):
  `red_box` 10.8-36.6 (median 29.2), `yellow_box` 21.7-52.6 (median 32.2) -- heavily overlapping ranges,
  consistent with the 2026-09-30 "crew footage intake" DECISION's observation that the yellow container's
  real hue leans lime/olive, close to red-orange. **Caveat: these hue numbers are contaminated by the
  collision above** -- on the ~40/60 frames where `yellow_box`'s selected box is actually the brown cardboard
  box (or the red container), its "hue" sample is not a clean read of the real lime container's color. A
  hue-based read of the *actual* lime container needs re-measurement from only the frames where the collision
  above did not happen (not done in this session -- flagged for whoever refines the area band).
- **Gloves**: none of the 12 sampled runs were the robustness/gloves runs (out of this spike's scope; the
  hand tracker's own gloves caveat is unchanged from the 2026-09-30 hand_landmarker smoke test entry above).
- **Artifacts** (all git-ignored, `data/`): `data/spikes/report.json` (every number above plus the full
  per-class area/hue sample lists and per-run stability table), `data/spikes/contact_sheet_01.jpg`
  through `_05.jpg` (12 frames per sheet, all 5 classes' boxes drawn and labelled with score), and three
  ad-hoc full-resolution debug renders named above (not produced by the committed spike script; made by hand
  in this session to diagnose the collision finding).
- **Code**: `training/spikes/` (`postprocess.py` -- tested pure helpers; `detector.py` -- thin Grounding DINO
  wrapper; `prompts.py` -- the candidate phrasings; `run_checkpoint1.py` -- the orchestration script, `python
  -m training.spikes.run_checkpoint1`), `tests/unit/training/test_spikes_postprocess.py` (16 tests, pure
  helpers only, no model). `scripts/check.py` full suite green (411 passed) after adding this code;
  `--status` unchanged (F1-F14 all GREEN, spike code is intentionally unmarked -- not a Tier-1 feature).
- Status: RESULT recorded; checkpoint 1 complete. Awaiting the operator's per-class bad-box counts from the
  contact sheets before checkpoint 2 (P1.2's own packet, IMPLEMENTATION_PLAN.md Part 10) proceeds.
## 2026-10-02 P2 fix - local environment: `tests/` shadowed by a site-packages `tests` package (F7)

- **Classification: LOCAL ENVIRONMENT interaction, not a product bug.** Nothing under `outputs/`,
  `perception/`, `state/` or `engine/` changed. `tests/` has no owner in `IMPLEMENTATION_PLAN.md` Part 4, so
  this was done in a P2 session.
- **Symptom:** `tests/unit/outputs/test_tts.py::test_degrades_quietly_when_engine_init_fails` failed (alone or
  in the full suite) with `ModuleNotFoundError: No module named 'tests.unit'` raised inside
  `multiprocessing/spawn.py` while the spawned `TTSWorker` child process tried to unpickle the test's
  module-level `_raising_engine_factory` callable (picklable-under-spawn, so it is referenced by its import
  path, `tests.unit.outputs.test_tts._raising_engine_factory`).
- **Cause:** this repo ships no `tests/__init__.py`. `ultralytics` 8.4.164 (the YOLO fallback detector,
  AGENTS.md rule 15) installs its own top-level `tests/__init__.py` directly into `.venv/Lib/site-packages/`
  (confirmed via `site-packages/ultralytics-8.4.164.dist-info/RECORD`, which lists `tests/__init__.py` at the
  package root, not under `ultralytics/`). A plain `import tests` in a *fresh* interpreter -- which is exactly
  what the spawned worker process does to unpickle the callable -- walks `sys.path` and resolves to that
  installed regular package instead of this repo's own `tests/` directory, which has no `__init__.py` of its
  own to win that resolution. Inside the main pytest process this is invisible: pytest's own
  `--import-mode=importlib` (`pyproject.toml`'s `[tool.pytest.ini_options]`) pre-populates
  `sys.modules['tests']` pointing at the repo before any test body runs, masking the hazard for anything
  checked in-process -- it only surfaces where a *new* interpreter re-imports by name, as the TTS worker does.
- **Fix:** added an empty `tests/__init__.py`. Nothing else changed.
- **Regression test:** `tests/unit/harness/test_tests_package_resolution.py` -- spawns a bare
  `sys.executable -c "import tests; ..."` subprocess (not pytest's own import machinery, which would hide the
  bug per the note above) and asserts `tests.__path__[0]` resolves inside this repo, not
  `site-packages`. Confirmed RED before the fix (`site-packages\tests` resolved), GREEN after.
- **Verified no regression:** toggled `tests/__init__.py` on/off against the current tree and compared
  `pytest tests/unit -m "not slow and not gold_video" --collect-only"` -- identical 397 collected / 1
  deselected either way; the fix changes only the pass/fail outcome of the two tests above, not which tests
  are collected. `python scripts/check.py` (full suite): 396 passed, 1 deselected, 0 failed. `--status`:
  F1 11, F2 no tests (pre-existing, unaffected), F3 22, F4 22, F5 31, F6 10, **F7 22/22 (was 21/22)**, F8 3,
  F9 16, F10 32, F11 7, F12 23, F13 25, F14 78 -- every count matches or exceeds its pre-fix value, all GREEN.
- Status: DONE. F7 is now the full 22/22.

## 2026-10-02 P1.2 checkpoint 1b - candidate cache, v2 selection, proxy metrics

- **Supersedes checkpoint 1's closing line.** "Awaiting the operator's per-class bad-box counts from the
  contact sheets" (2026-09-30 entry) is superseded: the v1 contact sheets (`data/spikes/contact_sheet_01..05.jpg`)
  must **not** be counted. This session re-ran the whole spike with a wider candidate net and a re-derived
  selection rule; `data/spikes/v2/bad_boxes.csv` (header only: `run_id,frame_id,class,reason`) is the per-class
  review artifact going forward, against `data/spikes/v2/contact_sheet_v2_01..05.jpg`.
- **Raw candidate cache** (`training/spikes/run_raw_cache.py`, no tests -- model + I/O, like
  `run_checkpoint1.py`): identical 12 train runs x 5 frames x 10 phrasings as checkpoint 1 (seed 0, frame list
  reused via `run_checkpoint1._select_runs` / `_sample_frame_indices` rather than re-derived -- confirmed these
  reproduce the exact `run_ids` checkpoint 1's `report.json` recorded). Checkpoint 1's own `report.json` only
  kept the post-hoc v1 winner, not the full candidate set, so it could not be reused for re-selection -- hence a
  fresh model pass. `box_threshold=0.20`, `text_threshold=0.20` (both lower than checkpoint 1's 0.25/0.20, to
  surface the smaller correct candidates checkpoint 1's own threshold had already screened out), `score_floor
  =0.20`, same resize as checkpoint 1 (`shortest_edge=480, longest_edge=800`), `HF_HUB_OFFLINE=1` set for the
  whole run. Up to 10 candidates kept per (run, frame, phrasing) call, written to
  `data/spikes/v2/raw_candidates.jsonl` (600 lines, one per call), resumable by skipping already-written
  (run,frame,phrase) keys (did not actually crash this session, but verified the skip logic against the
  partial file while the run was still in progress). **600 calls, mean 3.72s/call, median 3.70s/call, total
  37.2 min** -- in line with checkpoint 1's own ~4.08s/call.
- **Area bands, derived from report.json AND the raw cache, not assumed** (`training/spikes/run_checkpoint1b.py
  pick_area_band`): for each class, took every candidate box across all 600 calls belonging to that class,
  sorted their area fractions, and split them into clusters wherever a consecutive gap >= 3% of frame area
  opened up. **A first version of this (biggest-gap-only, no anchor) picked the WRONG cluster for `outer_box`**
  -- it assumed "the small cluster is always the real object" (true for the two containers: small = real
  container, large = the whole-cardboard-box collision) but that is backwards for `outer_box` itself, where the
  real object IS the large cluster. That version returned a band of (7.2%, 17.4%) for `outer_box`, entirely
  below its true size, and `outer_box` came back "missing" in 35/60 frames. **Fixed by anchoring each class's
  cluster choice on the minimum value already in checkpoint 1's own `report.json` `area_frac_samples` for that
  class** -- verified first that this minimum always falls inside the TRUE cluster for all 5 classes (outer_box
  0.2274, tray 0.1432, red_box 0.0437, yellow_box 0.0460, start_button 0.0193 -- all inside their class's
  correct range, confirmed against checkpoint 1's own qualitative findings before trusting this). Final bands
  (lo/hi, with the raw-cache cluster each came from):
  - `outer_box`: **[0.182, 0.328]** (cluster 0.227-0.273, 120/155 raw candidates -- matches checkpoint 1's own
    measured range almost exactly, as expected since checkpoint 1 found `outer_box` reliable).
  - `tray`: **[0.019, 0.188]** (cluster 0.024-0.157, 218/327).
  - `start_button`: **[0.001, 0.108]** (cluster 0.001-0.090, 217/328) -- cleanly excludes the 4 frames where
    checkpoint 1's own data shows `start_button` landing on the tray/book instead (area frac 0.143-0.194).
  - `red_box`: **[0.031, 0.188]** (cluster 0.039-0.157, 162/275).
  - `yellow_box`: **[0.023, 0.228]** (cluster 0.029-0.190, 142/216).
  These are wider than the packet's own starting-point suggestion (e.g. ~0.5%-15% for the containers) because
  the lower 0.20 score floor surfaces more borderline candidates bridging what would otherwise be a tighter
  cluster -- see the hand/skin finding below for what some of that bridging turns out to be.
- **Selection** (`training/spikes/select_v2.py`, 18 new unit tests, pure, no model): static classes
  (`outer_box`, `tray`, `start_button`) get a per-run consensus -- cluster that class's in-band candidates
  across the run's 5 frames by mutual IoU >= 0.5, take the cluster the most DISTINCT frames support, assign its
  median box to every frame with a candidate agreeing (IoU >= 0.5), else "missing" (never invents a box).
  Containers (`red_box`, `yellow_box`) reject any candidate overlapping (IoU > 0.5) the run's `outer_box`,
  `tray` or `start_button` consensus, then keep the highest remaining score; every rejection is tagged with a
  reason (`out_of_band`, `overlaps_<class>`, `not_highest_score`).
- **Proxy metrics, v1 (checkpoint 1's own rule, re-run on this richer cache) vs v2, both n=60 frames:**

  | metric | v1 | v2 |
  |---|---|---|
  | red_box/yellow_box IoU > 0.5 | 40 (67%) | 0 (0%) |
  | a container IoU > 0.5 with outer_box | 57 (95%) | 0 (0%, by construction) |
  | start_button IoU > 0.5 with a container or tray | 3 (5%) | 0 (0%) |
  | missing (v2 only): outer_box / tray / start_button / red_box / yellow_box | n/a | 0 / 0 / 4 / 8 / 0 |
  | mean static-box IoU vs run median | 0.873 | 1.0 (see caveat) |

  The v1 red/yellow overlap (40/60) **exactly reproduces checkpoint 1's own finding**, a good consistency check
  that re-deriving v1 on the lower-threshold cache didn't change the original result. The v2 row's "mean static
  IoU = 1.0" is trivial, not a real improvement number: v2 assigns the smoothed consensus box itself to every
  matched frame (essential-features.md section 14 Stage 3), so it is 1.0 by construction wherever a frame
  isn't "missing" -- the `missing` counts are v2's real stability signal, not this row. Likewise v2's
  container/outer-overlap is 0 by construction (select_container explicitly rejects those candidates), not an
  earned result -- the real result is row 1 (red/yellow confusion down from 67% to 0%).
  **Winning phrasing:** among chosen boxes, `red_box` picked "a red container." in 52/52 frames (100%; "a red
  box." never won); `yellow_box` picked "a lime green container." in 60/60 frames (100%; "a yellow box." never
  won) -- consistent with the 2026-09-30 DECISION that the real container's hue leans lime/olive. Static
  classes' final box is a cross-run median, not attributable to a single phrase/call, so this tally does not
  apply to them.
- **(b) start_button finding, measured (not the thumbnail reading).** The Lead's reading of the v1 contact
  sheets ("start_button lands on the lime container or the grey tray in roughly 8 of 60 frames") does not hold
  up against the cached candidates: the **measured v1-rule count is 3/60 (5%)**, not ~8/60. This measured
  number supersedes the thumbnail reading; the v1 sheets it came from are not to be counted either way (see
  above).
- **(c) Hue/saturation, central 50% of the box, OpenCV 0-179/0-255 scale, measured ONLY on frames with exactly
  one in-band, non-outer candidate for that colour** (`select_v2.single_clean_candidate`):
  - `red_box`: **n=11**, mean hue 4.51 (median 3.02, range 2.23-9.28), mean sat 123.06 (range 87.02-137.58).
  - `yellow_box`: **n=10**, mean hue 27.31 (median 33.65, range 10.35-35.12), mean sat 122.29 (range
    77.83-149.96).
  Hue separates the two colours cleanly at the median (3.0 vs 33.7) but the closest pair across classes (red's
  max 9.28, yellow's min 10.35) are under 1.1 hue units apart -- a real but thin margin, not a universal clean
  threshold on this small sample. Saturation does not add separation: the two ranges (77-150 vs 87-138)
  overlap almost completely.
  **Hand/skin confusion, measured, not hidden:** flagging frames with sat < 100 (an unvalidated heuristic, not
  a tuned threshold) in this same n=11/n=10 sample gives `red_box` 1/11 (9%) and `yellow_box` 3/10 (30%).
  Visually confirmed in the v2 contact sheets that this is **not limited to red** as the packet's own framing
  assumed: `yellow_box`'s final box lands on a hand/fingers rather than the lime container itself in
  `x034#156`, `x034#208` and `x031#472` (the last of these also affects `red_box` in the same frame -- a
  heavily occluded frame). Mechanism: a hand holding or covering a container produces an intermediate-sized
  false candidate that falls inside the SAME area band as the real container, so the area band cannot reject
  it; the static-consensus overlap exclusion (`outer_box`/`tray`/`start_button`) cannot catch it either, since
  a hand is none of those three objects. This is a real, open gap in the v2 rule, not something either rule
  solves.
- **(d) Not tested this session:** gloves, other performers, other lighting setups, and (per the packet's own
  rule) `val`/`test` runs -- all 12 sampled runs confirmed `train` split via `runs/manifest.csv` before this
  session touched them.
- **(e) None of this is accuracy.** There is no ground truth here -- every number above is a PROXY metric on
  zero-shot Grounding DINO boxes against each other and against checkpoint 1's own prior measurements. The
  real measure is the Lead's review of `data/spikes/v2/bad_boxes.csv` against the v2 contact sheets.
- **Code:** `training/spikes/run_raw_cache.py` (cache, no tests, model + I/O), `training/spikes/select_v2.py`
  (selection, 18 tests, pure), `training/spikes/run_checkpoint1b.py` (orchestration -- bands, proxy metrics,
  contact sheets, `bad_boxes.csv`; no tests, same stated scope as `run_checkpoint1.py`).
  `tests/unit/training/test_spikes_select_v2.py`. `python scripts/check.py` (full suite): **430 passed** (was
  412), 0 failed, ruff clean; `--status` unchanged (F1-F14 all GREEN; spike code remains intentionally
  unmarked, not a Tier-1 feature).
- **Artifacts** (all git-ignored, `data/spikes/v2/`): `raw_candidates.jsonl` (600 lines), `report_v2.json`
  (every number above plus the full per-frame chosen-box-or-missing-and-why table), `contact_sheet_v2_01.jpg`
  through `_05.jpg` (12 frames per sheet, v2's final boxes only, one fixed colour per class, a legend on every
  sheet, frame tag under each tile), `bad_boxes.csv` (header only), `run_raw_cache.log` (the cache run's log,
  including the per-call timings above).
- Status: RESULT recorded; checkpoint 1b complete, as scoped (box selection fix + v2 contact sheets). Awaiting
  the Lead's `bad_boxes.csv` review before checkpoint 2 or 3 (P1.2's own packet) proceeds. Per the Lead's
  checkpoint-1b instruction, this session stops here.

## 2026-10-02 P1.2 checkpoint 1c - Lead's v2 review, selection v3, hold-out

- **Lead's v2 review (validated by `training/spikes/review_csv.py`, 24 rows, 60 frames).** Bad = wrong_box / wrong_class / duplicate / missing-but-visible-or-partial; missing+hidden is reported separately and not counted. start_button 10/60 (16.7%) bad, plus 3 hidden-missing (x006#334, x010#474, x030#472); red_box 10/60 (16.7%); yellow_box 1/60 (1.7%); outer_box 0; tray 0.
- **Cause.** In x034 and x044 the start_button "consensus" was the RED CONTAINER (the container barely moved in most sampled frames, so it out-voted the card). The v2 rule "reject containers overlapping the start_button consensus" then removed the real red box (red_box missing in 8 frames). Two-container boxes (x034#77, #208) also passed the red band. The v2 proxy zeros (container/outer overlap 0, start_button overlap 0) were true by construction, not evidence.
- **Thresholds (OpenCV scale, central 50% of box).** White-paper test, card n=47 frames (v2 box in a frame the Lead did not flag, runs other than x034/x044): sat median 22.9, p95 43.3, max 75.2; val min 142.6, median 204.3. Containers (single clean candidates): red n=11 sat min 87.0 / median 127.4; lime n=10 sat min 77.8 / median 136.1; container val 152-250 overlaps the card, so val does not separate them. Frozen: start_button candidate needs sat <= 61 (midpoint of card p95 and lowest container sat) and val >= 128 (0.9 x card min val, a guard only). Container area band tightened from the 21 pooled clean areas (min 0.0381, p90 0.0555, max 0.0867): [0.0324, 0.0694] of frame (0.85 x min to 1.25 x p90); v2 was [0.031, 0.19]. Both two-container boxes (0.121, 0.131) are now out of band. Static bands unchanged from v2. Frozen in `data/spikes/v3/frozen_params.json`.
- **Was (b), a position prior, needed? No.** The white-paper test alone fixed x034 and x044; the card consensus is now the lower-frame card in all 12 runs. Not implemented.
- **Rule change.** The start_button consensus excludes containers only if its own measured sat/val passes the white-paper test (not circular any more); no candidate passing => "missing", never an invented box.
- **IN-SAMPLE result (the same 60 frames the thresholds came from; not evidence of generalization).** 23 (frame,class) cells changed vs v2 across 11 frames (`data/spikes/v3/changed_vs_v2.csv`). Of the Lead's 21 bad rows: 17 now have a box that looks right on the v3 sheets (my visual reading, not the Lead's), 4 had a wrong box removed and are now "missing" (x034#77 and #156 and x044#165 start_button, x034#156 yellow_box; the Lead should rule whether those objects were hidden). The 3 hidden-missing rows stay missing. Costs of the tight band: 2 new misses of visible-or-moving containers whose rotated box area is 0.0735 and 0.0725, just above the 0.0694 ceiling (x027#266 red_box, x044#165 yellow_box). Missing totals on 60 frames: start_button 6, red_box 1, yellow_box 2 (v2: 4, 8, 0). Phrasing tally: with only the chosen phrasing per class the 60-frame result is identical.
- **Hold-out (fresh).** Train runs x003, x004, x008, x021, x033 (seed 1), 4 frames each = 20 frames, one phrasing per class ("a brown cardboard box.", "a grey book.", "a white index card.", "a red container.", "a lime green container."; each chosen as the phrasing of the best-scoring candidate agreeing with the final box in the most frames of the 60-frame cache), 100 calls, mean 4.13 s/call. Frozen v3 applied unchanged: missing start_button 0, red_box 1, yellow_box 2, outer 0, tray 0; card consensus passed the white test in 5/5 runs. These are counts of missing boxes, not accuracy: the hold-out review sheet `data/spikes/holdout/bad_boxes.csv` is empty and awaits the Lead.
- **Occlusion.** The rule used here is "hidden-and-missing is not bad". The plan has no occlusion labeling policy yet; to be decided at checkpoint 3.
- **NOT tested:** gloves, other performers, other lighting, val/test runs (train runs only), anything on a different camera setup than the 17 train runs seen.
- Code: `training/spikes/review_csv.py`, `select_v3.py` (tested), `run_checkpoint1c.py`, `run_holdout.py` (orchestration, untested like 1b). Artifacts (git-ignored): `data/spikes/v3/`, `data/spikes/holdout/`. Status: checkpoint 1c complete; stopping before checkpoint 2.

## 2026-10-02 P1.2 DECISION - zero-shot auto-labeler (Grounding DINO tiny) and prop setup

Scope: S-C checkpoints 2 and 3, train runs only, frozen v3 rules unchanged (`data/spikes/v3/frozen_params.json`), `HF_HUB_OFFLINE=1`. Everything below is a feasibility check on 5 clips plus the Lead's reviews of 80 frames. It is not an accuracy claim. **Gloves were not tested.** No file in `contracts.py`, `config/experiment.json`, `state/`, `engine/`, `perception/` was edited. **No `config/experiment.json` change is proposed.**

### 1. Auto-labeler verdict
- **Grounding DINO tiny (Apache-2.0, `IDEA-Research/grounding-dino-tiny`) is usable as the Stage 4 first-pass labeler for this rig, with a human review of every frame class it is weak on (below).** It is not usable unreviewed: the three static classes had 0 bad frames on the reviewed frames (in-sample 60, hold-out 20), the two movable containers did not (1/60 and 3/60 in-sample, 1/20 and 2/20 hold-out, see section 3).
- Speed on this CPU: 3.72 s/call (600 calls, checkpoint 1b), 4.13 s/call (100 calls, hold-out), 4.05 s/call mean and 3.96 s median (773 calls, checkpoint 2, `shortest_edge=480`). About 3.7 to 4.1 s per call, one phrase per call.
- **YOLO-World was NOT tried** (not in the lockfile, no licence check done, never run). This entry therefore does not compare the two; it only says Grounding DINO tiny is good enough to continue with, and leaves YOLO-World as an untested alternative.
- Accuracy numbers are in section 3 (from the Lead's three CSVs). They are counts on 60 in-sample frames and 20 hold-out frames.

### 2. Checkpoint 2: the stack on 5 train clips (frozen v3 + hands -> StateTracker -> SequenceEngine)
Runs (train; none among x006 x007 x010 x020 x023 x027 x029 x030 x031 x034 x039 x044, none among the hold-out x003 x004 x008 x021 x033; the shortest unused clips of each type, to keep the ~4 s/call model pass short): **correct x011, x019, x001; skip x037; swap x043** (the only unused train skip; x043 chosen over x026 as the shorter swap). Frames at ~4 fps (frame ids on a 4 fps grid of the 30 fps clip, `t = frame_id / fps`): x011 75, x019 80, x001 86, x037 48, x043 60 = 349 frames. Static classes (outer_box, tray, start_button) by the v3 consensus on 5 frames spread across each clip, the consensus box and the median agreeing score applied to every frame; containers per frame, one phrasing each ("a red container.", "a lime green container."), 698 container calls + 75 static calls = 773 calls, 52.1 min. Config: `PerceptionConfig()` and `RuntimeConfig()` defaults (hysteresis 5, release 5, baseline 10 frames, floor 0.30, confirm 0.60, touch margin 0.10). Hands: `perception/hands.py` `HandTracker`, fresh landmarker per clip. **Hands-only: 38.8 fps overall (36.7 to 42.4 per clip), about 26 ms per frame, measured with the detector idle** (an earlier partial run measured 21 fps while the model was running on the same CPU).

| clip | expected deviations | produced | result |
|---|---|---|---|
| x019 correct | none | none; 7 steps in order, last at 18.51 s | matches |
| x011 correct | none | omission [yellow_out, yellow_in_tray]@6.24, out_of_order yellow_out@6.74, out_of_order yellow_in_tray@7.01, repeat start_pressed@15.01, omission [red_stowed]@18.01 | 5 spurious; red_stowed never fires |
| x001 correct | none | repeat start_pressed@16.50 (run completes at 20.00) | 1 spurious |
| x037 skip | omission [red_out, red_in_tray, yellow_out, yellow_in_tray] | omission [red_out, red_in_tray, yellow_out, yellow_in_tray, start_pressed, red_stowed]@6.74, run completed | type right, ids wrong |
| x043 swap | omission [red_out, red_in_tray]; out_of_order red_out; out_of_order red_in_tray; omission [start_pressed, red_stowed] | omission [red_out, red_in_tray, yellow_out, yellow_in_tray]@6.27; out_of_order red_out@6.50; out_of_order red_in_tray@6.50; omission [red_stowed]@9.50, run completed | 4 of 4 deviation types right, 2 of 4 id sets exact |

Fired steps, in order, with time in s (`?` = tagged `flagged_uncertain`, confidence < 0.60):
- x011: red_out 4.00?, red_in_tray 4.00?, start_pressed 6.24, yellow_out 6.74?, yellow_in_tray 7.01?, start_pressed 15.01, yellow_stowed 18.01?. Performed: the canonical 7. red_stowed never fires.
- x019: red_out 4.24?, red_in_tray 4.24?, yellow_out 8.24?, yellow_in_tray 8.24?, start_pressed 10.74, red_stowed 17.51?, yellow_stowed 18.51?. Performed: the canonical 7. Exact match.
- x001: red_out 3.50?, red_in_tray 3.50?, yellow_out 7.23?, yellow_in_tray 7.23?, start_pressed 7.23, red_stowed 16.00?, start_pressed 16.50, yellow_stowed 20.00?. Performed: the canonical 7.
- x037: yellow_stowed 6.74?, red_stowed 7.00? (the second is after run_completed and ignored). Performed: start_pressed only. start_pressed never fires.
- x043: start_pressed 6.27, red_out 6.50?, red_in_tray 6.50?, yellow_stowed 9.50? (run_completed), then red_out 9.73?, red_in_tray 9.73?, start_pressed 11.50, red_stowed 12.73? (all ignored by the engine after completion). Performed: yellow_out, yellow_in_tray, red_out, red_in_tray, yellow_stowed, red_stowed. yellow_out and yellow_in_tray never fire.

**Causes, from the per-frame data (`data/spikes/cp2/<run>/report.json`, `tables.md`) and the frames I looked at (hand landmarks and boxes drawn on x011, x001, x037, x043, x019). None was fixed; no threshold, rule or P2 code was touched.**

A. **START touch is not "fingertip on the card".** The rule (contracts.py section 3, implemented in `state/tracker.py`) is true when ANY of the 21 landmarks is in the card box grown 10 percent. Over the 5 clips the rule was true in 93 frames; the index fingertip (landmark 8) was in the grown card box in 9 of them, all in x019, the one clip whose real press is made with the hand mostly inside the frame. Of the 7 `start_pressed` events emitted, 1 (x019@10.74) is a real press; 6 are an arm or palm crossing the card (x011@6.24 and @15.01 while carrying a container, x001@7.23 and @16.50, x043@6.27 and @11.50). The real press in x001 (finger on the card at the frame's bottom edge, about 9.7 to 11.5 s) produced **no hand at all** in 14 consecutive frames (`n_hands` = 0), so it never fires. The real press in x011 (about 10 to 11.5 s) had a hand in 4 of the 9 frames between 9.5 and 11.5 s, never 5 in a row, and landmark 8 was off the visible fingertip in the frame I looked at (t = 10.01). In x037 the arm crossed the card from 2.0 to 5.2 s; the 10-frame baseline ends at 2.25 s, so START was latched at 2.0 s and stayed true until 5.24 s; it re-armed at 6.5 s and was then true for only 1 frame (7.5 s), so no event.
B. **Frame counts are tuned for a higher rate than 4 fps.** The defaults count frames: baseline 10 frames is 2.5 s at 4 fps (about 0.67 s at 15 fps), hysteresis 5 is 1.25 s, release 5 is 1.25 s. x043: the performer moved the yellow container before 2.25 s, so yellow_out and yellow_in_tray were already true in the baseline and were latched (never fire). Real transitions of 1 to 4 frames cannot reach hysteresis (e.g. x011 red_stowed true for 2 frames at 12.74). `RuntimeConfig.target_fps` is 15; the 4 fps here is the detector's own rate. Not tuned.
C. **Missing and low-score containers.** Missing container frames over the 5 clips (all reasons `out_of_band`: held and rotated above the 0.0694 ceiling, hand-merged or motion-blurred): red_box 40 and yellow_box 41 of 349 frames each (81 of 698 container-frames; not Lead-reviewed, includes hidden objects). Frames with a detection below the 0.30 floor (ignored by the tracker): red_box 41, yellow_box 3 (red_box scores 0.20 to 0.29 when it sits in the outer box in x011, which is why x011 red_stowed never fires). What the tracker does with a missing label: every rule on that label is false (inside and outside both need the label), so the hysteresis counter resets, and a fired or latched step gets a false frame; 5 missing frames re-arm it. That is how x037 fires yellow_stowed and red_stowed spuriously: the arm covers the red container for 11 frames (2.74 to 5.24 s) and the lime one for 15 (1.73 to 5.24 s), both stow steps re-arm, then both go true again.
D. **Stow steps (start true, latched), as the 2026-09-28 entry predicts:** red_stowed true in 3, 4, 8, 8, 9 of the first 10 frames and yellow_stowed true in 10, 10, 10, 2, 6 (x011, x019, x001, x037, x043) so all 10 were latched in the baseline; each re-arms after 5 false frames (red_stowed at 2.0, 2.23, 3.0, 3.24, 6.5 s; yellow_stowed at 6.51, 7.0, 7.23, 2.74, 2.73 s) and then fires when the container is put back (x019 17.51 and 18.51, x001 16.00 and 20.00, x043 12.73 and 9.50). They fire in the right order in x019 and x001. They fail when the container score is under the floor (x011 red) or the container is missing (x037 spurious).
- A contract-level change is **not** shown to be needed by these 5 clips. The fixes the data point at are on the P2 side (the touch rule and the frame-count config, tuned on val at P2.6 at the real rate) and on the detector side. They are listed for the Lead and P2 as observations, not as edits: touch rule true only for the fingertip or for several consecutive frames with a hand present; frame counts expressed in seconds; a hand cut off at the frame edge.
- Prop observation (untested proposal): the card sits at the frame's bottom edge, so a real press leaves only a fingertip in frame (x001). Moving the card up by about one hand length is a setup change that would need a re-recording to test.

### 3. Per-class bad counts (Lead's CSVs; bad = wrong_box / wrong_class / duplicate / missing-but-visible-or-partial; hidden-and-missing not counted)
v3 on the 60 frames the thresholds came from (**in-sample**): verdict from `changed_review.csv` for every cell it lists, else from the v2 `bad_boxes.csv`.

| class | bad / 60 | fraction | 95% Wilson interval |
|---|---|---|---|
| outer_box | 0 | 0.0% | 0 to 6.0% |
| tray | 0 | 0.0% | 0 to 6.0% |
| start_button | 0 | 0.0% | 0 to 6.0% |
| red_box | 1 (x027#266, rotated, 0.0735 above the ceiling) | 1.7% | 0.3 to 8.9% |
| yellow_box | 3 (x034#77 box stretched over the hand; x034#156 and x044#165 blurred and moving, 0.0725 above the ceiling) | 5.0% | 1.7 to 13.7% |

Hidden-and-missing, not counted: start_button x006#334, x010#474, x030#472, x034#77, x034#156, x044#165 (6). v2 for comparison: start_button 10, red_box 10, yellow_box 1, outer_box 0, tray 0 of 60 (16.7, 16.7, 1.7%).

Hold-out (fresh train runs x003 x004 x008 x021 x033, 20 frames): outer_box 0/20, tray 0/20, start_button 0/20, **red_box 1/20 (5.0%, x008#572 held at the left edge under the hand, only an outer-box-sized candidate)**, **yellow_box 2/20 (10.0%, x003#324 candidate 0.135 merged the hand and was rejected; x003#416 candidate 0.0277 below the 0.0324 floor)**. Intervals are wide at n = 20 (red 0.9 to 23.6%, yellow 2.8 to 30.1%; 0 of 20 is 0 to 16.1%). **One data note for the Lead:** the hold-out CSV row says `x004,416,yellow_box`; x004's sampled frames are 93, 222, 345, 506, and the 0.0277-area candidate the note describes is in x003#416 (its cache row). I counted it as x003#416 and did not edit the CSV; `load_bad_boxes` rejects the row as written.

### 4. Props
Keep: grey hardcover book as tray, white handwritten START card, lime container, red container. No prop change is supported by the data: the card passes the white-paper test (below) and was found in all 22 runs checked (12 + 5 + 5, v3 white test passed in each), tray and outer_box had 0 bad frames, and the remaining errors come from hands, motion and rotation, not from colour. The only prop-related proposal is the card position noted in section 2 (untested).

### 5. Recommended `training/prompts.yaml` wording (one phrasing per class; tally from the 60-frame v3 result, phrase of the best-scoring candidate agreeing with the final box)
- outer_box: "a brown cardboard box." (60/60; "a cardboard box." never won)
- tray: "a grey book." (51/60; "a tray." won 9/60)
- start_button: "a white index card." (54/54 frames with a box; "a start button." never won)
- red_box: "a red container." (59/59 frames with a box; "a red box." never won)
- yellow_box: "a lime green container." (58/58 frames with a box; "a yellow box." never won)
Prompts are lowercase and end with a period. One phrase per call (a multi-class query blends text spans, see the checkpoint 1 entry). Not tried: any other wording.

### 6. Recommended bands and the START white-paper test (from the Lead-reviewed 60 frames; thresholds already frozen, not re-derived)
- Container area band **[0.0324, 0.0694]** of frame area, from n = 21 pooled clean container areas (red 11, lime 10): min 0.0381, median 0.0458, p90 0.0555, max 0.0867; lower = 0.85 x min, upper = 1.25 x p90. Costs seen: 3 true containers above the ceiling (0.0725, 0.0735, and 0.0743 in x011#75, a held rotated red container) and 1 partly hidden container below the floor (0.0277). Static bands unchanged: outer_box [0.182, 0.328], tray [0.019, 0.188], start_button [0.001, 0.108].
- START card white-paper test, centre 50 percent of the box, OpenCV scale: **saturation <= 61 and value >= 128**. Card n = 47 frames: saturation median 22.9, p95 43.3, max 75.2; value min 142.6. Lowest container saturation n = 21: 77.8. The consensus passed the test in 22 of 22 runs (12 in checkpoint 1c, 5 hold-out, 5 here). Small samples, one performer, one setup.

### 7. Proposals for the Stage 4 labeler (PROPOSALS; each needs a NEW fresh sample before it is trusted, none is tested)
1. Container ceiling near **twice the clean median container area** (about 0.092 from the 0.0458 median) instead of 1.25 x p90, so rotated and held containers (0.0725 to 0.0743) survive; the wrong or merged boxes seen are 0.102 to 0.135. It would not reject x034#77's stretched lime box (0.0867), which is what proposal 3 is for.
2. A **colour-verified smaller floor**: accept a candidate below 0.0324 (e.g. 0.0277) only if its central-50-percent hue and saturation match the class colour (red hue 2.2 to 9.3, lime hue 10.4 to 35.1 from the checkpoint 1b sample, n = 11 and 10).
3. A **rule against boxes stretched over a hand**: reject or flag a candidate that is much wider than the container and overlaps the hand landmarks (x034#77 was 2.7 times the container width).

### 8. Occlusion labeling policy (PROPOSAL for the plan)
Hidden object: no box. Partly hidden: a box on the visible part only (the reviewed partial cases above are all of this kind). Review counts "missing but visible or partial" as bad and "missing and hidden" as not bad, which is the same rule.

### 9. NOT tested
Gloves (no glove clips); other performers; other lighting; other camera positions or any camera move; val and test runs (never opened); swap, idle and repeat clips in the hold-out (it had 4 correct runs and 1 skip); a second performer or setup (one performer, one setup throughout); YOLO-World.

Code: `training/spikes/cp2.py` (pure helpers, tested), `bad_counts.py` (tested), `run_cp2.py` (orchestration: plan, cache, assemble, tables). Tests: `tests/unit/training/test_spikes_cp2.py`, `test_spikes_bad_counts.py`. Artifacts (git-ignored): `data/spikes/cp2/` (`raw.jsonl` and `perception.jsonl` per run, `report.json`, `tables.md`, `summary.json`, `cache.log`). Status: checkpoints 2 and 3 complete; the Lead's call on section 2 (touch rule and frame-count config are P2's, a re-recording question for the card position) is open.

## 2026-10-02 P1.2 addendum - START press rule variants on train clips (measurement only)

Scope: S-C addendum, **train runs only** (every run whose `runs/manifest.csv` split is "train": 26 runs; no val or test run was opened, sampled or run). `HF_HUB_OFFLINE=1`; no install or download. **Nothing in `state/`, `engine/`, `perception/`, `contracts.py`, `config/experiment.json` or the F4 text was changed**; no threshold was tuned. Acceptance was pre-registered before any number was seen: a variant is ACCEPTABLE if it gives exactly the expected number of start presses in at least 80% of the train clips with script_type "correct" (12 of 14) AND extra events in at most 10% of all train clips (2 of 26).

**Result: no variant meets the acceptance (0 of 40 variant x hold x fps cells).** The best cells reach 9 of 14 correct clips exact (needs 12).

### 1. Clips
26 train runs: correct 14, skip 5, swap 5, idle 1, repeat 1 (14757 frames, 29.9 to 30.0 fps). Expected presses = count of `start_pressed` in `performed_steps`: 23 runs expect 1, x027 (repeat) expects 2, x023 (idle) and x043 (swap, start omitted) expect 0. **Excluded: none** (card found in 26 of 26, none below `detector_conf_floor`).

### 2. Card box (v3 static consensus, unchanged)
Five frames spread over each clip at 4 fps, phrasing "a white index card.", frozen white-paper test and area band; `data/spikes/cp3/<run>/raw.jsonl` (105 new model calls, 6.9 min; the five cp2 clips reused cp2's identical start_button calls). The card passed the white-paper test in 26 of 26 runs; the five cp2 boxes are identical to cp2's.

### 3. Hands (MediaPipe via `perception/hands.py`, every frame at the target rate, fresh landmarker per run and rate)
| target fps | frames (26 clips) | with a hand | share | hands-only fps |
|---|---|---|---|---|
| 4 | 1977 | 1386 | 70.1% | 44.5 |
| 8 | 3946 | 2979 | 75.5% | 44.9 |

Hands-only fps is this CPU laptop with nothing else running (a re-run of all 26 clips; the first pass, run beside the detector, was slower and is not used). The two passes gave byte-identical landmarks. The 8 fps stream has more frames with a hand than the 4 fps stream (75.5 vs 70.1%); a cause (tracking is easier at smaller motion) is a guess, not tested.
Share of frames with the index fingertip (landmark 8) inside the grown card box: margin 0.10 / 0.25 / 0.50 = 4.5 / 4.9 / 5.9% at 4 fps (89 / 96 / 117 frames), 8.6 / 9.3 / 10.4% at 8 fps (340 / 368 / 412 frames).

### 4. Validity check: PASSED
V0 with the tracker defaults (hysteresis 5, release 5, baseline 10, frame counts) on the 4 fps hands reproduces the real `StateTracker` start_pressed events of cp2 at the same times: x011 6.24 and 15.01, x019 10.74, x001 7.23 and 16.50, x037 none, x043 6.27 and 11.50. The tested helper (`press_events`) also agrees with `state.tracker.StateTracker` on four flag sequences.

### 5. Variants x hold x fps
Variants: V0 current (any of 21 landmarks, margin 0.10); V1a / V1b / V1c index tip only, margin 0.10 / 0.25 / 0.50; V2 any fingertip (4, 8, 12, 16, 20), margin 0.10. Hold H in seconds replaces the hysteresis (frames = ceil(H x fps)). Baseline and release are fixed in seconds at the cp2 defaults (2.5 s and 1.25 s = 10 and 5 frames at 4 fps), so 20 and 10 frames at 8 fps. Cells: **exact / missed / extra over all 26 clips ; same over the 14 correct clips.** (A clip is judged by event count only, so none is both missed and extra.)

| fps | variant | H = 0.25 s | H = 0.5 s | H = 0.75 s | H = 1.0 s |
|---|---|---|---|---|---|
| 4 | V0 | 3/0/23 ; 0/0/14 | 5/1/20 ; 0/0/14 | 4/3/19 ; 0/0/14 | 7/3/16 ; 1/0/13 |
| 4 | V1a | 8/17/1 ; 4/10/0 | 9/17/0 ; 4/10/0 | 8/18/0 ; 4/10/0 | 8/18/0 ; 4/10/0 |
| 4 | V1b | 9/16/1 ; 5/9/0 | 8/17/1 ; 4/10/0 | 8/18/0 ; 4/10/0 | 8/18/0 ; 4/10/0 |
| 4 | V1c | 11/12/3 ; 8/5/1 | 7/17/2 ; 4/10/0 | 8/17/1 ; 4/10/0 | 8/17/1 ; 4/10/0 |
| 4 | V2 | 13/7/6 ; 9/1/4 | 10/12/4 ; 6/5/3 | 10/15/1 ; 5/8/1 | 10/16/0 ; 5/9/0 |
| 8 | V0 | 5/0/21 ; 0/0/14 | 8/0/18 ; 0/0/14 | 8/2/16 ; 1/0/13 | 9/2/15 ; 1/0/13 |
| 8 | V1a | 14/11/1 ; 8/6/0 | 14/12/0 ; 7/7/0 | 14/12/0 ; 7/7/0 | 12/14/0 ; 5/9/0 |
| 8 | V1b | 14/11/1 ; 8/6/0 | 14/12/0 ; 7/7/0 | 14/12/0 ; 7/7/0 | 14/12/0 ; 7/7/0 |
| 8 | V1c | 15/10/1 ; 9/5/0 | 14/12/0 ; 7/7/0 | 14/12/0 ; 7/7/0 | 14/12/0 ; 7/7/0 |
| 8 | V2 | 12/6/8 ; 6/2/6 | 14/9/3 ; 7/4/3 | 16/10/0 ; 9/5/0 | 14/12/0 ; 7/7/0 |

Sensitivity: holding baseline and release at the tracker's raw frame counts (10 and 5 frames) instead of seconds changes any count by at most 3 clips in any cell and does not change the acceptance outcome (keys ending `|frames` in `data/spikes/cp3/results.json`).

### 6. What the data says
- **V0 (the current rule) gives extra events in 13 or 14 of the 14 correct clips at every hold and rate** (at 8 fps and 1.0 s, 15 of 26 clips have extra events). A longer hold does not fix it, because an arm or palm stays over the card for seconds. V0 misses at most 3 of 26 clips (a palm is nearly always there).
- **Index-tip-only variants trade extras for misses**: extras drop to 0 to 3 of 26 but 10 to 18 of 26 clips are missed. Widening the margin to 0.50 helps little (best correct-exact 9 of 14, V1c at 8 fps, 0.25 s).
- **V2 (any fingertip) at 8 fps and 0.75 s (6 frames) is the best cell**: 16 of 26 exact, 10 missed, **0 extra**; correct clips 9 exact, 5 missed, 0 extra (V1c at 8 fps and 0.25 s is the same 9 of 14 but 15 of 26 exact and 1 extra). Shorter V2 holds add extras (0.25 s: 8 of 26 extra), longer holds add misses (1.0 s: 12 missed). At 4 fps V2 needs H = 1.0 s for 0 extras and then misses 16 of 26.
- **Why clips are missed** (`data/spikes/cp3/diagnostics.md`, 8 fps, per clip). In 9 clips (x006 x007 x008 x012 x026 x027 x034 x037 x044) no variant sees a fingertip near the card long enough: the nearest index tip is 18 to 65 px from the card box (0.15 to 0.56 card widths) and the longest fingertip run inside the grown box is 0 to 3 frames, while a hand is detected in 47 to 74% of their frames. This matches the cp2 entry: the card sits at the frame's bottom edge, so a real press leaves little or no fingertip in frame (x001 had a 14-frame no-hand gap at 4 fps), or an arm covers the card (x037, latched at baseline in cp2). In x043 (expects 0) V0 has 28 frames with a landmark on the card, palm or arm only (nearest index tip 108 px). x004 is missed by V2 at 0.75 s only because its fingertip run is 0.62 s. x001 is missed by V1c at 0.25 s only because its index-tip run is 0.12 s (another fingertip stays 1.5 s). The one extra of V1c at 8 fps and 0.25 s is x031 (second event at 18.14 s, a second touch after release in a swap clip).
- Expected counts come from the scripts' `performed_steps`; there are no per-frame press labels, so an "exact" can in principle be a coincidence (a fingertip touch that is not the press). Press times were not checked against video for this addendum.

### 7. Recommendation (from train numbers only)
No variant is ACCEPTABLE, so no rule is recommended as passing. The numbers say the rule is not the bottleneck: with this layout the fingertip is detected near the card in too few frames for any hold to reach 12 of 14. If the Lead wants the best available rule for a P2 session to implement and test on val, the train numbers point to **V2 (any fingertip, margin 0.10) with a hold of 0.75 s** (6 frames at 8 fps; at 4 fps 3 frames gives 5 of 14 correct exact and 1 extra, so the rate matters), run at 8 fps or higher. The other train-supported change is on the perception side: the cp2 observation about moving the card up by about one hand length would address the 9 missed clips directly, but is untested (it needs a re-recording).

### 8. NOT known
Val and test were not opened, so nothing here is validated; one performer, one setup, one camera position, no gloves; 26 clips, 14 of them "correct" (one clip is 7 points against a threshold of 12 of 14); true press moments are not labeled; MediaPipe used its default 0.5 confidences; converting the hold in seconds to frames at the real frame rate is a P2.6 decision (here only ceil() at 4 and 8 fps); the hold replaces hysteresis only, release and baseline were not tuned; V2's landmark set and the margins were fixed before the run, not searched.

Code: `training/spikes/touch_variants.py` (pure helpers, tested), `run_cp3.py` (orchestration: plan, card cache, cards, hands, analyze, report). Tests: `tests/unit/training/test_spikes_touch_variants.py` (22). Artifacts (git-ignored): `data/spikes/cp3/` (`plan.json`, `cards.json`, `results.json`, `diagnostics.md`, `hands_timing.json`, per run `raw.jsonl`, `hands_4.jsonl`, `hands_8.jsonl`, logs). Stop here: the Lead decides the variant; a separate P2 session implements it.


## 2026-10-02 P2 DECISION - hand_touching uses fingertips only

**What.** `state/tracker.py` evaluates `hand_touching(label)` as true iff at least one fingertip landmark (module constant `FINGERTIP_LANDMARKS = (4, 8, 12, 16, 20)`, MediaPipe indices) of any hand lies inside the best floor-filtered detection of the label grown by `touch_margin_frac` on each side (box test inclusive, as before). Before: any of the 21 landmarks. Unchanged: the `HandTouchingRule(label)` type, the `PerceptionConfig` fields, `config/experiment.json`, margin 0.10, the touching hand's score in the step confidence, `hysteresis_frames` (default 5). New tests: `tests/unit/state/test_tracker_fingertips.py` (marker F4).

**Accepted limitation.** On the train clips the best fingertip variant (any fingertip, margin 0.10, hold 0.75 s at 8 fps) gives exactly one START press in 9 of 14 correct clips with 0 extras; the 80% acceptance bar was NOT met; the Lead accepted this as a documented limitation. One performer, one setup, no gloves; val and test not looked at (evidence: the P1.2 addendum and DECISION entries).

**P2.6 knobs.** The hold (`hysteresis_frames`; the measured hold was 0.75 s, i.e. 6 frames at 8 fps, the default stays 5 until then) and `touch_margin_frac`, tuned and checked on val only.

**See also.** The CONTRACT entry "2026-10-02 P2 CONTRACT - hand_touching uses fingertips only (Lead decision D74)" on branch `contract/touch-fingertips` (contracts.py comment and essential-features.md F4 text); that PR must be merged first.
## 2026-10-02 P2 CONTRACT - hand_touching uses fingertips only (Lead decision D74)

**What changes.** The `hand_touching(label)` rule text (contracts.py section 3 comment block, essential-features.md F4 item 3 and its pitfall sentence) changes from "at least one of the 21 landmarks of any hand lies inside the grown box" to "at least one FINGERTIP landmark (MediaPipe indices 4, 8, 12, 16, 20) of any hand lies inside the box grown by `touch_margin_frac` on each side". Comment and prose only. The behaviour change itself is implemented on branch `p2-runtime` (state/tracker.py, DECISION entry there).

**Why.** See the entries "2026-10-02 P1.2 addendum - START press rule variants on train clips (measurement only)" and "2026-10-02 P1.2 DECISION - zero-shot auto-labeler (Grounding DINO tiny) and prop setup" (section 2, cause A). Cited, not re-measured: with the any-landmark rule, 13 or 14 of the 14 correct train clips get extra START presses (an arm or palm over the card); the best fingertip variant (any fingertip, margin 0.10, hold 0.75 s at 8 fps) gives exactly one press in 9 of 14 correct clips with 0 extras.

**What does NOT change.** The `HandTouchingRule(label)` type and every other field, type, validator and default in contracts.py; the `PerceptionConfig` fields (`extra="forbid"`); `config/experiment.json`; best detection of the label (floor-filtered); `touch_margin_frac` 0.10; the touching hand's `score` in the step confidence; `hysteresis_frames` (default 5).

**Limitation (accepted by the Lead).** 9 of 14 correct train clips are recognised, 0 extras; the 80% acceptance bar (12 of 14) was NOT met. One performer, one setup, no gloves; val and test were not looked at.

**Knobs.** The hold (`hysteresis_frames`; the measured hold was 0.75 s, i.e. 6 frames at 8 fps) and the margin are P2.6 knobs, tuned and checked on val only.


## 2026-10-02 P1.4 S-D - dataset code, rules v4, review sheets

**Frames and time.** `python -m training.sample_frames` (stride round(fps/2), 32x32 de-dup at 6/255 or 1 s, cap 120 evenly subsampled): 1069 frames over 46 runs: train 624, val 220, test 225 (per run 4 to 35; x023 has 4 because the clip is 4 s). This is far below the 2,000 to 3,000 train images F14 expects; the runs are short (4 to 25 s). Density was NOT lowered. Labeling calls: 1069 x 2 movable + 687 static (5 spread frames x 3 classes per run) = 2825, about 3.8 s each measured = about 3 h (estimate by the packet formula: 3.2 h, under the 6 h limit).

**Rules.** Kept v3 (frozen in `data/labels/frozen_rules.json`, git-ignored, sha256 af7e6763...a7ba; also in the commit message of d8c783f): START white test sat <= 61, val >= 128; container band [0.032423, 0.069419]; static bands outer [0.1819, 0.3281], tray [0.0190, 0.1880], start [0.001, 0.1076]. Static boxes are now decided once per run and applied to every frame of it. v4 (evaluated by `training/spikes/run_v4_eval.py`, code in `select_v4.py`) = v3 + area ceiling 0.0922 (2 x clean median 0.0461) + colour-verified floor 0.015 (red hue -1.0..10.1 sat 87.7..149.8; lime hue 23.7..38.8 sat 112..162.7, central half of the box, 5th-95th percentile +-2 hue +-10 sat) + size cap 1.6 x median (261.8 x 188.0 px). Clean frames: 153 container boxes from the 80 reviewed frames (60 + 20) that v3 produced and the Lead did not mark bad.
**Safeguard result: tripped, v3 kept.** v4 changed 0 unmarked cells, 3 Lead-bad cells: x003#416 yellow and x027#266 red recovered (looked right on my sheet), but x034#156 yellow got back the box the Lead marked wrong_box in v2 (stretched over the hand, area 0.0867 below the ceiling, width 229 below the cap). Not recovered: x044#165 yellow (correct box 0.0725 but 193 px tall vs cap 188), x034#77 yellow (stretched box is only 1.29 x median wide), x003#324 and x008#572 (only oversized candidates). The colour test would not protect the ceiling band for red: hand+container boxes x020#84 and x033#103 (0.08, hue 9 to 10, sat 99 to 102) are inside the red range; they were not chosen only because better candidates existed. Only 3 cached candidates lie in 0.015..0.0324 (2 pass colour: x003#416 container, x030#294 container; 0 skin fragments); too few to say skin never passes. Re-check this on the full raw cache once it exists.

**Review files.** `data/label_review.csv` (400 rows, 80 train frames: 60 seeded stratified, at least 2 per train run, plus 20 flagged = largest thumbnail change, max 2 per run; a motion proxy, not auto-labeler flags) and `data/review/static_review.csv` (138 rows = 46 runs x 3 static classes, frame = first sampled frame of the run) are pre-filled verdict=ok. Columns are Plan 5.8's five plus `visibility` and `note` (EXTENSION of 5.8; visibility required when reason=missing, hidden-and-missing not counted as bad). Classes with no box drawn carry a note: ok is wrong there unless the object is absent. Sheets: `data/review/review_sheet_01..07.jpg` (12 frames each), `static_sheet_01..04.jpg`. Loader and gate (`<= 10 %` bad per class): `training/review_sheet.py`.
Label status of the labeled review sample so far: 62 ok, 21 excluded (missing red 11, yellow 8, both 2) = 25 %. Hard cases: of 21 excluded, 20 have a hand and 20 a hand overlapping the missing class (proxy: raw candidate or nearest box of the class); of 62 kept frames 38 have a hand over a container. Excluded frames are the hand-covered ones, so the training set under-represents them.

**Resume.** `python -m training.autolabel --status` (done/total, s per call, ETA); `--detach` starts the queue in a detached process (log `data/labels/autolabel.log`, pid file `autolabel.pid`), `--resume` runs it in the terminal; the raw cache is append-only, so a kill loses at most one call. Then `--build` (rules offline: COCO per split in `data/labels/coco/`, `missing_frames.csv`, `summary.json`, `frame_labels.json`), `--hands` (hand boxes, already done), `python -m training.build_dataset`. Job started 2026-10-03 (about 2.1 h remained at the last status).

**Code.** `sample_frames.py`, `autolabel.py`, `autolabel_rules.py`, `prompts.yaml`, `review_sheet.py`, `build_dataset.py`, `spikes/select_v4.py`, `spikes/run_v4_eval.py` and tests (`tests/unit/training/`). COCO category ids are 0-based = index in experiment.classes; whether RF-DETR wants a Roboflow-style placeholder id 0 is unverified (P1.5 to check).

**NOT done.** Gold subset, finetune, eval, label editor (`--edit`, `--gold`), building the real dataset, the Lead's review of either CSV, any look at val or test review (val and test frames are labeled by the queue but were not shown in gate sheets; the static sheet does show all 46 runs' static boxes as the packet asks).
**Known limits.** One performer, one setup, no gloves; 25 % of the sampled frames are excluded for a missing container (hand-covered or blurred); v3's known weaknesses remain (rotated, blurred or hand-covered containers; x034#77 stretched box); the review sample is small (80 frames, a single bad frame is 1.25 points).


## 2026-10-02 P1.4b S-D2 - review outcome and label editor

**Review outcome** (`python -m training.review_sheet --outcome`, strict loaders on `data/label_review.csv` and `data/review/static_review.csv`, written to `reports/dataset.json` under `label_review`; `build_dataset --review-done` records the same structure). Version 1, stratified 60 (the plan's gate, limit 10 %): red_box 8/60 = 13.3 % **FAIL**, yellow_box 5/60 = 8.3 %, outer_box, tray, start_button 0. Flagged 20 (reported separately, a hard-case sample, not a gate): red_box 5/20 = 25 %, yellow_box 6/20 = 30 %. Version 2, the kept-frame gate (bad fraction among frames the labeler did NOT exclude, i.e. the frames that become training labels): 59 kept of 80, yellow_box 1/59 = 1.7 %, every other class 0, **passes**; stratified-only 48 kept: yellow_box 1/48 = 2.1 %, passes. Static review: 46 runs, 0 bad rows of 138. The 24 bad rows are 23 missing + 1 wrong_box; none is hidden-and-missing (18 partial, 5 visible). The failure is coverage loss, not label noise: the labeler excludes a frame when a container has no box (21 of 80 frames), so those frames never become labels. **Check 23 = 23:** the Lead's 23 missing cells equal exactly the 23 cells the labeler noted "no box drawn" (the loader matches that note as a prefix, because the Lead appended remarks) and the 23 (frame, class) pairs of the labeler's excluded frames; no difference in either direction. The plan's gate wording is satisfied only through Stage 4b: correct or exclude, rebuild, review a fresh sample.

**Label editor.** `python -m training.review_sheet --edit --split <s> [--gold [N]] [--static] [--only-sample]` writes one static HTML page `data/review/editor_<s>.html` (vanilla JS, inline CSS, no server, no network, no CDN; frame list, auto boxes and proposals inlined as `window.EDITOR_DATA`; images by relative path `../frames/<s>/<run>_<frame>.jpg`, so open it from `data/review/`). Code: `training/label_editor/` (`build.py` generator, `editor_core.js` pure helpers, `editor_ui.js`, `editor.html`, `editor.css`). Frame order: (a) frames the labeler excluded, by run; (b) with `--gold` N (default 6) seeded frames per run, one drawn from each of N equal time chunks of the run (a short run gives all its frames); (c) with `--static` one frame per run showing only the three static boxes (an edit there becomes a `static_overrides` entry for the whole run). Pending frames are never shown. The page recomputes labels from the raw cache and the frozen rules (offline, no model), so it is current, not tied to `frame_labels.json`.
*Proposals.* For a class with no box, every cached raw candidate (up to 10, score floor 0.20, including those the rules rejected) is listed numbered with score, area fraction and the reason (select_v4's own rejection code with numbers, e.g. "area 0.2367 outside 0.0324 to 0.0694 (too large)", "overlaps the run's tray"; static classes: out of band or not the run's consensus), and drawn dashed and faint in the class colour with number and score.
*Keys.* Left/Right previous/next frame, N next unfinished; digits 1-9 (0 = tenth) accept candidate n for the highlighted missing class when nothing is selected (Tab = next missing class); with a box selected (click it, or B), 1-5 set its class in `experiment.classes` order (a box already in that class swaps, nothing is lost); Delete deletes; drag on a selected box moves or resizes it, drag anywhere else draws a box for the highlighted missing class; X excludes and moves on (X again un-excludes); V verifies (needs all five classes) and moves on; Z or Ctrl+Z undo (every edit, across frames); S downloads; ? help. Any edit clears "verified". Autosave to localStorage (only touched frames), "Load previous overlay" resumes from a downloaded file (entries for frames not on the page are kept and written out again), "Clear autosave".
*Overlay.* Plan 5.8 exactly (`version`, `split`, `static_overrides`, `frames`); a frame entry carries all five class boxes (auto plus accepted), original-resolution pixels, classes in experiment order; a frame that is not excluded and still lacks a class is NOT written (it would train as a negative) and is named in the download message; an entry also carries its run's static fix unless the user changed that box on that frame. The page downloads `<split>.json`; the Lead saves it as `data/corrections/<split>.json`. The static-check verdict (V on a static view) is not part of the overlay (no field for it in 5.8); it lives in the autosave only, and `static_review.csv` stays the record.

**Tests** (Python, 330 in `tests/unit/training`, check.py --quick 696+ green): generator (frame order, pending skipped, gold pick seeded and stratified, proposals with scores and reasons, colours BGR to RGB, safe inlining, node syntax check of the three inline scripts); 19 JS unit tests of the pure helpers run with node 22 (clamp, move, resize, hit test, class keys, accept, swap, overlay assembly and round trip, wrong split and unknown class refused); end to end through node and `build_dataset`: a simulated session accepts a candidate on an auto-excluded frame, the overlay passes build_dataset's assertions, the frame appears with the accepted box and all five classes, a frame entry replaces wholesale, a static override covers the run. Tests skip if node is absent (it is present here).

**Verified on the 80 reviewed train frames only** (`--edit --split train --only-sample --static`; page also holds 26 static check frames, one per train run whose static calls are finished): 21 excluded frames, 23 missing cells (red 13, yellow 10). Proposals per cell: red [4,1,2,1,1,2,2,2,1,2,1,2,2], yellow [3,1,1,1,3,2,3,4,4,4]; every missing cell has at least one cached candidate. Coverage proxy (area 0.015 to 0.11 of the frame and IoU <= 0.5 with each static box): 16 of 23 cells have a container-sized, non-static candidate (red 6 of 13, yellow 10 of 10). **This is a proxy: I did not look at which candidate is right, so it is not a recovery number.** Red is the weak class (7 cells whose only candidates are tiny, oversized or whole-box, e.g. x008#570 has one candidate of 23.6 % of the frame); those cells need a drawn box or exclusion. I drove the page in the in-app browser over a local http server (draw, select, move, accept, verify, exclude, download, load previous, autosave restore all worked); I did NOT test it from file:// in Chrome or Edge, which is the Lead's check, and the test state lived in a different origin so nothing was written to `data/corrections/`.

**NOT done.** Gold subset not verified (and `--gold` not generated for val or test; this session generated only train); no overlay written; the Lead has not used the page; no fresh review sample after correcting (the plan's gate stays open for red_box); the labeler was still running (see its status in the report; the page is for frames finished at generation time and should be regenerated before a full pass); GPU choice open (Colab/Kaggle). **Limits.** One box per class per frame (matches build_dataset); the page does not keep the labeler's "why excluded" beyond the reason string; image display scale is fit-to-window so small boxes need care; no touch support tested; the coverage proxy thresholds are my judgment calls; `localStorage` is per origin, so a page opened from a different path or by http has separate autosave.

## 2026-10-03 P1.5 prep (S-E) - GPU scripts, eval script, preflight, training guide

**Built** (new files only; no S-D file touched; nothing written under `data/labels`; the labeler was only queried with `--status`). All entry points take explicit paths and run unchanged on any machine; none downloads (pretrained weights must already be on disk, `--allow-download` is the Lead's-yes switch, RF-DETR only).
- `training/gpu_preflight.py` (`python -m training.gpu_preflight --model rfdetr|yolo11n --batch-size 4 --n-train N --epochs E [--require-cuda]`): GPU name and VRAM, CUDA availability, CPU vs CUDA torch build, torch and rfdetr versions; times K forward+backward iterations of the real network on synthetic input (5 on CUDA, 2 on CPU; the first is dropped as warm-up from 3 samples up) and prints seconds per iteration plus a labelled ESTIMATE of seconds per epoch and total (leaves out validation, data loading, optimizer and the criterion; the backward uses a surrogate loss on the raw outputs). `--require-cuda` without CUDA exits 2 with a message; missing weights exit 3.
- `training/finetune.py` (`python -m training.finetune --model rfdetr|yolo11n --dataset-dir D --output-dir O [--epochs] [--lr] [--batch-size 4] [--grad-accum 4] [--seed 0] [--device auto] [--resume] [--dry-run] [--weights] [--allow-download]`): RF-DETR path per Stage 6 (`RFDETRNano(pretrain_weights=<cached file>).train(dataset_dir, epochs, batch_size=4, grad_accum_steps=4, lr, output_dir, early_stopping=True, seed, class_names, tensorboard=False)`), best-EMA checkpoint (`checkpoint_best_ema.pth`) copied to `O/detector.pth`; YOLO11n path through `training/coco_to_yolo.py` and Ultralytics (same seed, AdamW at the same lr so `lr0` is honoured, batch 16, `best.pt` to `O/detector_yolo11n.pt`). `O/train_summary.json` carries model, versions (torch, torch build, rfdetr, ultralytics, python), seed, epochs, lr, batch settings, n_train, n_valid, measured seconds per iteration, total seconds, best validation mAP, dataset stamp (from `reports/dataset.json`), classes, detector file and sha256, device name, notes. Refuses (exit 2) before training: class names differ from `config/experiment.json` in order or ids not 0..n-1 (`build_dataset.check_coco`, called), split folder / annotation file / listed image missing, a run id in two splits (`build_dataset._parse_file_name`).
- `training/coco_to_yolo.py`: pure converter (normalised centre boxes, class index = id-sorted category position, empty images get an empty label file, test split never converted) plus the dataset yaml.
- `training/eval_detector.py` (`python -m training.eval_detector --model ... --weights ... --dataset-dir ... --split valid|test --out reports/detector_eval.json`): `supervision.metrics` only (MeanAveragePrecision, Precision, Recall; nothing re-implemented). Per-class precision and recall at IoU 0.5 and at `PerceptionConfig().detector_conf_floor` (read, not hardcoded; 0.30 today), mAP50 and mAP50-95 on predictions down to confidence 0.001; every metric twice: all labelled frames and the verified gold subset (`verified: true` in `data/corrections/<split>.json`; valid maps to the overlay's `val`); an empty gold set reports null and a note. Report carries generated_at, weights sha256, dataset stamp, the floor, acceptance sha256. **`--split test` is refused (exit 2) unless `config/acceptance.yaml` exists**, and then its sha256 is recorded (refusal tested first). Precision is null when a class has no prediction at the floor, recall null when it has no ground truth.
- `training/README_GPU.md`: path A (teammate's CUDA machine), path B (Kaggle, T4, 5-minute smoke test first), YOLO fallback, troubleshooting; every downloading step marked as needing the Lead's yes.

**Epoch and lr rule** (pure, tested at 1, 499, 500, 1999, 2000, 9999, 10000): n_train < 500: 100 to 200 epochs; 500 to 1,999: 50 to 100; 2,000 to 9,999: 30 to 50 (bands are [lo, hi), so 500 is in the second and 2,000 in the third); 10,000 and above: Stage 6 is silent, so `--epochs` is required. Default epochs = the upper end of the range (judgment call: early stopping on validation mAP bounds the cost); lr 5e-5 below 1,000 images, else 1e-4. For the expected 470 to 620 train images: 200 epochs at 470, 100 at 500 and above, lr 5e-5.

**COCO id finding** (installed rfdetr 1.11.0, read from `rfdetr/datasets/coco.py` and `rfdetr/detr.py`, then run on a synthetic dataset; pinned by `tests/unit/training/test_rfdetr_coco_ids.py`). rfdetr does **not** expect ids to start at 1 and does not need a Roboflow placeholder category 0: `CocoDetection(remap_category_ids=True)` builds `cat2label = {category id: position in the id-sorted category list}` from the train split (valid and test reuse it), so our ids 0..4 become label indices 0..4 and nothing is dropped. The Roboflow placeholder is removed only when other categories name it as their `supercategory` and it has no annotation; our categories have no supercategory, so id 0 (`outer_box`) is kept. Class count is 5 (`_detect_num_classes_for_training`); the head has 5 + 1 outputs, the spare slot is never a label; `predict()` returns `class_id` as a 0-based index into `class_names`. **No change to `build_dataset` is needed, so there is no proposed change for S-F1.** Caveat: `CocoDetection` itself could not be instantiated here (needs pycocotools, below), so the end-to-end check stops at the mapping helpers it calls.

**Measured on the Lead's CPU laptop** (torch 2.14.0+cpu, no CUDA): preflight RF-DETR-Nano at batch 4, 2 iterations: 6.70 s per iteration, which extrapolates to 790.6 s per epoch and 21.96 h for 470 images x 100 epochs (an estimate, a lower bound). Dry runs (6 synthetic images: 3 train, 2 valid, 1 test): YOLO11n completed end to end (1 epoch, 7.8 s, summary and `detector_yolo11n.pt` written; its mAP is meaningless). The RF-DETR dry run is PARTIAL: it built the model and ran 2 forward+backward iterations (3.44 s per iteration at batch 2) but `train()` was NOT called, see finding 1. The rfdetr `train()` argument names were checked against the installed `TrainConfig` (`get_train_config`).

**Surprises and findings.**
1. **The lockfile's `train` group cannot train RF-DETR.** `train = [run, supervision, pillow]` and `run` has `rfdetr>=1.10,<2` without the `[train]` extra, so pytorch_lightning, torchmetrics, pycocotools (and faster-coco-eval, vernier, ...) are not installed; `RFDETRNano().train()` raises ImportError (rfdetr's own message: install `rfdetr[train,loggers]`). Confirmed here: those three modules are missing. **Request to P2:** change the `train` group to depend on `rfdetr[train]` (check the added packages' licences, as the plan requires) and re-lock; until then a GPU machine needs the extra installed by hand, outside the lockfile, which must be recorded. `finetune.py` detects this and exits 4 with the missing names.
2. A CPU-only torch is what Windows gets from the pinned PyPI wheel; the CUDA build is outside the lockfile (guide step A5, only P2 edits `uv.lock`). Whether the Linux PyPI wheel has CUDA is unverified here.
3. A loaded Ultralytics checkpoint is frozen (`requires_grad` False); the preflight unfreezes it as its own trainer does.
4. `ruff format` over the whole `training/` tree reformats three S-D spike files; I reverted those and formatted only my own files (no existing file differs from develop).

**NOT done.** No training of any kind (no `train()` call, no GPU run), no Kaggle run, no CUDA machine seen, no test-split evaluation (and `eval_detector` was not run against real weights or a real dataset: `data/dataset` does not exist yet, the labeler is still finishing), `config/acceptance.yaml` not written, no `weights/detector.pth` and no `MANIFEST.json` change. `eval_detector`'s `RFDETR.from_checkpoint` load path is untested on a real fine-tuned checkpoint (its predict path was exercised only with a random 5-class head), the YOLO eval path is untested, `--resume` untested, `--allow-download` untested. The preflight numbers are an extrapolation, not a measurement of training time.

**Open: which GPU.** Not decided: a teammate's RTX 4090 or 4070 Ti, or Kaggle (T4 unverified, see README_GPU.md B3). Decide, then: P2 lockfile change (finding 1), record the exact CUDA torch command and versions in the training summary and here, run the preflight with `--require-cuda`, then `finetune`.

## 2026-10-03 P1.5 prep (S-F1a) - dataset built with corrections, acceptance.yaml, fresh review sample
**Built.** (1) `training/build_dataset.py`: `reports/dataset.json` now keeps its `label_review` block across rebuilds (`merge_label_review`: a recorded block is never replaced by `not_recorded`; before this a build without `--review-done` overwrote it). Overlay edge cases are `BuildError`s that name the frame (or run and class for a static override): duplicate class in one entry, reversed, zero-size, non-four-number or out-of-image box (nothing is clipped silently; a box touching the image edge is fine). A static override on a labeler-excluded frame leaves it excluded (documented in the module docstring, tested); only the frame's own entry can bring it back. Report adds `frames_labeler_excluded`, `frames_recovered`, `annotations_per_run` and, per split, `hand_over_container` (kept frames whose hand box intersects a red or yellow box, from the existing hand cache; proxy). **Stamp choice:** `dataset_stamp` hashes the three annotation files and the overlay files, NOT the JPEG bytes: the images are unmodified copies of the frame cache (a function of the run videos, `video_sha256` in the manifest) and the annotations name every image and size, so a changed image set changes the stamp; a re-encoded frame with an identical name would not. Accepted for cost and encoder independence; image counts per split are in the report. (2) `config/acceptance.yaml`: exactly the Plan 5.8 block, header says written 2026-10-03 before any evaluation on test, amended at most once at P2.6 with both people's ack. **Own commit `c89bee8`** (that commit touches only that file; no model or test evaluation exists before it). `eval_detector` reads no threshold keys: it records the file's sha256 and refuses `--split test` without it, so no key rename was needed. A test (`test_acceptance_yaml.py`) pins the values and validates against `contracts.AcceptanceConfig`. One S-E test (`test_cli_end_to_end_on_valid_with_a_fake_detector`) assumed the default acceptance path is absent; it now passes `--acceptance` explicitly. (3) `training/review_sheet.py`: `--fresh-sample [--n 60]` and `--fresh-outcome`; the `draw_review_sample` docstring now says random, not first.
**Dataset built** (`python -m training.build_dataset --review-done`; before it, `python -m training.autolabel --build` was run, offline, no model: it applied rules v3 to the finished raw cache and regenerated `data/labels/frame_labels.json` and `summary.json`, which were still the 83-frame state). Stamp `66f3106e49c14995`. Images per split: train 594 (of 624 sampled), valid 173 (of 220), test 187 (of 225). Boxes per class: 594 / 173 / 187 each for all five classes (every image has all five). Labeler-excluded frames: train 115, valid 47, test 38. Train overlay (version 1, 115 entries): 85 recovered and verified (all five classes), 30 stay excluded; 509 labeler-kept + 85 = 594 as expected; 26 runs, 4 to 31 images per run. Valid and test are auto-labeled kept frames only (no overlay, no verified frames). Share of kept frames with a hand over a container (proxy): train 370/594 = 62 %, valid 97/173 = 56 %, test 105/187 = 56 %; no frame lacks hand data. Layout checked on disk: folders `train`, `valid`, `test`; every JPEG is in its split's annotation file and vice versa; category names equal `experiment.classes` in order, ids 0..4; every image is of a run of that split (manifest); five annotations per image.
**Label review block in `reports/dataset.json` is still the OLD one** (the 80 frames reviewed before the overlay: red_box 8/60 = 13.3 % FAIL). It is kept, not replaced; the gate result after the repair comes only from the fresh sample below.
**Fresh review sample** (`python -m training.review_sheet --fresh-sample`, reads the BUILT train dataset, overlay applied): 60 train frames, seed 20261003 (new; the earlier was 20261002), none of the 80 in `data/review/sample.json`, 26 runs with 2 to 4 frames each (at least 2 everywhere), 11 of the 60 are overlay-corrected frames. Files: `data/review/fresh_sheet_01.jpg` to `fresh_sheet_05.jpg` (12 per sheet, same legend and `<run>#<frame>` tags), `data/review/fresh_sample.json` (seed, cells, the 11 corrected cells), prefilled `data/review/fresh_review.csv` (300 rows, all `ok`, columns run_id,frame_id,class,verdict,reason,visibility,note). After the Lead fills it in: `python -m training.review_sheet --fresh-outcome` loads it with the strict loader and applies the per-class gate (`bad_fractions`, limit 10 %, hidden-and-missing not counted).
**Training-size facts (no training run).** n_train 594, n_valid 173. By the Stage 6 rule (`finetune.epoch_range`): 500 to 2,000 images gives 50 to 100 epochs, default 100; lr 5e-5. Batch 4 x accumulation 4 = effective 16: 149 micro-batches (iterations of 4 images) per epoch, 38 optimizer steps per epoch. Reference only: the S-E preflight measured 6.70 s per iteration on the CPU laptop, which is about 998 s per epoch and about 27.7 h for 100 epochs on CPU; a CUDA GPU is needed.
**Correction note (append-only) to the 2026-10-02 P1.4 S-D entry:** "62 ok" and "59 kept" are not two counts of the same frames. "62 ok, 21 excluded" counted all 83 train frames labeled when that line was written; "59 kept of 80" counts only the 80 review-sample frames, 21 of which are excluded (the same 21). The 3 extra kept frames (62 - 59) are labeled frames outside the 80-frame sample (inferred from the two denominators; both lines share the 21 excluded). The gate uses the 80-frame figure.
**Surprises.** (a) After the first build, the three annotation files and `reports/dataset.json` were found filled with NUL bytes (right size, no content) some minutes later; the cause is not known (most likely an interrupted flush while the machine slept, nothing in the code reads or writes them again). The build was rerun, the files re-read byte by byte (no NUL) twice, and the stamp is the one above. If `data/dataset` ever fails to parse, rebuild. (b) `frame_labels.json` had to be rebuilt from the cache before the build could run (986 frames were still `pending` in it).
**NOT done.** No gold subset (no verified frames in valid or test), no val or test overlay, no training, no Kaggle action, no model run, no evaluation on any split; the fresh sample is not reviewed yet; the 80-frame review's red_box failure has not been re-gated.
**Open.** (1) The Lead reviews the 60 fresh frames, saves `fresh_review.csv`, then `--fresh-outcome`: per class at most 10 %. (2) Gold subset and val/test overlay (needed for the detector gate, which is checked on gold test frames). (3) Which GPU (see the S-E entry), P2 lockfile change for the CUDA torch build. (4) `eval_detector` does not compare to the thresholds itself; whoever reads `reports/detector_eval.json` does.

## 2026-10-03 P1.5 prep (S-F1b) - exclusion, verify_dataset, Kaggle package (prepared, not uploaded)
**Exclusion.** The fresh review's one bad row (`x012#570` yellow_box, the box sits on the red container; labeler-kept, so label noise) excludes that frame: rule "every bad row in a fresh review excludes its frame". New tool `python -m training.corrections exclude --split train --frame x012_570.jpg [--force]`: copies the overlay to `train.json.bak-<sha256 prefix>-<timestamp>` first, refuses an unknown split, a missing overlay, a file with NUL bytes, a frame that is not a sampled frame of the split, and a frame already corrected (unless `--force`), writes the new file to a temporary name, re-validates it with `build_dataset`'s own checks (`validate_overlay_path`, new, shared) and only then replaces the original. It writes LF bytes: the first real run used `write_text` and silently turned the file CRLF on Windows (58,841 to 63,420 bytes); that run was caught by the size, the file restored from its backup, the tool fixed (bytes, plus a test that no CR appears) and rerun. Result: overlay 116 entries = 85 corrected (verified) + 31 excluded, 58,924 bytes, sha256 prefix 5474f7695e1cf07f. Backups kept in `data/corrections/`: `train.json.bak-fbca27d7cbf2-20261003T135925` (the original, 58,841 bytes, sha256 fbca27d7...) and `...T135938` (same bytes, from the rerun); `train__5_.json` there is a browser duplicate of the original and is not read by the build. The overlay is force-added to git in its own commit (43aa603; a hand-made input of the stamp, about 59 KB, not 30 KB).
**Fresh outcome recorded.** `python -m training.review_sheet --fresh-outcome --record` writes `label_review.fresh_after_repair` into `reports/dataset.json`: 60 frames, seed 20261003, per-class bad: outer_box 0, tray 0, red_box 0, yellow_box 1 (1.7 %), start_button 0, gate_passes true, every bad cell by frame, class and reason, the CSV sha256, and `bad_frames_not_excluded` (now empty). The earlier 80-frame block (red_box 8/60 = 13.3 % FAIL, before the repair) is kept next to it; `build_dataset.merge_label_review` carries `fresh_after_repair` over when `--review-done` recomputes the old block (tested).
**Rebuild** (`python -m training.build_dataset --review-done`): stamp `622c277b5d0e5e06` (was 66f3106e49c14995). Images: train 593 (was 594), valid 173, test 187; annotations 2,965 / 865 / 935 (five per image). Train: 85 corrected, 85 verified, 31 overlay-excluded, 85 recovered. `x012_570.jpg` is not in `data/dataset/train`.
**verify_dataset** (`python -m training.verify_dataset [--expect-train N]`, 18 tests): every JSON parses; no NUL byte in any JSON or CSV (dataset folder, report, overlays, `runs/manifest.csv`, `data/label_review.csv`); per split the images on disk are exactly the annotated images; every image opens and has the annotated size; exactly one box per class per image; category names equal `experiment.classes` in order, ids 0..4; the report's classes and per-split counts agree; the overlays the report names exist with the recorded hash; the stamp recomputed from the annotation and overlay files equals the report's. Exit 1 with a list on any problem. Run after the rebuild (before it, it correctly reported the old stamp and the changed overlay hash), again just before packing (inside `kaggle_pack`, and by hand) and by the kernel on the unzipped data: `OK: train 593, valid 173, test 187; stamp 622c277b5d0e5e06 equals the report; no NUL bytes; every image opens`.
**Kaggle package (prepared, built from commit bf49a0d, under `data/kaggle_upload/`, git-ignored; `python -m training.kaggle_pack`).** CLI facts read from the installed Kaggle CLI 2.2.4 (help and its own source, offline): `datasets create` is private unless `-u`; dataset metadata needs `title` (6-50 chars), `id` (slug 6-50), exactly one `licenses` entry; `other` is one of its license names (all, cc, gpl, odb, other); `kernels push` takes `-p`, `-t`, `--accelerator`; the kernel metadata template has id, title, code_file, language, kernel_type, is_private, enable_gpu, enable_tpu, enable_internet, machine_shape, dataset_sources, competition_sources, kernel_sources, model_sources (only these are used). The accelerator names `--accelerator` accepts are not in the offline help, so the commands use `enable_gpu` and the web UI for a T4. Files (bytes, sha256):
- `dataset/dataset.zip` 97,266,189, 9a909e4a86085721018af9498815fb2a848e1b11150809ae5e16e1d88a22f5a6 (train/valid/test only, 956 members = 953 JPEG + 3 annotation files, stored, fixed timestamps; packed twice, identical bytes); `dataset/dataset_manifest.json` 88,777 (zip sha256 + sha256 of every file); `dataset/dataset-metadata.json` (id hemachandhara/sih26174-dataset, license other, private by default).
- `code/code.zip` 287,724, f24f9436faf63ed385013db32f31d8f4288f65cc58e8e88b312fddb1fb2ef5e7 (`git archive HEAD` of training/, config/, contracts.py, pyproject.toml, uv.lock, reports/dataset.json, data/corrections/train.json; perception/, state/ and engine/ are NOT included: finetune, gpu_preflight, build_dataset, verify_dataset and `load_classes` do not import them, and `autolabel` imports `perception.hands` only inside a function the training path never calls); `code/rf-detr-nano.pth` 366,287,238, d8d6b9ee57d4d0ed2b1f305163624712a0532cb7bce0c747317984fc5457440d (Apache-2.0); `code/yolo11n.pt` 5,613,764, 0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1 (AGPL-3.0, fallback only; log the licence decision if it is ever chosen); `code/code_manifest.json`; `code/dataset-metadata.json` (id hemachandhara/sih26174-code).
- `kernel/run_training.py` 32,583, 5d9271349a5628f15de6c82a80f4b37f9c69e2a6019f374293a11a0cfa3b597a (script kernel; `MODE = "SMOKE"` at the top; the sha256s above and the stamp baked into a generated block); `kernel/kernel-metadata.json` (id hemachandhara/sih26174-train, script, private, GPU on, internet on, dataset_sources both datasets). Plus `package_manifest.json` and `COMMANDS.md` (the Lead's commands; `python -m training.kaggle_pack --print-commands` prints them, `--set-mode FULL` flips the mode locally).
**Pins** (generated from `uv.lock` by `pins_from_lock`, tested on the real lock): `rfdetr[train]==1.11.0`, `supervision==0.30.5`, `ultralytics==8.4.164`. The lock names no `pycocotools`, `torchmetrics` or `pytorch-lightning` (the known gap: the `train` group lacks rfdetr's `[train]` extras), so those, and rfdetr's other train extras (peft, hotcoco, faster-coco-eval, ultrafast-pycocotools, vernier, scipy, simplejpeg, torch-hungarian, roboflow), are resolved by pip within rfdetr 1.11.0's own bounds at run time and recorded in the kernel's pip freeze; they are not locked. torch, torchvision and torchaudio are never installed or upgraded: pip runs with a constraints file freezing the preinstalled versions; a resolver conflict prints and stops with a message (exit 3), and a changed torch stack after the install also stops. numpy, opencv, pillow, pydantic and pyyaml are not pinned (the Kaggle image's own).
**The script** (tested pure parts, 25 tests; the real run is Kaggle's): finds the datasets by sentinel file (any mount layout), verifies the zip sha256 against the baked value and the manifest, unzips (or, if Kaggle unpacked the zip, checks every file against the manifest), runs `verify_dataset` on the unzipped data, installs the pins, then SMOKE: `gpu_preflight --require-cuda` for both models (n_train 593 and epochs from the S-E rule), `finetune --dry-run` for both, `smoke_report.json` (GPU, torch build, seconds per iteration, estimated minutes per epoch and total per model; flags a smoke part over 10 min); FULL: `finetune` rfdetr then yolo11n, seed 0, no `--epochs` or `--lr` (S-E rule), `--resume` when a checkpoint exists or an attached earlier output has one, logs, `train_summary.json`, best weights, `SHA256SUMS.txt`, `run_status.json`; a time guard (9 h session assumed, 25 min safety, no model started with under 30 min left) ends the run cleanly and says what finished. The child processes get `HF_HUB_OFFLINE=0` because `training.autolabel` sets it to 1 when unset and the kernel has internet. Checked locally against the real package with Kaggle's paths redirected to a temp folder: dataset zip and code zip sha256 verified, both unzipped, weights sha256 verified, `verify_dataset` OK on the unzipped data, pins read from the extracted lock.
**NOT done.** No network action of any kind: no Kaggle command was run (only `--version` and `--help` of the installed CLI, and reading its source), no upload, push to Kaggle, download, pip install, model run, training or evaluation; the `git push` of this branch is the only network use. The kernel script has not run anywhere (no GPU here): its pip step, the real `gpu_preflight` and `finetune` output on Kaggle, and Kaggle's handling of a `.zip` in a dataset are unverified. No gold subset, no val/test overlay.
**Open.** (1) Kaggle account state: is `hemachandhara` logged in with a verified phone (GPU needs it) and with GPU quota; the CLI here has not been tried against the account. (2) Which Python and torch the Kaggle image carries today and whether a P100 or a T4 is assigned (README_GPU B3); the smoke run answers it. (3) P2 lockfile change: `train` group to `rfdetr[train]` and a CUDA torch for the teammate's RTX 4070 Ti (Windows PyPI torch is a CPU build); until then the extras are unlocked. (4) Gold subset and val/test overlay for the detector gate. (5) RF-DETR may keep many checkpoints in the output folder; the script reports the output size but does not prune (20 GB Kaggle limit, unverified). (6) The Lead's backup at `D:\Z_PS2_2026\backup` was not found by this session (the repo is `D:\Z_PS2_2026\BAS`); the timestamped `.bak-` copies in `data/corrections/` and git commit 43aa603 hold the overlay now.

## 2026-10-03 P1.5 prep (S-F1c) - Kaggle kernel: dataset-root discovery, one GPU

**Smoke run 1 (kernel `hemachandhara/sih26174-train` v1) findings.** The environment printed well: Python 3.13.15, torch 2.11.0+cu128 (CUDA 12.8), TWO Tesla T4 (14.56 GB each, capability 7.5), pycocotools 2.0.11, pytorch-lightning 2.6.6, torchmetrics 1.9.0, peft and transformers 5.16.1 already installed. Then it stopped: "dataset.zip is not present; Kaggle unpacked it: copying the tree", then "STOP: dataset files differ from the packed manifest: ['missing: test/_annotations.coco.json', ...] (956)". 956 = 593 + 173 + 187 images + 3 annotation files: every file was "missing", so the script took the wrong directory (it assumed the data sits beside `dataset_manifest.json`) or the data is under an extra level or another mount path. Which one it was is NOT known: the only log is the script's stdout and nothing listed `/kaggle/input`. The new kernel prints that listing first, so run 2 settles it.
**Fixed (`training/kaggle/run_training.py`, 3 test files, 930 tests green).** (1) Discovery: `find_roots` walks at most 5 levels under `/kaggle/input`, probes with the first manifest path, then verifies every path; exactly one directory holding all of them is used (only the manifest's files are copied to `/kaggle/working/data/dataset`); two are "ambiguous" and stop with both listed; none stops with the directories examined, `found of total` per candidate and the first 10 missing paths. With no tree, `dataset.zip` is found anywhere, its sha256 checked against the baked and the manifest value, unzipped and searched the same way (so an extra top folder inside the zip is handled). Same for the code dataset (50 files) and the two weights files (found by name, sha256 verified after discovery, also compared with the sha baked into the script). Tested layouts: root, one extra folder, two extra folders, `datasets/<owner>/<slug>/`, zip at the root (also with a top folder), two roots (ambiguous), nothing found, only part of the files, deeper than 5 levels. Also run end to end on the real package: the packed script's `prepare()` against temp mounts built from the real zips (tree under `datasets/hemachandhara/sih26174-dataset/dataset/`, and zip plus unpacked code) verified all 956 + 50 files and both weights; a mount with only the manifests stopped with the examination. (2) Diagnostics at the very start of both modes: a depth-limited listing of `/kaggle/input` (4 levels, 200 lines, folders first, sizes for files, count and total size for folders over 10 files) and the free disk of `/kaggle/working`. (3) One GPU: `child_env()` sets `CUDA_VISIBLE_DEVICES=0` for every subprocess (verify, pip, `gpu_preflight`, `finetune`); the banner prints the GPUs seen and "the second is deliberately unused; effective batch stays 4 x 4 = 16"; `gpus_seen` and `gpu_used` are written to `smoke_report.json`, `run_status.json` and, by the kernel after each model (finetune itself is untouched), into `train_summary.json`. (4) Pins: the constraints file now freezes torch, torchvision, torchaudio, numpy, opencv (the three variants), pillow, scipy, pydantic, transformers, peft, pycocotools, pytorch-lightning and torchmetrics at what the image has; the optional lock pins (pycocotools, torchmetrics, pytorch-lightning) are dropped when the image already has the package (a different locked version could only conflict with the frozen one; a note says so); a pip failure, or any frozen package changing, stops with the package names pip blames. Pins kept: `rfdetr[train]==1.11.0`, `supervision==0.30.5`, `ultralytics==8.4.164`. (5) Python 3.13: a test reads the kernel script, `finetune`, `gpu_preflight`, `verify_dataset` and `coco_to_yolo` and finds no import of a module removed in 3.13 (cgi, imghdr, pipes, distutils, imp, ...); `tomllib` and `os.walk` are fine. The third-party stack under 3.13 is NOT verified here. (6) `training/kaggle_pack.py`: `--kernel-only` regenerates only `data/kaggle_upload/kernel/` (script in SMOKE mode, `kernel-metadata.json`) from the existing manifests, refuses if a zip or weights file no longer matches its manifest, refreshes `package_manifest.json` and `COMMANDS.md`, and compares a fresh `git archive HEAD` with `code.zip`. `COMMANDS.md` now uses slash-free paths.
**Re-pack result.** New kernel `run_training.py` sha256 `3dbfe4f154168fe7f1ba133abd038d10d3c0bfefb767edc7f6612855e859f132` (47922 bytes, MODE SMOKE); `kernel-metadata.json` sha256 `83f976eac6cfab64195e98e08ddc4181940082672e939491811fd50a64c18672`. `code.zip` and `dataset.zip` are byte-for-byte unchanged (sha256 `f24f9436...` and `9a909e4a...`), so no new dataset version: against HEAD only `training/kaggle/run_training.py` and `training/kaggle_pack.py` differ, and the kernel runs neither from the zip. Only the kernel is pushed. (If a file the kernel runs from `training/` ever changes: `uv tool run kaggle datasets version -p code -m "<message>"` from `data/kaggle_upload`.)
**Windows note for the Kaggle CLI.** A path with slashes fails on this machine. Run from `data/kaggle_upload` with a slash-free path: `cd data/kaggle_upload` then `uv tool run kaggle kernels push -p kernel` (likewise `-p dataset`, `-p code`, `-p smoke_output`).
**Expected, not verified: the pip step.** The image already has the training extras, so pip should only add rfdetr 1.11.0, supervision 0.30.5, ultralytics 8.4.164 and the small extras they need, and leave the frozen packages alone. Two ways it can stop, both with the package named: rfdetr 1.11.0 may bound `transformers` or `peft` below what the image has (the constraint then conflicts; per the run-1 log transformers is 5.16.1), or an extra without a Python 3.13 wheel ("No matching distribution found for X"). Do not loosen the freeze on Kaggle; log it.
**NOT done / still unverified.** No network action: no Kaggle command that contacts Kaggle (not even `--version` needed this session), no upload, no push to Kaggle, no pip install, no model run, no training. Unverified: where Kaggle mounts the datasets (run 2's listing answers it), the pip step, the real GPU measurement (`gpu_preflight` and `finetune --dry-run` on a T4 with `CUDA_VISIBLE_DEVICES=0`), that Kaggle gives 9 hours for a GPU session (the time guard assumes it), and the 20 GB output limit. The code dataset still holds the older template and packer inside `code.zip` (harmless: the pushed kernel is the script that runs).

## 2026-10-03 P1.5 prep (S-F1d) - Kaggle kernel: constraints, outputs, collect-all smoke

**Smoke run 2 (kernel v2) findings.** Verified on the real Kaggle image (Python 3.13.15, torch 2.11.0+cu128, two Tesla T4): dataset and code discovery works (mount `/kaggle/input/datasets/hemachandhara/<slug>/` with `dataset/` and `code/` already unpacked), the sha256 checks pass, `verify_dataset` is OK, the one-GPU policy works. It then stopped at the pip step: `rfdetr[train] 1.11.0 depends on torchmetrics<1.9.0 and >=1.8.2; extra == train. The user requested (constraint) torchmetrics==1.9.0`, because the constraints file froze 15 packages, among them the image's torchmetrics 1.9.0. Second problem: the kernel copied the dataset and the code under `/kaggle/working`, which is the kernel OUTPUT, so `kaggle kernels output` then tried to download hundreds of images.

**Offline compatibility result** (`tests/unit/training/test_kaggle_requirements.py`, 4 tests; reads `Requires-Dist` of rfdetr 1.11.0 with the `train` extra, supervision 0.30.5 and ultralytics 8.4.164 from this venv with importlib.metadata, evaluates the markers for Kaggle's environment (3.13, linux x86_64) and checks each requirement that names a package of the run-2 image against Kaggle's version; the fixture dict documents the facts; nothing downloaded). **Exactly one violation:** `rfdetr 1.11.0 needs torchmetrics<1.9.0,>=1.8.2, Kaggle has 1.9.0`. Everything else the pins name on the image is satisfied by what is already there, the core (torch, torchvision, numpy, scipy, pillow, opencv) included. Not on the image, so pip adds them: av, faster-coco-eval, hotcoco, pyDeprecate (rfdetr wants `<0.10,>=0.9`, supervision `<0.12,>=0.9`: 0.9.x satisfies both), roboflow, simplejpeg, torch-hungarian (pinned `==0.1.0rc0`, a pre-release; an exact pin is allowed by pip), ultrafast-pycocotools, ultralytics-platform, ultralytics-thop, vernier. Run 2's log named only av, pyDeprecate, ultralytics-thop and ultralytics-platform because it stopped at the first conflict; the rest of that list comes from the metadata, not from a Kaggle log.

**Fixed (`training/kaggle/run_training.py`; tests in `test_kaggle_collect.py` (26), `test_kaggle_requirements.py` (4), two older tests adjusted; 960 tests green).**
- **Constraints:** only the core is frozen: torch, torchvision, torchaudio, numpy, scipy, pillow, opencv-python, opencv-python-headless (and opencv-contrib-python if present). torchmetrics, pytorch-lightning, peft, transformers, pycocotools, pydantic are no longer frozen: pip's default only-if-needed strategy leaves them alone unless a pin needs a change (expected: torchmetrics moves to 1.8.x). `drop_installed_optional` is unchanged (the lock names none of them).
- **Pip step:** `pip install --dry-run` first when the image's pip has it (read from `pip install --help`); the `Would install ...` plan is parsed and printed as added/changed lines; a plan that touches a core package stops the run (exit 3) before anything is installed. After the real install the core versions are asserted unchanged and every package pip added or changed is printed; `outputs/pip_dry_run.log`, `pip_install.log`, `constraints.txt` and `pip_freeze.txt` are kept. A conflict in either pip call stops with the packages pip names.
- **Outputs:** datasets, code and training scratch are staged under `/kaggle/temp` if it exists with at least 4 GB free, otherwise `/tmp/work` (the choice and the reason are printed and put in `smoke_report.json`; neither with room is a hard stop naming both). `/kaggle/working` keeps only `outputs/`: logs, `smoke_report.json`, pip freeze, the dry runs' `train_summary.json` (their weights stay in temp), sha256 files. At the end the total size of `/kaggle/working` is printed (the biggest entries too when large); over 2 GB prints OUTPUT TOO LARGE and the run exits 1 (the stages and, in FULL, the training are not touched by it). The sha256 checks and `verify_dataset` are kept.
- **Collect-all smoke:** dataset verification and pip stay hard stops. Then five stages, each in its own guard (an exception is a failed stage, never an escape) with a 5-minute timeout: `preflight_rfdetr`, `preflight_yolo11n` (`gpu_preflight --require-cuda`, prints the machine, seconds per iteration and the epoch estimate), `dry_run_rfdetr`, `dry_run_yolo11n` (`finetune --dry-run` on the synthetic dataset), and `train_probe_rfdetr` (only if `dry_run_rfdetr` passed, otherwise recorded as skipped): three real RF-DETR-Nano training iterations on the real dataset at batch 4 in a child process that wraps pytorch_lightning's Trainer (`limit_train_batches=3`, no validation) and prints `PROBE iteration n: s`, then seconds per iteration (mean of iterations 2 and 3; data loading included) and the GPU memory peak allocated and reserved. Each stage records status (ok, failed, skipped), seconds, a message and, on failure, the last 30 lines of its output. A stage table is printed and written to `smoke_report.json` (all earlier fields kept, plus `stages`, `verdict`, `pip_changes`, `pip_plan`, `temp_root`, `outputs_size_text`); the last line is `SMOKE OK` or `SMOKE PARTIAL: <failing or skipped stage names>`.

**Re-pack (kernel only; `data/kaggle_upload` is git-ignored).** `kernel/run_training.py` 67,063 bytes, sha256 `8085057588ee3524dd7843bd2661b705211f89279d4b5dd78154dbb7b6a05de0` (MODE SMOKE); `kernel/kernel-metadata.json` unchanged (`83f976eac6cfab64195e98e08ddc4181940082672e939491811fd50a64c18672`); `package_manifest.json` refreshed. **code.zip is unchanged** (`f24f9436...`): against HEAD only `training/kaggle/run_training.py` and `training/kaggle_pack.py` differ, both outside what the kernel runs, so no new code dataset version is needed. Push, from `data/kaggle_upload`: `uv tool run kaggle kernels push -p kernel`.

**NOT done / still unverified.** No network action (no Kaggle command, no upload, no push to Kaggle, no pip install, no model run, no training). Unverified: (1) the pip install itself on Kaggle (the dry-run plan format, torchmetrics actually moving to 1.8.x, torch-hungarian's pre-release pin and the other new packages building on Python 3.13); (2) the RF-DETR train path on Python 3.13, including the new probe, which was only compiled and unit-tested, never run: it patches `pytorch_lightning.Trainer.__init__`, assumes rfdetr builds its Trainer from that class and that `train()` tolerates a run with no validation (any exception after the 3 batches is tolerated and noted); if it fails it only costs the one stage; (3) real GPU speed (the probe and the preflights are what will tell); (4) the 9-hour limit; (5) in FULL mode checkpoints are still written under `/kaggle/working/outputs/<model>` (resumable), so the 2 GB check can fail a FULL run on size alone: decide what to keep before the FULL push.

## 2026-10-03 P1.5 (S-F2a) - valid evaluation, CPU benchmark

**Scope.** Evidence only. No detector chosen, test split never evaluated or read, `weights/MANIFEST.json` not written. Branch p1-perception. Commits: eval_detector extensions a7026fb, valid reports 3d9c07a, benchmark_valid a6854d0, benchmark report 04b292f. Reports: `reports/detector_eval_valid_rfdetr.json`, `reports/detector_eval_valid_yolo11n.json`, `reports/benchmark_cpu_valid.json`. `reports/dataset.json` shows as modified in the working tree (the rebuild's own output, stamp 1bea0958699991df); it is not part of these commits.

**Weights** (copied to git-ignored `weights/`, sha256 equal to each `train_summary.json`): `detector_rfdetr_nano.pth` 82d9f126cb1d36d08713e6a490629aeeba37fd447db9cebf7381f5e64de804bb; `detector_yolo11n.pt` cb9cd840618525058823e4242e8ced9539118dc2f134456785667289e1ab1613. Both were trained on dataset stamp 622c277b5d0e5e06 (train 593 images, 173 valid images); the evaluation ran against the current dataset stamp 1bea0958699991df (valid 211 frames, 88 gold). The stamps differ because the overlays changed after the Lead's gold pass; whether the train images are identical in both stamps was not re-checked here. Floor `detector_conf_floor` = 0.3, IoU 0.5, CPU, valid only.

**Valid, all 211 frames** (precision / recall at the floor per class; each class has 211 ground-truth boxes):

| model | mAP50 | mAP50-95 | outer_box | tray | red_box | yellow_box | start_button |
|---|---|---|---|---|---|---|---|
| RF-DETR-Nano | 0.9982 | 0.9778 | 1.000/1.000 | 1.000/1.000 | 0.9953/0.9953 | 0.9765/0.9858 | 0.9906/1.000 |
| YOLO11n | 0.9984 | 0.9672 | 1.000/1.000 | 1.000/1.000 | 0.9765/0.9858 | 0.9673/0.9810 | 1.000/1.000 |

**Valid, 88 gold frames** (88 ground-truth boxes per class):

| model | mAP50 | mAP50-95 | outer_box | tray | red_box | yellow_box | start_button |
|---|---|---|---|---|---|---|---|
| RF-DETR-Nano | 0.9969 | 0.9683 | 1.000/1.000 | 1.000/1.000 | 1.000/1.000 | 0.9773/0.9773 | 0.9888/1.000 |
| YOLO11n | 0.9963 | 0.9570 | 1.000/1.000 | 1.000/1.000 | 0.9773/0.9773 | 0.9551/0.9659 | 1.000/1.000 |

**Rule (1) eligibility, gold, recall >= 0.85 and mAP50 >= 0.80:** RF-DETR-Nano: outer_box PASS 1.0000, tray PASS 1.0000, red_box PASS 1.0000, yellow_box PASS 0.9773, start_button PASS 1.0000, mAP50 PASS 0.9969 -> ELIGIBLE. YOLO11n: outer_box PASS 1.0000, tray PASS 1.0000, red_box PASS 0.9773, yellow_box PASS 0.9659, start_button PASS 1.0000, mAP50 PASS 0.9963 -> ELIGIBLE. Both are eligible, with a large margin.

**Gold recall by hand over a container** (build_dataset's definition: a hand box intersects a ground-truth red or yellow box; hand cache `data/labels/hands`): 62 gold frames with the hand over a container, 26 with the hand elsewhere, 0 without hand data. With hand over container: RF-DETR red 1.000, yellow 0.9677 (60/62); YOLO11n red 0.9677 (60/62), yellow 0.9516 (59/62); other classes 1.000 for both. Hand elsewhere (26): every class 1.000 for both. All misses of both models are in the hand-over-container group.

**red_box / yellow_box confusion** (each ground-truth box takes the best-IoU >= 0.5 prediction at the floor). All 211 frames: RF-DETR red: 210 red, 0 yellow, 1 missed; yellow: 208 yellow, 0 red, 3 missed. YOLO11n red: 208 red, 0 yellow, 3 missed; yellow: 207 yellow, 0 red, 4 missed. Gold: RF-DETR red 88 red, 0 missed, yellow 86 yellow, 2 missed; YOLO11n red 86, 2 missed, yellow 85, 3 missed. **No red/yellow swap in either model on either subset;** every error is a miss.

**CPU benchmark** (`training/benchmark_valid.py`; valid frames 848x480 BGR, seeded draw, 20 warm-up + 150 timed, per-call perf_counter; model input resolution 384; detector floor 0.3; hands = `perception.hands.HandTracker`, VIDEO mode, frames fed in shuffled order, so tracking gets no temporal help; overhead 5 ms assumed). Machine: AMD Ryzen 5 5600H, 6 cores / 12 logical, Windows Balanced power plan, **on AC power**, torch 2.14.0+cpu, 8 threads as recorded in the run (an interactive probe elsewhere printed 6). Run 1 order: rfdetr, rfdetr-optimized, yolo, hands; run 2 reversed. Median / p95 / mean in ms:

| candidate | run 1 | run 2 |
|---|---|---|
| RF-DETR-Nano (predict) | 163.1 / 178.3 / 165.5 | 139.1 / 153.4 / 140.4 |
| RF-DETR-Nano + optimize_for_inference (extra row) | 152.7 / 168.9 / 154.4 | 126.6 / 138.4 / 127.5 |
| YOLO11n | 20.0 / 23.0 / 20.3 | 19.0 / 20.3 / 18.8 |
| MediaPipe hands | 27.0 / 36.7 / 28.3 | 26.5 / 36.7 / 28.4 |

Pipeline view, median detector + median hands + 5 ms (budget 125 ms = 8 fps). Detector on every frame (k=1), every 2nd, every 3rd: average frame time and fps; the slow frame (detector + hands + overhead) is the same for every k:

| detector | run | k=1 | k=2 | k=3 | slow frame |
|---|---|---|---|---|---|
| RF-DETR-Nano | 1 | 195.1 ms, 5.1 fps | 113.6 ms, 8.8 fps | 86.4 ms, 11.6 fps | 195.1 |
| RF-DETR-Nano | 2 | 170.5 ms, 5.9 fps | 101.0 ms, 9.9 fps | 77.8 ms, 12.9 fps | 170.5 |
| RF-DETR-Nano optimized | 1 | 184.8 ms, 5.4 fps | 108.4 ms, 9.2 fps | 83.0 ms, 12.1 fps | 184.8 |
| RF-DETR-Nano optimized | 2 | 158.0 ms, 6.3 fps | 94.7 ms, 10.6 fps | 73.6 ms, 13.6 fps | 158.0 |
| YOLO11n | 1 | 52.1 ms, 19.2 fps | 42.1 ms, 23.8 fps | 38.7 ms, 25.8 fps | 52.1 |
| YOLO11n | 2 | 50.5 ms, 19.8 fps | 41.0 ms, 24.4 fps | 37.8 ms, 26.5 fps | 50.5 |

**Spread.** RF-DETR varied by about 15 percent between the runs (163 vs 139 ms; 153 vs 127 optimized); YOLO11n and hands by about 5 percent or less. The run with RF-DETR last was faster, so a position or thermal effect is likely, but run 2's reversed order confounds it with the candidate order and this benchmark cannot separate the two. Read against rule (2) as pre-registered (k=1): RF-DETR-Nano does NOT meet 125 ms in either run (170-195 ms), also not with optimize_for_inference (158-185 ms); YOLO11n meets it with a wide margin (50-52 ms). RF-DETR meets the budget only with the detector on every 2nd frame (average 101-114 ms), with up to 195 ms on the slow frames. Earlier benchmark (125-155 ms RF-DETR alone) is consistent with run 2. This session does not apply the rule.

**ONNX (optional item).** Works offline in this venv. `RFDETR.from_checkpoint(...).export(format="onnx", fp16=False)` to the scratchpad (not in the repo, nothing committed), ONNX Runtime CPU, 8 intra-op threads, same 150 frames: median 145.0 ms, p95 151.1, mean 145.1 (a single run, so only roughly comparable to the 139-163 ms PyTorch figures: no clear gain). Agreement with PyTorch on 20 valid images at the floor: 100 boxes each, same box count on all 20 frames, minimum IoU 0.99996, mean 0.999998, class agreement 100 percent, max confidence difference 6.8e-05. The exported model is numerically the same; ONNX does not change the speed picture.

**Not done.** No detector chosen; test split not evaluated or read; no MANIFEST; nothing downloaded or installed; no ONNX file committed; the benchmark is one laptop state (Balanced plan, AC), two runs; the pipeline fps is a model built from the three medians, not a measured end-to-end run.

**Caveats.** (1) Scores are against labels that partly come from the auto-labeler (agreement with the corrected labels, not independent truth). (2) Gold is 88 frames of one performer and one setup, so it says little about other people, lighting or arrangements. (3) Valid also drove early stopping and checkpoint selection of both trainings, so valid numbers are optimistic. (4) Hand data come from the same MediaPipe model the pipeline would use, so the hand-over-container split is a proxy. (5) Scores are near ceiling (every class recall >= 0.966, mAP50 >= 0.996), so rule (1) cannot separate the models; the difference between them is speed (and licence), not accuracy. (6) YOLO11n is AGPL-3.0; if rule (4) applies, the licence decision must be logged.

## 2026-10-03 P1.5 DECISION - detector choice: YOLO11n (default), RF-DETR-Nano alternative

**Written and committed BEFORE any test-split evaluation (S-F2b).** The test split has not been evaluated or read for any detector at the time of this entry.

**Pre-registered rule (written before any result, S-F2a).** (1) A detector is eligible if every class has recall >= 0.85 and mAP50 >= 0.80 on the valid gold subset at `detector_conf_floor`. (2) Among eligible detectors, prefer one meeting `min_pipeline_fps` 8 (median detector + hands + 5 ms <= 125 ms per frame on the demo laptop). (3) Both eligible and fast: RF-DETR-Nano. (4) Only YOLO11n fast enough: YOLO11n, with the AGPL-3.0 decision logged. (5) Neither fast enough: present options. (6) Test is evaluated ONCE, for the chosen detector only, afterwards.

**Eligibility, valid gold (88 frames, floor 0.3, IoU 0.5), recall per class and mAP50** (from the S-F2a entry): RF-DETR-Nano outer 1.000, tray 1.000, red 1.000, yellow 0.977, button 1.000, mAP50 0.9969, mAP50-95 0.9683. YOLO11n outer 1.000, tray 1.000, red 0.977, yellow 0.966, button 1.000, mAP50 0.9963, mAP50-95 0.9570. Both ELIGIBLE under rule (1), with a large margin.

**Speed, per frame with hands, detector on every frame, median detector + median hands + 5 ms** (budget 125 ms; two runs, this laptop: Ryzen 5 5600H, AC power, Balanced plan): RF-DETR-Nano 195.1 and 170.5 ms (5.1 and 5.9 fps); with optimize_for_inference 184.8 and 158.0 ms (5.4 and 6.3 fps); ONNX Runtime of RF-DETR 145 ms alone (no gain, outputs agree). YOLO11n 52.1 and 50.5 ms (19.2 and 19.8 fps). This laptop is the demo laptop.

**Application of rule (4).** RF-DETR-Nano does not meet 125 ms in either run (also not optimized or in ONNX); YOLO11n meets it with a wide margin. Both are eligible and only YOLO11n is fast enough, so rule (4) applies: **YOLO11n**, with the licence decision logged below. The Lead accepted YOLO11n as the DEFAULT detector. RF-DETR-Nano stays as the documented alternative.

**Accuracy cost of the choice (valid gold, 88 frames).** mAP50-95 0.957 (YOLO11n) versus 0.968 (RF-DETR-Nano); red_box recall 0.977 versus 1.000 (yellow 0.966 versus 0.977). All misses of both models are in frames with a hand over a container (YOLO11n: red 60/62, yellow 59/62 there; every class 1.000 with the hand elsewhere). No red/yellow swap in either model; every error is a miss. Scores are near ceiling, so the accuracy difference is small and the choice is driven by speed.

**Licence facts (factual, NOT legal advice).** Ultralytics YOLO11 code and weights are AGPL-3.0. Ultralytics also sells an Enterprise licence for commercial, internal and production use. An Ultralytics forum reply states that using AGPL software without an Enterprise licence requires open-sourcing the whole solution publicly, and that private or proprietary deployments need the Enterprise licence; trained models are covered too. This is the vendor's stated position, stricter than a minimal reading of the AGPL text; it is recorded as stated and not verified with a lawyer. Implication for hand-over: delivering the system (code plus weights) to ISRO/the ministry may count as conveying a combined work that has to be offered under AGPL-3.0, or may require an Enterprise licence. The SIH2024 guidelines say solution IP resides with the students and the ministry gets lifetime free access; the SIH 2026 text and the ISRO software policy were not found. RF-DETR-Nano is Apache-2.0 and raises none of this.

**Mitigations.**
- (a) A backend-only, launch-time detector toggle (manifest `active_detector` plus a command-line or environment override). No control in the frontend; never changed during a run; the chosen detector name and sha256 are recorded in every log header. To be built at the pipeline stage, NOT in this session.
- (b) `ultralytics` lives in its own optional dependency group and is imported only when YOLO11n is selected, so a licence-clean build can leave it out.
- (c) RF-DETR-Nano (Apache-2.0) is trained, evaluated on valid and kept as the documented alternative.
- (d) The choice is stated in the README, the slides and the manifest.
- (e) The Lead asks the mentor or the ISRO contact whether AGPL-3.0 is acceptable for hand-over and checks the SIH 2026 IP clause.

**Scope decision.** For the deadline, tuning and validation are completed for YOLO11n. RF-DETR-Nano's own cache, tuning profile and replay are done only when time permits, recorded as a follow-up. Until then RF-DETR-Nano is recorded as not validated for the pipeline and not evaluated on test.

**Caveats.** One performer, one setup, no gloves. Labels partly come from the auto-labeler (agreement with corrected labels, not independent truth). Valid drove early stopping and checkpoint selection, so valid numbers are optimistic. Gold is 88 frames (valid) and 80 frames (test). The benchmark is one laptop state (Balanced, AC), two runs, and the pipeline fps is a model from medians, not a measured end-to-end run.

**target_fps proposal: 10**, to be verified at the pipeline stage. Basis: about 50 ms per frame for detector and hands (YOLO11n 19-20 ms, hands 27 ms, plus 5 ms overhead) leaves about 50 ms of the 100 ms frame budget for tracking, streaming and speech. `min_pipeline_fps` stays 8.

**Follow-ups.** Adding `ultralytics` (and the detector backends) to the `run` dependency groups is a P2 lockfile change, not done here. The detector toggle code is not built here. Both are for the pipeline stage.

## 2026-10-03 P1.5 (S-F2b) - test evaluation of YOLO11n and MANIFEST

**Scope.** Record the choice (DECISION entry, commit f0e2983, pushed before any test evaluation), evaluate the CHOSEN detector once on test, write `weights/MANIFEST.json`. Branch p1-perception.

**Acceptance file.** `git diff c89bee8 -- config/acceptance.yaml` is empty (unchanged). sha256 `477b2be4a016c6bf3dd0c5e834dec391a155131fce1c851d777c0259eff30db5`, also recorded in the report.

**Test evaluation, YOLO11n, once** (`reports/detector_eval_test_yolo11n.json`; weights sha256 cb9cd840...1613; dataset stamp 1bea0958699991df; floor 0.3, IoU 0.5; CPU). The first attempt (`--device auto`) aborted on the first frame with "Invalid CUDA device=0" (the auto setting selects CUDA for YOLO, this laptop has none); it produced no prediction and no report, so nothing was seen. The same command with `--device cpu` (as in S-F2a) ran to completion and is the one evaluation. Follow-up: `eval_detector --device auto` should fall back to CPU when CUDA is unavailable (not changed here).

All 213 frames (precision / recall at the floor; mAP50 0.9971, mAP50-95 0.9615):

| class | precision | recall | n_gt | n_pred |
|---|---|---|---|---|
| outer_box | 1.000 | 1.000 | 213 | 213 |
| tray | 1.000 | 1.000 | 213 | 213 |
| red_box | 0.9724 | 0.9906 | 213 | 217 |
| yellow_box | 0.9628 | 0.9718 | 213 | 215 |
| start_button | 1.000 | 1.000 | 213 | 213 |

Gold, 80 frames (mAP50 0.9988, mAP50-95 0.9540):

| class | precision | recall | n_gt | n_pred |
|---|---|---|---|---|
| outer_box | 1.000 | 1.000 | 80 | 80 |
| tray | 1.000 | 1.000 | 80 | 80 |
| red_box | 0.9639 | 1.000 | 80 | 83 |
| yellow_box | 0.9620 | 0.950 | 80 | 79 |
| start_button | 1.000 | 1.000 | 80 | 80 |

Gold by hand: with the hand over a container 56 frames (mAP50 0.9979, mAP50-95 0.9389; red recall 1.000, yellow 0.929, others 1.000); hand elsewhere 24 frames (mAP50 1.000, mAP50-95 0.9868; every class precision and recall 1.000); 0 without hand data. All gold misses are in the hand-over-container group (4 yellow misses, all there).

Red/yellow confusion (best-IoU >= 0.5 at the floor). All frames: red 211 red, 1 yellow, 1 missed; yellow 207 yellow, 0 red, 6 missed. Gold: red 80 red, 0 yellow, 0 missed; yellow 76 yellow, 0 red, 4 missed. One red box in the all-frames set was taken for yellow (not in gold); there is no yellow-as-red case.

**Acceptance verdict (gold test vs `config/acceptance.yaml`):** recall >= 0.85 per class: outer_box PASS 1.000, tray PASS 1.000, red_box PASS 1.000, yellow_box PASS 0.950, start_button PASS 1.000; mAP50 >= 0.80: PASS 0.9988. **Detector acceptance: PASS.** (The pipeline, replay and label-review criteria are not part of this session.)

**MANIFEST** (`weights/MANIFEST.json`, committed; the weights stay git-ignored): `active_detector` yolo11n; `detectors` lists yolo11n (AGPL-3.0, test_evaluated true, validated_for_pipeline true, valid and test reports) and rfdetr_nano (Apache-2.0, validated_for_pipeline false, test_evaluated false, note "tune and replay when time permits", valid report only), each with file, sha256, size, classes in order, input size 384, floor 0.3, trained stamp 622c277b5d0e5e06, evaluated stamp 1bea0958699991df, training summary sha256 and the DECISION title; the existing `hand` entry (hand_landmarker.task, sha256 fbc2a300...cde1, Apache-2.0) is kept. `trained_at` is null: the training summaries carry no timestamp. The summaries live in the git-ignored `data/kaggle_upload/`, so the manifest records their hashes, not their paths in git. `tests/unit/training/test_manifest.py` (7 tests): hashes and sizes equal the files (skips when the weights are absent), exactly one active detector, the active one is the only one evaluated on test and its test report carries the same weights sha256, classes equal the experiment in order.

**Caveats.** One performer, one setup, no gloves. Labels partly from the auto-labeler. Valid drove early stopping, so valid numbers are optimistic; test was untouched until this run. Gold test is 80 frames: one yellow miss changes yellow recall by 1.25 points, and the lowest value (0.950) is 10 points above the threshold. Hand data come from the same MediaPipe model, so the hand split is a proxy. Near-ceiling scores mean this acceptance gate is easy to pass and says little about other people or lighting.

**NOT done.** RF-DETR-Nano was not evaluated on test; no retraining; no threshold change; no lockfile change; no toggle code; no runtime integration; no network, download or install. No other test run beyond the single completed evaluation. `reports/dataset.json` was already committed (cdc732f) before this session, so the tree was clean.

## 2026-10-03 P2 CONTRACT - compose_model_stamp gets an optional detector_name

**What changes.** `compose_model_stamp` in contracts.py gains a LAST parameter `detector_name: str = "rfdetr-nano"`, used as the first label of the stamp: `"<detector_name>:<sha256[:8]>|hand:<sha256[:8]>|pose:<sha256[:8]>"`. `detector_name` must match `^[a-z0-9]+(-[a-z0-9]+)*$`; an empty string, uppercase, `:`, `|`, spaces or a trailing newline raise `ValueError`, so a stamp always has exactly three `|`-separated parts with exactly one `:` each. Function and docstring only. Tests: `tests/unit/contracts/test_model_stamp.py`. Doc lines: IMPLEMENTATION_PLAN.md 5.2 (the stamp format) and the `weights/MANIFEST.json` line.

**Why.** The detector choice (D130/D131, see "2026-10-03 P1.5 DECISION - detector choice: YOLO11n (default), RF-DETR-Nano alternative") makes YOLO11n the default, with a backend-only launch-time toggle to RF-DETR-Nano. With the label hard-coded, a YOLO11n pipeline would stamp its perception caches and reports as `rfdetr-nano`. Caches are valid only while their `model_stamp` equals the pipeline's, so the label must name the detector actually loaded.

**Additive.** A new optional parameter with a default. `compose_model_stamp(d, h, p)` returns exactly what it returned before (tested against the original format). The one existing mention outside contracts.py is a docstring in `perception/hands.py`; no code calls the function yet, so no caller changes.

**What does NOT change.** Every other type, validator and default in contracts.py; the `hand:` and `pose:` labels; the 8-character sha256 prefix; `config/experiment.json`; the cache and manifest file shapes.

**Convention: pose disabled.** When pose is disabled (`enable_pose=False`, the default) callers pass `pose_sha256="none"`, so the stamp reads `pose:none`.

**Who must follow.** P1 (writer) composes the stamp from `weights/MANIFEST.json`, passing the manifest's detector name mapped to the stamp label: `yolo11n` -> `"yolo11n"`, `rfdetr_nano` -> `"rfdetr-nano"` (underscore in the manifest, hyphen in the stamp; never pass the manifest name through). P2 only compares stamp strings and never composes or parses them.

## 2026-10-03 P1 NOTE - Kaggle FULL run (facts)

Kernel version 4; one of the two T4 GPUs was used. **RF-DETR-Nano:** 100 epochs requested, early stop after about 35 epochs, best EMA epoch 24, effective mAP 0.98262 (RF-DETR's own metric), 2,787 s. **YOLO11n:** 100 epochs, 447 s, mAP50 0.995 and mAP50-95 0.974 on the then-valid 173 frames. Total 3,323 s. The kernel's output-size check (2.34 GB against a 2 GB self-imposed limit) tripped because of the per-epoch RF-DETR checkpoints. The weights were downloaded manually as `.zip` files that are the PyTorch archives themselves, and were verified by sha256 against the manifest. The two training summaries are now tracked as `reports/training/rfdetr_train_summary.json` and `reports/training/yolo11n_train_summary.json` (sha256 equal to the manifest's `training_summary_sha256`); `trained_at` is `2026-10-03`, the date of the Kaggle run log, because the summaries carry no timestamp.

## 2026-10-03 P1 CORRECTION - when the detector selection rule was written down

The selection rule was agreed in the project decision log and in the S-F2a agent prompt before its results existed. Git, however, shows the rule text only from commit 2271be6, which is after the valid results. The DECISION commit f0e2983 still precedes the test report commit a119ca2. So the order of record in git is: valid results and rule text (2271be6), decision (f0e2983), test report (a119ca2); the earlier agreement is not visible in git.

## 2026-10-03 P1 ACCEPTED CHANGE - manifest shape differs from the original Plan 5.8 wording

`weights/MANIFEST.json` carries `detectors[]` plus `active_detector` (per detector: `name`, `file`, `sha256`, `size_bytes`, `license`, `classes`, `input_size`, `detector_conf_floor`, `dataset_stamp_trained`, `dataset_stamp_evaluated`, `trained_at`, `validated_for_pipeline`, `test_evaluated`, training-summary path and sha256), not the single `detector` object and stored `model_stamp` that Plan 5.8 described. Plan 5.8 and the essential-features.md F2 line ("not a runtime switch" became "chosen at launch, never switched during a run") were updated in H0.4 of session S-G with the Lead's approval; the stamp is composed at runtime by `contracts.compose_model_stamp`.

## 2026-10-03 P1 NOTE (for P2) - perception/pipeline.py has landed; harness/replay.py --video must switch to load_pipeline

`perception/pipeline.py` now exists (S-G, P1.6.3). Two P2 tests in `tests/unit/harness/test_replay.py` asserted that importing `perception.pipeline` fails "until perception lands"; with the Lead's agreement, P1 changed only those two tests to stub the missing import (`monkeypatch.setitem(sys.modules, "perception.pipeline", None)`), nothing else in `harness/` or its tests. **Still P2's to do:** `harness/replay.py` `replay_from_video` calls `PerceptionPipeline(perception_config)`, which is not the real signature. The real constructor is `PerceptionPipeline(detector, hands, pose=None)`; the launch-time factory is `perception.pipeline.load_pipeline(manifest_path=..., detector=None, enable_pose=False)` (detector chosen by argument, then `SIH_DETECTOR`, then the manifest's `active_detector`). `replay --video` with a real video has therefore not been run against the real pipeline.

## 2026-10-03 P1.7 DECISION - target_fps = 10

**Measurement** (`python -m training.benchmark_pipeline`, `reports/benchmark_pipeline.json`; the real `PerceptionPipeline` with YOLO11n at imgsz 384 on CPU plus MediaPipe hands, frames decoded and processed one after another, 200 frames per clip after 10 warm-up frames, validation clips x015, x016, x017, Ryzen 5 5600H demo laptop; `model_stamp` yolo11n:cb9cd840|hand:fbc2a300|pose:none). Per frame, decode included: x015 median 47.3 ms / p95 51.0 ms; x016 48.1 / 60.7; x017 47.9 / 52.4. **All 600 frames: median 47.7 ms, p95 56.3 ms, mean 48.4 ms, which implies 20.9 fps at the median and 17.8 fps at the p95.** It excludes tracking, the engine, the JPEG stream and speech.

**Decision: `target_fps` = 10** (the proposal). It leaves 100 ms per frame against 47.7 ms used at the median (2.1x) and 56.3 ms at the p95 (1.8x); that headroom is for the state tracker and engine (cheap pure-Python), MJPEG encoding and Flask on other threads, and the speech worker process, all on the same 6-core CPU. A 30 fps recording is decimated exactly by 3 (`every = round(29.99 / 10) = 3`). 10 is sustainable; the pipeline view also clears `min_pipeline_fps` 8 (20.9 measured, detector + hands only). Not chosen: 15 (the `RuntimeConfig` default) would give 66.7 ms per frame, only 1.4x the median and 1.2x the p95 before streaming and speech, too thin for a live demo; RF-DETR-Nano (171 to 195 ms per frame for the detector alone, measured earlier) could not reach 10 fps and would run at about 4 to 5. `hysteresis_frames` is tuned at the real rate by P2.6 and `RuntimeConfig.target_fps` is P2's to set; the caches in this session are built at 10.

## 2026-10-03 P1.6/P1.7 (S-G) - pipeline, detector factory, caches

**Built (branch p1-perception).** H0: `eval_detector --device auto` falls back to CPU when CUDA is absent (YOLO and RF-DETR paths, tested with a mocked `torch.cuda.is_available`); `--split test` is refused while any `reports/detector_eval_test_*.json` exists unless `--allow-repeat-test "<reason>"` (recorded in the new report); Kaggle owner is `--owner` or `KAGGLE_OWNER`, no tracked default, personal path scrubbed from `README_GPU.md`; two doc lines reworded (essential-features F2, Plan 5.8); the two training summaries tracked under `reports/training/` with sha256 equal to the manifest, `trained_at` 2026-10-03 with its note; three ISSUES entries. P1.6.1: `perception/detector.py` (`Detector` protocol; `YoloDetector` imgsz 384, floor `DETECTOR_MIN_CONF` 0.10; `RfdetrDetector`, BGR to RGB at the boundary; `load_detector(manifest_path, name)` precedence argument, then `SIH_DETECTOR`, then `active_detector`; refuses on an unknown name, missing weights, weights sha256 differing from the manifest, or manifest classes differing from `config/experiment.json` in order; WARNING when `validated_for_pipeline` is false; `import perception.detector` imports neither ultralytics nor rfdetr, asserted in a clean subprocess). P1.6.2: `training/eval_detector.py` now imports `yolo_predict`, `rfdetr_predict` and `resolve_device` from `perception.detector`, so the evaluation and the runtime share one prediction code path. P1.6.3: `perception/pipeline.py` (`PerceptionPipeline(detector, hands, pose=None)` and the launch factory `load_pipeline`); a frame-level failure is logged with its frame id and re-raised, which is what `runtime/loop.py` catches (warning, skip frame). P1.6.4: `perception/cache.py` (atomic writer, validating reader that refuses a different `model_stamp` or `fps`, resumable builder, `python -m perception.cache build`). `training/benchmark_pipeline.py` and `training/cache_report.py` (timing and the report live in training/ because perception reads no clock, rule 8).

**Equivalence (P1.6.2).** On 20 valid images (every 10th of 211, YOLO11n, CPU) the new `YoloDetector` and the evaluation's original inline path (conf 0.001, imgsz 384, then filtered at 0.10) returned the same boxes, classes and confidences within 1e-4: 102 detections compared, same count on every image (`test_detector_equivalence.py`, marked slow, skips when weights or data are absent). The RF-DETR path is the same code as before, moved unchanged.

**Throughput (P1.7.1)** and the `target_fps` = 10 decision are in "2026-10-03 P1.7 DECISION - target_fps = 10": 600 frames over 3 valid clips, median 47.7 ms, p95 56.3 ms per frame (decode, YOLO11n, hands, sequential), 20.9 fps at the median, 17.8 at the p95.

**Caches (P1.7.2)**, `data/cache/<run_id>/perception.jsonl` (git-ignored), model_stamp `yolo11n:cb9cd840|hand:fbc2a300|pose:none`, fps 10 (every 3rd frame of the ~30 fps recordings), all 46 runs built, none failed, no frame skipped, 416.7 s wall in one background run (`reports/cache_build.json`):

| split | runs | frames | seconds |
|---|---|---|---|
| train | 26 | 4,928 | 244.7 |
| val | 10 | 1,691 | 84.5 |
| test | 10 | 1,763 | 87.4 |
| total | 46 | 8,382 | 416.7 |

Per-class detection statistics, descriptive only (no labels read on test, nothing tuned): frames with a detection at or above 0.30 (share of frames) and quantiles of the best such confidence per frame (min / q25 / median / q75 / max):

| class | val frames | val conf | test frames | test conf |
|---|---|---|---|---|
| outer_box | 1691 of 1691 (1.000) | .944 / .964 / .968 / .972 / .980 | 1763 of 1763 (1.000) | .877 / .959 / .965 / .969 / .982 |
| tray | 1691 (1.000) | .907 / .944 / .950 / .955 / .980 | 1763 (1.000) | .865 / .941 / .947 / .952 / .976 |
| red_box | 1687 (0.998) | .404 / .981 / .996 / .997 / 1.000 | 1760 (0.998) | .453 / .978 / .995 / .997 / 1.000 |
| yellow_box | 1675 (0.991) | .307 / .953 / .981 / .985 / .995 | 1750 (0.993) | .306 / .939 / .976 / .984 / .994 |
| start_button | 1691 (1.000) | .708 / .972 / .981 / .987 / .998 | 1763 (1.000) | .671 / .978 / .983 / .988 / .996 |

Detections at or above 0.30 exceed frames-with-detection for red and yellow (val 1,709 red and 1,709 yellow boxes in 1,687 and 1,675 frames; test 1,789 and 1,790): some frames carry a second, lower-confidence box of the same class above the floor. The tracker takes the best one per label (Plan 5.3), but P2.6 should know.

**Determinism and refusal (P1.7.3).** x015 rebuilt into a temporary folder is byte-identical to the cache (asserted in `test_cache_real.py`), and x030 rebuilt through the CLI into `data/cache_verify` has the same sha256 as `data/cache/x030` (14fce45e...); a second `build --run x001` skipped it as complete. The reader refuses the YOLO cache when the expected stamp is the RF-DETR one (`CacheMismatch`: "cache model_stamp 'yolo11n:...' != expected 'rfdetr-nano:82d9f126|...'"), and the reverse.

**Optional (P1.7.4), done.** `SIH_DETECTOR=rfdetr_nano python -m perception.cache build --run x015|x016 --out-root data/cache_rfdetr` built 188 and 220 frames, stamp `rfdetr-nano:82d9f126|hand:fbc2a300|pose:none`, with the `validated_for_pipeline=false` WARNING logged; about 34 s and 40 s per run (YOLO11n about 9 s). On those two val runs every class had a detection at or above 0.30 in all 408 frames (yellow 408 vs 406 for YOLO11n), so the RF-DETR class-id handling produces valid labels end to end. This proves the toggle only; it is not an evaluation. The RF-DETR caches are not committed (data/ is git-ignored).

**Quality gates.** `check.py --quick`: 1,080 passed (was 1,010); ruff clean; `check.py --status`: F1 11, F2 60, F3 34, F4 40, F5 31, F6 10, F7 22, F8 3, F9 16, F10 32, F11 7, F12 23, F13 25, F14 214 tests, all GREEN, 0 failed.

**Surprises.** (1) Adding `perception/pipeline.py` turned two P2 tests red (they asserted the import fails); with the Lead's agreement P1 stubbed the import in only those two tests (see the P2 NOTE above); `harness/replay.py --video` still calls the old constructor and is P2's to fix. (2) `PerceptionConfig`, `target_fps` and the harness replay were not touched. (3) The repo holds 46 runs; Plan G3 says "all 77 runs".

**G3 checklist.** MET and checked this session: detector chosen and stamped (YOLO11n active, stamp label in `compose_model_stamp`, `weights/MANIFEST.json` hashes verified at load); real `PerceptionPipeline` (built, 12 + 19 + 23 unit tests, and run on real video); caches for every run (46 of 46, train, val and test, at 10 fps); F1, F2, F3, F14 rows green (and F4 to F13). NOT re-verified here, taken from the earlier ISSUES entries (P1.1 to P1.5, PR #15) rather than checked in this session: all runs validated, static boxes reviewed for every run, dataset reviewed within the bad-label gate, gold subsets of val and test hand-verified (the test report used 80 gold frames). NOT met or open: the Plan's "77 runs" versus 46 recorded; replay of the caches through the tracker and engine (P2.6) and the harness `--video` path.

**NOT done.** RF-DETR full caches, tuning and replay; the dependency regrouping (ultralytics stays an optional group); the frontend toggle (none, by design); P2.6 threshold tuning; the harness `replay --video` switch to `load_pipeline`; pose (off, no pose model in the manifest). No network, download or install; test labels were never read.


## 2026-10-04 P2 R7 - dependency group `yolo` (S-H1, block D0)

**Change.** `pyproject.toml` gains the optional group `yolo = ["ultralytics==8.4.164"]` (the exact version already locked). `tools` now takes ultralytics through `{ include-group = "yolo" }` instead of its own `>=8.3,<9` line. The `run` group never required ultralytics (rfdetr's own dependencies do not include it, checked in `uv.lock`), so a licence-clean build leaves `yolo` out. `uv lock` (no `--upgrade`): 275 packages before and after, **no version changed, no package added or removed**; the lock diff is only the new `yolo` group entry and the `tools` specifier text. No package was downloaded and no `uv sync` was needed.

**Install commands (there is no README, so they are here).**
- Demo laptop: `uv sync --group run --group yolo` (YOLO11n is the default detector; AGPL-3.0, see the P1.5 DECISION). Never the `train` group.
- Licence-clean build: `uv sync --group run` (no ultralytics; the detector must then be RF-DETR-Nano, `SIH_DETECTOR=rfdetr_nano`, which is not yet validated for the pipeline).
- Developer laptop: `uv sync --group dev --group tools` (tools includes `yolo`).
- GPU training PC / Kaggle: install `rfdetr[train]` with pip in an isolated environment (see below); the lockfile's `train` group does not carry it.

**PARKED: `rfdetr[train]` in the lock.** An attempt to add `rfdetr[train]==1.11.0` to the `train` group was resolved and then reverted, changing nothing. It would have (a) changed the existing locked `typer` 0.27.2 -> 0.25.1 and (b) added `opencv-python-headless` 4.11.0.86 beside the locked `opencv-python`, which can shadow or break `cv2` on the demo laptop, plus 25 more packages (accelerate, aiohttp, faster-coco-eval, hotcoco, peft, pillow-avif-plugin, pyarrow, pycocotools, pytorch-lightning, roboflow, torch-hungarian, torchmetrics 1.8.2, ultrafast-pycocotools, vernier and small ones). RF-DETR training is only needed when time permits and the Kaggle route installs with pip. **To revisit only when RF-DETR is retrained:** use an isolated virtual environment, check the licences of roboflow, torch-hungarian, hotcoco, vernier, faster-coco-eval, ultrafast-pycocotools and pillow-avif-plugin, and check that `import cv2` and MediaPipe still work there. Nothing was checked for those licences in this session.

**Dry run.** `python -m training.finetune --model rfdetr --dry-run --output-dir <dir>` (the flag `--output-dir` is required) runs to the end with exit code 0 and a PARTIAL note: pytorch_lightning, torchmetrics and pycocotools are missing, so `train()` was NOT called, only the 2 synthetic forward+backward iterations. It reaches the `train()` call only with the extras installed; that is unchanged. It downloaded nothing.

## 2026-10-04 P2 NOTE - Kaggle FULL-run time (clarification)

The Kaggle FULL-run total of 3,323 s in "2026-10-03 P1 NOTE - Kaggle FULL run (facts)" is the kernel's wall time and includes about 89 s of setup; the two training times (RF-DETR-Nano 2,787 s, YOLO11n 447 s) sum to 3,234 s.
