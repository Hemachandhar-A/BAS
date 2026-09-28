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
