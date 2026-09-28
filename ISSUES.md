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
- Status: OPEN

## 2026-09-28 setup CONTRACT - a typo in a YAML config key is silently ignored
- Missing: `extra="forbid"` on `PerceptionConfig` and `RuntimeConfig`. Probe result: `hysteresis_frmes: 7` loads without error and the default (5) is used.
- Why it matters: the whole tuning step (P2.6) writes `config/perception.yaml`; a misspelled key would quietly revert a tuned threshold to its default and nobody would notice.
- Proposed fix at G0: `ConfigDict(frozen=True, extra="forbid")` on both models.
- Status: OPEN

## 2026-09-28 setup CONTRACT - `LogEntry.t_wall` is required but is stamped by the logger
- Missing: a way to construct a `LogEntry` before the logger stamps `t_wall`.
- Why it matters: contracts.py says wall time exists only in the logger, yet the field is mandatory at construction. Current workaround (documented in Plan 5.5): the router builds entries with `datetime(1970,1,1,tzinfo=utc)` and the logger overwrites it.
- Proposed fix at G0: `t_wall: datetime | None = None`, keeping the timezone-aware check when set; the logger always sets it.
- Status: OPEN

## 2026-09-28 setup CONTRACT - the small file seams are not typed
- Missing: Pydantic models for `config/acceptance.yaml` and the `reports/*.json` shapes (currently specified only as a table in Plan 5.8).
- Why it matters: they are written by one person and read by another (P1 writes acceptance thresholds; P2 reads them; reports feed the PPT).
- Proposed fix at G0: `AcceptanceConfig` (frozen, `extra="forbid"`) and a small `ReportHeader` (`generated_at`, `model_stamp`, input stamps). Report bodies may stay dicts.
- Status: OPEN

## 2026-09-28 setup DECISION - operating system of the two dev laptops and the demo laptop
- Decision needed: Windows / macOS / Linux for each machine.
- Why it matters: TTS backend (SAPI5 / NSSpeechSynthesizer / eSpeak), webcam backend (`CAP_DSHOW` on Windows), `multiprocessing` start method, and whether shell scripts would even run. The plan already avoids bare shell scripts, but the demo machine's voice and camera must be tested on that exact machine.
- Proposed: record the answer at G0; run the audio self-test and camera check on every machine.
- Status: OPEN

## 2026-09-28 setup DECISION - where the detector is fine-tuned (GPU access)
- Decision needed: who has a GPU (Colab or Kaggle notebook, a lab machine) and who runs stage 6 of the dataset pipeline.
- Why it matters: "CPU laptop only" is a deployment constraint; fine-tuning RF-DETR is a GPU job (its docs are written for T4/A100-class GPUs). No GPU means the fallback is a slow CPU fine-tune of YOLO11n, which is AGPL.
- Proposed: settle at G0; keep the fine-tune as a background script, not an open agent session.
- Status: OPEN
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
- Status: OPEN

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
