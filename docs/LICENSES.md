# Licences

Tags: **[repo]** read from a file in this repository, **[issues]** stated in `ISSUES.md`, **[user]** stated by the
project lead, **[unknown]** not checked. Status **confirmed** means the claim was read in the named file;
**to confirm** means it rests on a decision or a statement nobody has checked against the source.
**This is not legal advice.** Nothing here was reviewed by a lawyer.

Copyright line of the project: "Copyright (c) 2026 Hemachandhar A and the SIH 2026 PS 26174 team" (`LICENSE`).

| component | licence | where it applies | source of the claim | status |
|---|---|---|---|---|
| Project code (everything in this repository that is not listed below) | **MIT** | the whole git repository; text in `LICENSE` | [user] decision of 2026-10-07; `LICENSE` [repo] | confirmed (a project decision; the copyright name is for the lead to confirm) |
| Footage and labels (the 46 recorded runs, `script.json` files, COCO dataset, corrections, label review) | **CC BY 4.0**; credit line: "Credit: Hemachandhar A and the SIH 2026 PS 26174 team" | both Kaggle datasets and `runs/`, `data/` as published; **not** the model files | [user] decision of 2026-10-07 | confirmed as a project decision; consent of the people in the footage: "[user] consent confirmed by the Lead" |
| YOLO11n fine-tuned weights (`detector_yolo11n.pt`) | **AGPL-3.0** (Ultralytics; the vendor also sells an Enterprise licence) | `weights/`, the demo-kit dataset; cannot be made permissive | `weights/MANIFEST.json` field `license` [repo]; `ISSUES.md` 2026-10-03 P1.5 DECISION [issues] | confirmed |
| RF-DETR-Nano fine-tuned weights (`detector_rfdetr_nano.pth`) | **Apache-2.0**; trained and evaluated on validation only, **not validated for the pipeline**, not evaluated on test | `weights/`; in the full dataset only if built with `--with-weights` | `weights/MANIFEST.json` (`license`, `validated_for_pipeline: false`, `test_evaluated: false`) [repo] | confirmed (licence of the weights as recorded; the licence of every package behind `rfdetr[train]` is to confirm) |
| MediaPipe hand landmarker (`hand_landmarker.task`) | **Apache-2.0** | `weights/`, the demo-kit dataset | `weights/MANIFEST.json` field `license` [repo]; `ISSUES.md` 2026-09-30 P1 R7 [issues] | confirmed |
| Grounding DINO tiny (`IDEA-Research/grounding-dino-tiny`), the auto-labeler | **Apache-2.0** | dev time only; produced the first-pass labels; its weights are not shipped | `ISSUES.md` 2026-10-02 P1.2 R7 (Hugging Face model card, as recorded) [issues] | confirmed as recorded; the model card was not re-read for this change (**to confirm**) |
| Third-party Python dependencies | each package keeps its own licence | see `uv.lock`; full table in `IMPLEMENTATION_PLAN.md` section 3.1 | [repo] | to confirm (not re-checked; no long list is generated here) |
| `ultralytics` (optional dependency group `yolo`) | **AGPL-3.0** | installed only with `uv sync --group run --group yolo` | `pyproject.toml` comment on the `yolo` group [repo]; `ISSUES.md` P1.5 DECISION [issues] | confirmed |

A licence-clean build (no AGPL package) is `uv sync --group run`: it leaves out the `yolo` group, so `ultralytics` is
not installed. The detector must then be RF-DETR-Nano (see below).

## What the AGPL means for a YOLO11n user (plain words, not legal advice)

The YOLO11n detector and its package `ultralytics` are under the GNU Affero General Public Licence v3 (AGPL-3.0).
In neutral terms: the AGPL asks that if you run modified AGPL software and let other people use it over a network,
you offer those users the corresponding source code; distributing the software to others carries its own source
obligations. The project source is public at <https://github.com/Hemachandhar-A/BAS>. The detector is chosen once
at launch and is never switched during a run (`ISSUES.md`, 2026-10-03 P1.5 DECISION, mitigation (a)), so a user can
pick which licence applies to their deployment. RF-DETR-Nano (Apache-2.0) is the licence-clean alternative; it is
slower than the demo needs (about 5 frames per second, same entry) and is not validated for the pipeline. The vendor's
own stated position, recorded in `ISSUES.md`, is stricter than a minimal reading of the AGPL text and was not
checked with a lawyer. This page makes no claim about what any particular organisation may do with the weights;
whoever deploys them should read the licence themselves or ask a professional.

## Consent

Consent of the people in the footage: **[user] consent confirmed by the Lead.** No individual is named and the
consent records themselves are not in the repository.
