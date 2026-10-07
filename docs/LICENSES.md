# Licence notes (facts, not legal advice)

Tags: **[repo]** read from a file, **[issues]** stated in `ISSUES.md`, **[unknown]** not checked.
Every licence decision is **DECISION - Lead**. Nothing here was checked with a lawyer.

| item | licence as recorded | source |
|---|---|---|
| YOLO11n fine-tuned weights (`detector_yolo11n.pt`) and the `ultralytics` package | **AGPL-3.0** (Ultralytics also sells an Enterprise licence) | `weights/MANIFEST.json` [repo]; `ISSUES.md` 2026-10-03 P1.5 DECISION [issues] |
| RF-DETR-Nano fine-tuned weights (`detector_rfdetr_nano.pth`) | Apache-2.0; trained and evaluated on validation only, **not validated for the pipeline**, not evaluated on test | `weights/MANIFEST.json` [repo] |
| MediaPipe hand landmarker (`hand_landmarker.task`) | Apache-2.0 | `weights/MANIFEST.json` [repo] |
| Grounding DINO tiny (auto-labeller, dev time only) | Apache-2.0 (Hugging Face model card, as recorded) | `ISSUES.md` 2026-09-30 P1.2 R7 [issues] |
| Python dependencies of the `run` group | each package's own licence; the full table is `IMPLEMENTATION_PLAN.md` section 3.1; not re-checked for this kit | [repo] |
| `rfdetr[train]` extras (roboflow, torch-hungarian, hotcoco, vernier, ...) | **not checked** | `ISSUES.md` 2026-10-04 P2 R7 [issues] |
| the project's own code | the repository has no `LICENSE` file | [repo] |
| the recordings, labels and COCO dataset | not decided; the Kaggle placeholder is `other` | `docs/DATASET_CARD.md` |

**AGPL-3.0 and the YOLO weights.** The vendor's stated position, recorded in `ISSUES.md`, is that using AGPL
software without an Enterprise licence requires open-sourcing the whole solution publicly, and that this covers
trained models. This is the vendor's reading, not verified with a lawyer; it may be stricter than the AGPL text.
The demo kit therefore says so in `KIT_README.txt`, and a licence-clean build (`uv sync --group run`, no
`ultralytics`) is possible with the Apache-2.0 RF-DETR-Nano, which is slower than the demo needs (about 5 fps,
`ISSUES.md` 2026-10-03 P1.5 DECISION) and not validated for the pipeline. The question to the mentor or the ISRO contact
(mitigation (e) of that entry) is still open.

**Consent.** The repository records no consent of the person or people in the recordings: **DECISION - Lead.**
