# Dataset card: BAS "Sample Transfer" (46 recorded runs)

Source tags used below: **[repo]** read from a file in this repository, **[run]** observed by running a
command during the preparation of this kit, **[issues]** stated in `ISSUES.md`, **[user]** stated by
the project team, **[unknown]** nobody has recorded it. Anything not checked is marked **NOT VERIFIED**.
This card is not legal advice.

## 1. What it is

Video of one small, scripted experiment, **Sample Transfer**: a crew member takes a red and a yellow
sample box out of an outer box, puts them into a tray in a fixed order, presses a START button, then
stows both boxes. It is a stand-in for an on-orbit procedure, not a replica; the real procedures are
not public (`DATA_COLLECTION.md`, section 1) [repo]. The data feed an advisory step-checking assistant
(ISRO PS 26174, SIH 2026) that watches one fixed overhead camera, speaks the next step and alerts on a
skipped, out-of-order or repeated step.

Seven steps, in order: `red_out`, `red_in_tray`, `yellow_out`, `yellow_in_tray`, `start_pressed`,
`red_stowed`, `yellow_stowed` (`runs/x016/script.json`) [repo]. Five detector classes: `outer_box`,
`tray`, `red_box`, `yellow_box`, `start_button` (`reports/dataset.json`) [repo].

## 2. How it was recorded

* Everyday props (shoe box, two small boxes, a tray, a green START pad), a fixed camera, taped home
  positions, scripted runs: `DATA_COLLECTION.md` [repo].
* **Camera, operator and set-up are not recorded in the data:** `operator` and `camera_setup_id` are
  `unknown` in all 46 rows of `runs/manifest.csv` [repo]. `runs/run_plan.csv` plans 77 runs with four
  operators (O1 to O4) and two set-ups (S1, S2) and its `recorded` column is empty [repo]; the
  project's own caveat is "one performer, one setup, no gloves" [issues, 2026-10-03 P1.5 DECISION].
  **DECISION - Lead:** state here who performed and with which camera.
* 46 runs were recorded and kept, not the 77 planned [issues, 2026-10-03 P1.6/P1.7 "Surprises"].
* Footage is 848 x 480, 29.92 to 30.01 fps, H.264 in `.mp4`, 4.0 to 25.9 s per clip, 837.2 s and
  25,100 frames in all [run: computed from `runs/manifest.csv`]. The files are compressed (the
  originals' encoder settings are **[unknown]**).

## 3. Splits

The split is by run, never by frame (AGENTS.md rule 12) [repo]. Run counts per split and script type
[run: computed from `runs/manifest.csv`]:

| split | runs | correct | skip | swap | repeat | idle |
|---|---|---|---|---|---|---|
| train | 26 | 14 | 5 | 5 | 1 | 1 |
| val | 10 | 4 | 2 | 2 | 1 | 1 |
| test | 10 | 4 | 2 | 2 | 1 | 1 |

`runs/<id>/script.json` holds the script type, the performed steps and the deviations a correct
system must report (`expected_deviations`). **The test split is held out:** nothing was tuned on it
and the detector and the replay were evaluated on it once (`ISSUES.md`, "S-F2b" and "S-H2") [issues].
Please keep it that way.

## 4. The COCO detection dataset (`data/dataset/`)

Frames sampled from the runs, one COCO file per split (`_annotations.coco.json`), five boxes per kept
frame [repo: `reports/dataset.json`, dataset stamp `1bea0958699991df`].

| split | frames sampled | images kept | annotations |
|---|---|---|---|
| train | 624 | 593 | 2,965 |
| valid | 220 | 211 | 1,055 |
| test | 225 | 213 | 1,065 |

`python -m training.verify_dataset` reads the folder and says OK when it equals the report
[run: "OK: train 593, valid 211, test 213; stamp 1bea0958699991df equals the report"].

## 5. Labels and how they were checked

1. **Auto-labelled** with Grounding DINO tiny (zero-shot, Apache-2.0) with rule-based post-processing
   [issues, 2026-10-02 P1.2 DECISION and R7 entry; `training/autolabel.py`].
2. **Hand-corrected** frames, written to `data/corrections/{train,val,test}.json`: 85 train, 88 val and
   80 test frames corrected and verified (`frames_corrected` / `frames_verified` in
   `reports/dataset.json`) [repo]. The val (88) and test (80) corrected frames are the **gold subsets**
   used for the detector's acceptance numbers [issues, 2026-10-03 P1.5].
3. **Reviewed:** a stratified sample of 60 frames failed the 10 % bad-label gate for `red_box`
   (8 of 60 bad, 13.3 %) before repair; after repair a fresh sample of 60 frames (seed 20261003)
   passed (1 bad cell, `yellow_box` wrong class on `x012_570.jpg`, 1.7 %); a static review of 46 runs
   (138 rows) found 0 bad rows [repo: `reports/dataset.json`, `label_review`]. The review sheets are
   `data/review/` (not in the Kaggle bundle).
4. Frames the labeler or the overlay check excluded are not in the dataset: 115 / 47 / 38 labeler
   exclusions and 31 / 9 / 12 overlay exclusions for train / valid / test [repo: `reports/dataset.json`].

**What this does not give you:** the non-gold frames are auto-labelled, so agreement with them is
agreement with the labeler, not independent truth [issues]. `data/label_review.csv` (the reviewer's
verdict per box) is in the bundle.

## 6. Known errors and limits

* Per-run labels (`runs/provenance.csv`): 24 runs marked `bad` (a deviation) were watched by the Lead;
  the **22 `good` runs are `crew_label_only`: "assumed from crew label, not yet watched"** [repo].
  Notes such as `mixed` (x025, x027, x030, x034, x038, x040, x041, x043) mark runs that mix deviation
  types [repo]; x037's note says the yellow box was lifted but not out of the outer box, so not a step.
* One performer (as stated by the team, see section 2), one set-up, one lighting condition, no gloves,
  compressed 848 x 480 footage; scores on this data are near ceiling and say little about other people,
  cameras or lighting [issues, 2026-10-03 P1.5 caveats].
* Misses of the detector concentrate on frames with a hand over a container (yellow and red box)
  [issues]. START-press recognition is the weak point of the whole system: short taps, passes over the
  card and merged double taps are recognised only partly [issues, 2026-10-03 P2.7].
* Some clips end before the experiment does: the cached replay of x024 (val, skip, 9.8 s) and of x036
  (idle) ends without a `run_completed` event [run: `python scripts/replay.py --from-cache x024`].

## 7. Licence, consent and visibility

* **Licence of the footage and labels** (the recordings, `script.json`, the COCO files, the corrections and the
  label review): **CC BY 4.0** [user, decision of 2026-10-07]. Credit is required:
  **Credit: Hemachandhar A and the SIH 2026 PS 26174 team.**
  The Kaggle metadata field is `other` (see `docs/KIT.md`: the installed CLI documents no CC BY name offline), so
  the real licence is stated here and in the dataset description.
* **Consent of the people in the footage:** [user] consent confirmed by the Lead. No individual is named and the
  consent records are not in the repository. Whether faces or other identifying features are visible in the
  footage has not been checked [unknown].
* **Public visibility:** this dataset and the code repository (<https://github.com/Hemachandhar-A/BAS>) are
  public [user, decision of 2026-10-07].
* **The model weights are separate and keep their own licences:** YOLO11n fine-tune **AGPL-3.0**; RF-DETR-Nano
  fine-tune Apache-2.0; MediaPipe hand landmarker Apache-2.0 (`weights/MANIFEST.json`) [repo]. The project code is
  MIT. All of it is in `docs/LICENSES.md`. Not legal advice.

## 8. How to cite

> Hemachandhar A and the SIH 2026 PS 26174 team (2026). *BAS "Sample Transfer" dataset: 46 recorded runs with
> labels.* Kaggle dataset `hemachandhara/bas-dataset-full`,
> <https://www.kaggle.com/datasets/hemachandhara/bas-dataset-full>. Licence CC BY 4.0.
> Code: <https://github.com/Hemachandhar-A/BAS> (give the commit hash you used).

No DOI has been minted [unknown].

## 9. Integrity

`SHA256SUMS.txt` in the bundle lists every file (`<sha256>  <path>`, same format as `sha256sum -c`).
`runs/manifest.csv` carries the `video_sha256` of each clip; `python scripts/verify_assets.py --kit full`
checks every run video against it. `reports/dataset.json` carries the dataset stamp.
