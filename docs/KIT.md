# Judge replication kit: maintainer notes

For the Lead. The judges' instructions are in `README.md`; this file says how the two Kaggle bundles
were built, how to rebuild them and which commands publish them. Nothing here was uploaded: no network
call was made while preparing the kit, and the Kaggle CLI is not installed on the preparation machine,
so **every Kaggle command below is NOT VERIFIED**. Check each with `--help` first.

Tags: **[repo]** read from a file, **[run]** observed while preparing the kit, **[unknown]** not checked.

## 1. What the two bundles are

| bundle | folder (git-ignored) | purpose | size | Kaggle slug |
|---|---|---|---|---|
| A: demo kit | `kit_staging/bas-demo-kit/` | what a judge needs to RUN the demo | 30.4 MB, 9 files [run] | `<kaggle-user>/bas-demo-kit` |
| B: full dataset | `kit_staging/bas-dataset-full/` | what a researcher needs to REPRODUCE | 298.4 MB, 1,124 files [run], no weights | `<kaggle-user>/bas-dataset-full` |

With `--with-weights` bundle B also carries `weights/` (both detectors and the hand model, plus
`MANIFEST.json`): about 134 MB more (5.4 + 7.8 + 121.0 MB from `weights/MANIFEST.json`) [repo].

**A** is flat on purpose (the Kaggle CLI skips sub-folders by default, `--dir-mode skip` [unknown]):
`detector_yolo11n.pt`, `hand_landmarker.task`, `01_x016_clean.mp4`, `02_x032_skip.mp4`,
`03_x025_swap.mp4`, `04_x036_idle.mp4`, `playlist.txt`, `SHA256SUMS.txt`, `KIT_README.txt`,
`dataset-metadata.json`. The judge copies the files to `weights/` and `demo_videos/`.

**B** keeps the repository layout: `runs/<id>/video.mp4` and `script.json` (46 runs), `runs/manifest.csv`,
`runs/provenance.csv`, `runs/run_plan.csv`, `data/dataset/` (COCO), `data/corrections/{train,val,test}.json`,
`data/label_review.csv`, `reports/dataset.json`, `reports/training/*`, `DATASET_CARD.md`,
`SHA256SUMS.txt`, `dataset-metadata.json`. **Left out on purpose:** the perception caches `data/cache/`
(they contain the test split's perception output and are rebuilt in a few minutes), `data/frames/`,
`data/review/` sheets, `data/kaggle_upload/`, backup files in `data/corrections/`, `runs_out/`.

The demo clips are the **original** files: staged by copying `runs/<id>/video.mp4` under a new name,
after checking each against `video_sha256` in `runs/manifest.csv`. Nothing was re-encoded, trimmed or
stitched, and `D:\Z_PS2_2026\BAS\demo_videos\` (the Lead's own folder) was only read.

## 2. The demo order and why

| # | clip (run) | split | type | live result observed [run] |
|---|---|---|---|---|
| 1 | `01_x016_clean.mp4` (x016) | val | correct | 7 steps confirmed, no alert, `Experiment complete` |
| 2 | `02_x032_skip.mp4` (x032) | val | skip (START) | 5 confirmed, 1 omission "Step skipped: Start pressed" |
| 3 | `03_x025_swap.mp4` (x025) | val | swap | 1 confirmed, omission, 2 out-of-order, omission |
| 4 | `04_x036_idle.mp4` (x036) | val | idle | no step, no deviation, `Run ended early` |

Story: clean, skip, swap, idle. All are val-split originals of 16 to 22 s, so no `--allow-heldout` is
needed. The Lead's own `demo_videos/` folder holds x015, x025, x032, x038 and x022; the kit differs:
* **x015 -> x016** as the clean clip. Both are clean through the real flow (gated start, then the
  dashboard's POST: 7 steps, no alert [run]); x016 is the clip of the project's soak and of the
  `--replay-run x016` backup mode, and is a little longer (22 s).
* **x038** (repeat) is a **test-split** clip: it needs `--allow-heldout` and the Lead's yes. The
  playlist guard only looks at the path (section 6), so a renamed copy would play without the flag.
* **x022** is a *correct* run on which the START press is missed: it shows a false alarm ("Step skipped:
  Start pressed", flagged uncertain). Honest, but not a story opener; mention it in questions.
* A repeat on val: only **x040** (8.6 s, shorter than 15 s). Its cached and live results show the
  omission of four steps but not the repeat, so it does not show the deviation it is named for.
  The kit therefore has no repeat clip.

## 3. Rebuild (offline)

From the repository root with the venv interpreter:

```
.venv\Scripts\python.exe docs/kit/build_kit.py --kaggle-user <name>          # both bundles + assets/ASSETS.sha256
.venv\Scripts\python.exe docs/kit/build_kit.py --only demo                   # bundle A only
.venv\Scripts\python.exe docs/kit/build_kit.py --only full --with-weights    # bundle B with all weights
```

The script only copies, uses `contracts.sha256_of_file`, and **fails (exit 1)** if a weights file or a
clip does not match `weights/MANIFEST.json` / `runs/manifest.csv`, or if a clip is test-split. It
rewrites `assets/ASSETS.sha256` from bundle A; commit that file after a rebuild. `--kaggle-user` replaces
the `<kaggle-user>` placeholder in the dataset ids (and so in the hash list's header).

The list of clips is `DEMO_CLIPS` in `docs/kit/build_kit.py`; to change the demo order edit it and the
`PLAYLIST` text there, rebuild, and update the table in `README.md` section 5.

## 4. Kaggle commands for the Lead (NOT VERIFIED, run by hand)

The project's earlier Kaggle upload used `uv tool run kaggle ...` and slash-free paths from inside
the folder (`data/kaggle_upload/COMMANDS.md` [repo]). The same here. Your Kaggle credentials stay with you; nothing in
this kit reads or writes them.

Before the first upload:
1. Decide the **licence** and the **visibility** (section 5). Edit `licenses` in both
   `kit_staging/*/dataset-metadata.json` (currently `{"name": "other"}`, a placeholder).
2. Replace `<kaggle-user>` (rebuild with `--kaggle-user`, section 3) and re-commit `assets/ASSETS.sha256`.
3. Read `README.md` section 4 once: it has the same placeholder in the download commands.

```
cd kit_staging
uv tool run kaggle datasets create -p bas-demo-kit
uv tool run kaggle datasets create -p bas-dataset-full --dir-mode zip
```

* The metadata file sits at the root of each bundle folder: `kit_staging/bas-demo-kit/dataset-metadata.json`,
  `kit_staging/bas-dataset-full/dataset-metadata.json`.
* Datasets are created **private** unless the command is given a visibility flag [unknown: the CLI has
  `--public`; confirm with `uv tool run kaggle datasets create --help`]. There is no visibility field in the
  metadata files.
* `--dir-mode zip` uploads each sub-folder as a zip so the layout survives; whether Kaggle then
  shows them extracted or as zip files is **[unknown]**. Open the dataset page after the upload and
  check that `runs/x016/video.mp4` is reachable; if only zips appear, tell the judges to unzip them
  (README section 8) and tell the maintainer.
* A new version after a rebuild (the message is yours):

```
uv tool run kaggle datasets version -p bas-demo-kit -m "describe the change"
uv tool run kaggle datasets version -p bas-dataset-full -m "describe the change" --dir-mode zip
```

* Kaggle's limits on dataset size and file count: **[unknown]**, no network here. Confirm them on the
  Kaggle documentation page for datasets before uploading bundle B (298 MB, 1,124 files; 433 MB with
  weights). If the file count is a problem, upload `data/dataset/` as one zip and say so in the card.
* After the upload, from a clean machine: download bundle A (README section 4, both ways) and run
  `python scripts/verify_assets.py`; the last line must be `READY`.

## 5. Decisions only the Lead can make

1. **Consent** of the person or people in the footage (nothing is recorded in the repository).
2. **Licence** of the footage, labels and COCO files, and the licence note for the AGPL-3.0 YOLO11n weights
   (`ISSUES.md`, 2026-10-03 P1.5 DECISION, mitigation (e): the mentor/ISRO question is still open).
   The kit is Apache-2.0-clean if the judge uses RF-DETR-Nano, but that detector is not validated for
   the pipeline.
3. **Visibility** (private with invitations, or public) of both datasets, and who may access the repository
   (it is a private GitHub repository at the time of writing: judges need collaborator access
   [unknown: visibility not checked, the remote is `https://github.com/Hemachandhar-A/BAS.git`]).
4. Whether to include a **test-split** clip (x038, repeat) and so `--allow-heldout` in the live demo.
5. Whether bundle B carries the weights (`--with-weights`, includes the 121 MB RF-DETR-Nano).
6. The repository URL and the Kaggle user name that replace the placeholders.
7. The **performer** and **camera** facts for the dataset card (`unknown` in the manifest).

## 6. Findings made while building the kit (for ISSUES.md)

* **Held-out guard is path based.** `harness/live.py` decides "test split" from the clip's folder name
  (`runs/<id>/video.mp4`, then `script.json` / `manifest.csv`). A copy such as `04_repeat_x038.mp4` in
  `demo_videos/` is not refused without `--allow-heldout` [run: read of `resolve_source`, `_refuse_heldout`].
* **Cold first clip.** With `--no-wait-for-dashboard` the first clip of a playlist starts without the idle
  warm-up frame of the gated flow, and the clean clips x015 and x016 each lost `red_out` and `red_in_tray`
  there (a false omission alert at about 6 s; 2 runs of each as first clip). The same clips are clean
  through the real flow (gated start, then the dashboard's POST: x015 once, x016 once) and when they are
  not the first clip (x016 as clip 2 of a run without the gate). The judges' flow is the gated one.
* **A missing clip is skipped.** `demo.py` prints `warning: playlist: skipping '<clip>': file not found` and
  plays the rest, so a judge with a missing clip sees a shorter show: `python scripts/verify_assets.py`
  is what catches it.
* `demo.py --check` without `--playlist` also tries camera 0 and FAILs on a laptop without a camera: judges should pass `--playlist`.
