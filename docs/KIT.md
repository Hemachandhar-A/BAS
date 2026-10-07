# Judge replication kit: maintainer notes

For the Lead. The judges' instructions are in `README.md`; this file says how the two Kaggle bundles
were built, how to rebuild them and which commands publish them. Nothing here was uploaded: no network
call was made while preparing the kit. The Kaggle CLI (2.2.4, `uv tool`) is installed on the preparation
machine: every flag below was checked against its offline `--help` output [run], but **no Kaggle command
was run against Kaggle, so the commands' effects are NOT VERIFIED**.

Tags: **[repo]** read from a file, **[run]** observed while preparing the kit, **[unknown]** not checked.

## 1. What the two bundles are

| bundle | folder (git-ignored) | purpose | size | Kaggle slug |
|---|---|---|---|---|
| A: demo kit | `kit_staging/bas-demo-kit/` | what a judge needs to RUN the demo | 30.4 MB, 9 files + `SHA256SUMS.txt` [run] | `hemachandhara/bas-demo-kit` |
| B: full dataset | `kit_staging/bas-dataset-full/` | what a researcher needs to REPRODUCE | 298.4 MB, 1,124 files + `SHA256SUMS.txt` [run], no weights | `hemachandhara/bas-dataset-full` |

With `--with-weights` bundle B also carries `weights/` (both detectors and the hand model, plus
`MANIFEST.json`): about 134 MB more (5.4 + 7.8 + 121.0 MB from `weights/MANIFEST.json`) [repo].

**A** is flat on purpose (the Kaggle CLI skips sub-folders by default, `--dir-mode skip` [run: `kaggle datasets create --help`]):
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
.venv\Scripts\python.exe docs/kit/build_kit.py                               # both bundles + assets/ASSETS.sha256
.venv\Scripts\python.exe docs/kit/build_kit.py --only demo                   # bundle A only
.venv\Scripts\python.exe docs/kit/build_kit.py --only full --with-weights    # bundle B with all weights
```

The script only copies, uses `contracts.sha256_of_file`, and **fails (exit 1)** if a weights file or a
clip does not match `weights/MANIFEST.json` / `runs/manifest.csv`, or if a clip is test-split. It
rewrites `assets/ASSETS.sha256` from bundle A; commit that file after a rebuild. The Kaggle ids
(`hemachandhara/bas-demo-kit`, `hemachandhara/bas-dataset-full`), titles, subtitles, licence field and the
dataset descriptions are constants at the top of the script (`KAGGLE_USER`, `DEMO_DESCRIPTION`,
`FULL_DESCRIPTION`, `DATASET_META`); the demo kit's `KIT_README.txt` is `KIT_README` there. Edit them there and
rebuild: never hand-edit `kit_staging/` (it is git-ignored, so source and output would disagree).

The list of clips is `DEMO_CLIPS` in `docs/kit/build_kit.py`; to change the demo order edit it and the
`PLAYLIST` text there, rebuild, and update the table in `README.md` section 5.

To verify a staged bundle offline, lay its files out as the README says in a scratch folder (weights and clips from bundle A;
`runs/`, `data/`, `reports/` from bundle B; plus `assets/ASSETS.sha256` and `weights/MANIFEST.json`) and run
`python scripts/verify_assets.py --root <folder> --kit demo` and `--kit full`: both must end with `READY`.

## 4. Kaggle commands for the Lead (run by hand)

Your Kaggle credentials stay with you: the CLI authenticates from your existing token (outside the repository);
nothing in this kit reads, writes or prints it, and neither should you paste it anywhere. Both datasets are
**public** [user, 2026-10-07]. Run each block in PowerShell, once per bundle, from inside the bundle folder.

**What the installed CLI says offline** (Kaggle CLI 2.2.4, `uv tool run --offline kaggle datasets create --help`,
and its source) [run]:

* `datasets create` makes a **private** dataset by default. `-u` / `--public` ("Create publicly (default is
  private)") makes it public. The `isPrivate` field of `dataset-metadata.json` is **not read by `create`** (it is
  only read by `datasets metadata --update`), so the commands below pass `--public`. The metadata files still carry
  `"isPrivate": false` so that the intent is on record.
* `-r` / `--dir-mode {skip,zip,tar}`: `skip` ignores sub-folders (default), `zip` uploads each sub-folder as a zip.
* `datasets version` needs `-m` (the message); it also takes `-p`, `-r` and `-d` (delete old versions).
* `datasets status <owner>/<slug>` prints the status; `--format json` prints status and version number.
* Lengths enforced by the CLI: title 6 to 50 characters, slug 6 to 50, subtitle 20 to 80. The metadata files satisfy them
  (titles 12 and 16, subtitles 64 and 67 characters).
* **Licence names.** `create --help` does not list them and no metadata documentation ships with the package (the help
  only links to a GitHub page, not read: no network). The source shows only the examples `CC0-1.0` and `CC-BY-SA-4.0`.
  **`CC BY 4.0` is therefore NOT a documented accepted name**; both metadata files use `"other"` and state the real
  licence in the description and in the bundle's text file (`KIT_README.txt` and `DATASET_CARD.md`). You may try a
  CC BY 4.0 name on the Kaggle page after the upload (Settings, License), but I did not verify any name.

### Bundle A: the demo kit

```powershell
cd D:\Z_PS2_2026\BAS\kit_staging\bas-demo-kit
uv tool run kaggle datasets create -p . --dir-mode zip --public
uv tool run kaggle datasets status hemachandhara/bas-demo-kit
```

### Bundle B: the full dataset

```powershell
cd D:\Z_PS2_2026\BAS\kit_staging\bas-dataset-full
uv tool run kaggle datasets create -p . --dir-mode zip --public
uv tool run kaggle datasets status hemachandhara/bas-dataset-full
```

(`--dir-mode zip` changes nothing for the flat bundle A and keeps the folder layout of bundle B as zip files.
`-p .` contains no slash: see "The Windows path problem" below; the CLI's default is the current folder anyway.)

**Afterwards.** Open both dataset pages and check: visibility is public, the licence line reads as intended, the
description renders, and for bundle B that `runs`, `data`, `reports` appear (as zips if Kaggle keeps them zipped:
**[unknown]**, then say "unzip in place" to judges, as README section 8 already does). From a clean machine run the
README download commands and `python scripts/verify_assets.py` (the last line must be `READY`).

### A new version after a rebuild

Rebuild (section 3), then from inside the same bundle folder (the message is yours):

```powershell
uv tool run kaggle datasets version -p . -m "describe the change" --dir-mode zip
```

Whether `version` keeps the dataset public is **NOT VERIFIED**; check the page. Commit `assets/ASSETS.sha256` if it
changed.

### The Windows path problem

`ISSUES.md` (2026-10-03 P1.5 prep, S-F1c, "Windows note for the Kaggle CLI") records that **a path with slashes fails with
the Kaggle CLI on this machine**; the workaround is to `cd` into the folder and use a slash-free path (`-p dataset`
there). That is why every command above is run from inside the bundle folder with `-p .`. The label "D112" does not
appear in `ISSUES.md`; the note above is the entry found. If `-p .` itself is refused, omit `-p` (the default is the
current folder). **NOT VERIFIED** on this machine for datasets.

### If it fails

| what you see | cause and fix | source |
|---|---|---|
| an error about a path, the folder or "Invalid folder" | a path with slashes: `cd` into the bundle folder and use `-p .` | ISSUES S-F1c (seen) |
| the dataset page is created but private | `create` was run without `--public`: change the visibility on the dataset page (Settings), or `datasets metadata --update` after setting `isPrivate` | CLI help (not seen) |
| `The requested title ... is already in use` | a dataset with that id already exists (the account has other datasets): use `datasets version` instead, or choose another slug and rebuild | CLI source (not seen) |
| `Default slug detected` / `Default title detected` / `Please specify exactly one license` | the metadata file was replaced by the CLI template: rebuild the bundle (section 3) | CLI source (not seen) |
| `Subtitle length must be between 20 and 80 characters` (or title / slug length) | edit `DATASET_META` in `docs/kit/build_kit.py` and rebuild | CLI source (not seen) |
| an authentication or 401/403 error | the token is missing or expired: renew it in your Kaggle account settings; do not paste it into this repository | not seen |
| `uv` cannot find or install `kaggle` | `uv tool install kaggle` needs the network; it is already installed here as `kaggle` 2.2.4 | ISSUES 2026-09-30 P1 R7 (kaggle CLI) |
| the upload stalls or is refused for size or file count | Kaggle's limits are **[unknown]** (not read offline): upload `data/dataset/` as one zip and say so in the card; or use the browser upload | not seen |

**Fallback: browser upload.** On <https://www.kaggle.com/datasets> choose "New Dataset", add the files of the bundle
folder (for bundle B, zip the sub-folders first so the layout survives), set the title, subtitle, description and
visibility yourself (copy them from `dataset-metadata.json`), and set the licence to "Other" with the real licences in
the description. Open the page afterwards and check the same things.

## 5. What is decided and what is left for the Lead

Decided [user, 2026-10-07]: repository public; Kaggle user `hemachandhara`; both datasets public; code MIT; footage and
labels CC BY 4.0 with the credit line "Credit: Hemachandhar A and the SIH 2026 PS 26174 team"; consent of the people in
the footage confirmed by the Lead (no individual named); YOLO11n weights stay AGPL-3.0; copyright line
"Copyright (c) 2026 Hemachandhar A and the SIH 2026 PS 26174 team"; session S-K2 skipped, so the three known bugs remain
(README section 12) and the demo kit has four validation clips and no repeat clip.

Left for the Lead:

1. Run the upload blocks above and check the pages.
2. Upload the demo video and replace `[DEMO VIDEO LINK - Lead to add]` at the top of `README.md`.
3. Confirm the copyright name in `LICENSE`; fill in the mentors and the contact in `README.md` section 11.
4. Optionally ask the mentor or the ISRO contact whether AGPL-3.0 is acceptable for hand-over (mitigation (e) of the
   2026-10-03 P1.5 DECISION in `ISSUES.md`) and read the SIH 2026 IP clause.
5. Whether bundle B carries the weights (`--with-weights`, adds the 121 MB RF-DETR-Nano).
6. The performer and camera facts for the dataset card (`unknown` in the manifest).
7. Whether to show the test clip x038 (README section 12 (e)).

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
