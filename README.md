# BAS: an advisory step-checking assistant (demo and replication kit)

Source tags: **[repo]** read from a file here, **[run]** observed by running it while this kit was prepared,
**[issues]** stated in `ISSUES.md`, **[user]** stated by the project team, **[unknown]** nobody has checked.
Anything not checked is marked **NOT VERIFIED**. Placeholders in `<angle brackets>` are filled in by the
project lead before the kit is handed out.

## 1. What this is

BAS watches one fixed camera over a small scripted lab-style procedure, recognises each step with an object
detector and a hand tracker, speaks the next step and raises a voice alert when a step is skipped, done out of
order or repeated. It is **advisory only**: it observes, logs and alerts, and never controls or gates the
experiment. It was built for ISRO problem statement 26174 (SIH 2026) around **one** experiment, "Sample Transfer"
(take a red and a yellow box out of a box, put them in a tray in order, press START, stow them), and this kit
lets you run it on **recorded clips**, with no camera needed.

**What you will see.** A dashboard in your browser shows the video, the seven steps turning from pending to
confirmed, and a list of alerts; at the same time a synthetic voice speaks each next step and each alert. Four
short clips play one after another (a correct run, a skipped step, a swapped order, an idle clip) and each is
judged as its own run, with its own log in `runs_out/logs/`. The expected sounds and alerts are in section 5.

## 2. Quick start for judges

Windows, PowerShell, from an empty folder. Placeholders: `<repo-url>`, `<kaggle-user>`. Details in section 4.

```powershell
git clone <repo-url> BAS; cd BAS
uv sync --group run --group yolo; .venv\Scripts\Activate.ps1
kaggle datasets download <kaggle-user>/bas-demo-kit -p kit --unzip     # or the browser download, section 4
New-Item -ItemType Directory -Force demo_videos | Out-Null; Move-Item kit\*.pt, kit\*.task weights\; Move-Item kit\*.mp4, kit\playlist.txt demo_videos\
Copy-Item .env.example .env; notepad .env                                  # set STREAM_USER and STREAM_PASSWORD (not 'changeme')
python scripts/verify_assets.py                                            # last line must be READY
python scripts/demo.py --check --tls off --playlist demo_videos/playlist.txt
python scripts/demo.py --tls off --playlist demo_videos/playlist.txt --pause-between 5
# open the URL it prints (http://127.0.0.1:8443/?autostart=1), log in with the .env values
```

## 3. Requirements

* **Python 3.11 exactly** (`requires-python = "==3.11.*"` in `pyproject.toml` [repo]); `uv` installs it for you
  (<https://docs.astral.sh/uv/>). `git`. A browser. `kaggle` CLI only if you do not use the browser download.
* **OS: Windows.** The demo was built and rehearsed only on Windows 11 [run]. The `run` group pulls `pywin32` on
  Windows and `pyttsx3` for the voice [repo]; Linux and macOS are **NOT VERIFIED**.
* **Speakers or headphones** for the voice. No GPU is needed: the detector runs on the CPU at about 10 frames per
  second on the project laptop (Ryzen 5 5600H) [issues, 2026-10-03 P1.7 DECISION]. A slower CPU may drop more frames.
* **Disk and network.** The demo kit is **30.4 MB** (9 files) [run]; the full dataset bundle is 298.4 MB, 1,124 files,
  without weights [run]. `uv sync --group run --group yolo` installs 84 packages (76 without `yolo`) including
  `torch` [run: `uv sync --dry-run --offline --locked`]; the download size and the time it takes are **NOT VERIFIED**.
  Running the demo itself needs no network.
* Ports: the dashboard listens on `127.0.0.1:8443` (change with `--port`).

## 4. Step by step

**1. Clone.** `git clone <repo-url> BAS` and `cd BAS`. The repository is private at the time of writing
(`https://github.com/Hemachandhar-A/BAS.git` [repo, `git remote -v`]); ask the project lead to add you as a
collaborator and sign in with `git` or `gh` before cloning. *Fresh clone, timed: 3 s for a local clone [run].*

**2. Install** (needs the network, **NOT RUN** on the preparation machine: the dry run resolves from the lockfile):

```powershell
uv sync --group run --group yolo      # the demo laptop: YOLO11n is the default detector (AGPL-3.0, see section 9)
uv sync --group run                   # licence-clean build: no ultralytics; then the detector must be RF-DETR-Nano
.venv\Scripts\Activate.ps1            # use THIS interpreter for every `python` below
```

(Commands from `ISSUES.md`, 2026-10-04 "P2 R7 - dependency group yolo" [issues]. Never the `train` group.
The licence-clean build also needs the RF-DETR weights and `SIH_DETECTOR=rfdetr_nano`; that detector is **not
validated for the pipeline**, so the demo clips may behave differently with it. **NOT VERIFIED** end to end.)

**3. Download the demo kit**, Kaggle dataset `<kaggle-user>/bas-demo-kit`, in one of two ways:

* *Kaggle CLI* (needs a Kaggle account and an API token configured by you; nothing in this repo touches them):

  ```powershell
  kaggle datasets download <kaggle-user>/bas-demo-kit -p kit --unzip
  ```

  (The command shape is from the Kaggle CLI's usual syntax and was **NOT VERIFIED**: the CLI was not installed here.
  If `kaggle` is not on your PATH, `uv tool run kaggle datasets download ...` runs it, as the project did before.)
* *Browser:* open `https://www.kaggle.com/datasets/<kaggle-user>/bas-demo-kit`, press **Download**, unzip the file
  into a folder called `kit` inside the repository.

**4. Put every file where it belongs.** The kit is flat on purpose; this is where each file goes:

```
BAS/                               <- the repository you cloned
|-- weights/
|   |-- detector_yolo11n.pt        <- from the kit (YOLO11n fine-tune, 5.4 MB)
|   `-- hand_landmarker.task       <- from the kit (MediaPipe hand model, 7.8 MB)
|-- demo_videos/                   <- create it; git-ignored
|   |-- 01_x016_clean.mp4          <- from the kit
|   |-- 02_x032_skip.mp4
|   |-- 03_x025_swap.mp4
|   |-- 04_x036_idle.mp4
|   `-- playlist.txt               <- the order of the clips
`-- kit/  (SHA256SUMS.txt, KIT_README.txt, dataset-metadata.json: leave or delete)
```

```powershell
New-Item -ItemType Directory -Force demo_videos | Out-Null
Move-Item kit\*.pt, kit\*.task weights\
Move-Item kit\*.mp4, kit\playlist.txt demo_videos\
```

**5. Verify the files** (offline, read-only, standard library only; it does not load any model):

```powershell
python scripts/verify_assets.py
```

It prints `[OK]`, `[MISSING]` or `[BAD HASH]` for each of the 7 files, with the folder it belongs in and the
dataset it comes from, then `READY` or `NOT READY` (exit code 0 or 1). The expected hashes are in the committed file
`assets/ASSETS.sha256` and in `weights/MANIFEST.json`. `python scripts/verify_assets.py --kit full` also checks the
dataset bundle (section 8). It complements `demo.py --check` (step 7) and does not replace it.

**6. Credentials** (the dashboard is behind a login and refuses the placeholder values):

```powershell
Copy-Item .env.example .env       # then edit .env
```

Set `STREAM_USER` and `STREAM_PASSWORD` to **throwaway** values you invent (not `changeme`: it is refused). `.env` is
git-ignored. They are never printed. Alternatively set them for one session only:
`$env:STREAM_USER="demo"; $env:STREAM_PASSWORD="<something>"` (the check reports "from the process environment" [run]).

**7. TLS or not, then the self-check.**

* Optional TLS: `python scripts/gen_cert.py` writes a self-signed `certs/cert.pem` and `certs/key.pem`; then
  `demo.py` serves `https://127.0.0.1:8443/` and your browser warns once (expected). TLS in a real browser is **NOT VERIFIED**.
* No TLS: add `--tls off` (as in the commands here): the server binds `127.0.0.1` only and the URL is `http://`.
* The self-check, which loads the models and opens the first clip (about 8 s [run]):

  ```powershell
  python scripts/demo.py --check --tls off --playlist demo_videos/playlist.txt
  ```

  Every line must say `[OK]` and the last must say `self-check passed` (exit code 0). Always pass `--playlist`:
  without it the check also tries camera 0 and fails on a laptop that has none.

**8. Run the demo.**

```powershell
python scripts/demo.py --tls off --playlist demo_videos/playlist.txt --pause-between 5
```

What it prints, in order [run]: the self-check lines, then

```
READY  http://127.0.0.1:8443/?autostart=1
  login:   credentials from the process environment (no .env file) (values not shown)   <- or from your .env
  logs:    runs_out/logs
  ...
  the first frame shows at once; the run and the voice start when the page asks (?autostart=1) or when you press Start; Ctrl+C stops cleanly
```

Open that URL and log in. The first frame appears at once as a still; about half a second after the page has
loaded, the run starts and the voice speaks the first step. Then one terminal line per clip start and finish
(clip, run id, events), and `playlist finished` after the fourth clip (add `--once` to stop there; without it the
playlist loops until you press Ctrl+C). The whole show took about 90 s with a 2 s pause on the preparation laptop [run].

## 5. The demo order

Order and clips were chosen as a story: clean, skip, swap, idle. All four are **validation-split** original
recordings (no `--allow-heldout` needed). "Observed" is what the live pipeline logged when the clip was played through
the gated start (the flow you get with `?autostart=1`) [run]; the spoken text is the text the engine produced for the
alert in the cached replay `python scripts/replay.py --from-cache <run id>` [run]. **No audio was listened to**
while preparing the kit: that the voice speaks these words is from the code and logs, not from hearing it (**NOT VERIFIED**).

| # | clip | run type | what happens in the clip | you will hear (first cue, then alerts) |
|---|---|---|---|---|
| 1 | `01_x016_clean.mp4` (22 s) | correct | all seven steps in order; 7 confirmed, run ends `Experiment complete` | "Take out the red sample", then each next step ("Place red in the tray", ...); no alert |
| 2 | `02_x032_skip.mp4` (19 s) | skip | the START press is skipped; 5 confirmed, 1 omission, run ends `Experiment ended with skipped steps` | "Step skipped: Start pressed" at about 12 s |
| 3 | `03_x025_swap.mp4` (16 s) | swap | yellow is loaded before red; 1 confirmed, omission, 2 out-of-order, omission | "Step skipped: Red sample out, Red sample in tray", "Out of order: Red sample out", "Out of order: Red sample in tray", "Step skipped: Start pressed, Red sample stowed" |
| 4 | `04_x036_idle.mp4` (16 s) | idle | a hand roams, no step is performed; no step, no alert, run ends `Run ended early` | only the first cue at the start |

Notes. On clip 2 two steps are tagged `flagged_uncertain` in the log (acted on, but low confidence) [run]. A
**repeat** deviation (a step done twice) is not in the default order: the only validation repeat clip, x040, is
8.6 s and its replay shows an omission, not the repeat; the repeat test clip x038 is held out (section 6). A correct
run on which the START press is **missed** (x022: a false "Step skipped: Start pressed") is the known limit; it is in
the full dataset (section 8), not in the kit.

## 6. Controls, other clips and the rules

* **Dashboard.** It can **Start** and **Reset** a run, and it shows the video, the steps and the alerts. It has no
  clip dropdown, no camera switch and no detector switch, by design: the detector is chosen once at launch and never
  switched during a run (`ISSUES.md`, 2026-10-03 P1.5 DECISION, mitigation (a) [issues]), and the clip order is set by
  the presenter in the terminal. Routes: `API_ROUTES` in `contracts.py` (`/`, `/video_feed`, `/api/status`,
  `/api/experiment`, `/api/log`, `/api/run/start`, `/api/run/reset`), all behind login [repo].
* **Pace and order** (all in `python scripts/demo.py --help`): `--advance enter` holds the last frame until you press Enter;
  `--advance auto` (default) waits `--pause-between` seconds; `--shuffle --seed N` plays a reproducible shuffled order;
  `--once` stops after one pass; `--no-record` writes no `.avi`; `--speech-delay SECONDS` (default 0.2) moves the voice
  against the picture: raise it if the voice comes before the picture. Do not use `--no-wait-for-dashboard` for the show:
  it skips the idle warm-up and the first clip can lose its first two steps (a false alert) [run, `docs/KIT.md` section 6].
* **Another clip, another order.** Edit `demo_videos/playlist.txt` (one path per line, `#` comments) or point
  `--playlist` at a folder (all videos in name order). Clips of the 46 runs are in the dataset bundle as `runs/<id>/video.mp4`.
* **The original-clips rule.** Play the original files. **Never re-encode, trim or stitch clips into one file:** every clip
  must be its own run; a joined file is judged as one long run and the second clip as the continuation of the first
  (`demo.py --help`) [repo]. A re-encoded copy also changes the pixels and the result [issues, 2026-10-04 S-I1b].
  A listed file that is missing is skipped with a warning, so the show gets shorter: check with `verify_assets.py`.
* **Held-out clips.** The test split is never used for tuning. A clip of a test run needs `--allow-heldout`
  (`--replay-run x038 --allow-heldout`); the flag only prevents accidents. The guard looks at the **path** (`runs/<id>/`),
  so a renamed copy elsewhere is not recognised [run: reading `harness/live.py`]: do not rename test clips.

## 7. Using a live camera (optional, not verified on your hardware)

`python scripts/demo.py --source 0` (a camera index; `--source 1` for another) reads the camera at 1280 x 720, 30 fps if it
grants it. The system was built for **one fixed overhead camera** looking at the work area, with the same props as the
recordings (a shoe-box outer box, a tray, red and yellow boxes, a green START pad: `DATA_COLLECTION.md`). A different
viewpoint, lighting or props is a risk: the detector was trained on one set-up (section 9) and may fail on yours.
On the preparation laptop the built-in camera opened at 1280 x 720, 30 fps [run: `demo.py --check`]; no live run with the
props was done for this kit. **Optional, not verified on the judge's hardware.** A camera run starts from the dashboard (Start).

## 8. Reproducing the dataset and the results

Download the Kaggle dataset `<kaggle-user>/bas-dataset-full` (browser or `kaggle datasets download <kaggle-user>/bas-dataset-full -p full --unzip`,
**NOT VERIFIED**) and copy its folders over the cloned repository, same names:

```
BAS/runs/<id>/video.mp4  (46 runs: x001 ... x046)   BAS/runs/{manifest,provenance,run_plan}.csv
BAS/data/dataset/{train,valid,test}/                BAS/data/corrections/{train,val,test}.json   BAS/data/label_review.csv
BAS/reports/dataset.json  BAS/reports/training/     DATASET_CARD.md (also in docs/)   SHA256SUMS.txt
```

If the datasets arrive as zip files per folder (see `docs/KIT.md` section 4), unzip them in place. Check, offline:

```powershell
python scripts/verify_assets.py --kit full          # every run video against runs/manifest.csv, and the dataset folders
python -m training.verify_dataset                   # the COCO dataset equals reports/dataset.json (OK: train 593, valid 211, test 213 [run])
```

* **Perception cache** (about 85 s for the validation runs [issues, S-G]; writes `data/cache/`): `python -m perception.cache build --split val`
  (also `--split train`). Do not build `--split test` unless you intend to use the test replay.
* **Replay on validation** (cached, no models): `python scripts/replay.py --from-cache x016` prints the engine events of one run;
  repeat for any validation run (x015 x016 x017 x022 x024 x025 x028 x032 x036 x040).
* **Test replay is single-use.** `python scripts/replay.py --from-cache all --split test` was run once and its report is committed
  (`reports/replay_test.json`); a second run is refused while that report or the marker `reports/.replay_test_done` exists, unless
  `--allow-repeat-test "<reason>"` is given and the reason is recorded in the new report [issues, S-I1 F0]. The same guard protects
  `training.eval_detector --split test`. This is so nobody tunes on the test split by accident.
* **Detector evaluation:** `python -m training.eval_detector --model yolo11n --weights weights/detector_yolo11n.pt --dataset-dir data/dataset --split valid --device cpu`
  (see `--help`; it writes a report under `reports/`: pass `--reports-dir` to keep the committed one). The committed reports are
  `reports/detector_eval_valid_yolo11n.json` and `reports/detector_eval_test_yolo11n.json`.
* **Needs a GPU:** only **training** the detector (`python -m training.finetune`). It was run on Kaggle, T4; the steps, the
  commands and the Kaggle run are in `training/README_GPU.md`. Nothing else here needs a GPU.

## 9. Results and honest limits

Numbers are copied from the named file; nothing is rounded up.

* **Detector acceptance, gold test frames (YOLO11n).** 80 hand-verified test frames: recall per class 1.000 (outer_box, tray, red_box,
  start_button) and **0.950 (yellow_box)** against a threshold of 0.85; mAP50 **0.9988** against 0.80: PASS. On all 213 test images mAP50 is 0.9971
  and mAP50-95 0.9615 [`reports/detector_eval_test_yolo11n.json`; `ISSUES.md` 2026-10-03 S-F2b]. Valid drove early stopping, so valid numbers
  are optimistic; scores are near ceiling and say little about other people or lighting [issues].
* **Replay on the held-out test runs: 8 of 10 strict matches. The formal gate FAILED** (it requires 0 mismatches). The two mismatches, x014 (one
  extra START press) and x045 (a missed START press), are the documented START-press limitation: with START excluded the result is **10 of 10**
  (for information only; it does not change the verdict) [`reports/replay_test.json`; `ISSUES.md` 2026-10-03 P2.7]. The val result is also 8 of 10 [issues].
* **Frame rate.** `target_fps` is 10; the pipeline's median was 47.7 ms per frame (detector and hands) on one laptop [issues, P1.7 DECISION]. A live run
  achieved 9.18 fps over the processing window against a minimum of 8 [issues, S-I1 M1].
* **The data.** 46 recorded runs (26 train, 10 val, 10 test), **not the 77 planned** [issues; `runs/manifest.csv`]; one performer, one camera set-up,
  848 x 480 compressed footage, no gloves, one experiment [issues, `docs/DATASET_CARD.md`]. 22 "correct" runs carry a label assumed from the crew and
  were not watched; 24 runs with a deviation were watched by the lead [`runs/provenance.csv`]. Labels partly come from an auto-labeller (Grounding DINO tiny),
  corrected by hand on gold subsets (88 val, 80 test frames) [repo, `reports/dataset.json`].
* **START press** is the weak point: short taps, passes over the card and merged double taps are recognised only partly [issues].
* **Licence of the detector.** YOLO11n weights and code are **AGPL-3.0**. The vendor's stated position is that private or proprietary use needs
  its Enterprise licence or open-sourcing the whole solution; this was recorded but **not checked with a lawyer**, and the hand-over question is open
  [issues, 2026-10-03 P1.5 DECISION]. Alternative: RF-DETR-Nano, Apache-2.0, trained and evaluated on validation only, **not validated for the pipeline**,
  **not evaluated on test**. This is not legal advice.
* **"Offline".** Project code makes no outbound network call at run time (rule 7, `AGENTS.md`; tests block non-loopback sockets) [repo]. One open point,
  quoted from `ISSUES.md` (2026-10-04, S-I1, item 1, a PERCEPTION proposal; no later entry resolves it): MediaPipe's hand landmarker logs
  `portable_clearcut_uploader ... Failed to send to clearcut` each time it is recreated, "an attempted telemetry upload; offline it fails harmlessly",
  and "not verified here whether it actually sends when a network exists". **Status: open, not verified.** On a laptop with internet access that upload may succeed.
* **Not verified:** audio heard, TLS in a browser, a real camera with the props, Linux and macOS, the time and size of `uv sync`.

## 10. Troubleshooting

| symptom | cause and fix |
|---|---|
| `[FAIL] credentials ... changeme` or "STREAM_USER" not set | edit `.env` (section 4, step 6); `changeme` is refused |
| `[FAIL] port: 127.0.0.1:8443` busy | another demo is running: stop it, or `--port 8444` |
| `[FAIL] camera` | you ran `--check` without `--playlist`, or another app holds the camera: pass `--playlist demo_videos/playlist.txt`, or close the other app and try `--source 1` |
| no sound | the system keeps running silently; fix the Windows default output and run `python scripts/demo.py --tts-test` (you should hear "Audio check"); use the dashboard alerts meanwhile |
| `[BAD HASH]` or `[FAIL] detector weights: sha256 ... differs from the manifest` | the file is partial or wrong: download it again from the Kaggle dataset named in the line. Never edit `weights/MANIFEST.json` |
| `[MISSING]` | put the file in the folder the line names. A line "found at X: move it to Y" means it is in the wrong place |
| `weights/` has no `detector_yolo11n.pt` after `git clone` | by design: weights are git-ignored (`.gitignore`); they come from the Kaggle kit |
| a clip is skipped: `warning: playlist: skipping '...': file not found` | a clip file is missing or misspelled in `playlist.txt` |
| PowerShell says running scripts is disabled (`Activate.ps1`) | skip the activation and call the interpreter directly: `.venv\Scripts\python.exe scripts/demo.py ...` |
| clone or copy fails on a long Windows path | clone into a short folder such as `C:\BAS` |
| `'pytest' is not recognized` / `No module named ...` | wrong interpreter: activate `.venv\Scripts\Activate.ps1` or call `.venv\Scripts\python.exe`; `uv sync` first |
| `uv sync` cannot find Python 3.11 | `uv` downloads it (network); the project needs 3.11.x exactly |
| the browser warns about the certificate | expected with `gen_cert.py` (self-signed); or use `--tls off` |
| the page shows a still frame and nothing happens | the run starts when the page asks: open the URL that ends in `?autostart=1`, or press Start |
| the first clip raised alerts it should not | you used `--no-wait-for-dashboard`; leave it out (section 6) |
| something else | run `python scripts/demo.py --check --tls off --playlist demo_videos/playlist.txt` and read the first `[FAIL]` line |

## 11. Repo map, licence, credits, contacts

| folder or file | what it is |
|---|---|
| `scripts/` | entry points: `demo.py` (the demo), `dev.py`, `replay.py`, `gen_cert.py`, `verify_assets.py`, `check.py` (tests and lint), `install_hooks.py` |
| `perception/` | camera or file source, detector, hand tracker, the pipeline, perception caches |
| `state/`, `engine/` | step state tracker and the sequence engine (skip, out-of-order, repeat) |
| `runtime/`, `outputs/`, `server/` | the live loop and recorder; the JSONL logger and the voice; the Flask server and the dashboard files |
| `harness/` | live wiring behind `demo.py`, replay, tuning, held-out guard |
| `training/` | dataset pipeline, labelling, evaluation and fine-tuning (`README_GPU.md`) |
| `config/` | experiment definition, runtime and perception settings, acceptance criteria |
| `reports/`, `runs/` | measured results; run scripts and manifests (videos are not in git) |
| `weights/`, `data/`, `demo_videos/`, `kit_staging/`, `runs_out/`, `certs/` | git-ignored: models, datasets, clips, staged bundles, logs, TLS files |
| `assets/` | `ASSETS.sha256`, the expected hashes of the demo kit |
| `docs/` | `DATASET_CARD.md`, `KIT.md` (maintainer notes), `LICENSES.md`, `kit/build_kit.py` |
| `tests/` | unit and integration tests; `contracts.py` is the typed contract; `AGENTS.md`, `context.md`, `IMPLEMENTATION_PLAN.md`, `ISSUES.md` are the working documents |

**Licence.** The repository has no `LICENSE` file [repo]: **DECISION - Lead.** Third-party licences and the AGPL note: `docs/LICENSES.md`.
**Credits.** `<team names and mentors>` (**DECISION - Lead**). **Contacts.** `<name, email>` (**DECISION - Lead**).

---

# Appendix A. Pre-demo checks (for whoever presents)

Everything below needs real hardware and is done by a person; nothing here has been run for you.

1. **TLS and the dashboard in a real browser.** `python scripts/gen_cert.py`, then
   `python scripts/demo.py`. Open the printed `https://127.0.0.1:8443/` in Chrome or Edge, accept
   the self-signed warning, log in with the `.env` credentials. Confirm the video, the step
   status and the alerts update.
2. **Webcam at index 0.** The self-check line `[OK] camera: camera 0 opened: granted WxH ...` shows
   the granted mode (1280 x 720 at 30 fps is requested; the camera module logs
   `capture mode granted` too). The dashboard shows the live feed. If another mode is granted,
   note it.
3. **Real audio.** `python scripts/demo.py --tts-test` prints `[OK] TTS test ...` and you hear
   "Audio check". Wrong device or volume: fix the Windows default output and repeat.
4. **A full live run with the props on the table.** Press Start on the dashboard, perform the
   procedure. Note the spoken cues, then read the log in `runs_out/logs/<run_id>.jsonl` and play
   the saved file `runs_out/video/<run_id>.avi`.
5. **A deliberately skipped step** (the GOLD-1 pattern): skip one step. Expect exactly one spoken
   alert and exactly one `deviation_detected` line in the log.
6. **Unplug the camera for 10 seconds**, then plug it back in. Expect `feed_lost` in the log
   after about 3 seconds without frames, `feed_restored` when frames return, and the feed on the
   dashboard resuming. If the camera does not come back by itself, stop with Ctrl+C and restart.
7. **The recorded-video demo flow** (needs a real browser and speakers; not verified by the
   author). With the kit in place run
   `python scripts/demo.py --tls off --playlist demo_videos/playlist.txt --pause-between 5` and:
   1. Open the printed URL (it ends in `?autostart=1`): **the first frame appears immediately**,
      before the page has finished loading anything else.
   2. **The run starts about half a second later**: the banner changes from "waiting to start"
      to "running" and **the first step phrase is heard as the video starts moving**.
   3. **Alerts are heard when the matching event appears on screen.** If the voice comes before
      the picture raise `--speech-delay` (try 0.4), if it comes after, lower it; restart to apply.
   4. **A clip ends, the last frame holds, the next clip starts**: one terminal line per
      transition, the dashboard shows the new run id (it contains the clip name) and a fresh step
      list. With `--advance enter` it waits for your Enter.
   5. **Ctrl+C stops cleanly**: the running clip's run is finished in its log, no traceback.
8. **If something fails.** Run `python scripts/demo.py --check` and read the FAIL line, and see the
   table in section 10. Backup: `python scripts/demo.py --replay-run x016` plays a recorded val run
   in real time through the same loop, with speech and dashboard (needs `runs/x016/video.mp4`, i.e. the
   dataset bundle, or copy that one clip).

Other facts that still hold: TLS is on automatically when `certs/cert.pem` and `certs/key.pem` exist (or
`tls_cert` / `tls_key` are set in `config/runtime.yaml`); `--replay-run RUN_ID` refuses a test-split run
unless `--allow-heldout` is given; a file source does not start by itself, it shows its first frame and
waits for the dashboard (`?autostart=1`) or Start; Ctrl+C stops cleanly (run finished, recorder file
closed, log flushed, voice process stopped). The development server, with Start pressed by hand, is
`python scripts/dev.py`.
