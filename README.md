# Run the demo

(A short stub. The full README is the next session's work.)

```
python scripts/gen_cert.py          # once: self-signed TLS cert into certs/ (optional but recommended)
python scripts/demo.py --check      # the startup self-check only; exit code 1 on any FAIL
python scripts/demo.py              # the demo: camera 0, dashboard at the URL it prints
python scripts/demo.py --replay-run x016   # backup mode: a recorded run played in real time
python scripts/demo.py --playlist demo_videos --pause-between 5   # the recorded-video demo
python scripts/dev.py               # the same system for development (press Start on the dashboard)
```

### The recorded-video demo (one person runs and explains it)

```
python scripts/demo.py --playlist demo_videos --pause-between 5
```

1. **Prepare the folder.** Copy the chosen clips (for example held-out runs, with `--allow-heldout`)
   into `demo_videos/` next to this README; the clips play in name order (`01_x016.mp4`,
   `02_x024.mp4`, ...). Do not commit the clips: `demo_videos/` must stay out of git (it is not
   in `.gitignore` yet: add the line, or use `.git/info/exclude`). A text file with one clip path
   per line (relative to the file) works as the playlist as well.
2. **Never stitch the clips into one file.** Every clip is its own run (own `run_id`, log and
   `run_completed`); a joined file would be judged as one long run and the second clip as the
   continuation of the first.
3. **Open the printed URL.** `demo.py` prints `READY  https://127.0.0.1:8443/?autostart=1` (no
   credential in it). The page shows the first frame at once as a still; about half a second after
   it has loaded, the page starts the run and the voice speaks the first step as the video starts
   moving. Without `?autostart=1` press Start by hand: the clip waits for it either way.
4. **Voice versus picture.** The cue is held back by `--speech-delay` seconds (default 0.2) so that
   it lines up with the frame on the screen. Tune by eye: if the voice runs ahead of the picture
   raise it (`--speech-delay 0.4`), if it lags lower it (`0` switches the delay off).
5. **The presenter paces the show.** `--advance auto` (default) starts the next clip after
   `--pause-between` seconds; `--advance enter` holds the last frame until you press Enter in the
   terminal, so you can explain each result. `--shuffle --seed N` (reproducible order), `--once`
   (stop after one pass; the default loops until Ctrl+C), `--no-record` (no `.avi` files).
6. The server, the dashboard session and the voice stay up between clips; each clip is one line in
   the terminal (clip, `run_id`, events). The `run_id` shown on the dashboard contains the clip name.
7. Headless, tests and soak: add `--no-wait-for-dashboard` (the run starts at once). Camera sources
   never wait.

* Credentials: copy `.env.example` to `.env` (git-ignored) and set `STREAM_USER` and
  `STREAM_PASSWORD` to real values. `changeme` is refused. They are never printed.
* TLS is on automatically when `certs/cert.pem` and `certs/key.pem` exist (or when `tls_cert` /
  `tls_key` are set in `config/runtime.yaml`); otherwise the server binds 127.0.0.1 and the URL is
  `http://127.0.0.1:8443/`.
* `--replay-run RUN_ID` refuses a run of the test split unless `--allow-heldout` is given.
  Demonstrating a held-out run is honest and allowed; the flag only prevents accidents.
* A file source (`--replay-run`, `--source FILE`, `--playlist`) no longer starts by itself: it shows
  its first frame and waits until the dashboard starts the run (`?autostart=1` does it, or press
  Start). `--no-wait-for-dashboard` (or `--auto-start`) restores the immediate start, for tests,
  the soak (`--loop`) and headless runs.
* Ctrl+C stops cleanly: the run is finished, the recorder file closed, the log flushed, the voice
  process stopped.
* Known limits, stated honestly: the START press is recognised only partly (short taps, passes over
  the card, merged double taps); one performer, one setup, no gloves; the YOLO detector is
  AGPL-3.0 (optional dependency group `yolo`).

## Pre-demo checks

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
   author). Put two or three clips in `demo_videos/`, run
   `python scripts/demo.py --playlist demo_videos --pause-between 5` and:
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
8. **If something fails.** Run `python scripts/demo.py --check` and read the FAIL line.
   Typical fixes: credentials still `changeme` in `.env` (set both values); port in use (stop the
   other instance or pass `--port 8444`); camera busy (close other apps, try `--source 1`); no
   audio (the system keeps running silently, voice is then missing: use the dashboard and say so);
   weights hash mismatch (restore `weights/` from the backup, never edit `MANIFEST.json`).
   Backup: `python scripts/demo.py --replay-run x016` plays a recorded val run in real time
   through the same loop, with speech and dashboard.
