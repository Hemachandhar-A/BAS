# Run the demo

(A short stub. The full README is the next session's work.)

```
python scripts/gen_cert.py          # once: self-signed TLS cert into certs/ (optional but recommended)
python scripts/demo.py --check      # the startup self-check only; exit code 1 on any FAIL
python scripts/demo.py              # the demo: camera 0, dashboard at the URL it prints
python scripts/demo.py --replay-run x016   # backup mode: a recorded run played in real time
python scripts/dev.py               # the same system for development (press Start on the dashboard)
```

* Credentials: copy `.env.example` to `.env` (git-ignored) and set `STREAM_USER` and
  `STREAM_PASSWORD` to real values. `changeme` is refused. They are never printed.
* TLS is on automatically when `certs/cert.pem` and `certs/key.pem` exist (or when `tls_cert` /
  `tls_key` are set in `config/runtime.yaml`); otherwise the server binds 127.0.0.1 and the URL is
  `http://127.0.0.1:8443/`.
* `--replay-run RUN_ID` refuses a run of the test split unless `--allow-heldout` is given.
  Demonstrating a held-out run is honest and allowed; the flag only prevents accidents.
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
7. **If something fails.** Run `python scripts/demo.py --check` and read the FAIL line.
   Typical fixes: credentials still `changeme` in `.env` (set both values); port in use (stop the
   other instance or pass `--port 8444`); camera busy (close other apps, try `--source 1`); no
   audio (the system keeps running silently, voice is then missing: use the dashboard and say so);
   weights hash mismatch (restore `weights/` from the backup, never edit `MANIFEST.json`).
   Backup: `python scripts/demo.py --replay-run x016` plays a recorded val run in real time
   through the same loop, with speech and dashboard.
