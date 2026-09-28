# AGENTS.md — Session Rules for This Project

Read this at the start of every session, before touching code. It is deliberately short: `context.md` is the source of truth for the *why*; this file only says *what to do*. (Claude Code loads it through `CLAUDE.md`; run `/context` to confirm it appears under Memory files.)

---

## Fill in before starting (delete this line once filled)

```
Person:        [P1 / P2]
Track:         [P1 Perception & Data | P2 State, Engine & App]
Session:       [id + name from IMPLEMENTATION_PLAN.md Part 10, e.g. "P2.2 — SequenceEngine"]
Branch:        [p1-perception | p2-runtime | contract/<slug> | develop (G0 only)]
Features:      [F-numbers from essential-features.md]
Directories:   [the ONLY paths this session may edit — IMPLEMENTATION_PLAN.md Part 4]
Prerequisite:  [session(s) that must already be visible on origin/develop — "none" for G0]
```

## What this project is

An offline, standalone assistant for on-board experiments (ISRO PS 26174, SIH 2026). A fixed camera watches a predefined, linear experiment; the system recognizes each step, speaks the next step, voice-alerts on skipped / out-of-order / repeated steps, writes a timestamped JSONL log, streams video, and shows a dashboard. It is **advisory only**. Why: `context.md`. **How to build each feature (contract, libraries and calls, parameters, pitfalls): `essential-features.md` (F1–F14)**; deferred/excluded: `non-essential-features.md`. Who / order / behavior / tests / merging: `IMPLEMENTATION_PLAN.md` — **Part 10 is your work packet, Part 5 is the behavior, Part 7.4 is "done".** Every shape and signature: `contracts.py`. Recording and labeling by the crew: `DATA_COLLECTION.md`.

## Locked stack — do not add or swap

Python 3.11 · `uv` + one lockfile (groups `run`/`train`/`tools`/`dev`) · Pydantic v2 + NumPy · RF-DETR-Nano (YOLO11n or ONNX Runtime only if benchmarked) · MediaPipe Tasks API · OpenCV · RapidFuzz · `pyttsx3` in a worker process (a pre-rendered phrase cache is Tier 2) · Flask · vanilla JS with vendored CSS (**no framework, no CDN**) · pytest · ruff. The full table with licenses is `IMPLEMENTATION_PLAN.md` §3.1. **No cloud APIs. No FastAPI. No React.** A new dependency needs the other person's agreement, a license check, and an `ISSUES.md` entry; only P2 edits the lockfile.

## Environment

```
uv sync --group dev                      # runtime + dev deps from the lockfile (tools/train groups only where needed)
python scripts/check.py --quick          # must be green before you write code; also what the pre-commit hook runs
python scripts/check.py --status         # feature table, derived from test markers — never hand-edited
```

Four compute environments (`IMPLEMENTATION_PLAN.md` §3.2): dev laptops (CPU), a **borrowed GPU** (Colab/Kaggle) for the detector fine-tune only, the crew's recording rig, and the offline demo laptop. The team's OS is not fixed: **every tool is a Python entry point** (`python -m …` / `python scripts/x.py`), never a bare shell script or an OS-specific path.

## Non-negotiable rules

1. **Reuse libraries.** Never re-implement an algorithm a listed library provides.
2. **Stay inside your directories.** Call another owner's exposed function; never edit their files or re-implement it inline. Need something changed there → an `ISSUES.md` entry, not an edit.
3. **`contracts.py` is read-only for agents.** Found a gap? Append a `CONTRACT` entry to `ISSUES.md`, keep working against a `TEMP_<n>` stub, and do not wait. Never guess a field. Same for `config/experiment.json`.
4. **Test first, or alongside — never after.** A feature is done when its Part 7.4 row passes. Check with `python scripts/check.py --status`.
5. **GOLD-1 must never regress** (a skipped step → exactly one alert and one log line). Git hooks enforce it. Never bypass with `--no-verify`.
6. **Advisory only.** The system observes, logs and alerts. It never actuates or gates the experiment.
7. **Offline.** No outbound network call in runtime code; nothing is downloaded at runtime. Tests block non-loopback sockets.
8. **Time arrives on the data.** No clock reads in `perception/` (except `camera.py`), `state/` or `engine/`. Wall-clock time exists only inside `JsonlLogger`.
9. **Low confidence is never silently acted on.** Below `detector_conf_floor` a detection is ignored; between the floor and `confirm_conf` an event is acted on but tagged `flagged_uncertain`. Do not invent other handling.
10. **The vocabulary is closed.** Use only the words in `contracts.py` (deviation types, statuses, tags, event types). Never invent a status word on either side of a boundary.
11. **Voice is the primary channel**, and cues stay short (a phrase, not a sentence). The dashboard is reference, not the guidance channel.
12. **Split by run, never by frame. Never tune on `test`.**
13. **Determinism.** Fixed seeds; the same input gives the same output.
14. **Security.** No credentials in code, URLs, logs or git (`.env` only). Every route needs auth *and* an allowed IP. No TLS ⇒ loopback bind.
15. **Scope.** Tier 2 only after G5. Never start an explicitly excluded item (the dev-time label editor is not one: see non-essential-features.md, excluded 3). This round has one experiment (Sample Transfer); a second is Tier 2. YOLO (AGPL) only as the benchmarked fallback (chosen once at P1.5, never switched at runtime), with the license decision logged.
16. **Colour order.** Images are **BGR** everywhere. Convert to RGB *only at a model boundary* (RF-DETR, MediaPipe) — a silent BGR/RGB mix-up swaps red and yellow.
17. **Cross-platform.** Python entry points only; use `pathlib`; no bare shell scripts; nothing assumes one OS.

## One fresh session per work packet

Start a **new agent session for each numbered session in Part 10** — not one long thread. Everything a session needs to resume lives in git (commits, tests, `ISSUES.md`), not in chat. A fresh session re-orients from the repo, so it can pick up exactly where the last one stopped; a long thread fills with finished work and loses track of where it is. Finish a session (commit, push, report), close it, open the next.

## Session workflow, in order

1. **Branch.** `git checkout <Branch>` (`-b` off `origin/develop` if this is the track's first session). `git fetch origin`. Confirm with `git branch` — not still on `develop` or the last session's branch.
2. **Orient.** `git log --oneline -10`, `git status`, read `ISSUES.md`, read your types in `contracts.py`, read the existing tests. Run `python scripts/check.py --quick` as a smoke test. If it is red, fix it or log a `BLOCKER` before adding anything.
3. **Prerequisite.** `git log origin/develop --oneline -10`. If the prerequisite session's work is not visible there, it has not landed: do not build against someone's branch; ask or log a `BLOCKER`. A merged session is not a live route — for routes, confirm the URL map equals `API_ROUTES`.
4. **Read your packet** (Part 10), the **feature spec(s) it names in `essential-features.md`**, and the Part 5 section it names. Note what is out of scope. If the work touches the crew (recording, review), also read `DATA_COLLECTION.md`.
5. **Test first → implement → commit and push after each passing piece.** Reuse the named library; stay in your directories.
6. **Validate the whole Part 7.4 row(s)**, not just the last test: `python scripts/check.py --status`.
7. **Close.** `git fetch origin && git rebase origin/develop`, `git push --force-with-lease`, open a PR into `develop`, and request the other person's review; the reviewer merges. Stubs and fakes merge the same hour.
8. **Report.** What is built, which rows pass, `ISSUES.md` entries added, and the next session and whether its prerequisite is met. **If the session must end early,** commit and push anyway and say plainly what is unfinished.

## Prompt — joint G0 session

```
<goal>Joint G0 session. I am [P1/P2]; the other person is present. The repo has only main.</goal>
<orient>P2 runs: git checkout main; git checkout -b develop; git push -u origin develop — the only time
anyone is on main. Read AGENTS.md; IMPLEMENTATION_PLAN.md Parts 1–6 and 9; "G0 / P2.0" in Part 10;
and contracts.py in full.</orient>
<task>P2 drives the scaffold listed under G0 in Part 10. Both read contracts.py line by line and apply
agreed changes now — the only session where contracts are designed rather than read. Author the DRAFT
config/experiment.json with position rules; each step's rule must be able to go false before its turn (the two stow steps start true and are latched, see ISSUES.md 2026-09-28).</task>
<signoff>Nothing closes until BOTH people say yes in this sitting. Each adds a sign-off line to ISSUES.md,
both run install + python scripts/check.py green locally, plus the audio self-test and the camera check, then tag g0.</signoff>
```

## Prompt — first session on a track

```
<goal>I am [Person], starting [session id — name], the first session on this track. Repo cloned,
contracts frozen.</goal>
<orient>git checkout -b [p1-perception | p2-runtime] from origin/develop. Read AGENTS.md; my packet in
IMPLEMENTATION_PLAN.md Part 10 and its prerequisite; the Part 5 section it names; contracts.py for MY exact
types (read them, don't skim — anything unclear is an ISSUES.md CONTRACT entry now, not a guess later);
essential-features.md [F-ids].</orient>
<environment_check>Install from the lockfile and run python scripts/check.py green before writing any code. If it
is red, stop and fix or log it.</environment_check>
<first_task>Exactly what the packet says. Where it says to write a stub or fake first, merge it the same
hour.</first_task>
<rules>Follow every rule in AGENTS.md. Test first. Gaps go to ISSUES.md + a TEMP_ stub; never edit
contracts.py.</rules>
```

## Prompt — every later session

```
<goal>I am [Person], continuing [track]. Session [id — name]. Objective: [the specific next piece].</goal>
<orient>git checkout [my branch] && git fetch origin; confirm with git branch. git log --oneline -10;
git status; read ISSUES.md. Confirm the prerequisite is visible on origin/develop; if so git rebase
origin/develop. If not, it has not landed — ask or log a BLOCKER; never guess from elapsed time. Run
python scripts/check.py --quick.</orient>
<task>[Paste the Part 7.4 row(s) this session targets.]</task>
<rules>Test first. Stay in my directories. Log contract gaps; never edit contracts.py. No outbound network.</rules>
<end_of_session>Commit and push (--force-with-lease if rebased) and open the PR into develop. Report what is
built, which 7.4 rows pass (paste `python scripts/check.py --status`), ISSUES.md entries added, and the next
session and whether its prerequisite is met — marked plainly if this one is unfinished.</end_of_session>
```

## Git essentials

- Branches: `main` (untouched until P2.8), `develop`, `p1-perception`, `p2-runtime`, short-lived `contract/<slug>`. Full model: `IMPLEMENTATION_PLAN.md` Part 8.
- **Rebase, not pull:** `git fetch origin && git rebase origin/develop`. Push a rebased branch with `git push --force-with-lease`, never plain `--force`.
- **Merge at every session close**, not at the end of a track — an unmerged branch blocks the other person's prerequisite.
- Shared files (`contracts.py`, `config/experiment.json`, the lockfile, `ISSUES.md`) each have a rule in Part 8. `ISSUES.md` is append-only.
