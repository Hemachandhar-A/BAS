# DATA_COLLECTION.md — Crew Guide: Recording the Experiment Data

**Who this is for:** the four crew teammates and anyone who performs in front of the camera. You do **not** need to read the engineering documents. This guide is complete on its own. (If you are curious *why* something is asked, section 9 says what happens to your recordings, and `context.md` §5 has the full reasoning.)

**Why your work matters most.** The AI learns what our objects look like from your videos, and it is *graded* on your videos. Consistency and honesty beat volume: a run where you did something different from what you declared is worse than no run.

**Two crew jobs.** *Data crew* (two people) run the recording, the label checks and the label corrections. *Comms crew* (the other two) build the PPT and demo video — see `IMPLEMENTATION_PLAN.md` Part 10. Everyone can be a performer.

---

## 1. The experiment: "Sample Transfer"

ISRO's problem statement gives one hint: *a box that contains two smaller boxes, red and yellow*. The rest is ours to design. We built a small **analogue** of a real on-orbit procedure with everyday objects.

**The story.** In an experiment workspace, a crew member takes two sample containers out of a stowage box, loads them into an experiment tray **in a fixed order** (red first, then yellow), presses START, then stows the samples back in the same order.

**Why this resembles the real domain (evidence, not decoration).**
- The ISS Microgravity Science Glovebox is described as a "[workbench](https://ntrs.nasa.gov/api/citations/20090017703/downloads/20090017703.pdf)" environment: a bounded work area where the crew installs and handles experiment hardware, with [powered video drawers and cameras monitoring the investigations](https://www.eusoc.upm.es/microgravity-science-glovebox/). That is our fixed-camera, bounded-workspace setup.
- Crews routinely [change samples and perform operations](https://www.sciencedaily.com/releases/2002/06/020617075754.htm) in that workspace. NASA's daily reports describe [sample-preparation steps inside a glovebox](https://www.nasa.gov/blogs/stationreport/2020/10/page/2/), and ground-written procedures are [performed in sequence](https://spaceref.com/?p=94029) by the crew. That is our ordered-steps structure.
- **Honest limit:** the real BAS procedures are not public. This is the same *structure* (bounded workspace, fixed camera, ordered handling of containers, a panel action), not a replica.

### Objects (five things; everything is everyday stuff)

| Object (detector class) | What to use | Requirements |
|---|---|---|
| `outer_box` | The open bottom of a shoe box or a shallow cardboard box, about 30 × 20 cm. **Lid removed** | Must be open-topped so the camera sees the boxes inside. Plain colour that is **not** red/yellow/green |
| `red_box` | A small box about 6 × 6 × 4 cm (tea box, medicine box, wrapped in red paper if needed) | Saturated **red**, matte, no white lettering across the top |
| `yellow_box` | Same size as the red box, saturated **yellow** | Matte. Same size as red so only colour tells them apart |
| `tray` | A flat plastic lunch-box lid, baking tray or placemat, about 25 × 20 cm | Big enough for both small boxes side by side. **Blue, black or white**, not red/yellow/green |
| `start_button` | A **green** coaster or sticky-note pad, at least 6 cm across, with "START" written on it | Fixed to the table with tape. It must not move during a run |

### Layout (view from above)

```
┌──────────────────────────────────────────────────┐  ← tape rectangle ≈ 50 × 35 cm
│                                                  │     = the "work volume"
│   ┌────────────────┐         ┌───────────────┐   │
│   │   OUTER BOX    │         │     TRAY      │   │
│   │  [RED] [YELLOW]│         │   (empty)     │   │
│   └────────────────┘         └───────────────┘   │
│                                       ( START )  │
│            ✕ parking spot (bare table)           │
└──────────────────────────────────────────────────┘
```

**Tape the home position** of every object with masking-tape corners so each run starts identically to within about 2 cm. **The outer box, tray and START button never move during a run.** Keep the tray and the outer box **apart** (no overlap in the camera image) and keep START away from the boxes' travel path.

### The seven steps

| # | Step (id) | You do | Spoken cue |
|---|---|---|---|
| 1 | `red_out` | Lift the red box **out of** the outer box | "Take out the red sample" |
| 2 | `red_in_tray` | Put the red box **into the tray** | "Place red in the tray" |
| 3 | `yellow_out` | Lift the yellow box out of the outer box | "Take out the yellow sample" |
| 4 | `yellow_in_tray` | Put the yellow box into the tray | "Place yellow in the tray" |
| 5 | `start_pressed` | Touch and **hold** the START button for **at least 1.5 s** | "Press the start button" |
| 6 | `red_stowed` | Put the red box back **into the outer box** | "Stow the red sample" |
| 7 | `yellow_stowed` | Put the yellow box back into the outer box | "Stow the yellow sample" |

The draft definition lives in `config/experiment.json`. It is a **draft until the feasibility check (session P1.2)** confirms the camera and detector can tell all five objects apart. If they change it, the run plan keeps working because it uses step ids, and you will be told.

---

## 2. The environment (set this up once; do not touch it during a session)

| Item | Requirement | Why |
|---|---|---|
| **Camera position** | **Straight down** (overhead) on a tripod, a gooseneck arm or a stack of books, about 50–60 cm above the table. All five objects fully in frame with about 10 % margin | The AI decides "inside / outside" from the 2-D image. Overhead removes the depth ambiguity a front camera has |
| **Camera settings** | 1280 × 720 at 30 fps if the webcam allows, otherwise its best fixed mode. **Same mode for every run.** Lock exposure and white balance if the camera software has the option | Auto-exposure "hunting" shifts colours mid-run and confuses red vs yellow |
| **Surface** | A plain, matte table or sheet. **Not red, yellow or green** | Those are our object colours |
| **Lighting — setup `S1`** | Steady room light, no direct sunlight patch, no strong shadow across the workspace | The main condition |
| **Lighting — setup `S2`** | Deliberately different: a single desk lamp from the side, or room light off with a window. Used for about 1 in 7 runs | Tests robustness to lighting |
| **Performer** | Neutral sleeves (**no red/yellow/green clothing**), one hand does the work, natural speed | Colour confusion, and the hand tracker follows one clear hand |
| **Sound** | Speakers on. The system speaks later, and you will rehearse with it | Demo condition |

**`camera_setup_id`:** `S1` (standard light) or `S2` (alternate light). **If you move the camera or lights mid-session, add a letter (`S1b`) and tell the Data crew.** The system records this so a moved camera never silently corrupts the data.

### Equipment checklist
Overhead camera stand · laptop with the repo installed · 5 objects · masking tape · marker · a desk lamp (for `S2`) · thin gloves (for robustness runs) · 3–4 unrelated objects (pen, mug, phone) for distractor runs · a timer or phone stopwatch · a shared folder (Drive/USB) for the videos.

---

## 3. People and roles

Each run needs **two people**:
- **Performer:** does the steps.
- **Director:** runs the recording tool, reads the run's instruction aloud, checks the preview, and confirms what was actually performed.

**Operators.** `O1`, `O2`, `O3` are three teammates who record throughout. **`O4` is a fourth teammate who appears only in the test set** (the plan marks which runs). O4 must **not** perform in any train or val run and must not be shown the recordings in advance. That is how we measure whether the AI works on a hand it has never seen. Write real names against O1–O4 in `runs/run_plan.csv` before you start.

---

## 4. The run plan: `runs/run_plan.csv`

77 planned runs, already ordered for recording (setup `S1` block first, then `S2`; shuffled within each block so tiredness never lines up with one kind of run).

> ### Pilot first: record rows 1–12, then STOP
> Rows 1–12 are a deliberately chosen pilot: **5 correct runs plus one each of skip, swap, repeat, idle, gloves, distractor and fast**, all from the `train` split and lighting `S1`, so the feasibility check never looks at validation or test recordings.
>
> The experiment is a **draft** until P1 has checked, on real footage, that the camera and detector can tell the five objects apart and that "inside/outside" behaves. That check (session P1.2) can change the props or the steps. So: **record only rows 1–12 (the pilot), validate them, tell P1, and wait.** P1 will tell you "the experiment is frozen" (gate G2); only then record rows 13–77. If the props change, the pilot is discarded, which costs about 25 minutes instead of a whole afternoon.

| Split | Runs | Used for |
|---|---|---|
| `train` | 36 | Teaching the object detector |
| `val` | 14 | Tuning thresholds (not the detector) |
| `test` | 27 | The final honest score. Never used for tuning |

The counts are **judgment calls, not evidence-derived**: no source gives "the right number". They were sized so that (a) train frames land in the range RF-DETR's own documentation covers for fine-tuning (about 500–2,000+ images) after near-duplicate frames are removed, (b) every deviation variant appears in both val and test, and (c) the whole plan takes about **2.6 hours** of recording (≈ 2 min per run including reset). If the detector under-performs, we add more `correct` runs first.

Types of run: **22 correct · 15 skip · 11 swap · 10 repeat · 5 idle · 14 robustness.** Every row has an `intent` column that tells you exactly what to do. Summary:

| Type | What you do | Example intent (row text) |
|---|---|---|
| `correct` | All 7 steps in order | Perform every step in order |
| `skip` (a/b/c) | Leave out something on purpose | a: take red out but don't put it in the tray · b: forget START · c: never touch the yellow box |
| `swap` (a/b/c) | Right steps, wrong order | a: load yellow first · b: yellow out, START, then yellow in tray · c: stow yellow before red |
| `repeat` (a/b) | Do a step twice | a: put red in the tray, lift it out for 1 s, put it in again · b: press START, lift away 1 s, press again |
| `idle` (a/b) | **No steps at all** | a: nobody in view, 30 s · b: hands wander 40 s but touch nothing |
| `robustness` | Correct run under stress | dim light · gloves · distractor objects and a second hand at the frame edge · double speed |

**Performing rules (these matter more than they look):**
1. **Natural pace, about 1 s per move**, then **pause about 1 s after each step.**
2. **Hold contact ≥ 1.5 s** on the START button. Brief taps are not counted by the system, by design (a single-frame touch must not trigger anything).
3. For `repeat`, **lift the box fully out for a full second** before placing it again. Otherwise it is one long placement, not two.
4. **Never move the outer box, tray or START button.**
5. Do not cover a box with your whole forearm for a long time.
6. If you make a genuine mistake that is *not* the planned intent, **record what you actually did (see section 6), or discard and redo.**

---

## 5. Recording one run (the Director's checklist)

Tool: `python -m perception.record --plan runs/run_plan.csv` (built in session P1.1; until then, rehearse the layout with any camera app).

1. Objects on their tape marks. Camera and lights unchanged since the last run (else new `camera_setup_id`).
2. The tool shows the next planned run: read its **run_id, operator and `intent` aloud** to the performer.
3. Check the **live preview**: all five objects visible, colours look right, no glare. If not, fix and restart the run.
4. The tool counts down 3-2-1 and starts recording. Performer performs. Director watches for mistakes.
5. Press **S** when the performer says "done" (idle runs stop at their timer). Press **X** to discard a run.
6. The tool asks: *"Was it performed exactly as planned?"* Press **Enter** for yes. If not, **type the steps actually performed** (ids from the table in section 1).
7. Reset the objects to their tape marks. Next run.

**Every session, before you start:** do **3 practice runs** (discarded) to confirm layout, lighting and audio.

---

## 6. Truthful declarations and mistakes

The performer's **declaration of what was actually done is the ground truth** for the whole project. Nobody labels frames by hand; the system compares its own output with your declaration. So:
- If the run went exactly as planned → confirm it.
- If it did not (a dropped box, an accidental swap) → **type what really happened**. That is still a valid, useful recording (it becomes a different kind of run).
- If you are unsure → discard and redo. **Never edit a video or a declaration afterwards.**

The system re-derives what the AI *should* report from your declared steps and fails loudly if a declaration is impossible, so honest errors are caught, not hidden.

---

## 7. After every session (Data crew)

1. Run `python -m perception.record --validate runs/`. It checks every run's files and regenerates `runs/manifest.csv`. Fix or re-record anything it flags.
2. Copy `runs/` to the **shared drive**. Videos are **not** committed to git (they are large); only the small `script.json` files and `runs/manifest.csv` are.
3. Tick the run in `runs/run_plan.csv` (`recorded`, `recorded_by`, `notes`) and commit the small files.
4. Tell P1 which session finished and any `notes` (a lamp got bumped, a box was replaced). **After the pilot: wait for "experiment frozen" before recording more.**

---

## 8. The label check and corrections (Data crew, after P1 sends the review sheets)

P1's tools automatically draw boxes on the frames. **You check them by eye, and fix the ones that are wrong.**
1. Open the contact sheet (`data/review/index.html`). Each page is one object class, showing about 40 randomly chosen frames with the box and class name drawn.
2. In `data/label_review.csv`, for each frame write `ok` or `bad`, and for `bad` one reason: `wrong_box`, `missing`, `duplicate` or `wrong_class`.
3. **Rule of thumb:** if more than **1 in 10** frames of a class is `bad` (a judgment threshold), P1 fixes the labeling settings and sends you a fresh sample. Re-check that sample; do not assume it is fixed.
4. **Static boxes, one frame per run.** Open the static-box sheet. Each run shows one frame with the boxes for the outer box, the tray and the START button. Mark each `ok`, or fix it in the label editor (step 5). One fix covers every frame in that run, so this is the highest-value check.
5. **The label editor** (`data/review/edit_<split>.html`, opens in your browser; nothing to install). Drag a box to move or resize it. Press a number key to change its class (the page shows which number is which class). Draw a box that is missing. Delete a wrong one. Press *Exclude* to drop a hopeless frame, and *Confirm* when a frame is right. When you finish a page, press *Download corrections* and put the file in `data/corrections/`. Your corrections are stored separately; the original boxes are never overwritten.
6. **Gold frames (val and test).** P1 marks about 6 frames per val and test run for you to verify. They are used to score the AI honestly, so they must be right: check every box, fix it, then confirm. **For test frames, correct from the image alone; do not look at what the AI predicted.**

---

## 9. What happens to your recordings (so you can spot problems)

1. **Validate** (you). 2. **Sample frames**: about 2 per second, dropping near-identical frames. 3. **Auto-label**: a zero-shot detector draws boxes from text prompts ("red box", "green button"). Fixed objects are smoothed. 4. **You check and correct** boxes (section 8). 5. **Train** the RF-DETR detector on `train` frames. 6. **Cache** the AI's per-frame output for every run. 7. **Tune** thresholds on `val`. 8. **Score** on `test`. Full detail: `IMPLEMENTATION_PLAN.md` Part 5.7.

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| Colours look washed out or drifting | Lock exposure and white balance; remove direct sunlight; re-check the red vs yellow boxes |
| A box is hard to see in the preview | More light, a plainer surface, or a box of a different colour. Tell P1 |
| The preview cuts an object off | Lower or re-aim the camera, then **new `camera_setup_id`** |
| The performer's hand leaves the frame a lot | Re-tape the home positions closer to the centre |
| A run went wrong | Section 6: declare what really happened, or discard and redo |
| The tool says the video is unreadable | Re-record; check free disk space and the camera cable |

## 11. Contacts

**P1** (recording tool, labels, detector) for anything about recording or review. **P2** (results, reports) for PPT numbers.
