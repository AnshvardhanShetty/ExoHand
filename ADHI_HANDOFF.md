# Adhi — Hardware Handoff

Six things to do, in this order. Do the earlier ones first — they matter most. If you run out of time, just skip the later ones.

Before you start each task, read the "Why" and "What you make" bits so you know what "done" looks like.

---

## Save everything, always

**The single most important rule across all six tasks: save every single measurement, every raw file, and every intermediate result. Nothing gets thrown away.**

For every task you do, we need enough saved that someone else — or us in a month — could re-run the whole experiment from scratch and reproduce exactly what you got.

That means:

- Every individual measurement, not just the summary numbers (all 20 latency trials, not just the mean; every person's raw recording, not just the aggregated result)
- Every raw file the measurement came from (raw video files, raw sensor recordings, logic analyzer captures, power meter logs, photos before any editing)
- Every intermediate file (edited videos, processed data, trained models, calibration files)
- Every setup detail (which sensor was on which pin, room lighting, subject position, any weird thing that happened during a run — write it down in a `notes.md` next to the data)
- The exact version of any code or firmware you used (commit it to git, or save the `.ino` file alongside the data)

**Rule of thumb: if it took you time to produce it, save it. Don't ever throw a measurement away because "we already computed the average."** Averages hide bugs and outliers; the raw data is where the truth lives.

Every task below has a "Where it goes" line. That's the location for the finished artifact. Alongside it, always save a `raw/` subfolder with everything you started from, and a `notes.md` describing what you did.

---

## Order to do things in

1. **A1** — take new photos of the hardware and check the parts list
2. **A2** — cut down the demo video
3. **A3** — actually measure how long the system takes to react
4. **A6/T1** — set up the Teensy so we can feed it recorded data instead of live signals
5. **A4** — record three healthy people using the exoskeleton
6. **A5** — write down what the Teensy uses (memory, power, etc.)

---

## A1 — New hardware photos + parts list check

**Why:** The paper says we built a working device for £180. The people reading it will want to see it. Right now we only have old pictures.

**What you make:**

- Two or three sharp photos of the current hardware:
  - One with someone wearing it (just their arm in the frame — no face)
  - One from above showing where the sensors sit on the arm
  - Optional: a close-up of the Teensy and its wires
- Use good light and a plain background. A phone camera is fine if the room is bright.
- Look at the parts list (`analysis/system/cost_itemization.md`). If anything on it has changed — new servo, different 3D prints, whatever — update it. If nothing has changed, just tell me it's still right.

**Where it goes:**

- Final photos → `report_figures/hardware_v2/`, save as `.png`
- **Raw photos (straight out of the camera, before any cropping/editing)** → `report_figures/hardware_v2/raw/`
- Parts list → edit `analysis/system/cost_itemization.md` in place
- `notes.md` alongside the photos → what setup, what light source, which parts version, anything about the current build that's different from the old one

---

## A2 — Cut the demo video down

**Why:** The current demo video (`ExoHand Demo Final/`) is styled like a startup product pitch. That looks wrong in an academic paper. It needs to be short and plain so people take the science seriously.

**What you make:**

- A shorter video, no longer than about a minute and a half
- Only the arm should be visible — no faces (the review process is anonymous)
- Cut out any login screens or app interface. Just show the hand doing gestures and the exoskeleton moving in response
- No voiceover. Silent, or with quiet instrumental music
- Add one text card at the very start that says:
  > *"End-to-end closed-loop demonstration. Healthy adult wearing the exoskeleton; classifier output drives actuation during cued gestures."*
- Save it as an `.mp4`, keep the file size under 25 MB (there's an upload limit)

**Where it goes:**

- Trimmed video → `report_figures/hardware_v2/demo_trimmed.mp4`
- **Raw uncut video (the original, untouched)** → `report_figures/hardware_v2/raw/demo_original.mov`
- Editing project file (iMovie project or whatever you used) → `report_figures/hardware_v2/raw/demo_edit_project/`
- `notes.md` — what you cut out, what you kept, why

**Tool:** iMovie on a Mac. Free.

---

## A3 — Measure how long the system takes to react

**Why:** Right now the paper says the system takes about 275 milliseconds from muscle signal to motor moving. But that number was worked out on paper — we added up the estimated time for each step. Someone reading the paper will ask "did you actually measure it?" and we need a real number.

**What you make:**

Between 20 and 50 attempts, each one measuring:
- Time from when the muscle signal starts → when the exoskeleton visibly starts moving

Then write down the average, the middle value (median), the fastest 5%, the slowest 5%, and the slowest single one.

**Two ways to measure this — pick whichever you can do:**

### Option A3a — Slow-motion phone video

Cheapest and easiest.

- Set an iPhone to record in slow-motion at 240 frames per second
- Point it at the person and the exoskeleton
- Person does a sudden gesture with a clear signal (like tapping their finger, or flashing a light when the classifier fires)
- Play the video back frame-by-frame
- Count the frames from the cue to when the exoskeleton starts moving
- Each frame is about 4 milliseconds. Multiply frames × 4.

### Option A3b — Logic analyzer

Cleaner and more precise, but only if you have one.

- Tap two signals: (1) EMG going over a threshold, (2) servo motor changing position
- The gap between them is your latency
- Automatic and precise

If you don't already have a logic analyzer (a Saleae or similar), just do A3a. Don't buy one.

**Where it goes:**

- Summary + full trial-by-trial table → `analysis/system/results/latency_measured.md`
  - **Every single trial listed as its own row**, not just the summary stats. If you did 30 trials, the table has 30 rows.
- **Raw slow-motion video files (A3a) or raw logic analyzer captures (A3b)** → `analysis/system/results/latency_raw/`
  - For A3a: the actual `.mov` slow-mo files, plus a `frame_counts.csv` showing what frame you counted for each trial
  - For A3b: the `.sal` or `.csv` capture files from the logic analyzer, one per trial
- `notes.md` alongside — describe the setup (which cue you used, how you triggered the gesture, anything that went sideways during a trial)
- Also update `analysis/system/HARDWARE_LATENCY.md` to point at the new measured number (keep the old component-sum estimate there too, so people can compare the two)

---

## A6/T1 — Firmware switch so we can replay recorded data

**Why:** Right now the Teensy only reads real live signals from the sensors. We want to feed it recorded stroke patient data (from a computer, over USB) so we can measure how the deployed system would perform on those patients. That turns "we simulated the deployment" into "we actually ran the deployed hardware on the recorded data."

**What you make:**

- The Teensy firmware, updated with a little switch that lets it read data from USB instead of from the sensors when we tell it to
- A verified working Teensy — meaning you've sent it some known test data and confirmed it responds correctly
- The Teensy plugged into a laptop, ready for me to run my scripts against

**I'll write all the computer-side scripts. You just need to get the Teensy behaving.**

### What actually happens once your firmware is ready

Once your firmware is verified, I'll run a script that streams the recorded EMG from 48 stroke patients through your Teensy — one patient after another, one 50 ms window at a time — catches what the Teensy outputs for every window, and computes how well the deployed system would have performed on each patient's real recorded session.

That takes a few hours to run end-to-end. The Teensy just needs to be plugged in and left alone during the run. No one wears it; no exoskeleton actuates; the sensors don't need to be connected. Just Teensy + laptop + USB cable, sitting on a desk.

Every window's output from the Teensy gets saved (raw per-patient traces + per-window predictions). If the run gets interrupted, we can resume from wherever it stopped — nothing has to be re-done.

### The firmware change

The current file is `teensy_emg/teensy_emg.ino`. I've already made the change — you just need to look at it, understand it, and flash it.

Here's what the change does:

1. At the top there's a line that says `#define TEST_MODE 0`.
2. When it's `0`, everything works normally — sensors → Teensy → output.
3. When it's `1`, the Teensy stops listening to the sensors and instead reads data coming in over the USB cable. It still processes that data the same way (calculates peak-to-peak) and sends the answer back.

Keep it at `0` for normal use. Set it to `1` only when we're doing the replay run.

### The rules for how the two sides talk to each other

**This is the important bit. Don't change these numbers without telling me — my computer scripts assume them exactly.**

**Every 50 milliseconds, the computer sends the Teensy:**

- Exactly **800 bytes**
- That's 100 samples × 4 channels × 2 bytes per sample
- The order is: `[sample 0 channel 0, sample 0 channel 1, sample 0 channel 2, sample 0 channel 3, sample 1 channel 0, ...]`
- Byte order is little-endian, and each value is a 16-bit signed integer
- Values range from 0 to 4095 (this matches what the Teensy's ADC normally produces)

**Every 50 milliseconds, the Teensy sends the computer back one line of text:**

```
pp_ch0<TAB>pp_ch1<TAB>pp_ch2<TAB>pp_ch3<newline>
```

Each value is a whole number between 0 and 4095.

### How to check it's working before the big run

1. Compile the firmware with `TEST_MODE 1` and upload it to the Teensy
2. Send 800 bytes of all zeros. The Teensy should reply `0 0 0 0` (all zeros, tab-separated)
3. Send 800 bytes that look like a 2 kHz sine wave with amplitude 500. The Teensy should reply `1000 1000 1000 1000` (peak-to-peak of a ±500 wave is 1000)
4. If both work, we're ready

**Save everything from the verification too:**

- The exact `.ino` firmware file you flashed (with a git commit hash if you can) → `analysis/revision/T1_hardware_replay/raw/teensy_emg_TEST_MODE.ino`
- The three verification test replies (screenshot or copy the terminal output) → `analysis/revision/T1_hardware_replay/raw/verification_log.txt`
- `notes.md` — Teensy model, Arduino IDE version, which USB port, any weird behaviour

**When we're done with the replay, flash the normal firmware back (with `TEST_MODE 0`) before anyone puts the exoskeleton on. This is non-negotiable — see the safety note at the very bottom.**

---

## A4 — Record three healthy people using the system

**Why:** Right now we have live-deployment data from just one person (you). Getting three people (you + Ansh + Yash) makes the demo evidence stronger and stops reviewers from saying "your live number is based on one person."

**What you make:**

- Run the same 12-minute recording protocol we've been using, on all three people
- Update `analysis/system/live_deployment_eval.py` so it loops over all three and reports the combined result
- Report: average accuracy across the three people, plus each person's individual number

**Use the exact same setup as before:**

- 12-minute cued session with three gestures (rest, close, open)
- Same MyoWare sensor positions on the forearm
- Same Teensy
- Same calibration steps

**Where it goes:**

- Per-subject folder → `analysis/system/results/live_deployment_n3/<subject_name>/`
  - **Raw EMG recording (the full session, every sample)** → `raw_emg.parquet` or whatever format the recorder produces
  - Calibration data (windows used for training the per-session model) → `cal.parquet`
  - Trained model file → `model.pkl`
  - Per-window predictions during the evaluation portion → `predictions.parquet`
  - Accuracy summary → `results.json`
  - `notes.md` — subject id, date (relative — e.g. "session 1"), any issues during the run, sensor placement description
- Combined summary across all three → `analysis/system/results/live_deployment_n3/summary.md`

---

## A5 — Teensy resource check

**Why:** For one of the papers ("tight settings" focus) reviewers will want the actual numbers on what the Teensy uses. So we can prove the deployment claim.

**What you make:**

Write these numbers in `analysis/system/results/teensy_resource_census.md`:

- **How big is the trained model on disk?** In kilobytes.
- **How long would inference take if we ran the model ON the Teensy?** (Right now we run it on a laptop that's connected. If we tried to squeeze the model into the Teensy itself, how long would each prediction take? If the model is too big to even fit in the Teensy's memory, just say that and mention the memory limit.)
- **How much RAM does the Teensy use?** Peak, during normal operation.
- **How much electric current does it draw?** Use a USB power meter (about £10 on Amazon if we don't already own one). Measure two numbers: when it's running predictions vs when it's just idle.
- **How long would it run on a battery?** Assume a standard-sized rechargeable battery (3000 mAh). Given the current draw you measured, how many hours would it last?

Some of these you can look up online (like the model file size, or Teensy datasheet numbers). Some you need to actually measure (like current draw). Do what you can, and note anything you couldn't figure out.

**Where it goes:**

- Summary → `analysis/system/results/teensy_resource_census.md`
- **Raw measurements** → `analysis/system/results/teensy_resource_census_raw/`
  - Power meter readings — every individual sample, not just the average. If the meter can log to CSV, save the log. If not, take a photo of the display for each measurement and save them numbered
  - RAM measurements — the actual Teensy serial output showing free memory over time
  - Any datasheet screenshots or URLs you looked up → `sources.md` with links
- `notes.md` — which power meter you used (model + serial), which battery model you assumed for battery-life estimate, room temperature (affects current draw slightly), anything else

---

## The one thing that must be right — SAFETY

**Never flash the `TEST_MODE 1` firmware onto a Teensy that someone is about to wear.**

If it accidentally gets used on a real person, the Teensy won't be reading their muscle signals anymore — it'll be waiting for data from a USB cable that isn't there, and it'll process garbage. That could make the exoskeleton do unpredictable things on a person's hand.

`TEST_MODE 1` is only ever for the replay rig sitting on your desk, plugged into a laptop with no one wearing it. Before anything else touches a person, always flash back to `TEST_MODE 0`.

If you're ever not sure which version is currently on the Teensy, don't guess — reflash the normal firmware to be safe.

---

## When to check in with me

- If you get stuck on any step, ping me the same day — don't sit on it
- If the firmware won't compile or won't behave, tell me early so I have time to fall back to a different plan
- Send me a short update whenever you finish one of the six items, so I can plan the next things around what's done
