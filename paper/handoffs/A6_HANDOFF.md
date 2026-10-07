# A6 — Hardware Replay of Stroke Patient Data

## What this whole thing is, in plain words

The Teensy is a tiny computer inside the exoskeleton. Normally it listens to the muscle sensors on someone's arm and does its work with those live signals.

For this task, we want to change one small thing: instead of listening to the sensors, we want the Teensy to be able to listen to a **cable** — so I can send it recorded muscle signals from stroke patients (that we saved earlier) and see what it does with them.

Why: this proves the real hardware would have worked on real stroke patients, not just a laptop simulation.

**Your job is only two real things:**

1. Flip a tiny switch in the Teensy's code so it listens to the cable instead of the sensors, and check it's actually working.
2. When we're done, flip the switch back and *check it really did switch back* by wearing the sensors and doing a quick test.

Everything in between (the actual streaming of the 48 patients through the Teensy) is done by my computer scripts. You just plug the Teensy in and leave it running.

---

## The map of what you'll do

| Step | What it is | Sensors on your arm? |
|---|---|:---:|
| 1 | Open the code, flip one line from `0` to `1` | No |
| 2 | Send the code to the Teensy over USB | No |
| 3 | Read this section — the rules for how the Teensy and my script talk (don't change any of these) | No |
| 4 | Run a quick test script — sends three test signals in, checks the Teensy responds correctly | No |
| 5 | We agree in writing what "success" looks like *before* the big run — so we can't cheat later | No |
| 6 | I run the big script that sends 48 patients' data through the Teensy over a few hours. You just leave the Teensy plugged in. | No |
| 7 | Afterwards: flip the switch back to `0`, send the code to the Teensy again, **wear the sensors, do one hand-close and one hand-open**, check the exoskeleton responds. Write down that you did this. | **Yes** |

That's it. Six of the seven steps you can do at your desk with no sensors on you. Only Step 7 needs you to actually wear the sensors.

---

## Step 1 — Flip the switch in the code

The Teensy's code lives in a file called:

```
teensy_emg/teensy_emg.ino
```

Open it. Near the top (around line 15) you'll see this line:

```c
#define TEST_MODE 0
```

That `0` means "normal mode — listen to the sensors."

Change it to:

```c
#define TEST_MODE 1
```

That `1` means "test mode — listen to the cable instead."

**Save the file.** That's the entire code change. I already wrote everything else — you're just flipping the number.

---

## Step 2 — Send the code to the Teensy

1. Open the **Arduino IDE** program (already installed on your Mac)
2. Open the file `teensy_emg/teensy_emg.ino` in it
3. Plug the Teensy into your Mac with a USB cable
4. At the top menu: **Tools → Board → Teensyduino → Teensy 4.0**
5. At the top menu: **Tools → Port →** click the one that looks like `tty.usbmodem12345` (the number after `usbmodem` will vary — pick the one that's there now that wasn't there before you plugged in)
6. Click the **arrow button** at the top left (this uploads the code)
7. Wait until the bottom of the window says **"Done uploading"**

If the Arduino IDE doesn't have "Teensy 4.0" as an option, install Teensyduino first from this page: <https://www.pjrc.com/teensy/td_download.html>. Then come back to step 4.

---

## Step 3 — The rules for how the two sides talk (DO NOT CHANGE)

**This is the most important thing to not touch.** My computer script and your firmware have to agree exactly on what data flows in which direction. If either side changes and the other doesn't, everything breaks silently — the numbers will look wrong and we won't know why for hours.

You don't have to *do* anything in this step. Just don't change any of these numbers if you're editing the firmware.

**Every 50 milliseconds, the computer sends the Teensy:**
- Exactly **800 bytes**
- Which is: 100 samples × 4 channels × 2 bytes per sample
- In this order: `[sample 0 channel 0, sample 0 channel 1, sample 0 channel 2, sample 0 channel 3, sample 1 channel 0, ...]` and so on
- The two bytes for each sample are written little-endian (low byte first, high byte second)
- Each sample is a number between 0 and 4095

**Every 50 milliseconds, the Teensy sends the computer back:**
- One line of text like this: `1234<TAB>567<TAB>890<TAB>2345<newline>`
- Four numbers, separated by tabs, ending with a newline
- Each number is between 0 and 4095

---

## Step 4 — Run three quick tests to make sure it works

Before the big run, we send three test signals in and check the Teensy responds correctly.

I already wrote the script. Run it like this in a terminal:

```bash
python3 analysis/revision/T1_hardware_replay/inject_and_capture.py \
    --verify --port /dev/tty.usbmodem12345
```

Replace `12345` with whatever your Teensy's port is (the one you picked in Step 2, item 5).

The script runs three tests automatically:

**Test 1 — All zeros**
- We send 800 bytes of zero
- Teensy should reply: `0<TAB>0<TAB>0<TAB>0<newline>`
- Why: if the signal is flat at zero the whole time, the "highest minus lowest" is zero

**Test 2 — All the same value**
- We send 800 bytes representing a flat value of 2048 (mid-range)
- Teensy should reply: `0<TAB>0<TAB>0<TAB>0<newline>`
- Why: same as above — if the signal never changes, highest minus lowest is zero

**Test 3 — A known wiggly signal (sine wave)**
- We send 800 bytes representing a signal that wiggles up and down by 500 units
- Teensy should reply: `1000<TAB>1000<TAB>1000<TAB>1000<newline>` (within 5 units)
- Why: if the signal wiggles by ±500, the highest minus lowest is 1000

**If any of the three tests come back wrong, STOP.** Something's broken. Tell me and don't proceed.

**If all three tests pass, we're ready.**

### What to save (even from the tests)

**Everything goes in `analysis/revision/T1_hardware_replay/raw/`.** After you pull the latest commit, this folder should already exist (with a placeholder `.gitkeep` file inside). If it doesn't, create it:

- macOS / Linux: `mkdir -p analysis/revision/T1_hardware_replay/raw`
- Windows PowerShell: `New-Item -ItemType Directory -Force -Path "analysis\revision\T1_hardware_replay\raw"`
- Windows cmd: `mkdir analysis\revision\T1_hardware_replay\raw`

Then drop these three files into that folder:

- **verification_log.txt** — the terminal output from the three tests (just copy-paste from Terminal into a new text file)
- **teensy_emg_TEST_MODE.ino** — the exact `.ino` file you uploaded (just copy `teensy_emg/teensy_emg.ino` into this folder, as-is — it captures the version you actually flashed)
- **notes.md** — a plain text file with:
  - Which Teensy model this is (should be Teensy 4.0)
  - Which version of Arduino IDE (Arduino IDE → About)
  - Which macOS/Windows version you're on
  - Which USB port name showed up
  - Anything weird that happened

---

## Step 5 — What "success" means (agreed in writing, NOW)

We write down what "the big run succeeded" means **before** the big run. Not after. This way if something's slightly off we can't fudge the numbers to make it "look fine."

**This is the target we have to hit:**

We pick 5 patients (mix of chronic and acute). We stream at least 5,000 windows through the Teensy from those 5 patients. For every single one of those 5,000 windows, we compare:

- What the Teensy calculated for each channel
- What the software simulation (a Python version of the same math) calculated for the same input

**We need TWO things to both be true:**

1. **The two calculations should never differ by more than 1 unit** on any channel of any window. (They should basically be identical — the software and the Teensy are doing the same math on the same numbers. Allowing 1 unit is just a tiny safety margin for weird edge cases.)

2. **The downstream classifier's decision should be the same at least 99.5% of the time** between the two paths. (Because the classifier is deterministic — same input, same output — this should really be 100%. Allowing 0.5% is a tiny margin.)

**If either one fails on the pre-check subset, we investigate why. We do not lower the target.** These numbers are being written in this document, right now, before any big run happens.

---

## Step 6 — I run the big thing

Once Step 4's tests pass and Step 5's target is met on the pre-check subset, I run this command:

```bash
python3 analysis/revision/T1_hardware_replay/run_T1_all_patients.py \
    --port /dev/tty.usbmodem12345
```

This sends recorded EMG from 48 stroke patients through your Teensy — one patient at a time, one 50 ms window at a time — and saves what the Teensy replies for every single window.

**You don't do anything during this.** Just plug the Teensy in, keep the laptop awake, and let it run. Takes a few hours.

Nobody wears the exoskeleton during this. No sensors need to be connected. The exoskeleton doesn't need to be attached. It's just Teensy + laptop + USB cable, sitting on a desk.

**What gets saved (this is the important bit):**

We save the **raw output** from every single window, from both the Teensy AND from the software simulation. Not just the summary. This way if patient 34 has one weird window, we can go look at exactly which window it was and what happened.

Four files get created in `analysis/revision/results/`:

| File | What's in it |
|---|---|
| `T1_deployed_stream_per_window.parquet` | What the Teensy said, one row per window, all 48 patients |
| `T1_softsim_stream_per_window.parquet` | What the software simulation said on the same inputs |
| `T1_hardware_vs_sim_diff.parquet` | The difference between the two, per window |
| `T1_deployed_accuracy_per_patient.csv` | Summary: how well the Teensy performed on each patient |

**If the run gets interrupted** (laptop restarts, USB unplugs, whatever), just re-run the same command. It picks up where it left off. Nothing has to be redone.

---

## Step 7 — Put the Teensy back to normal AND check it really is

**After the big run finishes, the Teensy is still in test mode (listening to the cable, not the sensors).** If someone puts the exoskeleton on with the Teensy in test mode, the sensors do nothing and the Teensy waits for a cable signal that isn't coming. That could mean the exoskeleton stays still, or does something unpredictable if the input floats.

**You must put it back to normal mode AND check that it worked, before anyone wears it again.** Not one or the other — both.

Do these five things in order:

**1. Flip the switch back.** Open `teensy_emg/teensy_emg.ino` again. Change:

```c
#define TEST_MODE 1
```

Back to:

```c
#define TEST_MODE 0
```

Save the file.

**2. Send the code to the Teensy again.** Same as Step 2 — Arduino IDE, click the arrow, wait for "Done uploading."

**3. Put the sensors on your arm.** Normal placement — the same way you'd wear them for any normal session.

**4. Do the live check.** Do these gestures:
- One clear **hand close** (make a fist)
- One clear **hand open** (spread your fingers)

Watch the classifier / exoskeleton. Both gestures should be detected correctly and the exoskeleton should respond as normal.

**5. Write down that you did the check.** In this file:

```
analysis/revision/T1_hardware_replay/raw/revert_log.txt
```

Write something like:

```
Revert check after replay session 1
Subject: Adhi
Sensor placement: normal
Gestures tried: 1× close, 1× open
Result: both detected correctly, exoskeleton responded normally
Status: SAFE — Teensy confirmed back in normal mode
```

**If Step 4 (the live check) doesn't work** — no response, weird response, opposite response — the Teensy is either still in test mode (upload didn't take) or something else is wrong. **Do not use it.** Go back to Step 1 and try again. If it still doesn't work, tell me immediately.

**Only after the live check works** is the Teensy safe to hand back for demos or real use.

---

## The five rules that must not break

1. **Don't change the rules from Step 3** (byte count, sample order, byte order, reply format). If they drift, everything breaks silently.
2. **Don't move on to the big run if Step 4's three tests fail.**
3. **Don't accept the big run's result if Step 5's target isn't met.** Investigate. Don't lower the target.
4. **Save the raw per-window outputs from both paths (Teensy AND software).** Not just the summary.
5. **Don't consider the Teensy safe to wear until the live check in Step 7 works and is logged.**

---

## Summary of what I do vs what you do

**You:**
- Step 1 — flip switch to `1`
- Step 2 — upload
- Step 4 — run verification script, check three tests pass, save the logs
- (I run Step 6, you just leave the Teensy plugged in)
- Step 7 — flip switch back to `0`, upload, wear sensors, do live check, write it down

**Me:**
- Wrote all the code
- Run the verification script (or you can — takes 30 seconds)
- Run the big 48-patient replay
- Analyse the output for the paper

That's the whole thing.
