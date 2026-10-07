# A6 hardware replay — step-by-step for Adhi

Read this once through end-to-end before you start anything — some steps take hours so you want to plan when to kick them off.

---

## Step 1 — Download the file

Click this Drive link:

<https://drive.google.com/file/d/11NyA8W231guWzHUD9lUL5zLBjuAOJasy/view?usp=sharing>

Click the download button (arrow icon at the top of Drive). It's 3.8 GB, so on a normal home WiFi it'll take between 30 minutes and 2 hours depending on your connection. Kick this off before you go do something else — you don't need to sit and watch it.

You'll end up with a file called `physiomio_bundle.zip` in your Downloads folder (`C:\Users\adhir\Downloads\physiomio_bundle.zip` or wherever your browser puts downloads).

---

## Step 2 — Unzip it into the ExoHand folder

This is the important bit — you need to unzip it **into your ExoHand folder** specifically, not somewhere else. The zip file already knows where everything goes; you just have to point it at the right base folder.

Right-click `physiomio_bundle.zip` → **Extract All…**. When it asks where to extract to, type or browse to:

```
C:\Users\adhir\ExoHand
```

Click **Extract**. Wait a minute or two while it unpacks.

---

## Step 3 — Check the files landed in the right place

Open File Explorer and navigate to `C:\Users\adhir\ExoHand\data\physiomio\data\`. You should see 48 folders named `patient1`, `patient2`, …, `patient48`, plus a file called `metadata.csv`. If you see those, you're good.

Also check that `C:\Users\adhir\ExoHand\data\physiomio_channel_picks.csv` exists. It's a tiny file, ~8 KB.

If either isn't there, tell me before doing anything else — the extraction went to the wrong place.

---

## Step 4 — Make sure the Teensy is set up

Before you kick off the big run:

1. Plug the Teensy in via USB
2. The Teensy should still have the `TEST_MODE = 1` firmware on it — the one you verified last week. If you've flashed something else since, re-flash the TEST_MODE version.
3. Confirm Windows sees it as COM5. If Device Manager shows it as a different COM port, note the number (you'll swap it into the command in the next step).

---

## Step 5 — Kick off the big run

Open PowerShell (Start menu → type "PowerShell"). In the terminal:

```powershell
cd C:\Users\adhir\ExoHand
python analysis\revision\T1_hardware_replay\run_T1_all_patients.py --port COM5
```

If your Teensy is on a different port, change `COM5` to whatever it shows up as.

You'll see the terminal start printing lines like:

```
[1/48] patient1: 2100 windows replayed  [0.2min, eta 12min]
[2/48] patient2: 1980 windows replayed  [0.4min, eta 11min]
...
```

---

## Step 6 — Let it run

The full 48-patient run takes about 3 to 4 hours. The easiest thing is to kick it off before you go to bed, leave the laptop plugged in with the Teensy plugged in, and wake up to a completed run.

A few things to know while it runs:

- **Don't unplug the Teensy** mid-run.
- **Don't close the PowerShell window.**
- **Windows might want to sleep.** Go to Settings → System → Power & battery → set "Screen and sleep" to "Never" while plugged in, then set it back after.
- **If the run errors out mid-way** for any reason (Teensy disconnected, laptop rebooted, whatever), don't panic — just re-run the exact same command. It picks up from wherever it left off. Nothing has to be redone.

---

## Step 7 — When it finishes

When the run completes, the terminal will print something like `Wrote analysis\revision\results\T1_deployed_accuracy_per_patient.csv` and stop.

Two output files will now exist:

- `C:\Users\adhir\ExoHand\analysis\revision\results\T1_deployed_accuracy_per_patient.csv` — small, ~50 KB
- `C:\Users\adhir\ExoHand\analysis\revision\results\T1_deployed_stream_per_window.parquet` — bigger, maybe 50-100 MB

Send me both. Easiest way: commit them to git and push.

```powershell
cd C:\Users\adhir\ExoHand
git add analysis\revision\results\T1_deployed_accuracy_per_patient.csv analysis\revision\results\T1_deployed_stream_per_window.parquet
git commit -m "A6 hardware replay results, 48 patients"
git push origin main
```

---

## Step 8 — Safety: revert the Teensy firmware to normal mode

**This is critical, don't skip it.** The Teensy is still in `TEST_MODE = 1`, which means it's expecting data over USB and won't work if anyone tries to wear it. Before doing anything else with the Teensy:

1. Open `teensy_emg\teensy_emg.ino` in the Arduino IDE
2. Find the line `#define TEST_MODE 1`, change the 1 back to 0 so it reads `#define TEST_MODE 0`
3. Save the file
4. Upload to the Teensy (same steps as before — Board = Teensy 4.0, Port = whichever COM, click Upload arrow)
5. Wait for "Done uploading"

---

## Step 9 — Live gesture check to confirm the revert worked

Put the MyoWare sensors on your own forearm, plug the Teensy in. Do one clear hand-close gesture (make a fist) and one clear hand-open gesture (spread your fingers). Watch the exoskeleton — both should be detected correctly and the exoskeleton should respond as normal.

- **If it responds correctly** → the Teensy is back in normal mode, safe to hand back / store / demo.
- **If it doesn't respond** → the revert didn't take. Re-upload the firmware and try again. If it still doesn't work, tell me before doing anything else.

---

## Step 10 — Log that you did the revert

Create a text file at `C:\Users\adhir\ExoHand\analysis\revision\T1_hardware_replay\raw\revert_log.txt` with something like:

```
Revert check after A6 hardware replay
Subject: Adhi (own arm)
Sensor placement: normal MyoWare positions
Gestures tried: 1x close (fist), 1x open (spread fingers)
Result: both detected correctly, exoskeleton responded normally
Status: SAFE — Teensy confirmed back in TEST_MODE 0
```

Commit and push that too:

```powershell
git add analysis\revision\T1_hardware_replay\raw\revert_log.txt
git commit -m "A6 safety revert confirmed"
git push origin main
```

---

## That's it — you're done with A6

If anything breaks, errors out, or looks weird at any step: **stop, take a screenshot of the terminal, and send it to me before trying to fix it yourself.** Don't reflash or reboot the Teensy in the middle of a run unless the run has already crashed.
