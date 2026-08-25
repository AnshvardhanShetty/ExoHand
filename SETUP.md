# Getting the ExoHand system running on your laptop

This walks you through everything you need to do, from a fresh laptop to
a working system controlling the exoskeleton with real EMG. No prior
knowledge assumed.

When you finish you'll have:

- A browser window at `http://localhost:5173` running the ExoHand web app
- A Teensy microcontroller streaming EMG from the MyoWare sensors
- A Python process classifying intents and sending servo commands
- The servo actually opening and closing the exoskeleton hand

Total time: **30–60 minutes** the first time, ~2 minutes every subsequent
time.

---

## What you need before starting

- The physical device: **Teensy 4.0** flashed with the firmware, **4× MyoWare
  2.0 sensors** on the forearm, servo wired to the tendon-driven exoskeleton.
- A laptop running **macOS**, **Windows**, or **Linux**.
- A **USB-C cable** (or micro-USB, depending on your Teensy) to connect the
  Teensy to the laptop.
- **~2 GB of free disk space** for dependencies.
- An internet connection for the first-time installs.

You do NOT need a GPU, a cloud account, or the physical device to be a
specific version — the firmware auto-detects channels.

---

## Step 1 · Install the three tools you need

You need **git**, **Python 3.8 or newer**, and **Node.js 18 or newer**.

### macOS

Install [Homebrew](https://brew.sh/) if you don't have it, then:

```bash
brew install git python@3.11 node
```

### Windows

Download and run installers, in order:

1. Git for Windows — <https://git-scm.com/download/win>
2. Python 3.11 — <https://www.python.org/downloads/>  
   *IMPORTANT: on the first installer screen, tick **"Add python.exe to PATH"**.*
3. Node.js LTS — <https://nodejs.org/>

### Linux (Ubuntu / Debian)

```bash
sudo apt update
sudo apt install -y git python3 python3-pip nodejs npm
```

### Confirm the three are installed

Open a fresh terminal and run:

```bash
git --version
python3 --version   # or `python --version` on Windows
node --version
npm --version
```

Every command should print a version. If any says "command not found",
close the terminal and reopen it, or reinstall the missing one.

---

## Step 2 · Get the code

Pick a folder on your laptop where you want the project to live (Desktop
is fine). Open a terminal in that folder and run:

```bash
git clone https://github.com/AnshvardhanShetty/ExoHand.git
cd ExoHand
```

You now have a folder called `ExoHand` with the whole codebase in it.
Everything in the rest of this guide assumes your terminal is inside that
folder.

### The two important model files

There are two `.pkl` files at the root of the repo you need:
- `exohand_model.pkl` — the base model trained on GrabMyo (43 healthy subjects)
- `exohand_adapted_model.pkl` — the patient-adapted model (only used if you
  ran calibration for a specific patient)

Both should already be there after `git clone`. Confirm:

```bash
ls -lh exohand*.pkl
```

If you see two files of ~20–40 MB each, you're good. If they're tiny
(under 1 KB) they were downloaded as Git LFS pointers rather than the
real files — install Git LFS and re-fetch:

```bash
# macOS
brew install git-lfs
# Windows / Linux: see https://git-lfs.com

git lfs install
git lfs pull
```

---

## Step 3 · Install the Python dependencies

The runtime that talks to the Teensy is written in Python. Install its
dependencies:

```bash
pip3 install -r requirements.txt
```

*(On Windows use `pip install ...` — no `3`.)*

This grabs `numpy`, `scipy`, `scikit-learn`, `joblib`, `pyserial`, and
`pandas`. Takes 30 seconds to a couple of minutes depending on your
connection.

When it's done, verify:

```bash
python3 -c "import numpy, scipy, sklearn, joblib, serial; print('ok')"
```

If you see `ok` you're set.

---

## Step 4 · Install the web app dependencies

There are two Node projects: the **server** (backend) and the **client**
(frontend). Both need their packages installed. From the ExoHand folder:

```bash
cd server
npm install
cd ../client
npm install
cd ..
```

Each `npm install` takes 1–3 minutes and downloads a few hundred MB into
`node_modules/`. Ignore any yellow "warnings" — only red errors matter.

---

## Step 5 · Flash the Teensy firmware

The Teensy needs the ExoHand firmware. If it was already flashed by the
previous person you skipped this step, otherwise:

1. Install the **Arduino IDE** — <https://www.arduino.cc/en/software>
2. Install **Teensyduino** (adds Teensy support to Arduino IDE) —
   <https://www.pjrc.com/teensy/td_download.html>
3. Open the Arduino IDE, then open the file
   `exohand_combined/exohand_combined.ino` from this repo.
4. Top of window: select **Tools → Board → Teensyduino → Teensy 4.0**.
5. Plug the Teensy into your laptop with a USB cable.
6. Select **Tools → Port → [the one that appeared when you plugged it in]**.
7. Click the **Upload button** (right-arrow icon).

You should see "Done uploading" at the bottom.

---

## Step 6 · Plug everything in

1. Connect the **4 MyoWare sensors** to the Teensy analog pins (per the
   pinout diagram in `teensy_emg/` — should already be wired).
2. Connect the **servo** to the Teensy PWM pin and power.
3. Put the sensors on your forearm — 2 on flexor side, 2 on extensor side.
4. Plug the Teensy into the laptop via USB.

---

## Step 7 · Find the Teensy's serial port name

Every OS calls the Teensy something different. In a terminal:

### macOS

```bash
ls /dev/tty.usbmodem*
```

You'll see something like `/dev/tty.usbmodem176627901`. Copy the full
path — you'll need it in step 8. If you see nothing, unplug/replug the
Teensy.

### Linux

```bash
ls /dev/ttyACM*
```

You'll see something like `/dev/ttyACM0`.

### Windows

Open **Device Manager → Ports (COM & LPT)** and look for the Teensy
entry. It'll be something like `COM3`.

---

## Step 8 · Start the services

There are two entry points into the deployed pipeline, and they're **alternatives, not layers** — pick one, don't run both. Only one process at a time can hold the Teensy's serial port.

- **Path A — React web app (recommended)**: Node backend + Vite client. Serves the React UI at http://localhost:5173. Includes the "Start 22s Test (Beta)" button and all the therapist/patient workflows. Node owns the serial port and spawns the Python calibration script on demand.
- **Path B — Legacy Python UI**: `run_exohand.py --web` runs standalone and serves a simpler UI at http://localhost:8000. Python owns the serial port.

**Do not run both paths at the same time** — they'll fight for the serial port and the second one to start will crash with "port in use" (macOS/Linux) or hang (Windows).

The setup below is for **Path A**. If you specifically want the legacy Python UI, skip to the very bottom of this section for Path B.

---

### Path A · Two terminals: Node backend + Vite client

Open **two terminal windows/tabs** in the `ExoHand` folder.

### Terminal 1 · Backend server

The backend needs to know which serial port the Teensy is on, so it can spawn calibration processes with the right port. **Set the `SERIAL_PORT` env var before running the server**, using the same port name from step 7.

**macOS / Linux:**

```bash
cd server
SERIAL_PORT=<YOUR_PORT> npm run dev
```

**Windows PowerShell:**

```powershell
cd server
$env:SERIAL_PORT="<YOUR_PORT>"
npm run dev
```

**Windows cmd:**

```cmd
cd server
set SERIAL_PORT=<YOUR_PORT>
npm run dev
```

Wait until you see:

```
[SERIAL] Using port <YOUR_PORT>
ExoHand server running on http://localhost:3001
WebSocket available on ws://localhost:3001
```

If you see `[SERIAL] No SERIAL_PORT set — running without hardware (simulation mode)`, the env var wasn't picked up. Login and dashboard will still load, but **calibration will fail** because the Node server won't know which port to hand to the Python calibration script. Kill the server (Ctrl+C), set the env var, restart.

**Keep this terminal open.**

### Terminal 2 · Web client

```bash
cd client
npm run dev
```

Wait until you see:

```
VITE v6.x.x  ready in Xms
Local: http://localhost:5173/
```

**Keep this terminal open.**

---

### Path B (alternative) · Legacy standalone Python UI

Skip this if you're already running Path A above.

If you want the older self-contained Python UI (no React app, no Node backend), open **one terminal** and run:

```bash
python3 runtime/run_exohand.py --port <YOUR_PORT> --model exohand_model.pkl --web
```

Wait for `Connected. Streaming.` and open <http://localhost:8000> in a browser.

**Important:** if you already have the Node server (Path A Terminal 1) running, kill it first. Both processes try to open the same serial port and only one can hold it at a time. On Windows this manifests as a hang or a "port in use" error; on macOS/Linux you'll see the same error more clearly.

---

## Step 9 · Open the browser

Go to <http://localhost:5173> in Chrome, Firefox, or Safari.

You should see the ExoHand landing page. Scroll down to the **"Try
Simulation"** button — but ignore that, we're using the real device.
Click **"Login"** in the top-right.

### First-time login

The default PINs are:
- `0000` — logs in as a demo patient
- `9999` — logs in as a therapist

(If you added a real patient earlier those PINs also work.)

### Fastest way to test the setup end-to-end (recommended for first run)

Use **`0000` (patient mode)** — it drops you straight into the calibration flow with no navigation needed.

1. Enter PIN `0000` and press Enter
2. You'll land on a "Session Calibration" screen with a single **Start Calibration** button
3. Click it and follow the cues (rest → close → open, ~90 seconds)
4. Once calibration finishes, you'll be taken to the session screen
5. Move your hand — the classifier prediction should update in the top bar and the exoskeleton should follow. **Rest = stay still, close = squeeze, open = extend fingers.**

If this works, the whole pipeline is wired correctly.

### If you logged in as therapist (`9999`) instead

Therapist mode lands you on a dashboard with a sidebar (Dashboard / Patients / Add Patient), not directly on calibration. To get to a calibration screen from there:

1. Sidebar → **Patients**
2. If the list is empty, sidebar → **Add Patient** and create one (any name works — e.g., "Test Patient")
3. Click on the patient you just created (or an existing one) to open their detail page
4. On the patient detail page, find and click the **Calibrate** action
5. You'll now see the calibration mode chooser with three stacked buttons: **Start Full Calibration**, **Start Quick Calibration**, **Start 22s Test (Beta)**
6. Click whichever mode you want to test

Therapist mode gives you access to Full and the 22s Beta protocols; patient mode only ever runs Quick.

### If nothing shows up

- Check terminal 1 (Python): should say `Starting web server at http://localhost:8000` or similar. If it errored, you'll see a traceback.
- Check terminal 2 (Node server): should say `ExoHand server running on http://localhost:3001`. If it says `[SERIAL] No SERIAL_PORT set — running without hardware (simulation mode)`, set the env var before starting Node (see Step 8).
- Check the client terminal (Vite): should say `Local: http://localhost:5173/`. If it's not running, `cd client && npm run dev`.

---

## Troubleshooting

### "Waiting for connection..." on the calibration screen

The web app can't talk to the backend. Check terminal 2 (server) —
should say "listening on port 3001". Restart it if it died.

### Backend says "serial port in use" or won't start

The Python runtime already has the port. That's the correct order —
terminal 1 (Python) grabs the port, terminal 2 (server) then connects
to the Python runtime via a local socket, not to the serial port
directly. If terminal 2 complains about the serial port, kill and
restart terminal 1 first, then terminal 2.

### Servo doesn't move even though EMG is streaming

- Confirm the servo has its own power. USB alone often can't drive it.
- Confirm the state changed in the browser — "REST → CLOSE" should be
  visible. If it stays on REST, the classifier isn't firing (bad model
  or the sensors aren't detecting muscle activity).
- Try a new session with `--assist-level 5` on the Python command to
  make the classifier more sensitive.

### The 3D hand in the browser is stuck

Refresh the page. The 3D viewer uses WebGL — if your browser has WebGL
disabled or your GPU driver is old, the hand just won't render. The
classifier and servo still work; only the visualisation is broken.

### Nothing happens when I click "Start Session"

Look in the browser console (F12 → Console tab). Errors show up in red.
Usually one of:
- Missing an exercise definition — go to the therapist view and add one.
- The Node backend crashed — check the backend terminal.
- The Vite client failed to hot-reload — refresh the browser tab.
- The calibration Python script errored — the Node backend terminal will show its stderr.

### I want to use the deployed public demo instead

That's <https://exohand-demo.vercel.app> (once deployed). It doesn't need
any hardware or setup — just click and go. But it plays back a canned
recording; it isn't running the real classifier against your hardware.

---

## Every time after the first time

The install steps only run once. To use the system on subsequent days (Path A / React app):

1. Plug the Teensy in.
2. Open **two** terminals in the `ExoHand` folder.
3. Terminal 1 — Node backend, with the serial port set:
   - PowerShell: `cd server; $env:SERIAL_PORT="<YOUR_PORT>"; npm run dev`
   - macOS/Linux: `cd server && SERIAL_PORT=<YOUR_PORT> npm run dev`
4. Terminal 2 — Vite client: `cd client && npm run dev`
5. Browser: <http://localhost:5173>

That's it.

If you only want the legacy Python UI at localhost:8000 instead, run only:

```bash
python3 runtime/run_exohand.py --port <YOUR_PORT> --model exohand_model.pkl --web
```

Don't run both — they'll fight for the serial port.

---

## Optional · Testing the paper's 22-second calibration protocol

The shipped system uses a 6-minute initial patient calibration + 30-second per-session re-cal (see the README for why). Our research paper reports numbers on a shorter protocol — a 22-second cued calibration on 4 reps × 3 classes (close/open/rest), matching the PhysioMio dataset's 432-window balanced cal budget.

If you want to try the paper's exact calibration protocol on the live system (useful for validating that the reported protocol works end-to-end on real hardware, or as a shorter alternative for a quick session), you can run it in one of two ways.

### What the paper22s protocol runs

- 12 cued gestures total (4 reps × [close, open, rest], interleaved)
- 1.8 seconds hold + 0.2 seconds transition per gesture
- ~24 seconds total wall-clock (~34 s with the rest baseline at the start)
- 432 balanced training windows at 20 Hz stride (exact match to the paper)

### Option A · From the web UI (easiest)

**Do this the same way you'd start any calibration through the browser:**

1. Follow Step 8 Path A to start the Node backend + Vite client.
2. Open <http://localhost:5173> and log in as the therapist role (the button only appears in therapist mode — patient-facing landing pages still show only the standard session cal).
3. Navigate to the calibration screen for a patient.
4. You'll see three stacked buttons: **Start Full Calibration**, **Start Quick Calibration**, and **Start 22s Test (Beta)**.
5. Click **Start 22s Test (Beta)**. The countdown, cued gestures, progress bar, and completion flow all work identically to the other two modes — only the protocol underneath is different.
6. When it finishes, the trained model is saved to the standard location and the exoskeleton is immediately usable in a session.

### Option B · From the command line (headless)

If you want to run it without the web UI at all — e.g., for scripted testing or on a machine without a browser — use the CLI directly:

```bash
python3 runtime/calibrate_patient.py \
    --web-mode \
    --port <YOUR_PORT> \
    --model exohand_model.pkl \
    --patient-id <YOUR_NAME> \
    --mode paper22s
```

Replace `<YOUR_PORT>` with your Teensy's port (see Step 7) and `<YOUR_NAME>` with any patient identifier string.

### What it produces (either option)

- A trained model saved to the same location as any other calibration mode
- JSON progress events on stdout (Option B) or via the web UI (Option A)
- A standard calibration report on completion

After calibration, launch `runtime/run_exohand.py` pointing at the saved model — same as after any other calibration mode — and the exoskeleton runs with the paper's classifier configuration.

### The three calibration modes at a glance

| `--mode` | Duration | What it is | When to use |
|---|---|---|---|
| `full` | ~6 min | Multi-phase patient personalisation (rest baseline, familiarisation, sustained holds, quick contractions, variable effort) | First-time patient registration |
| `quick` | ~90 s | Session re-cal (3 reps × 3 classes, 5 s hold + 4 s rest) | Returning patient, per-session drift correction (default) |
| `paper22s` | ~24 s | Paper's PhysioMio-matched cued protocol (4 reps × 3 classes, 1.8 s hold) | Validating the paper's reported calibration budget on live hardware, or when the shortest possible cued cal is preferred |
