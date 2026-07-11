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

## Step 8 · Start the three services (three terminals)

You need three things running at the same time. Open **three separate
terminal windows/tabs**, all in the `ExoHand` folder.

### Terminal 1 · Python runtime

Replace `<YOUR_PORT>` with what you found in step 7 (e.g.
`/dev/tty.usbmodem176627901` or `COM3`):

```bash
python3 runtime/run_exohand.py --port <YOUR_PORT> --model exohand_model.pkl --web
```

Wait until you see:

```
Loading model...
Model loaded (X features)
Connecting to serial port...
Connected. Streaming.
```

If you get a serial-port error, either the port name is wrong (redo step
7) or another process is using it (unplug/replug the Teensy).

**Keep this terminal open.**

### Terminal 2 · Backend server

```bash
cd server
npm run dev
```

Wait until you see:

```
Server listening on port 3001
WebSocket ready
```

**Keep this terminal open.**

### Terminal 3 · Web client

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

Log in, work through the calibration flow, then start a session.

**If everything is wired correctly, moving your hand should move the
exoskeleton.** Rest = stay still, close = squeeze, open = extend fingers.

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
- The Python runtime crashed — check terminal 1.
- The backend crashed — check terminal 2.

### I want to use the deployed public demo instead

That's <https://exohand-demo.vercel.app> (once deployed). It doesn't need
any hardware or setup — just click and go. But it plays back a canned
recording; it isn't running the real classifier against your hardware.

---

## Every time after the first time

The install steps only run once. To use the system on subsequent days:

1. Plug the Teensy in.
2. Open three terminals in the `ExoHand` folder.
3. Terminal 1: `python3 runtime/run_exohand.py --port <YOUR_PORT> --model exohand_model.pkl --web`
4. Terminal 2: `cd server && npm run dev`
5. Terminal 3: `cd client && npm run dev`
6. Browser: <http://localhost:5173>

That's it.
