# Adhi — Hardware Handoff for NeurIPS Submissions

**Two submission deadlines:**
- **ICBINB** — Aug 29 (6 days)
- **TS-LIMITS** — Sep 5 (13 days)

Both are being submitted. Full deployment work in scope. Six tasks below, prioritised. Read the "What you're doing and why" bit for each before touching anything.

---

## Priority order

1. **A1** — new hardware photos + BOM (must-have for ICBINB)
2. **A2** — demo video trim (must-have for ICBINB)
3. **A3** — measured end-to-end latency (must-have for TS-LIMITS, nice-to-have for ICBINB)
4. **A6/T1** — hardware-in-the-loop stroke replay (the big TS-LIMITS artifact)
5. **A4** — live closed-loop on n=3 healthy
6. **A5** — Teensy resource census

Any item that doesn't ship by its deadline just gets dropped or downgraded. Don't skip 1-3 to spend all your time on 4-6.

---

## A1 — New hardware photos + BOM update

**Deadline:** Aug 26
**Effort:** ~2 hours
**Why:** ICBINB reviewers judge deployment claims on visible evidence. The paper says "£180 embedded system on real hardware" — we need photos and an itemised BOM to back that.

**What you deliver:**
- 2-3 paper-quality photos of the new hardware:
  - Front view worn on a person (arm-only frame — anonymised)
  - Top-down showing the sensor placement
  - Optional: close-up showing the Teensy + wiring
- Well-lit, plain background, sharp focus. Phone camera is fine if lighting is good
- Updated BOM if any parts changed (new servo, different 3D-printed parts, etc.). If unchanged, confirm the £180 line items are still accurate

**Where to put it:** `report_figures/hardware_v2/` — create folder, drop photos as `.png`. BOM update as `analysis/system/cost_itemization.md` (edit existing file).

---

## A2 — Demo video trim

**Deadline:** Aug 26
**Effort:** ~2 hours
**Why:** Existing demo video (`ExoHand Demo Final/`) is a product-style pitch that will read as "startup demo" to academic reviewers. Needs to be trimmed to a scientific artifact.

**What you deliver:**
- Cut to 60-90 seconds max
- Arm-only framing — no faces (anonymisation for double-blind review)
- Remove UI/login screens; go straight to the closed-loop demo (hand gestures + exoskeleton actuation)
- Silent or with plain instrumental (no voiceover selling the product)
- Add one text card at start: *"End-to-end closed-loop demonstration. Healthy adult wearing the exoskeleton; classifier output drives actuation during cued gestures."*
- Export as `.mp4`, keep it under 25 MB (OpenReview supplementary limit)

**Where to put it:** `report_figures/hardware_v2/demo_trimmed.mp4`.

**Tool:** iMovie or Final Cut on Mac. Free.

---

## A3 — Measured end-to-end latency

**Deadline:** Aug 28 (soft) / Sep 3 (hard, for TS-LIMITS)
**Effort:** ~3-5 hours
**Why:** The paper currently reports ~275 ms end-to-end latency as a *component-sum estimate* (Teensy sample window + software prediction + servo slew, added together). That's the weakest sentence in the deployment section — a TS-LIMITS reviewer will ask "did you actually measure it?" We need a real stopwatch number.

**What you deliver:**
- 20-50 trials, each measuring: **time from EMG stimulus onset → visible motor motion**
- Report as: mean, median, 5th percentile, 95th percentile, max
- Break down per stage if possible

**Two implementation options (pick the one you can do):**

### Option A3a — Phone at 240 fps
Cheapest. Point a phone camera (iPhone slo-mo at 240 fps) at the setup. Have the subject do a sudden gesture with a visible cue (finger tap, or the classifier's "close" indicator light if you add one). Frame-by-frame in the video app, count frames from cue → exoskeleton visibly starts moving. Multiply by 4.17 ms per frame at 240 fps.

- Trials: 20 gestures, log each
- Precision: ~4 ms per measurement (limited by frame rate)
- Total effort: 1 hour recording + 1 hour frame counting

### Option A3b — Logic analyser tap
Cleaner if you have one. Tap two signals: (i) EMG input crossing a threshold, (ii) servo PWM changing. Difference = end-to-end latency. Automated, precise to microseconds.

- If you own a Saleae Logic or similar, use this
- If not, skip to A3a

**Where to put it:** `analysis/system/results/latency_measured.md` — table of measurements + summary stats. Update `analysis/system/HARDWARE_LATENCY.md` to reference the measured number (keep the old component-sum for comparison).

---

## A6/T1 — Hardware-in-the-loop stroke replay

**Deadline:** Aug 30 - Sep 2 (for TS-LIMITS)
**Effort:** ~6-10 hours (spread over 2 days)
**Why:** Converts "we simulated deployment on stroke data" into "we replayed real stroke data through the actual deployed Teensy hardware and got X accuracy on 58 patients." That's the highest-value TS-LIMITS artifact. Also unlocks stream metrics (event F1, transition latency, false-activation rate) on stroke data.

**What you deliver:**
- Modified Teensy firmware with a `TEST_MODE` flag
- Verified that the firmware works: send known test samples in, verify P-P output matches expected
- Physical Teensy plugged into laptop, ready for the injection script to run against it

**I'll write the host-side injection scripts (`inject_and_capture.py`, `run_T1_all_patients.py`). You just need to get the firmware working.**

### Firmware modification

The current `teensy_emg/teensy_emg.ino` reads samples from `analogRead()`. Add a compile-time flag that switches to reading samples from serial instead.

I've drafted the modified firmware — see `teensy_emg/teensy_emg.ino` (I've committed the diff). The change is:

1. Add `#define TEST_MODE 0` at top (keep at 0 for production; set to 1 when running T1)
2. Inside `loop()`, wrap the sampling code in `#if TEST_MODE / #else / #endif`
3. In test mode: read 100 samples × 4 channels × 2 bytes each from serial per 50 ms window; compute P-P as normal; emit as normal

### Interface contract (this is the API between your firmware and my host script — don't change without telling me)

**Host → Teensy per 50 ms window:**
- Exactly **800 bytes**
- 100 samples × 4 channels × 2 bytes per sample
- Order: `[sample0_ch0, sample0_ch1, sample0_ch2, sample0_ch3, sample1_ch0, sample1_ch1, ...]`
- Endianness: **little-endian int16**
- Value range: 0-4095 (12-bit ADC range, matches Teensy 4.0 analogRead default)

**Teensy → Host per 50 ms window:**
- One text line: `pp_ch0\tpp_ch1\tpp_ch2\tpp_ch3\n`
- Values: unsigned ints, 0-4095

### Verification steps

Before we run the full 58-patient replay:

1. Compile firmware with `TEST_MODE 1`, upload to Teensy
2. Manually send 800 zero-bytes (or use a quick Python script). Teensy should reply `0\t0\t0\t0\n`
3. Send 800 bytes representing a known ~2 kHz sinewave of amplitude 500. Teensy should reply `1000\t1000\t1000\t1000\n` (peak-to-peak of a sinewave of amplitude 500 is 1000)
4. If both check out, we're go for the full run

**After T1 runs:** flash the production firmware back (with `TEST_MODE 0`) before any real patient contact. Non-negotiable.

---

## A4 — Live closed-loop on n=3 healthy

**Deadline:** Sep 1
**Effort:** ~4 hours (1 hour per subject + processing)
**Why:** Current live-deployment claim uses n=1 (you). Upgrading to n=3 (you + Ansh + Yash) doesn't validate on stroke patients but honestly upgrades the demo evidence and preempts "your live number is n=1" reviewer comments.

**What you deliver:**
- Run the same 12-minute cued session protocol on 3 healthy subjects (you + Ansh + Yash)
- Save the recordings + trained models + evaluation to `analysis/system/results/live_deployment_n3/`
- Update `analysis/system/live_deployment_eval.py` to iterate over 3 subjects and aggregate
- Report: mean within-session accuracy across 3 subjects, per-subject numbers

**Same protocol as before:**
- 12-minute cued session, 3-class (rest / close / open)
- Same MyoWare placement, same Teensy, same calibration protocol
- Same held-out evaluation

---

## A5 — Teensy resource census

**Deadline:** Sep 1
**Effort:** ~2-3 hours
**Why:** TS-LIMITS is a "tight settings" workshop. Reviewers want raw resource numbers to prove the deployment claim.

**What you deliver:**
Report the following in `analysis/system/results/teensy_resource_census.md`:

- **Model flash size (KB)**: how big is the pickled HGB when serialised for the deployed pipeline?
- **Inference time on-device (ms)**: if we ran the model ON the Teensy (we don't currently — inference is on host), what would inference take? If we can't measure this because HGB doesn't fit in Teensy RAM, state that with the memory budget.
- **RAM peak (KB)**: measure Teensy free memory during operation
- **Current draw (mA)**: use a USB power meter (~£10 on Amazon if we don't have one) — measure current during sustained inference vs idle
- **Estimated battery life**: given current draw and a battery capacity (assume a standard 3000 mAh LiPo), how many hours of continuous operation?

Some of these you can look up (flash size from datasheet), some you need to measure (current draw). Do what you can, note what you couldn't.

---

## Overall time budget

If you work concentrated for the next 12 days:

| Item | Deadline | Hours |
|---|---|---:|
| A1 photos + BOM | Aug 26 | 2 |
| A2 video trim | Aug 26 | 2 |
| A3 latency | Aug 28 | 3-5 |
| A6/T1 firmware | Aug 30 | 2 |
| A6/T1 run + support | Aug 30 - Sep 2 | 3-5 |
| A4 n=3 live | Sep 1 | 4 |
| A5 resource census | Sep 1 | 2-3 |
| **Total** | | **~18-23 hours over 12 days** |

Roughly 2 hours a day. Front-load A1/A2 for the ICBINB deadline.

---

## What Ansh is doing in parallel

- ICBINB paper prose (~10 hrs)
- TS-LIMITS paper reframe (~5 hrs)
- Host-side scripts for T1 (~4 hrs) — the injection script and 58-patient driver
- Stream metrics analysis (T4) from T1 output (~2 hrs)
- Sizing comparison table for TS-LIMITS (~1 hr)
- Figure production (~3 hrs)

We overlap only on the T1 interface (spec above). Everything else is independent.

---

## Communication cadence

- Post updates in whatever channel we're using each morning
- If any deadline slips or you hit a blocker, flag it same-day
- If the T1 firmware won't compile / behave, tell me by Aug 28 latest so I can pivot to Approach B-lite (software-only replay through deployed post-processing)

---

## The one thing that must be right

**Never flash TEST_MODE=1 firmware onto a device that will be used on a real person.** If it accidentally gets there and someone connects, the classifier will process garbage samples and might issue unpredictable motor commands. Fine for the T1 replay rig sitting on your desk; not fine for anything else. Always revert to `TEST_MODE 0` before any patient / subject use.
