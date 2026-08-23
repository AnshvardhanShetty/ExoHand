// teensy_emg.ino — EMG P-P envelope sampler for ExoHand deployment.
//
// Emits per-channel peak-to-peak amplitude every 50 ms (20 Hz output rate)
// over USB serial at 115 200 baud.
//
// TWO MODES:
//   TEST_MODE = 0 (default, PRODUCTION): reads real EMG via analogRead()
//   TEST_MODE = 1 (T1 hardware-in-the-loop replay): reads samples via USB serial
//
// SAFETY: NEVER flash with TEST_MODE = 1 for real patient / subject use.
// The device will process arbitrary serial data as EMG, which produces
// unpredictable motor commands. Always revert to TEST_MODE = 0 before
// any live use.

#define TEST_MODE 0   // 0 for production. Set to 1 only for T1 replay.

// Per-window sample count when TEST_MODE == 1.
// 100 samples × 4 channels × 2 bytes = 800 bytes per 50 ms window.
// (Corresponds to a 2 kHz sample rate — matches PhysioMio raw data.)
#define TEST_SAMPLES_PER_WINDOW 100

void setup() {
  Serial.begin(115200);
#if TEST_MODE
  // Give the host time to open the port and start streaming.
  while (!Serial) { ; }
#endif
}

void loop() {
  int maxVal0 = 0, minVal0 = 4095;
  int maxVal1 = 0, minVal1 = 4095;
  int maxVal2 = 0, minVal2 = 4095;
  int maxVal3 = 0, minVal3 = 4095;

#if TEST_MODE
  // ---- T1 REPLAY MODE ----
  // Read exactly TEST_SAMPLES_PER_WINDOW × 4 samples (2 bytes each,
  // little-endian int16) from serial, update per-channel min/max.
  //
  // Byte order per sample-tuple: ch0 lo, ch0 hi, ch1 lo, ch1 hi, ch2 lo, ch2 hi, ch3 lo, ch3 hi
  for (int i = 0; i < TEST_SAMPLES_PER_WINDOW; i++) {
    int v[4];
    for (int c = 0; c < 4; c++) {
      // Block until 2 bytes available for this channel's sample.
      while (Serial.available() < 2) { ; }
      int lo = Serial.read();
      int hi = Serial.read();
      v[c] = lo | (hi << 8);
    }
    if (v[0] > maxVal0) maxVal0 = v[0];
    if (v[0] < minVal0) minVal0 = v[0];
    if (v[1] > maxVal1) maxVal1 = v[1];
    if (v[1] < minVal1) minVal1 = v[1];
    if (v[2] > maxVal2) maxVal2 = v[2];
    if (v[2] < minVal2) minVal2 = v[2];
    if (v[3] > maxVal3) maxVal3 = v[3];
    if (v[3] < minVal3) minVal3 = v[3];
  }
#else
  // ---- PRODUCTION MODE ----
  // Sample analog inputs for 50 ms, track per-channel min/max.
  unsigned long start = millis();
  while (millis() - start < 50) {
    int v0 = analogRead(A0);
    int v1 = analogRead(A1);
    int v2 = analogRead(A2);
    int v3 = analogRead(A4);

    if (v0 > maxVal0) maxVal0 = v0;
    if (v0 < minVal0) minVal0 = v0;
    if (v1 > maxVal1) maxVal1 = v1;
    if (v1 < minVal1) minVal1 = v1;
    if (v2 > maxVal2) maxVal2 = v2;
    if (v2 < minVal2) minVal2 = v2;
    if (v3 > maxVal3) maxVal3 = v3;
    if (v3 < minVal3) minVal3 = v3;
  }
#endif

  // Emit per-channel peak-to-peak amplitude over serial.
  // Format is identical in both modes: "pp0\tpp1\tpp2\tpp3\n"
  Serial.print(maxVal0 - minVal0);
  Serial.print('\t');
  Serial.print(maxVal1 - minVal1);
  Serial.print('\t');
  Serial.print(maxVal2 - minVal2);
  Serial.print('\t');
  Serial.println(maxVal3 - minVal3);
}
