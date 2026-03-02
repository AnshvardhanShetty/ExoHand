/*
 * ExoHand Combined — EMG + Motor on a single Teensy
 *
 * EMG:   4-channel peak-to-peak amplitude (A0, A1, A2, A4)
 *        Sends tab-separated values every ~50ms
 *        Format: "123\t456\t789\t012\n"
 *
 * Motor: Receives "A###\n" commands over serial
 *        Moves servo to angle (110=open, 145=rest, 180=closed)
 *        Responds with "OK:###\n"
 *
 * Both share the same Serial (USB) connection at 115200 baud.
 */

#include <Servo.h>

Servo handServo;

const int SERVO_PIN = 9;
const int LED_PIN   = 13;

// Serial command buffer
char buf[16];
int  bufIdx = 0;
int  currentAngle = 145;  // Start at rest position

void parseSerial() {
  while (Serial.available()) {
    char c = Serial.read();

    if (c == '\n' || c == '\r') {
      buf[bufIdx] = '\0';

      if (bufIdx >= 4 && buf[0] == 'A') {
        int angle = atoi(&buf[1]);
        angle = constrain(angle, 110, 180);
        handServo.write(angle);
        currentAngle = angle;

        Serial.print("OK:");
        Serial.println(angle);

        digitalWrite(LED_PIN, !digitalRead(LED_PIN));
      }

      bufIdx = 0;
    } else if (bufIdx < 15) {
      buf[bufIdx++] = c;
    }
  }
}

void setup() {
  Serial.begin(115200);
  handServo.attach(SERVO_PIN);
  handServo.write(currentAngle);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, HIGH);
  Serial.println("READY");
}

void loop() {
  // ── Tight 50ms EMG sampling window (identical to teensy_emg.ino) ──
  int maxVal0 = 0, minVal0 = 1023;
  int maxVal1 = 0, minVal1 = 1023;
  int maxVal2 = 0, minVal2 = 1023;
  int maxVal3 = 0, minVal3 = 1023;

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

  Serial.print(maxVal0 - minVal0);
  Serial.print('\t');
  Serial.print(maxVal1 - minVal1);
  Serial.print('\t');
  Serial.print(maxVal2 - minVal2);
  Serial.print('\t');
  Serial.println(maxVal3 - minVal3);

  // ── Check for motor commands between sampling windows ──
  parseSerial();
}
