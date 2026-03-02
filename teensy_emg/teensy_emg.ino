void setup() {
  Serial.begin(115200);
}

void loop() {
  // Sample for 50ms and report the peak-to-peak amplitude on 4 channels
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
}
