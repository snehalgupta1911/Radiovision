#include <Arduino.h>

void setup() {
  Serial.begin(115200);
  delay(3000);
  Serial.println("HELLO board is alive");
}

void loop() {
  Serial.println("still alive");
  delay(1000);
}