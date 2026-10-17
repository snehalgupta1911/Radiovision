#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>

const char* ssid = "XIAO_CSI_TX";
const char* password = "123456789";

WiFiUDP udp;
uint8_t payload[128];

void setup() {
  Serial.begin(115200);
  WiFi.mode(WIFI_AP);
  WiFi.softAP(ssid, password, 6);
  Serial.printf("TX started. IP: %s\n", WiFi.softAPIP().toString().c_str());
  udp.begin(3333);
  memset(payload, 0xAB, sizeof(payload));
}

void loop() {
  udp.beginPacket(IPAddress(192,168,4,2), 3333);  // RX fixed IP
  udp.write(payload, sizeof(payload));
  udp.endPacket();
  delay(10);
}