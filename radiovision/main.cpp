#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <esp_task_wdt.h>

const char *ssid     = "XIAO_CSI_TX";
const char *password = "123456789";

WiFiUDP udp;
uint8_t payload[100];
int count = 0;

void setup() {
  Serial.begin(115200);
  Serial.println("TX booting...");
  
  esp_task_wdt_init(10, true);  // 10 second watchdog
  esp_task_wdt_add(NULL);
  
  WiFi.mode(WIFI_AP);
  WiFi.softAP(ssid, password);
  Serial.print("TX started. IP: ");
  Serial.println(WiFi.softAPIP());
  udp.begin(3333);
  memset(payload, 0xAB, sizeof(payload));
  Serial.printf("AP Channel: %d\n", WiFi.channel());
  Serial.println("Broadcasting...");
}

void loop() {
  esp_task_wdt_reset();  // pet the watchdog
  
  int result = udp.beginPacket(IPAddress(192,168,4,255), 3333);
  if (result) {
    udp.write(payload, sizeof(payload));
    udp.endPacket();
  }
  
  count++;
  if (count % 100 == 0) {
    Serial.printf("Sent %d packets\n", count);
  }
  
  delay(10);
}