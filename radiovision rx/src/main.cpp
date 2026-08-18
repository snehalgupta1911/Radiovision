#include <Arduino.h>
#include <WiFi.h>
#include <esp_wifi.h>

const char* ssid = "XIAO_CSI_TX";
const char* password = "123456789";

void csi_callback(void *ctx, wifi_csi_info_t *info) {
  if (!info || !info->buf) return;
  if (info->mac[0] != 0xAE) return;  // TX MAC filter
  Serial.printf("CSI,%u,%d,%d,[",
    info->rx_ctrl.timestamp,
    info->rx_ctrl.rssi,
    info->len);
  for (int i = 0; i < info->len; i++) {
    Serial.printf("%d", info->buf[i]);
    if (i < info->len - 1) Serial.print(" ");
  }
  Serial.println("]");
}

void connectToTX() {
  Serial.print("Connecting to TX");
  WiFi.begin(ssid, password);
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 40) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\nConnected! Channel:%d\n", WiFi.channel());
    esp_wifi_set_csi(true);
  } else {
    Serial.println("\nFailed - retrying...");
  }
}

void setup() {
  Serial.begin(115200);
  while(!Serial);
  Serial.println("RX starting...");

  WiFi.mode(WIFI_STA);

  wifi_csi_config_t cfg = {};
  cfg.lltf_en = true;
  cfg.htltf_en = true;
  cfg.stbc_htltf2_en = true;
  cfg.ltf_merge_en = true;
  cfg.channel_filter_en = false;
  cfg.manu_scale = false;
  esp_wifi_set_csi_config(&cfg);
  esp_wifi_set_csi_rx_cb(&csi_callback, NULL);

  connectToTX();
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("Reconnecting...");
    esp_wifi_set_csi(false);
    connectToTX();
    return;
  }

  static WiFiUDP udp;
  static bool started = false;
  if (!started) { udp.begin(3333); started = true; }

  udp.beginPacket(WiFi.gatewayIP(), 3333);
  udp.write((uint8_t*)"ping", 4);
  udp.endPacket();
  delay(1);  // 1ms = ~1000 packets/sec constant stream
}