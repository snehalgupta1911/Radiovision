#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <esp_wifi.h>
#include <freertos/FreeRTOS.h>
#include <freertos/queue.h>

const char* ssid = "XIAO_CSI_TX";
const char* password = "123456789";

static const uint32_t EMIT_HZ = 100;
static const uint32_t EMIT_INTERVAL_MS = 1000 / EMIT_HZ;

static uint8_t ap_bssid[6];
static bool have_bssid = false;
static uint32_t seq = 0;

#define CSI_MAX_LEN 512

struct CsiRecord {
  uint32_t seq;
  uint32_t ts;
  int8_t rssi;
  uint16_t len;
  int8_t buf[CSI_MAX_LEN];
};

static QueueHandle_t csi_q;

void csi_callback(void *ctx, wifi_csi_info_t *info) {
  if (!info || !info->buf || !have_bssid) return;
  if (memcmp(info->mac, ap_bssid, 6) != 0) return;
  if (info->len <= 0 || info->len > CSI_MAX_LEN) return;

  static uint32_t last_emit_ms = 0;
  uint32_t now = millis();

  if (now - last_emit_ms < EMIT_INTERVAL_MS) return;
  last_emit_ms = now;

  CsiRecord rec;
  rec.seq = seq++;
  rec.ts = info->rx_ctrl.timestamp;
  rec.rssi = info->rx_ctrl.rssi;
  rec.len = info->len;

  memcpy(rec.buf, info->buf, info->len);

  xQueueSend(csi_q, &rec, 0);
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
    memcpy(ap_bssid, WiFi.BSSID(), 6);
    have_bssid = true;

    Serial.printf("\nConnected! Channel:%d\n", WiFi.channel());

    esp_wifi_set_csi(true);
  } else {
    Serial.println("\nFailed - retrying...");
  }
}

void setup() {
  Serial.begin(115200);
  while (!Serial);

  Serial.println("RX starting...");

  csi_q = xQueueCreate(128, sizeof(CsiRecord));

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
    esp_wifi_set_csi(false);
    have_bssid = false;
    connectToTX();
    return;
  }

  static WiFiUDP udp;
  static bool started = false;

  if (!started) {
    udp.begin(3333);
    started = true;
  }

  udp.beginPacket(WiFi.gatewayIP(), 3333);
  udp.write((uint8_t*)"ping", 4);
  udp.endPacket();

  CsiRecord rec;

  while (xQueueReceive(csi_q, &rec, 0) == pdTRUE) {
    Serial.printf(
      "CSI,%u,%u,%d,%d,[",
      rec.seq,
      rec.ts,
      rec.rssi,
      rec.len
    );

    for (int i = 0; i < rec.len; i++) {
      Serial.print(rec.buf[i]);

      if (i < rec.len - 1)
        Serial.print(' ');
    }

    Serial.println("]");
  }

  delay(5);
}
