/*
  yoga_assist.ino
  ---------------
  ESP32-CAM firmware for AI Yoga Assist.

  What it does every loop:
    1. Capture a JPEG frame from the OV2640 camera
    2. POST it to the laptop server  POST /process
    3. Parse the returned {"voice_id": N}
    4. If voice_id is new: GET /voice/N  and stream the MP3 over I2S

  Hardware assumed
  ----------------
  - AI-Thinker ESP32-CAM  (PSRAM available)
  - MAX98357A I2S amplifier on pins I2S_BCLK / I2S_LRC / I2S_DOUT
  - Speaker on MAX98357A output

  Libraries required (install via Arduino Library Manager)
  --------------------------------------------------------
  - ESP32-audioI2S  by schreibfaul1   (handles HTTP MP3 streaming)
  - ArduinoJson     by Benoit Blanchon

  Board: "AI Thinker ESP32-CAM"  in Arduino IDE
  Partition scheme: "Huge APP (3MB No OTA)"  — gives enough flash for audio lib
*/

#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include "Audio.h"          // ESP32-audioI2S

// ── Wi-Fi ──────────────────────────────────────────────────────────────────
#define WIFI_SSID     "YOUR_SSID"
#define WIFI_PASSWORD "YOUR_PASSWORD"

// ── Server ─────────────────────────────────────────────────────────────────
#define SERVER_IP   "192.168.1.100"   // <-- your laptop's LAN IP
#define SERVER_PORT 8000
#define PROCESS_URL "http://" SERVER_IP ":" STR(SERVER_PORT) "/process"
#define VOICE_URL   "http://" SERVER_IP ":" STR(SERVER_PORT) "/voice/"
#define STR(x)      STR2(x)
#define STR2(x)     #x

// ── I2S pins (MAX98357A) ───────────────────────────────────────────────────
#define I2S_BCLK  26
#define I2S_LRC   25
#define I2S_DOUT  22

// ── Camera pin map — AI-Thinker ESP32-CAM ─────────────────────────────────
#define PWDN_GPIO_NUM   32
#define RESET_GPIO_NUM  -1
#define XCLK_GPIO_NUM    0
#define SIOD_GPIO_NUM   26
#define SIOC_GPIO_NUM   27
#define Y9_GPIO_NUM     35
#define Y8_GPIO_NUM     34
#define Y7_GPIO_NUM     39
#define Y6_GPIO_NUM     36
#define Y5_GPIO_NUM     21
#define Y4_GPIO_NUM     19
#define Y3_GPIO_NUM     18
#define Y2_GPIO_NUM      5
#define VSYNC_GPIO_NUM  25
#define HREF_GPIO_NUM   23
#define PCLK_GPIO_NUM   22

// ── Globals ────────────────────────────────────────────────────────────────
Audio audio;
int   lastVoiceId = -1;

// ── Camera init ────────────────────────────────────────────────────────────
bool initCamera() {
  camera_config_t cfg;
  cfg.ledc_channel = LEDC_CHANNEL_0;
  cfg.ledc_timer   = LEDC_TIMER_0;
  cfg.pin_d0       = Y2_GPIO_NUM;
  cfg.pin_d1       = Y3_GPIO_NUM;
  cfg.pin_d2       = Y4_GPIO_NUM;
  cfg.pin_d3       = Y5_GPIO_NUM;
  cfg.pin_d4       = Y6_GPIO_NUM;
  cfg.pin_d5       = Y7_GPIO_NUM;
  cfg.pin_d6       = Y8_GPIO_NUM;
  cfg.pin_d7       = Y9_GPIO_NUM;
  cfg.pin_xclk     = XCLK_GPIO_NUM;
  cfg.pin_pclk     = PCLK_GPIO_NUM;
  cfg.pin_vsync    = VSYNC_GPIO_NUM;
  cfg.pin_href     = HREF_GPIO_NUM;
  cfg.pin_sscb_sda = SIOD_GPIO_NUM;
  cfg.pin_sscb_scl = SIOC_GPIO_NUM;
  cfg.pin_pwdn     = PWDN_GPIO_NUM;
  cfg.pin_reset    = RESET_GPIO_NUM;
  cfg.xclk_freq_hz = 20000000;
  cfg.pixel_format = PIXFORMAT_JPEG;

  // Use PSRAM for larger frame buffer if available
  if (psramFound()) {
    cfg.frame_size   = FRAMESIZE_VGA;   // 640×480
    cfg.jpeg_quality = 12;              // 0–63, lower = better quality
    cfg.fb_count     = 2;
  } else {
    cfg.frame_size   = FRAMESIZE_QVGA;  // 320×240 fallback
    cfg.jpeg_quality = 20;
    cfg.fb_count     = 1;
  }

  esp_err_t err = esp_camera_init(&cfg);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    return false;
  }
  return true;
}

// ── POST frame → returns voice_id (-1 on error, 0 if null) ────────────────
int postFrame(camera_fb_t* fb) {
  HTTPClient http;
  http.begin(PROCESS_URL);
  http.addHeader("Content-Type", "image/jpeg");

  int httpCode = http.POST(fb->buf, fb->len);
  if (httpCode != 200) {
    Serial.printf("POST /process failed: %d\n", httpCode);
    http.end();
    return -1;
  }

  String body = http.getString();
  http.end();

  // Parse {"voice_id": N}  or {"voice_id": null}
  StaticJsonDocument<128> doc;
  DeserializationError err = deserializeJson(doc, body);
  if (err) {
    Serial.printf("JSON parse error: %s\n", err.c_str());
    return -1;
  }

  if (doc["voice_id"].isNull()) return 0;   // server said "stay silent"
  return doc["voice_id"].as<int>();
}

// ── Stream MP3 from /voice/{id} over I2S ──────────────────────────────────
void playVoice(int voiceId) {
  char url[64];
  snprintf(url, sizeof(url), "%s%03d.mp3", VOICE_URL, voiceId);
  Serial.printf("Playing: %s\n", url);
  audio.connecttohost(url);
}

// ── Arduino lifecycle ──────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);

  // Camera
  if (!initCamera()) {
    Serial.println("Camera init failed — halting");
    while (true) delay(1000);
  }

  // Wi-Fi
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.printf("\nConnected — IP: %s\n", WiFi.localIP().toString().c_str());

  // I2S audio
  audio.setPinout(I2S_BCLK, I2S_LRC, I2S_DOUT);
  audio.setVolume(18);   // 0–21

  // Verify server is reachable
  HTTPClient http;
  http.begin("http://" SERVER_IP ":" STR(SERVER_PORT) "/health");
  int code = http.GET();
  Serial.printf("Server health check: %d\n", code);
  http.end();
}

void loop() {
  // Keep audio streaming ticking
  audio.loop();

  // Capture frame
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed");
    delay(500);
    return;
  }

  // Send to server
  int voiceId = postFrame(fb);
  esp_camera_fb_return(fb);

  // Play if we got a new voice_id
  if (voiceId > 0 && voiceId != lastVoiceId) {
    lastVoiceId = voiceId;
    playVoice(voiceId);
  }

  // ~5 fps — matches server-side feedback tick
  // audio.loop() must keep running during this delay
  unsigned long t = millis();
  while (millis() - t < 200) {
    audio.loop();
    delay(5);
  }
}

// ── ESP32-audioI2S optional callbacks ────────────────────────────────────
void audio_info(const char* info) {
  Serial.printf("[audio] %s\n", info);
}
void audio_eof_mp3(const char* info) {
  Serial.printf("[audio] finished: %s\n", info);
  lastVoiceId = -1;   // allow replaying same correction after playback ends
}
