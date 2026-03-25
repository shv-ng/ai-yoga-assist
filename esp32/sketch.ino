#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

const char* SSID     = "Wokwi-GUEST";   // works on Wokwi out of the box
const char* PASSWORD = "";
const char* SERVER   = "http://YOUR_PC_IP:5000";

// On real hardware: #include "esp_camera.h" and init the cam here
// For Wokwi: we fake a 1-byte JPEG body just to test the HTTP round-trip

void setup() {
  Serial.begin(115200);
  WiFi.begin(SSID, PASSWORD);
  while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
  Serial.println("\nWiFi connected: " + WiFi.localIP().toString());
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) return;

  HTTPClient http;
  http.begin(String(SERVER) + "/process");
  http.addHeader("Content-Type", "image/jpeg");

  // --- swap this block with real cam capture on hardware ---
  uint8_t fake_jpg[] = {0xFF, 0xD8, 0xFF, 0xD9};  // minimal valid JPEG stub
  int code = http.POST(fake_jpg, sizeof(fake_jpg));
  // ---------------------------------------------------------

  if (code == 200) {
    String body = http.getString();
    StaticJsonDocument<64> doc;
    deserializeJson(doc, body);
    int voice_id = doc["voice_id"];
    Serial.printf("Voice ID: %d\n", voice_id);
    playVoice(voice_id);
  } else {
    Serial.printf("HTTP error: %d\n", code);
  }
  http.end();
  delay(3000);   // capture rate: 1 frame every 3 s for testing
}

void playVoice(int vid) {
  // On real hardware: fetch /voice/<vid>, decode MP3, push to I2S speaker
  // For Wokwi: just log it — no audio hardware in simulator
  Serial.printf("→ Playing voice file %03d.mp3\n", vid);
}
