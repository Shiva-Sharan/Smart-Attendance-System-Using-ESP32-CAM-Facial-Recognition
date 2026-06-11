#include "esp_camera.h"
#include <WiFi.h>
#include "esp_http_server.h"

// ==========================================
// 1. WIFI SETTINGS
// ==========================================
const char* ssid = "poco";
const char* password = "1122334455";

// ==========================================
// 2. AI-THINKER CAMERA PINS & FLASH PIN
// ==========================================
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

#define FLASH_LED_PIN      4

// ==========================================
// 2.1 CAMERA TUNING DEFAULTS (natural look)
// ==========================================
#define CAM_DEFAULT_HMIRROR    0   // 0 = real-world orientation (not mirrored)
#define CAM_DEFAULT_VFLIP      1   // AI-Thinker often needs this for upright image
#define CAM_DEFAULT_BRIGHTNESS 0   // range: -2..2
#define CAM_DEFAULT_CONTRAST   0   // range: -2..2
#define CAM_DEFAULT_SATURATION 0   // range: -2..2

static httpd_handle_t control_httpd = nullptr;
static httpd_handle_t stream_httpd = nullptr;
static bool server_started = false;

// ==========================================
// 3. HTTP SERVER HANDLERS
// ==========================================
#define PART_BOUNDARY "123456789000000000000987654321"
static const char* STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

static int clamp_int(int v, int lo, int hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

static void apply_camera_tuning(sensor_t* s) {
  if (!s) {
    return;
  }

  // Auto controls for stable real-world color/exposure.
  s->set_whitebal(s, 1);
  s->set_awb_gain(s, 1);
  s->set_wb_mode(s, 0);
  s->set_exposure_ctrl(s, 1);
  s->set_aec2(s, 1);
  s->set_ae_level(s, 0);
  s->set_gain_ctrl(s, 1);

  // Neutral profile to avoid strong color cast.
  s->set_brightness(s, clamp_int(CAM_DEFAULT_BRIGHTNESS, -2, 2));
  s->set_contrast(s, clamp_int(CAM_DEFAULT_CONTRAST, -2, 2));
  s->set_saturation(s, clamp_int(CAM_DEFAULT_SATURATION, -2, 2));
  s->set_special_effect(s, 0);

  // Orientation.
  s->set_hmirror(s, CAM_DEFAULT_HMIRROR ? 1 : 0);
  s->set_vflip(s, CAM_DEFAULT_VFLIP ? 1 : 0);

  // Sensor corrections.
  s->set_bpc(s, 1);
  s->set_wpc(s, 1);
  s->set_raw_gma(s, 1);
  s->set_lenc(s, 1);
}

static esp_err_t stream_handler(httpd_req_t* req) {
  camera_fb_t* fb = nullptr;
  esp_err_t res = ESP_OK;
  char part_buf[64];

  res = httpd_resp_set_type(req, STREAM_CONTENT_TYPE);
  if (res != ESP_OK) {
    return res;
  }
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

  while (true) {
    fb = esp_camera_fb_get();
    if (!fb) {
      delay(2);
      continue;
    }

    res = httpd_resp_send_chunk(req, STREAM_BOUNDARY, strlen(STREAM_BOUNDARY));
    if (res == ESP_OK) {
      size_t hlen = snprintf(part_buf, sizeof(part_buf), STREAM_PART, fb->len);
      res = httpd_resp_send_chunk(req, part_buf, hlen);
    }
    if (res == ESP_OK) {
      res = httpd_resp_send_chunk(req, reinterpret_cast<const char*>(fb->buf), fb->len);
    }

    esp_camera_fb_return(fb);
    if (res != ESP_OK) {
      break;
    }
  }

  return res;
}

static esp_err_t cmd_handler(httpd_req_t* req) {
  char buf[100] = {0};
  bool handled = false;
  sensor_t* s = esp_camera_sensor_get();

  if (httpd_req_get_url_query_str(req, buf, sizeof(buf)) == ESP_OK) {
    char var[32] = {0};
    char val[32] = {0};

    if (httpd_query_key_value(buf, "var", var, sizeof(var)) == ESP_OK &&
        httpd_query_key_value(buf, "val", val, sizeof(val)) == ESP_OK) {
      if (strcmp(var, "led_intensity") == 0) {
        int intensity = atoi(val);
        bool turn_on = intensity > 0;
        digitalWrite(FLASH_LED_PIN, turn_on ? HIGH : LOW);
        Serial.printf("[FLASH] %s (val=%d)\n", turn_on ? "ON" : "OFF", intensity);
        handled = true;
      } else if (s && strcmp(var, "hmirror") == 0) {
        int v = atoi(val) > 0 ? 1 : 0;
        s->set_hmirror(s, v);
        Serial.printf("[CAM] hmirror=%d\n", v);
        handled = true;
      } else if (s && strcmp(var, "vflip") == 0) {
        int v = atoi(val) > 0 ? 1 : 0;
        s->set_vflip(s, v);
        Serial.printf("[CAM] vflip=%d\n", v);
        handled = true;
      } else if (s && strcmp(var, "brightness") == 0) {
        int v = clamp_int(atoi(val), -2, 2);
        s->set_brightness(s, v);
        Serial.printf("[CAM] brightness=%d\n", v);
        handled = true;
      } else if (s && strcmp(var, "contrast") == 0) {
        int v = clamp_int(atoi(val), -2, 2);
        s->set_contrast(s, v);
        Serial.printf("[CAM] contrast=%d\n", v);
        handled = true;
      } else if (s && strcmp(var, "saturation") == 0) {
        int v = clamp_int(atoi(val), -2, 2);
        s->set_saturation(s, v);
        Serial.printf("[CAM] saturation=%d\n", v);
        handled = true;
      } else if (s && strcmp(var, "reset_tuning") == 0) {
        apply_camera_tuning(s);
        Serial.println("[CAM] tuning reset to defaults");
        handled = true;
      }
    }
  }

  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  if (!handled) {
    httpd_resp_set_status(req, "400 Bad Request");
    return httpd_resp_send(req, "Invalid command", HTTPD_RESP_USE_STRLEN);
  }

  return httpd_resp_send(req, "OK", HTTPD_RESP_USE_STRLEN);
}

void startCameraServer() {
  if (server_started) {
    return;
  }

  httpd_config_t control_config = HTTPD_DEFAULT_CONFIG();
  control_config.server_port = 80;
  control_config.max_open_sockets = 4;

  httpd_config_t stream_config = HTTPD_DEFAULT_CONFIG();
  stream_config.server_port = 81;
  stream_config.ctrl_port = control_config.ctrl_port + 1;
  stream_config.max_open_sockets = 4;

  httpd_uri_t stream_uri = {
      .uri = "/stream",
      .method = HTTP_GET,
      .handler = stream_handler,
      .user_ctx = nullptr,
  };

  httpd_uri_t cmd_uri = {
      .uri = "/control",
      .method = HTTP_GET,
      .handler = cmd_handler,
      .user_ctx = nullptr,
  };

  bool control_ok = (httpd_start(&control_httpd, &control_config) == ESP_OK);
  if (control_ok) {
    httpd_register_uri_handler(control_httpd, &cmd_uri);
  }

  bool stream_ok = (httpd_start(&stream_httpd, &stream_config) == ESP_OK);
  if (stream_ok) {
    httpd_register_uri_handler(stream_httpd, &stream_uri);
  }

  if (control_ok && stream_ok) {
    server_started = true;
    Serial.println("Camera HTTP servers started.");
    Serial.println("  Stream:  http://<ESP32_IP>:81/stream");
    Serial.println("  Control: http://<ESP32_IP>/control?var=led_intensity&val=1");
  } else {
    Serial.println("Failed to start one or more HTTP servers.");

    if (control_httpd != nullptr) {
      httpd_stop(control_httpd);
      control_httpd = nullptr;
    }
    if (stream_httpd != nullptr) {
      httpd_stop(stream_httpd);
      stream_httpd = nullptr;
    }
  }
}

void connectWifiBlocking() {
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(ssid, password);

  Serial.print("Connecting to WiFi");
  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 20000) {
    delay(300);
    Serial.print('.');
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("WiFi connected.");
    Serial.print("Stream Link: http://");
    Serial.print(WiFi.localIP());
    Serial.println(":81/stream");
    Serial.print("Control Link: http://");
    Serial.print(WiFi.localIP());
    Serial.println("/control?var=led_intensity&val=1");
    Serial.println("Tuning examples:");
    Serial.println("  /control?var=hmirror&val=0");
    Serial.println("  /control?var=vflip&val=1");
    Serial.println("  /control?var=brightness&val=0");
    Serial.println("  /control?var=contrast&val=0");
    Serial.println("  /control?var=saturation&val=0");
    Serial.println("  /control?var=reset_tuning&val=1");
  } else {
    Serial.println("WiFi connection timed out. Reconnect will continue in loop().");
  }
}

// ==========================================
// 4. SETUP & LOOP
// ==========================================
void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);

  pinMode(FLASH_LED_PIN, OUTPUT);
  digitalWrite(FLASH_LED_PIN, LOW);

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;

  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_QVGA;
  config.jpeg_quality = 12;
  config.fb_count = psramFound() ? 2 : 1;
  config.grab_mode = CAMERA_GRAB_LATEST;
  config.fb_location = psramFound() ? CAMERA_FB_IN_PSRAM : CAMERA_FB_IN_DRAM;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed! Error 0x%x\n", err);
    return;
  }

  sensor_t* s = esp_camera_sensor_get();
  apply_camera_tuning(s);

  connectWifiBlocking();
  startCameraServer();
}

void loop() {
  static unsigned long lastWifiCheck = 0;
  unsigned long now = millis();

  if (now - lastWifiCheck >= 2000) {
    lastWifiCheck = now;
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("WiFi lost, reconnecting...");
      WiFi.reconnect();
    }
  }

  delay(10);
}
