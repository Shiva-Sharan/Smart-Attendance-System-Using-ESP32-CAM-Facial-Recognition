#include <WiFi.h>
#include <PubSubClient.h>
#include <LiquidCrystal.h>
#include <string.h>

// Network credentials
const char* ssid = "poco";
const char* password = "1122334455";

// MQTT broker details
const char* mqtt_server = "broker.hivemq.com";
const char* mqtt_topic = "shiva_edgeid/lcd_display";

constexpr uint16_t MQTT_PORT = 1883;
constexpr uint16_t LCD_COLS = 16;
constexpr uint16_t LCD_ROWS = 2;
constexpr size_t MQTT_MSG_MAX = 128;
constexpr unsigned long MQTT_RETRY_MS = 3000;

WiFiClient espClient;
PubSubClient client(espClient);

// ESP32 pins connected to the LCD
const int rs = 22, en = 23, d4 = 5, d5 = 18, d6 = 19, d7 = 21;
LiquidCrystal lcd(rs, en, d4, d5, d6, d7);

unsigned long lastMqttAttemptMs = 0;
char lastMessage[MQTT_MSG_MAX + 1] = {0};

void lcdPrintLine(uint8_t row, const char* text) {
  lcd.setCursor(0, row);
  for (uint8_t i = 0; i < LCD_COLS; i++) {
    char c = text[i];
    if (c == '\0') {
      for (; i < LCD_COLS; i++) {
        lcd.print(' ');
      }
      return;
    }
    lcd.print(c);
  }
}

void renderMessage(const char* msg) {
  char line1[LCD_COLS + 1] = {0};
  char line2[LCD_COLS + 1] = {0};

  const char* split = strchr(msg, '\n');
  if (split) {
    size_t len1 = min(static_cast<size_t>(LCD_COLS), static_cast<size_t>(split - msg));
    memcpy(line1, msg, len1);

    const char* second = split + 1;
    size_t len2 = min(static_cast<size_t>(LCD_COLS), strlen(second));
    memcpy(line2, second, len2);
  } else {
    size_t len = min(static_cast<size_t>(LCD_COLS), strlen(msg));
    memcpy(line1, msg, len);
  }

  lcdPrintLine(0, line1);
  lcdPrintLine(1, line2);
}

void callback(char* topic, byte* payload, unsigned int length) {
  (void)topic;
  if (length == 0) {
    return;
  }

  size_t copyLen = min(static_cast<size_t>(length), static_cast<size_t>(MQTT_MSG_MAX));
  char msg[MQTT_MSG_MAX + 1] = {0};
  memcpy(msg, payload, copyLen);

  if (strncmp(msg, lastMessage, MQTT_MSG_MAX) == 0) {
    return;
  }

  strncpy(lastMessage, msg, MQTT_MSG_MAX);
  lastMessage[MQTT_MSG_MAX] = '\0';

  Serial.print("LCD payload: ");
  Serial.println(msg);
  renderMessage(msg);
}

void connectWiFiIfNeeded() {
  if (WiFi.status() == WL_CONNECTED) {
    return;
  }

  Serial.println("WiFi disconnected, reconnecting...");
  WiFi.disconnect();
  WiFi.begin(ssid, password);
}

bool connectMqtt() {
  char clientId[32];
  snprintf(clientId, sizeof(clientId), "ESP32LCDClient-%08lx", static_cast<unsigned long>(esp_random()));

  if (!client.connect(clientId)) {
    Serial.print("MQTT connect failed, rc=");
    Serial.println(client.state());
    return false;
  }

  if (!client.subscribe(mqtt_topic, 0)) {
    Serial.println("MQTT subscribe failed");
    client.disconnect();
    return false;
  }

  Serial.println("MQTT connected and subscribed");
  renderMessage("System Ready");
  return true;
}

void setup() {
  Serial.begin(115200);

  lcd.begin(LCD_COLS, LCD_ROWS);
  renderMessage("Booting EdgeID");

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(ssid, password);

  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 20000) {
    delay(250);
    Serial.print('.');
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("WiFi connected. IP: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("WiFi not connected yet; continuing with background reconnects.");
  }

  client.setServer(mqtt_server, MQTT_PORT);
  client.setBufferSize(MQTT_MSG_MAX + 16);
  client.setKeepAlive(30);
  client.setSocketTimeout(2);
  client.setCallback(callback);
}

void loop() {
  connectWiFiIfNeeded();

  if (WiFi.status() == WL_CONNECTED) {
    if (!client.connected()) {
      unsigned long now = millis();
      if (now - lastMqttAttemptMs >= MQTT_RETRY_MS) {
        lastMqttAttemptMs = now;
        connectMqtt();
      }
    } else {
      client.loop();
    }
  }

  delay(10);
}
