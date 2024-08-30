#include "Arduino_LED_Matrix.h"
#include <WiFiS3.h>
#include <ArduinoHttpClient.h>
#include <ArduinoJson.h>


ArduinoLEDMatrix matrix;

const char* ssid = "NG";
const char* password = "whackamole9112";
const bool serial_mode = false;
const bool debug = false;

const char* serverIp = "35.3.118.16";  // Replace with your server IP
const int port = 9000;  // Ensure this matches the port your Flask server is running on
WiFiClient wifiClient;
HttpClient client = HttpClient(wifiClient, serverIp, port);

const uint32_t animation[][4] = {
  {0x19819, 0x80001f83, 0xc204000, 66},
  {0x19819, 0x80002041, 0xf8000000, 66}
};

void setup() {
  if (debug) {
    Serial.begin(115200);
  }
  matrix.begin();
  delay(5000);
  matrix.loadFrame(animation[0]);
  Serial.flush();
  if (!serial_mode) {
    if (debug) {
      Serial.print("Connecting to Network: ");
      Serial.println(ssid);
      Serial.flush();
    }
    while (WiFi.status() != WL_CONNECTED) {
      WiFi.begin(ssid, password);
      delay(500);
    }

    if (debug) {
      Serial.println("Connected to Network");
      printWiFiStatus();
    }
  }
}

void loop() {
  if (!serial_mode) {
    if (WiFi.status() != WL_CONNECTED) {
      if (debug) {
        Serial.println("WiFi connection lost. Reconnecting...");
      }
      WiFi.begin(ssid, password);
      delay(5000);
    }

    client.get("/api/signals");

    int statusCode = client.responseStatusCode();
    if (statusCode == 200) {
      String response = client.responseBody();

      // Parse JSON response
      StaticJsonDocument<200> doc;  // Adjust size as needed
      DeserializationError error = deserializeJson(doc, response);
      if (!error) {
        int frameNumber = doc["signal"];
        if (debug) {
          Serial.print("Signal value: ");
          Serial.println(frameNumber);
        }
        if (frameNumber >= 0 && frameNumber < 2) {
          matrix.loadFrame(animation[frameNumber]); // THIS IS WHERE YOU CONNECT TO SUPPORTED IOT DEVICE
        }
      } else {
        if (debug) {
          Serial.print("JSON parsing failed: ");
          Serial.println(error.c_str());
        }
      }
    } else {
      if (debug) {
        Serial.println(statusCode);
      }
    }
    delay(10);
  } else {
    if (Serial.available() > 0) {
      char receivedChar = Serial.read();
      int frameNumber = receivedChar - '0';
      
      if (frameNumber >= 0 && frameNumber < 2) {
        matrix.loadFrame(animation[frameNumber]);
        Serial.print("Loaded frame: ");
        Serial.println(frameNumber);
      }
    }
  }
}

void printWiFiStatus() {
  Serial.print("SSID: ");
  Serial.println(WiFi.SSID());
  IPAddress ip = WiFi.localIP();
  Serial.print("IP Address: ");
  Serial.println(ip);
  long rssi = WiFi.RSSI();
  Serial.print("Signal strength (RSSI): ");
  Serial.print(rssi);
  Serial.println(" dBm");
}
