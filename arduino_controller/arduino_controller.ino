#include "Arduino_LED_Matrix.h"
#include <WiFiS3.h>
#include <ArduinoJson.h>

ArduinoLEDMatrix matrix;

const char* ssid = "NG";
const char* password = "whackamole9112";
const bool serial_mode = false;
const bool debug = true;

const char* serverIp = "172.17.87.239";  // Replace with your server IP
const int port = 9000;  // Ensure this matches the port your Flask server is running on
WiFiClient client;

const uint32_t animation[][4] = {
  {0x19819, 0x80001f83, 0xc204000, 66},
  {0x19819, 0x80002041, 0xf8000000, 66}
};
int counter = 0;

const int connectionTimeout = 5000; // 5 seconds timeout

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    ; // wait for serial port to connect. Needed for native USB port only
  }
  
  matrix.begin();
  delay(5000);
  matrix.loadFrame(animation[0]);
  
  if (!serial_mode) {
    connectToWiFi();
  }
}

void loop() {
  if (!serial_mode) {
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("WiFi connection lost. Reconnecting...");
      connectToWiFi();
    }
    
    sendApiRequest();
    delay(1000); // Wait for 1 second before next request
  } else {
    handleSerialInput();
  }
}

void connectToWiFi() {
  Serial.print("Connecting to Network: ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);
  
  unsigned long startAttemptTime = millis();
  
  while (WiFi.status() != WL_CONNECTED && millis() - startAttemptTime < connectionTimeout) {
    delay(100);
    Serial.print(".");
  }
  
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("\nFailed to connect to WiFi. Please check your credentials and try again.");
  } else {
    Serial.println("\nConnected to Network");
    printWiFiStatus();
  }
}

void sendApiRequest() {
  Serial.print("Sending API request ");
  Serial.println(counter);
  
  unsigned long requestStart = millis();
  
  if (client.connect(serverIp, port)) {
    Serial.println("Connected to server");
    
    // Send HTTP request
    client.println("GET /api/signals HTTP/1.1");
    client.println("Host: " + String(serverIp));
    client.println("Connection: close");
    client.println();
    
    unsigned long requestDuration = millis() - requestStart;
    Serial.print("Request took ");
    Serial.print(requestDuration);
    Serial.println(" ms");
    
    // Wait for the response
    while (client.connected()) {
      String line = client.readStringUntil('\n');
      if (line == "\r") {
        Serial.println("Headers received");
        break;
      }
    }
    
    // Read the response
    String response = client.readString();
    Serial.println("Response: " + response);
    
    parseAndHandleResponse(response);
    
    client.stop();
  } else {
    Serial.println("Connection to server failed");
  }
  
  counter++;
}

void parseAndHandleResponse(String response) {
  StaticJsonDocument<200> doc;
  DeserializationError error = deserializeJson(doc, response);
  
  if (!error) {
    int frameNumber = doc["signal"];
    Serial.print("Signal value: ");
    Serial.println(frameNumber);
    
    if (frameNumber >= 0 && frameNumber < 2) {
      matrix.loadFrame(animation[frameNumber]);
    }
  } else {
    Serial.print("JSON parsing failed: ");
    Serial.println(error.c_str());
  }
}

void handleSerialInput() {
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