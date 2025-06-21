/*
 * Arduino Dashboard Example
 * This sketch demonstrates how to send sensor data to the Python dashboard
 * 
 * Supported data formats:
 * 1. JSON: {"temperature": 25.5, "humidity": 60.2, "light": 512, "pressure": 1013.25}
 * 2. CSV: 25.5,60.2,512,1013.25
 * 3. Key-value: temperature:25.5,humidity:60.2,light:512,pressure:1013.25
 */

// Pin definitions (modify these based on your sensor connections)
const int TEMP_SENSOR_PIN = A0;    // Analog temperature sensor
const int LIGHT_SENSOR_PIN = A1;   // Analog light sensor
const int HUMIDITY_SENSOR_PIN = A2; // Analog humidity sensor

// Data transmission interval (milliseconds)
const unsigned long TRANSMISSION_INTERVAL = 1000; // 1 second
unsigned long lastTransmission = 0;

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  
  // Initialize analog pins
  pinMode(TEMP_SENSOR_PIN, INPUT);
  pinMode(LIGHT_SENSOR_PIN, INPUT);
  pinMode(HUMIDITY_SENSOR_PIN, INPUT);
  
  Serial.println("Arduino Dashboard Sensor Node Ready");
  Serial.println("Data format: temperature,humidity,light,pressure");
}

void loop() {
  unsigned long currentTime = millis();
  
  // Send data at regular intervals
  if (currentTime - lastTransmission >= TRANSMISSION_INTERVAL) {
    // Read sensor values
    float temperature = readTemperature();
    float humidity = readHumidity();
    int light = readLight();
    float pressure = readPressure();
    
    // Send data in CSV format (easiest to parse)
    Serial.print(temperature, 1);
    Serial.print(",");
    Serial.print(humidity, 1);
    Serial.print(",");
    Serial.print(light);
    Serial.print(",");
    Serial.println(pressure, 2);
    
    lastTransmission = currentTime;
  }
}

// Sensor reading functions - modify these based on your actual sensors

float readTemperature() {
  // Example: LM35 temperature sensor (10mV/°C)
  int rawValue = analogRead(TEMP_SENSOR_PIN);
  float voltage = (rawValue * 5.0) / 1024.0; // Convert to voltage
  float temperature = voltage * 100; // Convert to Celsius
  
  // Add some realistic variation for demo purposes
  temperature += random(-2, 3) * 0.1;
  
  return temperature;
}

float readHumidity() {
  // Example: DHT22 or similar humidity sensor
  int rawValue = analogRead(HUMIDITY_SENSOR_PIN);
  float humidity = map(rawValue, 0, 1023, 0, 100); // Map to 0-100%
  
  // Add some realistic variation for demo purposes
  humidity += random(-5, 6) * 0.1;
  humidity = constrain(humidity, 0, 100); // Keep within bounds
  
  return humidity;
}

int readLight() {
  // Example: LDR (Light Dependent Resistor)
  int lightValue = analogRead(LIGHT_SENSOR_PIN);
  
  // Add some realistic variation for demo purposes
  lightValue += random(-20, 21);
  lightValue = constrain(lightValue, 0, 1023); // Keep within bounds
  
  return lightValue;
}

float readPressure() {
  // Example: BMP280 or similar pressure sensor
  // For demo purposes, simulate atmospheric pressure around 1013.25 hPa
  float pressure = 1013.25;
  
  // Add some realistic variation for demo purposes
  pressure += random(-10, 11) * 0.01;
  
  return pressure;
}

/*
 * Alternative data formats you can use:
 * 
 * JSON format:
 * void sendJsonData() {
 *   Serial.print("{\"temperature\":");
 *   Serial.print(readTemperature(), 1);
 *   Serial.print(",\"humidity\":");
 *   Serial.print(readHumidity(), 1);
 *   Serial.print(",\"light\":");
 *   Serial.print(readLight());
 *   Serial.print(",\"pressure\":");
 *   Serial.print(readPressure(), 2);
 *   Serial.println("}");
 * }
 * 
 * Key-value format:
 * void sendKeyValueData() {
 *   Serial.print("temperature:");
 *   Serial.print(readTemperature(), 1);
 *   Serial.print(",humidity:");
 *   Serial.print(readHumidity(), 1);
 *   Serial.print(",light:");
 *   Serial.print(readLight());
 *   Serial.print(",pressure:");
 *   Serial.println(readPressure(), 2);
 * }
 */ 