/*
 * Health Monitoring Arduino Example
 * Simulates heart rate, temperature, humidity, and activity data
 * Compatible with the Health Monitoring Dashboard
 */

// Pin definitions (for real sensors)
const int TEMP_PIN = A0;      // Temperature sensor pin
const int HUMIDITY_PIN = A1;  // Humidity sensor pin
const int LIGHT_PIN = A2;     // Light/activity sensor pin
const int HEART_RATE_PIN = A3; // Heart rate sensor pin

// Data transmission interval
const unsigned long TRANSMISSION_INTERVAL = 2000; // 2 seconds
unsigned long lastTransmission = 0;

// Simulated data ranges
const int MIN_HEART_RATE = 60;
const int MAX_HEART_RATE = 100;
const float MIN_TEMP = 20.0;
const float MAX_TEMP = 30.0;
const float MIN_HUMIDITY = 40.0;
const float MAX_HUMIDITY = 80.0;
const int MIN_ACTIVITY = 0;
const int MAX_ACTIVITY = 1000;

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  
  // Initialize analog pins
  pinMode(TEMP_PIN, INPUT);
  pinMode(HUMIDITY_PIN, INPUT);
  pinMode(LIGHT_PIN, INPUT);
  pinMode(HEART_RATE_PIN, INPUT);
  
  Serial.println("Health Monitoring Arduino Started");
  Serial.println("Data format: heartRate,temperature,humidity,activity");
}

void loop() {
  unsigned long currentTime = millis();
  
  // Send data every TRANSMISSION_INTERVAL milliseconds
  if (currentTime - lastTransmission >= TRANSMISSION_INTERVAL) {
    // Read sensor data (or generate simulated data)
    int heartRate = readHeartRate();
    float temperature = readTemperature();
    float humidity = readHumidity();
    int activity = readActivity();
    
    // Send data in CSV format
    Serial.print(heartRate);
    Serial.print(",");
    Serial.print(temperature, 1);
    Serial.print(",");
    Serial.print(humidity, 1);
    Serial.print(",");
    Serial.println(activity);
    
    lastTransmission = currentTime;
  }
  
  // Small delay to prevent overwhelming the serial port
  delay(100);
}

int readHeartRate() {
  // Simulate heart rate with some variation
  static int baseHeartRate = 75;
  static int variation = 0;
  static int direction = 1;
  
  // Gradually vary the heart rate
  variation += direction * random(1, 4);
  
  // Change direction when reaching limits
  if (variation > 15) {
    direction = -1;
  } else if (variation < -15) {
    direction = 1;
  }
  
  int heartRate = baseHeartRate + variation;
  
  // Ensure heart rate stays within reasonable bounds
  heartRate = constrain(heartRate, MIN_HEART_RATE, MAX_HEART_RATE);
  
  return heartRate;
}

float readTemperature() {
  // Simulate temperature reading with gradual changes
  static float currentTemp = 25.0;
  static float targetTemp = 25.0;
  
  // Occasionally change target temperature
  if (random(100) < 5) {
    targetTemp = random(MIN_TEMP * 10, MAX_TEMP * 10) / 10.0;
  }
  
  // Gradually move toward target temperature
  if (currentTemp < targetTemp) {
    currentTemp += 0.1;
  } else if (currentTemp > targetTemp) {
    currentTemp -= 0.1;
  }
  
  return currentTemp;
}

float readHumidity() {
  // Simulate humidity reading
  static float currentHumidity = 60.0;
  static float targetHumidity = 60.0;
  
  // Occasionally change target humidity
  if (random(100) < 3) {
    targetHumidity = random(MIN_HUMIDITY * 10, MAX_HUMIDITY * 10) / 10.0;
  }
  
  // Gradually move toward target humidity
  if (currentHumidity < targetHumidity) {
    currentHumidity += 0.2;
  } else if (currentHumidity > targetHumidity) {
    currentHumidity -= 0.2;
  }
  
  return currentHumidity;
}

int readActivity() {
  // Simulate activity level (0 = no activity, 1000 = high activity)
  static int currentActivity = 200;
  static int targetActivity = 200;
  
  // Occasionally change activity level
  if (random(100) < 10) {
    targetActivity = random(MIN_ACTIVITY, MAX_ACTIVITY);
  }
  
  // Gradually move toward target activity
  if (currentActivity < targetActivity) {
    currentActivity += random(10, 30);
  } else if (currentActivity > targetActivity) {
    currentActivity -= random(10, 30);
  }
  
  // Ensure activity stays within bounds
  currentActivity = constrain(currentActivity, MIN_ACTIVITY, MAX_ACTIVITY);
  
  return currentActivity;
}

/*
 * Alternative data formats you can use:
 * 
 * JSON Format:
 * Serial.println("{\"heartRate\": " + String(heartRate) + ", \"temperature\": " + String(temperature, 1) + ", \"humidity\": " + String(humidity, 1) + ", \"activity\": " + String(activity) + "}");
 * 
 * Key-Value Format:
 * Serial.println("heartRate:" + String(heartRate) + ",temperature:" + String(temperature, 1) + ",humidity:" + String(humidity, 1) + ",activity:" + String(activity));
 * 
 * The dashboard supports all three formats automatically.
 */ 