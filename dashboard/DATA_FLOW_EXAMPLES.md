# User Input Data Flow - Code Examples

## 1. User Profile Input Flow

### Step 1: User Enters Data (Frontend)
**File:** `templates/dashboard.html`
```html
<form id="user-profile-form">
    <input type="number" id="user-age" name="age" min="1" max="120">
    <select id="user-gender" name="gender">
        <option value="male">Male</option>
        <option value="female">Female</option>
        <option value="other">Other</option>
    </select>
    <button type="submit">Save Profile</button>
</form>
```

### Step 2: JavaScript Handles Form Submission
**File:** `static/js/dashboard.js`
```javascript
// Form submission handler
userProfileForm.addEventListener('submit', function(e) {
    e.preventDefault();
    saveUserProfile();
});

function saveUserProfile() {
    const age = document.getElementById('user-age').value;
    const gender = document.getElementById('user-gender').value;
    
    if (age && gender) {
        userProfile = {
            age: parseInt(age),
            gender: gender
        };
        
        // Save to localStorage (Client-side persistence)
        localStorage.setItem('userProfile', JSON.stringify(userProfile));
        
        // Send to server via WebSocket
        socket.emit('save_profile', userProfile);
        
        // Update UI
        updateProfileDisplay();
        updateHealthInsights();
    }
}
```

### Step 3: Server Receives Data
**File:** `arduino_dashboard.py`
```python
@socketio.on('save_profile')
def handle_save_profile(data):
    """Handle profile save request via WebSocket"""
    global user_profiles
    user_id = data.get('userId', 'default')
    
    # Store in server memory
    user_profiles[user_id] = {
        'age': data.get('age'),
        'gender': data.get('gender'),
        'timestamp': time.time()
    }
    
    # Send confirmation back to client
    emit('profile_saved', {'status': 'success', 'message': 'Profile saved successfully'})
```

### Step 4: Health Insights Generation
**File:** `static/js/dashboard.js`
```javascript
function updateHealthInsights() {
    if (!userProfile.age || !userProfile.gender) {
        healthInsights.innerHTML = `
            <div class="alert alert-warning">
                Please save your profile to see personalized health insights.
            </div>
        `;
        return;
    }

    let insights = [];
    
    // Age-based insights
    if (userProfile.age < 18) {
        insights.push('You are in the adolescent age group. Normal resting heart rate: 60-100 BPM');
    } else if (userProfile.age < 65) {
        insights.push('You are in the adult age group. Normal resting heart rate: 60-100 BPM');
    } else {
        insights.push('You are in the senior age group. Normal resting heart rate: 60-100 BPM');
    }
    
    // Gender-based insights
    if (userProfile.gender === 'female') {
        insights.push('Women typically have slightly higher resting heart rates than men.');
    }
    
    // Display insights
    const insightsHtml = insights.map(insight => `<li>${insight}</li>`).join('');
    healthInsights.innerHTML = `
        <div class="alert alert-info">
            <h6>Personalized Health Insights:</h6>
            <ul>${insightsHtml}</ul>
        </div>
    `;
}
```

## 2. Real-time Health Data Flow

### Step 1: Arduino Sends Data
**File:** `arduino_health_example.ino`
```cpp
void loop() {
    // Read sensor data
    int heartRate = readHeartRate();
    float temperature = readTemperature();
    float humidity = readHumidity();
    int activity = readActivity();
    
    // Send in CSV format
    Serial.print(heartRate);
    Serial.print(",");
    Serial.print(temperature, 1);
    Serial.print(",");
    Serial.print(humidity, 1);
    Serial.print(",");
    Serial.println(activity);
    
    delay(2000);
}
```

### Step 2: Python Reads and Parses Data
**File:** `arduino_dashboard.py`
```python
def read_arduino_data():
    """Read data from Arduino in a separate thread"""
    global sensor_data, is_connected
    
    while True:
        if arduino and is_connected:
            try:
                if arduino.in_waiting > 0:
                    data = arduino.readline().decode('utf-8').strip()
                    if data:
                        parsed_data = parse_arduino_data(data)
                        if parsed_data:
                            # Add timestamp
                            parsed_data['timestamp'] = time.time()
                            
                            # Update sensor data storage
                            for key, value in parsed_data.items():
                                if key in sensor_data:
                                    sensor_data[key].append(value)
                                    # Keep only last 100 data points
                                    if len(sensor_data[key]) > 100:
                                        sensor_data[key].pop(0)
                            
                            # Emit to all connected clients
                            socketio.emit('arduino_data', parsed_data)
                            
            except Exception as e:
                print(f"Error reading from Arduino: {e}")
                is_connected = False
                break
        else:
            time.sleep(1)

def parse_arduino_data(data_string):
    """Parse data from Arduino string format"""
    try:
        # CSV format: heartRate,temperature,humidity,activity
        if ',' in data_string:
            values = data_string.strip().split(',')
            if len(values) >= 4:
                return {
                    'heartRate': float(values[0]),
                    'temperature': float(values[1]),
                    'humidity': float(values[2]),
                    'activity': float(values[3])
                }
        return None
    except Exception as e:
        print(f"Error parsing data: {e}")
        return None
```

### Step 3: Browser Receives Real-time Updates
**File:** `static/js/dashboard.js`
```javascript
// WebSocket event handler for real-time data
socket.on('arduino_data', function(data) {
    updateDashboard(data);
    addToDataLog(data);
});

function updateDashboard(data) {
    // Update real-time values
    if (data.heartRate !== undefined) {
        document.getElementById('heart-rate-value').textContent = `${data.heartRate.toFixed(0)} BPM`;
    }
    if (data.temperature !== undefined) {
        document.getElementById('temp-value').textContent = `${data.temperature.toFixed(1)}°C`;
    }
    if (data.humidity !== undefined) {
        document.getElementById('humidity-value').textContent = `${data.humidity.toFixed(1)}%`;
    }
    if (data.activity !== undefined) {
        document.getElementById('activity-value').textContent = data.activity.toFixed(0);
    }

    // Update charts
    updateCharts(data);
    
    // Update health insights with new data
    updateHealthInsights();
}
```

## 3. Data Storage Locations

### Client-Side Storage (localStorage)
```javascript
// Stored in browser's localStorage
{
    "userProfile": {
        "age": 25,
        "gender": "female"
    }
}
```

### Server-Side Storage (In-Memory)
```python
# User profiles (lost on server restart)
user_profiles = {
    "default": {
        "age": 25,
        "gender": "female",
        "timestamp": 1640995200.0
    }
}

# Sensor data (last 100 points each)
sensor_data = {
    "heartRate": [75, 76, 74, 77, 75, ...],
    "temperature": [25.5, 25.6, 25.4, 25.7, 25.5, ...],
    "humidity": [60.2, 60.1, 60.3, 60.0, 60.2, ...],
    "activity": [512, 515, 510, 518, 512, ...],
    "timestamp": [1640995200.0, 1640995202.0, ...]
}
```

## 4. API Endpoints for External Access

### REST API for User Profile
```python
@app.route('/api/profile', methods=['GET', 'POST'])
def handle_profile():
    global user_profiles
    
    if request.method == 'POST':
        # Save profile
        data = request.get_json()
        user_id = data.get('userId', 'default')
        user_profiles[user_id] = {
            'age': data.get('age'),
            'gender': data.get('gender'),
            'timestamp': time.time()
        }
        return jsonify({'status': 'success', 'message': 'Profile saved'})
    
    elif request.method == 'GET':
        # Get profile
        user_id = request.args.get('userId', 'default')
        profile = user_profiles.get(user_id)
        return jsonify({'status': 'success', 'profile': profile})
```

### REST API for Health Insights
```python
@app.route('/api/health-insights', methods=['GET'])
def get_health_insights():
    user_id = request.args.get('userId', 'default')
    profile = user_profiles.get(user_id)
    
    if not profile:
        return jsonify({'status': 'error', 'message': 'No profile found'})
    
    # Get latest heart rate data
    latest_heart_rate = sensor_data['heartRate'][-1] if sensor_data['heartRate'] else None
    
    insights = []
    
    # Age-based insights
    age = profile.get('age')
    if age:
        if age < 18:
            insights.append('Adolescent age group. Normal resting heart rate: 60-100 BPM')
        elif age < 65:
            insights.append('Adult age group. Normal resting heart rate: 60-100 BPM')
        else:
            insights.append('Senior age group. Normal resting heart rate: 60-100 BPM')
    
    # Heart rate analysis
    if latest_heart_rate:
        if latest_heart_rate < 60:
            insights.append('Heart rate below normal range. Consider consulting healthcare provider.')
        elif latest_heart_rate > 100:
            insights.append('Heart rate above normal range. Consider consulting healthcare provider.')
        else:
            insights.append('Heart rate within normal range.')
    
    return jsonify({
        'status': 'success',
        'insights': insights,
        'latest_heart_rate': latest_heart_rate
    })
```

## 5. Data Flow Summary

```
User Input Flow:
HTML Form → JavaScript → localStorage → WebSocket → Python Server → In-Memory Storage → Health Insights

Real-time Data Flow:
Arduino Sensors → Serial Port → Python Parser → In-Memory Storage → WebSocket → JavaScript → Charts & Insights

External API Flow:
HTTP Request → Flask Route → In-Memory Storage → JSON Response
```

## 6. Key Data Transformation Points

1. **Form Input → JavaScript Object**: HTML form values converted to JavaScript object
2. **JavaScript Object → JSON String**: localStorage serialization
3. **JSON String → Python Dictionary**: WebSocket deserialization
4. **Python Dictionary → Health Logic**: Age/gender rules applied
5. **Health Logic → HTML**: Insights rendered in browser
6. **Arduino CSV → Python Dictionary**: Serial data parsing
7. **Python Dictionary → WebSocket JSON**: Real-time transmission
8. **WebSocket JSON → Chart Data**: Frontend visualization 