# Health Monitoring Dashboard - System Architecture

## Overview
The Health Monitoring Dashboard is a real-time web application that collects user profile data, processes health metrics, and provides personalized insights. Here's how the data flows through the system.

## System Architecture Diagram

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Browser  │    │  Flask Server   │    │  Arduino/Device │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │ User Profile│ │    │ │ WebSocket   │ │    │ │ Health      │ │
│ │ Form        │ │    │ │ Server      │ │    │ │ Sensors     │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │ Real-time   │ │    │ │ REST API    │ │    │ │ Serial      │ │
│ │ Charts      │ │    │ │ Endpoints   │ │    │ │ Output      │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │ Health      │ │    │ │ Data        │ │    │ │ CSV/JSON    │ │
│ │ Insights    │ │    │ │ Storage     │ │    │ │ Format      │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                    Data Flow                                │
    └─────────────────────────────────────────────────────────────┘
```

## Detailed Data Flow

### 1. User Profile Input Flow

```
User Input → Browser Storage → Server Storage → Health Insights
     │              │              │              │
     ▼              ▼              ▼              ▼
┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
│ Age &   │   │localStor│   │In-Memory│   │Personal │
│ Gender  │   │age      │   │Dict     │   │ized     │
│ Form    │   │         │   │         │   │Analysis │
└─────────┘   └─────────┘   └─────────┘   └─────────┘
```

**Step-by-step process:**

1. **User enters age and gender** in the profile form
2. **JavaScript saves to localStorage** for persistence
3. **WebSocket sends to server** for processing
4. **Server stores in memory** (user_profiles dictionary)
5. **Health insights are generated** based on profile + real-time data

### 2. Real-time Health Data Flow

```
Arduino/Device → Serial Port → Python Parser → WebSocket → Browser Charts
      │              │              │              │              │
      ▼              ▼              ▼              ▼              ▼
┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
│ Sensors │   │CSV/JSON │   │Data Dict│   │Real-time│   │Charts & │
│ (HR,    │   │Format   │   │         │   │Emit     │   │Insights │
│ Temp,   │   │         │   │         │   │         │   │         │
│ etc.)   │   │         │   │         │   │         │   │         │
└─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘
```

**Step-by-step process:**

1. **Arduino reads sensors** and sends data via Serial
2. **Python reads Serial port** and parses data
3. **Data is stored** in sensor_data dictionary
4. **WebSocket emits** to all connected clients
5. **Browser updates** charts and insights in real-time

## Component Details

### Frontend (Browser)
- **HTML Templates**: User interface structure
- **JavaScript**: Real-time updates and user interactions
- **CSS**: Styling and responsive design
- **localStorage**: Client-side data persistence

### Backend (Flask Server)
- **Flask App**: Main web server
- **SocketIO**: Real-time WebSocket communication
- **Serial Communication**: Arduino/device connection
- **Data Storage**: In-memory storage (user_profiles, sensor_data)

### Hardware (Arduino/Device)
- **Sensors**: Heart rate, temperature, humidity, activity
- **Serial Output**: Data transmission to computer
- **Data Format**: CSV, JSON, or key-value pairs

## Data Storage Architecture

### Client-Side Storage (localStorage)
```javascript
// User profile data
{
  "userProfile": {
    "age": 25,
    "gender": "female"
  }
}
```

### Server-Side Storage (In-Memory)
```python
# User profiles storage
user_profiles = {
    "default": {
        "age": 25,
        "gender": "female",
        "timestamp": 1640995200.0
    }
}

# Sensor data storage
sensor_data = {
    "heartRate": [75, 76, 74, ...],
    "temperature": [25.5, 25.6, 25.4, ...],
    "humidity": [60.2, 60.1, 60.3, ...],
    "activity": [512, 515, 510, ...],
    "timestamp": [1640995200.0, 1640995202.0, ...]
}
```

## API Endpoints

### REST API Endpoints
```
GET  /api/profile?userId=<id>     # Get user profile
POST /api/profile                 # Save user profile
GET  /api/health-insights?userId=<id>  # Get health insights
```

### WebSocket Events
```
Client → Server:
- connect_arduino
- disconnect_arduino
- request_data
- save_profile

Server → Client:
- connection_status
- arduino_data
- sensor_data
- profile_saved
```

## Data Processing Pipeline

### 1. User Profile Processing
```
Input → Validation → Storage → Health Logic → Insights
  │         │          │          │           │
  ▼         ▼          ▼          ▼           ▼
Age &   Check for   Save to   Apply age/   Display
Gender  valid data  memory    gender      personalized
                      │        rules      recommendations
                      ▼
                Persist for
                session
```

### 2. Health Data Processing
```
Raw Data → Parse → Validate → Store → Analyze → Display
   │        │        │        │       │        │
   ▼        ▼        ▼        ▼       ▼        ▼
Serial   JSON/    Check    Add to   Compare   Update
Output   CSV      ranges   arrays   with      charts &
         Parse            │        normal    insights
                          ▼        ranges
                    Keep last
                    100 points
```

## Security Considerations

### Current Implementation
- **Client-side validation**: Basic form validation
- **Server-side storage**: In-memory (not persistent)
- **No authentication**: Single-user system
- **Local network**: Runs on localhost

### Production Recommendations
- **Database storage**: PostgreSQL/MongoDB for persistence
- **User authentication**: JWT tokens or session-based auth
- **Data encryption**: HTTPS and encrypted storage
- **Input sanitization**: Server-side validation
- **Rate limiting**: Prevent abuse

## Scalability Considerations

### Current Limitations
- **Single server**: No load balancing
- **In-memory storage**: Data lost on restart
- **Single user**: No multi-user support
- **Local deployment**: No cloud hosting

### Scaling Options
- **Microservices**: Separate API, WebSocket, and data services
- **Database**: Move from in-memory to persistent storage
- **Load balancer**: Multiple server instances
- **Message queue**: Redis/RabbitMQ for real-time data
- **Cloud deployment**: AWS/Azure/GCP hosting

## File Structure and Dependencies

```
dashboard/
├── arduino_dashboard.py          # Main Flask application
├── arduino_health_example.ino    # Arduino sketch
├── requirements.txt              # Python dependencies
├── templates/
│   └── dashboard.html            # Main UI template
└── static/
    ├── css/
    │   └── dashboard.css         # Styles
    └── js/
        └── dashboard.js          # Frontend logic
```

### Key Dependencies
- **Flask**: Web framework
- **Flask-SocketIO**: WebSocket support
- **PySerial**: Arduino communication
- **Chart.js**: Frontend charts
- **Bootstrap**: UI framework

## Performance Considerations

### Current Performance
- **Real-time updates**: 2-second intervals
- **Data retention**: Last 100 data points
- **Memory usage**: Minimal (in-memory storage)
- **Network**: Local WebSocket communication

### Optimization Opportunities
- **Data compression**: Reduce WebSocket payload size
- **Caching**: Cache frequently accessed data
- **Database indexing**: For large datasets
- **CDN**: For static assets in production
- **Background processing**: Async data processing 