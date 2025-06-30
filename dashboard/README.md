# Health Monitoring Dashboard

A comprehensive health monitoring system with real-time sensor data visualization and integrated heat stroke risk assessment. This unified dashboard combines health monitoring capabilities with medical-grade heat stroke assessment tools.

## Features

### 🏥 Health Monitoring
- **Real-time Sensor Data**: Monitor heart rate, temperature, humidity, and activity levels
- **Interactive Charts**: Live-updating charts using Chart.js
- **Data Logging**: Comprehensive data logging with timestamps
- **User Profiles**: Personalized health insights based on age and gender
- **Connection Management**: Easy Arduino device connection/disconnection

### 🌡️ Heat Stroke Assessment
- **Medical-Grade Assessment**: Based on established medical guidelines
- **Comprehensive Evaluation**: Core temperature, mental status, vital signs, environmental factors
- **Risk Scoring**: Advanced risk calculation algorithm
- **Emergency Alerts**: Critical risk detection with immediate alerts
- **Personalized Recommendations**: Actionable advice based on risk level
- **Assessment History**: Track assessment results over time

### 🎯 Unified Interface
- **Tabbed Navigation**: Seamless switching between monitoring and assessment
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Real-time Updates**: Live data updates via WebSocket connections
- **Modern UI**: Clean, professional interface with Bootstrap 5

## Quick Start

### Prerequisites
- Python 3.7 or higher
- pip3
- Arduino (optional - simulation mode available)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd dashboard
   ```

2. **Run the startup script**
   ```bash
   chmod +x start_dashboard.sh
   ./start_dashboard.sh
   ```

3. **Access the dashboard**
   - Open your browser and go to: http://localhost:5000
   - The dashboard will automatically start in simulation mode if no Arduino is connected

### Manual Installation

If you prefer manual installation:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
python3 arduino_dashboard.py
```

## Usage

### Health Monitoring Tab

1. **Set Up Profile**
   - Enter your age and gender
   - Click "Save Profile" to enable personalized insights

2. **Connect Device**
   - Click "Connect Device" to connect to Arduino
   - If no Arduino is available, the system will run in simulation mode

3. **Monitor Data**
   - View real-time vital signs in the data cards
   - Watch live-updating charts
   - Check health insights for personalized recommendations
   - Review data log for historical information

### Heat Stroke Assessment Tab

1. **Complete Assessment Form**
   - **Core Temperature**: Enter body temperature and measurement method
   - **Mental Status**: Select current mental state (critical for diagnosis)
   - **Vital Signs**: Input heart rate, respiratory rate, blood pressure
   - **Environmental Factors**: Enter ambient temperature, humidity, activity level
   - **Symptoms**: Check all applicable symptoms
   - **Risk Factors**: Select relevant medical history and risk factors

2. **Review Results**
   - **Risk Level**: Low, Moderate, High, or Critical
   - **Risk Score**: Numerical score out of 100
   - **Emergency Actions**: Immediate steps if critical risk detected
   - **Recommendations**: Personalized advice based on assessment

3. **Track History**
   - View previous assessments
   - Monitor risk trends over time
   - Access historical recommendations

## Architecture

### Frontend
- **HTML5**: Semantic markup with Bootstrap 5 framework
- **CSS3**: Custom styling with responsive design
- **JavaScript**: 
  - Chart.js for data visualization
  - Socket.IO for real-time communication
  - Custom modules for dashboard and assessment functionality

### Backend
- **Flask**: Web framework for API endpoints
- **Flask-SocketIO**: Real-time WebSocket communication
- **PySerial**: Arduino communication
- **Custom Assessment Engine**: Medical-grade heat stroke risk calculation

### Data Flow
```
Arduino Sensors → Serial Communication → Flask Backend → WebSocket → Frontend Charts
User Input → Assessment Engine → Risk Calculation → Personalized Recommendations
```

## API Endpoints

### Health Monitoring
- `GET /` - Main dashboard
- `GET /health` - Health check endpoint
- `GET /api/health-data` - Current sensor data
- `POST /api/profile` - Save user profile

### Heat Stroke Assessment
- `POST /api/assessment` - Perform assessment
- `GET /api/assessment-history` - Assessment history

### WebSocket Events
- `arduino_data` - Real-time sensor data
- `connection_status` - Device connection status
- `heat_stroke_assessment` - Assessment submission
- `assessment_result` - Assessment results
- `emergency_alert` - Critical risk alerts

## Arduino Integration

### Hardware Requirements
- Arduino board (Uno, Nano, or similar)
- Heart rate sensor (Pulse sensor, MAX30100, etc.)
- Temperature sensor (DHT22, LM35, etc.)
- Humidity sensor (DHT22, etc.)
- Activity sensor (accelerometer, etc.)

### Arduino Code
See `arduino_health_example.ino` for a complete example that includes:
- Heart rate monitoring
- Temperature and humidity sensing
- Activity level detection
- Serial communication protocol

### Data Format
Arduino should send data in this format:
```
HR:75,TEMP:37.2,HUM:65,ACT:moderate
```

## Heat Stroke Assessment Guidelines

The assessment system is based on established medical guidelines:

### Critical Indicators
- **Core Temperature ≥ 104°F (40°C)**
- **Mental Status Changes**: Confusion, delirium, unconsciousness, seizures
- **Multiple Risk Factors**: Age >65, cardiovascular disease, medications

### Risk Levels
- **Low Risk (0-49)**: Normal monitoring, hydration
- **Moderate Risk (50-79)**: Cool environment, rest, medical evaluation if needed
- **High Risk (80-99)**: Immediate medical attention, cooling measures
- **Critical Risk (100+)**: Emergency medical care, call 911

### Emergency Actions
- Move to cool environment
- Remove excess clothing
- Apply cool compresses
- Call emergency services if critical
- Monitor vital signs continuously

## Configuration

### Environment Variables
- `FLASK_ENV`: Set to 'development' for debug mode
- `ARDUINO_PORT`: Custom Arduino port (default: auto-detect)

### Customization
- Modify `dashboard.css` for styling changes
- Update assessment parameters in `heat_stroke_assessment.py`
- Adjust chart configurations in `dashboard.js`

## Troubleshooting

### Common Issues

1. **Arduino Connection Failed**
   - Check USB connection
   - Verify correct port in Arduino IDE
   - Ensure Arduino code is uploaded

2. **Charts Not Updating**
   - Check browser console for JavaScript errors
   - Verify WebSocket connection status
   - Refresh page if needed

3. **Assessment Not Working**
   - Ensure all required fields are filled
   - Check browser console for errors
   - Verify mental status is selected

### Debug Mode
Enable debug mode by setting environment variable:
```bash
export FLASK_ENV=development
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Medical Disclaimer

This system is for educational and monitoring purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare providers for medical concerns.

## Support

For support and questions:
- Check the troubleshooting section
- Review the API documentation
- Open an issue on GitHub

---

**Health Monitoring Dashboard** - Empowering health awareness through technology 