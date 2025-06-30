# Arduino Dashboard with Heart Rate Monitoring

A real-time web dashboard for monitoring Arduino sensor data with heart rate analysis from PPG signals.

## Features

- **Real-time Sensor Monitoring**: Temperature, humidity, light, pressure, and PPG signals
- **Heart Rate Analysis**: PPG signal processing with spectrogram visualization
- **CSV Analysis**: Automatic analysis of PPG data from CSV files
- **Responsive Design**: Modern UI with Bootstrap and custom styling
- **WebSocket Communication**: Real-time data updates

## Data Formats

The dashboard supports multiple data formats from Arduino:

### JSON Format (Recommended)
```json
{
  "temperature": 25.5,
  "humidity": 60.2,
  "light": 512,
  "pressure": 1013.25,
  "ppg": 258
}
```

### CSV Format
```
25.5,60.2,512,1013.25,258
```

### Key-Value Format
```
temp:25.5,humidity:60.2,light:512,pressure:1013.25,ppg:258
```

## Installation

1. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Dashboard**:
   ```bash
   ./start_dashboard.sh
   ```
   Or manually:
   ```bash
   python3 arduino_dashboard.py
   ```

3. **Open in Browser**:
   Navigate to `http://localhost:5002`

## Arduino Setup

### Basic Setup
1. Connect your Arduino to your computer
2. Upload the example sketch (`arduino_example.ino`)
3. Open the dashboard and click "Connect Arduino"

## Dashboard Features

### Real-time Cards
- **Temperature**: Current temperature in °C
- **Humidity**: Current humidity percentage
- **Light**: Light sensor reading
- **Pressure**: Atmospheric pressure in hPa
- **Heart Rate**: Real-time BPM from PPG analysis
- **PPG Signal**: Raw PPG sensor value

### Heart Rate Analysis
- PPG signal processing with bandpass filtering
- Spectrogram generation for frequency analysis
- Confidence scoring for heart rate estimates
- Automatic CSV file analysis from BPM directory

### Data Logging
- Real-time data log with timestamps
- Sensor values and heart rate data
- Automatic log rotation (keeps last 50 entries)

## File Structure

```
dashboard/
├── arduino_dashboard.py      # Main Flask application
├── arduino_example.ino       # Arduino example sketch
├── requirements.txt          # Python dependencies
├── start_dashboard.sh        # Startup script
├── static/
│   ├── css/
│   │   └── dashboard.css     # Custom styling
│   └── js/
│       └── dashboard.js      # Frontend JavaScript
└── templates/
    └── dashboard.html        # Main dashboard template
```

## Troubleshooting

### Port Issues
If port 5000/5001/5002 is in use:
- On macOS, disable AirPlay Receiver in System Preferences
- Or modify the port in `arduino_dashboard.py`

### Arduino Connection
- Check if Arduino is properly connected
- Verify the correct port in the code
- Ensure baud rate matches (9600)

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source and available under the MIT License. 