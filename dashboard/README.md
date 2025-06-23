# Arduino Dashboard

A real-time web dashboard for monitoring Arduino sensor data with beautiful visualizations and responsive design.

## Features

- 🔌 **Automatic Arduino Detection** - Automatically finds and connects to your Arduino
- 📊 **Real-time Charts** - Live updating charts for temperature, humidity, light, and pressure
- 📱 **Responsive Design** - Works on desktop, tablet, and mobile devices
- 🔄 **Multiple Data Formats** - Supports JSON, CSV, and key-value data formats
- 📝 **Data Logging** - Real-time data log with timestamp
- 🎨 **Modern UI** - Beautiful gradient design with smooth animations
- ⚡ **WebSocket Communication** - Real-time data updates without page refresh

## Prerequisites

- Python 3.7 or higher
- Arduino board (Uno, Nano, Mega, etc.)
- USB cable to connect Arduino to computer
- Sensors (optional - the app includes demo data)

## Installation

1. **Clone or download this project**
   ```bash
   cd /path/to/arduino-dashboard
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Upload Arduino sketch** (optional)
   - Open `arduino_example.ino` in Arduino IDE
   - Connect your Arduino via USB
   - Upload the sketch to your Arduino

## Usage

### Starting the Dashboard

1. **Run the Python application**
   ```bash
   python arduino_dashboard.py
   ```

2. **Open your web browser**
   - Navigate to `http://localhost:5000`
   - The dashboard will automatically appear

3. **Connect to Arduino**
   - Click the "Connect Arduino" button
   - The app will automatically detect your Arduino
   - If no Arduino is found, it will show a message

### Arduino Data Format

The dashboard supports three data formats from your Arduino:

#### 1. CSV Format (Recommended)
```
25.5,60.2,512,1013.25
```
Format: `temperature,humidity,light,pressure`

#### 2. JSON Format
```json
{"temperature": 25.5, "humidity": 60.2, "light": 512, "pressure": 1013.25}
```

#### 3. Key-Value Format
```
temperature:25.5,humidity:60.2,light:512,pressure:1013.25
```

### Arduino Sketch Setup

1. **Basic Setup**
   ```cpp
   void setup() {
     Serial.begin(9600);
   }
   ```

2. **Send Data**
   ```cpp
   void loop() {
     // Read your sensors
     float temp = readTemperature();
     float humidity = readHumidity();
     int light = readLight();
     float pressure = readPressure();
     
     // Send in CSV format
     Serial.print(temp, 1);
     Serial.print(",");
     Serial.print(humidity, 1);
     Serial.print(",");
     Serial.print(light);
     Serial.print(",");
     Serial.println(pressure, 2);
     
     delay(1000); // Send every second
   }
   ```

## Configuration

### Port Settings

The app automatically detects Arduino ports. If you need to specify a custom port, edit `arduino_dashboard.py`:

```python
# Change this line in the find_arduino_port() function
ports = glob.glob('/dev/tty.usbmodem*') + glob.glob('/dev/ttyACM*') + glob.glob('COM*')
```

### Baud Rate

Default baud rate is 9600. To change it, modify:

```python
BAUD_RATE = 9600  # Change to your Arduino's baud rate
```

### Data Update Interval

Change how often data is sent from Arduino by modifying the `TRANSMISSION_INTERVAL` in the Arduino sketch:

```cpp
const unsigned long TRANSMISSION_INTERVAL = 1000; // 1 second
```

## Troubleshooting

### Arduino Not Detected

1. **Check USB connection**
   - Ensure Arduino is properly connected via USB
   - Try a different USB cable or port

2. **Check port permissions** (Linux/macOS)
   ```bash
   sudo chmod 666 /dev/ttyUSB0
   ```

3. **Verify Arduino IDE connection**
   - Open Arduino IDE
   - Check if Arduino appears in Tools > Port menu

### No Data Displayed

1. **Check Arduino serial output**
   - Open Arduino IDE Serial Monitor
   - Verify data is being sent in correct format

2. **Check baud rate**
   - Ensure Arduino and Python app use same baud rate (default: 9600)

3. **Check data format**
   - Verify Arduino sends data in one of the supported formats
   - Check for extra characters or formatting issues

### Dashboard Not Loading

1. **Check Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Check port availability**
   - Ensure port 5000 is not in use
   - Change port in `arduino_dashboard.py` if needed

3. **Check firewall settings**
   - Allow Python/Flask through firewall

## File Structure

```
arduino-dashboard/
├── arduino_dashboard.py      # Main Python application
├── requirements.txt          # Python dependencies
├── arduino_example.ino       # Example Arduino sketch
├── README.md                 # This file
├── templates/
│   └── dashboard.html        # Dashboard HTML template
└── static/
    ├── css/
    │   └── dashboard.css     # Dashboard styles
    └── js/
        └── dashboard.js      # Dashboard JavaScript
```

## Customization

### Adding New Sensors

1. **Modify Arduino sketch**
   - Add new sensor reading functions
   - Include new data in serial output

2. **Update Python parser**
   - Modify `parse_arduino_data()` function in `arduino_dashboard.py`
   - Add new sensor to `sensor_data` dictionary

3. **Update dashboard**
   - Add new data cards in `dashboard.html`
   - Update JavaScript in `dashboard.js`
   - Add new chart datasets

### Changing Dashboard Theme

Edit `static/css/dashboard.css` to customize:
- Colors and gradients
- Card styles and animations
- Typography and spacing
- Responsive breakpoints

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this project.

## License

This project is open source and available under the MIT License.

## Support

If you encounter any issues or have questions:
1. Check the troubleshooting section above
2. Review the Arduino serial monitor output
3. Check the Python console for error messages
4. Ensure all dependencies are properly installed 