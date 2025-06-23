import serial
import json
import time
import threading
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import re

app = Flask(__name__)
app.config['SECRET_KEY'] = 'arduino_dashboard_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Arduino connection settings
ARDUINO_PORT = '/dev/tty.usbmodem*'  # Common on macOS, adjust for your system
BAUD_RATE = 9600
arduino = None
is_connected = False

# Data storage
sensor_data = {
    'temperature': [],
    'humidity': [],
    'light': [],
    'pressure': [],
    'timestamp': []
}

def find_arduino_port():
    """Find the Arduino port automatically"""
    import glob
    ports = glob.glob('/dev/tty.usbmodem*') + glob.glob('/dev/ttyACM*') + glob.glob('COM*')
    return ports[0] if ports else None

def connect_arduino():
    """Connect to Arduino"""
    global arduino, is_connected
    try:
        port = find_arduino_port()
        if port:
            arduino = serial.Serial(port, BAUD_RATE, timeout=1)
            is_connected = True
            print(f"Connected to Arduino on {port}")
            return True
        else:
            print("No Arduino found. Please check connection.")
            return False
    except Exception as e:
        print(f"Error connecting to Arduino: {e}")
        return False

def parse_arduino_data(data_string):
    """Parse data from Arduino string format"""
    try:
        # Common Arduino data formats
        # Format 1: JSON {"temp": 25.5, "humidity": 60.2}
        if data_string.strip().startswith('{'):
            return json.loads(data_string)
        
        # Format 2: CSV temp,humidity,light,pressure
        elif ',' in data_string:
            values = data_string.strip().split(',')
            if len(values) >= 4:
                return {
                    'temperature': float(values[0]),
                    'humidity': float(values[1]),
                    'light': float(values[2]),
                    'pressure': float(values[3])
                }
        
        # Format 3: Key-value pairs temp:25.5,humidity:60.2
        elif ':' in data_string:
            data = {}
            pairs = data_string.strip().split(',')
            for pair in pairs:
                if ':' in pair:
                    key, value = pair.split(':')
                    data[key.strip()] = float(value.strip())
            return data
        
        return None
    except Exception as e:
        print(f"Error parsing data: {e}")
        return None

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
                            
                            # Update sensor data
                            for key, value in parsed_data.items():
                                if key in sensor_data:
                                    sensor_data[key].append(value)
                                    # Keep only last 100 data points
                                    if len(sensor_data[key]) > 100:
                                        sensor_data[key].pop(0)
                            
                            # Emit data to web clients
                            socketio.emit('arduino_data', parsed_data)
                            print(f"Received: {parsed_data}")
                
                time.sleep(0.1)  # Small delay to prevent CPU overuse
                
            except Exception as e:
                print(f"Error reading from Arduino: {e}")
                is_connected = False
                break
        else:
            time.sleep(1)  # Wait before trying to reconnect

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('connection_status', {'connected': is_connected})

@socketio.on('request_data')
def handle_data_request():
    """Send current sensor data to client"""
    emit('sensor_data', sensor_data)

@socketio.on('connect_arduino')
def handle_arduino_connect():
    """Handle Arduino connection request"""
    global is_connected
    if connect_arduino():
        is_connected = True
        emit('connection_status', {'connected': True, 'message': 'Arduino connected successfully'})
    else:
        emit('connection_status', {'connected': False, 'message': 'Failed to connect to Arduino'})

@socketio.on('disconnect_arduino')
def handle_arduino_disconnect():
    """Handle Arduino disconnection request"""
    global arduino, is_connected
    if arduino:
        arduino.close()
        arduino = None
    is_connected = False
    emit('connection_status', {'connected': False, 'message': 'Arduino disconnected'})

if __name__ == '__main__':
    # Start Arduino data reading thread
    arduino_thread = threading.Thread(target=read_arduino_data, daemon=True)
    arduino_thread.start()
    
    print("Arduino Dashboard starting...")
    print("Open http://localhost:5000 in your browser")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True) 