import serial
import json
import time
import threading
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import re
import random
from datetime import datetime
import os

# Import heat stroke assessment module
from heat_stroke_assessment import HeatStrokeAssessment

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
arduino_connected = False
arduino_port = None
user_profiles = {}
assessment_history = []

# Initialize heat stroke assessment
heat_stroke_assessor = HeatStrokeAssessment()

# Arduino connection settings
ARDUINO_PORT = '/dev/tty.usbmodem*'  # Common on macOS, adjust for your system
BAUD_RATE = 9600

# Data storage
sensor_data = {
    'heartRate': [],
    'temperature': [],
    'humidity': [],
    'activity': [],
    'timestamp': []
}

def find_arduino_port():
    """Find the Arduino port automatically"""
    import glob
    ports = glob.glob('/dev/tty.usbmodem*') + glob.glob('/dev/ttyACM*') + glob.glob('COM*')
    return ports[0] if ports else None

def connect_arduino():
    """Connect to Arduino"""
    global arduino_connected, arduino_port
    try:
        port = find_arduino_port()
        if port:
            arduino_port = serial.Serial(port, BAUD_RATE, timeout=1)
            arduino_connected = True
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
        # Format 1: JSON {"heartRate": 75, "temperature": 25.5, "humidity": 60.2, "activity": 512}
        if data_string.strip().startswith('{'):
            return json.loads(data_string)
        
        # Format 2: CSV heartRate,temperature,humidity,activity
        elif ',' in data_string:
            values = data_string.strip().split(',')
            if len(values) >= 4:
                return {
                    'heartRate': float(values[0]),
                    'temperature': float(values[1]),
                    'humidity': float(values[2]),
                    'activity': float(values[3])
                }
        
        # Format 3: Key-value pairs heartRate:75,temperature:25.5,humidity:60.2,activity:512
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
    global arduino_connected, arduino_port
    
    while arduino_connected and arduino_port:
        try:
            if arduino_port.in_waiting > 0:
                data = arduino_port.readline().decode('utf-8').strip()
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
            arduino_connected = False
            socketio.emit('connection_status', {'connected': False, 'message': 'Arduino connection lost'})
            break

@app.route('/')
def dashboard():
    """Main dashboard route - unified interface"""
    return render_template('dashboard.html')

@app.route('/assessment')
def assessment():
    """Legacy assessment route - redirects to main dashboard"""
    return render_template('dashboard.html')

@app.route('/api/profile', methods=['GET', 'POST'])
def api_profile():
    """API endpoint for user profile management"""
    if request.method == 'POST':
        data = request.get_json()
        user_id = request.headers.get('X-User-ID', 'default')
        user_profiles[user_id] = data
        return jsonify({'status': 'success', 'message': 'Profile saved'})
    else:
        user_id = request.headers.get('X-User-ID', 'default')
        return jsonify(user_profiles.get(user_id, {}))

@app.route('/api/assessment-history')
def api_assessment_history():
    """API endpoint for assessment history"""
    return jsonify(assessment_history)

@app.route('/api/health-data')
def api_health_data():
    """API endpoint for current health data"""
    # Return simulated data for now
    data = {
        'heart_rate': random.randint(60, 100),
        'temperature': round(random.uniform(36.5, 37.5), 1),
        'humidity': random.randint(40, 70),
        'activity_level': random.choice(['resting', 'light', 'moderate', 'strenuous']),
        'timestamp': datetime.now().isoformat()
    }
    return jsonify(data)

@app.route('/api/assessment', methods=['POST'])
def api_assessment():
    """API endpoint for heat stroke assessment"""
    try:
        data = request.get_json()
        result = heat_stroke_assessor.assess_heat_stroke_risk(data)
        
        # Store assessment
        assessment_record = {
            'timestamp': datetime.now().isoformat(),
            'data': data,
            'result': result
        }
        assessment_history.append(assessment_record)
        
        return jsonify({
            'status': 'success',
            'result': result,
            'timestamp': assessment_record['timestamp']
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('connection_status', {'connected': arduino_connected, 'message': 'Connected to server'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('connect_arduino')
def handle_connect_arduino():
    global arduino_connected, arduino_port
    try:
        # Try to connect to Arduino
        arduino_port = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
        arduino_connected = True
        emit('connection_status', {'connected': True, 'message': 'Arduino connected successfully'})
        
        # Start data reading thread
        threading.Thread(target=read_arduino_data, daemon=True).start()
        
    except Exception as e:
        print(f"Failed to connect to Arduino: {e}")
        emit('connection_status', {'connected': False, 'message': f'Failed to connect: {str(e)}'})
        
        # Start simulation mode if Arduino connection fails
        threading.Thread(target=simulate_arduino_data, daemon=True).start()

@socketio.on('disconnect_arduino')
def handle_disconnect_arduino():
    global arduino_connected, arduino_port
    if arduino_port:
        arduino_port.close()
    arduino_connected = False
    emit('connection_status', {'connected': False, 'message': 'Arduino disconnected'})

@socketio.on('save_profile')
def handle_save_profile(data):
    """Save user profile data"""
    user_id = request.sid
    user_profiles[user_id] = data
    emit('profile_saved', {'status': 'success', 'message': 'Profile saved successfully'})

@socketio.on('heat_stroke_assessment')
def handle_heat_stroke_assessment(data):
    """Handle heat stroke assessment submission"""
    try:
        assessment_data = data.get('data', {})
        client_result = data.get('result', {})
        
        # Perform server-side assessment for validation
        server_result = heat_stroke_assessor.assess_heat_stroke_risk(assessment_data)
        
        # Store assessment in history
        assessment_record = {
            'timestamp': datetime.now().isoformat(),
            'data': assessment_data,
            'client_result': client_result,
            'server_result': server_result
        }
        assessment_history.append(assessment_record)
        
        # Keep only last 100 assessments
        if len(assessment_history) > 100:
            assessment_history.pop(0)
        
        # Emit assessment result
        emit('assessment_result', {
            'status': 'success',
            'result': server_result,
            'timestamp': assessment_record['timestamp']
        })
        
        # Emit emergency alert if critical
        if server_result.get('risk_level') == 'critical':
            emit('emergency_alert', {
                'type': 'heat_stroke_critical',
                'message': 'CRITICAL HEAT STROKE RISK DETECTED - IMMEDIATE MEDICAL ATTENTION REQUIRED',
                'data': assessment_data
            })
            
    except Exception as e:
        print(f"Error processing heat stroke assessment: {e}")
        emit('assessment_result', {
            'status': 'error',
            'message': f'Assessment error: {str(e)}'
        })

def simulate_arduino_data():
    """Simulate Arduino data for testing"""
    print("Starting Arduino data simulation...")
    
    while True:
        try:
            # Simulate realistic health data
            data = {
                'heart_rate': random.randint(60, 100),
                'temperature': round(random.uniform(36.5, 37.5), 1),
                'humidity': random.randint(40, 70),
                'activity_level': random.choice(['resting', 'light', 'moderate', 'strenuous']),
                'timestamp': datetime.now().isoformat()
            }
            
            socketio.emit('arduino_data', data)
            time.sleep(2)  # Update every 2 seconds
            
        except Exception as e:
            print(f"Error in data simulation: {e}")
            time.sleep(5)

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# Health check endpoint
@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'arduino_connected': arduino_connected,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    print("Starting Health Monitoring Dashboard...")
    print("Dashboard will be available at: http://localhost:5000")
    print("Health check available at: http://localhost:5000/health")
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=True) 