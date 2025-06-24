import serial
import json
import time
import threading
from flask import Flask, render_template, jsonify, send_file
from flask_socketio import SocketIO, emit
import re
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, periodogram
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import io
import base64
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'arduino_dashboard_secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

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

# Global variable to store the latest analysis results
latest_analysis = {
    'bpm': 0,
    'confidence': 0,
    'spectrogram_b64': None,
    'time_domain_b64': None,
    'analysis_time': None
}

# Demo mode settings
DEMO_MODE = False  # Set to False to disable demo data
demo_data_counter = 0

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
        # Format 1: JSON {"temp": 25.5, "humidity": 60.2, "ppg": 258}
        if data_string.strip().startswith('{'):
            return json.loads(data_string)
        
        # Format 2: CSV temp,humidity,light,pressure,ppg
        elif ',' in data_string:
            values = data_string.strip().split(',')
            if len(values) >= 5:
                return {
                    'temperature': float(values[0]),
                    'humidity': float(values[1]),
                    'light': float(values[2]),
                    'pressure': float(values[3]),
                    'ppg': float(values[4])
                }
            elif len(values) >= 4:
                return {
                    'temperature': float(values[0]),
                    'humidity': float(values[1]),
                    'light': float(values[2]),
                    'pressure': float(values[3])
                }
        
        # Format 3: Key-value pairs temp:25.5,humidity:60.2,ppg:258
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

def analyze_csv_file(csv_path, fs=10):
    """
    Analyze CSV file using the same algorithm as newA.py
    """
    try:
        # Load the CSV
        df = pd.read_csv(csv_path, skiprows=1)
        ppg_signal = df["IR_Value"].astype(float).values
        
        print(f"Analyzing {csv_path}: {len(ppg_signal)} samples")
        
        # Check if signal is too short for filtering
        if len(ppg_signal) < 10:
            print("Warning: Signal is very short. Using simple frequency analysis.")
            # Use simple periodogram approach for short signals
            freqs, power = periodogram(ppg_signal, fs)
            
            # Limit to heart rate band (40-240 BPM = 0.67-4 Hz)
            mask = (freqs >= 0.67) & (freqs <= 4.0)
            if np.any(mask):
                freqs, power = freqs[mask], power[mask]
                peak_freq = freqs[np.argmax(power)]
                bpm = peak_freq * 60
                confidence = power.max() / np.sum(power)
            else:
                bpm = 0
                confidence = 0
                
            # Generate simple plots for short signals
            time_domain_b64 = generate_time_domain_plot(ppg_signal, None, fs, bpm, confidence)
            spectrogram_b64 = None
            
            return bpm, confidence, spectrogram_b64, time_domain_b64
        
        # For longer signals, use bandpass filtering
        # Bandpass filter (40–240 BPM = 0.67–4 Hz)
        low, high = 0.67, 4.0
        nyq = 0.5 * fs
        
        # Adjust filter order for short signals
        filter_order = min(2, len(ppg_signal) // 3 - 1)
        if filter_order < 1:
            filter_order = 1
        
        try:
            b, a = butter(filter_order, [low / nyq, high / nyq], btype="band")
            filtered = filtfilt(b, a, ppg_signal)
        except ValueError as e:
            print(f"Filtering failed: {e}. Using unfiltered signal.")
            filtered = ppg_signal

        # Generate spectrogram similar to main algorithm
        if len(filtered) > 8:
            # Calculate window and overlap parameters
            window_size = min(len(filtered), int(fs * 8))  # 8 second window
            overlap_size = int(window_size * 0.75)  # 75% overlap
            
            # Ensure minimum window size
            if window_size < 4:
                window_size = 4
                overlap_size = 2
            
            # Create spectrogram
            spec, freqs, times, im = plt.specgram(
                filtered, 
                NFFT=window_size, 
                Fs=fs, 
                noverlap=overlap_size,
                cmap='viridis'
            )
            
            # Find peak frequency and convert to BPM
            peak_freq = freqs[np.argmax(np.mean(spec, axis=1))]
            bpm = peak_freq * 60
            confidence = np.max(np.mean(spec, axis=1)) / np.sum(np.mean(spec, axis=1))
            
            # Generate plots
            spectrogram_b64 = generate_spectrogram_plot(filtered, fs, window_size, overlap_size, bpm)
            time_domain_b64 = generate_time_domain_plot(ppg_signal, filtered, fs, bpm, confidence)
            
            return bpm, confidence, spectrogram_b64, time_domain_b64
        else:
            # Simple periodogram for short signals
            freqs, power = periodogram(filtered, fs)
            mask = (freqs >= low) & (freqs <= high)
            if np.any(mask):
                freqs, power = freqs[mask], power[mask]
                peak_freq = freqs[np.argmax(power)]
                bpm = peak_freq * 60
                confidence = power.max() / np.sum(power)
                time_domain_b64 = generate_time_domain_plot(ppg_signal, filtered, fs, bpm, confidence)
                spectrogram_b64 = None
                return bpm, confidence, spectrogram_b64, time_domain_b64
            else:
                time_domain_b64 = generate_time_domain_plot(ppg_signal, filtered, fs, 0, 0)
                return 0, 0, None, time_domain_b64
        
    except Exception as e:
        print(f"CSV analysis error: {e}")
        return 0, 0, None, None

def generate_spectrogram_plot(filtered_signal, fs, window_size, overlap_size, bpm):
    """Generate spectrogram plot and return as base64"""
    try:
        plt.figure(figsize=(8, 6))
        plt.specgram(filtered_signal, NFFT=window_size, Fs=fs, noverlap=overlap_size, cmap='viridis')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency (BPM)')
        plt.title(f'PPG Signal Spectrogram - Estimated BPM: {bpm:.1f}')
        plt.axhline(y=bpm, color='red', linestyle='--', alpha=0.8, label=f'Estimated BPM: {bpm:.1f}')
        plt.legend()
        plt.ylim(40, 240)
        plt.tight_layout()
        
        # Convert plot to base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        spectrogram_b64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return spectrogram_b64
    except Exception as e:
        print(f"Error generating spectrogram: {e}")
        return None

def generate_time_domain_plot(ppg_signal, filtered_signal=None, fs=10, bpm=0, confidence=0):
    """Generate time domain plot and return as base64"""
    try:
        plt.figure(figsize=(10, 6))
        time_axis = np.arange(len(ppg_signal)) / fs
        
        if filtered_signal is not None:
            plt.subplot(2, 1, 1)
            plt.plot(time_axis, ppg_signal, 'b-', alpha=0.7, label='Raw PPG Signal')
            plt.xlabel('Time (seconds)')
            plt.ylabel('IR Value')
            plt.title('Raw PPG Signal from Arduino')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 1, 2)
            plt.plot(time_axis, filtered_signal, 'g-', alpha=0.7, label='Filtered PPG Signal')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Filtered Value')
            plt.title(f'Band-pass Filtered PPG Signal (40-240 BPM) - BPM: {bpm:.1f}, Confidence: {confidence:.3f}')
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            plt.plot(time_axis, ppg_signal, 'b-o', label='Raw PPG Signal')
            plt.xlabel('Time (seconds)')
            plt.ylabel('IR Value')
            plt.title(f'Raw PPG Signal - Estimated BPM: {bpm:.1f}, Confidence: {confidence:.3f}')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert plot to base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        time_domain_b64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return time_domain_b64
        
    except Exception as e:
        print(f"Error generating time domain plot: {e}")
        return None

def load_and_analyze_csv():
    """Load and analyze the A.csv file from the BPM directory"""
    global latest_analysis
    
    csv_path = os.path.join('..', 'BPM', 'A.csv')
    if os.path.exists(csv_path):
        print(f"Analyzing {csv_path}...")
        result = analyze_csv_file(csv_path)
        
        if result:
            # Handle different return formats from analyze_csv_file
            if len(result) == 4:
                bpm, confidence, spectrogram_b64, time_domain_b64 = result
            elif len(result) == 3:
                bpm, confidence, time_domain_b64 = result
                spectrogram_b64 = None
            else:
                print(f"Unexpected result format: {result}")
                return None
            
            latest_analysis = {
                'bpm': bpm,
                'confidence': confidence,
                'spectrogram_b64': spectrogram_b64,
                'time_domain_b64': time_domain_b64,
                'analysis_time': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            print(f"Analysis complete: BPM={bpm:.1f}, Confidence={confidence:.3f}")
            
            # Emit the analysis results to connected clients
            socketio.emit('csv_analysis_result', latest_analysis)
            
            return latest_analysis
        else:
            print(f"CSV analysis failed")
            return None
    else:
        print(f"CSV file not found: {csv_path}")
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
                            current_time = time.time()
                            parsed_data['timestamp'] = current_time
                            
                            # Update sensor data
                            for key, value in parsed_data.items():
                                if key in sensor_data:
                                    sensor_data[key].append(value)
                                    # Keep only last 100 data points
                                    if len(sensor_data[key]) > 100:
                                        sensor_data[key].pop(0)
                            
                            # Emit sensor data to web clients
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

@app.route('/api/csv_analysis')
def get_csv_analysis():
    """API endpoint to get the latest CSV analysis results"""
    return jsonify(latest_analysis)

@app.route('/api/analyze_csv')
def trigger_csv_analysis():
    """API endpoint to trigger CSV analysis"""
    result = load_and_analyze_csv()
    if result:
        return jsonify({'success': True, 'data': result})
    else:
        return jsonify({'success': False, 'error': 'CSV file not found'})

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    
    if is_connected:
        emit('connection_status', {'connected': True, 'message': 'Arduino connected'})
    elif DEMO_MODE:
        emit('connection_status', {'connected': False, 'message': 'Demo mode active - showing sample data'})
    else:
        emit('connection_status', {'connected': False, 'message': 'No Arduino connected'})
    
    # Send current CSV analysis if available
    if latest_analysis['analysis_time']:
        emit('csv_analysis_result', latest_analysis)

@socketio.on('request_data')
def handle_data_request():
    """Handle data request from client"""
    if sensor_data['temperature']:
        latest_data = {
            'temperature': sensor_data['temperature'][-1],
            'humidity': sensor_data['humidity'][-1],
            'light': sensor_data['light'][-1],
            'pressure': sensor_data['pressure'][-1],
            'timestamp': sensor_data['timestamp'][-1]
        }
        emit('arduino_data', latest_data)

@socketio.on('request_csv_analysis')
def handle_csv_analysis_request():
    """Handle CSV analysis request from client"""
    load_and_analyze_csv()

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

@app.route('/analyze_csv', methods=['POST'])
def analyze_csv():
    """Analyze uploaded CSV file for heart rate"""
    try:
        from flask import request
        if 'file' not in request.files:
            return {'error': 'No file uploaded'}, 400
        
        file = request.files['file']
        if file.filename == '':
            return {'error': 'No file selected'}, 400
        
        # Save uploaded file temporarily
        temp_path = os.path.join('/tmp', 'uploaded_csv.csv')
        file.save(temp_path)
        
        # Analyze the file
        bpm, confidence, spectrogram_b64, time_domain_b64 = analyze_csv_file(temp_path, fs=10)
        
        # Clean up
        os.remove(temp_path)
        
        return {
            'bpm': float(bpm),
            'confidence': float(confidence),
            'spectrogram': spectrogram_b64,
            'time_domain': time_domain_b64,
            'signal_length': 0,  # Will be calculated from the signal
            'duration': 0  # Will be calculated from the signal
        }
        
    except Exception as e:
        return {'error': str(e)}, 500

if __name__ == '__main__':
    # Start Arduino data reading thread
    arduino_thread = threading.Thread(target=read_arduino_data, daemon=True)
    arduino_thread.start()
    
    # Load and analyze CSV file on startup
    print("Loading and analyzing A.csv file...")
    load_and_analyze_csv()
    
    print("Arduino Dashboard starting...")
    print("Open http://localhost:5002 in your browser")
    socketio.run(app, host='0.0.0.0', port=5002, debug=True) 