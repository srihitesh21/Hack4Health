import serial
import json
import time
import threading
from flask import Flask, render_template, jsonify, send_file, request
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
from datetime import datetime

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
    'analysis_time': None,
    'infection_score': 0.0,
    'dehydration_score': 0.0,
    'arrhythmia_score': 0.0
}

# Store combined health assessment data (demographics + health questionnaire)
demographics_data = []

# Demo mode settings
DEMO_MODE = False  # Set to False to disable demo data
demo_data_counter = 0

# Global variables
arduino_connected = False
arduino_port = None
demo_mode = True
csv_analysis_results = None

def compute_pvr_score_components(pulse_rates, skin_temps=None):
    """
    Compute health risk scores based on pulse rate and skin temperature data.
    
    Args:
        pulse_rates (list or np.array): Array of pulse rate values in BPM
        skin_temps (list or np.array, optional): Array of skin temperature values in °C
        
    Returns:
        tuple: (infection_score, dehydration_score, arrhythmia_score)
    """
    pulse_rates = np.array(pulse_rates)
    pulse_rates = pulse_rates[pulse_rates > 0]  # Remove invalid values
    
    if len(pulse_rates) < 2:
        return 0.0, 0.0, 0.0

    mean_hr = np.mean(pulse_rates)
    std_long = np.std(pulse_rates)
    std_short = np.std(np.diff(pulse_rates))

    # Risk Score 1: Infection Risk (elevated HR + elevated skin temp)
    infection_score = 0.0
    if mean_hr > 90:
        infection_score += 0.7  # Heart rate component
    if skin_temps is not None and len(skin_temps) > 0:
        mean_temp = np.mean(skin_temps)
        if mean_temp > 37.5:  # Above 37.5°C indicates fever
            infection_score += 0.3  # Temperature component
    if infection_score > 0:
        infection_score = 1.0  # Normalize to 1.0 if any risk detected

    # Risk Score 2: Dehydration Risk (elevated HR + low HRV + low skin temp)
    dehydration_score = 0.0
    if mean_hr > 85 and std_long < 1.0 and std_short < 0.5:
        dehydration_score += 0.6  # Heart rate component
    if skin_temps is not None and len(skin_temps) > 0:
        mean_temp = np.mean(skin_temps)
        if mean_temp < 35.5:  # Below 35.5°C indicates poor circulation/dehydration
            dehydration_score += 0.4  # Temperature component
    if dehydration_score > 0:
        dehydration_score = 1.0  # Normalize to 1.0 if any risk detected

    # Risk Score 3: Arrhythmia Risk (abnormal values - no clear temp relationship)
    if np.min(pulse_rates) < 45 or np.max(pulse_rates) > 130 or std_short > 5:
        arrhythmia_score = 1.0
    else:
        arrhythmia_score = 0.0

    return infection_score, dehydration_score, arrhythmia_score

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
        skin_temps = df["SkinTemp(C)"].astype(float).values  # Extract skin temperature data
        
        print(f"Analyzing {csv_path}: {len(ppg_signal)} samples")
        print(f"Skin temperature range: {np.min(skin_temps):.1f}°C - {np.max(skin_temps):.1f}°C")
        print(f"Average skin temperature: {np.mean(skin_temps):.1f}°C")
        
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
                
            # Calculate health risk scores
            if bpm > 0:
                # Create a pulse rate array for risk assessment
                pulse_rates = [bpm + np.random.normal(0, 2) for _ in range(10)]
                infection_score, dehydration_score, arrhythmia_score = compute_pvr_score_components(pulse_rates, skin_temps)
            else:
                infection_score, dehydration_score, arrhythmia_score = 0.0, 0.0, 0.0
                
            # Generate simple plots for short signals
            time_domain_b64 = generate_time_domain_plot(ppg_signal, None, fs, bpm, confidence)
            spectrogram_b64 = None
            
            return bpm, confidence, spectrogram_b64, time_domain_b64, infection_score, dehydration_score, arrhythmia_score
        
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
            
            # Calculate health risk scores
            if bpm > 0:
                # Create a pulse rate array for risk assessment
                pulse_rates = [bpm + np.random.normal(0, 2) for _ in range(10)]
                infection_score, dehydration_score, arrhythmia_score = compute_pvr_score_components(pulse_rates, skin_temps)
            else:
                infection_score, dehydration_score, arrhythmia_score = 0.0, 0.0, 0.0
            
            # Generate plots
            spectrogram_b64 = generate_spectrogram_plot(filtered, fs, window_size, overlap_size, bpm)
            time_domain_b64 = generate_time_domain_plot(ppg_signal, filtered, fs, bpm, confidence)
            
            return bpm, confidence, spectrogram_b64, time_domain_b64, infection_score, dehydration_score, arrhythmia_score
        else:
            # Simple periodogram for short signals
            freqs, power = periodogram(filtered, fs)
            mask = (freqs >= low) & (freqs <= high)
            if np.any(mask):
                freqs, power = freqs[mask], power[mask]
                peak_freq = freqs[np.argmax(power)]
                bpm = peak_freq * 60
                confidence = power.max() / np.sum(power)
                
                # Calculate health risk scores
                if bpm > 0:
                    # Create a pulse rate array for risk assessment
                    pulse_rates = [bpm + np.random.normal(0, 2) for _ in range(10)]
                    infection_score, dehydration_score, arrhythmia_score = compute_pvr_score_components(pulse_rates, skin_temps)
                else:
                    infection_score, dehydration_score, arrhythmia_score = 0.0, 0.0, 0.0
                
                time_domain_b64 = generate_time_domain_plot(ppg_signal, filtered, fs, bpm, confidence)
                spectrogram_b64 = None
                return bpm, confidence, spectrogram_b64, time_domain_b64, infection_score, dehydration_score, arrhythmia_score
            else:
                time_domain_b64 = generate_time_domain_plot(ppg_signal, filtered, fs, 0, 0)
                return 0, 0, None, time_domain_b64, 0.0, 0.0, 0.0
        
    except Exception as e:
        print(f"CSV analysis error: {e}")
        return 0, 0, None, None, 0.0, 0.0, 0.0

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
            if len(result) == 7:
                bpm, confidence, spectrogram_b64, time_domain_b64, infection_score, dehydration_score, arrhythmia_score = result
            elif len(result) == 4:
                bpm, confidence, spectrogram_b64, time_domain_b64 = result
                infection_score, dehydration_score, arrhythmia_score = 0.0, 0.0, 0.0
            elif len(result) == 3:
                bpm, confidence, time_domain_b64 = result
                spectrogram_b64 = None
                infection_score, dehydration_score, arrhythmia_score = 0.0, 0.0, 0.0
            else:
                print(f"Unexpected result format: {result}")
                return None
            
            latest_analysis = {
                'bpm': bpm,
                'confidence': confidence,
                'spectrogram_b64': spectrogram_b64,
                'time_domain_b64': time_domain_b64,
                'analysis_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'infection_score': infection_score,
                'dehydration_score': dehydration_score,
                'arrhythmia_score': arrhythmia_score
            }
            
            print(f"Analysis complete: BPM={bpm:.1f}, Confidence={confidence:.3f}")
            print(f"Risk Scores - Infection: {infection_score:.1f}, Dehydration: {dehydration_score:.1f}, Arrhythmia: {arrhythmia_score:.1f}")
            
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
        result = analyze_csv_file(temp_path, fs=10)
        
        # Clean up
        os.remove(temp_path)
        
        if len(result) == 7:
            bpm, confidence, spectrogram_b64, time_domain_b64, infection_score, dehydration_score, arrhythmia_score = result
        else:
            bpm, confidence, spectrogram_b64, time_domain_b64 = result[:4]
            infection_score, dehydration_score, arrhythmia_score = 0.0, 0.0, 0.0
        
        return {
            'bpm': float(bpm),
            'confidence': float(confidence),
            'spectrogram': spectrogram_b64,
            'time_domain': time_domain_b64,
            'infection_score': float(infection_score),
            'dehydration_score': float(dehydration_score),
            'arrhythmia_score': float(arrhythmia_score),
            'signal_length': 0,  # Will be calculated from the signal
            'duration': 0  # Will be calculated from the signal
        }
        
    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/health_assessment', methods=['GET', 'POST'])
def health_assessment():
    """Health Risk Assessment page"""
    if request.method == 'POST':
        data = request.get_json()
        if data:
            # Store combined assessment data
            assessment_data = {
                'age': data.get('age'),
                'gender': data.get('gender'),
                'risk_factors': {
                    'high_blood_pressure': data.get('high_blood_pressure'),
                    'diabetes': data.get('diabetes'),
                    'smoking': data.get('smoking'),
                    'obesity': data.get('obesity'),
                    'family_history': data.get('family_history'),
                    'physical_inactivity': data.get('physical_inactivity'),
                    'high_cholesterol': data.get('high_cholesterol'),
                    'previous_stroke': data.get('previous_stroke')
                },
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            demographics_data.append(assessment_data)
            return jsonify({'success': True, 'message': 'Assessment submitted successfully'})
    
    return render_template('demographics.html', assessments=demographics_data)

@app.route('/demographics', methods=['GET', 'POST'])
def demographics():
    """Demographics/Health Assessment page - alias for health_assessment"""
    return health_assessment()

@app.route('/api/demographics')
def get_demographics():
    """API endpoint to get stored demographics data"""
    return jsonify(demographics_data)

@app.route('/submit_demographics', methods=['POST'])
def submit_demographics():
    """Handle demographics form submission"""
    try:
        data = request.get_json()
        if data:
            # Store combined assessment data
            assessment_data = {
                'age': data.get('age'),
                'gender': data.get('gender'),
                'symptoms': data.get('symptoms', []),
                'medical_history': data.get('medical_history', []),
                'risk_factors': data.get('risk_factors', []),
                'timestamp': time.time()
            }
            
            # Add the latest risk scores from CSV analysis
            if csv_analysis_results:
                assessment_data['risk_scores'] = {
                    'infection_score': csv_analysis_results.get('infection_score', 0.0),
                    'dehydration_score': csv_analysis_results.get('dehydration_score', 0.0),
                    'arrhythmia_score': csv_analysis_results.get('arrhythmia_score', 0.0)
                }
            
            demographics_data.append(assessment_data)
            
            # Keep only the last 10 assessments
            if len(demographics_data) > 10:
                demographics_data.pop(0)
            
            return jsonify({
                'success': True,
                'message': 'Assessment submitted successfully',
                'risk_scores': assessment_data.get('risk_scores', {})
            })
        else:
            return jsonify({'success': False, 'error': 'No data received'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

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