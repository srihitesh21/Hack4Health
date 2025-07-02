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
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
import requests

# Free weather and location APIs (no API keys required)
WEATHER_BASE_URL = "https://api.open-meteo.com/v1/forecast"
LOCATION_BASE_URL = "http://ip-api.com/json"

def get_weather_data(lat=None, lon=None, city=None):
    """
    Get current weather data for a location using free Open-Meteo API
    Args:
        lat: latitude (optional)
        lon: longitude (optional) 
        city: city name (optional)
    Returns:
        dict: weather data including temperature, location, etc.
    """
    try:
        # If city is provided, we need to geocode it first
        if city and not lat and not lon:
            # Use a free geocoding service
            geocode_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
            geocode_response = requests.get(geocode_url, timeout=10)
            if geocode_response.status_code == 200:
                geocode_data = geocode_response.json()
                if geocode_data.get('results'):
                    lat = geocode_data['results'][0]['latitude']
                    lon = geocode_data['results'][0]['longitude']
                    city_name = geocode_data['results'][0]['name']
                else:
                    # Fallback to default location
                    lat = 40.7128
                    lon = -74.0060
                    city_name = "New York"
            else:
                # Fallback to default location
                lat = 40.7128
                lon = -74.0060
                city_name = "New York"
        elif not lat and not lon:
            # Default to New York if no coordinates provided
            lat = 40.7128
            lon = -74.0060
            city_name = "New York"
        else:
            # Try to get location name from coordinates using reverse geocoding
            try:
                reverse_geocode_url = f"https://geocoding-api.open-meteo.com/v1/search?latitude={lat}&longitude={lon}&count=1"
                reverse_response = requests.get(reverse_geocode_url, timeout=5)
                if reverse_response.status_code == 200:
                    reverse_data = reverse_response.json()
                    if reverse_data.get('results'):
                        city_name = reverse_data['results'][0]['name']
                    else:
                        city_name = f"Location ({lat:.2f}, {lon:.2f})"
                else:
                    city_name = f"Location ({lat:.2f}, {lon:.2f})"
            except:
                city_name = f"Location ({lat:.2f}, {lon:.2f})"
        
        # Get weather data from Open-Meteo API
        params = {
            'latitude': lat,
            'longitude': lon,
            'current': 'temperature_2m,relative_humidity_2m,apparent_temperature,pressure_msl,weather_code',
            'timezone': 'auto'
        }
        
        response = requests.get(WEATHER_BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        
        weather_data = response.json()
        current = weather_data['current']
        
        # Map weather codes to descriptions
        weather_descriptions = {
            0: "Clear sky",
            1: "Mainly clear",
            2: "Partly cloudy",
            3: "Overcast",
            45: "Foggy",
            48: "Depositing rime fog",
            51: "Light drizzle",
            53: "Moderate drizzle",
            55: "Dense drizzle",
            61: "Slight rain",
            63: "Moderate rain",
            65: "Heavy rain",
            71: "Slight snow",
            73: "Moderate snow",
            75: "Heavy snow",
            77: "Snow grains",
            80: "Slight rain showers",
            81: "Moderate rain showers",
            82: "Violent rain showers",
            85: "Slight snow showers",
            86: "Heavy snow showers",
            95: "Thunderstorm",
            96: "Thunderstorm with slight hail",
            99: "Thunderstorm with heavy hail"
        }
        
        weather_code = current.get('weather_code', 0)
        weather_description = weather_descriptions.get(weather_code, "Unknown")
        
        return {
            'temperature': current.get('temperature_2m'),
            'feels_like': current.get('apparent_temperature'),
            'humidity': current.get('relative_humidity_2m'),
            'pressure': current.get('pressure_msl'),
            'description': weather_description,
            'location': city_name,
            'country': 'Unknown',  # Open-Meteo doesn't provide country info
            'latitude': lat,
            'longitude': lon,
            'timestamp': datetime.now().isoformat()
        }
        
    except requests.RequestException as e:
        print(f"⚠️ Weather API error: {e}")
        return {
            'error': 'Weather data unavailable',
            'temperature': None,
            'location': 'Unknown',
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        print(f"⚠️ Weather data processing error: {e}")
        return {
            'error': 'Weather data processing failed',
            'temperature': None,
            'location': 'Unknown',
            'timestamp': datetime.now().isoformat()
        }

def get_user_location():
    """
    Get user's location using free IP geolocation service
    Returns:
        dict: location data
    """
    try:
        # Use free IP geolocation service
        response = requests.get(LOCATION_BASE_URL, timeout=5)
        if response.status_code == 200:
            data = response.json()
            location_data = {
                'city': data.get('city', 'Unknown'),
                'region': data.get('regionName', 'Unknown'),
                'country': data.get('country', 'Unknown'),
                'lat': data.get('lat'),
                'lon': data.get('lon')
            }
            print(f"🌍 IP Location detected: {location_data['city']}, {location_data['region']}, {location_data['country']}")
            return location_data
    except Exception as e:
        print(f"⚠️ Location detection error: {e}")
    
    print("⚠️ Using fallback location data")
    return {
        'city': 'Unknown',
        'region': 'Unknown', 
        'country': 'Unknown',
        'lat': None,
        'lon': None
    }

# MediaPipe disabled for camera compatibility
MEDIAPIPE_AVAILABLE = False
mediapipe_model = None
print("📷 MediaPipe disabled - using traditional analysis for camera compatibility")

# Import stress fatigue detector
try:
    from stress_fatigue_detector import FacialWellnessAnalyzer
    STRESS_FATIGUE_AVAILABLE = True
    stress_fatigue_detector = None
    print("✅ Stress fatigue detection available")
except ImportError as e:
    STRESS_FATIGUE_AVAILABLE = False
    print(f"⚠️ Stress fatigue detection not available: {e}")

# Import somnolence detection
try:
    import sys
    sys.path.append('../somnolence-detection')
    from somnolence_detection import DrowsinessDetector
    SOMNOLENCE_AVAILABLE = True
    somnolence_detector = None
    print("✅ Somnolence detection available")
except ImportError as e:
    SOMNOLENCE_AVAILABLE = False
    print(f"⚠️ Somnolence detection not available: {e}")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'arduino_dashboard_secret'
# Use threading mode to avoid eventlet SSL issues with Python 3.12
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', logger=False, engineio_logger=False)

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

# Load existing health assessment data from file
def load_health_assessment_data():
    """Load health assessment data from file if it exists"""
    global demographics_data
    try:
        if os.path.exists('health_assessment_data.json'):
            with open('health_assessment_data.json', 'r') as f:
                demographics_data = json.load(f)
            print(f"✅ Loaded {len(demographics_data)} health assessments from file")
        else:
            print("📝 No existing health assessment data file found")
    except Exception as e:
        print(f"⚠️  Warning: Could not load health assessment data: {e}")
        demographics_data = []

# Load data on startup
load_health_assessment_data()

# Demo mode settings
DEMO_MODE = False  # Set to False to disable demo data
demo_data_counter = 0

# Global variables
arduino_connected = False
arduino_port = None
demo_mode = True
csv_analysis_results = None

# Initialize heatstroke predictor globally
heatstroke_predictor = None

def initialize_stress_fatigue_detector():
    """Initialize the stress fatigue detector if available"""
    global stress_fatigue_detector
    if STRESS_FATIGUE_AVAILABLE and stress_fatigue_detector is None:
        try:
            stress_fatigue_detector = FacialWellnessAnalyzer()
            print("✅ Stress fatigue detector initialized")
            return True
        except Exception as e:
            print(f"⚠️ Failed to initialize stress fatigue detector: {e}")
            return False
    return STRESS_FATIGUE_AVAILABLE

class PretrainedStressFatigueModel:
    """
    Pretrained model for stress and fatigue detection using ensemble learning
    Combines multiple ML models for robust predictions
    """
    
    def __init__(self):
        self.stress_model = None
        self.fatigue_model = None
        self.feature_scaler = None
        self.is_loaded = False
        self.model_path = os.path.join(os.path.dirname(__file__), 'pretrained_models')
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_path, exist_ok=True)
        
        # Initialize or load models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize or load pretrained models"""
        try:
            # Try to load existing models
            if self._load_models():
                print("✅ Loaded pretrained stress/fatigue models")
                self.is_loaded = True
            else:
                # Train new models if none exist
                print("🔄 Training new stress/fatigue models...")
                self._train_models()
                self.is_loaded = True
        except Exception as e:
            print(f"⚠️ Model initialization failed: {e}")
            self.is_loaded = False
    
    def _load_models(self):
        """Load pretrained models from disk"""
        try:
            model_files = [
                'stress_model.pkl',
                'fatigue_model.pkl', 
                'feature_scaler.pkl'
            ]
            
            # Check if all model files exist
            for file in model_files:
                if not os.path.exists(os.path.join(self.model_path, file)):
                    return False
            
            # Load models
            with open(os.path.join(self.model_path, 'stress_model.pkl'), 'rb') as f:
                self.stress_model = pickle.load(f)
            
            with open(os.path.join(self.model_path, 'fatigue_model.pkl'), 'rb') as f:
                self.fatigue_model = pickle.load(f)
            
            with open(os.path.join(self.model_path, 'feature_scaler.pkl'), 'rb') as f:
                self.feature_scaler = pickle.load(f)
            
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def _save_models(self):
        """Save trained models to disk"""
        try:
            with open(os.path.join(self.model_path, 'stress_model.pkl'), 'wb') as f:
                pickle.dump(self.stress_model, f)
            
            with open(os.path.join(self.model_path, 'fatigue_model.pkl'), 'wb') as f:
                pickle.dump(self.fatigue_model, f)
            
            with open(os.path.join(self.model_path, 'feature_scaler.pkl'), 'wb') as f:
                pickle.dump(self.feature_scaler, f)
            
            print("✅ Models saved successfully")
            
        except Exception as e:
            print(f"Error saving models: {e}")
    
    def _generate_training_data(self, num_samples=1000):
        """Generate synthetic training data based on medical research"""
        np.random.seed(42)  # For reproducibility
        
        data = []
        labels_stress = []
        labels_fatigue = []
        
        for i in range(num_samples):
            # Generate realistic feature combinations
            sample = self._generate_realistic_sample()
            data.append(sample)
            
            # Generate labels based on feature patterns
            stress_label = self._calculate_stress_label(sample)
            fatigue_label = self._calculate_fatigue_label(sample)
            
            labels_stress.append(stress_label)
            labels_fatigue.append(fatigue_label)
        
        return np.array(data), np.array(labels_stress), np.array(labels_fatigue)
    
    def _generate_realistic_sample(self):
        """Generate a realistic feature sample"""
        # Facial features (0-1 scale)
        eye_openness = np.random.normal(0.6, 0.2)
        eye_openness = np.clip(eye_openness, 0.1, 1.0)
        
        mouth_tension = np.random.normal(0.4, 0.2)
        mouth_tension = np.clip(mouth_tension, 0.0, 1.0)
        
        brow_furrow = np.random.normal(0.3, 0.2)
        brow_furrow = np.clip(brow_furrow, 0.0, 1.0)
        
        jaw_clenching = np.random.normal(0.3, 0.2)
        jaw_clenching = np.clip(jaw_clenching, 0.0, 1.0)
        
        blink_rate = np.random.normal(0.5, 0.2)
        blink_rate = np.clip(blink_rate, 0.1, 1.0)
        
        pupil_dilation = np.random.normal(0.4, 0.2)
        pupil_dilation = np.clip(pupil_dilation, 0.1, 1.0)
        
        # Physiological features
        heart_rate = np.random.normal(75, 15)
        heart_rate = np.clip(heart_rate, 50, 120)
        
        hrv = np.random.normal(40, 15)
        hrv = np.clip(hrv, 10, 80)
        
        skin_temp = np.random.normal(36.5, 1.0)
        skin_temp = np.clip(skin_temp, 34.0, 39.0)
        
        respiration_rate = np.random.normal(16, 4)
        respiration_rate = np.clip(respiration_rate, 10, 30)
        
        # Lifestyle features
        age = np.random.randint(18, 80)
        sleep_hours = np.random.normal(7, 2)
        sleep_hours = np.clip(sleep_hours, 4, 12)
        
        work_hours = np.random.normal(8, 3)
        work_hours = np.clip(work_hours, 0, 16)
        
        exercise_freq = np.random.choice([0, 1, 2, 3], p=[0.3, 0.3, 0.2, 0.2])  # 0=none, 3=high
        
        return [
            eye_openness, mouth_tension, brow_furrow, jaw_clenching,
            blink_rate, pupil_dilation, heart_rate, hrv, skin_temp,
            respiration_rate, age, sleep_hours, work_hours, exercise_freq
        ]
    
    def _calculate_stress_label(self, features):
        """Calculate stress label based on feature patterns"""
        eye_openness, mouth_tension, brow_furrow, jaw_clenching, \
        blink_rate, pupil_dilation, heart_rate, hrv, skin_temp, \
        respiration_rate, age, sleep_hours, work_hours, exercise_freq = features
        
        stress_score = 0.0
        
        # Facial stress indicators
        if brow_furrow > 0.6: stress_score += 0.2
        if jaw_clenching > 0.6: stress_score += 0.15
        if eye_openness < 0.4: stress_score += 0.1
        if mouth_tension > 0.6: stress_score += 0.1
        if pupil_dilation > 0.6: stress_score += 0.1
        
        # Physiological stress indicators
        if heart_rate > 85: stress_score += 0.15
        if hrv < 30: stress_score += 0.1
        if respiration_rate > 20: stress_score += 0.1
        
        # Lifestyle stress indicators
        if work_hours > 10: stress_score += 0.1
        if sleep_hours < 6: stress_score += 0.1
        if exercise_freq == 0: stress_score += 0.05
        
        return min(stress_score, 1.0)
    
    def _calculate_fatigue_label(self, features):
        """Calculate fatigue label based on feature patterns"""
        eye_openness, mouth_tension, brow_furrow, jaw_clenching, \
        blink_rate, pupil_dilation, heart_rate, hrv, skin_temp, \
        respiration_rate, age, sleep_hours, work_hours, exercise_freq = features
        
        fatigue_score = 0.0
        
        # Facial fatigue indicators
        if eye_openness < 0.4: fatigue_score += 0.25
        if blink_rate < 0.3: fatigue_score += 0.15
        if pupil_dilation < 0.3: fatigue_score += 0.1
        
        # Physiological fatigue indicators
        if heart_rate > 80: fatigue_score += 0.1
        if hrv < 25: fatigue_score += 0.15
        if skin_temp < 35.5: fatigue_score += 0.1
        
        # Lifestyle fatigue indicators
        if sleep_hours < 6: fatigue_score += 0.2
        if work_hours > 10: fatigue_score += 0.1
        if exercise_freq == 0: fatigue_score += 0.05
        
        return min(fatigue_score, 1.0)
    
    def _train_models(self):
        """Train the stress and fatigue detection models"""
        try:
            # Generate training data
            X, y_stress, y_fatigue = self._generate_training_data(2000)
            
            # Initialize feature preprocessing
            self.feature_scaler = StandardScaler()
            
            # Scale features
            X_scaled = self.feature_scaler.fit_transform(X)
            
            # Train stress model (Random Forest for classification)
            self.stress_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            self.stress_model.fit(X_scaled, (y_stress > 0.5).astype(int))
            
            # Train fatigue model (Gradient Boosting for regression)
            self.fatigue_model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                random_state=42,
                learning_rate=0.1
            )
            self.fatigue_model.fit(X_scaled, y_fatigue)
            
            # Save models
            self._save_models()
            
            print("✅ Models trained and saved successfully")
            
        except Exception as e:
            print(f"Error training models: {e}")
            import traceback
            traceback.print_exc()
    
    def extract_features(self, facial_data, physiological_data, demographic_data):
        """Extract and normalize features from input data"""
        try:
            features = []
            
            # Facial features (normalize to 0-1)
            features.extend([
                facial_data.get('eye_openness', 0.5),
                facial_data.get('mouth_tension', 0.5),
                facial_data.get('brow_furrow', 0.5),
                facial_data.get('jaw_clenching', 0.5),
                facial_data.get('blink_rate', 0.5),
                facial_data.get('pupil_dilation', 0.5)
            ])
            
            # Physiological features (normalize)
            heart_rate = physiological_data.get('heart_rate', 70)
            hrv = physiological_data.get('hrv', 50)
            skin_temp = physiological_data.get('skin_temperature', 36.5)
            respiration_rate = physiological_data.get('respiration_rate', 16)
            
            # Normalize physiological values
            heart_rate_norm = (heart_rate - 50) / (120 - 50)  # 50-120 range
            hrv_norm = (hrv - 10) / (80 - 10)  # 10-80 range
            skin_temp_norm = (skin_temp - 34) / (39 - 34)  # 34-39 range
            resp_norm = (respiration_rate - 10) / (30 - 10)  # 10-30 range
            
            features.extend([heart_rate_norm, hrv_norm, skin_temp_norm, resp_norm])
            
            # Demographic features
            age = demographic_data.get('age', 30)
            sleep_hours = demographic_data.get('sleep_hours', 7)
            work_hours = demographic_data.get('work_hours', 8)
            exercise_freq = self._encode_exercise_frequency(
                demographic_data.get('exercise_frequency', 'moderate')
            )
            
            # Normalize demographic values
            age_norm = (age - 18) / (80 - 18)  # 18-80 range
            sleep_norm = (sleep_hours - 4) / (12 - 4)  # 4-12 range
            work_norm = (work_hours - 0) / (16 - 0)  # 0-16 range
            
            features.extend([age_norm, sleep_norm, work_norm, exercise_freq])
            
            return np.array(features)
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return np.zeros(14)  # Return zero features if error
    
    def _encode_exercise_frequency(self, exercise_freq):
        """Encode exercise frequency to numeric value"""
        mapping = {
            'none': 0.0,
            'low': 0.33,
            'moderate': 0.67,
            'high': 1.0
        }
        return mapping.get(exercise_freq.lower(), 0.5)
    
    def predict_stress_fatigue(self, facial_data, physiological_data, demographic_data):
        """Predict stress and fatigue using pretrained models"""
        if not self.is_loaded:
            return {
                'stress_score': 0.0,
                'fatigue_score': 0.0,
                'confidence': 0.0,
                'model_available': False
            }
        
        try:
            # Extract features
            features = self.extract_features(facial_data, physiological_data, demographic_data)
            
            # Scale features
            features_scaled = self.feature_scaler.transform(features.reshape(1, -1))
            
            # Make predictions
            stress_prob = self.stress_model.predict_proba(features_scaled)[0]
            stress_score = stress_prob[1] if len(stress_prob) > 1 else stress_prob[0]  # Probability of high stress
            
            fatigue_score = self.fatigue_model.predict(features_scaled)[0]
            fatigue_score = np.clip(fatigue_score, 0.0, 1.0)
            
            # Calculate confidence based on model certainty
            stress_confidence = max(stress_prob) if len(stress_prob) > 1 else 0.8
            fatigue_confidence = 0.8  # Default confidence for regression
            
            overall_confidence = (stress_confidence + fatigue_confidence) / 2
            
            return {
                'stress_score': round(float(stress_score), 3),
                'fatigue_score': round(float(fatigue_score), 3),
                'confidence': round(float(overall_confidence), 3),
                'model_available': True,
                'model_version': '1.0'
            }
            
        except Exception as e:
            print(f"Error in model prediction: {e}")
            return {
                'stress_score': 0.0,
                'fatigue_score': 0.0,
                'confidence': 0.0,
                'model_available': False,
                'error': str(e)
            }

# Initialize the pretrained model
pretrained_model = PretrainedStressFatigueModel()

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
                'facial_analysis': data.get('facial_analysis'),
                'weather_data': data.get('weather_data', {}),
                'timestamp': time.time(),
                'date': datetime.now().strftime('%d/%m/%Y, %H:%M:%S')
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
            
            # Save to file for persistence
            try:
                with open('health_assessment_data.json', 'w') as f:
                    json.dump(demographics_data, f, indent=2)
                print(f"✅ Health assessment data saved to file: {len(demographics_data)} assessments")
            except Exception as save_error:
                print(f"⚠️  Warning: Could not save to file: {save_error}")
            
            # 🔥 AUTOMATIC HEATSTROKE PREDICTION
            print("\n🔥 Running automatic heatstroke prediction...")
            heatstroke_result = None
            
            try:
                # Get BPM and temperature data from newA.py analysis
                bpm_data = get_bpm_and_temperature_data()
                print(f"📊 Using real BPM data: {bpm_data['bpm']:.1f} BPM, {bpm_data['skin_temperature']:.1f}°C")
                
                # Run heatstroke prediction with real physiological data
                heatstroke_result = heatstroke_predictor.predict(assessment_data, bpm_data)
                
                if 'error' not in heatstroke_result:
                    print(f"✅ Heatstroke prediction completed:")
                    print(f"   Risk: {'HIGH' if heatstroke_result['heatstroke_prediction'] else 'LOW'}")
                    print(f"   Probability: {heatstroke_result['heatstroke_probability']*100:.1f}%")
                    print(f"   Risk Level: {heatstroke_result['risk_level']}")
                    
                    # Print key features
                    if heatstroke_result.get('feature_values'):
                        print(f"\n🔍 Key Features:")
                        for feature, value in heatstroke_result['feature_values'].items():
                            if value != 0:
                                print(f"   {feature}: {value}")
                else:
                    print(f"❌ Heatstroke prediction error: {heatstroke_result['error']}")
                    
            except Exception as heatstroke_error:
                print(f"❌ Error running heatstroke prediction: {heatstroke_error}")
                heatstroke_result = {'error': str(heatstroke_error)}
            
            return jsonify({
                'success': True,
                'message': 'Assessment submitted successfully',
                'risk_scores': assessment_data.get('risk_scores', {}),
                'heatstroke_prediction': heatstroke_result
            })
        else:
            return jsonify({'success': False, 'error': 'No data received'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})



def perform_advanced_stress_analysis(facial_data, physiological_data, demographic_data):
    """
    Advanced stress and fatigue analysis using multi-modal data fusion
    with MediaPipe facial landmarks and machine learning
    """
    try:

        
        # Fallback to traditional analysis
        print("🔄 Using traditional analysis as fallback")
        
        # Get predictions from pretrained model
        ml_predictions = pretrained_model.predict_stress_fatigue(
            facial_data, physiological_data, demographic_data
        )
        
        # Traditional rule-based analysis
        rule_based_stress = 0.0
        rule_based_fatigue = 0.0
        rule_based_confidence = 0.0
        
        # 1. Enhanced Facial Analysis (40% weight)
        facial_stress = analyze_enhanced_facial_stress(facial_data)
        facial_fatigue = analyze_enhanced_facial_fatigue(facial_data)
        
        # 2. Enhanced Physiological Analysis (35% weight)
        physiological_stress = analyze_enhanced_physiological_stress(physiological_data)
        physiological_fatigue = analyze_enhanced_physiological_fatigue(physiological_data)
        
        # 3. Enhanced Lifestyle Analysis (25% weight)
        lifestyle_stress = analyze_enhanced_lifestyle_stress(demographic_data)
        lifestyle_fatigue = analyze_enhanced_lifestyle_fatigue(demographic_data)
        
        # Combine rule-based scores
        rule_based_stress = (
            facial_stress * 0.4 +
            physiological_stress * 0.35 +
            lifestyle_stress * 0.25
        )
        
        rule_based_fatigue = (
            facial_fatigue * 0.4 +
            physiological_fatigue * 0.35 +
            lifestyle_fatigue * 0.25
        )
        
        # Combine ML and rule-based predictions (70% ML, 30% rule-based)
        final_stress = (
            ml_predictions.get('stress_score', 0.0) * 0.7 +
            rule_based_stress * 0.3
        )
        
        final_fatigue = (
            ml_predictions.get('fatigue_score', 0.0) * 0.7 +
            rule_based_fatigue * 0.3
        )
        
        # Calculate confidence
        confidence = (
            ml_predictions.get('confidence', 0.0) * 0.7 +
            rule_based_confidence * 0.3
        )
        
        # Determine levels
        stress_level = "Low" if final_stress < 0.3 else "Moderate" if final_stress < 0.7 else "High"
        fatigue_level = "Low" if final_fatigue < 0.3 else "Moderate" if final_fatigue < 0.7 else "High"
        
        # Generate recommendations
        recommendations = generate_recommendations(final_stress, final_fatigue, demographic_data)
        
        return {
            'stress_score': round(final_stress, 3),
            'fatigue_score': round(final_fatigue, 3),
            'confidence': round(confidence, 3),
            'stress_level': stress_level,
            'fatigue_level': fatigue_level,
            'model_version': 'Ensemble v2.0',
            'analysis_method': 'ML + Rule-based Analysis',
            'recommendations': recommendations,
            'ml_predictions': ml_predictions,
            'rule_based_scores': {
                'facial_stress': facial_stress,
                'facial_fatigue': facial_fatigue,
                'physiological_stress': physiological_stress,
                'physiological_fatigue': physiological_fatigue,
                'lifestyle_stress': lifestyle_stress,
                'lifestyle_fatigue': lifestyle_fatigue
            },
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Error in stress analysis: {e}")
        import traceback
        traceback.print_exc()
        return {
            'error': str(e),
            'stress_score': 0.0,
            'fatigue_score': 0.0,
            'confidence': 0.0,
            'stress_level': 'Unknown',
            'fatigue_level': 'Unknown'
        }

def analyze_enhanced_facial_stress(facial_data):
    """Enhanced facial stress analysis with machine learning features"""
    stress_score = 0.0
    
    # Extract comprehensive facial features
    eye_openness = facial_data.get('eye_openness', 0.5)
    mouth_tension = facial_data.get('mouth_tension', 0.5)
    brow_furrow = facial_data.get('brow_furrow', 0.5)
    jaw_clenching = facial_data.get('jaw_clenching', 0.5)
    blink_rate = facial_data.get('blink_rate', 0.5)
    pupil_dilation = facial_data.get('pupil_dilation', 0.5)
    facial_asymmetry = facial_data.get('facial_asymmetry', 0.5)
    skin_tone_variation = facial_data.get('skin_tone_variation', 0.5)
    
    # Advanced stress indicators with weighted scoring
    # 1. Corrugator activity (brow furrowing) - Strong stress indicator
    if brow_furrow > 0.7:
        stress_score += 0.25
    elif brow_furrow > 0.5:
        stress_score += 0.15
    
    # 2. Masseter activity (jaw clenching) - Chronic stress indicator
    if jaw_clenching > 0.7:
        stress_score += 0.20
    elif jaw_clenching > 0.5:
        stress_score += 0.10
    
    # 3. Orbicularis oculi activity (eye tension)
    if eye_openness < 0.3:
        stress_score += 0.15
    elif eye_openness < 0.5:
        stress_score += 0.08
    
    # 4. Orbicularis oris activity (mouth tension)
    if mouth_tension > 0.6:
        stress_score += 0.12
    elif mouth_tension > 0.4:
        stress_score += 0.06
    
    # 5. Pupil dilation (autonomic response)
    if pupil_dilation > 0.7:
        stress_score += 0.10
    elif pupil_dilation > 0.5:
        stress_score += 0.05
    
    # 6. Abnormal blink rate (cognitive load)
    if blink_rate < 0.2 or blink_rate > 0.8:
        stress_score += 0.08
    
    # 7. Facial asymmetry (muscle tension)
    if facial_asymmetry > 0.6:
        stress_score += 0.05
    
    # 8. Skin tone variation (vasoconstriction)
    if skin_tone_variation > 0.6:
        stress_score += 0.05
    
    return min(stress_score, 1.0)

def analyze_enhanced_facial_fatigue(facial_data):
    """Enhanced facial fatigue analysis with comprehensive features"""
    fatigue_score = 0.0
    
    # Extract facial features
    eye_openness = facial_data.get('eye_openness', 0.5)
    eye_bags = facial_data.get('eye_bags', 0.5)
    skin_tone = facial_data.get('skin_tone', 0.5)
    facial_asymmetry = facial_data.get('facial_asymmetry', 0.5)
    eye_redness = facial_data.get('eye_redness', 0.5)
    facial_droop = facial_data.get('facial_droop', 0.5)
    blink_frequency = facial_data.get('blink_frequency', 0.5)
    
    # Advanced fatigue indicators
    # 1. Ptosis (droopy eyelids) - Primary fatigue indicator
    if eye_openness < 0.3:
        fatigue_score += 0.30
    elif eye_openness < 0.5:
        fatigue_score += 0.20
    
    # 2. Periorbital edema (eye bags) - Chronic fatigue
    if eye_bags > 0.7:
        fatigue_score += 0.25
    elif eye_bags > 0.5:
        fatigue_score += 0.15
    
    # 3. Conjunctival injection (eye redness) - Sleep deprivation
    if eye_redness > 0.6:
        fatigue_score += 0.15
    elif eye_redness > 0.4:
        fatigue_score += 0.08
    
    # 4. Facial asymmetry (muscle fatigue)
    if facial_asymmetry > 0.6:
        fatigue_score += 0.10
    elif facial_asymmetry > 0.4:
        fatigue_score += 0.05
    
    # 5. Dull skin tone (poor circulation)
    if skin_tone < 0.3:
        fatigue_score += 0.10
    elif skin_tone < 0.5:
        fatigue_score += 0.05
    
    # 6. Facial droop (muscle weakness)
    if facial_droop > 0.6:
        fatigue_score += 0.05
    
    # 7. Reduced blink frequency (cognitive fatigue)
    if blink_frequency < 0.3:
        fatigue_score += 0.05
    
    return min(fatigue_score, 1.0)

def analyze_enhanced_physiological_stress(physiological_data):
    """Enhanced physiological stress analysis with advanced metrics"""
    stress_score = 0.0
    
    # Extract comprehensive physiological parameters
    heart_rate = physiological_data.get('heart_rate', 70)
    heart_rate_variability = physiological_data.get('hrv', 50)
    skin_temperature = physiological_data.get('skin_temperature', 36.5)
    respiration_rate = physiological_data.get('respiration_rate', 16)
    blood_pressure_systolic = physiological_data.get('bp_systolic', 120)
    blood_pressure_diastolic = physiological_data.get('bp_diastolic', 80)
    skin_conductance = physiological_data.get('skin_conductance', 5.0)
    temperature_variation = physiological_data.get('temp_variation', 0.5)
    
    # Advanced stress indicators with age-adjusted thresholds
    # 1. Heart rate analysis (age-adjusted)
    age = physiological_data.get('age', 30)
    max_hr = 220 - age
    resting_hr_threshold = max_hr * 0.4  # 40% of max HR
    
    if heart_rate > resting_hr_threshold + 20:
        stress_score += 0.25
    elif heart_rate > resting_hr_threshold + 10:
        stress_score += 0.15
    
    # 2. Heart rate variability (RMSSD-based)
    if heart_rate_variability < 20:
        stress_score += 0.25
    elif heart_rate_variability < 35:
        stress_score += 0.15
    elif heart_rate_variability < 50:
        stress_score += 0.08
    
    # 3. Skin conductance (electrodermal activity)
    if skin_conductance > 8.0:
        stress_score += 0.15
    elif skin_conductance > 6.0:
        stress_score += 0.08
    
    # 4. Blood pressure elevation
    if blood_pressure_systolic > 140 or blood_pressure_diastolic > 90:
        stress_score += 0.15
    elif blood_pressure_systolic > 130 or blood_pressure_diastolic > 85:
        stress_score += 0.08
    
    # 5. Respiratory rate (hyperventilation)
    if respiration_rate > 22:
        stress_score += 0.12
    elif respiration_rate > 18:
        stress_score += 0.06
    
    # 6. Skin temperature elevation (stress response)
    if skin_temperature > 37.8:
        stress_score += 0.08
    elif skin_temperature > 37.2:
        stress_score += 0.04
    
    return min(stress_score, 1.0)

def analyze_enhanced_physiological_fatigue(physiological_data):
    """Enhanced physiological fatigue analysis with comprehensive metrics"""
    fatigue_score = 0.0
    
    # Extract physiological parameters
    heart_rate = physiological_data.get('heart_rate', 70)
    heart_rate_variability = physiological_data.get('hrv', 50)
    skin_temperature = physiological_data.get('skin_temperature', 36.5)
    blood_pressure_systolic = physiological_data.get('bp_systolic', 120)
    blood_pressure_diastolic = physiological_data.get('bp_diastolic', 80)
    oxygen_saturation = physiological_data.get('oxygen_saturation', 98)
    respiratory_rate = physiological_data.get('respiration_rate', 16)
    
    # Advanced fatigue indicators
    # 1. Reduced heart rate variability (autonomic dysfunction)
    if heart_rate_variability < 15:
        fatigue_score += 0.30
    elif heart_rate_variability < 25:
        fatigue_score += 0.20
    elif heart_rate_variability < 35:
        fatigue_score += 0.10
    
    # 2. Compensatory tachycardia
    if heart_rate > 85:
        fatigue_score += 0.20
    elif heart_rate > 75:
        fatigue_score += 0.10
    
    # 3. Poor peripheral circulation (low skin temperature)
    if skin_temperature < 35.0:
        fatigue_score += 0.20
    elif skin_temperature < 36.0:
        fatigue_score += 0.10
    
    # 4. Blood pressure dysregulation
    if blood_pressure_systolic > 150 or blood_pressure_systolic < 90:
        fatigue_score += 0.15
    elif blood_pressure_diastolic > 95 or blood_pressure_diastolic < 60:
        fatigue_score += 0.10
    
    # 5. Reduced oxygen saturation
    if oxygen_saturation < 95:
        fatigue_score += 0.10
    elif oxygen_saturation < 97:
        fatigue_score += 0.05
    
    # 6. Irregular breathing pattern
    if respiratory_rate < 12 or respiratory_rate > 20:
        fatigue_score += 0.05
    
    return min(fatigue_score, 1.0)

def analyze_enhanced_lifestyle_stress(demographic_data):
    """Enhanced lifestyle stress analysis with comprehensive factors"""
    stress_score = 0.0
    
    # Extract comprehensive lifestyle data
    age = demographic_data.get('age', 30)
    sleep_hours = demographic_data.get('sleep_hours', 7)
    sleep_quality = demographic_data.get('sleep_quality', 'good')
    exercise_frequency = demographic_data.get('exercise_frequency', 'moderate')
    work_hours = demographic_data.get('work_hours', 8)
    stress_level = demographic_data.get('stress_level', 'low')
    caffeine_intake = demographic_data.get('caffeine_intake', 'moderate')
    alcohol_consumption = demographic_data.get('alcohol_consumption', 'low')
    smoking_status = demographic_data.get('smoking_status', 'none')
    social_support = demographic_data.get('social_support', 'good')
    financial_stress = demographic_data.get('financial_stress', 'low')
    
    # Advanced stress factors with weighted scoring
    # 1. Sleep quality and quantity
    if sleep_hours < 5:
        stress_score += 0.25
    elif sleep_hours < 6:
        stress_score += 0.15
    elif sleep_hours < 7:
        stress_score += 0.08
    
    if sleep_quality == 'poor':
        stress_score += 0.20
    elif sleep_quality == 'fair':
        stress_score += 0.10
    
    # 2. Self-reported stress level
    if stress_level == 'very_high':
        stress_score += 0.25
    elif stress_level == 'high':
        stress_score += 0.20
    elif stress_level == 'moderate':
        stress_score += 0.10
    
    # 3. Work-related stress
    if work_hours > 12:
        stress_score += 0.20
    elif work_hours > 10:
        stress_score += 0.15
    elif work_hours > 8:
        stress_score += 0.08
    
    # 4. Physical activity (inverse relationship)
    if exercise_frequency == 'none':
        stress_score += 0.15
    elif exercise_frequency == 'low':
        stress_score += 0.08
    
    # 5. Substance use
    if caffeine_intake == 'very_high':
        stress_score += 0.12
    elif caffeine_intake == 'high':
        stress_score += 0.08
    
    if alcohol_consumption == 'high':
        stress_score += 0.10
    elif alcohol_consumption == 'moderate':
        stress_score += 0.05
    
    if smoking_status == 'current':
        stress_score += 0.15
    elif smoking_status == 'recent':
        stress_score += 0.08
    
    # 6. Social and financial factors
    if social_support == 'poor':
        stress_score += 0.15
    elif social_support == 'fair':
        stress_score += 0.08
    
    if financial_stress == 'high':
        stress_score += 0.20
    elif financial_stress == 'moderate':
        stress_score += 0.10
    
    return min(stress_score, 1.0)

def analyze_enhanced_lifestyle_fatigue(demographic_data):
    """Enhanced lifestyle fatigue analysis with comprehensive factors"""
    fatigue_score = 0.0
    
    # Extract lifestyle data
    sleep_hours = demographic_data.get('sleep_hours', 7)
    sleep_quality = demographic_data.get('sleep_quality', 'good')
    sleep_latency = demographic_data.get('sleep_latency', 15)
    exercise_frequency = demographic_data.get('exercise_frequency', 'moderate')
    caffeine_intake = demographic_data.get('caffeine_intake', 'moderate')
    alcohol_consumption = demographic_data.get('alcohol_consumption', 'low')
    diet_quality = demographic_data.get('diet_quality', 'good')
    hydration_level = demographic_data.get('hydration_level', 'good')
    screen_time = demographic_data.get('screen_time', 4)
    
    # Advanced fatigue factors
    # 1. Sleep quantity and quality
    if sleep_hours < 5:
        fatigue_score += 0.30
    elif sleep_hours < 6:
        fatigue_score += 0.20
    elif sleep_hours < 7:
        fatigue_score += 0.10
    
    if sleep_quality == 'very_poor':
        fatigue_score += 0.25
    elif sleep_quality == 'poor':
        fatigue_score += 0.20
    elif sleep_quality == 'fair':
        fatigue_score += 0.10
    
    if sleep_latency > 30:
        fatigue_score += 0.15
    elif sleep_latency > 20:
        fatigue_score += 0.08
    
    # 2. Physical activity (inverse relationship)
    if exercise_frequency == 'none':
        fatigue_score += 0.20
    elif exercise_frequency == 'low':
        fatigue_score += 0.10
    
    # 3. Substance use patterns
    if caffeine_intake == 'very_high':
        fatigue_score += 0.15
    elif caffeine_intake == 'high':
        fatigue_score += 0.10
    
    if alcohol_consumption == 'high':
        fatigue_score += 0.15
    elif alcohol_consumption == 'moderate':
        fatigue_score += 0.08
    
    # 4. Nutritional factors
    if diet_quality == 'poor':
        fatigue_score += 0.15
    elif diet_quality == 'fair':
        fatigue_score += 0.08
    
    if hydration_level == 'poor':
        fatigue_score += 0.12
    elif hydration_level == 'fair':
        fatigue_score += 0.06
    
    # 5. Digital fatigue
    if screen_time > 8:
        fatigue_score += 0.10
    elif screen_time > 6:
        fatigue_score += 0.05
    
    return min(fatigue_score, 1.0)

# Helper functions for confidence calculation and temporal analysis
def calculate_facial_confidence(facial_data):
    """Calculate confidence in facial analysis based on data quality"""
    confidence = 0.0
    valid_features = 0
    
    features = ['eye_openness', 'mouth_tension', 'brow_furrow', 'jaw_clenching', 
               'blink_rate', 'pupil_dilation', 'facial_asymmetry']
    
    for feature in features:
        if feature in facial_data and facial_data[feature] is not None:
            valid_features += 1
    
    confidence = valid_features / len(features)
    return min(confidence, 1.0)

def calculate_physiological_confidence(physiological_data):
    """Calculate confidence in physiological analysis"""
    confidence = 0.0
    valid_features = 0
    
    features = ['heart_rate', 'hrv', 'skin_temperature', 'respiration_rate']
    
    for feature in features:
        if feature in physiological_data and physiological_data[feature] is not None:
            valid_features += 1
    
    confidence = valid_features / len(features)
    return min(confidence, 1.0)

def calculate_lifestyle_confidence(demographic_data):
    """Calculate confidence in lifestyle analysis"""
    confidence = 0.0
    valid_features = 0
    
    features = ['sleep_hours', 'exercise_frequency', 'stress_level', 'work_hours']
    
    for feature in features:
        if feature in demographic_data and demographic_data[feature] is not None:
            valid_features += 1
    
    confidence = valid_features / len(features)
    return min(confidence, 1.0)

# Global variables for temporal analysis
stress_history = []
fatigue_history = []

def apply_temporal_smoothing(score, metric_type):
    """Apply temporal smoothing to reduce noise and improve stability"""
    global stress_history, fatigue_history
    
    if metric_type == 'stress':
        stress_history.append(score)
        if len(stress_history) > 5:  # Keep last 5 readings
            stress_history.pop(0)
        return sum(stress_history) / len(stress_history)
    else:
        fatigue_history.append(score)
        if len(fatigue_history) > 5:
            fatigue_history.pop(0)
        return sum(fatigue_history) / len(fatigue_history)

def get_enhanced_stress_level(stress_score, demographic_data):
    """Enhanced stress level determination with age and context adjustment"""
    age = demographic_data.get('age', 30) if demographic_data else 30
    
    # Age-adjusted thresholds
    if age > 65:
        # Elderly: more sensitive to stress
        if stress_score >= 0.7:
            return 'Critical'
        elif stress_score >= 0.5:
            return 'High'
        elif stress_score >= 0.3:
            return 'Moderate'
        elif stress_score >= 0.15:
            return 'Low'
        else:
            return 'Minimal'
    elif age < 25:
        # Young adults: more resilient
        if stress_score >= 0.8:
            return 'Critical'
        elif stress_score >= 0.6:
            return 'High'
        elif stress_score >= 0.4:
            return 'Moderate'
        elif stress_score >= 0.2:
            return 'Low'
        else:
            return 'Minimal'
    else:
        # Standard thresholds for adults
        if stress_score >= 0.8:
            return 'Critical'
        elif stress_score >= 0.6:
            return 'High'
        elif stress_score >= 0.4:
            return 'Moderate'
        elif stress_score >= 0.2:
            return 'Low'
        else:
            return 'Minimal'

def get_enhanced_fatigue_level(fatigue_score, demographic_data):
    """Enhanced fatigue level determination with context adjustment"""
    age = demographic_data.get('age', 30) if demographic_data else 30
    sleep_hours = demographic_data.get('sleep_hours', 7) if demographic_data else 7
    
    # Adjust thresholds based on sleep patterns
    if sleep_hours < 6:
        # Lower thresholds for sleep-deprived individuals
        if fatigue_score >= 0.7:
            return 'Severe'
        elif fatigue_score >= 0.5:
            return 'Moderate'
        elif fatigue_score >= 0.3:
            return 'Mild'
        elif fatigue_score >= 0.15:
            return 'Slight'
        else:
            return 'None'
    else:
        # Standard thresholds
        if fatigue_score >= 0.8:
            return 'Severe'
        elif fatigue_score >= 0.6:
            return 'Moderate'
        elif fatigue_score >= 0.4:
            return 'Mild'
        elif fatigue_score >= 0.2:
            return 'Slight'
        else:
            return 'None'

def identify_risk_factors(stress_score, fatigue_score, demographic_data, physiological_data):
    """Identify specific risk factors contributing to stress and fatigue"""
    risk_factors = []
    
    if demographic_data:
        # Sleep-related risks
        sleep_hours = demographic_data.get('sleep_hours', 7)
        if sleep_hours < 6:
            risk_factors.append({
                'category': 'Sleep',
                'factor': 'Insufficient sleep',
                'severity': 'High' if sleep_hours < 5 else 'Moderate',
                'description': f'Only {sleep_hours} hours of sleep per night'
            })
        
        # Work-related risks
        work_hours = demographic_data.get('work_hours', 8)
        if work_hours > 10:
            risk_factors.append({
                'category': 'Work',
                'factor': 'Long work hours',
                'severity': 'High',
                'description': f'{work_hours} hours of work per day'
            })
        
        # Lifestyle risks
        exercise_frequency = demographic_data.get('exercise_frequency', 'moderate')
        if exercise_frequency == 'none':
            risk_factors.append({
                'category': 'Lifestyle',
                'factor': 'Lack of exercise',
                'severity': 'Moderate',
                'description': 'No regular physical activity'
            })
    
    if physiological_data:
        # Physiological risks
        heart_rate = physiological_data.get('heart_rate', 70)
        if heart_rate > 85:
            risk_factors.append({
                'category': 'Physiological',
                'factor': 'Elevated heart rate',
                'severity': 'Moderate',
                'description': f'Heart rate: {heart_rate} BPM'
            })
        
        hrv = physiological_data.get('hrv', 50)
        if hrv < 30:
            risk_factors.append({
                'category': 'Physiological',
                'factor': 'Low heart rate variability',
                'severity': 'High',
                'description': f'HRV: {hrv} ms'
            })
    
    return risk_factors

def generate_advanced_recommendations(stress_score, fatigue_score, demographic_data, physiological_data):
    """Generate advanced, personalized recommendations"""
    recommendations = []
    
    # Stress management recommendations
    if stress_score >= 0.7:
        recommendations.append({
            'category': 'Immediate Stress Relief',
            'priority': 'Critical',
            'recommendations': [
                'Practice 4-7-8 breathing technique immediately',
                'Take a 10-minute walk in nature',
                'Listen to calming music or guided meditation',
                'Consider speaking with a mental health professional',
                'Implement stress-reduction breaks every 30 minutes'
            ]
        })
    elif stress_score >= 0.5:
        recommendations.append({
            'category': 'Stress Management',
            'priority': 'High',
            'recommendations': [
                'Practice mindfulness meditation for 15-20 minutes daily',
                'Engage in progressive muscle relaxation',
                'Maintain a consistent sleep schedule (7-9 hours)',
                'Limit caffeine intake, especially after 2 PM',
                'Consider yoga or tai chi classes'
            ]
        })
    
    # Fatigue management recommendations
    if fatigue_score >= 0.7:
        recommendations.append({
            'category': 'Fatigue Recovery',
            'priority': 'Critical',
            'recommendations': [
                'Prioritize 8-9 hours of quality sleep',
                'Create a relaxing bedtime routine',
                'Avoid screens 2 hours before bedtime',
                'Consider a sleep study consultation',
                'Maintain regular meal times and stay hydrated'
            ]
        })
    elif fatigue_score >= 0.5:
        recommendations.append({
            'category': 'Fatigue Management',
            'priority': 'High',
            'recommendations': [
                'Aim for 7-8 hours of consistent sleep',
                'Create a dark, quiet sleep environment',
                'Avoid caffeine after 3 PM',
                'Take short power naps (20 minutes max)',
                'Maintain regular exercise routine'
            ]
        })
    
    # Lifestyle optimization
    if demographic_data:
        sleep_hours = demographic_data.get('sleep_hours', 7)
        if sleep_hours < 7:
            recommendations.append({
                'category': 'Sleep Optimization',
                'priority': 'Medium',
                'recommendations': [
                    'Gradually increase sleep duration by 15 minutes',
                    'Establish a consistent bedtime routine',
                    'Keep bedroom cool (65-68°F) and dark',
                    'Avoid large meals before bedtime',
                    'Consider sleep tracking to identify patterns'
                ]
            })
        
        exercise_frequency = demographic_data.get('exercise_frequency', 'moderate')
        if exercise_frequency in ['none', 'low']:
            recommendations.append({
                'category': 'Physical Activity',
                'priority': 'Medium',
                'recommendations': [
                    'Start with 10-15 minutes of daily walking',
                    'Gradually increase to 30 minutes, 5 days/week',
                    'Consider low-impact activities like swimming',
                    'Find activities you enjoy to maintain consistency',
                    'Consult with a fitness professional if needed'
                ]
            })
    
    # Prevention and maintenance
    if stress_score < 0.4 and fatigue_score < 0.4:
        recommendations.append({
            'category': 'Wellness Maintenance',
            'priority': 'Low',
            'recommendations': [
                'Continue current healthy habits',
                'Regular health check-ups',
                'Maintain social connections',
                'Practice gratitude and positive thinking',
                'Consider stress management workshops'
            ]
        })
    
    return recommendations





# --- Heatstroke Predictor Integration ---
import sys
sys.path.append('../CSV_datasets')
sys.path.append('../BPM')  # Add BPM directory to path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

def get_bpm_and_temperature_data():
    """
    Get current BPM and skin temperature data from newA.py analysis
    
    Returns:
        dict: Contains 'bpm', 'skin_temperature', 'confidence', and other metrics
    """
    try:
        # Import the BPM analysis function
        from newA import run_pulse_rate_from_csv
        
        # Analyze the latest Arduino data
        bpm, confidence, infection_score, dehydration_score, arrhythmia_score = run_pulse_rate_from_csv(
            "../BPM/A.csv", 
            fs=10, 
            plot_spectrogram=False
        )
        
        # Get skin temperature from the CSV file
        import pandas as pd
        try:
            df = pd.read_csv("../BPM/A.csv", skiprows=1)
            skin_temps = df["SkinTemp(C)"].astype(float).values
            avg_skin_temp = float(np.mean(skin_temps))
            min_skin_temp = float(np.min(skin_temps))
            max_skin_temp = float(np.max(skin_temps))
        except Exception as e:
            print(f"⚠️ Could not read skin temperature from CSV: {e}")
            avg_skin_temp = 34.6  # Fallback value from logs
            min_skin_temp = 34.0
            max_skin_temp = 35.0
        
        return {
            'bpm': float(bpm),
            'skin_temperature': avg_skin_temp,
            'skin_temp_min': min_skin_temp,
            'skin_temp_max': max_skin_temp,
            'confidence': float(confidence),
            'infection_score': float(infection_score),
            'dehydration_score': float(dehydration_score),
            'arrhythmia_score': float(arrhythmia_score),
            'data_source': 'newA.py analysis',
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
    except Exception as e:
        print(f"⚠️ Could not get BPM data from newA.py: {e}")
        # Return fallback data
        return {
            'bpm': 72.0,
            'skin_temperature': 37.0,
            'skin_temp_min': 36.5,
            'skin_temp_max': 37.5,
            'confidence': 0.0,
            'infection_score': 0.0,
            'dehydration_score': 0.0,
            'arrhythmia_score': 0.0,
            'data_source': 'fallback values',
            'timestamp': pd.Timestamp.now().isoformat()
        }

class DashboardHeatstrokePredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = []
        self.is_trained = False
        self._initialize()

    def _initialize(self):
        # Only train once and cache
        try:
            train_df = pd.read_csv('../CSV_datasets/health_heatstroke_train_2000_temp (1).csv')
            test_df = pd.read_csv('../CSV_datasets/health_heatstroke_test_2000_temp (1).csv')
            # --- Fix: Standardize label column name ---
            for df in [train_df, test_df]:
                if 'heat stroke' in df.columns:
                    df.rename(columns={'heat stroke': 'Heatstroke'}, inplace=True)
            combined_df = pd.concat([train_df, test_df], ignore_index=True)
            X, y = self._preprocess_features(combined_df)
            train_size = len(train_df)
            X_train = X[:train_size]
            y_train = y[:train_size]
            self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            self.model.fit(X_train_scaled, y_train)
            self.is_trained = True
            print("✅ Heatstroke model trained and ready.")
        except Exception as e:
            print(f"❌ Error initializing heatstroke model: {e}")
            self.is_trained = False

    def _preprocess_features(self, df):
        data = df.copy()
        categorical_cols = ['Gender', 'Headache', 'Dizziness', 'Nausea', 'Vomiting', 
            'Muscle_Cramps', 'Weakness', 'Fatigue', 'Hot_Skin', 'Dry_Skin',
            'Rapid_Breathing', 'Cardiovascular_Disease', 'Diabetes', 
            'Obesity', 'Elderly', 'Heat_Sensitive_Medications', 'Dehydration',
            'Alcohol_Use', 'Previous_Heat_Illness', 'Poor_Heat_Acclimation',
            'Prolonged_Exertion', 'High_Blood_Pressure', 'Smoking']
        for col in categorical_cols:
            if col in data.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    data[col] = self.label_encoders[col].fit_transform(data[col].astype(str))
                else:
                    data[col] = data[col].astype(str)
                    data[col] = data[col].map(lambda x: x if x in self.label_encoders[col].classes_ else 'Unknown')
                    data[col] = self.label_encoders[col].transform(data[col])
        feature_cols = ['Age', 'Gender', 'Temperature', 'Heart_Rate', 'Blood_Pressure_Systolic',
            'Blood_Pressure_Diastolic', 'Headache', 'Dizziness', 'Nausea', 'Vomiting',
            'Muscle_Cramps', 'Weakness', 'Fatigue', 'Hot_Skin', 'Dry_Skin',
            'Rapid_Breathing', 'Cardiovascular_Disease', 'Diabetes', 'Obesity',
            'Elderly', 'Heat_Sensitive_Medications', 'Dehydration', 'Alcohol_Use',
            'Previous_Heat_Illness', 'Poor_Heat_Acclimation', 'Prolonged_Exertion',
            'High_Blood_Pressure', 'Smoking']
        available_features = [col for col in feature_cols if col in data.columns]
        self.feature_names = available_features
        X = data[available_features].fillna(0)
        y = data['Heatstroke'].astype(int)
        return X, y

    def predict(self, health_data, bpm_data=None):
        if not self.is_trained:
            return {'error': 'Model not trained'}
        features = []
        for feature in self.feature_names:
            if feature == 'Age':
                features.append(health_data.get('age', 30))
            elif feature == 'Gender':
                gender = health_data.get('gender', 'male')
                if gender in self.label_encoders.get('Gender', {}).classes_:
                    features.append(self.label_encoders['Gender'].transform([gender])[0])
                else:
                    features.append(0)
            elif feature == 'Temperature':
                if bpm_data and 'skin_temperature' in bpm_data:
                    features.append(bpm_data['skin_temperature'])
                else:
                    features.append(37.0)
            elif feature == 'Heart_Rate':
                if bpm_data and 'bpm' in bpm_data:
                    features.append(bpm_data['bpm'])
                else:
                    features.append(72.0)
            elif feature == 'Blood_Pressure_Systolic':
                features.append(120.0)
            elif feature == 'Blood_Pressure_Diastolic':
                features.append(80.0)
            else:
                value = 0
                if feature.lower() in [s.lower() for s in health_data.get('symptoms', [])]:
                    value = 1
                elif feature.lower() in [s.lower() for s in health_data.get('medical_history', [])]:
                    value = 1
                elif feature.lower() in [s.lower() for s in health_data.get('risk_factors', [])]:
                    value = 1
                if feature == 'High_Blood_Pressure' and 'high_blood_pressure' in health_data.get('risk_factors', []):
                    value = 1
                elif feature == 'Diabetes' and ('diabetes' in health_data.get('medical_history', []) or 'diabetes' in health_data.get('risk_factors', [])):
                    value = 1
                elif feature == 'Elderly' and health_data.get('age', 0) > 65:
                    value = 1
                features.append(value)
        features_scaled = self.scaler.transform([features])
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        result = {
            'heatstroke_prediction': bool(prediction),
            'heatstroke_probability': float(probability[1] if len(probability) > 1 else probability[0]),
            'risk_level': self._get_risk_level(probability[1] if len(probability) > 1 else probability[0]),
            'features_used': self.feature_names,
            'feature_values': dict(zip(self.feature_names, features))
        }
        
        # Add BPM data to the result if available
        if bpm_data:
            result['bpm_data'] = bpm_data
            
        return result
    def _get_risk_level(self, probability):
        if probability < 0.3:
            return "Low"
        elif probability < 0.7:
            return "Moderate"
        else:
            return "High"

# Initialize heatstroke predictor
if heatstroke_predictor is None:
    heatstroke_predictor = DashboardHeatstrokePredictor()

@app.route('/api/heatstroke_prediction', methods=['POST'])
def api_heatstroke_prediction():
    data = request.get_json()
    health_data = data.get('health_data', {})
    bpm_data = data.get('bpm_data', {})
    result = heatstroke_predictor.predict(health_data, bpm_data)
    return jsonify(result)

def print_latest_heatstroke_prediction():
    """Print the heatstroke prediction for the latest assessment in demographics_data."""
    if not demographics_data:
        print("❌ No health assessment data in memory. Submit an assessment first.")
        sys.stdout.flush()
        return
    latest = demographics_data[-1]
    print("\n--- Latest Health Assessment ---")
    print(f"Age: {latest.get('age')}")
    print(f"Gender: {latest.get('gender')}")
    print(f"Symptoms: {latest.get('symptoms', [])}")
    print(f"Medical History: {latest.get('medical_history', [])}")
    print(f"Risk Factors: {latest.get('risk_factors', [])}")
    print(f"Date: {latest.get('date', latest.get('timestamp'))}")
    
    # Get BPM and temperature data from newA.py analysis
    bpm_data = get_bpm_and_temperature_data()
    print(f"📊 BPM Data: {bpm_data['bpm']:.1f} BPM, Confidence: {bpm_data['confidence']:.3f}")
    print(f"🌡️ Skin Temperature: {bpm_data['skin_temperature']:.1f}°C (Range: {bpm_data['skin_temp_min']:.1f}-{bpm_data['skin_temp_max']:.1f}°C)")
    print(f"📈 Data Source: {bpm_data['data_source']}")
    
    # Use the DashboardHeatstrokePredictor
    result = heatstroke_predictor.predict(latest, bpm_data)
    
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
    else:
        print(f"\n🔥 Heatstroke Risk Prediction:")
        print(f"   Risk: {'HIGH' if result['heatstroke_prediction'] else 'LOW'}")
        print(f"   Probability: {result['heatstroke_probability']*100:.1f}%")
        print(f"   Risk Level: {result['risk_level']}")
        
        # Show key features that influenced the prediction
        print(f"\n🔍 Key Features:")
        for feature, value in result['feature_values'].items():
            if value != 0:  # Only show non-zero features
                print(f"   {feature}: {value}")
    print("\n✅ Heatstroke prediction complete.\n")
    sys.stdout.flush()

@app.route('/api/latest_heatstroke_prediction')
def get_latest_heatstroke_prediction():
    """Get heatstroke prediction for the latest stored assessment data."""
    if not demographics_data:
        return jsonify({
            'success': False,
            'error': 'No health assessment data available. Please submit an assessment first.'
        })
    
    latest = demographics_data[-1]
    
    # Get BPM and temperature data from newA.py analysis
    bpm_data = get_bpm_and_temperature_data()
    
    # Get prediction
    result = heatstroke_predictor.predict(latest, bpm_data)
    
    if 'error' in result:
        return jsonify({
            'success': False,
            'error': result['error']
        })
    
    return jsonify({
        'success': True,
        'prediction': result,
        'health_data': latest,
        'bpm_data': bpm_data
    })

@app.route('/api/stress_fatigue_analysis', methods=['POST'])
def api_stress_fatigue_analysis():
    """API endpoint for stress and fatigue analysis using MediaPipe face mesh"""
    if not STRESS_FATIGUE_AVAILABLE:
        return jsonify({
            'error': 'Stress fatigue detection not available',
            'stress_level': 0.0,
            'fatigue_level': 0.0,
            'confidence': 0.0
        })
    
    try:
        # Initialize detector if needed
        if not initialize_stress_fatigue_detector():
            return jsonify({
                'error': 'Failed to initialize stress fatigue detector',
                'stress_level': 0.0,
                'fatigue_level': 0.0,
                'confidence': 0.0
            })
        
        # Get analysis summary
        summary = stress_fatigue_detector.generate_wellness_summary()
        
        return jsonify({
            'stress_level': summary['tension_level'],
            'fatigue_level': summary['exhaustion_level'],
            'confidence': summary['analysis_confidence'],
            'trends': summary['wellness_trends'],
            'recommendations': summary['wellness_recommendations'],
            'success': True
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Analysis failed: {str(e)}',
            'stress_level': 0.0,
            'fatigue_level': 0.0,
            'confidence': 0.0
        })

@app.route('/api/reset_stress_fatigue', methods=['POST'])
def api_reset_stress_fatigue():
    """Reset stress fatigue analysis data"""
    try:
        # Reset global variables
        global latest_analysis
        latest_analysis.update({
            'stress_score': 0.0,
            'fatigue_score': 0.0,
            'stress_confidence': 0.0,
            'fatigue_confidence': 0.0
        })
        
        return jsonify({'success': True, 'message': 'Stress fatigue analysis reset successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/test_sunstroke', methods=['POST'])
def test_sunstroke():
    """API endpoint to run the sunstroke test command"""
    try:
        import subprocess
        import sys
        import os
        
        # Get the current script path
        script_path = os.path.abspath(__file__)
        
        # Run the command
        result = subprocess.run([
            sys.executable,  # Use the same Python interpreter
            script_path,
            '--print-latest-heatstroke'
        ], capture_output=True, text=True, cwd=os.path.dirname(script_path))
        
        if result.returncode == 0:
            return jsonify({
                'success': True,
                'output': result.stdout,
                'message': 'Sunstroke test completed successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': result.stderr,
                'output': result.stdout
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/face_landmarks', methods=['POST'])
def api_face_landmarks():
    """API endpoint for face landmark detection using server-side MediaPipe"""
    try:
        data = request.get_json()
        if not data or 'image_data' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Initialize somnolence detector if not already done
        global somnolence_detector
        if SOMNOLENCE_AVAILABLE and somnolence_detector is None:
            try:
                somnolence_detector = DrowsinessDetector()
                print("✅ Somnolence detector initialized")
            except Exception as e:
                print(f"⚠️ Failed to initialize somnolence detector: {e}")
                return jsonify({'error': 'Face detection not available'}), 500
        
        if not SOMNOLENCE_AVAILABLE or somnolence_detector is None:
            return jsonify({'error': 'Face detection not available'}), 500
        
        # Process the image data
        image_data = data['image_data']
        
        # Remove data URL prefix if present
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        # Decode base64 image
        import base64
        import cv2
        import numpy as np
        
        try:
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return jsonify({'error': 'Invalid image data'}), 400
            
            # Get face landmarks using somnolence detector
            landmarks = somnolence_detector.get_face_landmarks(image)
            
            if landmarks is not None:
                # Convert landmarks to list format for JSON serialization
                landmarks_list = []
                for point in landmarks:
                    landmarks_list.append({
                        'x': float(point[0]),
                        'y': float(point[1])
                    })
                
                return jsonify({
                    'success': True,
                    'landmarks': landmarks_list,
                    'face_detected': True
                })
            else:
                return jsonify({
                    'success': True,
                    'landmarks': [],
                    'face_detected': False
                })
                
        except Exception as e:
            print(f"Error processing image: {e}")
            return jsonify({'error': 'Image processing failed'}), 500
            
    except Exception as e:
        print(f"Face landmarks API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/weather', methods=['GET'])
def api_weather():
    """API endpoint to get current weather data"""
    try:
        # Get location from query parameters or use IP geolocation
        lat = request.args.get('lat', type=float)
        lon = request.args.get('lon', type=float)
        city = request.args.get('city')
        
        if not lat and not lon and not city:
            # Try to get location from IP
            location = get_user_location()
            lat = location.get('lat')
            lon = location.get('lon')
            city = location.get('city')
            
            # If we got a city name from IP geolocation, use it
            if city and city != 'Unknown':
                print(f"📍 Using IP-detected location: {city}")
        
        # Get weather data
        weather_data = get_weather_data(lat=lat, lon=lon, city=city)
        
        # If we have IP location data, enhance the weather response
        if not lat and not lon and not city:
            location = get_user_location()
            if location.get('city') != 'Unknown':
                weather_data['location'] = f"{location['city']}, {location['region']}"
                weather_data['country'] = location.get('country', 'Unknown')
        
        return jsonify({
            'success': True,
            'weather': weather_data
        })
        
    except Exception as e:
        print(f"Weather API error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'weather': {
                'error': 'Weather data unavailable',
                'temperature': None,
                'location': 'Unknown'
            }
        })

@app.route('/api/location', methods=['GET'])
def api_location():
    """API endpoint to get user's current location"""
    try:
        location = get_user_location()
        return jsonify({
            'success': True,
            'location': location
        })
    except Exception as e:
        print(f"Location API error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'location': {
                'city': 'Unknown',
                'region': 'Unknown',
                'country': 'Unknown'
            }
        })

if __name__ == '__main__':
    import sys
    
    # Check if this is a CLI command
    if len(sys.argv) > 1 and sys.argv[1] == "--print-latest-heatstroke":
        print_latest_heatstroke_prediction()
        sys.exit(0)
    
    # Start Arduino data reading thread
    arduino_thread = threading.Thread(target=read_arduino_data, daemon=True)
    arduino_thread.start()
    
    # Load and analyze CSV file on startup
    print("Loading and analyzing A.csv file...")
    load_and_analyze_csv()
    
    print("Arduino Dashboard starting...")
    print("Open http://localhost:5003 in your browser")
    # Use SocketIO with threading mode to avoid SSL issues
    socketio.run(app, host='0.0.0.0', port=5003, debug=True, allow_unsafe_werkzeug=True)