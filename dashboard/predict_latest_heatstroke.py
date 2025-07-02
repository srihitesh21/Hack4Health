import json
import os
import sys
from datetime import datetime

# Add CSV datasets directory to path for model import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../CSV_datasets')))

try:
    from arduino_dashboard import DashboardHeatstrokePredictor
except ImportError:
    print("❌ Could not import DashboardHeatstrokePredictor from arduino_dashboard.py")
    sys.exit(1)

def load_latest_assessment():
    data_file = 'health_assessment_data.json'
    if not os.path.exists(data_file):
        print("❌ No health assessment data file found. Please submit an assessment first.")
        return None
    with open(data_file, 'r') as f:
        data = json.load(f)
    if not data:
        print("❌ No assessment data found in file.")
        return None
    latest = data[-1]
    print(f"\nLatest Assessment (Date: {latest.get('date','N/A')}):")
    print(json.dumps(latest, indent=2))
    return latest

def main():
    latest = load_latest_assessment()
    if not latest:
        return
    # Prepare health_data and bpm_data for the predictor
    health_data = {
        'age': latest.get('age'),
        'gender': latest.get('gender'),
        'symptoms': latest.get('symptoms', []),
        'medical_history': latest.get('medical_history', []),
        'risk_factors': latest.get('risk_factors', [])
    }
    # Use risk_scores if available for extra info
    bpm_data = None  # If you want to add BPM/temperature, add here
    predictor = DashboardHeatstrokePredictor()
    result = predictor.predict(health_data, bpm_data)
    print("\n--- Heatstroke Prediction Result ---")
    print(json.dumps(result, indent=2))
    print(f"\nRisk Level: {result.get('risk_level','N/A')} (Probability: {result.get('probability','N/A'):.2f})")

if __name__ == "__main__":
    main() 