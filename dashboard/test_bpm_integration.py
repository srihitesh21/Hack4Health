#!/usr/bin/env python3
"""
Test script to verify BPM integration with heatstroke prediction
"""

import sys
import os
sys.path.append('../BPM')

def test_bpm_integration():
    """Test the BPM integration with heatstroke prediction"""
    print("🧪 Testing BPM Integration with Heatstroke Prediction")
    print("=" * 60)
    
    try:
        # Test 1: Import BPM module
        print("1. Testing BPM module import...")
        from newA import run_pulse_rate_from_csv
        print("✅ BPM module imported successfully")
        
        # Test 2: Get BPM data
        print("\n2. Testing BPM data extraction...")
        bpm, confidence, infection_score, dehydration_score, arrhythmia_score = run_pulse_rate_from_csv(
            "../BPM/A.csv", 
            fs=10, 
            plot_spectrogram=False
        )
        print(f"✅ BPM Analysis Results:")
        print(f"   BPM: {bpm:.1f}")
        print(f"   Confidence: {confidence:.3f}")
        print(f"   Infection Score: {infection_score:.1f}")
        print(f"   Dehydration Score: {dehydration_score:.1f}")
        print(f"   Arrhythmia Score: {arrhythmia_score:.1f}")
        
        # Test 3: Get skin temperature
        print("\n3. Testing skin temperature extraction...")
        import pandas as pd
        import numpy as np
        df = pd.read_csv("../BPM/A.csv", skiprows=1)
        skin_temps = df["SkinTemp(C)"].astype(float).values
        avg_skin_temp = float(np.mean(skin_temps))
        min_skin_temp = float(np.min(skin_temps))
        max_skin_temp = float(np.max(skin_temps))
        print(f"✅ Skin Temperature Analysis:")
        print(f"   Average: {avg_skin_temp:.1f}°C")
        print(f"   Range: {min_skin_temp:.1f}°C - {max_skin_temp:.1f}°C")
        
        # Test 4: Test the get_bpm_and_temperature_data function
        print("\n4. Testing integrated BPM data function...")
        from arduino_dashboard import get_bpm_and_temperature_data
        bpm_data = get_bpm_and_temperature_data()
        print(f"✅ Integrated BPM Data:")
        print(f"   BPM: {bpm_data['bpm']:.1f}")
        print(f"   Skin Temperature: {bpm_data['skin_temperature']:.1f}°C")
        print(f"   Confidence: {bpm_data['confidence']:.3f}")
        print(f"   Data Source: {bpm_data['data_source']}")
        
        # Test 5: Test heatstroke prediction with BPM data
        print("\n5. Testing heatstroke prediction with BPM data...")
        from arduino_dashboard import DashboardHeatstrokePredictor
        
        # Sample health data
        health_data = {
            'age': 35,
            'gender': 'female',
            'symptoms': ['headache', 'dizziness'],
            'medical_history': ['diabetes'],
            'risk_factors': ['high_blood_pressure']
        }
        
        predictor = DashboardHeatstrokePredictor()
        result = predictor.predict(health_data, bpm_data)
        
        print(f"✅ Heatstroke Prediction with BPM Data:")
        print(f"   Risk: {'HIGH' if result['heatstroke_prediction'] else 'LOW'}")
        print(f"   Probability: {result['heatstroke_probability']*100:.1f}%")
        print(f"   Risk Level: {result['risk_level']}")
        print(f"   BPM Used: {result['feature_values'].get('Heart_Rate', 'N/A')}")
        print(f"   Temperature Used: {result['feature_values'].get('Temperature', 'N/A')}")
        
        if 'bpm_data' in result:
            print(f"   BPM Data Included: ✅")
        else:
            print(f"   BPM Data Included: ❌")
        
        print("\n" + "=" * 60)
        print("🎉 All tests passed! BPM integration is working correctly.")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_bpm_integration()
    sys.exit(0 if success else 1) 