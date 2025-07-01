#!/usr/bin/env python3

import requests
import json

def test_heatstroke_prediction():
    """Test the heatstroke prediction functionality"""
    
    # Test data
    test_data = {
        "age": 35,
        "gender": "female", 
        "symptoms": ["headache", "dizziness"],
        "medical_history": ["diabetes"],
        "risk_factors": ["high_blood_pressure"],
        "facial_analysis": None
    }
    
    try:
        # Test the submit_demographics endpoint
        print("Testing demographics submission with heatstroke prediction...")
        response = requests.post(
            'http://localhost:5003/submit_demographics',
            headers={'Content-Type': 'application/json'},
            json=test_data
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Demographics submission successful!")
            print(f"Success: {result.get('success')}")
            print(f"Full response: {json.dumps(result, indent=2)}")
            
            if result.get('heatstroke_prediction'):
                prediction = result['heatstroke_prediction']
                print("\n🔥 Heatstroke Prediction Results:")
                print(f"   Risk Level: {prediction.get('risk_level', 'N/A')}")
                print(f"   Probability: {prediction.get('heatstroke_probability', 0) * 100:.1f}%")
                print(f"   High Risk: {prediction.get('heatstroke_prediction', False)}")
                
                if prediction.get('feature_values'):
                    print("\n🔍 Key Features:")
                    for feature, value in prediction['feature_values'].items():
                        if value != 0:
                            print(f"   {feature}: {value}")
            else:
                print("❌ No heatstroke prediction in response")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Error testing heatstroke prediction: {e}")

if __name__ == "__main__":
    test_heatstroke_prediction() 