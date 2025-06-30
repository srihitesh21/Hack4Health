#!/usr/bin/env python3
import json
import sys
import os

# Add the CSV datasets directory to path
sys.path.append('../CSV datasets')

try:
    from heatstroke_prediction_integrated import HeatstrokePredictor
except ImportError:
    print("❌ Could not import HeatstrokePredictor. Make sure the CSV datasets directory is accessible.")
    sys.exit(1)

def check_stored_assessment_data():
    """Check stored health assessment data and calculate heatstroke risk"""
    
    # Check if health assessment data file exists
    data_file = 'health_assessment_data.json'
    if not os.path.exists(data_file):
        print("❌ No health assessment data file found")
        print("   Please complete a health assessment first")
        return
    
    try:
        # Load stored health assessment data
        with open(data_file, 'r') as f:
            stored_data = json.load(f)
        
        print('📋 Stored Health Assessment Data:')
        print(json.dumps(stored_data, indent=2))
        
        # Get the latest assessment
        if stored_data:
            latest_assessment = stored_data[-1] if isinstance(stored_data, list) else stored_data
            print(f'\n🔍 Latest Assessment:')
            print(f'Age: {latest_assessment.get("age", "N/A")}')
            print(f'Gender: {latest_assessment.get("gender", "N/A")}')
            print(f'Symptoms: {latest_assessment.get("symptoms", [])}')
            print(f'Medical History: {latest_assessment.get("medical_history", [])}')
            print(f'Risk Factors: {latest_assessment.get("risk_factors", [])}')
            
            # Calculate heatstroke risk
            print(f'\n🧮 Calculating Heatstroke Risk...')
            predictor = HeatstrokePredictor()
            
            # Get current BPM data (from the logs, BPM=82.5, skin temp=34.6°C)
            bpm_data = {
                'bpm': 82.5,
                'skin_temperature': 34.6
            }
            
            # Prepare prediction data
            prediction_data = {
                'health_data': {
                    'age': int(latest_assessment.get('age', 30)),
                    'gender': latest_assessment.get('gender', 'male'),
                    'symptoms': latest_assessment.get('symptoms', []),
                    'medical_history': latest_assessment.get('medical_history', []),
                    'risk_factors': latest_assessment.get('risk_factors', [])
                },
                'bpm_data': bpm_data
            }
            
            # Get prediction
            result = predictor.predict_heatstroke(prediction_data)
            
            print(f'\n🌡️ HEATSTROKE RISK ASSESSMENT:')
            print(f'Risk Level: {"HIGH" if result["heatstroke_prediction"] else "LOW"}')
            print(f'Probability: {result["probability"]*100:.1f}%')
            print(f'Current BPM: {bpm_data["bpm"]} BPM')
            print(f'Skin Temperature: {bpm_data["skin_temperature"]}°C')
            
            if result['heatstroke_prediction']:
                print('⚠️  HIGH RISK - Immediate attention recommended!')
                print('   • Seek medical attention immediately')
                print('   • Move to a cool environment')
                print('   • Remove excess clothing')
                print('   • Apply cool compresses')
            else:
                print('✅ LOW RISK - Continue monitoring')
                print('   • Stay hydrated')
                print('   • Take regular breaks')
                print('   • Monitor for any new symptoms')
                
        else:
            print('No stored assessment data found')
            
    except FileNotFoundError:
        print('No health assessment data file found')
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_stored_assessment_data() 