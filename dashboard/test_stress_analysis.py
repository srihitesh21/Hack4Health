#!/usr/bin/env python3
"""
Test script for the advanced stress and fatigue analysis functionality
"""

import json
import requests
import time

def test_stress_analysis():
    """Test the stress and fatigue analysis endpoint"""
    
    # Test data
    test_data = {
        "facial_data": {
            "eye_openness": 0.3,  # Low eye openness (fatigue indicator)
            "brow_furrow": 0.7,   # High brow furrow (stress indicator)
            "mouth_tension": 0.6,  # Moderate mouth tension
            "jaw_clenching": 0.8,  # High jaw clenching (stress indicator)
            "blink_rate": 0.4,     # Normal blink rate
            "eye_bags": 0.6,       # Moderate eye bags (fatigue indicator)
            "skin_tone": 0.4,      # Dull skin tone (fatigue indicator)
            "facial_asymmetry": 0.3  # Low asymmetry
        },
        "physiological_data": {
            "heart_rate": 95,      # Elevated heart rate (stress indicator)
            "hrv": 25,             # Low HRV (stress/fatigue indicator)
            "skin_temperature": 37.8,  # Elevated temperature (stress indicator)
            "respiration_rate": 22,    # Elevated respiration (stress indicator)
            "blood_pressure": 145      # Elevated blood pressure
        },
        "demographic_data": {
            "age": 35,
            "sleep_hours": 5,      # Poor sleep (stress/fatigue indicator)
            "exercise_frequency": "none",  # No exercise (stress indicator)
            "work_hours": 12,      # Long work hours (stress indicator)
            "stress_level": "high", # Self-reported high stress
            "sleep_quality": "poor", # Poor sleep quality (fatigue indicator)
            "caffeine_intake": "high"  # High caffeine (compensatory for fatigue)
        }
    }
    
    print("🧠 Testing Advanced Stress & Fatigue Analysis")
    print("=" * 50)
    
    try:
        # Test the analysis endpoint
        print("📊 Sending test data to analysis endpoint...")
        response = requests.post(
            'http://localhost:5002/stress_fatigue_analysis',
            json=test_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if response.status_code == 200:
            results = response.json()
            print("✅ Analysis completed successfully!")
            print("\n📈 Results:")
            print(f"   Stress Score: {results.get('stress_score', 0):.2f}")
            print(f"   Fatigue Score: {results.get('fatigue_score', 0):.2f}")
            print(f"   Confidence: {results.get('confidence', 0):.2f}")
            print(f"   Stress Level: {results.get('stress_level', 'Unknown')}")
            print(f"   Fatigue Level: {results.get('fatigue_level', 'Unknown')}")
            
            # Display recommendations
            recommendations = results.get('recommendations', [])
            if recommendations:
                print(f"\n💡 Recommendations ({len(recommendations)} categories):")
                for i, category in enumerate(recommendations, 1):
                    print(f"   {i}. {category['category']} ({category['priority']} priority)")
                    for rec in category['recommendations']:
                        print(f"      • {rec}")
            else:
                print("\n💡 No specific recommendations generated")
                
        else:
            print(f"❌ Analysis failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the dashboard server")
        print("   Make sure the dashboard is running on http://localhost:5002")
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
    except Exception as e:
        print(f"❌ Error during testing: {e}")

def test_individual_functions():
    """Test individual analysis functions"""
    
    print("\n🔬 Testing Individual Analysis Functions")
    print("=" * 50)
    
    # Import the analysis functions
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        from arduino_dashboard import (
            analyze_facial_stress_indicators,
            analyze_facial_fatigue_indicators,
            analyze_physiological_stress,
            analyze_physiological_fatigue,
            analyze_lifestyle_stress_factors,
            analyze_lifestyle_fatigue_factors
        )
        
        # Test facial analysis
        facial_data = {
            "eye_openness": 0.3,
            "brow_furrow": 0.7,
            "mouth_tension": 0.6,
            "jaw_clenching": 0.8,
            "blink_rate": 0.4
        }
        
        stress_score = analyze_facial_stress_indicators(facial_data)
        fatigue_score = analyze_facial_fatigue_indicators(facial_data)
        
        print(f"📹 Facial Analysis:")
        print(f"   Stress Score: {stress_score:.2f}")
        print(f"   Fatigue Score: {fatigue_score:.2f}")
        
        # Test physiological analysis
        physio_data = {
            "heart_rate": 95,
            "hrv": 25,
            "skin_temperature": 37.8,
            "respiration_rate": 22
        }
        
        physio_stress = analyze_physiological_stress(physio_data)
        physio_fatigue = analyze_physiological_fatigue(physio_data)
        
        print(f"\n💓 Physiological Analysis:")
        print(f"   Stress Score: {physio_stress:.2f}")
        print(f"   Fatigue Score: {physio_fatigue:.2f}")
        
        # Test lifestyle analysis
        lifestyle_data = {
            "age": 35,
            "sleep_hours": 5,
            "exercise_frequency": "none",
            "work_hours": 12,
            "stress_level": "high",
            "sleep_quality": "poor",
            "caffeine_intake": "high"
        }
        
        lifestyle_stress = analyze_lifestyle_stress_factors(lifestyle_data)
        lifestyle_fatigue = analyze_lifestyle_fatigue_factors(lifestyle_data)
        
        print(f"\n🏃 Lifestyle Analysis:")
        print(f"   Stress Score: {lifestyle_stress:.2f}")
        print(f"   Fatigue Score: {lifestyle_fatigue:.2f}")
        
        # Calculate weighted scores
        total_stress = (stress_score * 0.4 + physio_stress * 0.35 + lifestyle_stress * 0.25)
        total_fatigue = (fatigue_score * 0.4 + physio_fatigue * 0.35 + lifestyle_fatigue * 0.25)
        
        print(f"\n📊 Combined Analysis:")
        print(f"   Total Stress Score: {total_stress:.2f}")
        print(f"   Total Fatigue Score: {total_fatigue:.2f}")
        
    except ImportError as e:
        print(f"❌ Could not import analysis functions: {e}")
        print("   Make sure arduino_dashboard.py is in the same directory")

if __name__ == "__main__":
    print("🧠 Advanced Stress & Fatigue Analysis Test Suite")
    print("=" * 60)
    
    # Test individual functions first
    test_individual_functions()
    
    # Test the web endpoint
    test_stress_analysis()
    
    print("\n✅ Test suite completed!")
    print("\nTo use the advanced analysis:")
    print("1. Start the dashboard: python3 arduino_dashboard.py")
    print("2. Open http://localhost:5002/stress_fatigue_analysis")
    print("3. Allow camera access and run the analysis") 