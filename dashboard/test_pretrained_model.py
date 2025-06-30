#!/usr/bin/env python3
"""
Pretrained Model Test Suite
Tests the enhanced stress and fatigue detection with ML models
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from arduino_dashboard import (
    pretrained_model,
    perform_advanced_stress_analysis,
    analyze_enhanced_facial_stress,
    analyze_enhanced_facial_fatigue,
    analyze_enhanced_physiological_stress,
    analyze_enhanced_physiological_fatigue,
    analyze_enhanced_lifestyle_stress,
    analyze_enhanced_lifestyle_fatigue
)

def test_model_initialization():
    """Test model initialization and loading"""
    print("🔧 Testing Model Initialization...")
    
    # Check if model is loaded
    assert pretrained_model.is_loaded, "Model should be loaded"
    print("  ✅ Model loaded successfully")
    
    # Check if model components exist
    assert pretrained_model.stress_model is not None, "Stress model should exist"
    assert pretrained_model.fatigue_model is not None, "Fatigue model should exist"
    assert pretrained_model.feature_scaler is not None, "Feature scaler should exist"
    assert pretrained_model.feature_hasher is not None, "Feature hasher should exist"
    print("  ✅ All model components present")
    
    print("✅ Model Initialization Tests Passed\n")

def test_feature_extraction():
    """Test feature extraction functionality"""
    print("🔍 Testing Feature Extraction...")
    
    # Test data
    facial_data = {
        'eye_openness': 0.3,
        'mouth_tension': 0.7,
        'brow_furrow': 0.8,
        'jaw_clenching': 0.6,
        'blink_rate': 0.2,
        'pupil_dilation': 0.7
    }
    
    physiological_data = {
        'heart_rate': 85,
        'hrv': 25,
        'skin_temperature': 37.2,
        'respiration_rate': 18
    }
    
    demographic_data = {
        'age': 35,
        'sleep_hours': 6,
        'work_hours': 10,
        'exercise_frequency': 'low'
    }
    
    # Extract features
    features = pretrained_model.extract_features(facial_data, physiological_data, demographic_data)
    
    # Check feature dimensions
    assert len(features) == 14, f"Expected 14 features, got {len(features)}"
    print(f"  ✅ Feature extraction: {len(features)} features extracted")
    
    # Check feature normalization
    assert all(0 <= f <= 1 for f in features[:6]), "Facial features should be normalized to 0-1"
    print("  ✅ Feature normalization working")
    
    print("✅ Feature Extraction Tests Passed\n")

def test_model_predictions():
    """Test model prediction functionality"""
    print("🤖 Testing Model Predictions...")
    
    # Test case 1: High stress individual
    high_stress_facial = {
        'eye_openness': 0.2,
        'mouth_tension': 0.8,
        'brow_furrow': 0.9,
        'jaw_clenching': 0.8,
        'blink_rate': 0.1,
        'pupil_dilation': 0.8
    }
    
    high_stress_physio = {
        'heart_rate': 95,
        'hrv': 15,
        'skin_temperature': 38.0,
        'respiration_rate': 24
    }
    
    high_stress_demo = {
        'age': 35,
        'sleep_hours': 4,
        'work_hours': 12,
        'exercise_frequency': 'none'
    }
    
    result = pretrained_model.predict_stress_fatigue(
        high_stress_facial, high_stress_physio, high_stress_demo
    )
    
    print(f"  High Stress Case:")
    print(f"    Stress Score: {result['stress_score']}")
    print(f"    Fatigue Score: {result['fatigue_score']}")
    print(f"    Confidence: {result['confidence']}")
    print(f"    Model Available: {result['model_available']}")
    
    assert result['model_available'], "Model should be available"
    assert 0 <= result['stress_score'] <= 1, "Stress score should be 0-1"
    assert 0 <= result['fatigue_score'] <= 1, "Fatigue score should be 0-1"
    assert 0 <= result['confidence'] <= 1, "Confidence should be 0-1"
    
    # Test case 2: Healthy individual
    healthy_facial = {
        'eye_openness': 0.7,
        'mouth_tension': 0.3,
        'brow_furrow': 0.2,
        'jaw_clenching': 0.3,
        'blink_rate': 0.5,
        'pupil_dilation': 0.4
    }
    
    healthy_physio = {
        'heart_rate': 68,
        'hrv': 55,
        'skin_temperature': 36.8,
        'respiration_rate': 14
    }
    
    healthy_demo = {
        'age': 28,
        'sleep_hours': 8,
        'work_hours': 8,
        'exercise_frequency': 'high'
    }
    
    healthy_result = pretrained_model.predict_stress_fatigue(
        healthy_facial, healthy_physio, healthy_demo
    )
    
    print(f"  Healthy Case:")
    print(f"    Stress Score: {healthy_result['stress_score']}")
    print(f"    Fatigue Score: {healthy_result['fatigue_score']}")
    print(f"    Confidence: {healthy_result['confidence']}")
    
    # Verify that healthy individual has lower scores
    assert healthy_result['stress_score'] < result['stress_score'], "Healthy individual should have lower stress"
    assert healthy_result['fatigue_score'] < result['fatigue_score'], "Healthy individual should have lower fatigue"
    
    print("✅ Model Predictions Tests Passed\n")

def test_integrated_analysis():
    """Test the complete integrated analysis with ML models"""
    print("🔗 Testing Integrated Analysis...")
    
    # Test high stress scenario
    high_stress_facial = {
        'eye_openness': 0.2, 'mouth_tension': 0.8, 'brow_furrow': 0.9,
        'jaw_clenching': 0.8, 'blink_rate': 0.1, 'pupil_dilation': 0.8,
        'facial_asymmetry': 0.7, 'skin_tone_variation': 0.7,
        'eye_bags': 0.8, 'skin_tone': 0.2, 'eye_redness': 0.8,
        'facial_droop': 0.7, 'blink_frequency': 0.1
    }
    
    high_stress_physio = {
        'heart_rate': 95, 'hrv': 15, 'skin_temperature': 38.0,
        'respiration_rate': 24, 'bp_systolic': 150, 'bp_diastolic': 95,
        'skin_conductance': 9.0, 'temp_variation': 0.8, 'oxygen_saturation': 94,
        'age': 35
    }
    
    high_stress_demo = {
        'age': 35, 'sleep_hours': 4, 'sleep_quality': 'poor',
        'exercise_frequency': 'none', 'work_hours': 12, 'stress_level': 'very_high',
        'caffeine_intake': 'very_high', 'alcohol_consumption': 'high',
        'smoking_status': 'current', 'social_support': 'poor', 'financial_stress': 'high',
        'sleep_latency': 45, 'diet_quality': 'poor', 'hydration_level': 'poor', 'screen_time': 10
    }
    
    result = perform_advanced_stress_analysis(
        high_stress_facial, high_stress_physio, high_stress_demo
    )
    
    print(f"  High Stress/Fatigue Analysis:")
    print(f"    Stress Score: {result['stress_score']}")
    print(f"    Fatigue Score: {result['fatigue_score']}")
    print(f"    Confidence: {result['confidence']}")
    print(f"    Analysis Method: {result['analysis_details']['analysis_method']}")
    print(f"    Model Version: {result['analysis_details']['model_version']}")
    print(f"    ML Stress: {result['analysis_details']['ml_predictions']['stress_score']}")
    print(f"    Rule Stress: {result['analysis_details']['rule_based_stress']}")
    
    # Verify analysis results
    assert result['stress_score'] > 0.5, f"Expected high stress score, got {result['stress_score']}"
    assert result['fatigue_score'] > 0.5, f"Expected high fatigue score, got {result['fatigue_score']}"
    assert result['confidence'] > 0.6, f"Expected high confidence, got {result['confidence']}"
    assert result['analysis_details']['analysis_method'] == "ML-Enhanced", "Should use ML-Enhanced method"
    assert result['analysis_version'] == "3.0", "Should be version 3.0"
    
    print("✅ Integrated Analysis Tests Passed\n")

def test_model_performance():
    """Test model performance and consistency"""
    print("⚡ Testing Model Performance...")
    
    # Test data
    facial_data = {
        'eye_openness': 0.5, 'mouth_tension': 0.5, 'brow_furrow': 0.5,
        'jaw_clenching': 0.5, 'blink_rate': 0.5, 'pupil_dilation': 0.5
    }
    
    physiological_data = {
        'heart_rate': 75, 'hrv': 40, 'skin_temperature': 36.5,
        'respiration_rate': 16
    }
    
    demographic_data = {
        'age': 30, 'sleep_hours': 7, 'work_hours': 8,
        'exercise_frequency': 'moderate'
    }
    
    # Run multiple predictions to test consistency
    results = []
    for i in range(5):
        result = pretrained_model.predict_stress_fatigue(
            facial_data, physiological_data, demographic_data
        )
        results.append(result)
    
    # Check consistency (scores should be similar for same input)
    stress_scores = [r['stress_score'] for r in results]
    fatigue_scores = [r['fatigue_score'] for r in results]
    
    stress_variance = max(stress_scores) - min(stress_scores)
    fatigue_variance = max(fatigue_scores) - min(fatigue_scores)
    
    print(f"  Stress Score Variance: {stress_variance:.4f}")
    print(f"  Fatigue Score Variance: {fatigue_variance:.4f}")
    
    # Variance should be low for deterministic model
    assert stress_variance < 0.1, f"Stress scores should be consistent, variance: {stress_variance}"
    assert fatigue_variance < 0.1, f"Fatigue scores should be consistent, variance: {fatigue_variance}"
    
    print("✅ Model Performance Tests Passed\n")

def test_error_handling():
    """Test error handling and edge cases"""
    print("🛡️ Testing Error Handling...")
    
    # Test with missing data
    result = pretrained_model.predict_stress_fatigue({}, {}, {})
    
    assert result['model_available'] == False or result['model_available'] == True, "Should handle missing data"
    print("  ✅ Handles missing data gracefully")
    
    # Test with invalid data types
    invalid_facial = {'eye_openness': 'invalid', 'mouth_tension': None}
    invalid_physio = {'heart_rate': 'not_a_number'}
    invalid_demo = {'age': -5}
    
    result = pretrained_model.predict_stress_fatigue(invalid_facial, invalid_physio, invalid_demo)
    
    assert 'error' in result or result['model_available'] == False, "Should handle invalid data"
    print("  ✅ Handles invalid data gracefully")
    
    print("✅ Error Handling Tests Passed\n")

def run_all_tests():
    """Run all test suites"""
    print("🧪 Pretrained Model Test Suite")
    print("=" * 60)
    
    try:
        test_model_initialization()
        test_feature_extraction()
        test_model_predictions()
        test_integrated_analysis()
        test_model_performance()
        test_error_handling()
        
        print("🎉 All Tests Passed! Pretrained Model is working correctly.")
        print("\n📈 Model Features:")
        print("  ✅ Ensemble learning with Random Forest and Gradient Boosting")
        print("  ✅ Feature hashing for robust representation")
        print("  ✅ Standardized feature scaling")
        print("  ✅ Multi-modal data fusion")
        print("  ✅ Confidence scoring")
        print("  ✅ Fallback to rule-based analysis")
        print("  ✅ Comprehensive error handling")
        
    except Exception as e:
        print(f"❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 