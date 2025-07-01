#!/usr/bin/env python3
"""
Test script for MediaPipe Stress and Fatigue Model
"""

import sys
import os
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_mediapipe_import():
    """Test if MediaPipe model can be imported"""
    try:
        from mediapipe_stress_fatigue import mediapipe_model
        print("✅ MediaPipe model imported successfully")
        return mediapipe_model
    except ImportError as e:
        print(f"❌ Failed to import MediaPipe model: {e}")
        return None

def test_model_initialization(model):
    """Test if the model is properly initialized"""
    if model is None:
        print("❌ Model is None")
        return False
    
    try:
        if hasattr(model, 'is_initialized'):
            print(f"✅ Model initialization status: {model.is_initialized}")
            return model.is_initialized
        else:
            print("⚠️ Model doesn't have is_initialized attribute")
            return False
    except Exception as e:
        print(f"❌ Error checking model initialization: {e}")
        return False

def test_facial_feature_extraction(model):
    """Test facial feature extraction with sample data"""
    if model is None:
        print("❌ Model is None, skipping feature extraction test")
        return False
    
    try:
        # Create sample facial features (14 values)
        sample_facial_features = [
            0.6, 0.7,  # Left and right eye openness
            0.5, 0.5,  # Left and right brow height
            0.3, 0.4, 0.4,  # Mouth openness, left corner, right corner
            0.4, 0.3,  # Jaw tension, cheek tension
            0.1, 0.0, 0.0,  # Eye, brow, mouth asymmetry
            0.5, 0.5   # Blink rate, pupil dilation
        ]
        
        # Create sample physiological data
        sample_physiological_data = {
            'heart_rate': 75,
            'hrv': 40,
            'skin_temperature': 36.5,
            'respiration_rate': 16
        }
        
        # Create sample demographic data
        sample_demographic_data = {
            'age': 30,
            'gender': 'male',
            'sleep_hours': 7,
            'exercise_frequency': 'moderate'
        }
        
        print("🧪 Testing facial feature extraction...")
        result = model.predict_stress_fatigue(
            sample_facial_features, 
            sample_physiological_data, 
            sample_demographic_data
        )
        
        print("✅ Feature extraction test completed")
        print(f"   Stress Score: {result.get('stress_score', 'N/A')}")
        print(f"   Fatigue Score: {result.get('fatigue_score', 'N/A')}")
        print(f"   Confidence: {result.get('confidence', 'N/A')}")
        print(f"   Model Available: {result.get('model_available', 'N/A')}")
        
        if result.get('model_available', False):
            print("✅ Model is working correctly!")
            return True
        else:
            print(f"❌ Model not available: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error in feature extraction test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mediapipe_face_mesh():
    """Test if MediaPipe Face Mesh can be initialized"""
    try:
        import mediapipe as mp
        
        # Initialize MediaPipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        print("✅ MediaPipe Face Mesh initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error initializing MediaPipe Face Mesh: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Testing MediaPipe Stress and Fatigue Model")
    print("=" * 50)
    
    # Test 1: Import
    model = test_mediapipe_import()
    
    # Test 2: Initialization
    if model:
        init_success = test_model_initialization(model)
        
        # Test 3: Feature extraction
        if init_success:
            feature_success = test_facial_feature_extraction(model)
        else:
            feature_success = False
    else:
        init_success = False
        feature_success = False
    
    # Test 4: MediaPipe Face Mesh
    face_mesh_success = test_mediapipe_face_mesh()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"   Import: {'✅ PASS' if model else '❌ FAIL'}")
    print(f"   Initialization: {'✅ PASS' if init_success else '❌ FAIL'}")
    print(f"   Feature Extraction: {'✅ PASS' if feature_success else '❌ FAIL'}")
    print(f"   Face Mesh: {'✅ PASS' if face_mesh_success else '❌ FAIL'}")
    
    if model and init_success and feature_success and face_mesh_success:
        print("\n🎉 All tests passed! MediaPipe model is ready to use.")
        return True
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 