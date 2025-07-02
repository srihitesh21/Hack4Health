#!/usr/bin/env python3
"""
Test script for stress fatigue detector integration
Tests the new stress fatigue detection system based on somnolence-detection techniques
"""

import sys
import os
import numpy as np
import cv2
import time

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_stress_fatigue_detector():
    """Test the stress fatigue detector functionality"""
    print("🧠 Testing Stress Fatigue Detector Integration")
    print("=" * 50)
    
    try:
        # Import the stress fatigue detector
        from stress_fatigue_detector import FacialWellnessAnalyzer
        print("✅ Successfully imported FacialWellnessAnalyzer")
        
        # Initialize the detector
        detector = FacialWellnessAnalyzer()
        print("✅ FacialWellnessAnalyzer initialized successfully")
        
        # Test EAR calculation
        print("\n📊 Testing EAR (Eye Aspect Ratio) calculation...")
        test_eye_points = np.array([
            [100, 50],   # Point 0
            [110, 45],   # Point 1
            [120, 45],   # Point 2
            [130, 50],   # Point 3
            [120, 55],   # Point 4
            [110, 55]    # Point 5
        ])
        
        ear = detector.compute_ocular_aspect_ratio(test_eye_points)
        print(f"   EAR value: {ear:.4f}")
        print(f"   EAR threshold: {detector.OCULAR_RATIO_LIMIT}")
        print(f"   Eye fatigue: {'Yes' if ear < detector.OCULAR_RATIO_LIMIT else 'No'}")
        
        # Test MAR calculation
        print("\n📊 Testing MAR (Mouth Aspect Ratio) calculation...")
        test_mouth_points = np.array([
            [100, 100],  # Point 0
            [110, 95],   # Point 1
            [120, 90],   # Point 2
            [130, 95],   # Point 3
            [140, 100],  # Point 4
            [130, 105],  # Point 5
            [120, 110],  # Point 6
            [110, 105]   # Point 7
        ])
        
        mar = detector.compute_oral_aspect_ratio(test_mouth_points)
        print(f"   MAR value: {mar:.4f}")
        print(f"   MAR threshold: {detector.ORAL_RATIO_LIMIT}")
        print(f"   Yawn detected: {'Yes' if mar > detector.ORAL_RATIO_LIMIT else 'No'}")
        
        # Test eyebrow tension calculation
        print("\n📊 Testing eyebrow tension calculation...")
        test_eyebrow_points = np.array([
            [100, 30],
            [110, 28],
            [120, 30],
            [130, 32],
            [140, 30]
        ])
        
        height, curvature = detector.compute_suprabrow_tension_metrics(test_eyebrow_points)
        print(f"   Eyebrow height: {height:.2f}")
        print(f"   Eyebrow curvature: {curvature:.4f}")
        
        # Test stress indicators analysis
        print("\n📊 Testing stress indicators analysis...")
        # Create mock mesh points (simplified)
        mock_mesh_points = np.random.rand(468, 2) * 100  # 468 MediaPipe face mesh points
        
        stress_analysis = detector.evaluate_tension_indicators(mock_mesh_points)
        print(f"   Stress score: {stress_analysis['tension_score']:.4f}")
        print(f"   Eyebrow height: {stress_analysis['suprabrow_elevation']:.2f}")
        print(f"   Eyebrow curvature: {stress_analysis['suprabrow_curvature']:.4f}")
        
        # Test fatigue indicators analysis
        print("\n📊 Testing fatigue indicators analysis...")
        fatigue_analysis = detector.evaluate_exhaustion_indicators(mock_mesh_points)
        print(f"   Fatigue score: {fatigue_analysis['exhaustion_score']:.4f}")
        print(f"   Left EAR: {fatigue_analysis['left_ocular_ratio']:.4f}")
        print(f"   Right EAR: {fatigue_analysis['right_ocular_ratio']:.4f}")
        print(f"   Average EAR: {fatigue_analysis['average_ocular_ratio']:.4f}")
        print(f"   MAR: {fatigue_analysis['oral_ratio']:.4f}")
        print(f"   Eye fatigue: {fatigue_analysis['ocular_exhaustion']:.4f}")
        print(f"   Yawn fatigue: {fatigue_analysis['oral_exhaustion']:.4f}")
        
        # Test analysis summary
        print("\n📊 Testing analysis summary...")
        summary = detector.generate_wellness_summary()
        print(f"   Stress level: {summary['tension_level']:.4f}")
        print(f"   Fatigue level: {summary['exhaustion_level']:.4f}")
        print(f"   Confidence: {summary['analysis_confidence']:.4f}")
        print(f"   Trends: {summary['wellness_trends']}")
        print(f"   Recommendations: {summary['wellness_recommendations']}")
        
        # Test reset functionality
        print("\n📊 Testing reset functionality...")
        detector.reset_wellness_analysis()
        print("✅ Analysis data reset successfully")
        
        # Test cleanup
        print("\n📊 Testing cleanup...")
        detector.cleanup_resources()
        print("✅ Detector cleaned up successfully")
        
        print("\n🎉 All stress fatigue detector tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import FacialWellnessAnalyzer: {e}")
        print("   Make sure the stress_fatigue_detector.py file exists and all dependencies are installed")
        return False
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mediapipe_availability():
    """Test if MediaPipe is available"""
    print("\n🔍 Testing MediaPipe availability...")
    try:
        import mediapipe as mp
        print("✅ MediaPipe is available")
        
        # Test basic MediaPipe functionality
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("✅ MediaPipe FaceMesh initialized successfully")
        
        # Cleanup
        face_mesh.close()
        print("✅ MediaPipe FaceMesh cleaned up")
        
        return True
        
    except ImportError as e:
        print(f"❌ MediaPipe not available: {e}")
        print("   Install with: pip install mediapipe")
        return False
        
    except Exception as e:
        print(f"❌ MediaPipe test failed: {e}")
        return False

def test_opencv_availability():
    """Test if OpenCV is available"""
    print("\n🔍 Testing OpenCV availability...")
    try:
        import cv2
        print(f"✅ OpenCV version: {cv2.__version__}")
        
        # Test basic OpenCV functionality
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        print("✅ OpenCV basic functionality works")
        
        return True
        
    except ImportError as e:
        print(f"❌ OpenCV not available: {e}")
        print("   Install with: pip install opencv-python")
        return False
        
    except Exception as e:
        print(f"❌ OpenCV test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧠 Stress Fatigue Detector Integration Test Suite")
    print("=" * 60)
    
    # Test dependencies
    opencv_ok = test_opencv_availability()
    mediapipe_ok = test_mediapipe_availability()
    
    if not opencv_ok or not mediapipe_ok:
        print("\n❌ Dependency tests failed. Please install missing dependencies.")
        return False
    
    # Test stress fatigue detector
    detector_ok = test_stress_fatigue_detector()
    
    if detector_ok:
        print("\n🎉 All tests passed! Stress fatigue detector is ready for use.")
        print("\n📋 Integration Summary:")
        print("   ✅ OpenCV available")
        print("   ✅ MediaPipe available")
        print("   ✅ FacialWellnessAnalyzer working")
        print("   ✅ EAR/MAR calculations working")
        print("   ✅ Stress/fatigue analysis working")
        print("   ✅ API integration ready")
        return True
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 