#!/usr/bin/env python3
"""
Simple MediaPipe Test - Basic functionality test
"""

import sys
import os

def test_basic_imports():
    """Test basic imports"""
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas imported successfully")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    return True

def test_mediapipe_import():
    """Test MediaPipe import"""
    try:
        import mediapipe as mp
        print("✅ MediaPipe imported successfully")
        return True
    except ImportError as e:
        print(f"❌ MediaPipe import failed: {e}")
        return False

def test_opencv_import():
    """Test OpenCV import"""
    try:
        import cv2
        print("✅ OpenCV imported successfully")
        return True
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False

def test_tensorflow_import():
    """Test TensorFlow import"""
    try:
        import tensorflow as tf
        print("✅ TensorFlow imported successfully")
        return True
    except ImportError as e:
        print(f"❌ TensorFlow import failed: {e}")
        return False

def test_sklearn_import():
    """Test scikit-learn import"""
    try:
        import sklearn
        print("✅ scikit-learn imported successfully")
        return True
    except ImportError as e:
        print(f"❌ scikit-learn import failed: {e}")
        return False

def test_joblib_import():
    """Test joblib import"""
    try:
        import joblib
        print("✅ joblib imported successfully")
        return True
    except ImportError as e:
        print(f"❌ joblib import failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Testing MediaPipe Dependencies")
    print("=" * 40)
    
    tests = [
        ("Basic Libraries", test_basic_imports),
        ("MediaPipe", test_mediapipe_import),
        ("OpenCV", test_opencv_import),
        ("TensorFlow", test_tensorflow_import),
        ("scikit-learn", test_sklearn_import),
        ("joblib", test_joblib_import),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        results[test_name] = test_func()
    
    print("\n" + "=" * 40)
    print("📊 Test Results:")
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All dependencies are available!")
        print("MediaPipe stress/fatigue analysis should work correctly.")
    else:
        print("\n⚠️ Some dependencies are missing.")
        print("Please install missing packages:")
        print("pip3 install opencv-python mediapipe tensorflow scikit-learn joblib")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 