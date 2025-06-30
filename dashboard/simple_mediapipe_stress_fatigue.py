#!/usr/bin/env python3
"""
Simplified MediaPipe-based Stress and Fatigue Detection Model
Uses facial landmarks for accurate prediction without complex dependencies
"""

import numpy as np
import json
from datetime import datetime
import os

class SimpleMediaPipeStressFatigueModel:
    """
    Simplified stress and fatigue detection using facial landmarks
    """
    
    def __init__(self):
        self.is_initialized = True
        self.model_version = "2.0"
        self.analysis_method = "Facial Landmark Analysis"
        
        print("✅ Simple MediaPipe model initialized successfully")
    
    def predict_stress_fatigue(self, facial_data, physiological_data, demographic_data):
        """Predict stress and fatigue using facial features"""
        try:
            # Extract facial features (14 values)
            facial_features = np.array(facial_data[:14])
            
            # Extract physiological features
            physiological_features = self._extract_physiological_features(physiological_data)
            
            # Calculate stress score based on facial features
            stress_score = self._calculate_stress_score(facial_features, physiological_features)
            
            # Calculate fatigue score based on facial features
            fatigue_score = self._calculate_fatigue_score(facial_features, physiological_features)
            
            # Calculate confidence
            confidence = self._calculate_confidence(facial_features, physiological_features)
            
            # Determine levels
            stress_level = self._get_stress_level(stress_score)
            fatigue_level = self._get_fatigue_level(fatigue_score)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(stress_score, fatigue_score, demographic_data)
            
            return {
                'stress_score': round(float(stress_score), 3),
                'fatigue_score': round(float(fatigue_score), 3),
                'confidence': round(float(confidence), 3),
                'model_available': True,
                'model_version': self.model_version,
                'analysis_method': self.analysis_method,
                'timestamp': datetime.now().isoformat(),
                'stress_level': stress_level,
                'fatigue_level': fatigue_level,
                'facial_features': facial_features.tolist(),
                'recommendations': recommendations
            }
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return {
                'stress_score': 0.0,
                'fatigue_score': 0.0,
                'confidence': 0.0,
                'model_available': False,
                'error': str(e)
            }
    
    def _extract_physiological_features(self, physiological_data):
        """Extract physiological features from data"""
        try:
            heart_rate = physiological_data.get('heart_rate', 75)
            hrv = physiological_data.get('hrv', 40)
            skin_temperature = physiological_data.get('skin_temperature', 36.5)
            respiration_rate = physiological_data.get('respiration_rate', 16)
            
            return np.array([heart_rate, hrv, skin_temperature, respiration_rate])
        except Exception as e:
            print(f"Error extracting physiological features: {e}")
            return np.array([75, 40, 36.5, 16])  # Default values
    
    def _calculate_stress_score(self, facial_features, physiological_features):
        """Calculate stress score based on facial and physiological features"""
        stress_score = 0.0
        
        # Facial stress indicators
        left_eye, right_eye = facial_features[0], facial_features[1]
        left_brow, right_brow = facial_features[2], facial_features[3]
        jaw_tension = facial_features[7]
        cheek_tension = facial_features[8]
        eye_asymmetry = facial_features[9]
        brow_asymmetry = facial_features[10]
        
        # Physiological stress indicators
        heart_rate = physiological_features[0]
        hrv = physiological_features[1]
        skin_temperature = physiological_features[2]
        respiration_rate = physiological_features[3]
        
        # Calculate stress score
        if left_eye < 0.4 or right_eye < 0.4:  # Eye squinting
            stress_score += 0.2
        if left_brow < 0.3 or right_brow < 0.3:  # Furrowed brows
            stress_score += 0.2
        if jaw_tension > 0.7:  # Jaw clenching
            stress_score += 0.2
        if cheek_tension > 0.6:  # Cheek tension
            stress_score += 0.1
        if eye_asymmetry > 0.3 or brow_asymmetry > 0.3:  # Facial asymmetry
            stress_score += 0.1
        
        # Physiological indicators
        if heart_rate > 90:  # High heart rate
            stress_score += 0.2
        if hrv < 20:  # Low HRV
            stress_score += 0.2
        if skin_temperature > 37.5:  # Elevated temperature
            stress_score += 0.1
        if respiration_rate > 20:  # High respiration rate
            stress_score += 0.1
        
        return np.clip(stress_score, 0.0, 1.0)
    
    def _calculate_fatigue_score(self, facial_features, physiological_features):
        """Calculate fatigue score based on facial and physiological features"""
        fatigue_score = 0.0
        
        # Facial fatigue indicators
        left_eye, right_eye = facial_features[0], facial_features[1]
        left_brow, right_brow = facial_features[2], facial_features[3]
        mouth_openness = facial_features[4]
        blink_rate = facial_features[12]
        pupil_dilation = facial_features[13]
        
        # Physiological fatigue indicators
        heart_rate = physiological_features[0]
        hrv = physiological_features[1]
        skin_temperature = physiological_features[2]
        respiration_rate = physiological_features[3]
        
        # Calculate fatigue score
        if left_eye < 0.3 or right_eye < 0.3:  # Droopy eyes
            fatigue_score += 0.3
        if left_brow < 0.2 or right_brow < 0.2:  # Droopy brows
            fatigue_score += 0.2
        if mouth_openness > 0.5:  # Open mouth (yawning)
            fatigue_score += 0.2
        if blink_rate < 0.3:  # Reduced blink rate
            fatigue_score += 0.1
        if pupil_dilation < 0.2:  # Constricted pupils
            fatigue_score += 0.1
        
        # Physiological indicators
        if heart_rate < 60:  # Low heart rate
            fatigue_score += 0.1
        if hrv < 15:  # Very low HRV
            fatigue_score += 0.1
        if skin_temperature < 35.5:  # Low temperature
            fatigue_score += 0.1
        if respiration_rate < 12:  # Slow respiration
            fatigue_score += 0.1
        
        return np.clip(fatigue_score, 0.0, 1.0)
    
    def _calculate_confidence(self, facial_features, physiological_features):
        """Calculate confidence in the prediction"""
        # Base confidence
        confidence = 0.8
        
        # Adjust based on feature quality
        if np.any(facial_features < 0.1) or np.any(facial_features > 0.9):
            confidence -= 0.1  # Extreme values reduce confidence
        
        if np.any(physiological_features < 0) or np.any(physiological_features > 200):
            confidence -= 0.1  # Unrealistic physiological values
        
        return np.clip(confidence, 0.5, 1.0)
    
    def _get_stress_level(self, stress_score):
        """Get stress level description"""
        if stress_score < 0.3:
            return "Low Stress"
        elif stress_score < 0.6:
            return "Moderate Stress"
        else:
            return "High Stress"
    
    def _get_fatigue_level(self, fatigue_score):
        """Get fatigue level description"""
        if fatigue_score < 0.3:
            return "Low Fatigue"
        elif fatigue_score < 0.6:
            return "Moderate Fatigue"
        else:
            return "High Fatigue"
    
    def _generate_recommendations(self, stress_score, fatigue_score, demographic_data):
        """Generate personalized recommendations"""
        recommendations = []
        
        if stress_score > 0.6:
            recommendations.append("🧘 <strong>Practice deep breathing exercises</strong> - Take 5-10 deep breaths to activate your parasympathetic nervous system")
            recommendations.append("💧 <strong>Stay hydrated</strong> - Dehydration can increase stress levels")
            recommendations.append("🏃 <strong>Take a short walk</strong> - Physical activity helps reduce cortisol levels")
        
        if fatigue_score > 0.6:
            recommendations.append("😴 <strong>Consider taking a short break</strong> - 10-15 minutes of rest can improve alertness")
            recommendations.append("☕ <strong>Limit caffeine intake</strong> - Too much caffeine can worsen fatigue later")
            recommendations.append("💡 <strong>Ensure proper lighting</strong> - Poor lighting can contribute to eye strain and fatigue")
        
        if stress_score < 0.3 and fatigue_score < 0.3:
            recommendations.append("✅ <strong>Great job!</strong> - Your stress and fatigue levels are well-managed")
            recommendations.append("🎯 <strong>Maintain your routine</strong> - Keep up with your healthy habits")
        
        return recommendations

# Initialize the simplified MediaPipe model
simple_mediapipe_model = SimpleMediaPipeStressFatigueModel() 