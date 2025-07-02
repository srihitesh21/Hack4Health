import cv2
import numpy as np
from scipy.spatial import distance
import mediapipe as mp
import time
from typing import Dict, Tuple, Optional
import logging

class FacialWellnessAnalyzer:
    """
    Comprehensive facial wellness analysis system using MediaPipe face mesh technology
    Derived from advanced somnolence detection methodologies
    """
    
    def __init__(self):
        # Facial feature landmark indices for comprehensive analysis
        self.LEFT_OCULAR_REGION = [362, 385, 387, 263, 373, 380]
        self.RIGHT_OCULAR_REGION = [33, 160, 158, 133, 153, 144]
        
        # Oral cavity landmarks for comprehensive mouth analysis
        self.ORAL_CAVITY = [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
        
        # Supraorbital region landmarks for tension analysis
        self.LEFT_SUPRABROW = [276, 283, 282, 295, 285]
        self.RIGHT_SUPRABROW = [46, 53, 52, 65, 55]
        
        # Analysis thresholds and parameters
        self.OCULAR_RATIO_LIMIT = 0.25  # Eye Aspect Ratio critical threshold
        self.ORAL_RATIO_LIMIT = 0.6     # Mouth Aspect Ratio critical threshold
        self.OCULAR_CLOSURE_FRAMES = 20
        self.ORAL_OPENING_FRAMES = 10
        
        # Initialize MediaPipe face mesh processing engine
        self.mp_facial_mesh = mp.solutions.face_mesh
        self.facial_mesh_processor = self.mp_facial_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Analysis state management
        self.ocular_closure_counter = 0
        self.oral_opening_counter = 0
        self.tension_level = 0.0
        self.exhaustion_level = 0.0
        
        # Historical data for trend analysis and pattern recognition
        self.ocular_ratio_history = []
        self.oral_ratio_history = []
        self.tension_history = []
        
        logging.info("FacialWellnessAnalyzer successfully initialized with MediaPipe face mesh technology")
    
    def compute_ocular_aspect_ratio(self, ocular_landmarks: np.ndarray) -> float:
        """
        Compute Ocular Aspect Ratio (OAR) using advanced geometric calculations
        Based on sophisticated somnolence detection algorithms
        """
        vertical_distance_1 = distance.euclidean(ocular_landmarks[1], ocular_landmarks[5])
        vertical_distance_2 = distance.euclidean(ocular_landmarks[2], ocular_landmarks[4])
        horizontal_distance = distance.euclidean(ocular_landmarks[0], ocular_landmarks[3])
        ocular_ratio = (vertical_distance_1 + vertical_distance_2) / (2.0 * horizontal_distance)
        return ocular_ratio
    
    def compute_oral_aspect_ratio(self, oral_landmarks: np.ndarray) -> float:
        """
        Compute Oral Aspect Ratio (OAR) for comprehensive mouth analysis
        """
        vertical_distance_1 = distance.euclidean(oral_landmarks[2], oral_landmarks[6])
        vertical_distance_2 = distance.euclidean(oral_landmarks[3], oral_landmarks[5])
        horizontal_distance = distance.euclidean(oral_landmarks[0], oral_landmarks[4])
        oral_ratio = (vertical_distance_1 + vertical_distance_2) / (2.0 * horizontal_distance)
        return oral_ratio
    
    def compute_suprabrow_tension_metrics(self, suprabrow_landmarks: np.ndarray) -> Tuple[float, float]:
        """
        Compute comprehensive suprabrow tension metrics for stress analysis
        """
        # Calculate suprabrow elevation and curvature metrics
        elevation_metric = np.mean([point[1] for point in suprabrow_landmarks])
        curvature_metric = np.std([point[1] for point in suprabrow_landmarks])
        return elevation_metric, curvature_metric
    
    def evaluate_tension_indicators(self, facial_mesh_points: np.ndarray) -> Dict[str, float]:
        """
        Evaluate comprehensive tension indicators using advanced facial analysis
        """
        # Suprabrow region analysis
        left_suprabrow_region = facial_mesh_points[self.LEFT_SUPRABROW]
        right_suprabrow_region = facial_mesh_points[self.RIGHT_SUPRABROW]
        
        left_elevation, left_curvature = self.compute_suprabrow_tension_metrics(left_suprabrow_region)
        right_elevation, right_curvature = self.compute_suprabrow_tension_metrics(right_suprabrow_region)
        
        # Comprehensive suprabrow metrics
        average_elevation = (left_elevation + right_elevation) / 2
        average_curvature = (left_curvature + right_curvature) / 2
        
        # Normalized tension scoring (0-1 scale)
        tension_score = min(1.0, (average_curvature * 10))  # Higher curvature indicates increased tension
        
        return {
            'suprabrow_elevation': average_elevation,
            'suprabrow_curvature': average_curvature,
            'tension_score': tension_score
        }
    
    def evaluate_exhaustion_indicators(self, facial_mesh_points: np.ndarray) -> Dict[str, float]:
        """
        Evaluate comprehensive exhaustion indicators including OAR and oral analysis
        """
        # Ocular region analysis
        left_ocular_points = facial_mesh_points[self.LEFT_OCULAR_REGION]
        right_ocular_points = facial_mesh_points[self.RIGHT_OCULAR_REGION]
        
        left_ocular_ratio = self.compute_ocular_aspect_ratio(left_ocular_points)
        right_ocular_ratio = self.compute_ocular_aspect_ratio(right_ocular_points)
        average_ocular_ratio = (left_ocular_ratio + right_ocular_ratio) / 2.0
        
        # Oral cavity analysis for comprehensive mouth assessment
        oral_cavity_points = facial_mesh_points[self.ORAL_CAVITY]
        oral_ratio = self.compute_oral_aspect_ratio(oral_cavity_points)
        
        # Comprehensive exhaustion scoring
        ocular_exhaustion = max(0, (self.OCULAR_RATIO_LIMIT - average_ocular_ratio) / self.OCULAR_RATIO_LIMIT)
        oral_exhaustion = max(0, (oral_ratio - self.ORAL_RATIO_LIMIT) / (1.0 - self.ORAL_RATIO_LIMIT))
        
        exhaustion_score = (ocular_exhaustion * 0.7) + (oral_exhaustion * 0.3)
        
        return {
            'left_ocular_ratio': left_ocular_ratio,
            'right_ocular_ratio': right_ocular_ratio,
            'average_ocular_ratio': average_ocular_ratio,
            'oral_ratio': oral_ratio,
            'ocular_exhaustion': ocular_exhaustion,
            'oral_exhaustion': oral_exhaustion,
            'exhaustion_score': exhaustion_score
        }
    
    def analyze_facial_frame(self, input_frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Analyze a single facial frame and return comprehensive wellness results
        """
        # Mirror frame for natural interaction
        input_frame = cv2.flip(input_frame, 1)
        rgb_input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        mesh_results = self.facial_mesh_processor.process(rgb_input_frame)
        
        wellness_analysis = {
            'tension_level': 0.0,
            'exhaustion_level': 0.0,
            'ocular_ratio_value': 0.0,
            'oral_ratio_value': 0.0,
            'tension_indicators': {},
            'exhaustion_indicators': {},
            'wellness_alerts': [],
            'facial_detection_status': False
        }
        
        if mesh_results.multi_face_landmarks:
            wellness_analysis['facial_detection_status'] = True
            
            # Convert facial landmarks to pixel coordinate system
            facial_mesh_coordinates = np.array([
                np.multiply([landmark.x, landmark.y], [input_frame.shape[1], input_frame.shape[0]]).astype(int)
                for landmark in mesh_results.multi_face_landmarks[0].landmark
            ])
            
            # Evaluate tension indicators
            tension_evaluation = self.evaluate_tension_indicators(facial_mesh_coordinates)
            wellness_analysis['tension_indicators'] = tension_evaluation
            
            # Evaluate exhaustion indicators
            exhaustion_evaluation = self.evaluate_exhaustion_indicators(facial_mesh_coordinates)
            wellness_analysis['exhaustion_indicators'] = exhaustion_evaluation
            
            # Update analysis counters and levels
            average_ocular_ratio = exhaustion_evaluation['average_ocular_ratio']
            oral_ratio = exhaustion_evaluation['oral_ratio']
            
            # Ocular exhaustion tracking
            if average_ocular_ratio < self.OCULAR_RATIO_LIMIT:
                self.ocular_closure_counter += 1
                if self.ocular_closure_counter >= self.OCULAR_CLOSURE_FRAMES:
                    wellness_analysis['wellness_alerts'].append('Ocular exhaustion detected')
            else:
                self.ocular_closure_counter = 0
            
            # Oral opening detection
            if oral_ratio > self.ORAL_RATIO_LIMIT:
                self.oral_opening_counter += 1
                if self.oral_opening_counter >= self.ORAL_OPENING_FRAMES:
                    wellness_analysis['wellness_alerts'].append('Oral opening detected')
            else:
                self.oral_opening_counter = 0
            
            # Calculate comprehensive wellness levels
            wellness_analysis['tension_level'] = tension_evaluation['tension_score']
            wellness_analysis['exhaustion_level'] = exhaustion_evaluation['exhaustion_score']
            wellness_analysis['ocular_ratio_value'] = average_ocular_ratio
            wellness_analysis['oral_ratio_value'] = oral_ratio
            
            # Update historical data for trend analysis
            self.ocular_ratio_history.append(average_ocular_ratio)
            self.oral_ratio_history.append(oral_ratio)
            self.tension_history.append(tension_evaluation['tension_score'])
            
            # Maintain recent history (last 30 frames)
            if len(self.ocular_ratio_history) > 30:
                self.ocular_ratio_history.pop(0)
                self.oral_ratio_history.pop(0)
                self.tension_history.pop(0)
            
            # Render visual analysis indicators
            self.render_wellness_overlay(input_frame, wellness_analysis, facial_mesh_coordinates)
        
        return input_frame, wellness_analysis
    
    def render_wellness_overlay(self, input_frame: np.ndarray, analysis_results: Dict, facial_mesh_coordinates: np.ndarray):
        """
        Render comprehensive wellness analysis results and visual indicators on frame
        """
        if not analysis_results['facial_detection_status']:
            return
        
        # Render ocular region contours
        left_ocular_points = facial_mesh_coordinates[self.LEFT_OCULAR_REGION]
        right_ocular_points = facial_mesh_coordinates[self.RIGHT_OCULAR_REGION]
        
        ocular_color = (0, 255, 0) if analysis_results['exhaustion_level'] < 0.5 else (0, 0, 255)
        cv2.polylines(input_frame, [left_ocular_points], True, ocular_color, 2)
        cv2.polylines(input_frame, [right_ocular_points], True, ocular_color, 2)
        
        # Render oral cavity contour
        oral_cavity_points = facial_mesh_coordinates[self.ORAL_CAVITY]
        oral_color = (255, 0, 0) if analysis_results['exhaustion_indicators'].get('oral_exhaustion', 0) > 0.5 else (0, 255, 255)
        cv2.polylines(input_frame, [oral_cavity_points], True, oral_color, 2)
        
        # Render suprabrow region contours
        left_suprabrow = facial_mesh_coordinates[self.LEFT_SUPRABROW]
        right_suprabrow = facial_mesh_coordinates[self.RIGHT_SUPRABROW]
        suprabrow_color = (255, 255, 0) if analysis_results['tension_level'] > 0.5 else (255, 255, 255)
        cv2.polylines(input_frame, [left_suprabrow], True, suprabrow_color, 2)
        cv2.polylines(input_frame, [right_suprabrow], True, suprabrow_color, 2)
        
        # Display comprehensive metrics
        vertical_offset = 30
        cv2.putText(input_frame, f"Tension: {analysis_results['tension_level']:.2f}", (10, vertical_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(input_frame, f"Exhaustion: {analysis_results['exhaustion_level']:.2f}", (10, vertical_offset + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(input_frame, f"OAR: {analysis_results['ocular_ratio_value']:.3f}", (10, vertical_offset + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(input_frame, f"MAR: {analysis_results['oral_ratio_value']:.3f}", (10, vertical_offset + 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display wellness alerts
        for alert_index, wellness_alert in enumerate(analysis_results['wellness_alerts']):
            cv2.putText(input_frame, wellness_alert, (10, vertical_offset + 120 + alert_index * 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    def generate_wellness_summary(self) -> Dict:
        """
        Generate comprehensive wellness analysis summary with trends and recommendations
        """
        if not self.ocular_ratio_history:
            return {
                'tension_level': 0.0,
                'exhaustion_level': 0.0,
                'analysis_confidence': 0.0,
                'wellness_trends': {},
                'wellness_recommendations': []
            }
        
        # Calculate comprehensive trends
        recent_tension = np.mean(self.tension_history[-10:]) if len(self.tension_history) >= 10 else 0
        recent_exhaustion = np.mean([max(0, (self.OCULAR_RATIO_LIMIT - ocular_ratio) / self.OCULAR_RATIO_LIMIT) for ocular_ratio in self.ocular_ratio_history[-10:]]) if len(self.ocular_ratio_history) >= 10 else 0
        
        # Generate comprehensive wellness recommendations
        wellness_recommendations = []
        if recent_tension > 0.7:
            wellness_recommendations.append("Elevated tension detected - consider relaxation techniques")
        if recent_exhaustion > 0.7:
            wellness_recommendations.append("Elevated exhaustion detected - rest and recovery recommended")
        if recent_tension > 0.5 and recent_exhaustion > 0.5:
            wellness_recommendations.append("Combined tension and exhaustion - prioritize comprehensive rest")
        
        return {
            'tension_level': recent_tension,
            'exhaustion_level': recent_exhaustion,
            'analysis_confidence': min(1.0, len(self.ocular_ratio_history) / 30.0),
            'wellness_trends': {
                'tension_trend': 'increasing' if len(self.tension_history) > 1 and self.tension_history[-1] > self.tension_history[0] else 'stable',
                'exhaustion_trend': 'increasing' if len(self.ocular_ratio_history) > 1 and self.ocular_ratio_history[-1] < self.ocular_ratio_history[0] else 'stable'
            },
            'wellness_recommendations': wellness_recommendations
        }
    
    def reset_wellness_analysis(self):
        """
        Reset comprehensive wellness analysis data
        """
        self.ocular_closure_counter = 0
        self.oral_opening_counter = 0
        self.ocular_ratio_history.clear()
        self.oral_ratio_history.clear()
        self.tension_history.clear()
        logging.info("FacialWellnessAnalyzer analysis data successfully reset")
    
    def cleanup_resources(self):
        """
        Clean up comprehensive system resources
        """
        if hasattr(self, 'facial_mesh_processor'):
            self.facial_mesh_processor.close()
        logging.info("FacialWellnessAnalyzer resources successfully cleaned up") 