"""
Heat Stroke Assessment Module
Integrates with Health Monitoring Dashboard to provide real-time heat stroke risk assessment
"""

import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

class HeatStrokeAssessment:
    """
    Comprehensive heat stroke assessment system based on medical guidelines
    """
    
    def __init__(self):
        self.assessment_history = []
        self.risk_factors = {
            'age_risk': {'elderly': 65, 'young_children': 5},
            'temp_threshold': 104.0,  # Fahrenheit
            'temp_threshold_celsius': 40.0,
            'heart_rate_threshold': 100,
            'respiratory_rate_threshold': 20
        }
    
    def assess_heat_stroke_risk(self, 
                               core_temp: float,
                               mental_status: str,
                               heart_rate: int,
                               respiratory_rate: int,
                               age: int,
                               activity_level: str,
                               environmental_temp: float,
                               humidity: float,
                               symptoms: List[str],
                               medical_history: Dict[str, bool],
                               hydration_status: str) -> Dict:
        """
        Comprehensive heat stroke risk assessment
        
        Args:
            core_temp: Core body temperature in Fahrenheit
            mental_status: Mental status assessment
            heart_rate: Heart rate in BPM
            respiratory_rate: Respiratory rate per minute
            age: Patient age
            activity_level: Current activity level
            environmental_temp: Environmental temperature in Fahrenheit
            humidity: Environmental humidity percentage
            symptoms: List of reported symptoms
            medical_history: Dictionary of medical conditions
            hydration_status: Hydration assessment
            
        Returns:
            Dictionary containing risk assessment and recommendations
        """
        
        assessment = {
            'timestamp': datetime.now().isoformat(),
            'risk_level': 'LOW',
            'heat_stroke_probability': 0.0,
            'critical_factors': [],
            'recommendations': [],
            'emergency_action_required': False,
            'assessment_score': 0
        }
        
        score = 0
        critical_factors = []
        
        # 1. Core Temperature Assessment (Most Critical)
        temp_score = self._assess_temperature(core_temp)
        score += temp_score['score']
        if temp_score['critical']:
            critical_factors.append(f"Critical temperature: {core_temp}°F")
        
        # 2. Mental Status Evaluation
        mental_score = self._assess_mental_status(mental_status)
        score += mental_score['score']
        if mental_score['critical']:
            critical_factors.append(f"Altered mental status: {mental_status}")
        
        # 3. Vital Signs Assessment
        vitals_score = self._assess_vital_signs(heart_rate, respiratory_rate)
        score += vitals_score['score']
        
        # 4. Age and Risk Factors
        risk_score = self._assess_risk_factors(age, medical_history)
        score += risk_score['score']
        
        # 5. Environmental Factors
        env_score = self._assess_environmental_factors(environmental_temp, humidity)
        score += env_score['score']
        
        # 6. Symptoms Assessment
        symptom_score = self._assess_symptoms(symptoms)
        score += symptom_score['score']
        
        # 7. Hydration Status
        hydration_score = self._assess_hydration(hydration_status)
        score += hydration_score['score']
        
        # Calculate overall risk
        assessment['assessment_score'] = score
        assessment['critical_factors'] = critical_factors
        
        # Determine risk level and probability
        if score >= 80 or len(critical_factors) >= 2:
            assessment['risk_level'] = 'CRITICAL'
            assessment['heat_stroke_probability'] = 0.9
            assessment['emergency_action_required'] = True
        elif score >= 60:
            assessment['risk_level'] = 'HIGH'
            assessment['heat_stroke_probability'] = 0.7
        elif score >= 40:
            assessment['risk_level'] = 'MODERATE'
            assessment['heat_stroke_probability'] = 0.4
        elif score >= 20:
            assessment['risk_level'] = 'LOW'
            assessment['heat_stroke_probability'] = 0.2
        else:
            assessment['risk_level'] = 'MINIMAL'
            assessment['heat_stroke_probability'] = 0.05
        
        # Generate recommendations
        assessment['recommendations'] = self._generate_recommendations(
            assessment['risk_level'], 
            critical_factors,
            core_temp,
            mental_status
        )
        
        # Store assessment in history
        self.assessment_history.append(assessment)
        
        return assessment
    
    def _assess_temperature(self, core_temp: float) -> Dict:
        """Assess core temperature risk"""
        if core_temp >= self.risk_factors['temp_threshold']:
            return {'score': 30, 'critical': True}
        elif core_temp >= 102:
            return {'score': 20, 'critical': False}
        elif core_temp >= 100:
            return {'score': 10, 'critical': False}
        else:
            return {'score': 0, 'critical': False}
    
    def _assess_mental_status(self, mental_status: str) -> Dict:
        """Assess mental status for CNS dysfunction"""
        critical_mental_states = [
            'confused', 'disoriented', 'agitated', 'delirious',
            'hallucinating', 'unconscious', 'seizure', 'slurred_speech'
        ]
        
        if mental_status.lower() in critical_mental_states:
            return {'score': 25, 'critical': True}
        elif mental_status.lower() in ['drowsy', 'lethargic', 'irritable']:
            return {'score': 15, 'critical': False}
        elif mental_status.lower() == 'normal':
            return {'score': 0, 'critical': False}
        else:
            return {'score': 10, 'critical': False}
    
    def _assess_vital_signs(self, heart_rate: int, respiratory_rate: int) -> Dict:
        """Assess vital signs"""
        score = 0
        
        if heart_rate >= 120:
            score += 15
        elif heart_rate >= 100:
            score += 10
        
        if respiratory_rate >= 25:
            score += 10
        elif respiratory_rate >= 20:
            score += 5
        
        return {'score': score, 'critical': False}
    
    def _assess_risk_factors(self, age: int, medical_history: Dict[str, bool]) -> Dict:
        """Assess age and medical risk factors"""
        score = 0
        
        # Age risk factors
        if age >= self.risk_factors['age_risk']['elderly'] or age <= self.risk_factors['age_risk']['young_children']:
            score += 10
        
        # Medical conditions
        high_risk_conditions = [
            'heart_disease', 'diabetes', 'respiratory_condition',
            'mental_illness', 'obesity', 'dehydration'
        ]
        
        for condition in high_risk_conditions:
            if medical_history.get(condition, False):
                score += 5
        
        return {'score': score, 'critical': False}
    
    def _assess_environmental_factors(self, env_temp: float, humidity: float) -> Dict:
        """Assess environmental risk factors"""
        score = 0
        
        if env_temp >= 95:
            score += 15
        elif env_temp >= 85:
            score += 10
        elif env_temp >= 75:
            score += 5
        
        if humidity >= 80:
            score += 10
        elif humidity >= 60:
            score += 5
        
        return {'score': score, 'critical': False}
    
    def _assess_symptoms(self, symptoms: List[str]) -> Dict:
        """Assess reported symptoms"""
        high_risk_symptoms = [
            'headache', 'dizziness', 'nausea', 'vomiting',
            'muscle_cramps', 'weakness', 'fatigue', 'hot_skin',
            'dry_skin', 'rapid_breathing', 'weak_pulse'
        ]
        
        score = 0
        for symptom in symptoms:
            if symptom.lower() in high_risk_symptoms:
                score += 5
        
        return {'score': min(score, 20), 'critical': False}
    
    def _assess_hydration(self, hydration_status: str) -> Dict:
        """Assess hydration status"""
        if hydration_status.lower() in ['dehydrated', 'severely_dehydrated']:
            return {'score': 15, 'critical': False}
        elif hydration_status.lower() == 'mildly_dehydrated':
            return {'score': 10, 'critical': False}
        elif hydration_status.lower() == 'well_hydrated':
            return {'score': 0, 'critical': False}
        else:
            return {'score': 5, 'critical': False}
    
    def _generate_recommendations(self, risk_level: str, critical_factors: List[str], 
                                core_temp: float, mental_status: str) -> List[str]:
        """Generate recommendations based on risk level"""
        recommendations = []
        
        if risk_level == 'CRITICAL':
            recommendations.extend([
                "🚨 EMERGENCY: Call 911 immediately",
                "🚨 Initiate immediate cooling measures",
                "🚨 Move to cool environment",
                "🚨 Remove excess clothing",
                "🚨 Apply ice packs to neck, armpits, and groin",
                "🚨 Monitor vital signs continuously"
            ])
        elif risk_level == 'HIGH':
            recommendations.extend([
                "⚠️ Seek immediate medical attention",
                "⚠️ Move to air-conditioned environment",
                "⚠️ Apply cool compresses",
                "⚠️ Drink cool fluids if conscious",
                "⚠️ Monitor for worsening symptoms"
            ])
        elif risk_level == 'MODERATE':
            recommendations.extend([
                "📋 Rest in cool environment",
                "📋 Increase fluid intake",
                "📋 Monitor symptoms closely",
                "📋 Avoid strenuous activity",
                "📋 Seek medical attention if symptoms worsen"
            ])
        elif risk_level == 'LOW':
            recommendations.extend([
                "✅ Stay hydrated",
                "✅ Take breaks in cool areas",
                "✅ Monitor for new symptoms",
                "✅ Avoid prolonged heat exposure"
            ])
        else:
            recommendations.extend([
                "✅ Continue normal activities",
                "✅ Maintain good hydration",
                "✅ Monitor environmental conditions"
            ])
        
        return recommendations
    
    def get_assessment_history(self, limit: int = 10) -> List[Dict]:
        """Get recent assessment history"""
        return self.assessment_history[-limit:] if self.assessment_history else []
    
    def get_risk_trend(self) -> Dict:
        """Analyze risk trend over time"""
        if len(self.assessment_history) < 2:
            return {'trend': 'insufficient_data', 'change': 0}
        
        recent_scores = [a['assessment_score'] for a in self.assessment_history[-5:]]
        if len(recent_scores) >= 2:
            change = recent_scores[-1] - recent_scores[0]
            if change > 10:
                trend = 'increasing'
            elif change < -10:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
            change = 0
        
        return {'trend': trend, 'change': change}

# Global instance for dashboard integration
heat_stroke_assessor = HeatStrokeAssessment() 