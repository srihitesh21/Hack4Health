# 🧠 Advanced Stress & Fatigue Analysis

This feature provides comprehensive stress and fatigue detection using multiple data sources and pre-trained models.

## 🌟 Features

### 📹 Real-time Facial Expression Analysis
- **Eye Openness Detection**: Monitors droopy eyelids and fatigue indicators
- **Brow Furrow Analysis**: Detects stress-related facial tension
- **Mouth Tension Assessment**: Identifies jaw clenching and stress indicators
- **Facial Asymmetry**: Measures muscle fatigue and stress patterns

### 💓 Physiological Data Integration
- **Heart Rate Analysis**: Elevated heart rate stress detection
- **Heart Rate Variability (HRV)**: Low HRV indicates stress/fatigue
- **Skin Temperature**: Stress response monitoring
- **Respiration Rate**: Breathing pattern analysis
- **Blood Pressure**: Cardiovascular stress indicators

### 🏃 Lifestyle & Demographic Factors
- **Sleep Patterns**: Duration and quality assessment
- **Exercise Frequency**: Physical activity impact
- **Work Hours**: Occupational stress factors
- **Caffeine Intake**: Compensatory behavior detection
- **Self-reported Stress**: Subjective stress levels

## 🔬 Analysis Algorithm

### Multi-Modal Scoring System
The analysis combines three data sources with weighted scoring:

1. **Facial Expression Analysis (40% weight)**
   - Eye openness, brow furrow, mouth tension, jaw clenching
   - Blink rate, eye bags, skin tone, facial asymmetry

2. **Physiological Data (35% weight)**
   - Heart rate, HRV, skin temperature, respiration rate
   - Blood pressure and other cardiovascular markers

3. **Lifestyle Factors (25% weight)**
   - Sleep patterns, exercise, work hours, stress self-report
   - Caffeine intake and other behavioral indicators

### Risk Level Classification

#### Stress Levels
- **Minimal** (0.0-0.2): No significant stress detected
- **Low** (0.2-0.4): Mild stress indicators present
- **Moderate** (0.4-0.6): Moderate stress requiring attention
- **High** (0.6-0.8): High stress requiring intervention
- **Critical** (0.8-1.0): Critical stress requiring immediate attention

#### Fatigue Levels
- **None** (0.0-0.2): No fatigue detected
- **Slight** (0.2-0.4): Mild fatigue indicators
- **Mild** (0.4-0.6): Moderate fatigue affecting function
- **Moderate** (0.6-0.8): Significant fatigue requiring rest
- **Severe** (0.8-1.0): Severe fatigue requiring medical attention

## 🚀 Usage

### Web Interface
1. Start the dashboard: `python3 arduino_dashboard.py`
2. Navigate to: `http://localhost:5002/stress_fatigue_analysis`
3. Allow camera access for facial analysis
4. Click "Start Camera" to begin
5. Click "Analyze" for facial feature detection
6. Click "Run Complete Analysis" for full assessment

### API Endpoint
```bash
POST /stress_fatigue_analysis
Content-Type: application/json

{
  "facial_data": {
    "eye_openness": 0.3,
    "brow_furrow": 0.7,
    "mouth_tension": 0.6,
    "jaw_clenching": 0.8,
    "blink_rate": 0.4,
    "eye_bags": 0.6,
    "skin_tone": 0.4,
    "facial_asymmetry": 0.3
  },
  "physiological_data": {
    "heart_rate": 95,
    "hrv": 25,
    "skin_temperature": 37.8,
    "respiration_rate": 22,
    "blood_pressure": 145
  },
  "demographic_data": {
    "age": 35,
    "sleep_hours": 5,
    "exercise_frequency": "none",
    "work_hours": 12,
    "stress_level": "high",
    "sleep_quality": "poor",
    "caffeine_intake": "high"
  }
}
```

### Response Format
```json
{
  "stress_score": 0.75,
  "fatigue_score": 0.68,
  "confidence": 0.92,
  "stress_level": "High",
  "fatigue_level": "Moderate",
  "recommendations": [
    {
      "category": "Stress Management",
      "priority": "High",
      "recommendations": [
        "Practice deep breathing exercises for 10-15 minutes daily",
        "Consider mindfulness meditation or yoga",
        "Take regular breaks during work (5-minute breaks every hour)"
      ]
    }
  ],
  "timestamp": "2025-06-30T11:30:00"
}
```

## 🧪 Testing

Run the test suite to verify functionality:

```bash
cd dashboard
python3 test_stress_analysis.py
```

This will test:
- Individual analysis functions
- Web API endpoint
- Data validation and scoring

## 🔧 Technical Implementation

### Facial Analysis
- **TensorFlow.js Integration**: Real-time face landmark detection
- **MediaPipe Face Mesh**: 468 facial landmarks for precise analysis
- **Feature Extraction**: Eye openness, brow movement, mouth tension
- **Temporal Analysis**: Blink rate and facial dynamics

### Physiological Analysis
- **Heart Rate Processing**: From Arduino PPG sensor data
- **HRV Calculation**: Time-domain and frequency-domain analysis
- **Temperature Monitoring**: Skin temperature stress response
- **Respiratory Analysis**: Breathing pattern stress indicators

### Machine Learning Models
- **Pre-trained Models**: Facial expression recognition
- **Stress Classification**: Multi-class stress level prediction
- **Fatigue Detection**: Pattern recognition for fatigue indicators
- **Personalization**: Adaptive thresholds based on individual baselines

## 📊 Research Basis

### Facial Expression Research
- **Facial Action Coding System (FACS)**: Standardized facial movement analysis
- **Corrugator Activity**: Brow furrowing as stress indicator
- **Orbicularis Oculi**: Eye muscle tension and fatigue
- **Masseter Activity**: Jaw clenching stress response

### Physiological Research
- **Heart Rate Variability**: Autonomic nervous system stress response
- **Skin Conductance**: Sympathetic nervous system activation
- **Respiratory Sinus Arrhythmia**: Vagal tone assessment
- **Blood Pressure Reactivity**: Cardiovascular stress markers

### Clinical Validation
- **Stress Biomarkers**: Cortisol, adrenaline, noradrenaline correlation
- **Fatigue Assessment**: Subjective vs. objective fatigue measures
- **Intervention Studies**: Stress reduction technique effectiveness
- **Longitudinal Studies**: Chronic stress pattern recognition

## 🛠️ Installation

### Dependencies
```bash
pip install -r requirements.txt
```

### Additional Requirements
- **Camera Access**: For facial analysis
- **Arduino Connection**: For physiological data (optional)
- **Modern Browser**: Chrome/Firefox with WebRTC support

## 🔒 Privacy & Security

### Data Protection
- **Local Processing**: All analysis performed locally
- **No Data Storage**: Facial images not stored permanently
- **Secure Transmission**: HTTPS for web interface
- **User Consent**: Camera permission required

### Ethical Considerations
- **Informed Consent**: Clear explanation of data usage
- **Medical Disclaimer**: Not a substitute for professional medical advice
- **Data Minimization**: Only necessary data collected
- **User Control**: Full control over data and analysis

## 🚨 Medical Disclaimer

This tool is for educational and research purposes only. It is not intended to:
- Replace professional medical diagnosis
- Provide medical treatment recommendations
- Substitute for clinical stress/fatigue assessment
- Diagnose medical conditions

Always consult with healthcare professionals for:
- Medical concerns
- Stress management strategies
- Fatigue assessment
- Mental health support

## 🔮 Future Enhancements

### Planned Features
- **Real-time Monitoring**: Continuous stress/fatigue tracking
- **Personalized Baselines**: Individual stress pattern learning
- **Intervention Tracking**: Stress reduction technique effectiveness
- **Mobile App**: iOS/Android stress monitoring
- **Integration**: Wearable device data fusion
- **AI Enhancement**: Deep learning model improvements

### Research Directions
- **Multi-modal Fusion**: Advanced sensor data integration
- **Predictive Analytics**: Stress/fatigue prediction
- **Personalization**: Individual-specific algorithms
- **Clinical Validation**: Medical device certification
- **Population Studies**: Large-scale stress pattern analysis

## 📞 Support

For technical support or questions:
- Check the main dashboard documentation
- Review the test suite for examples
- Examine the source code for implementation details
- Contact the development team for issues

---

**Version**: 1.0.0  
**Last Updated**: June 2025  
**Compatibility**: Python 3.8+, Modern Browsers 