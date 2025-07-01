import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import warnings
import sys
import os

# Add the BPM directory to path to import newA.py
sys.path.append('../BPM')
from newA import run_pulse_rate_from_csv

warnings.filterwarnings('ignore')

class HeatstrokePredictor:
    """
    Heatstroke prediction using Random Forest model trained on health assessment data
    combined with real-time BPM data from newA.py
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = []
        self.is_trained = False
        
    def load_and_preprocess_data(self):
        """Load and preprocess the heatstroke datasets"""
        print("Loading heatstroke datasets...")
        
        # Load training and test datasets
        train_df = pd.read_csv('health_heatstroke_train_2000_temp (1).csv')
        test_df = pd.read_csv('health_heatstroke_test_2000_temp (1).csv')
        
        print(f"Training dataset shape: {train_df.shape}")
        print(f"Test dataset shape: {test_df.shape}")
        
        # Combine datasets for preprocessing
        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        
        # Preprocess the data
        X, y = self._preprocess_features(combined_df)
        
        # Split back into train and test
        train_size = len(train_df)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        return X_train, X_test, y_train, y_test
    
    def _preprocess_features(self, df):
        """Preprocess features for machine learning"""
        # Create a copy to avoid modifying original data
        data = df.copy()
        
        # Convert categorical variables to numerical
        categorical_cols = ['Gender', 'Headache', 'Dizziness', 'Nausea', 'Vomiting', 
                           'Muscle_Cramps', 'Weakness', 'Fatigue', 'Hot_Skin', 'Dry_Skin',
                           'Rapid_Breathing', 'Cardiovascular_Disease', 'Diabetes', 
                           'Obesity', 'Elderly', 'Heat_Sensitive_Medications', 'Dehydration',
                           'Alcohol_Use', 'Previous_Heat_Illness', 'Poor_Heat_Acclimation',
                           'Prolonged_Exertion', 'High_Blood_Pressure', 'Smoking']
        
        for col in categorical_cols:
            if col in data.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    data[col] = self.label_encoders[col].fit_transform(data[col].astype(str))
                else:
                    # Handle unseen categories in test data
                    data[col] = data[col].astype(str)
                    data[col] = data[col].map(lambda x: x if x in self.label_encoders[col].classes_ else 'Unknown')
                    data[col] = self.label_encoders[col].transform(data[col])
        
        # Select features for the model
        feature_cols = ['Age', 'Gender', 'Temperature', 'Heart_Rate', 'Blood_Pressure_Systolic',
                       'Blood_Pressure_Diastolic', 'Headache', 'Dizziness', 'Nausea', 'Vomiting',
                       'Muscle_Cramps', 'Weakness', 'Fatigue', 'Hot_Skin', 'Dry_Skin',
                       'Rapid_Breathing', 'Cardiovascular_Disease', 'Diabetes', 'Obesity',
                       'Elderly', 'Heat_Sensitive_Medications', 'Dehydration', 'Alcohol_Use',
                       'Previous_Heat_Illness', 'Poor_Heat_Acclimation', 'Prolonged_Exertion',
                       'High_Blood_Pressure', 'Smoking']
        
        # Filter columns that exist in the dataset
        available_features = [col for col in feature_cols if col in data.columns]
        self.feature_names = available_features
        
        X = data[available_features].fillna(0)
        y = data['Heatstroke'].astype(int)
        
        return X, y
    
    def train_model(self, X_train, y_train):
        """Train the Random Forest model"""
        print("Training Random Forest model...")
        
        # Initialize and train the model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        print("✅ Model trained successfully!")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        return feature_importance
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the model performance"""
        if not self.is_trained:
            print("❌ Model not trained yet!")
            return
        
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nModel Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return accuracy
    
    def predict_heatstroke(self, health_assessment_data, bpm_data=None):
        """
        Predict heatstroke risk using health assessment data and BPM data
        
        Args:
            health_assessment_data (dict): Data from dashboard health assessment
            bpm_data (dict): Data from newA.py analysis (optional)
        """
        if not self.is_trained:
            print("❌ Model not trained yet! Please train the model first.")
            return None
        
        # Create feature vector from health assessment data
        features = self._extract_features_from_assessment(health_assessment_data, bpm_data)
        
        # Scale features
        features_scaled = self.scaler.transform([features])
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        return {
            'heatstroke_prediction': bool(prediction),
            'heatstroke_probability': probability[1] if len(probability) > 1 else probability[0],
            'risk_level': self._get_risk_level(probability[1] if len(probability) > 1 else probability[0]),
            'features_used': self.feature_names,
            'feature_values': dict(zip(self.feature_names, features))
        }
    
    def _extract_features_from_assessment(self, health_data, bpm_data):
        """Extract features from health assessment data"""
        features = []
        
        for feature in self.feature_names:
            if feature == 'Age':
                features.append(health_data.get('age', 30))
            elif feature == 'Gender':
                gender = health_data.get('gender', 'male')
                if gender in self.label_encoders.get('Gender', {}).classes_:
                    features.append(self.label_encoders['Gender'].transform([gender])[0])
                else:
                    features.append(0)
            elif feature == 'Temperature':
                # Use skin temperature from BPM data if available
                if bpm_data and 'skin_temperature' in bpm_data:
                    features.append(bpm_data['skin_temperature'])
                else:
                    features.append(37.0)  # Default body temperature
            elif feature == 'Heart_Rate':
                # Use BPM from newA.py if available
                if bpm_data and 'bpm' in bpm_data:
                    features.append(bpm_data['bpm'])
                else:
                    features.append(72.0)  # Default heart rate
            elif feature == 'Blood_Pressure_Systolic':
                features.append(120.0)  # Default
            elif feature == 'Blood_Pressure_Diastolic':
                features.append(80.0)  # Default
            else:
                # Check if feature is in symptoms, medical_history, or risk_factors
                value = 0
                if feature.lower() in [s.lower() for s in health_data.get('symptoms', [])]:
                    value = 1
                elif feature.lower() in [s.lower() for s in health_data.get('medical_history', [])]:
                    value = 1
                elif feature.lower() in [s.lower() for s in health_data.get('risk_factors', [])]:
                    value = 1
                
                # Handle specific mappings
                if feature == 'High_Blood_Pressure' and 'high_blood_pressure' in health_data.get('risk_factors', []):
                    value = 1
                elif feature == 'Diabetes' and ('diabetes' in health_data.get('medical_history', []) or 
                                               'diabetes' in health_data.get('risk_factors', [])):
                    value = 1
                elif feature == 'Elderly' and health_data.get('age', 0) > 65:
                    value = 1
                
                features.append(value)
        
        return features
    
    def _get_risk_level(self, probability):
        """Convert probability to risk level"""
        if probability < 0.3:
            return "Low"
        elif probability < 0.7:
            return "Moderate"
        else:
            return "High"
    
    def analyze_current_data(self):
        """Analyze current health assessment data with BPM data"""
        print("🔍 Analyzing current health data for heatstroke prediction...")
        
        # Get BPM data from newA.py
        try:
            print("📊 Getting BPM data from newA.py...")
            bpm, confidence, infection_score, dehydration_score, arrhythmia_score = run_pulse_rate_from_csv("../BPM/A.csv", fs=10, plot_spectrogram=False)
            
            bpm_data = {
                'bpm': bpm,
                'confidence': confidence,
                'infection_score': infection_score,
                'dehydration_score': dehydration_score,
                'arrhythmia_score': arrhythmia_score,
                'skin_temperature': 34.6  # From the logs showing average skin temperature
            }
            
            print(f"✅ BPM Analysis Results:")
            print(f"   Heart Rate: {bpm:.1f} BPM")
            print(f"   Confidence: {confidence:.3f}")
            print(f"   Dehydration Score: {dehydration_score:.1f}")
            
        except Exception as e:
            print(f"⚠️ Could not get BPM data: {e}")
            bpm_data = None
        
        # Sample health assessment data (you can replace this with actual data from your dashboard)
        sample_health_data = {
            'age': 35,
            'gender': 'male',
            'symptoms': ['headache', 'fatigue', 'hot_skin'],
            'medical_history': ['dehydration'],
            'risk_factors': ['prolonged_exertion']
        }
        
        print(f"📋 Sample Health Assessment Data:")
        print(f"   Age: {sample_health_data['age']}")
        print(f"   Gender: {sample_health_data['gender']}")
        print(f"   Symptoms: {sample_health_data['symptoms']}")
        print(f"   Medical History: {sample_health_data['medical_history']}")
        print(f"   Risk Factors: {sample_health_data['risk_factors']}")
        
        # Make prediction
        prediction_result = self.predict_heatstroke(sample_health_data, bpm_data)
        
        if prediction_result:
            print(f"\n🎯 Heatstroke Prediction Results:")
            print(f"   Prediction: {'HEATSTROKE RISK' if prediction_result['heatstroke_prediction'] else 'No Heatstroke Risk'}")
            print(f"   Probability: {prediction_result['heatstroke_probability']:.3f}")
            print(f"   Risk Level: {prediction_result['risk_level']}")
            
            # Show key features that influenced the prediction
            print(f"\n🔍 Key Features:")
            for feature, value in prediction_result['feature_values'].items():
                if value != 0:  # Only show non-zero features
                    print(f"   {feature}: {value}")
        
        return prediction_result

def main():
    """Main function to run the heatstroke prediction"""
    print("🔥 Heatstroke Prediction using Health Assessment + BPM Data")
    print("=" * 60)
    
    # Initialize predictor
    predictor = HeatstrokePredictor()
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test = predictor.load_and_preprocess_data()
    
    # Train model
    feature_importance = predictor.train_model(X_train, y_train)
    
    # Evaluate model
    accuracy = predictor.evaluate_model(X_test, y_test)
    
    print("\n" + "=" * 60)
    print("🎯 PREDICTING ON CURRENT DATA")
    print("=" * 60)
    
    # Analyze current data
    prediction_result = predictor.analyze_current_data()
    
    print("\n" + "=" * 60)
    print("✅ Analysis Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main() 