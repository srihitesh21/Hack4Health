import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """Load and preprocess the heatstroke datasets"""
    print("Loading datasets...")
    
    # Load training and test datasets
    train_df = pd.read_csv('health_heatstroke_train_2000_temp (1).csv')
    test_df = pd.read_csv('health_heatstroke_test_2000_temp (1).csv')
    
    print(f"Training dataset shape: {train_df.shape}")
    print(f"Test dataset shape: {test_df.shape}")
    
    # Combine datasets for preprocessing
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Separate features and target
    X = combined_df.drop('heat stroke', axis=1)
    y = combined_df['heat stroke']
    
    # Handle categorical variables
    categorical_columns = ['gender', 'Infection Risk', 'Dehydration Risk', 'Arrhythmia Risk', 
                          'Stress Level', 'Fatigue Analysis']
    
    # Create label encoders for categorical variables
    label_encoders = {}
    for col in categorical_columns:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le
    
    # Handle binary categorical variables (Yes/No columns)
    binary_columns = ['Headache', 'Dizziness', 'Nausea', 'Vomiting', 'Muscle Cramps', 
                     'Weakness', 'Fatigue', 'Hot Skin', 'Dry Skin', 'Rapid Breathing',
                     'Cardiovascular Disease', 'Diabetes', 'Obesity', 'Elderly (>65 years)',
                     'Heat-Sensitive Medications', 'Dehydration', 'Alcohol Use', 
                     'Previous Heat Illness', 'Poor Heat Acclimation', 'Prolonged Exertion',
                     'High Blood Pressure', 'Smoking', 'Family History', 'Physical Inactivity',
                     'High Cholesterol', 'Atrial Fibrillation']
    
    for col in binary_columns:
        if col in X.columns:
            X[col] = (X[col] == 'Yes').astype(int)
    
    # Split back into train and test
    train_size = len(train_df)
    X_train = X[:train_size]
    X_test = X[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]
    
    print(f"Preprocessed training features shape: {X_train.shape}")
    print(f"Preprocessed test features shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, label_encoders

def train_random_forest(X_train, y_train):
    """Train Random Forest model with hyperparameter tuning"""
    print("\nTraining Random Forest model...")
    
    # Initialize Random Forest with optimized parameters
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    # Train the model
    rf_model.fit(X_train, y_train)
    
    print("Model training completed!")
    return rf_model

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Evaluate the Random Forest model"""
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    # Training accuracy
    train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_pred)
    print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    
    # Test accuracy
    test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_pred)
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Detailed classification report
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, test_pred, target_names=['No Heat Stroke', 'Heat Stroke']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, test_pred)
    print("\nConfusion Matrix (Test Set):")
    print(cm)
    
    return test_accuracy, test_pred

def analyze_feature_importance(model, X_train):
    """Analyze and visualize feature importance"""
    print("\n" + "="*50)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*50)
    
    # Get feature importance
    feature_importance = model.feature_importances_
    feature_names = X_train.columns
    
    # Create DataFrame for better visualization
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    print("Top 15 Most Important Features:")
    print(importance_df.head(15))
    
    # Visualize feature importance
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(15)
    plt.barh(range(len(top_features)), top_features['Importance'])
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 15 Feature Importance for Heat Stroke Prediction')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the plot to avoid display issues
    
    return importance_df

def analyze_predictions(y_test, test_pred, X_test):
    """Analyze prediction patterns"""
    print("\n" + "="*50)
    print("PREDICTION ANALYSIS")
    print("="*50)
    
    # Create results DataFrame
    results_df = X_test.copy()
    results_df['Actual'] = y_test.values
    results_df['Predicted'] = test_pred
    results_df['Correct'] = (y_test.values == test_pred)
    
    # Analyze correct vs incorrect predictions
    correct_predictions = results_df[results_df['Correct'] == True]
    incorrect_predictions = results_df[results_df['Correct'] == False]
    
    print(f"Correct predictions: {len(correct_predictions)} ({len(correct_predictions)/len(results_df)*100:.2f}%)")
    print(f"Incorrect predictions: {len(incorrect_predictions)} ({len(incorrect_predictions)/len(results_df)*100:.2f}%)")
    
    # Analyze false positives and false negatives
    false_positives = results_df[(results_df['Actual'] == 0) & (results_df['Predicted'] == 1)]
    false_negatives = results_df[(results_df['Actual'] == 1) & (results_df['Predicted'] == 0)]
    
    print(f"False Positives (predicted heat stroke but no heat stroke): {len(false_positives)}")
    print(f"False Negatives (missed heat stroke): {len(false_negatives)}")
    
    return results_df

def main():
    """Main function to run the complete analysis"""
    print("HEAT STROKE PREDICTION USING RANDOM FOREST")
    print("="*60)
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, label_encoders = load_and_preprocess_data()
    
    # Train model
    rf_model = train_random_forest(X_train, y_train)
    
    # Evaluate model
    test_accuracy, test_pred = evaluate_model(rf_model, X_train, X_test, y_train, y_test)
    
    # Analyze feature importance
    importance_df = analyze_feature_importance(rf_model, X_train)
    
    # Analyze predictions
    results_df = analyze_predictions(y_test, test_pred, X_test)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Model successfully trained on {len(X_train)} samples")
    print(f"Model tested on {len(X_test)} samples")
    print("Feature importance analysis saved to 'feature_importance.png'")
    
    return rf_model, test_accuracy, importance_df

if __name__ == "__main__":
    # Run the complete analysis
    model, accuracy, importance = main() 