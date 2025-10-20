#!/usr/bin/env python
"""
Example of integrating a real scikit-learn model with the MLOps monitoring system
This shows how to train a model, save it, and then use it with monitoring
"""

import requests
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import time
import uuid

# Configuration
MONITORING_API_URL = "http://127.0.0.1:8000/api"
MODEL_ID = None

def create_sample_data():
    """Create sample data for training a model"""
    # Generate synthetic credit approval data
    np.random.seed(42)
    n_samples = 1000
    
    # Features
    age = np.random.randint(18, 80, n_samples)
    income = np.random.normal(50000, 20000, n_samples)
    credit_score = np.random.normal(650, 100, n_samples)
    loan_amount = np.random.normal(15000, 10000, n_samples)
    employment_length = np.random.randint(0, 40, n_samples)
    
    # Ensure realistic ranges
    income = np.clip(income, 10000, 200000)
    credit_score = np.clip(credit_score, 300, 850)
    loan_amount = np.clip(loan_amount, 1000, 100000)
    
    # Create target variable based on features (simplified logic)
    approval_prob = (
        (age - 18) / 62 * 0.2 +
        np.clip((income - 10000) / 190000, 0, 1) * 0.3 +
        np.clip((credit_score - 300) / 550, 0, 1) * 0.4 +
        (40 - np.abs(employment_length - 20)) / 40 * 0.1
    )
    approved = np.random.binomial(1, approval_prob, n_samples)
    
    # Create DataFrame
    data = pd.DataFrame({
        'age': age,
        'income': income,
        'credit_score': credit_score,
        'loan_amount': loan_amount,
        'employment_length': employment_length,
        'approved': approved
    })
    
    return data

def train_and_save_model():
    """Train a model and save it to disk"""
    print("Training a new model...")
    
    # Create sample data
    data = create_sample_data()
    
    # Prepare features and target
    feature_columns = ['age', 'income', 'credit_score', 'loan_amount', 'employment_length']
    X = data[feature_columns]
    y = data['approved']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model trained with accuracy: {accuracy:.4f}")
    
    # Save model and scaler
    joblib.dump(model, 'credit_approval_model.pkl')
    joblib.dump(scaler, 'feature_scaler.pkl')
    print("Model and scaler saved to disk")
    
    return model, scaler

class MonitoredCreditModel:
    """Wrapper class for your ML model with monitoring integration"""
    
    def __init__(self, model_path='credit_approval_model.pkl', 
                 scaler_path='feature_scaler.pkl'):
        """Load the trained model and scaler"""
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.feature_columns = ['age', 'income', 'credit_score', 'loan_amount', 'employment_length']
            print("✓ Model and scaler loaded successfully")
        except FileNotFoundError:
            print("✗ Model files not found. Please train a model first.")
            raise
    
    def predict_single(self, input_data):
        """
        Make a single prediction
        input_data should be a dictionary with the required features
        """
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Select and order features
        X = input_df[self.feature_columns]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0].max()
        
        return {
            'approved': bool(prediction),
            'probability': float(probability),
            'confidence': float(probability)  # Using probability as confidence
        }

def register_model_with_monitoring():
    """Register the model with the monitoring system"""
    model_data = {
        "name": "ScikitLearnCreditModel",
        "version": "v1.0",
        "description": "Random Forest model for credit approval with MLOps monitoring"
    }
    
    try:
        response = requests.post(f"{MONITORING_API_URL}/models/", json=model_data)
        if response.status_code == 201:
            model_info = response.json()
            print(f"✓ Model registered with monitoring system")
            print(f"  Model ID: {model_info['id']}")
            return model_info['id']
        else:
            print(f"✗ Failed to register model: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"✗ Error registering model: {e}")
        return None

def send_prediction_to_monitoring(model_id, input_data, prediction_result, latency_ms):
    """Send prediction to monitoring system"""
    payload = {
        "model": model_id,
        "input_data": input_data,
        "prediction": prediction_result,
        "latency_ms": round(latency_ms, 2)
    }
    
    try:
        response = requests.post(f"{MONITORING_API_URL}/predictions/", json=payload)
        if response.status_code == 201:
            return True
        else:
            print(f"Warning: Failed to send prediction to monitoring: {response.status_code}")
            return False
    except Exception as e:
        print(f"Warning: Error sending prediction to monitoring: {e}")
        return False

def predict_with_monitoring(monitored_model, input_data, model_id):
    """
    Make a prediction and send it to the monitoring system
    This is what you would call in your production application
    """
    start_time = time.time()
    
    # Make prediction
    prediction_result = monitored_model.predict_single(input_data)
    
    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    
    # Send to monitoring if model is registered
    if model_id:
        send_prediction_to_monitoring(model_id, input_data, prediction_result, latency_ms)
    
    return prediction_result

def main():
    print("Scikit-learn Model Integration with MLOps Monitoring")
    print("=" * 55)
    
    # Step 1: Train and save model (do this once)
    try:
        train_and_save_model()
    except Exception as e:
        print(f"Error training model: {e}")
        return
    
    # Step 2: Load the trained model
    print("\nStep 1: Loading trained model...")
    try:
        monitored_model = MonitoredCreditModel()
    except Exception as e:
        print(f"Cannot load model: {e}")
        return
    
    # Step 3: Register with monitoring system
    print("\nStep 2: Registering model with monitoring system...")
    model_id = register_model_with_monitoring()
    
    if not model_id:
        print("Cannot proceed without model registration")
        return
    
    # Step 4: Make predictions with monitoring
    print("\nStep 3: Making predictions with monitoring...")
    
    # Sample customer applications
    customers = [
        {
            "age": 35,
            "income": 50000,
            "credit_score": 720,
            "loan_amount": 15000,
            "employment_length": 5
        },
        {
            "age": 28,
            "income": 35000,
            "credit_score": 650,
            "loan_amount": 25000,
            "employment_length": 2
        },
        {
            "age": 45,
            "income": 80000,
            "credit_score": 780,
            "loan_amount": 10000,
            "employment_length": 15
        }
    ]
    
    for i, customer in enumerate(customers, 1):
        print(f"\nProcessing customer {i}...")
        print(f"  Input: Age={customer['age']}, Income=${customer['income']}, "
              f"Credit Score={customer['credit_score']}")
        
        # Make prediction with monitoring
        result = predict_with_monitoring(monitored_model, customer, model_id)
        decision = "APPROVED" if result['approved'] else "DENIED"
        print(f"  Decision: {decision} (Probability: {result['probability']:.4f})")
    
    print("\n" + "=" * 55)
    print("Integration complete!")
    print("Your model is now making predictions and they are being monitored.")
    print(f"View dashboard at: http://127.0.0.1:8000/models/{model_id}/")

if __name__ == "__main__":
    main()