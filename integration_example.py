#!/usr/bin/env python
"""
Example of how to integrate your ML model with the MLOps monitoring system
This shows how to modify your existing application to send predictions to the monitoring system
"""

import requests
import json
import joblib  # For loading scikit-learn models
import pandas as pd
import numpy as np
from datetime import datetime
import time

# Configuration for the monitoring system
MONITORING_API_URL = "http://127.0.0.1:8000/api"
MODEL_ID = None  # This will be set when we register the model

class CreditApprovalModel:
    """Example ML model class - replace with your actual model"""
    
    def __init__(self):
        # In a real scenario, you would load your trained model here
        # self.model = joblib.load('path/to/your/model.pkl')
        pass
    
    def predict(self, input_data):
        """
        Make a prediction using your model
        Replace this with your actual model's prediction logic
        """
        # This is just a mock prediction for demonstration
        # In reality, you would use your trained model:
        # prediction = self.model.predict(input_data)
        
        # Mock prediction logic based on simple rules
        age = input_data.get('age', 0)
        income = input_data.get('income', 0)
        credit_score = input_data.get('credit_score', 0)
        
        # Simple rule-based "model" for demonstration
        if credit_score > 700 and income > 40000 and age > 25:
            probability = min(0.95, 0.5 + (credit_score - 700) * 0.001 + (income - 40000) * 0.00001)
            decision = "approved"
        else:
            probability = max(0.05, 0.5 - (700 - credit_score) * 0.001 - (40000 - income) * 0.00001)
            decision = "denied"
        
        return {
            "decision": decision,
            "probability": round(probability, 4),
            "confidence": round(0.8 + np.random.random() * 0.2, 4)  # Random confidence for demo
        }

def register_model_with_monitoring():
    """Register your model with the monitoring system"""
    model_data = {
        "name": "MyCreditApprovalModel",
        "version": "v1.0",
        "description": "Custom credit approval model with monitoring"
    }
    
    try:
        response = requests.post(f"{MONITORING_API_URL}/models/", json=model_data)
        if response.status_code == 201:
            model_info = response.json()
            print(f"✓ Model registered with monitoring system")
            print(f"  Model ID: {model_info['id']}")
            print(f"  Model Name: {model_info['name']}")
            return model_info['id']
        else:
            print(f"✗ Failed to register model: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"✗ Error registering model: {e}")
        return None

def send_prediction_to_monitoring(model_id, input_data, prediction_result, latency_ms):
    """Send prediction data to the monitoring system"""
    payload = {
        "model": model_id,
        "input_data": input_data,
        "prediction": prediction_result,
        "latency_ms": latency_ms
    }
    
    try:
        response = requests.post(f"{MONITORING_API_URL}/predictions/", json=payload)
        if response.status_code == 201:
            print("✓ Prediction sent to monitoring system")
            return True
        else:
            print(f"✗ Failed to send prediction: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"✗ Error sending prediction: {e}")
        return False

def make_prediction_with_monitoring(model, input_data, model_id):
    """
    Make a prediction and send it to the monitoring system
    This is the main function you would call in your application
    """
    start_time = time.time()
    
    # Make the actual prediction using your model
    prediction_result = model.predict(input_data)
    
    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    
    # Send to monitoring system
    if model_id:
        send_prediction_to_monitoring(model_id, input_data, prediction_result, latency_ms)
    
    return prediction_result

def main():
    print("ML Model Integration with MLOps Monitoring System")
    print("=" * 50)
    
    # Step 1: Initialize your ML model
    print("Step 1: Loading ML model...")
    model = CreditApprovalModel()
    print("✓ Model loaded successfully")
    
    # Step 2: Register model with monitoring system
    print("\nStep 2: Registering model with monitoring system...")
    model_id = register_model_with_monitoring()
    
    if not model_id:
        print("Cannot proceed without model registration")
        return
    
    # Step 3: Make predictions with monitoring
    print("\nStep 3: Making predictions with monitoring...")
    
    # Example customer data
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
    
    for i, customer_data in enumerate(customers, 1):
        print(f"\nProcessing customer {i}...")
        print(f"  Input: {customer_data}")
        
        # Make prediction with monitoring
        result = make_prediction_with_monitoring(model, customer_data, model_id)
        print(f"  Prediction: {result}")
        
        # Simulate some delay between requests
        time.sleep(0.5)
    
    print("\n" + "=" * 50)
    print("Integration complete!")
    print(f"View your model dashboard at: http://127.0.0.1:8000/models/{model_id}/")
    print("You should see all predictions in the 'Recent Predictions' section")

if __name__ == "__main__":
    main()