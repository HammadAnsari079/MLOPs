#!/usr/bin/env python
"""
Test script to upload sample data to the MLOps monitoring system
"""

import requests
import json
import random
from datetime import datetime

# API endpoint
BASE_URL = "http://127.0.0.1:8000/api"

def register_model():
    """Register a new model"""
    model_data = {
        "name": "CreditRiskModel",
        "version": "v2.1",
        "description": "Credit risk assessment model"
    }
    
    response = requests.post(f"{BASE_URL}/models/", json=model_data)
    if response.status_code == 201:
        model = response.json()
        print(f"Model registered successfully: {model['name']} (ID: {model['id']})")
        return model['id']
    else:
        print(f"Failed to register model: {response.status_code} - {response.text}")
        return None

def log_prediction(model_id):
    """Log a single prediction"""
    # Generate realistic sample data
    prediction_data = {
        "model": model_id,
        "input_data": {
            "loan_amount": round(random.uniform(1000, 100000), 2),
            "interest_rate": round(random.uniform(2.5, 15.0), 2),
            "borrower_age": random.randint(18, 80),
            "credit_score": random.randint(300, 850),
            "employment_length": random.randint(0, 40),
            "debt_to_income": round(random.uniform(0.1, 0.8), 2)
        },
        "prediction": {
            "probability": round(random.uniform(0.01, 0.99), 4),
            "confidence": round(random.uniform(0.5, 1.0), 4),
            "risk_class": random.choice(["low", "medium", "high"])
        },
        "latency_ms": round(random.uniform(10, 200), 2)
    }
    
    response = requests.post(f"{BASE_URL}/predictions/", json=prediction_data)
    if response.status_code == 201:
        print(f"Prediction logged successfully")
        return True
    else:
        print(f"Failed to log prediction: {response.status_code} - {response.text}")
        return False

def log_multiple_predictions(model_id, count=10):
    """Log multiple predictions"""
    success_count = 0
    for i in range(count):
        if log_prediction(model_id):
            success_count += 1
        # Add a small delay to simulate real-time predictions
        # import time
        # time.sleep(0.1)
    
    print(f"Logged {success_count}/{count} predictions successfully")

def main():
    print("MLOps Monitoring System - Data Upload Test")
    print("=" * 50)
    
    # Register a model
    print("1. Registering model...")
    model_id = register_model()
    
    if model_id:
        print("\n2. Logging predictions...")
        log_multiple_predictions(model_id, 20)
        
        print("\n3. Test completed!")
        print(f"View your model dashboard at: http://127.0.0.1:8000/models/{model_id}/")
    else:
        print("Failed to register model. Exiting.")

if __name__ == "__main__":
    main()