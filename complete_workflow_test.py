#!/usr/bin/env python
"""
Complete workflow test for the MLOps monitoring system
This script demonstrates the entire process from model registration to prediction submission
"""

import requests
import json
import time

# Configuration
BASE_URL = "http://127.0.0.1:8000/api"

def test_api_connection():
    """Test if the API is accessible"""
    try:
        response = requests.get(f"{BASE_URL}/test/")
        if response.status_code == 200:
            print("✓ API is accessible")
            return True
        else:
            print(f"✗ API connection failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ API connection failed: {e}")
        return False

def register_model():
    """Register a new model"""
    model_data = {
        "name": "CreditApprovalModel",
        "version": "v1.0",
        "description": "Model for credit approval decisions"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/models/", json=model_data)
        if response.status_code == 201:
            model = response.json()
            print(f"✓ Model registered successfully: {model['name']} (ID: {model['id']})")
            return model['id']
        else:
            print(f"✗ Failed to register model: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"✗ Error registering model: {e}")
        return None

def send_predictions(model_id):
    """Send sample predictions to the monitoring system"""
    predictions = [
        {
            "input_data": {
                "age": 35,
                "income": 50000,
                "credit_score": 720,
                "loan_amount": 15000,
                "employment_length": 5
            },
            "prediction": {
                "decision": "approved",
                "probability": 0.85,
                "confidence": 0.92
            },
            "latency_ms": 45.5
        },
        {
            "input_data": {
                "age": 28,
                "income": 35000,
                "credit_score": 650,
                "loan_amount": 25000,
                "employment_length": 2
            },
            "prediction": {
                "decision": "denied",
                "probability": 0.32,
                "confidence": 0.87
            },
            "latency_ms": 38.2
        },
        {
            "input_data": {
                "age": 45,
                "income": 80000,
                "credit_score": 780,
                "loan_amount": 10000,
                "employment_length": 15
            },
            "prediction": {
                "decision": "approved",
                "probability": 0.95,
                "confidence": 0.98
            },
            "latency_ms": 52.1
        }
    ]
    
    success_count = 0
    for i, pred_data in enumerate(predictions, 1):
        try:
            payload = {
                "model": model_id,
                "input_data": pred_data["input_data"],
                "prediction": pred_data["prediction"],
                "latency_ms": pred_data["latency_ms"]
            }
            
            response = requests.post(f"{BASE_URL}/predictions/", json=payload)
            if response.status_code == 201:
                print(f"✓ Prediction {i} sent successfully")
                success_count += 1
            else:
                print(f"✗ Failed to send prediction {i}: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"✗ Error sending prediction {i}: {e}")
        
        # Small delay between requests
        time.sleep(0.5)
    
    print(f"Successfully sent {success_count}/{len(predictions)} predictions")
    return success_count == len(predictions)

def get_model_health(model_id):
    """Check the health of the registered model"""
    try:
        response = requests.get(f"{BASE_URL}/models/{model_id}/health/")
        if response.status_code == 200:
            health_data = response.json()
            print(f"✓ Model health check successful")
            print(f"  Health Score: {health_data['health_score']}/100")
            print(f"  Status: {health_data['status']}")
            print(f"  Total Predictions: {health_data['total_predictions']}")
            return True
        else:
            print(f"✗ Failed to get model health: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"✗ Error checking model health: {e}")
        return False

def main():
    print("MLOps Monitoring System - Complete Workflow Test")
    print("=" * 50)
    
    # Step 1: Test API connection
    print("Step 1: Testing API connection...")
    if not test_api_connection():
        return
    
    # Step 2: Register a model
    print("\nStep 2: Registering model...")
    model_id = register_model()
    if not model_id:
        return
    
    # Step 3: Send predictions
    print("\nStep 3: Sending sample predictions...")
    if not send_predictions(model_id):
        print("Warning: Some predictions failed to send")
    
    # Step 4: Check model health
    print("\nStep 4: Checking model health...")
    time.sleep(2)  # Wait a moment for processing
    get_model_health(model_id)
    
    # Step 5: Provide instructions for viewing the dashboard
    print("\nStep 5: View results in the dashboard")
    print(f"Open your browser and navigate to: http://127.0.0.1:8000/models/{model_id}/")
    print("You should see:")
    print("  - The model listed in the dashboard")
    print("  - The predictions you sent in the 'Recent Predictions' section")
    print("  - Updated health metrics")
    
    print("\nComplete! The MLOps monitoring system is working correctly.")

if __name__ == "__main__":
    main()