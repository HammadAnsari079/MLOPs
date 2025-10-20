#!/usr/bin/env python
"""
Example script showing how to send predictions to the MLOps monitoring system
"""

import requests
import json
import uuid

# Configuration
BASE_URL = "http://127.0.0.1:8000/api"

def register_model():
    """Register a new model with the monitoring system"""
    model_data = {
        "name": "ExampleModel",
        "version": "v1.0",
        "description": "An example model for demonstration"
    }
    
    response = requests.post(f"{BASE_URL}/models/", json=model_data)
    if response.status_code == 201:
        model = response.json()
        print(f"Model registered successfully!")
        print(f"Model ID: {model['id']}")
        print(f"Model Name: {model['name']}")
        return model['id']
    else:
        print(f"Failed to register model: {response.status_code}")
        print(response.text)
        return None

def send_prediction(model_id, input_data, prediction_result, latency_ms=None):
    """Send a prediction to the monitoring system"""
    prediction_data = {
        "model": model_id,
        "input_data": input_data,
        "prediction": prediction_result
    }
    
    # Add latency if provided
    if latency_ms is not None:
        prediction_data["latency_ms"] = latency_ms
    
    response = requests.post(f"{BASE_URL}/predictions/", json=prediction_data)
    if response.status_code == 201:
        print("Prediction sent successfully!")
        return True
    else:
        print(f"Failed to send prediction: {response.status_code}")
        print(response.text)
        return False

def main():
    print("MLOps Monitoring System - Prediction Submission Example")
    print("=" * 55)
    
    # Step 1: Register a model (you would typically do this once)
    print("Step 1: Registering model...")
    model_id = register_model()
    
    if not model_id:
        print("Cannot proceed without a model ID")
        return
    
    print("\nStep 2: Sending sample predictions...")
    
    # Example 1: Simple classification prediction
    input_data_1 = {
        "age": 35,
        "income": 50000,
        "credit_score": 720,
        "loan_amount": 15000
    }
    
    prediction_result_1 = {
        "class": "approved",
        "probability": 0.85,
        "confidence": 0.92
    }
    
    print("\nSending prediction 1...")
    send_prediction(model_id, input_data_1, prediction_result_1, 45.5)
    
    # Example 2: Regression prediction
    input_data_2 = {
        "temperature": 25.5,
        "humidity": 60,
        "pressure": 1013.25,
        "wind_speed": 12.3
    }
    
    prediction_result_2 = {
        "predicted_value": 32.7,
        "confidence_interval": [30.5, 34.9]
    }
    
    print("\nSending prediction 2...")
    send_prediction(model_id, input_data_2, prediction_result_2, 32.1)
    
    # Example 3: NLP classification
    input_data_3 = {
        "text": "This product is amazing! I love it so much.",
        "text_length": 42,
        "exclamation_count": 2
    }
    
    prediction_result_3 = {
        "sentiment": "positive",
        "score": 0.95,
        "emotion": "joy"
    }
    
    print("\nSending prediction 3...")
    send_prediction(model_id, input_data_3, prediction_result_3, 67.8)
    
    print("\nStep 3: View your model dashboard")
    print(f"Open your browser and go to: http://127.0.0.1:8000/models/{model_id}/")
    print("You should see the predictions you just sent in the 'Recent Predictions' section")

if __name__ == "__main__":
    main()