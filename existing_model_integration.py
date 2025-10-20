#!/usr/bin/env python
"""
Example of integrating an existing trained model with the MLOps monitoring system
This shows how to wrap your existing model to send predictions to the monitoring system
"""

import requests
import joblib
import pandas as pd
import numpy as np
import time

# Configuration
MONITORING_API_URL = "http://127.0.0.1:8000/api"
MODEL_ID = None

class ModelWithMonitoring:
    """
    Wrapper class for your existing ML model with monitoring integration
    Replace this with your actual model loading and prediction logic
    """
    
    def __init__(self, model_path):
        """Initialize with your existing model"""
        # Load your existing model
        # Replace this with your actual model loading code
        self.model = joblib.load(model_path)
        
        # If you have a scaler or other preprocessing components, load them too
        # self.scaler = joblib.load('path/to/scaler.pkl')
        
        print("✓ Your existing model loaded successfully")
    
    def predict(self, input_features):
        """
        Make a prediction using your existing model
        Replace this with your actual prediction logic
        """
        # Convert input to the format your model expects
        # This is just an example - adjust based on your model's requirements
        if isinstance(input_features, dict):
            # Convert dict to DataFrame
            X = pd.DataFrame([input_features])
        else:
            # Assume it's already in the right format
            X = input_features
        
        # Apply any preprocessing your model needs
        # X_scaled = self.scaler.transform(X)
        
        # Make prediction
        # Replace this with your actual model's prediction method
        prediction = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        # Format the result as needed by your application
        return {
            'prediction': prediction[0],
            'probability': float(probabilities[0].max()),
            'probabilities_all_classes': probabilities[0].tolist()
        }

def register_your_model():
    """Register your model with the monitoring system"""
    # Replace with your model's information
    model_data = {
        "name": "YourExistingModel",  # Replace with your model name
        "version": "v1.0",            # Replace with your model version
        "description": "Your existing model integrated with MLOps monitoring"
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

def send_to_monitoring(model_id, input_data, prediction_result, latency_ms):
    """Send prediction data to the monitoring system"""
    payload = {
        "model": model_id,
        "input_data": input_data,
        "prediction": {
            "class": str(prediction_result['prediction']),
            "probability": prediction_result['probability']
        },
        "latency_ms": round(latency_ms, 2)
    }
    
    try:
        response = requests.post(f"{MONITORING_API_URL}/predictions/", json=payload)
        if response.status_code == 201:
            print("✓ Prediction sent to monitoring system")
            return True
        else:
            print(f"✗ Failed to send prediction: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error sending prediction: {e}")
        return False

def predict_with_monitoring(wrapped_model, input_data, model_id):
    """
    Make a prediction and send it to the monitoring system
    This is the main function you would use in your application
    """
    start_time = time.time()
    
    # Make the actual prediction using your model
    prediction_result = wrapped_model.predict(input_data)
    
    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    
    # Send to monitoring system
    if model_id:
        send_to_monitoring(model_id, input_data, prediction_result, latency_ms)
    
    return prediction_result

def main():
    print("Integrating Your Existing Model with MLOps Monitoring")
    print("=" * 55)
    
    # Step 1: Wrap your existing model
    print("Step 1: Loading your existing model...")
    try:
        # Replace 'your_model.pkl' with the path to your actual model file
        wrapped_model = ModelWithMonitoring('your_model.pkl')
    except FileNotFoundError:
        print("Model file not found. Creating a simple example model...")
        # Create a simple example model for demonstration
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        
        # Create example data and model
        X, y = make_classification(n_samples=1000, n_features=5, n_classes=2, random_state=42)
        example_model = RandomForestClassifier(n_estimators=10, random_state=42)
        example_model.fit(X, y)
        
        # Save example model
        joblib.dump(example_model, 'example_model.pkl')
        
        # Load the example model
        wrapped_model = ModelWithMonitoring('example_model.pkl')
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Step 2: Register with monitoring system
    print("\nStep 2: Registering model with monitoring system...")
    model_id = register_your_model()
    
    if not model_id:
        print("Cannot proceed without model registration")
        return
    
    # Step 3: Make predictions with monitoring
    print("\nStep 3: Making predictions with monitoring...")
    
    # Example input data - replace with your actual input format
    sample_inputs = [
        {
            "feature1": 1.5,
            "feature2": 2.3,
            "feature3": 0.8,
            "feature4": 1.1,
            "feature5": 3.2
        },
        {
            "feature1": 2.1,
            "feature2": 1.7,
            "feature3": 1.9,
            "feature4": 0.5,
            "feature5": 2.8
        }
    ]
    
    for i, input_data in enumerate(sample_inputs, 1):
        print(f"\nProcessing input {i}...")
        print(f"  Input: {input_data}")
        
        # Make prediction with monitoring
        result = predict_with_monitoring(wrapped_model, input_data, model_id)
        print(f"  Prediction: {result['prediction']}")
        print(f"  Probability: {result['probability']:.4f}")
    
    print("\n" + "=" * 55)
    print("Integration complete!")
    print("Your existing model is now making predictions and they are being monitored.")
    print(f"View dashboard at: http://127.0.0.1:8000/models/{model_id}/")
    print("\nTo use this with your actual model:")
    print("1. Replace 'your_model.pkl' with your model file path")
    print("2. Modify the predict() method to match your model's interface")
    print("3. Update the model registration information")
    print("4. Adjust the input data format as needed")

if __name__ == "__main__":
    main()