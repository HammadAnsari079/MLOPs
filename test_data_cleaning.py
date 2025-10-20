#!/usr/bin/env python
"""
Test script for the data cleaning API
"""

import requests
import json

# API endpoint
BASE_URL = "http://127.0.0.1:8000/api"

def test_data_cleaning():
    """Test the data cleaning API endpoint"""
    print("Testing Data Cleaning API")
    print("=" * 30)
    
    # Sample data with issues
    sample_data = [
        {
            "size": 2000,
            "bedrooms": 3,
            "bathrooms": 2,
            "age": 10,
            "location_score": 8.5,
            "price": None  # Missing value
        },
        {
            "size": 2500,
            "bedrooms": None,  # Missing value
            "bathrooms": 3,
            "age": 5,
            "location_score": 9.2,
            "price": 350000
        },
        {
            "size": 1800,
            "bedrooms": 4,
            "bathrooms": 2,
            "age": 15,
            "location_score": 7.8,
            "price": 280000
        },
        {
            "size": 3000,
            "bedrooms": 5,
            "bathrooms": 4,
            "age": 2,
            "location_score": 9.8,
            "price": 500000  # Outlier
        }
    ]
    
    # Cleaning configuration
    config = {
        "handle_missing": True,
        "missing_strategy": "auto",
        "handle_outliers": True,
        "outlier_method": "iqr",
        "outlier_threshold": 1.5,
        "outlier_action": "cap",
        "remove_duplicates": True,
        "encode_categorical": False
    }
    
    # Send request to API
    payload = {
        "data": sample_data,
        "config": config
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/clean-data/",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Data cleaning successful!")
            print(f"  Rows processed: {result['rows_processed']}")
            print("\nCleaned data:")
            print(json.dumps(result['cleaned_data'], indent=2))
            return True
        else:
            print(f"✗ Data cleaning failed with status {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"✗ Error testing data cleaning: {e}")
        return False

def test_house_price_prediction():
    """Test house price prediction with the monitoring system"""
    print("\nTesting House Price Prediction")
    print("=" * 30)
    
    # Register a model (if not already registered)
    model_data = {
        "name": "TestHousePriceModel",
        "version": "v1.0",
        "description": "Test model for house price prediction"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/models/",
            json=model_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 201:
            model_info = response.json()
            model_id = model_info['id']
            print(f"✓ Model registered: {model_info['name']} (ID: {model_id})")
        else:
            print("✗ Failed to register model")
            return False
    except Exception as e:
        print(f"✗ Error registering model: {e}")
        return False
    
    # Send a prediction
    prediction_data = {
        "model": model_id,
        "input_data": {
            "size": 2200,
            "bedrooms": 4,
            "bathrooms": 3,
            "age": 8,
            "location_score": 8.7
        },
        "prediction": {
            "predicted_price": 320000
        },
        "latency_ms": 45.5
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predictions/",
            json=prediction_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 201:
            print("✓ Prediction submitted successfully!")
            return True
        else:
            print(f"✗ Failed to submit prediction: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"✗ Error submitting prediction: {e}")
        return False

def main():
    print("MLOps Data Cleaning and Prediction Test")
    print("=" * 40)
    
    # Test data cleaning
    cleaning_success = test_data_cleaning()
    
    # Test prediction
    prediction_success = test_house_price_prediction()
    
    print("\n" + "=" * 40)
    print("TEST RESULTS")
    print("=" * 40)
    print(f"Data Cleaning: {'PASS' if cleaning_success else 'FAIL'}")
    print(f"Prediction Submission: {'PASS' if prediction_success else 'FAIL'}")
    
    if cleaning_success and prediction_success:
        print("\n🎉 All tests passed! The system is working correctly.")
        print("You can now:")
        print("1. Use the data cleaning API to preprocess your datasets")
        print("2. Submit predictions to the monitoring system")
        print("3. View results in the dashboard")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main()