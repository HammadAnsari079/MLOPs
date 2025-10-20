#!/usr/bin/env python
"""
Test script for the data cleaning API endpoint
"""

import requests
import json
import pandas as pd

def test_cleaning_api():
    """Test the actual API endpoint"""
    print("Testing data cleaning API endpoint...")
    
    # Sample data with issues
    sample_data = [
        {"feature1": 1.5, "feature2": "A", "feature3": 10},
        {"feature1": None, "feature2": "B", "feature3": 20},
        {"feature1": 3.2, "feature2": None, "feature3": 30},
        {"feature1": 4.1, "feature2": "A", "feature3": None},
        {"feature1": 5.7, "feature2": "C", "feature3": 50},
        {"feature1": 100.0, "feature2": "A", "feature3": 25},  # Outlier
    ]
    
    # Configuration similar to frontend
    config = {
        "handle_missing": True,
        "missing_strategy": "auto",
        "handle_outliers": True,
        "outlier_method": "iqr",
        "outlier_threshold": 1.5,
        "outlier_action": "cap",
        "remove_duplicates": True,
        "encode_categorical": True,
        "standardize_features": True
    }
    
    # Payload
    payload = {
        "data": sample_data,
        "config": config
    }
    
    print("Sending request to API...")
    print(f"Data rows: {len(sample_data)}")
    print(f"Config: {json.dumps(config, indent=2)}")
    
    try:
        # Make the API request
        response = requests.post(
            "http://127.0.0.1:8000/api/clean-data/",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Response status code: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print("Success! Response:")
            print(json.dumps(result, indent=2))
            return True
        else:
            print(f"Error! Response content: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server. Is the Django server running?")
        return False
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_with_different_config():
    """Test with a simpler configuration"""
    print("\n" + "="*50)
    print("Testing with minimal configuration...")
    
    # Simple data
    sample_data = [
        {"feature1": 1.5, "feature2": "A"},
        {"feature1": None, "feature2": "B"},
        {"feature1": 3.2, "feature2": "A"},
    ]
    
    # Minimal config
    config = {
        "handle_missing": True,
        "missing_strategy": "mean"
    }
    
    payload = {
        "data": sample_data,
        "config": config
    }
    
    try:
        response = requests.post(
            "http://127.0.0.1:8000/api/clean-data/",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Response status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("Success! Response:")
            print(json.dumps(result, indent=2))
            return True
        else:
            print(f"Error! Response content: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server. Is the Django server running?")
        return False
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing data cleaning API endpoint...")
    print("="*50)
    
    success1 = test_cleaning_api()
    success2 = test_with_different_config()
    
    if success1 and success2:
        print("\nAll API tests passed!")
    else:
        print("\nSome API tests failed. Check the error messages above.")