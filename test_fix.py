#!/usr/bin/env python
"""
Test script to verify the data cleaning fix
"""

import requests
import json

def test_with_encode_categorical():
    """Test with encode_categorical=True"""
    print("Testing with encode_categorical=True...")
    
    # Sample data with categorical values
    sample_data = [
        {"feature1": 1.5, "category": "A", "value": 10},
        {"feature1": None, "category": "B", "value": 20},
        {"feature1": 3.2, "category": "A", "value": 30},
    ]
    
    # Configuration with encode_categorical=True
    config = {
        "handle_missing": True,
        "missing_strategy": "mean",
        "encode_categorical": True  # This was causing the issue
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
        
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("SUCCESS!")
            print(f"Rows processed: {result['rows_processed']}")
            print("First row of cleaned data:")
            print(json.dumps(result['cleaned_data'][0], indent=2))
            return True
        else:
            print(f"ERROR: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server. Is the Django server running?")
        return False
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_without_encode_categorical():
    """Test without encode_categorical"""
    print("\nTesting without encode_categorical...")
    
    # Sample data without categorical values
    sample_data = [
        {"feature1": 1.5, "feature2": 10},
        {"feature1": None, "feature2": 20},
        {"feature1": 3.2, "feature2": 30},
    ]
    
    # Configuration without encode_categorical
    config = {
        "handle_missing": True,
        "missing_strategy": "mean"
        # encode_categorical is not present or is False
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
        
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("SUCCESS!")
            print(f"Rows processed: {result['rows_processed']}")
            print("First row of cleaned data:")
            print(json.dumps(result['cleaned_data'][0], indent=2))
            return True
        else:
            print(f"ERROR: {response.text}")
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
    print("Testing data cleaning fix...")
    print("="*50)
    
    success1 = test_with_encode_categorical()
    success2 = test_without_encode_categorical()
    
    if success1 and success2:
        print("\nAll tests passed! The fix is working correctly.")
    else:
        print("\nSome tests failed. The fix may not be complete.")