#!/usr/bin/env python
"""
Test script for the file download functionality
"""

import requests
import json
import pandas as pd

def test_file_download():
    """Test the file download API endpoint"""
    print("Testing file download API endpoint...")
    
    # Sample cleaned data
    sample_data = [
        {"feature1": 1.5, "category": "A", "value": 10},
        {"feature1": 2.3, "category": "B", "value": 20},
        {"feature1": 3.2, "category": "A", "value": 30},
    ]
    
    # Test CSV format
    print("\nTesting CSV format...")
    payload = {
        "data": sample_data,
        "format": "csv"
    }
    
    try:
        response = requests.post(
            "http://127.0.0.1:8000/api/download-cleaned-data/",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ CSV download prepared successfully")
            print(f"  Filename: {result['filename']}")
            print(f"  Content type: {result['content_type']}")
            return True
        else:
            print(f"✗ CSV download failed with status {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"✗ Error testing CSV download: {e}")
        return False

def test_excel_download():
    """Test Excel download"""
    print("\nTesting Excel format...")
    
    # Sample cleaned data
    sample_data = [
        {"feature1": 1.5, "category": "A", "value": 10},
        {"feature1": 2.3, "category": "B", "value": 20},
        {"feature1": 3.2, "category": "A", "value": 30},
    ]
    
    payload = {
        "data": sample_data,
        "format": "xlsx"
    }
    
    try:
        response = requests.post(
            "http://127.0.0.1:8000/api/download-cleaned-data/",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Excel download prepared successfully")
            print(f"  Filename: {result['filename']}")
            print(f"  Content type: {result['content_type']}")
            return True
        else:
            print(f"✗ Excel download failed with status {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"✗ Error testing Excel download: {e}")
        return False

if __name__ == "__main__":
    print("Testing file download functionality...")
    print("=" * 50)
    
    success1 = test_file_download()
    success2 = test_excel_download()
    
    if success1 and success2:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed. Check the error messages above.")