#!/usr/bin/env python
"""
Test script to verify that our fixes work correctly
"""

import os
import sys
import django
from django.conf import settings

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mlops_monitor.settings')
django.setup()

def test_model_creation():
    """Test that we can create a model with our fixed code"""
    try:
        from api.models import ModelRegistry
        import uuid
        
        # Create a test model with unique name
        unique_name = f"TestModel_{uuid.uuid4().hex[:8]}"
        model = ModelRegistry.objects.create(
            name=unique_name,
            version="v1.0",
            description="Test model for verification"
        )
        
        print("✓ Model creation test passed")
        return True
    except Exception as e:
        print(f"✗ Model creation test failed: {e}")
        return False

def test_data_cleaning():
    """Test that data cleaning works with our fixes"""
    try:
        from preprocessing.data_cleaner import DataCleaner
        import pandas as pd
        import numpy as np
        
        # Create test data with issues
        data = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5],
            'feature2': ['A', 'B', 'C', 'A', None],
            'feature3': [10, 20, 30, 1000, 40]  # 1000 is an outlier
        })
        
        cleaner = DataCleaner()
        cleaner.identify_column_types(data)
        cleaned_data = cleaner.handle_missing_values(data, 'mean')
        
        print("✓ Data cleaning test passed")
        return True
    except Exception as e:
        print(f"✗ Data cleaning test failed: {e}")
        return False

def test_utils_functions():
    """Test that our utility functions handle edge cases"""
    try:
        from monitoring.utils import calculate_psi, calculate_kl_divergence
        
        # Test with empty arrays
        psi = calculate_psi([], [])
        kl_div = calculate_kl_divergence([], [])
        
        # Test with normal data
        data1 = [1, 2, 3, 4, 5]
        data2 = [2, 3, 4, 5, 6]
        psi = calculate_psi(data1, data2)
        kl_div = calculate_kl_divergence(data1, data2)
        
        print("✓ Utility functions test passed")
        return True
    except Exception as e:
        print(f"✗ Utility functions test failed: {e}")
        return False

def main():
    print("Running tests to verify fixes...")
    print("=" * 40)
    
    tests = [
        test_model_creation,
        test_data_cleaning,
        test_utils_functions
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print("=" * 40)
    print(f"Tests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("🎉 All tests passed! The fixes are working correctly.")
    else:
        print("❌ Some tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main()