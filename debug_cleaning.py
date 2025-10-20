#!/usr/bin/env python
"""
Debug script for data cleaning issues
"""

import pandas as pd
import numpy as np
import json
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing.data_cleaner import DataCleaner

def test_data_cleaner():
    """Test the DataCleaner class with sample data"""
    print("Testing DataCleaner class...")
    
    # Create sample data with various issues
    sample_data = [
        {"feature1": 1.5, "feature2": "A", "feature3": 10},
        {"feature1": None, "feature2": "B", "feature3": 20},
        {"feature1": 3.2, "feature2": None, "feature3": 30},
        {"feature1": 4.1, "feature2": "A", "feature3": None},
        {"feature1": 5.7, "feature2": "C", "feature3": 50},
        {"feature1": 100.0, "feature2": "A", "feature3": 25},  # Outlier
    ]
    
    print("Sample data:")
    for row in sample_data:
        print(f"  {row}")
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(sample_data)
        print(f"\nOriginal DataFrame shape: {df.shape}")
        print(f"Missing values:\n{df.isnull().sum()}")
        print(f"Data types:\n{df.dtypes}")
        
        # Apply cleaning
        cleaner = DataCleaner()
        
        # Identify column types
        cleaner.identify_column_types(df)
        print(f"\nColumn types:")
        print(f"  Numeric: {cleaner.numeric_columns}")
        print(f"  Categorical: {cleaner.categorical_columns}")
        
        # Handle missing values
        print("\nHandling missing values...")
        df = cleaner.handle_missing_values(df, 'auto')
        print(f"Missing values after handling:\n{df.isnull().sum()}")
        
        # Handle outliers
        print("\nHandling outliers...")
        df = cleaner.handle_outliers(df, 'iqr', 1.5, 'cap')
        print("Outliers handled")
        
        # Remove duplicates
        print("\nRemoving duplicates...")
        df = cleaner.remove_duplicates(df)
        print(f"DataFrame shape after deduplication: {df.shape}")
        
        # Encode categorical variables
        print("\nEncoding categorical variables...")
        df = cleaner.encode_categorical_variables(df, 'label')
        print("Categorical variables encoded")
        
        # Standardize features
        print("\nStandardizing features...")
        df = cleaner.standardize_numeric_features(df)
        print("Features standardized")
        
        # Convert back to records
        cleaned_data = df.to_dict('records')
        print(f"\nCleaned data (first 2 rows):")
        for i, row in enumerate(cleaned_data[:2]):
            print(f"  {row}")
            
        print("\nData cleaning completed successfully!")
        return True
        
    except Exception as e:
        print(f"\nError during data cleaning: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_api_payload():
    """Test with a payload similar to what the API receives"""
    print("\n" + "="*50)
    print("Testing API payload format...")
    
    # Sample data similar to what frontend sends
    data = [
        {"feature1": 1.5, "feature2": "A", "feature3": 10},
        {"feature1": None, "feature2": "B", "feature3": 20},
        {"feature1": 3.2, "feature2": None, "feature3": 30},
        {"feature1": 4.1, "feature2": "A", "feature3": None},
        {"feature1": 5.7, "feature2": "C", "feature3": 50},
    ]
    
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
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(data)
        print(f"Original DataFrame shape: {df.shape}")
        
        # Apply cleaning based on config
        cleaner = DataCleaner()
        
        # Identify column types
        cleaner.identify_column_types(df)
        
        # Handle missing values
        if config.get('handle_missing', True):
            strategy = config.get('missing_strategy', 'auto')
            df = cleaner.handle_missing_values(df, strategy)
        
        # Handle outliers
        if config.get('handle_outliers', True):
            method = config.get('outlier_method', 'iqr')
            threshold = config.get('outlier_threshold', 1.5)
            action = config.get('outlier_action', 'cap')
            df = cleaner.handle_outliers(df, method, threshold, action)
        
        # Remove duplicates
        if config.get('remove_duplicates', True):
            df = cleaner.remove_duplicates(df)
        
        # Encode categorical variables
        if config.get('encode_categorical', False):
            method = config.get('encoding_method', 'label')
            df = cleaner.encode_categorical_variables(df, method)
        
        # Standardize features
        if config.get('standardize_features', False):
            df = cleaner.standardize_numeric_features(df)
        
        # Convert back to JSON
        cleaned_data = df.to_dict('records')
        
        print(f"Cleaned data shape: {len(cleaned_data)} rows")
        print("Data cleaning via API simulation completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during API simulation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Debugging data cleaning issues...")
    print("="*50)
    
    success1 = test_data_cleaner()
    success2 = test_api_payload()
    
    if success1 and success2:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed. Check the error messages above.")