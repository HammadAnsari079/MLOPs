#!/usr/bin/env python
"""
Demo script showing how to use the model training and data cleaning capabilities
"""

import requests
import json
import random
import pandas as pd
from sklearn.datasets import make_classification

def generate_sample_data():
    """Generate sample data for demonstration"""
    # Create a synthetic classification dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Convert to DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Add some categorical features and missing values to make it more realistic
    df['category'] = [random.choice(['A', 'B', 'C']) for _ in range(len(df))]
    df['region'] = [random.choice(['North', 'South', 'East', 'West']) for _ in range(len(df))]
    
    # Introduce some missing values
    for col in df.columns:
        if random.random() < 0.05:  # 5% missing values
            missing_indices = random.sample(range(len(df)), int(0.05 * len(df)))
            df.loc[missing_indices, col] = None
    
    return df

def demo_data_cleaning():
    """Demonstrate data cleaning capabilities"""
    print("=== Data Cleaning Demo ===")
    
    # Generate sample data
    df = generate_sample_data()
    print(f"Original data shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    
    # Clean the data using our API
    data_dict = df.to_dict('records')
    
    payload = {
        'data': data_dict,
        'config': {
            'handle_missing': True,
            'remove_duplicates': True,
            'handle_outliers': True,
            'encode_categorical': True,
            'scale_features': True
        }
    }
    
    try:
        response = requests.post('http://127.0.0.1:8000/api/clean-data/', json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"Cleaned data shape: {len(result['cleaned_data'])} rows")
            print(f"Data cleaning report: {result['quality_report']}")
            print("Data cleaning completed successfully!")
            return result['cleaned_data']
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error connecting to API: {e}")

def demo_model_training():
    """Demonstrate model training capabilities"""
    print("\n=== Model Training Demo ===")
    
    # Generate sample data
    df = generate_sample_data()
    
    # Prepare data for training
    data_dict = df.to_dict('records')
    
    payload = {
        'data': data_dict,
        'target_column': 'target',
        'model_name': 'FraudDetectionDemo',
        'algorithm': 'random_forest'
    }
    
    try:
        response = requests.post('http://127.0.0.1:8000/api/train/', json=payload)
        if response.status_code == 201:
            result = response.json()
            print(f"Model trained successfully!")
            print(f"Model ID: {result['model_id']}")
            print(f"Training metrics: {result['training_metrics']}")
            return result['model_id']
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error connecting to API: {e}")

def main():
    print("MLOps Platform - Training and Data Cleaning Demo")
    print("=" * 50)
    
    # Demo data cleaning
    cleaned_data = demo_data_cleaning()
    
    # Demo model training
    model_id = demo_model_training()
    
    print("\n=== Demo Completed ===")
    print("You can now use the trained model for predictions and monitoring!")

if __name__ == "__main__":
    main()