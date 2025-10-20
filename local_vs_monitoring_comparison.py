#!/usr/bin/env python
"""
Comparison between local metrics calculation and MLOps monitoring
This shows why you need the monitoring system even when you can calculate metrics locally
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import requests
import time

# Local metrics calculation (what you can do without monitoring)
def calculate_local_metrics():
    """Calculate metrics locally - what you can do without monitoring"""
    print("LOCAL METRICS CALCULATION")
    print("=" * 30)
    
    # Simulate some predictions
    np.random.seed(42)
    actual = np.random.normal(300000, 100000, 100)  # Actual house prices
    predicted = actual + np.random.normal(0, 30000, 100)  # Predictions with some error
    
    # Calculate metrics
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    
    print(f"Local Metrics (based on 100 predictions):")
    print(f"  MAE: ${mae:,.2f}")
    print(f"  RMSE: ${rmse:,.2f}")
    print(f"  R²: {r2:.4f}")
    
    # Try to create a simple plot if matplotlib is available
    try:
        import importlib
        matplotlib_spec = importlib.util.find_spec("matplotlib")
        if matplotlib_spec is not None:
            import matplotlib.pyplot as plt
            
            # Create a simple plot
            plt.figure(figsize=(10, 4))
            
            plt.subplot(1, 2, 1)
            plt.scatter(actual, predicted, alpha=0.6)
            plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--')
            plt.xlabel('Actual Price')
            plt.ylabel('Predicted Price')
            plt.title('Actual vs Predicted (Local)')
            
            plt.subplot(1, 2, 2)
            errors = actual - predicted
            plt.hist(errors, bins=20, alpha=0.7)
            plt.xlabel('Prediction Error')
            plt.ylabel('Frequency')
            plt.title('Error Distribution (Local)')
            
            plt.tight_layout()
            plt.savefig('local_metrics.png')
            plt.close()
            
            print("\nLocal metrics plot saved as 'local_metrics.png'")
        else:
            print("\nMatplotlib not available. Install it with: pip install matplotlib")
            print("Skipping plot generation.")
    except Exception as e:
        print(f"\nCould not generate plots: {e}")
        print("Skipping plot generation.")
    
    return {"mae": mae, "rmse": rmse, "r2": r2}

# What the monitoring system provides (why you need it)
def explain_monitoring_benefits():
    """Explain what the monitoring system provides beyond local metrics"""
    print("\n\nMLOPS MONITORING SYSTEM BENEFITS")
    print("=" * 40)
    
    benefits = [
        {
            "title": "Real-time Monitoring",
            "description": "Tracks model performance continuously as new predictions are made, not just on static test sets."
        },
        {
            "title": "Data Drift Detection",
            "description": "Alerts you when input data changes significantly from training data (e.g., market shifts)."
        },
        {
            "title": "Concept Drift Detection",
            "description": "Detects when the relationship between inputs and outputs changes over time."
        },
        {
            "title": "Historical Trend Analysis",
            "description": "Shows how model performance changes over weeks/months, not just a single evaluation."
        },
        {
            "title": "Automated Alerts",
            "description": "Sends notifications when performance drops below thresholds, so you don't have to check manually."
        },
        {
            "title": "Multi-dimensional Analysis",
            "description": "Breaks down performance by different segments (e.g., house types, price ranges)."
        },
        {
            "title": "Data Quality Monitoring",
            "description": "Detects missing values, outliers, and schema violations in real-time."
        },
        {
            "title": "Collaboration Dashboard",
            "description": "Provides a web interface for teams to monitor model health together."
        }
    ]
    
    for i, benefit in enumerate(benefits, 1):
        print(f"{i}. {benefit['title']}")
        print(f"   {benefit['description']}")
        print()

# Simulate what happens without monitoring
def simulate_problems_without_monitoring():
    """Show what problems can occur without monitoring"""
    print("\n\nPROBLEMS WITHOUT MONITORING")
    print("=" * 30)
    
    scenarios = [
        {
            "scenario": "Market Shift",
            "problem": "Housing market crashes, but your model still predicts high prices",
            "impact": "Business loses money on bad recommendations",
            "detection": "Without monitoring: You never notice. With monitoring: Immediate alert."
        },
        {
            "scenario": "Data Quality Issues",
            "problem": "New data source has missing location scores",
            "impact": "Model accuracy degrades significantly",
            "detection": "Without monitoring: Silent failures. With monitoring: Data quality alerts."
        },
        {
            "scenario": "Feature Changes",
            "problem": "Real estate agents start using new features not in training data",
            "impact": "Model becomes less relevant",
            "detection": "Without monitoring: Gradual performance decay. With monitoring: Drift detection."
        },
        {
            "scenario": "Seasonal Patterns",
            "problem": "Model works well in summer but fails in winter",
            "impact": "Seasonal performance issues",
            "detection": "Without monitoring: Annual surprise. With monitoring: Seasonal trend analysis."
        }
    ]
    
    for scenario in scenarios:
        print(f"Scenario: {scenario['scenario']}")
        print(f"  Problem: {scenario['problem']}")
        print(f"  Impact: {scenario['impact']}")
        print(f"  Detection: {scenario['detection']}")
        print()

# Show how to use the monitoring system
def show_monitoring_usage():
    """Show how to use the monitoring system"""
    print("\n\nHOW TO USE MLOPS MONITORING")
    print("=" * 30)
    
    steps = [
        "1. Register your model once with the monitoring system",
        "2. Send each prediction to the monitoring API",
        "3. View real-time metrics in the web dashboard",
        "4. Receive alerts when issues are detected",
        "5. Analyze historical trends and patterns"
    ]
    
    for step in steps:
        print(step)
    
    print("\nExample API calls:")
    print("""
# Register model
POST /api/models/
{
  "name": "HousePriceModel",
  "version": "v1.0",
  "description": "My house price prediction model"
}

# Send prediction
POST /api/predictions/
{
  "model": "MODEL_UUID",
  "input_data": {"size": 2000, "bedrooms": 3},
  "prediction": {"price": 350000},
  "actual_value": {"price": 340000},  # Optional, for supervised feedback
  "latency_ms": 45
}
    """)

def main():
    print("LOCAL METRICS vs MLOPS MONITORING")
    print("=" * 50)
    
    print("You're right that you can calculate metrics like F1 score,")
    print("precision, recall, and plot graphs locally. But there's more...")
    
    # Calculate local metrics
    local_metrics = calculate_local_metrics()
    
    # Explain monitoring benefits
    explain_monitoring_benefits()
    
    # Show problems without monitoring
    simulate_problems_without_monitoring()
    
    # Show how to use monitoring
    show_monitoring_usage()
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print("Local metrics are good for initial evaluation, but:")
    print("✓ Monitoring provides continuous oversight")
    print("✓ Monitoring detects problems you can't see locally")
    print("✓ Monitoring alerts you automatically")
    print("✓ Monitoring shows trends over time")
    print("✓ Monitoring helps collaborate with your team")
    print("\nThe monitoring system doesn't replace local evaluation -")
    print("it complements it by providing production-level oversight!")

if __name__ == "__main__":
    main()