#!/usr/bin/env python
"""
Example of integrating a house price prediction model with the MLOps monitoring system
This shows why monitoring is important even for models that seem to work well
"""

import requests
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import warnings
warnings.filterwarnings('ignore')

# Configuration
MONITORING_API_URL = "http://127.0.0.1:8000/api"
MODEL_ID = None

def create_house_price_data():
    """Create synthetic house price data for training"""
    np.random.seed(42)
    n_samples = 1000
    
    # Features that affect house prices
    size = np.random.normal(2000, 500, n_samples)  # Square feet
    bedrooms = np.random.randint(1, 6, n_samples)
    bathrooms = np.random.randint(1, 4, n_samples)
    age = np.random.randint(0, 50, n_samples)  # Years old
    location_score = np.random.uniform(1, 10, n_samples)  # Location quality (1-10)
    
    # Ensure realistic ranges
    size = np.clip(size, 500, 5000)
    
    # Create price based on features (simplified formula)
    base_price = (
        size * 100 +  # $100 per square foot
        bedrooms * 5000 +  # $5000 per bedroom
        bathrooms * 7000 +  # $7000 per bathroom
        (50 - age) * 1000 +  # Newer houses worth more
        location_score * 10000  # Location premium
    )
    
    # Add some noise
    noise = np.random.normal(0, 20000, n_samples)
    price = base_price + noise
    
    # Ensure positive prices
    price = np.clip(price, 50000, 1000000)
    
    # Create DataFrame
    data = pd.DataFrame({
        'size': size,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'age': age,
        'location_score': location_score,
        'price': price
    })
    
    return data

def train_house_price_model():
    """Train a house price prediction model"""
    print("Training house price prediction model...")
    
    # Create sample data
    data = create_house_price_data()
    
    # Prepare features and target
    feature_columns = ['size', 'bedrooms', 'bathrooms', 'age', 'location_score']
    X = data[feature_columns]
    y = data['price']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"  MAE: ${mae:,.2f}")
    print(f"  RMSE: ${rmse:,.2f}")
    print(f"  R²: {r2:.4f}")
    
    # Save model
    joblib.dump(model, 'house_price_model.pkl')
    print("Model saved to 'house_price_model.pkl'")
    
    # Also save test data for later use
    test_data = pd.concat([X_test, pd.Series(y_test, name='actual_price')], axis=1)
    test_data.to_csv('test_data.csv', index=False)
    print("Test data saved to 'test_data.csv'")
    
    return model, X_test, y_test

def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Additional metrics
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # Mean Absolute Percentage Error
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'mape': mape
    }

def plot_performance_comparison(y_true, y_pred):
    """Create a simple performance comparison plot"""
    try:
        import matplotlib.pyplot as plt
        
        # Create scatter plot
        plt.figure(figsize=(10, 6))
        
        # Actual vs Predicted
        plt.subplot(1, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title('Actual vs Predicted Prices')
        
        # Residuals
        plt.subplot(1, 2, 2)
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Price')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        
        plt.tight_layout()
        plt.savefig('house_price_performance.png')
        plt.close()
        
        print("Performance plots saved to 'house_price_performance.png'")
        return True
    except ImportError:
        print("Matplotlib not available. Install it with: pip install matplotlib")
        return False

class HousePriceModelWithMonitoring:
    """House price prediction model with MLOps monitoring"""
    
    def __init__(self, model_path='house_price_model.pkl'):
        """Load the trained model"""
        try:
            self.model = joblib.load(model_path)
            self.feature_columns = ['size', 'bedrooms', 'bathrooms', 'age', 'location_score']
            print("✓ House price model loaded successfully")
        except FileNotFoundError:
            print("✗ Model file not found. Please train a model first.")
            raise
    
    def predict_price(self, house_features):
        """
        Predict house price and send to monitoring
        house_features: dict with keys size, bedrooms, bathrooms, age, location_score
        """
        # Convert to DataFrame
        input_df = pd.DataFrame([house_features])
        
        # Select and order features
        X = input_df[self.feature_columns]
        
        # Make prediction
        predicted_price = self.model.predict(X)[0]
        
        return {
            'predicted_price': float(predicted_price),
            'features': house_features
        }

def register_model_with_monitoring():
    """Register the house price model with the monitoring system"""
    model_data = {
        "name": "HousePricePredictor",
        "version": "v1.0",
        "description": "Random Forest model for house price prediction with MLOps monitoring"
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

def send_prediction_to_monitoring(model_id, input_data, prediction_result, actual_price=None, latency_ms=None):
    """Send prediction to monitoring system"""
    # Prepare prediction data for monitoring
    prediction_payload = {
        "model": model_id,
        "input_data": input_data,
        "prediction": {
            "predicted_price": prediction_result['predicted_price']
        }
    }
    
    # Add latency if provided
    if latency_ms is not None:
        prediction_payload["latency_ms"] = round(latency_ms, 2)
    
    # Add actual value if provided (for supervised learning feedback)
    if actual_price is not None:
        prediction_payload["actual_value"] = {"actual_price": actual_price}
    
    try:
        response = requests.post(f"{MONITORING_API_URL}/predictions/", json=prediction_payload)
        if response.status_code == 201:
            return True
        else:
            print(f"Warning: Failed to send prediction to monitoring: {response.status_code}")
            return False
    except Exception as e:
        print(f"Warning: Error sending prediction to monitoring: {e}")
        return False

def predict_house_price_with_monitoring(model_wrapper, house_features, model_id, actual_price=None):
    """
    Predict house price and send to monitoring system
    """
    start_time = time.time()
    
    # Make prediction
    prediction_result = model_wrapper.predict_price(house_features)
    
    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    
    # Send to monitoring if model is registered
    if model_id:
        send_prediction_to_monitoring(model_id, house_features, prediction_result, actual_price, latency_ms)
    
    return prediction_result

def demonstrate_model_drift():
    """Demonstrate why monitoring is important by simulating model drift"""
    print("\n" + "="*60)
    print("DEMONSTRATING WHY MONITORING IS IMPORTANT")
    print("="*60)
    
    print("\nScenario: Market conditions change over time")
    print("Your model was trained on historical data, but now:")
    print("  - Inflation has increased house prices by 20%")
    print("  - Remote work has increased demand for suburban homes")
    print("  - Interest rates have changed buyer behavior")
    
    print("\nWithout monitoring, you wouldn't know that your model's predictions")
    print("are becoming less accurate over time. The MLOps system detects this!")

def main():
    print("House Price Prediction Model with MLOps Monitoring")
    print("=" * 55)
    
    # Step 1: Train model (do this once)
    try:
        model, X_test, y_test = train_house_price_model()
    except Exception as e:
        print(f"Error training model: {e}")
        return
    
    # Step 2: Calculate and display initial metrics
    print("\nStep 1: Evaluating initial model performance...")
    y_pred_initial = model.predict(X_test)
    initial_metrics = calculate_metrics(y_test, y_pred_initial)
    
    print("Initial Model Performance:")
    print(f"  MAE: ${initial_metrics['mae']:,.2f}")
    print(f"  RMSE: ${initial_metrics['rmse']:,.2f}")
    print(f"  R²: {initial_metrics['r2']:.4f}")
    print(f"  MAPE: {initial_metrics['mape']:.2f}%")
    
    # Step 3: Create performance plots
    print("\nStep 2: Creating performance visualization...")
    plot_performance_comparison(y_test, y_pred_initial)
    
    # Step 4: Load the trained model
    print("\nStep 3: Loading trained model for monitoring...")
    try:
        model_wrapper = HousePriceModelWithMonitoring()
    except Exception as e:
        print(f"Cannot load model: {e}")
        return
    
    # Step 5: Register with monitoring system
    print("\nStep 4: Registering model with monitoring system...")
    model_id = register_model_with_monitoring()
    
    if not model_id:
        print("Cannot proceed without model registration")
        return
    
    # Step 6: Make predictions with monitoring
    print("\nStep 5: Making predictions with monitoring...")
    
    # Sample house listings
    houses = [
        {
            "size": 2500,
            "bedrooms": 4,
            "bathrooms": 3,
            "age": 5,
            "location_score": 8.5
        },
        {
            "size": 1800,
            "bedrooms": 3,
            "bathrooms": 2,
            "age": 15,
            "location_score": 6.2
        },
        {
            "size": 3200,
            "bedrooms": 5,
            "bathrooms": 4,
            "age": 2,
            "location_score": 9.1
        }
    ]
    
    for i, house in enumerate(houses, 1):
        print(f"\nProcessing house {i}...")
        print(f"  Features: {house}")
        
        # Make prediction with monitoring
        result = predict_house_price_with_monitoring(model_wrapper, house, model_id)
        print(f"  Predicted Price: ${result['predicted_price']:,.2f}")
    
    # Demonstrate why monitoring is important
    demonstrate_model_drift()
    
    print("\n" + "=" * 55)
    print("Integration complete!")
    print("Your house price model is now making predictions and they are being monitored.")
    print(f"View dashboard at: http://127.0.0.1:8000/models/{model_id}/")
    print("\nWhat the monitoring system tracks for you:")
    print("  ✓ Real-time performance metrics (MAE, RMSE, R²)")
    print("  ✓ Data drift detection (changes in input features)")
    print("  ✓ Concept drift detection (degrading prediction accuracy)")
    print("  ✓ Data quality issues (missing values, outliers)")
    print("  ✓ Performance alerts when metrics drop below thresholds")
    print("  ✓ Historical performance trends")

if __name__ == "__main__":
    main()