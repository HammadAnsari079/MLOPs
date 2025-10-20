"""
Example Logistic Regression Model Template for MLOps Platform

This is a template showing how to structure your Python model file for upload to the MLOps platform.
The model should have methods for prediction and be compatible with the monitoring system.

Requirements:
- Model must be a Python class with a 'predict' method
- The predict method should accept input data and return predictions
- Model should be serializable (using joblib, pickle, etc.)
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import joblib

class LogisticRegressionModel:
    def __init__(self):
        """Initialize the model"""
        self.model = LogisticRegression(random_state=42)
        self.input_features = 5  # Number of input features
        self.output_classes = 5   # Number of output classes
        
    def train(self, X, y):
        """
        Train the model with provided data
        
        Args:
            X: Input features (numpy array or pandas DataFrame)
            y: Target labels (numpy array or pandas Series)
        """
        # Ensure X has the right shape
        if X.shape[1] != self.input_features:
            raise ValueError(f"Expected {self.input_features} features, got {X.shape[1]}")
            
        # Train the model
        self.model.fit(X, y)
        
    def predict(self, X):
        """
        Make predictions on input data
        
        Args:
            X: Input features (numpy array or pandas DataFrame)
            
        Returns:
            predictions: Model predictions
        """
        # Ensure X has the right shape
        if hasattr(X, 'shape') and len(X.shape) == 1:
            # Reshape single sample
            X = X.reshape(1, -1)
            
        if hasattr(X, 'shape') and X.shape[1] != self.input_features:
            raise ValueError(f"Expected {self.input_features} features, got {X.shape[1]}")
            
        # Make predictions
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        return {
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist()
        }
    
    def predict_single(self, input_data):
        """
        Make a prediction for a single input sample
        
        Args:
            input_data: Dictionary or list with input features
            
        Returns:
            Dictionary with prediction results
        """
        # Convert input to numpy array
        if isinstance(input_data, dict):
            # Extract values in order (you may need to adjust this based on your feature names)
            feature_names = [f'feature_{i}' for i in range(1, self.input_features + 1)]
            X = np.array([input_data.get(name, 0) for name in feature_names]).reshape(1, -1)
        else:
            # Assume it's a list or similar
            X = np.array(input_data).reshape(1, -1)
            
        # Make prediction
        result = self.predict(X)
        
        return {
            'predicted_class': result['predictions'][0],
            'class_probabilities': result['probabilities'][0]
        }

# Example usage (for testing):
if __name__ == "__main__":
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=5, n_classes=5, 
                             n_informative=4, n_redundant=1, random_state=42)
    
    # Create and train model
    model = LogisticRegressionModel()
    model.train(X, y)
    
    # Test prediction
    sample_input = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = model.predict_single(sample_input)
    print("Sample prediction:", result)
    
    # Save model
    joblib.dump(model, 'logistic_model_example.pkl')
    print("Model saved as logistic_model_example.pkl")