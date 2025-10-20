"""
Model training pipeline for the MLOps platform
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import uuid
from datetime import datetime
from api.models import ModelRegistry
import os

class ModelTrainer:
    def __init__(self, model_name, algorithm='random_forest'):
        self.model_name = model_name
        self.algorithm = algorithm
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
    def prepare_data(self, df, target_column):
        """Prepare data for training"""
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Handle categorical variables
        for column in X.columns:
            if X[column].dtype == 'object':
                le = LabelEncoder()
                X[column] = le.fit_transform(X[column].astype(str))
                self.label_encoders[column] = le
        
        # Handle target variable if categorical
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
            self.label_encoders[target_column] = le
            
        return X, y
    
    def train(self, X, y, test_size=0.2, random_state=42):
        """Train the model"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model based on algorithm
        if self.algorithm == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        elif self.algorithm == 'logistic_regression':
            self.model = LogisticRegression(random_state=random_state)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
            
        # Fit model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        return metrics
    
    def predict(self, X):
        """Make predictions with the trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
            
        # Apply same preprocessing
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        return predictions, probabilities
    
    def save_model(self, model_id):
        """Save the trained model to disk"""
        # Create models directory if it doesn't exist
        models_dir = 'trained_models'
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            
        # Save model components
        model_path = os.path.join(models_dir, f'{model_id}_model.pkl')
        scaler_path = os.path.join(models_dir, f'{model_id}_scaler.pkl')
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'algorithm': self.algorithm,
            'feature_names': self.feature_names,
            'label_encoders': list(self.label_encoders.keys()),
            'trained_at': datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(models_dir, f'{model_id}_metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
            
        return {
            'model_path': model_path,
            'scaler_path': scaler_path,
            'metadata_path': metadata_path
        }
    
    def load_model(self, model_id):
        """Load a trained model from disk"""
        models_dir = 'trained_models'
        model_path = os.path.join(models_dir, f'{model_id}_model.pkl')
        scaler_path = os.path.join(models_dir, f'{model_id}_scaler.pkl')
        
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            
        return self.model

def train_model_from_data(data, target_column, model_name, algorithm='random_forest'):
    """
    Train a model from raw data
    
    Args:
        data: DataFrame or dict with training data
        target_column: Name of the target column
        model_name: Name for the model
        algorithm: Algorithm to use ('random_forest' or 'logistic_regression')
    
    Returns:
        dict: Training results and model metadata
    """
    # Convert to DataFrame if needed
    if isinstance(data, dict):
        df = pd.DataFrame(data)
    else:
        df = data.copy()
    
    # Initialize trainer
    trainer = ModelTrainer(model_name, algorithm)
    
    # Prepare data
    X, y = trainer.prepare_data(df, target_column)
    
    # Train model
    metrics = trainer.train(X, y)
    
    # Generate model ID
    model_id = str(uuid.uuid4())
    
    # Save model
    paths = trainer.save_model(model_id)
    
    return {
        'model_id': model_id,
        'metrics': metrics,
        'paths': paths,
        'feature_names': trainer.feature_names
    }