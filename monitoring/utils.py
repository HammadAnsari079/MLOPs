import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
import json
from datetime import datetime, timedelta
from django.utils import timezone
from api.models import Prediction, PerformanceMetric, DriftMetric, DataQualityMetric, Alert
from collections import defaultdict
import logging

# Set up logging
logger = logging.getLogger(__name__)

def calculate_psi(expected, actual, buckets=10):
    """
    Calculate Population Stability Index (PSI)
    """
    # Handle edge case of empty arrays
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    
    # Convert to numpy arrays
    expected = np.array(expected)
    actual = np.array(actual)
    
    # Create bins
    breakpoints = np.linspace(expected.min(), expected.max(), buckets + 1)
    
    # Calculate percentages in each bin
    expected_counts, _ = np.histogram(expected, breakpoints)
    actual_counts, _ = np.histogram(actual, breakpoints)
    
    # Convert to percentages
    expected_pct = expected_counts / len(expected) if len(expected) > 0 else np.zeros_like(expected_counts, dtype=float)
    actual_pct = actual_counts / len(actual) if len(actual) > 0 else np.zeros_like(actual_counts, dtype=float)
    
    # Replace zeros with small value to avoid division by zero
    epsilon = 1e-10
    expected_pct = np.where(expected_pct == 0, epsilon, expected_pct)
    actual_pct = np.where(actual_pct == 0, epsilon, actual_pct)
    
    # Calculate PSI
    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    psi = np.sum(psi_values)
    
    return psi

def calculate_ks_test(expected, actual):
    """
    Calculate Kolmogorov-Smirnov test statistic
    """
    # Handle edge case of empty arrays
    if len(expected) == 0 or len(actual) == 0:
        return 0.0, 1.0
    
    # Convert to numpy arrays
    expected = np.array(expected)
    actual = np.array(actual)
    
    ks_statistic, p_value = stats.ks_2samp(expected, actual)
    return ks_statistic, p_value

def calculate_kl_divergence(expected, actual, buckets=10):
    """
    Calculate Kullback-Leibler divergence
    """
    # Handle edge case of empty arrays
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    
    # Convert to numpy arrays
    expected = np.array(expected)
    actual = np.array(actual)
    
    # Create bins
    all_values = np.concatenate([expected, actual])
    if len(all_values) == 0:
        return 0.0
        
    breakpoints = np.linspace(all_values.min(), all_values.max(), buckets + 1)
    
    # Calculate histograms
    expected_counts, _ = np.histogram(expected, breakpoints)
    actual_counts, _ = np.histogram(actual, breakpoints)
    
    # Convert to probabilities
    expected_prob = expected_counts / len(expected) if len(expected) > 0 else np.zeros_like(expected_counts, dtype=float)
    actual_prob = actual_counts / len(actual) if len(actual) > 0 else np.zeros_like(actual_counts, dtype=float)
    
    # Replace zeros with small value to avoid division by zero
    epsilon = 1e-10
    expected_prob = np.where(expected_prob == 0, epsilon, expected_prob)
    actual_prob = np.where(actual_prob == 0, epsilon, actual_prob)
    
    # Calculate KL divergence
    kl_div = np.sum(expected_prob * np.log(expected_prob / actual_prob))
    
    return kl_div

def detect_data_drift(training_data, production_data, feature_name):
    """
    Detect data drift for a specific feature using multiple methods
    """
    # Handle edge case of empty arrays
    if len(training_data) == 0 or len(production_data) == 0:
        return {
            'psi': 0.0,
            'ks_statistic': 0.0,
            'ks_p_value': 1.0,
            'kl_divergence': 0.0,
            'drift_detected': False
        }
    
    # Convert to numpy arrays
    train_array = np.array(training_data)
    prod_array = np.array(production_data)
    
    # Calculate drift metrics
    psi = calculate_psi(train_array, prod_array)
    ks_stat, ks_p_value = calculate_ks_test(train_array, prod_array)
    kl_div = calculate_kl_divergence(train_array, prod_array)
    
    # Determine if drift is detected (consensus approach)
    drift_detected = 0
    if psi >= 0.2:
        drift_detected += 1
    if ks_p_value < 0.05:
        drift_detected += 1
    if kl_div >= 0.1:
        drift_detected += 1
    
    # Drift is detected if 2+ methods agree
    is_drift = drift_detected >= 2
    
    return {
        'psi': psi,
        'ks_statistic': ks_stat,
        'ks_p_value': ks_p_value,
        'kl_divergence': kl_div,
        'drift_detected': is_drift
    }

def calculate_performance_metrics(predictions, actuals=None, is_classification=True):
    """
    Calculate performance metrics for predictions
    """
    if len(predictions) == 0:
        return {}
    
    metrics = {
        'sample_size': len(predictions)
    }
    
    if is_classification:
        # For classification tasks
        if actuals is not None and len(actuals) == len(predictions):
            # Convert to binary if needed
            pred_binary = [1 if p > 0.5 else 0 for p in predictions]
            
            try:
                # Calculate confusion matrix
                tn, fp, fn, tp = confusion_matrix(actuals, pred_binary).ravel()
                
                # Calculate metrics with division by zero protection
                metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
                metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                
                precision = metrics['precision']
                recall = metrics['recall']
                metrics['f1_score'] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
                metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            except ValueError as e:
                # Handle case where confusion matrix can't be calculated
                logger.warning(f"Could not calculate confusion matrix: {e}")
                metrics['accuracy'] = 0.0
                metrics['precision'] = 0.0
                metrics['recall'] = 0.0
                metrics['f1_score'] = 0.0
                metrics['false_positive_rate'] = 0.0
                metrics['false_negative_rate'] = 0.0
    else:
        # For regression tasks
        if actuals is not None and len(actuals) == len(predictions):
            try:
                metrics['mae'] = mean_absolute_error(actuals, predictions)
                metrics['rmse'] = np.sqrt(mean_squared_error(actuals, predictions))
                metrics['r_squared'] = r2_score(actuals, predictions)
                
                # Handle potential division by zero in MAPE
                actuals_array = np.array(actuals)
                predictions_array = np.array(predictions)
                non_zero_actuals = actuals_array != 0
                if np.any(non_zero_actuals):
                    mape = np.mean(np.abs((actuals_array[non_zero_actuals] - predictions_array[non_zero_actuals]) / actuals_array[non_zero_actuals])) * 100
                    metrics['mape'] = mape
                else:
                    metrics['mape'] = 0.0
            except Exception as e:
                logger.error(f"Error calculating regression metrics: {e}")
                metrics['mae'] = 0.0
                metrics['rmse'] = 0.0
                metrics['r_squared'] = 0.0
                metrics['mape'] = 0.0
    
    return metrics

def check_data_quality(predictions):
    """
    Check data quality for a batch of predictions
    """
    quality_metrics = {
        'missing_counts': {},
        'missing_percentages': {},
        'outlier_counts': {},
        'type_mismatches': {},
        'schema_violations': {}
    }
    
    if len(predictions) == 0:
        return quality_metrics
    
    # Check each prediction
    total_predictions = len(predictions)
    
    for pred in predictions:
        input_data = pred.input_data
        
        # Check for missing values
        for key, value in input_data.items():
            if value is None or (isinstance(value, str) and value == ''):
                if key not in quality_metrics['missing_counts']:
                    quality_metrics['missing_counts'][key] = 0
                quality_metrics['missing_counts'][key] += 1
        
        # Check for outliers (simplified - would need baseline data in production)
        # This is a placeholder - in production, you'd compare against training data statistics
    
    # Calculate percentages
    for key, count in quality_metrics['missing_counts'].items():
        quality_metrics['missing_percentages'][key] = (count / total_predictions) * 100 if total_predictions > 0 else 0.0
    
    return quality_metrics

def detect_concept_drift(model_id, window_size=1000, threshold=0.05):
    """
    Detect concept drift by comparing accuracy over time windows
    """
    # Get recent predictions
    recent_predictions = Prediction.objects.filter(
        model_id=model_id,
        timestamp__gte=timezone.now() - timedelta(hours=24)
    ).order_by('-timestamp')
    
    if len(recent_predictions) < window_size * 2:
        return None  # Not enough data
    
    # Split into two windows
    current_window = recent_predictions[:window_size]
    previous_window = recent_predictions[window_size:window_size*2]
    
    # Calculate accuracy for each window (simplified)
    # In practice, you'd need actual values to compare against
    current_count = len(current_window)
    previous_count = len(previous_window)
    
    current_accuracy = len([p for p in current_window if p.prediction.get('confidence', 0) > 0.8]) / current_count if current_count > 0 else 0.0
    previous_accuracy = len([p for p in previous_window if p.prediction.get('confidence', 0) > 0.8]) / previous_count if previous_count > 0 else 0.0
    
    # Check for significant drop
    accuracy_drop = previous_accuracy - current_accuracy
    if accuracy_drop > threshold:
        return {
            'current_accuracy': current_accuracy,
            'previous_accuracy': previous_accuracy,
            'drift_detected': True,
            'accuracy_drop': accuracy_drop
        }
    
    return {
        'current_accuracy': current_accuracy,
        'previous_accuracy': previous_accuracy,
        'drift_detected': False,
        'accuracy_drop': accuracy_drop
    }