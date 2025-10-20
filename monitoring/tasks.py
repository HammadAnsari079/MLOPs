# Simple version without Celery dependency
from django.utils import timezone
from datetime import timedelta
from api.models import ModelRegistry, Prediction, PerformanceMetric, DriftMetric, DataQualityMetric, Alert
from .utils import calculate_performance_metrics, check_data_quality, detect_data_drift, detect_concept_drift
import numpy as np
import json
import logging
import uuid

# Set up logging
logger = logging.getLogger(__name__)

def calculate_metrics_task(model_id=None):
    """
    Calculate performance metrics for predictions
    """
    try:
        # Get models to process
        if model_id:
            try:
                uuid.UUID(str(model_id))
                models = ModelRegistry.objects.filter(id=model_id)
            except ValueError:
                logger.error(f"Invalid model ID format: {model_id}")
                return "Invalid model ID format"
        else:
            models = ModelRegistry.objects.all()
        
        processed_count = 0
        for model in models:
            try:
                # Get recent predictions (last 5 minutes)
                since_time = timezone.now() - timedelta(minutes=5)
                predictions = Prediction.objects.filter(
                    model=model,
                    timestamp__gte=since_time
                )
                
                if not predictions.exists():
                    continue
                    
                # Extract prediction values and actual values (if available)
                pred_values = []
                actual_values = []
                
                for pred in predictions:
                    try:
                        # Extract prediction value (assuming binary classification with probability)
                        if isinstance(pred.prediction, dict):
                            if 'probability' in pred.prediction:
                                pred_values.append(pred.prediction['probability'])
                            elif 'score' in pred.prediction:
                                pred_values.append(pred.prediction['score'])
                            else:
                                # Try to get any numeric value
                                for key, value in pred.prediction.items():
                                    if isinstance(value, (int, float)):
                                        pred_values.append(float(value))
                                        break
                        elif isinstance(pred.prediction, (int, float)):
                            pred_values.append(float(pred.prediction))
                        
                        # Extract actual value if available
                        if pred.actual_value is not None:
                            if isinstance(pred.actual_value, (int, float)):
                                actual_values.append(float(pred.actual_value))
                            elif isinstance(pred.actual_value, dict):
                                # Try to get any numeric value
                                for key, value in pred.actual_value.items():
                                    if isinstance(value, (int, float)):
                                        actual_values.append(float(value))
                                        break
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Skipping prediction {pred.id} due to data conversion error: {e}")
                        continue
                
                # Calculate performance metrics
                metrics = calculate_performance_metrics(pred_values, actual_values)
                
                if metrics:
                    # Save metrics
                    perf_metric = PerformanceMetric(
                        model=model,
                        sample_size=metrics.get('sample_size', 0),
                        accuracy=metrics.get('accuracy'),
                        precision=metrics.get('precision'),
                        recall=metrics.get('recall'),
                        f1_score=metrics.get('f1_score'),
                        false_positive_rate=metrics.get('false_positive_rate'),
                        false_negative_rate=metrics.get('false_negative_rate'),
                        mae=metrics.get('mae'),
                        rmse=metrics.get('rmse'),
                        r_squared=metrics.get('r_squared'),
                        mape=metrics.get('mape')
                    )
                    perf_metric.save()
                    
                    # Check for performance degradation alerts
                    if metrics.get('accuracy') and metrics['accuracy'] < 0.8:
                        Alert.objects.create(
                            model=model,
                            alert_type='PERFORMANCE',
                            level='CRITICAL',
                            title=f'Performance Degradation for {model.name}',
                            description=f'Model accuracy dropped below 80%: {metrics["accuracy"]:.2f}'
                        )
                    elif metrics.get('accuracy') and metrics['accuracy'] < 0.9:
                        Alert.objects.create(
                            model=model,
                            alert_type='PERFORMANCE',
                            level='WARNING',
                            title=f'Performance Warning for {model.name}',
                            description=f'Model accuracy below 90%: {metrics["accuracy"]:.2f}'
                        )
                
                processed_count += 1
            except Exception as e:
                logger.error(f"Error processing model {model.id}: {str(e)}")
                continue
        
        return f"Calculated metrics for {processed_count} models"
    except Exception as e:
        logger.error(f"Error in calculate_metrics_task: {str(e)}")
        return f"Error calculating metrics: {str(e)}"

def detect_drift_task(model_id=None):
    """
    Detect data and concept drift
    """
    try:
        # Get models to process
        if model_id:
            try:
                uuid.UUID(str(model_id))
                models = ModelRegistry.objects.filter(id=model_id)
            except ValueError:
                logger.error(f"Invalid model ID format: {model_id}")
                return "Invalid model ID format"
        else:
            models = ModelRegistry.objects.all()
        
        processed_count = 0
        for model in models:
            try:
                # Get recent predictions (last 24 hours)
                since_time = timezone.now() - timedelta(hours=24)
                predictions = Prediction.objects.filter(
                    model=model,
                    timestamp__gte=since_time
                )
                
                if not predictions.exists():
                    continue
                    
                # Extract feature data for drift detection
                feature_data = {}
                
                # Collect feature values
                for pred in predictions:
                    try:
                        input_data = pred.input_data
                        for key, value in input_data.items():
                            if isinstance(value, (int, float)):
                                if key not in feature_data:
                                    feature_data[key] = []
                                feature_data[key].append(float(value))
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Skipping prediction {pred.id} due to data conversion error: {e}")
                        continue
                
                # Detect data drift for each feature
                feature_drifts = {}
                overall_drift_score = 0
                drift_detected = False
                
                for feature_name, values in feature_data.items():
                    if len(values) < 10:  # Need sufficient data
                        continue
                        
                    # For demonstration, we'll compare first half with second half
                    half_point = len(values) // 2
                    training_data = values[:half_point]
                    production_data = values[half_point:]
                    
                    if len(training_data) > 0 and len(production_data) > 0:
                        drift_result = detect_data_drift(np.array(training_data), np.array(production_data), feature_name)
                        feature_drifts[feature_name] = drift_result
                        
                        if drift_result['drift_detected']:
                            overall_drift_score += 1
                            drift_detected = True
                
                # Detect concept drift
                concept_drift = detect_concept_drift(model.id)
                
                # Save drift metrics
                drift_metric = DriftMetric(
                    model=model,
                    feature_drifts=feature_drifts,
                    concept_drift=concept_drift or {},
                    overall_drift_score=overall_drift_score,
                    drift_detected=drift_detected
                )
                drift_metric.save()
                
                # Create alerts for significant drift
                if drift_detected:
                    Alert.objects.create(
                        model=model,
                        alert_type='DRIFT',
                        level='CRITICAL',
                        title=f'Data Drift Detected for {model.name}',
                        description=f'Significant data drift detected in {overall_drift_score} features'
                    )
                
                if concept_drift and concept_drift.get('drift_detected'):
                    Alert.objects.create(
                        model=model,
                        alert_type='DRIFT',
                        level='WARNING',
                        title=f'Concept Drift Detected for {model.name}',
                        description=f'Concept drift detected: accuracy dropped by {concept_drift["accuracy_drop"]:.2f}'
                    )
                
                processed_count += 1
            except Exception as e:
                logger.error(f"Error processing model {model.id} for drift detection: {str(e)}")
                continue
        
        return f"Drift detection completed for {processed_count} models"
    except Exception as e:
        logger.error(f"Error in detect_drift_task: {str(e)}")
        return f"Error detecting drift: {str(e)}"

def check_data_quality_task(model_id=None):
    """
    Check data quality for predictions
    """
    try:
        # Get models to process
        if model_id:
            try:
                uuid.UUID(str(model_id))
                models = ModelRegistry.objects.filter(id=model_id)
            except ValueError:
                logger.error(f"Invalid model ID format: {model_id}")
                return "Invalid model ID format"
        else:
            models = ModelRegistry.objects.all()
        
        processed_count = 0
        for model in models:
            try:
                # Get recent predictions (last 5 minutes)
                since_time = timezone.now() - timedelta(minutes=5)
                predictions = Prediction.objects.filter(
                    model=model,
                    timestamp__gte=since_time
                )
                
                if not predictions.exists():
                    continue
                    
                # Check data quality
                quality_metrics = check_data_quality(predictions)
                
                # Save quality metrics
                quality_metric = DataQualityMetric(
                    model=model,
                    missing_counts=quality_metrics['missing_counts'],
                    missing_percentages=quality_metrics['missing_percentages'],
                    outlier_counts=quality_metrics['outlier_counts'],
                    type_mismatches=quality_metrics['type_mismatches'],
                    schema_violations=quality_metrics['schema_violations']
                )
                quality_metric.save()
                
                # Create alerts for data quality issues
                for feature, percentage in quality_metrics['missing_percentages'].items():
                    if percentage > 5:  # More than 5% missing values
                        Alert.objects.create(
                            model=model,
                            alert_type='DATA_QUALITY',
                            level='WARNING',
                            title=f'High Missing Values in {feature}',
                            description=f'{percentage:.2f}% of values are missing for feature {feature}'
                        )
                
                processed_count += 1
            except Exception as e:
                logger.error(f"Error processing model {model.id} for data quality check: {str(e)}")
                continue
        
        return f"Data quality check completed for {processed_count} models"
    except Exception as e:
        logger.error(f"Error in check_data_quality_task: {str(e)}")
        return f"Error checking data quality: {str(e)}"

def periodic_monitoring_task():
    """
    Run all monitoring tasks periodically
    """
    try:
        # Run all tasks
        result1 = calculate_metrics_task()
        result2 = detect_drift_task()
        result3 = check_data_quality_task()
        
        return f"Periodic monitoring tasks completed: {result1}, {result2}, {result3}"
    except Exception as e:
        logger.error(f"Error in periodic_monitoring_task: {str(e)}")
        return f"Error in periodic monitoring: {str(e)}"