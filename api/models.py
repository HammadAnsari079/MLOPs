from django.db import models
import uuid

class ModelRegistry(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    version = models.CharField(max_length=50)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        unique_together = ('name', 'version')
        indexes = [
            models.Index(fields=['name']),
            models.Index(fields=['created_at']),
        ]
    
    def __str__(self):
        return f"{self.name} v{self.version}"

class Prediction(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    model = models.ForeignKey(ModelRegistry, on_delete=models.CASCADE, related_name='predictions')
    input_data = models.JSONField()  # Store input features as JSON
    prediction = models.JSONField()   # Store prediction output
    actual_value = models.JSONField(null=True, blank=True)  # For supervised learning feedback
    latency_ms = models.FloatField(null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    
    # Data quality fields
    missing_values = models.JSONField(default=dict)  # {feature_name: count}
    outliers = models.JSONField(default=dict)        # {feature_name: [outlier_values]}
    
    class Meta:
        indexes = [
            models.Index(fields=['model', '-timestamp']),
            models.Index(fields=['timestamp']),
        ]
    
    def __str__(self):
        return f"Prediction for {self.model.name} at {self.timestamp}"

class DataQualityMetric(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    model = models.ForeignKey(ModelRegistry, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)
    
    # Missing values tracking
    missing_counts = models.JSONField(default=dict)  # {feature_name: count}
    missing_percentages = models.JSONField(default=dict)  # {feature_name: percentage}
    
    # Outliers tracking
    outlier_counts = models.JSONField(default=dict)  # {feature_name: count}
    
    # Type mismatches
    type_mismatches = models.JSONField(default=dict)  # {feature_name: error_details}
    
    # Schema changes
    schema_violations = models.JSONField(default=dict)  # {error_type: details}
    
    class Meta:
        indexes = [
            models.Index(fields=['model', '-timestamp']),
        ]
    
    def __str__(self):
        return f"Data Quality for {self.model.name} at {self.timestamp}"

class PerformanceMetric(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    model = models.ForeignKey(ModelRegistry, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)
    
    # Classification metrics
    accuracy = models.FloatField(null=True, blank=True)
    precision = models.FloatField(null=True, blank=True)
    recall = models.FloatField(null=True, blank=True)
    f1_score = models.FloatField(null=True, blank=True)
    false_positive_rate = models.FloatField(null=True, blank=True)
    false_negative_rate = models.FloatField(null=True, blank=True)
    
    # Regression metrics
    mae = models.FloatField(null=True, blank=True)  # Mean Absolute Error
    rmse = models.FloatField(null=True, blank=True)  # Root Mean Square Error
    r_squared = models.FloatField(null=True, blank=True)  # R-squared
    mape = models.FloatField(null=True, blank=True)  # Mean Absolute Percentage Error
    
    # Infrastructure metrics
    avg_latency_ms = models.FloatField(null=True, blank=True)
    p95_latency_ms = models.FloatField(null=True, blank=True)
    p99_latency_ms = models.FloatField(null=True, blank=True)
    error_rate = models.FloatField(null=True, blank=True)
    
    # Sample size
    sample_size = models.IntegerField()
    
    class Meta:
        indexes = [
            models.Index(fields=['model', '-timestamp']),
        ]
    
    def __str__(self):
        return f"Performance for {self.model.name} at {self.timestamp}"

class DriftMetric(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    model = models.ForeignKey(ModelRegistry, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)
    
    # Feature drift metrics
    feature_drifts = models.JSONField(default=dict)  # {feature_name: {psi, ks_stat, kl_div}}
    
    # Prediction drift
    prediction_drift = models.JSONField(default=dict)  # {psi, ks_stat, kl_div}
    
    # Concept drift (accuracy over time windows)
    concept_drift = models.JSONField(default=dict)  # {window_start_time: accuracy}
    
    # Overall drift status
    overall_drift_score = models.FloatField()
    drift_detected = models.BooleanField(default=False)
    
    class Meta:
        indexes = [
            models.Index(fields=['model', '-timestamp']),
        ]
    
    def __str__(self):
        return f"Drift for {self.model.name} at {self.timestamp}"

class Alert(models.Model):
    ALERT_TYPES = [
        ('DATA_QUALITY', 'Data Quality Issue'),
        ('PERFORMANCE', 'Performance Degradation'),
        ('DRIFT', 'Data/Concept Drift'),
        ('INFRASTRUCTURE', 'Infrastructure Issue'),
    ]
    
    ALERT_LEVELS = [
        ('INFO', 'Information'),
        ('WARNING', 'Warning'),
        ('CRITICAL', 'Critical'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    model = models.ForeignKey(ModelRegistry, on_delete=models.CASCADE)
    alert_type = models.CharField(max_length=20, choices=ALERT_TYPES)
    level = models.CharField(max_length=10, choices=ALERT_LEVELS)
    title = models.CharField(max_length=255)
    description = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    resolved = models.BooleanField(default=False)
    resolved_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['model', '-timestamp']),
            models.Index(fields=['alert_type']),
            models.Index(fields=['level']),
        ]
    
    def __str__(self):
        return f"{self.title} - {self.level}"