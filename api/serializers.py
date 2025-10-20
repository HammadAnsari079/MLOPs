from rest_framework import serializers
from .models import ModelRegistry, Prediction, DataQualityMetric, PerformanceMetric, DriftMetric, Alert

class ModelRegistrySerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelRegistry
        fields = '__all__'

class PredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prediction
        fields = '__all__'
        read_only_fields = ('id', 'timestamp')

class DataQualityMetricSerializer(serializers.ModelSerializer):
    class Meta:
        model = DataQualityMetric
        fields = '__all__'

class PerformanceMetricSerializer(serializers.ModelSerializer):
    class Meta:
        model = PerformanceMetric
        fields = '__all__'

class DriftMetricSerializer(serializers.ModelSerializer):
    class Meta:
        model = DriftMetric
        fields = '__all__'

class AlertSerializer(serializers.ModelSerializer):
    class Meta:
        model = Alert
        fields = '__all__'
        read_only_fields = ('id', 'timestamp')