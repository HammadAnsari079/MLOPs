from rest_framework import generics, status
from rest_framework.response import Response
from rest_framework.views import APIView
from django.shortcuts import get_object_or_404
from django.http import Http404
from .models import ModelRegistry, Prediction, DataQualityMetric, PerformanceMetric, DriftMetric, Alert
from .serializers import (
    ModelRegistrySerializer, 
    PredictionSerializer, 
    DataQualityMetricSerializer,
    PerformanceMetricSerializer,
    DriftMetricSerializer,
    AlertSerializer
)
import json
import numpy as np
from datetime import datetime, timedelta
from django.utils import timezone
from django.db.models import Count, Avg
from collections import defaultdict
import pandas as pd
from django.http import JsonResponse
from rest_framework.parsers import MultiPartParser, FormParser
import io
import logging
import uuid

# Set up logging
logger = logging.getLogger(__name__)

class ModelRegistryListCreateView(generics.ListCreateAPIView):
    queryset = ModelRegistry.objects.all().order_by('-created_at')
    serializer_class = ModelRegistrySerializer

class ModelRegistryDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = ModelRegistry.objects.all()
    serializer_class = ModelRegistrySerializer

class PredictionCreateView(APIView):
    def post(self, request, format=None):
        try:
            serializer = PredictionSerializer(data=request.data)
            if serializer.is_valid():
                # Perform data quality checks before saving
                prediction = serializer.save()
                
                # Here we would normally trigger async data quality checks
                # For now, we'll save directly
                return Response(serializer.data, status=status.HTTP_201_CREATED)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            logger.error(f"Error creating prediction: {str(e)}")
            return Response(
                {"error": "Failed to create prediction"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class PredictionListView(generics.ListAPIView):
    serializer_class = PredictionSerializer
    
    def get_queryset(self):
        try:
            model_id = self.kwargs['model_id']
            # Validate UUID format
            uuid.UUID(str(model_id))
            return Prediction.objects.filter(model_id=model_id).order_by('-timestamp')
        except (ValueError, KeyError):
            return Prediction.objects.none()

class MetricsSummaryView(APIView):
    def get(self, request, model_id):
        try:
            # Validate model_id
            uuid.UUID(str(model_id))
            
            # Get the latest metrics for a model
            try:
                latest_performance = PerformanceMetric.objects.filter(
                    model_id=model_id
                ).latest('timestamp')
                
                latest_drift = DriftMetric.objects.filter(
                    model_id=model_id
                ).latest('timestamp')
                
                latest_quality = DataQualityMetric.objects.filter(
                    model_id=model_id
                ).latest('timestamp')
                
                # Get recent alerts
                recent_alerts = Alert.objects.filter(
                    model_id=model_id,
                    timestamp__gte=timezone.now() - timedelta(hours=24)
                ).order_by('-timestamp')[:10]
                
                response_data = {
                    'performance': PerformanceMetricSerializer(latest_performance).data,
                    'drift': DriftMetricSerializer(latest_drift).data,
                    'quality': DataQualityMetricSerializer(latest_quality).data,
                    'recent_alerts': AlertSerializer(recent_alerts, many=True).data
                }
                
                return Response(response_data)
            except PerformanceMetric.DoesNotExist:
                return Response({'error': 'Performance metrics not available'}, status=404)
            except DriftMetric.DoesNotExist:
                return Response({'error': 'Drift metrics not available'}, status=404)
            except DataQualityMetric.DoesNotExist:
                return Response({'error': 'Data quality metrics not available'}, status=404)
        except ValueError:
            return Response({'error': 'Invalid model ID format'}, status=400)
        except Exception as e:
            logger.error(f"Error fetching metrics summary: {str(e)}")
            return Response({'error': 'Failed to fetch metrics'}, status=500)

class AlertListView(generics.ListAPIView):
    serializer_class = AlertSerializer
    
    def get_queryset(self):
        try:
            model_id = self.kwargs.get('model_id')
            hours = int(self.request.query_params.get('hours', 24))
            
            queryset = Alert.objects.filter(
                timestamp__gte=timezone.now() - timedelta(hours=hours)
            ).order_by('-timestamp')
            
            if model_id:
                # Validate UUID format
                uuid.UUID(str(model_id))
                queryset = queryset.filter(model_id=model_id)
                
            return queryset
        except (ValueError, TypeError):
            return Alert.objects.none()

class AlertResolveView(APIView):
    def post(self, request, alert_id):
        try:
            # Validate alert_id
            uuid.UUID(str(alert_id))
            alert = get_object_or_404(Alert, id=alert_id)
            alert.resolved = True
            alert.resolved_at = timezone.now()
            alert.save()
            return Response({'status': 'resolved'})
        except ValueError:
            return Response({'error': 'Invalid alert ID format'}, status=400)
        except Exception as e:
            logger.error(f"Error resolving alert: {str(e)}")
            return Response({'error': 'Failed to resolve alert'}, status=500)

class ModelHealthView(APIView):
    def get(self, request, model_id):
        try:
            # Validate model_id
            uuid.UUID(str(model_id))
            
            # Calculate health score based on recent metrics
            try:
                # Get recent performance metrics (last 24 hours)
                perf_metrics = PerformanceMetric.objects.filter(
                    model_id=model_id,
                    timestamp__gte=timezone.now() - timedelta(hours=24)
                )
                
                # Get recent drift metrics
                drift_metrics = DriftMetric.objects.filter(
                    model_id=model_id,
                    timestamp__gte=timezone.now() - timedelta(hours=24)
                )
                
                # Get recent alerts
                alerts = Alert.objects.filter(
                    model_id=model_id,
                    timestamp__gte=timezone.now() - timedelta(hours=24)
                )
                
                # Calculate health score (simplified algorithm)
                health_score = 100  # Start with perfect score
                
                # Deduct points for performance degradation
                if perf_metrics.exists():
                    latest_perf = perf_metrics.latest('timestamp')
                    if latest_perf.accuracy and latest_perf.accuracy < 0.8:
                        health_score -= 20
                    elif latest_perf.accuracy and latest_perf.accuracy < 0.9:
                        health_score -= 10
                        
                # Deduct points for drift
                if drift_metrics.exists():
                    latest_drift = drift_metrics.latest('timestamp')
                    if latest_drift.drift_detected:
                        health_score -= 25
                        
                # Deduct points for alerts
                critical_alerts = alerts.filter(level='CRITICAL').count()
                warning_alerts = alerts.filter(level='WARNING').count()
                
                health_score -= (critical_alerts * 15 + warning_alerts * 5)
                
                # Ensure health score is between 0 and 100
                health_score = max(0, min(100, health_score))
                
                # Determine status based on health score
                if health_score >= 80:
                    status_text = 'Healthy'
                elif health_score >= 60:
                    status_text = 'Degraded'
                else:
                    status_text = 'Unhealthy'
                    
                return Response({
                    'health_score': health_score,
                    'status': status_text,
                    'total_predictions': Prediction.objects.filter(model_id=model_id).count(),
                    'alerts_24h': {
                        'critical': critical_alerts,
                        'warning': warning_alerts,
                        'info': alerts.filter(level='INFO').count()
                    }
                })
            except Exception as e:
                logger.error(f"Error calculating model health: {str(e)}")
                return Response({'error': 'Failed to calculate health score'}, status=500)
        except ValueError:
            return Response({'error': 'Invalid model ID format'}, status=400)
        except Exception as e:
            logger.error(f"Error in model health view: {str(e)}")
            return Response({'error': str(e)}, status=500)

# Test endpoint for initial setup
class TestView(APIView):
    def get(self, request):
        return Response({"message": "MLOps Monitoring API is running!"})

# Monitoring task endpoints
class RunMetricsTaskView(APIView):
    def post(self, request):
        try:
            model_id = request.data.get('model_id')
            if model_id:
                # Validate model_id
                uuid.UUID(str(model_id))
            
            from monitoring.tasks import calculate_metrics_task
            result = calculate_metrics_task(model_id)
            return Response(
                {"message": "Performance metrics calculated successfully", "result": result}, 
                status=status.HTTP_200_OK
            )
        except ValueError:
            return Response(
                {"error": "Invalid model ID format"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            logger.error(f"Failed to calculate metrics: {str(e)}")
            return Response(
                {"error": f"Failed to calculate metrics: {str(e)}"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class RunDriftTaskView(APIView):
    def post(self, request):
        try:
            model_id = request.data.get('model_id')
            if model_id:
                # Validate model_id
                uuid.UUID(str(model_id))
            
            from monitoring.tasks import detect_drift_task
            result = detect_drift_task(model_id)
            return Response(
                {"message": "Drift detection completed successfully", "result": result}, 
                status=status.HTTP_200_OK
            )
        except ValueError:
            return Response(
                {"error": "Invalid model ID format"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            logger.error(f"Failed to detect drift: {str(e)}")
            return Response(
                {"error": f"Failed to detect drift: {str(e)}"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class RunDataQualityTaskView(APIView):
    def post(self, request):
        try:
            model_id = request.data.get('model_id')
            if model_id:
                # Validate model_id
                uuid.UUID(str(model_id))
            
            from monitoring.tasks import check_data_quality_task
            result = check_data_quality_task(model_id)
            return Response(
                {"message": "Data quality check completed successfully", "result": result}, 
                status=status.HTTP_200_OK
            )
        except ValueError:
            return Response(
                {"error": "Invalid model ID format"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            logger.error(f"Failed to check data quality: {str(e)}")
            return Response(
                {"error": f"Failed to check data quality: {str(e)}"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

# New endpoints for training and data cleaning
class TrainModelView(APIView):
    def post(self, request):
        try:
            return Response(
                {"message": "Model training endpoint is working!"}, 
                status=status.HTTP_200_OK
            )
        except Exception as e:
            logger.error(f"Error in train model view: {str(e)}")
            return Response(
                {"error": f"Model training failed: {str(e)}"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class CleanDataView(APIView):
    def post(self, request):
        try:
            # Get data from request
            data = request.data.get('data', [])
            config = request.data.get('config', {})
            file_format = request.data.get('file_format', 'json')  # Track original file format
            
            if not data:
                return Response(
                    {"error": "No data provided"}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Convert to DataFrame
            import pandas as pd
            df = pd.DataFrame(data)
            
            # Apply cleaning based on config
            from preprocessing.data_cleaner import DataCleaner
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
                df = cleaner.encode_categorical_variables(df, 'label')
            
            # Standardize features
            if config.get('standardize_features', False):
                df = cleaner.standardize_numeric_features(df)
            
            # Convert back to appropriate format
            cleaned_data = df.to_dict('records')
            
            return Response(
                {
                    "message": "Data cleaning completed successfully",
                    "cleaned_data": cleaned_data,
                    "rows_processed": len(cleaned_data),
                    "original_format": file_format  # Include original format in response
                }, 
                status=status.HTTP_200_OK
            )
        except Exception as e:
            logger.error(f"Data cleaning failed: {str(e)}")
            return Response(
                {"error": f"Data cleaning failed: {str(e)}"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class DownloadCleanedDataView(APIView):
    def post(self, request):
        try:
            # Get data and format from request
            data = request.data.get('data', [])
            file_format = request.data.get('format', 'json')
            
            if not data:
                return Response(
                    {"error": "No data provided"}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Convert to DataFrame
            import pandas as pd
            import io
            import base64
            df = pd.DataFrame(data)
            
            # Create file in requested format
            if file_format == 'csv':
                # Convert to CSV
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                file_content = csv_buffer.getvalue()
                content_type = 'text/csv'
                filename = 'cleaned_data.csv'
            elif file_format in ['xlsx', 'xls']:
                # Convert to Excel
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Cleaned_Data')
                file_content = excel_buffer.getvalue()
                content_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                filename = f'cleaned_data.{file_format}'
            else:
                # Default to JSON
                file_content = json.dumps(data, indent=2)
                content_type = 'application/json'
                filename = 'cleaned_data.json'
            
            # Encode file content in base64 for transmission
            if isinstance(file_content, str):
                file_base64 = base64.b64encode(file_content.encode('utf-8')).decode('utf-8')
            else:
                file_base64 = base64.b64encode(file_content).decode('utf-8')
            
            return Response(
                {
                    "message": "File prepared for download",
                    "file_content": file_base64,
                    "content_type": content_type,
                    "filename": filename
                }, 
                status=status.HTTP_200_OK
            )
        except Exception as e:
            logger.error(f"File preparation failed: {str(e)}")
            return Response(
                {"error": f"File preparation failed: {str(e)}"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class FileUploadView(APIView):
    parser_classes = (MultiPartParser, FormParser)
    
    def post(self, request, format=None):
        try:
            file_obj = request.FILES['file']
            file_extension = file_obj.name.split('.')[-1].lower()
            
            # Read file content
            if file_extension == 'csv':
                df = pd.read_csv(file_obj)
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(file_obj)
            else:
                return Response(
                    {"error": "Unsupported file format"}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Convert to JSON-serializable format
            data = df.to_dict('records')
            
            return Response(
                {"message": "File processed successfully", "data": data},
                status=status.HTTP_200_OK
            )
        except Exception as e:
            logger.error(f"File processing failed: {str(e)}")
            return Response(
                {"error": f"File processing failed: {str(e)}"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )