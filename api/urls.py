from django.urls import path
from . import views

urlpatterns = [
    # Model Registry
    path('models/', views.ModelRegistryListCreateView.as_view(), name='model-list-create'),
    path('models/<uuid:pk>/', views.ModelRegistryDetailView.as_view(), name='model-detail'),
    
    # Predictions
    path('predictions/', views.PredictionCreateView.as_view(), name='prediction-create'),
    path('models/<uuid:model_id>/predictions/', views.PredictionListView.as_view(), name='prediction-list'),
    path('models/<uuid:model_id>/bulk-predictions/', views.BulkPredictionUploadView.as_view(), name='bulk-prediction-upload'),
    
    # Metrics
    path('models/<uuid:model_id>/metrics/', views.MetricsSummaryView.as_view(), name='metrics-summary'),
    
    # Alerts
    path('alerts/', views.AlertListView.as_view(), name='alert-list'),
    path('models/<uuid:model_id>/alerts/', views.AlertListView.as_view(), name='model-alert-list'),
    path('alerts/<uuid:alert_id>/resolve/', views.AlertResolveView.as_view(), name='alert-resolve'),
    
    # Health
    path('models/<uuid:model_id>/health/', views.ModelHealthView.as_view(), name='model-health'),
    
    # Monitoring Tasks
    path('run-metrics/', views.RunMetricsTaskView.as_view(), name='run-metrics'),
    path('run-drift/', views.RunDriftTaskView.as_view(), name='run-drift'),
    path('run-data-quality/', views.RunDataQualityTaskView.as_view(), name='run-data-quality'),
    
    # Training and Data Cleaning
    path('train/', views.TrainModelView.as_view(), name='train-model'),
    path('clean-data/', views.CleanDataView.as_view(), name='clean-data'),
    path('download-cleaned-data/', views.DownloadCleanedDataView.as_view(), name='download-cleaned-data'),
    path('upload-file/', views.FileUploadView.as_view(), name='file-upload'),
    
    # Test endpoint
    path('test/', views.TestView.as_view(), name='test'),
]