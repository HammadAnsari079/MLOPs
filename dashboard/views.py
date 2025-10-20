from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from api.models import ModelRegistry, Prediction, PerformanceMetric, DriftMetric, Alert
from django.utils import timezone
from datetime import timedelta

def dashboard(request):
    """Main dashboard view showing all registered models"""
    models = ModelRegistry.objects.all().order_by('-created_at')
    
    # Get recent alerts across all models
    recent_alerts = Alert.objects.filter(
        timestamp__gte=timezone.now() - timedelta(hours=24)
    ).order_by('-timestamp')[:10]
    
    context = {
        'models': models,
        'recent_alerts': recent_alerts,
    }
    return render(request, 'dashboard/index.html', context)

def model_detail(request, model_id):
    """Detailed view for a specific model"""
    model = get_object_or_404(ModelRegistry, id=model_id)
    
    # Get recent predictions (reduced from 50 to 20 for better performance)
    recent_predictions = Prediction.objects.filter(
        model=model
    ).order_by('-timestamp')[:20]
    
    # Get latest metrics
    try:
        latest_performance = PerformanceMetric.objects.filter(
            model=model
        ).latest('timestamp')
    except PerformanceMetric.DoesNotExist:
        latest_performance = None
        
    try:
        latest_drift = DriftMetric.objects.filter(
            model=model
        ).latest('timestamp')
    except DriftMetric.DoesNotExist:
        latest_drift = None
    
    context = {
        'model': model,
        'recent_predictions': recent_predictions,
        'latest_performance': latest_performance,
        'latest_drift': latest_drift,
    }
    return render(request, 'dashboard/model_detail.html', context)

def data_upload(request):
    """View for data upload and processing"""
    return render(request, 'dashboard/data_upload.html')

def websocket_test(request):
    """Simple view for testing websocket connections"""
    return JsonResponse({'status': 'WebSocket endpoint ready'})