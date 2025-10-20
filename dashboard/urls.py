from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('models/<uuid:model_id>/', views.model_detail, name='model-detail'),
    path('data-upload/', views.data_upload, name='data-upload'),
    path('ws/dashboard/', views.websocket_test, name='websocket-test'),
]