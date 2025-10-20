from django.core.management.base import BaseCommand
from api.models import ModelRegistry, Prediction
import random
import json
from datetime import datetime, timedelta
from django.utils import timezone

class Command(BaseCommand):
    help = 'Generate sample data for testing the monitoring system'

    def add_arguments(self, parser):
        parser.add_argument('--models', type=int, default=3, help='Number of models to create')
        parser.add_argument('--predictions', type=int, default=100, help='Number of predictions per model')

    def handle(self, *args, **options):
        models_count = options['models']
        predictions_count = options['predictions']
        
        # Create sample models
        models = []
        for i in range(models_count):
            model, created = ModelRegistry.objects.get_or_create(
                name=f'FraudDetectionModel_{i+1}',
                version=f'v1.{i+1}',
                defaults={
                    'description': f'Fraud detection model version {i+1}'
                }
            )
            models.append(model)
            if created:
                self.stdout.write(f'Created model: {model.name}')
            else:
                self.stdout.write(f'Using existing model: {model.name}')
        
        # Generate sample predictions for each model
        for model in models:
            self.stdout.write(f'Generating {predictions_count} predictions for {model.name}...')
            
            for j in range(predictions_count):
                # Generate realistic sample data
                prediction_data = {
                    'probability': random.uniform(0.01, 0.99),
                    'confidence': random.uniform(0.5, 1.0),
                    'class': 'fraud' if random.random() < 0.1 else 'legitimate'
                }
                
                input_data = {
                    'transaction_amount': round(random.uniform(10, 5000), 2),
                    'user_age': random.randint(18, 80),
                    'account_age_days': random.randint(1, 3650),
                    'num_transactions_today': random.randint(1, 20),
                    'device_type': random.choice(['mobile', 'desktop', 'tablet']),
                    'country_risk_score': random.uniform(0, 1)
                }
                
                # Randomly introduce some missing values
                if random.random() < 0.05:  # 5% chance of missing values
                    missing_field = random.choice(list(input_data.keys()))
                    input_data[missing_field] = None
                
                # Create prediction with some latency variation
                Prediction.objects.create(
                    model=model,
                    input_data=input_data,
                    prediction=prediction_data,
                    latency_ms=random.uniform(10, 200),
                    actual_value={'is_fraud': random.choice([0, 1])} if random.random() < 0.8 else None
                )
            
            self.stdout.write(
                self.style.SUCCESS(
                    f'Successfully generated {predictions_count} predictions for {model.name}'
                )
            )
        
        self.stdout.write(
            self.style.SUCCESS(
                f'Sample data generation complete! Created {models_count} models with {predictions_count} predictions each.'
            )
        )