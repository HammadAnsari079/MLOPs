from django.core.management.base import BaseCommand
from monitoring.tasks import calculate_metrics_task, detect_drift_task, check_data_quality_task
from api.models import ModelRegistry
import time

class Command(BaseCommand):
    help = 'Run monitoring tasks for all models'

    def add_arguments(self, parser):
        parser.add_argument('--model-id', type=str, help='Specific model ID to monitor (default: all models)')

    def handle(self, *args, **options):
        model_id = options['model_id']
        
        # Get models to process
        if model_id:
            try:
                models = ModelRegistry.objects.filter(id=model_id)
                if not models.exists():
                    self.stdout.write(
                        self.style.ERROR(f'No model found with ID: {model_id}')
                    )
                    return
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f'Error finding model with ID {model_id}: {str(e)}')
                )
                return
        else:
            models = ModelRegistry.objects.all()
        
        if not models.exists():
            self.stdout.write(
                self.style.WARNING('No models found to monitor')
            )
            return
        
        self.stdout.write(
            self.style.SUCCESS(f'Starting monitoring for {models.count()} models...')
        )
        
        # Run monitoring tasks
        for model in models:
            self.stdout.write(f'Processing model: {model.name} (ID: {model.id})')
            
            # Calculate performance metrics
            self.stdout.write('  - Calculating performance metrics...')
            try:
                result = calculate_metrics_task(str(model.id))
                self.stdout.write(f'    Result: {result}')
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f'    Error calculating metrics: {str(e)}')
                )
            
            # Check data quality
            self.stdout.write('  - Checking data quality...')
            try:
                result = check_data_quality_task(str(model.id))
                self.stdout.write(f'    Result: {result}')
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f'    Error checking data quality: {str(e)}')
                )
            
            # Detect drift
            self.stdout.write('  - Detecting drift...')
            try:
                result = detect_drift_task(str(model.id))
                self.stdout.write(f'    Result: {result}')
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f'    Error detecting drift: {str(e)}')
                )
            
            self.stdout.write('')
        
        self.stdout.write(
            self.style.SUCCESS('Monitoring tasks completed!')
        )