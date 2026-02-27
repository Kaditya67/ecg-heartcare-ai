import os
import csv
from django.core.management.base import BaseCommand
from api.models import ECGRecord, ECGLabel
from django.db.models import Q

class Command(BaseCommand):
    help = 'Curates high-value records for manual labeling based on AI confidence and model disagreement.'

    def add_arguments(self, parser):
        parser.add_argument('--threshold', type=float, default=0.7, help='Confidence threshold (default: 0.7)')
        parser.add_argument('--limit', type=int, default=100, help='Max records to export (default: 100)')
        parser.add_argument('--output', type=str, default='high_value_records.csv', help='Output CSV filename')

    def handle(self, *args, **options):
        threshold = options['threshold']
        limit = options['limit']
        output_file = options['output']

        self.stdout.write(self.style.SUCCESS(f'Scanning for high-value records (Confidence < {threshold})...'))

        # 1. Fetch records where AI is unsure (Low Confidence)
        # and ignore already verified ones
        low_conf_qs = ECGRecord.objects.filter(
            is_verified=False,
            ai_confidence__lt=threshold,
            ai_confidence__isnull=False
        ).order_by('ai_confidence')[:limit]

        # 2. Fetch records with NO AI label yet (Untouched)
        unprocessed_qs = ECGRecord.objects.filter(
            ai_label__isnull=True,
            is_verified=False
        ).order_by('?')[:limit//2]  # Add some random variety

        combined_records = list(low_conf_qs) + list(unprocessed_qs)
        
        if not combined_records:
            self.stdout.write(self.style.WARNING('No low-confidence records found. Ensure you have run Auto-Labeling first.'))
            return

        # Export to CSV
        with open(output_file, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Record ID', 'Patient ID', 'AI Label', 'AI Confidence', 'Heart Rate', 'Reason'])
            
            for record in combined_records:
                reason = "Low Confidence" if record.ai_confidence and record.ai_confidence < threshold else "Untouched"
                writer.writerow([
                    record.id,
                    record.patient_id,
                    record.ai_label.name if record.ai_label else "N/A",
                    f"{record.ai_confidence:.4f}" if record.ai_confidence else "N/A",
                    record.heart_rate,
                    reason
                ])

        self.stdout.write(self.style.SUCCESS(f'Successfully exported {len(combined_records)} records to {output_file}'))
        self.stdout.write(self.style.MIGRATE_LABEL('These records are "High-Value" — labeling them will significantly improve your next model release.'))
