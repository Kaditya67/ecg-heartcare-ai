from django.db import models

class ECGFile(models.Model):
    file_name = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(
        max_length=50,
        choices=[("processing", "Processing"), ("completed", "Completed"), ("failed", "Failed")],
        default="processing"
    )
    total_records = models.IntegerField(default=0)

    def __str__(self):
        return f"{self.file_name} ({self.status})"


class ECGRecord(models.Model):
    STATUS_CHOICES = [
        ("untouched", "Untouched"),
        ("skipped", "Skipped"),
        ("marked", "Marked"),
        ("deleted", "Deleted"),
    ]

    file = models.ForeignKey(ECGFile, on_delete=models.CASCADE, related_name="records")

    patient_id = models.CharField(max_length=100, db_index=True)
    ecg_wave = models.JSONField()  # Stores list/array of 2604 points
    heart_rate = models.FloatField()
    label = models.IntegerField(null=True, blank=True)  # ✅ numerical for ML

    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="untouched")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["patient_id"]),
            models.Index(fields=["status"]),
            models.Index(fields=["label"]),
        ]

    def __str__(self):
        return f"Patient {self.patient_id} - Status: {self.status}"
