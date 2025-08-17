from django.db import models

class ECGLabel(models.Model):
    value = models.IntegerField(unique=True)  # numerical value, e.g., 0,1,2
    name = models.CharField(max_length=100)  # friendly display name, e.g., "Normal"
    color = models.CharField(max_length=7, default="#000000")  # Hex color code like "#22c55e"
    description = models.TextField(blank=True, null=True)  # Optional description
    
    def __str__(self):
        return f"{self.name} ({self.value})"

class ECGFile(models.Model):
    file_name = models.CharField(max_length=255, unique=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(
        max_length=50,
        choices=[("processing", "Processing"), ("completed", "Completed"), ("failed", "Failed")],
        default="processing",
    )
    total_records = models.IntegerField(default=0)

    def __str__(self):
        return f"{self.file_name} ({self.status})"

class ECGRecord(models.Model):
    STATUS_CHOICES = [
        ("untouched", "Untouched"),
        ("skipped", "Skipped"),
        ("deleted", "Deleted"),
    ]

    file = models.ForeignKey(ECGFile, on_delete=models.CASCADE, related_name="records")

    patient_id = models.CharField(max_length=100, db_index=True)
    ecg_wave = models.JSONField()  # Stores list/array of 2604 points
    heart_rate = models.FloatField()
    label = models.ForeignKey(ECGLabel, null=True, blank=True, on_delete=models.SET_NULL)  # FK to label table
    
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="untouched")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["patient_id"]),
            models.Index(fields=["status"]),
            models.Index(fields=["label"]),
        ]

    def __str__(self):
        label_name = self.label.name if self.label else "No Label"
        return f"Patient {self.patient_id} - Status: {self.status} - Label: {label_name}"

from django.contrib.auth.models import User
from django.db import models

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    is_authorized = models.BooleanField(default=False)  # Admins set this

    def __str__(self):
        return f"{self.user.username} Profile"
