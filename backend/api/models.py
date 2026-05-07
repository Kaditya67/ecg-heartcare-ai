from django.db import models
from django.contrib.auth.models import User

class ECGLabel(models.Model):
    value = models.IntegerField(unique=True)  # numerical value, e.g., 0,1,2
    name = models.CharField(max_length=100)  # friendly display name, e.g., "Normal"
    color = models.CharField(max_length=7, default="#000000")  # Hex color code like "#22c55e"
    description = models.TextField(blank=True, null=True)  # Optional description
    
    def __str__(self):
        return f"{self.name} ({self.value})"


class DeployedModel(models.Model):
    SOURCE_BUILTIN = "builtin"
    SOURCE_UPLOADED = "uploaded"
    SOURCE_CHOICES = [
        (SOURCE_BUILTIN, "Built In"),
        (SOURCE_UPLOADED, "Uploaded"),
    ]

    key = models.CharField(max_length=100, unique=True)
    label = models.CharField(max_length=150)
    base_model_key = models.CharField(max_length=100)
    source_type = models.CharField(max_length=20, choices=SOURCE_CHOICES, default=SOURCE_BUILTIN)
    weights_path = models.CharField(max_length=255)
    input_size = models.IntegerField()
    num_classes = models.IntegerField()
    trainable = models.BooleanField(default=True)
    is_active = models.BooleanField(default=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=["key"]),
            models.Index(fields=["base_model_key"]),
            models.Index(fields=["source_type", "is_active"]),
        ]

    def __str__(self):
        return f"{self.label} [{self.key}]"


class PatientModelAssignment(models.Model):
    patient_id = models.CharField(max_length=100, unique=True, db_index=True)
    model = models.ForeignKey(DeployedModel, on_delete=models.CASCADE, related_name="patient_assignments")
    assigned_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=["patient_id"]),
        ]

    def __str__(self):
        return f"Patient {self.patient_id} -> {self.model.key}"

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
    LABELED_BY_CHOICES = [
        ("import", "Imported"),
        ("human", "Human"),
        ("ai", "AI"),
    ]

    file = models.ForeignKey(ECGFile, on_delete=models.CASCADE, related_name="records")

    patient_id = models.CharField(max_length=100, db_index=True)
    ecg_wave = models.JSONField()  # Stores list/array of 2604 points
    heart_rate = models.FloatField()

    # ── Human label (manually set by annotator) ───────────────────────────
    label = models.ForeignKey(
        ECGLabel, null=True, blank=True, on_delete=models.SET_NULL,
        related_name="records"
    )
    labeled_by = models.CharField(
        max_length=10, choices=LABELED_BY_CHOICES, default="import"
    )
    is_verified = models.BooleanField(
        default=False,
        help_text="True means a human has confirmed this label is correct."
    )

    # ── AI-predicted label (never overwrites manual label) ────────────────
    ai_label = models.ForeignKey(
        ECGLabel, null=True, blank=True, on_delete=models.SET_NULL,
        related_name="ai_records",
        help_text="Label predicted by the AI model. Stored separately from the human label."
    )
    ai_model_name = models.CharField(
        max_length=100, null=True, blank=True,
        help_text="Name of the model that produced ai_label."
    )
    ai_confidence = models.FloatField(
        null=True, blank=True,
        help_text="Confidence core (e.g., max probability) for the AI prediction."
    )
    ai_probabilities = models.JSONField(
        null=True, blank=True,
        help_text="Full probability distribution across all labels."
    )

    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="untouched")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["patient_id"]),
            models.Index(fields=["status"]),
            models.Index(fields=["label"]),
            models.Index(fields=["is_verified"]),
            models.Index(fields=["ai_label"]),
        ]

    def __str__(self):
        label_name = self.label.name if self.label else "No Label"
        return f"Patient {self.patient_id} - Label: {label_name} - Verified: {self.is_verified}"
class Profile(models.Model):
    ROLE_ADMIN = "admin"
    ROLE_DOCTOR = "doctor"
    ROLE_PATIENT = "patient"
    ROLE_CHOICES = [
        (ROLE_ADMIN, "Admin"),
        (ROLE_DOCTOR, "Doctor"),
        (ROLE_PATIENT, "Patient"),
    ]

    user = models.OneToOneField(User, on_delete=models.CASCADE)
    is_authorized = models.BooleanField(default=False)  # Admins set this
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default=ROLE_DOCTOR)
    patient_id = models.CharField(
        max_length=100,
        blank=True,
        null=True,
        help_text="Required when the user is a patient. Limits access to this patient's records.",
    )
    show_missing_weights = models.BooleanField(
        default=True,
        help_text="Show models that do not have weights available in the UI.",
    )
    default_plot_library = models.CharField(
        max_length=50,
        default="echarts",
        help_text="Default plotting library to use for ECG charts.",
    )

    @property
    def effective_role(self):
        if self.user.is_superuser or self.user.is_staff:
            return self.ROLE_ADMIN
        return self.role

    def save(self, *args, **kwargs):
        if self.effective_role != self.ROLE_PATIENT:
            self.patient_id = None
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.user.username} Profile ({self.effective_role})"
