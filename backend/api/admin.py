from django.contrib import admin
from .models import ECGFile, ECGRecord, ECGLabel, Profile, DeployedModel, PatientModelAssignment

# Register your models here.
admin.site.site_header = "ECG Labeling System Admin"
admin.site.site_title = "ECG Labeling System Admin Portal"

admin.site.index_title = "Welcome to the ECG Labeling System Admin Portal"

admin.site.register(ECGFile)
admin.site.register(ECGRecord)
admin.site.register(ECGLabel)
admin.site.register(DeployedModel)
admin.site.register(PatientModelAssignment)


@admin.register(Profile)
class ProfileAdmin(admin.ModelAdmin):
    list_display = ("user", "role", "patient_id", "is_authorized")
    list_filter = ("role", "is_authorized")
    search_fields = ("user__username", "patient_id")
