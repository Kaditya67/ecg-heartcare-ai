from django.contrib import admin
from .models import ECGFile, ECGRecord, ECGLabel, Profile

# Register your models here.
admin.site.site_header = "ECG Labeling System Admin"
admin.site.site_title = "ECG Labeling System Admin Portal"

admin.site.index_title = "Welcome to the ECG Labeling System Admin Portal"

admin.site.register(ECGFile)    
admin.site.register(ECGRecord)
admin.site.register(ECGLabel)
admin.site.register(Profile)