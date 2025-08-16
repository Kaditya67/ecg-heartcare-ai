from rest_framework import serializers
from .models import ECGFile, ECGRecord

class ECGRecordSerializer(serializers.ModelSerializer):
    wave_length = serializers.SerializerMethodField()

    class Meta:
        model = ECGRecord
        fields = ["patient_id", "heart_rate", "label", "wave_length"]

    def get_wave_length(self, obj):
        return len(obj.ecg_wave)

class ECGFileSerializer(serializers.ModelSerializer):
    record_count = serializers.IntegerField(source="records.count", read_only=True)

    class Meta:
        model = ECGFile
        fields = ["id", "file_name", "uploaded_at", "record_count"]