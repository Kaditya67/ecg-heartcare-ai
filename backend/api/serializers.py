from rest_framework import serializers
from .models import ECGRecord, ECGFile 
from .models import ECGLabel, ECGRecord

class ECGLabelSerializer(serializers.ModelSerializer):
    class Meta:
        model = ECGLabel
        fields = ['id', 'name', 'value', 'color']

class ECGRecordSerializer(serializers.ModelSerializer):
    label = ECGLabelSerializer(read_only=True)

    class Meta:
        model = ECGRecord
        fields = ["id", "patient_id", "heart_rate", "label"]


class ECGRecordDetailSerializer(serializers.ModelSerializer):
    class Meta:
        model = ECGRecord
        fields = ["id", "patient_id", "heart_rate", "label", "ecg_wave"]

class ECGWaveSerializer(serializers.ModelSerializer):
    class Meta:
        model = ECGRecord
        fields = ["ecg_wave"]

class ECGFileSerializer(serializers.ModelSerializer):
    record_count = serializers.IntegerField(source="total_records", read_only=True)

    class Meta:
        model = ECGFile
        fields = ["id", "file_name", "uploaded_at", "record_count"]
