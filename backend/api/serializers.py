from rest_framework import serializers
from .models import ECGFile, ECGRecord

class ECGRecordSerializer(serializers.ModelSerializer):
    class Meta:
        model = ECGRecord
        fields = "__all__"

class ECGFileSerializer(serializers.ModelSerializer):
    records = ECGRecordSerializer(many=True, read_only=True)

    class Meta:
        model = ECGFile
        fields = "__all__"
