from rest_framework import serializers
from .models import ECGRecord, ECGFile, ECGLabel

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

from rest_framework import serializers
from rest_framework.reverse import reverse
from .models import ECGFile

class ECGFileSerializer(serializers.ModelSerializer):
    record_count = serializers.IntegerField(source="total_records", read_only=True)
    download_csv_url = serializers.SerializerMethodField()
    download_xlsx_url = serializers.SerializerMethodField()

    class Meta:
        model = ECGFile
        fields = [
            "id",
            "file_name",
            "uploaded_at",
            "record_count",
            "download_csv_url",
            "download_xlsx_url",
        ]

    def get_download_csv_url(self, obj):
        request = self.context.get("request")
        return reverse(
            'ecgfile-download-records-csv', kwargs={'pk': obj.pk}, request=request
        )

    def get_download_xlsx_url(self, obj):
        request = self.context.get("request")
        return reverse(
            'ecgfile-download-records-xlsx', kwargs={'pk': obj.pk}, request=request
        )
