from django.contrib.auth.models import User
from rest_framework import serializers
from rest_framework.reverse import reverse
from .models import ECGRecord, ECGFile, ECGLabel


class ECGLabelSerializer(serializers.ModelSerializer):
    class Meta:
        model = ECGLabel
        fields = ['id', 'name', 'value', 'color']


class ECGRecordSerializer(serializers.ModelSerializer):
    """Lightweight serializer — used in the paginated list (no wave data)."""
    label = ECGLabelSerializer(read_only=True)
    ai_label = ECGLabelSerializer(read_only=True)

    class Meta:
        model = ECGRecord
        fields = ["id", "patient_id", "heart_rate", "label", "labeled_by",
                  "is_verified", "ai_label", "ai_model_name"]


class ECGRecordDetailSerializer(serializers.ModelSerializer):
    """Full serializer including ECG wave — used for the plot view."""
    label = ECGLabelSerializer(read_only=True)
    ai_label = ECGLabelSerializer(read_only=True)

    class Meta:
        model = ECGRecord
        fields = ["id", "patient_id", "heart_rate", "label", "labeled_by",
                  "is_verified", "ai_label", "ai_model_name", "ecg_wave"]


class ECGWaveSerializer(serializers.ModelSerializer):
    class Meta:
        model = ECGRecord
        fields = ["ecg_wave"]


class ECGFileSerializer(serializers.ModelSerializer):
    record_count = serializers.IntegerField(source="total_records", read_only=True)
    download_csv_url = serializers.SerializerMethodField()
    download_xlsx_url = serializers.SerializerMethodField()

    class Meta:
        model = ECGFile
        fields = [
            "id", "file_name", "uploaded_at", "record_count",
            "download_csv_url", "download_xlsx_url",
        ]

    def get_download_csv_url(self, obj):
        request = self.context.get("request")
        return reverse('ecgfile-download-records-csv', kwargs={'pk': obj.pk}, request=request)

    def get_download_xlsx_url(self, obj):
        request = self.context.get("request")
        return reverse('ecgfile-download-records-xlsx', kwargs={'pk': obj.pk}, request=request)


class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)

    class Meta:
        model = User
        fields = ['username', 'password']

    def create(self, validated_data):
        return User.objects.create_user(
            username=validated_data['username'],
            password=validated_data['password']
        )
