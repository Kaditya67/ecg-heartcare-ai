from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, generics
from django.db import transaction
import pandas as pd
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework.decorators import api_view
from .models import ECGFile, ECGRecord
from .serializers import ECGRecordSerializer, ECGFileSerializer, ECGWaveSerializer, ECGRecordDetailSerializer
from .utils.redis_client import set_ecg_wave, get_ecg_wave, preload_page_waves
import json

# ---------------------------
# File Upload
# ---------------------------
class FileUploadView(APIView):
    def post(self, request, format=None):
        file = request.FILES.get("file")
        if not file:
            return Response({"error": "No file provided"}, status=status.HTTP_400_BAD_REQUEST)

        ecg_file = ECGFile.objects.create(file_name=file.name, status="processing")

        try:
            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
            elif file.name.endswith((".xls", ".xlsx")):
                df = pd.read_excel(file)
            else:
                return Response({"error": "Invalid file format"}, status=status.HTTP_400_BAD_REQUEST)

            required_cols = {"patient_id", "ecgWave", "Heartrate", "Label"}
            if not required_cols.issubset(df.columns):
                return Response({"error": f"Missing columns. Required: {required_cols}"}, status=400)

            records = [
                ECGRecord(
                    file=ecg_file,
                    patient_id=row["patient_id"],
                    ecg_wave=row["ecgWave"],   # Assume JSON/list-like
                    heart_rate=row["Heartrate"],
                    label=int(row["Label"]) if not pd.isna(row["Label"]) else None,
                )
                for _, row in df.iterrows()
            ]

            with transaction.atomic():
                ECGRecord.objects.bulk_create(records, batch_size=5000)

            ecg_file.total_records = len(records)
            ecg_file.status = "completed"
            ecg_file.save()

            return Response(ECGFileSerializer(ecg_file).data, status=status.HTTP_201_CREATED)

        except Exception as e:
            ecg_file.status = "failed"
            ecg_file.save()
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ---------------------------
# Paginated Records (with optional wave caching)
# ---------------------------
class ECGRecordListView(generics.ListAPIView):
    serializer_class = ECGRecordSerializer
    queryset = ECGRecord.objects.all()

    def list(self, request, *args, **kwargs):
        include_wave = request.query_params.get("include_wave", "false").lower() == "true"
        page = self.paginate_queryset(self.get_queryset())

        # Case 1: metadata only
        if not include_wave:
            serializer = self.get_serializer(page, many=True)
            return self.get_paginated_response(serializer.data)

        # Case 2: include_wave=true → use Redis
        data_with_wave = []
        for record in page:
            ecg_wave = get_ecg_wave(record.id)
            if ecg_wave is None:
                ecg_wave = record.ecg_wave
                set_ecg_wave(record.id, ecg_wave, timeout=120)

            data_with_wave.append({
                "patient_id": record.patient_id,
                "heart_rate": record.heart_rate,
                "label": record.label,
                "wave_length": len(ecg_wave),
                "ecg_wave": ecg_wave,  # Only when requested
            })

        return self.get_paginated_response(data_with_wave)

class ECGRecordDetailView(generics.RetrieveAPIView):
    queryset = ECGRecord.objects.all()
    serializer_class = ECGRecordDetailSerializer

# ---------------------------
# Fetch ECG Wave (per record, uses cache)
# ---------------------------
@api_view(["GET"])
def get_ecg_wave_view(request, record_id):
    ecg_wave = get_ecg_wave(record_id)
    if ecg_wave is not None:
        return Response({"id": record_id, "ecg_wave": ecg_wave, "source": "cache"})

    # 2. Fallback DB
    try:
        record = ECGRecord.objects.get(id=record_id)
    except ECGRecord.DoesNotExist:
        return Response({"error": "Record not found"}, status=status.HTTP_404_NOT_FOUND)

    serializer = ECGWaveSerializer(record)
    ecg_wave_data = serializer.data["ecg_wave"]

    # 3. Store in Redis
    set_ecg_wave(record_id, ecg_wave_data, timeout=120)

    return Response({"id": record_id, "ecg_wave": ecg_wave_data, "source": "db"})
