from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, generics
from django.db import transaction
import pandas as pd
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework.decorators import api_view
from .models import ECGFile, ECGRecord, ECGLabel
from .serializers import ECGRecordSerializer, ECGFileSerializer, ECGWaveSerializer, ECGRecordDetailSerializer
from .utils.redis_client import set_ecg_wave, get_ecg_wave, preload_page_waves
import json
from django.db import IntegrityError

# ---------------------------
# File Upload
# ---------------------------
class FileUploadView(APIView):
    def post(self, request, format=None):
        file = request.FILES.get("file")
        if not file:
            return Response({"error": "No file provided"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            ecg_file = ECGFile.objects.create(file_name=file.name, status="processing")
        except IntegrityError:
            return Response({"error": "A file with this name already exists."}, status=400)

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

        # Case 2: include_wave=true → use Redis and include serialized label info
        data_with_wave = []
        for record in page:
            ecg_wave = get_ecg_wave(record.id)
            if ecg_wave is None:
                ecg_wave = record.ecg_wave
                set_ecg_wave(record.id, ecg_wave, timeout=120)
            
            label_obj = record.label  # assuming FK to ECGLabel
            label_data = None
            if label_obj:
                label_data = {
                    "id": label_obj.id,
                    "name": label_obj.name,
                    "value": label_obj.value,
                    "color": label_obj.color,
                }

            data_with_wave.append({
                "id": record.id,
                "patient_id": record.patient_id,
                "heart_rate": record.heart_rate,
                "label": label_data,
                "wave_length": len(ecg_wave),
                "ecg_wave": ecg_wave,  # Only when include_wave=true
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
    patient_id = request.query_params.get("patient_id")

    try:
        record = ECGRecord.objects.get(id=record_id)
    except ECGRecord.DoesNotExist:
        return Response({"error": "Record not found"}, status=status.HTTP_404_NOT_FOUND)

    # Optional: validate patient_id
    if patient_id and str(record.patient_id) != str(patient_id):
        return Response({"error": "Patient ID does not match record"}, status=status.HTTP_400_BAD_REQUEST)

    ecg_wave = get_ecg_wave(record_id)
    if ecg_wave is None:
        ecg_wave = record.ecg_wave
        set_ecg_wave(record_id, ecg_wave, timeout=120)

    serializer = ECGWaveSerializer(record)
    ecg_wave_data = serializer.data["ecg_wave"]

    # --- Fetch all available label options ---
    all_labels = ECGLabel.objects.all().order_by('value')
    label_options = [
        {
            "id": label.id,
            "value": label.value,
            "name": label.name,
            "color": label.color,
            "description": label.description,
        }
        for label in all_labels
    ]

    return Response({
        "id": record_id,
        "patient_id": record.patient_id,
        "heart_rate": record.heart_rate,
        "label": record.label.value if record.label else None,  # Only value for ML, or send full label as dict if needed
        "ecg_wave": ecg_wave_data,
        "source": "cache" if ecg_wave else "db",
        "label_options": label_options,
    })

# ---------------------------
# Bulk Label Update
# ---------------------------

from django.db import transaction
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .models import ECGRecord, ECGLabel

@api_view(["POST"])
def bulk_label_update_view(request):
    """
    Bulk update labels for multiple ECG records.
    Expected POST payload:
    {
      "records": [
        {"id": <record_id>, "patient_id": <patient_id>, "label": <label_value or label_id>},
        ...
      ]
    }
    """
    records = request.data.get("records", [])
    if not isinstance(records, list):
        return Response({"error": "Invalid payload, 'records' must be a list."}, status=status.HTTP_400_BAD_REQUEST)

    unique_labels = set(rec.get("label") for rec in records if rec.get("label") is not None)
    label_map = {}
    if unique_labels:
        labels_qs = ECGLabel.objects.filter(value__in=unique_labels)
        label_map = {label.value: label for label in labels_qs}

    updated = []
    errors = []
    to_update = []

    with transaction.atomic():
        for rec in records:
            print(rec)
            record_id = rec.get("id")
            patient_id = rec.get("patient_id")
            label_value = rec.get("label")

            if record_id is None or label_value is None or patient_id is None:
                errors.append(f"Record id, patient_id, or label missing in entry: {rec}")
                continue

            try:
                ecg_record = ECGRecord.objects.get(id=record_id, patient_id=patient_id)
            except ECGRecord.DoesNotExist:
                errors.append(f"Record id {record_id} with patient_id {patient_id} does not exist")
                continue

            label_obj = label_map.get(label_value)
            if not label_obj:
                errors.append(f"Label ({label_value}) for record {record_id} does not exist")
                continue

            if ecg_record.label_id != label_obj.id:
                ecg_record.label = label_obj
                to_update.append(ecg_record)
                updated.append(record_id)

        if to_update:
            ECGRecord.objects.bulk_update(to_update, ['label'])

    if errors:
        return Response({"updated": updated, "errors": errors}, status=status.HTTP_207_MULTI_STATUS)

    return Response({"updated": updated}, status=status.HTTP_200_OK)


# ---------------------------
# ECGFile ViewSet
# ---------------------------
import io
import pandas as pd
from django.http import HttpResponse
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import ECGFile, ECGRecord
from .serializers import ECGFileSerializer, ECGRecordSerializer, ECGRecordDetailSerializer


class ECGFileViewSet(viewsets.ModelViewSet):  # <-- Changed here to allow DELETE
    """
    List, retrieve, delete ECG files with record counts.
    """
    queryset = ECGFile.objects.all().order_by('-uploaded_at')
    serializer_class = ECGFileSerializer

    @action(detail=True, methods=['get'])
    def records(self, request, pk=None):
        try:
            ecg_file = self.get_object()
        except ECGFile.DoesNotExist:
            return Response({"detail": "File not found."}, status=status.HTTP_404_NOT_FOUND)

        records = ECGRecord.objects.filter(file=ecg_file)
        serializer = ECGRecordSerializer(records, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['get'])
    def records_detail(self, request, pk=None):
        try:
            ecg_file = self.get_object()
        except ECGFile.DoesNotExist:
            return Response({"detail": "File not found."}, status=status.HTTP_404_NOT_FOUND)

        records = ECGRecord.objects.filter(file=ecg_file)
        serializer = ECGRecordDetailSerializer(records, many=True)
        return Response(serializer.data)
    
    @action(detail=True, methods=['get'], url_path='download_records/csv')
    def download_records_csv(self, request, pk=None):
        ecg_file = self.get_object()
        records_qs = ECGRecord.objects.filter(file=ecg_file).order_by('created_at')

        data = [{
            'ID': idx + 1,  # 1-based index
            'Patient ID': record.patient_id,
            'Heart Rate': record.heart_rate,
            'EcgWave': record.ecg_wave,
            'Label': record.label.value if record.label else ''
        } for idx, record in enumerate(records_qs)]

        df = pd.DataFrame(data)
        csv_data = df.to_csv(index=False)
        response = HttpResponse(csv_data, content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="{ecg_file.file_name}_records.csv"'
        return response

    @action(detail=True, methods=['get'], url_path='download_records/xlsx')
    def download_records_xlsx(self, request, pk=None):
        ecg_file = self.get_object()
        records_qs = ECGRecord.objects.filter(file=ecg_file).order_by('created_at')

        data = [{
            'ID': idx + 1,  # 1-based index
            'Patient ID': record.patient_id,
            'Heart Rate': record.heart_rate,
            'EcgWave': record.ecg_wave,
            'Label': record.label.value if record.label else ''
        } for idx, record in enumerate(records_qs)]

        df = pd.DataFrame(data)

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='ECG Records')
        output.seek(0)
        response = HttpResponse(
            output.read(),
            content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        response['Content-Disposition'] = f'attachment; filename="{ecg_file.file_name}_records.xlsx"'
        return response