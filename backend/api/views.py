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
    
# backend/views.py

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import JSONParser
from rest_framework import status
from django.db.models import Count
from collections import defaultdict
import pandas as pd
import io
from django.http import HttpResponse

from .models import ECGFile, ECGRecord, ECGLabel


class ECGFileListView(APIView):
    def get(self, request):
        files = ECGFile.objects.all().annotate(total_records=Count('records'))
        file_list = [
            {
                'file_name': f.file_name,
                'total_records': f.total_records,
                'uploaded_at': f.uploaded_at,
                'status': f.status
            }
            for f in files
        ]
        return Response(file_list)


class LabelsPatientsByFilesView(APIView):
    parser_classes = [JSONParser]

    def post(self, request):
        file_names = request.data.get('files', [])
        if not file_names:
            return Response({"error": "No files specified."}, status=status.HTTP_400_BAD_REQUEST)

        # Exclude records with label name 'No Label' or null label
        records = ECGRecord.objects.filter(file__file_name__in=file_names) \
            .exclude(label__name='No Label') \
            .exclude(label__isnull=True)

        # Label counts among these files excluding 'No Label' and null labels
        label_counts_qs = (
            records.values('label_id', 'label__name')
            .annotate(count=Count('id'))
            .order_by('label__name')
        )
        label_counts = [
            {
                'id': item['label_id'],
                'name': item['label__name'],
                'count': item['count']
            }
            for item in label_counts_qs
        ]

        # Patient counts among these files
        patient_counts_qs = (
            records.values('patient_id')
            .annotate(record_count=Count('id'))
            .order_by('patient_id')
        )
        patient_counts = [
            {
                'patient_id': item['patient_id'],
                'record_count': item['record_count']
            }
            for item in patient_counts_qs
        ]

        return Response({
            'labels': label_counts,
            'patients': patient_counts,
        })


class ECGCustomExportView(APIView):
    parser_classes = [JSONParser]

    def post(self, request):
        files = request.data.get('files', [])
        patients = request.data.get('patients', [])
        labels = request.data.get('labels', [])
        file_format = request.data.get('format', 'csv').lower()

        if not files:
            return Response({"error": "Must specify at least one file"}, status=status.HTTP_400_BAD_REQUEST)

        patient_ids = [p['patient_id'] for p in patients if p.get('numRecords', 0) > 0]
        label_ids = [l['id'] for l in labels if l.get('numRecords', 0) > 0]
        file_names = [f['file_name'] for f in files if f.get('numRecords', 0) > 0]

        queryset = ECGRecord.objects.filter(file__file_name__in=file_names)

        if patient_ids:
            queryset = queryset.filter(patient_id__in=patient_ids)
        if label_ids:
            queryset = queryset.filter(label_id__in=label_ids)

        patient_limits = {p['patient_id']: p.get('numRecords', 0) for p in patients}
        label_limits = {l['id']: l.get('numRecords', 0) for l in labels}
        file_limits = {f['file_name']: f.get('numRecords', 0) for f in files}

        patient_counts = defaultdict(int)
        label_counts = defaultdict(int)
        file_counts = defaultdict(int)

        data = []
        idx = 1

        for rec in queryset.order_by('file__file_name', 'patient_id', 'label_id', 'created_at'):
            if file_counts[rec.file.file_name] >= file_limits.get(rec.file.file_name, 0):
                continue
            if patient_limits and patient_counts[rec.patient_id] >= patient_limits.get(rec.patient_id, 0):
                continue
            label_id = rec.label_id if rec.label_id is not None else -1
            if label_limits and label_counts[label_id] >= label_limits.get(label_id, 0):
                continue

            data.append({
                "ID": idx,
                "Patient ID": rec.patient_id,
                "Heart Rate": rec.heart_rate,
                "ECG Wave": rec.ecg_wave if rec.ecg_wave else '',
                "Label": rec.label.value if rec.label else '',
            })
            idx += 1
            file_counts[rec.file.file_name] += 1
            patient_counts[rec.patient_id] += 1
            label_counts[label_id] += 1

        if not data:
            return Response({"error": "No records match the given filters."}, status=status.HTTP_400_BAD_REQUEST)

        df = pd.DataFrame(data)

        if file_format == 'xlsx':
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='ECG Records')
            output.seek(0)
            response = HttpResponse(
                output.read(),
                content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
            response['Content-Disposition'] = 'attachment; filename=ecg_custom_export.xlsx'
            return response

        else:  # CSV fallback
            csv_data = df.to_csv(index=False)
            response = HttpResponse(csv_data, content_type='text/csv')
            response['Content-Disposition'] = 'attachment; filename=ecg_custom_export.csv'
            return response

from rest_framework.views import APIView
from rest_framework.response import Response
from .models import ECGLabel  # import your model

class LabelCountView(APIView):
    def get(self, request):
        # Example logic; adapt as needed
        labels = ECGLabel.objects.all().values('id', 'name')
        return Response(list(labels))

from rest_framework.views import APIView
from rest_framework.response import Response
from .models import ECGRecord

class PatientCountView(APIView):
    def get(self, request):
        patients = ECGRecord.objects.values('patient_id').annotate(record_count=Count('id')).order_by('patient_id')
        response = [
            {'patient_id': p['patient_id'], 'record_count': p['record_count']}
            for p in patients
        ]
        return Response(response)

class ECGFileSummaryView(APIView):
    def get(self, request):
        print("Fetching ECG file summaries...")
        files = ECGFile.objects.all().annotate(record_count=Count('records'))
        file_summaries = []
        for f in files:
            records = ECGRecord.objects.filter(file=f)
            label_counts_qs = records.values('label_id', 'label__name').annotate(count=Count('id'))

            # Filter out label entries named "No Label"
            label_counts = [
                {
                    'label_id': lc['label_id'],
                    'label_name': lc['label__name'],
                    'count': lc['count'],
                }
                for lc in label_counts_qs
                if lc['label__name'] and lc['label__name'] != 'No Label'
            ]

            patient_count = records.values('patient_id').distinct().count()

            file_summaries.append({
                'file_name': f.file_name,
                'uploaded_at': f.uploaded_at,
                'status': f.status,
                'total_records': f.record_count,
                'label_counts': label_counts,
                'unique_patient_count': patient_count,
            })
        return Response(file_summaries)
