from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, generics, viewsets
from rest_framework.decorators import api_view, action, permission_classes, authentication_classes
from rest_framework.permissions import AllowAny, IsAuthenticated, IsAdminUser
from rest_framework_simplejwt.tokens import RefreshToken
from django.db import transaction, IntegrityError
from django.db.models import Count
from django.contrib.auth import authenticate
from collections import defaultdict
import pandas as pd
import io
from django.http import HttpResponse
import json
from django.contrib.auth.models import User

from .models import ECGFile, ECGRecord, ECGLabel
from .serializers import (
    ECGRecordSerializer,
    ECGFileSerializer,
    ECGWaveSerializer,
    ECGRecordDetailSerializer,
    RegisterSerializer,
)
from .utils.redis_client import set_ecg_wave, get_ecg_wave

from rest_framework.parsers import JSONParser

# ---------------------------
# File Upload
# ---------------------------
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.db import transaction, IntegrityError
import pandas as pd
from .models import ECGFile, ECGRecord
from .serializers import ECGFileSerializer

# Define canonical column names and their possible aliases
COLUMN_ALIASES = {
    "patient_id": ["patient_id", "PatientID", "patientID", "PatientId"],
    "ecg_wave": ["ecgWave", "ValueStr", "ECG", "Ecgwave"],
    "heart_rate": ["Heartrate", "Value", "HeartRate", "HR"],
    "label": ["Label", "label", "Diagnosis", "Class"],
}


def normalize_columns(df):
    """
    Rename columns in DataFrame to canonical column names if a match for any aliases is found.
    """
    new_columns = {}
    for canonical_col, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns:
                new_columns[alias] = canonical_col
                break
    df = df.rename(columns=new_columns)
    return df


class FileUploadView(APIView):
    # permission_classes = [IsAuthenticated]

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
                return Response({"error": "Invalid file format, expected CSV or Excel file."}, status=status.HTTP_400_BAD_REQUEST)

            # Normalize columns to canonical form
            df = normalize_columns(df)

            required_cols = {"patient_id", "ecg_wave", "heart_rate", "label"}
            if not required_cols.issubset(df.columns):
                missing = required_cols - set(df.columns)
                return Response({"error": f"Missing required columns after normalization: {missing}"}, status=400)

            records = [
                ECGRecord(
                    file=ecg_file,
                    patient_id=row["patient_id"],
                    ecg_wave=row["ecg_wave"],  # expected format string/list
                    heart_rate=row["heart_rate"],
                    label=int(row["label"]) if not pd.isna(row["label"]) else None,
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
from django_filters import rest_framework as filters
from .models import ECGRecord

class ECGRecordFilter(filters.FilterSet):
    # Use exact param name so frontend and backend match perfectly
    file__file_name__in = filters.BaseInFilter(field_name='file__file_name', lookup_expr='in')

    class Meta:
        model = ECGRecord
        fields = {
            'patient_id': ['exact'],
            'label__value': ['exact'],
            # Do NOT put file__file_name__in or other non-model fields here
        }


from rest_framework import generics
from .serializers import ECGRecordSerializer

class ECGRecordListView(generics.ListAPIView):
    serializer_class = ECGRecordSerializer
    queryset = ECGRecord.objects.all()
    filter_backends = [filters.DjangoFilterBackend]
    filterset_class = ECGRecordFilter

    def list(self, request, *args, **kwargs):
        include_wave = request.query_params.get("include_wave", "false").lower() == "true"
        page = self.paginate_queryset(self.filter_queryset(self.get_queryset()))

        if not include_wave:
            serializer = self.get_serializer(page, many=True)
            return self.get_paginated_response(serializer.data)

        data_with_wave = []
        for record in page:
            ecg_wave = get_ecg_wave(record.id)
            if ecg_wave is None:
                ecg_wave = record.ecg_wave
                set_ecg_wave(record.id, ecg_wave, timeout=120)

            label_obj = record.label
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
                "ecg_wave": ecg_wave,
            })

        return self.get_paginated_response(data_with_wave)


class ECGRecordDetailView(generics.RetrieveAPIView):
    queryset = ECGRecord.objects.all()
    serializer_class = ECGRecordDetailSerializer
    # permission_classes = [IsAuthenticated]


# ---------------------------
# Fetch ECG Wave (per record, uses cache)
# ---------------------------
@api_view(["GET"])
# @permission_classes([IsAuthenticated])
def get_ecg_wave_view(request, record_id):
    patient_id = request.query_params.get("patient_id")

    try:
        record = ECGRecord.objects.get(id=record_id)
    except ECGRecord.DoesNotExist:
        return Response({"error": "Record not found"}, status=status.HTTP_404_NOT_FOUND)

    if patient_id and str(record.patient_id) != str(patient_id):
        return Response({"error": "Patient ID does not match record"}, status=status.HTTP_400_BAD_REQUEST)

    ecg_wave = get_ecg_wave(record_id)
    if ecg_wave is None:
        ecg_wave = record.ecg_wave
        set_ecg_wave(record_id, ecg_wave, timeout=120)

    serializer = ECGWaveSerializer(record)
    ecg_wave_data = serializer.data["ecg_wave"]

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
        "label": record.label.value if record.label else None,
        "ecg_wave": ecg_wave_data,
        "source": "cache" if ecg_wave else "db",
        "label_options": label_options,
    })


# ---------------------------
# Bulk Label Update
# ---------------------------
@api_view(["POST"])
# @permission_classes([IsAuthenticated])
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
@permission_classes([AllowAny])  # Allow public access to list/retrieve/delete files as example
class ECGFileViewSet(viewsets.ModelViewSet):
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
    @permission_classes([AllowAny])
    @authentication_classes([])
    def download_records_csv(self, request, pk=None):
        ecg_file = self.get_object()
        records_qs = ECGRecord.objects.filter(file=ecg_file).order_by('created_at')

        data = [{
            'ID': idx + 1,
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
    @permission_classes([AllowAny])
    @authentication_classes([])
    def download_records_xlsx(self, request, pk=None):
        ecg_file = self.get_object()
        records_qs = ECGRecord.objects.filter(file=ecg_file).order_by('created_at')

        data = [{
            'ID': idx + 1,
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


# ---------------------------
# Additional APIViews for Summaries and Counts
# ---------------------------
class ECGFileListView(APIView):
    # permission_classes = [IsAuthenticated]

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
    # permission_classes = [IsAuthenticated]
    # parser_classes = [JSONParser]

    def post(self, request):
        file_names = request.data.get('files', [])
        if not file_names:
            return Response({"error": "No files specified."}, status=status.HTTP_400_BAD_REQUEST)

        records = ECGRecord.objects.filter(file__file_name__in=file_names) \
            .exclude(label__name='No Label') \
            .exclude(label__isnull=True)

        label_counts_qs = records.values('label_id', 'label__name') \
            .annotate(count=Count('id')) \
            .order_by('label__name')
        label_counts = [
            {
                'id': item['label_id'],
                'name': item['label__name'],
                'count': item['count']
            }
            for item in label_counts_qs
        ]

        patient_counts_qs = records.values('patient_id') \
            .annotate(record_count=Count('id')) \
            .order_by('patient_id')
        patient_counts = [
            {
                'patient_id': item['patient_id'],
                'record_count': item['record_count']
            }
            for item in patient_counts_qs
        ]

        return Response({'labels': label_counts, 'patients': patient_counts})


class ECGCustomExportView(APIView):
    # permission_classes = [IsAuthenticated]
    # parser_classes = [JSONParser]

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
                content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            )
            response['Content-Disposition'] = 'attachment; filename=ecg_custom_export.xlsx'
            return response
        else:
            csv_data = df.to_csv(index=False)
            response = HttpResponse(csv_data, content_type='text/csv')
            response['Content-Disposition'] = 'attachment; filename=ecg_custom_export.csv'
            return response


class LabelCountView(APIView):
    # permission_classes = [IsAuthenticated]

    def get(self, request):
        labels = ECGLabel.objects.all().values('id', 'name')
        return Response(list(labels))


class PatientCountView(APIView):
    # permission_classes = [IsAuthenticated]

    def get(self, request):
        patients = ECGRecord.objects.values('patient_id').annotate(record_count=Count('id')).order_by('patient_id')
        response = [{'patient_id': p['patient_id'], 'record_count': p['record_count']} for p in patients]
        return Response(response)


class ECGFileSummaryView(APIView):
    # permission_classes = [IsAuthenticated]

    def get(self, request):
        try:
            files = ECGFile.objects.all().annotate(record_count=Count('records'))
            file_summaries = []
            for f in files:
                records = ECGRecord.objects.filter(file=f)
                label_counts_qs = records.values('label_id', 'label__name').annotate(count=Count('id'))

                label_counts = [
                    {
                        'label_id': lc['label_id'],
                        'label_name': lc['label__name'],
                        'count': lc['count']
                    }
                    for lc in label_counts_qs if lc['label__name'] and lc['label__name'] != 'No Label'
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

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class DashboardSummaryView(APIView):
    # permission_classes = [IsAuthenticated]

    def get(self, request):
        try:
            total_files = ECGFile.objects.count()
            total_records = ECGRecord.objects.count()
            labelled_records = ECGRecord.objects.exclude(label=None).count()
            normal_records = ECGRecord.objects.filter(label__name="Normal").count()

            labels_data_qs = ECGLabel.objects.annotate(record_count=Count('ecgrecord')).values('name', 'record_count')
            labels_data = {item['name']: item['record_count'] for item in labels_data_qs}

            return Response({
                "total_files": total_files,
                "total_records": total_records,
                "labelled_records": labelled_records,
                "normal_records": normal_records,
                "labels_data": labels_data,
            }, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class AuthorizeUserView(APIView):
    permission_classes = [IsAdminUser]

    def post(self, request):
        username = request.data.get('username')
        try:
            user = User.objects.get(username=username)
            user.profile.is_authorized = True
            user.profile.save()
            return Response({"authorized": True})
        except User.DoesNotExist:
            return Response({"error": "User does not exist"}, status=status.HTTP_404_NOT_FOUND)


from rest_framework.views import APIView
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework import status
from django.contrib.auth import authenticate
from rest_framework_simplejwt.tokens import RefreshToken
from .serializers import RegisterSerializer

class RegisterView(APIView):
    permission_classes = [AllowAny]  # Anyone can register
    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response({'msg': 'User registered successfully!'}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class LoginView(APIView):
    permission_classes = [AllowAny]  # Anyone can login
    def post(self, request):
        username = request.data.get('username')
        password = request.data.get('password')
        user = authenticate(username=username, password=password)
        if user is not None:
            refresh = RefreshToken.for_user(user)
            return Response({
                'refresh': str(refresh),
                'access': str(refresh.access_token),
            })
        return Response({'error': 'Invalid credentials'}, status=status.HTTP_401_UNAUTHORIZED)
