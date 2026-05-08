import glob
import io
import json
import logging
import os
import re
import threading
import uuid
from collections import defaultdict
from pathlib import Path
from os import path as ospath

import gdown
import pandas as pd
import torch
from django.contrib.auth import authenticate
from django.contrib.auth.models import User
from django.db import OperationalError, transaction
from django.db.models import Count, Q
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework import generics, status, viewsets
from rest_framework.decorators import action, api_view, permission_classes
from rest_framework.parsers import JSONParser
from rest_framework.permissions import AllowAny, IsAdminUser, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.tokens import RefreshToken
from django_filters import rest_framework as filters

from .models import (
    DeployedModel,
    ECGFile,
    ECGLabel,
    ECGRecord,
    PatientModelAssignment,
    Profile,
)
from .Model_architechture.MLModels import (
    AdaBoostECGWrapper,
    ECGXGBoostWrapper,
    RandomForestECGWrapper,
)
from .models_loader import MODEL_MAP
from .permissions import IsAdminRole, IsDoctorOrAdmin, get_user_role
from .serializers import (DeployedModelSerializer, ECGFileSerializer,
                          ECGRecordDetailSerializer, ECGRecordSerializer,
                          ECGWaveSerializer, PatientModelAssignmentSerializer,
                          ProfileSerializer, RegisterSerializer, UserSerializer)
from .utils.redis_client import get_ecg_wave, set_ecg_wave

logger = logging.getLogger(__name__)
LOCAL_MODELS_FOLDER = "api/models"
UPLOADED_MODELS_FOLDER = "api/models/uploads"
os.makedirs(Path(Path(__file__).resolve().parent / "models"), exist_ok=True)
os.makedirs(Path(Path(__file__).resolve().parent / "models" / "uploads"), exist_ok=True)

# Define canonical column names and their possible aliases
COLUMN_ALIASES = {
    "patient_id": ["patient_id", "Patient ID", "patient id", "PatientID", "patientID", "PatientId"],
    "ecg_wave": ["ecg_wave", "ECG Wave", "ecg wave", "ecgWave", "ValueStr", "ECG", "Ecgwave"],
    "heart_rate": ["heart_rate", "Heart Rate", "heart rate", "Heartrate", "Value", "HeartRate", "HR"],
    "label": ["label", "Label", "Diagnosis", "Class"],
}
REQUIRED_UPLOAD_COLUMNS = {"patient_id", "ecg_wave", "heart_rate", "label"}


def normalize_column_key(value):
    normalized = str(value).replace("\ufeff", "").strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def load_tabular_file(uploaded_file):
    file_name = uploaded_file.name.lower()
    uploaded_file.seek(0)
    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if file_name.endswith((".xls", ".xlsx")):
        return pd.read_excel(uploaded_file)
    raise ValueError("Invalid file format, expected CSV or Excel file.")


def build_unique_file_name(file_name):
    base_name, extension = ospath.splitext(str(file_name))
    candidate = f"{base_name}{extension}"
    counter = 2

    while ECGFile.objects.filter(file_name=candidate).exists():
        candidate = f"{base_name}_{counter}{extension}"
        counter += 1

    return candidate


def build_upload_mapping_response(df, missing_columns):
    available_columns = [str(col) for col in df.columns]
    suggested_mapping = {}
    normalized_available = {normalize_column_key(col): str(col) for col in df.columns}

    for canonical_col, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            matched = normalized_available.get(normalize_column_key(alias))
            if matched:
                suggested_mapping[canonical_col] = matched
                break

    preview_rows = (
        df.head(5)
        .fillna("")
        .astype(str)
        .to_dict(orient="records")
    )

    return {
        "error": f"Missing required columns after normalization: {missing_columns}",
        "missing_required_columns": sorted(missing_columns),
        "available_columns": available_columns,
        "suggested_mapping": suggested_mapping,
        "preview_rows": preview_rows,
    }


def extract_column_mapping(request_data):
    raw_mapping = request_data.get("column_mapping")
    if raw_mapping:
        try:
            return json.loads(raw_mapping) if isinstance(raw_mapping, str) else raw_mapping
        except json.JSONDecodeError:
            raise ValueError("Invalid column_mapping payload.")

    fallback_mapping = {}
    for canonical_col in REQUIRED_UPLOAD_COLUMNS:
        mapped_value = request_data.get(f"column_mapping_{canonical_col}")
        if mapped_value:
            fallback_mapping[canonical_col] = mapped_value

    return fallback_mapping or None


def apply_column_mapping(df, column_mapping):
    if not column_mapping:
        return df

    rename_map = {}
    used_sources = set()
    normalized_columns = {normalize_column_key(col): col for col in df.columns}
    for canonical_col, source_col in column_mapping.items():
        if canonical_col not in COLUMN_ALIASES:
            continue
        if not source_col:
            continue
        requested_source = str(source_col)
        source_col = normalized_columns.get(normalize_column_key(requested_source), requested_source)
        if source_col not in df.columns:
            raise ValueError(f'Mapped source column "{source_col}" was not found in the file.')
        if source_col in used_sources:
            raise ValueError(f'Column "{source_col}" was mapped more than once.')
        rename_map[source_col] = canonical_col
        used_sources.add(source_col)

    return df.rename(columns=rename_map)


def resolve_uploaded_label(raw_value, labels_by_value):
    if pd.isna(raw_value) or str(raw_value).strip() == "":
        return None

    try:
        normalized_value = int(float(raw_value))
    except (TypeError, ValueError):
        raise ValueError(f'Invalid label value "{raw_value}". Expected a numeric label such as 0, 1, or 2.')

    label_obj = labels_by_value.get(normalized_value)
    if not label_obj:
        known_values = ", ".join(str(value) for value in sorted(labels_by_value.keys()))
        raise ValueError(
            f'Label value "{normalized_value}" does not exist in ECG labels. '
            f'Available label values: {known_values}.'
        )

    return label_obj


def normalize_columns(df):
    """
    Rename columns in DataFrame to canonical column names if a match for any aliases is found.
    """
    new_columns = {}
    normalized_columns = {normalize_column_key(col): col for col in df.columns}

    for canonical_col, aliases in COLUMN_ALIASES.items():
        if canonical_col in df.columns:
            continue
        for alias in aliases:
            matched_column = normalized_columns.get(normalize_column_key(alias))
            if matched_column:
                new_columns[matched_column] = canonical_col
                break
    df = df.rename(columns=new_columns)
    return df


def get_or_create_profile(user):
    if not user or not user.is_authenticated:
        return None
    profile, _ = Profile.objects.get_or_create(user=user)
    return profile


def get_effective_role(user):
    if not user or not user.is_authenticated:
        return Profile.ROLE_DOCTOR
    if user.is_superuser or user.is_staff:
        return Profile.ROLE_ADMIN
    profile = get_or_create_profile(user)
    return profile.effective_role if profile else Profile.ROLE_DOCTOR


def scope_records_for_user(user, queryset=None):
    queryset = queryset if queryset is not None else ECGRecord.objects.all()
    role = get_effective_role(user)
    if role in {Profile.ROLE_ADMIN, Profile.ROLE_DOCTOR}:
        return queryset

    profile = get_or_create_profile(user)
    patient_id = getattr(profile, "patient_id", None)
    if role == Profile.ROLE_PATIENT and patient_id:
        return queryset.filter(patient_id=str(patient_id))
    return queryset.none()


def can_access_record(user, record):
    return scope_records_for_user(user, ECGRecord.objects.filter(id=record.id)).exists()


def resolve_model_path(model_info):
    from django.conf import settings

    return str(Path(settings.BASE_DIR) / model_info["path"])


SKLEARN_WRAPPERS = (ECGXGBoostWrapper, AdaBoostECGWrapper, RandomForestECGWrapper)


def ensure_builtin_models():
    for key, info in MODEL_MAP.items():
        DeployedModel.objects.update_or_create(
            key=key,
            defaults={
                "label": info.get("label", key),
                "base_model_key": key,
                "source_type": DeployedModel.SOURCE_BUILTIN,
                "weights_path": info["path"],
                "input_size": info["input_size"],
                "num_classes": info["num_classes"],
                "trainable": info["class"] not in SKLEARN_WRAPPERS,
                "is_active": True,
            },
        )


def deployed_model_to_runtime_info(model_obj):
    base_info = MODEL_MAP.get(model_obj.base_model_key)
    if not base_info:
        raise KeyError(f'Base model "{model_obj.base_model_key}" is not registered in MODEL_MAP.')

    runtime_info = dict(base_info)
    runtime_info.update({
        "key": model_obj.key,
        "label": model_obj.label,
        "path": model_obj.weights_path,
        "input_size": model_obj.input_size,
        "num_classes": model_obj.num_classes,
        "trainable": model_obj.trainable,
        "source_type": model_obj.source_type,
        "base_model_key": model_obj.base_model_key,
        "db_id": model_obj.id,
    })
    return runtime_info


def get_active_deployed_models():
    ensure_builtin_models()
    return DeployedModel.objects.filter(is_active=True).order_by("label", "key")


def get_model_runtime_catalog():
    catalog = {}
    for model_obj in get_active_deployed_models():
        catalog[model_obj.key] = deployed_model_to_runtime_info(model_obj)
    return catalog


def get_model_runtime_info(model_key):
    return get_model_runtime_catalog().get(model_key)


def get_patient_assignment(patient_id):
    if not patient_id:
        return None
    ensure_builtin_models()
    return (
        PatientModelAssignment.objects
        .select_related("model")
        .filter(patient_id=str(patient_id), model__is_active=True)
        .first()
    )


def get_visible_model_runtime_catalog_for_user(user):
    catalog = get_model_runtime_catalog()
    role = get_effective_role(user)
    if role != Profile.ROLE_PATIENT:
        return catalog

    profile = get_or_create_profile(user)
    assignment = get_patient_assignment(getattr(profile, "patient_id", None))
    if not assignment:
        return {}

    assigned_info = catalog.get(assignment.model.key)
    return {assignment.model.key: assigned_info} if assigned_info else {}


def load_inference_model(model_name):
    if model_name in loaded_models:
        return loaded_models[model_name], None

    model_info = get_model_runtime_info(model_name)
    if not model_info:
        return None, f'Model "{model_name}" not found.'

    model_path = resolve_model_path(model_info)
    if not os.path.exists(model_path):
        return None, f'Weights for "{model_name}" not found at {model_path}.'

    model_class = model_info["class"]
    extra_kwargs = model_info.get("kwargs", {})

    try:
        if model_class in SKLEARN_WRAPPERS:
            model = model_class(model_path=model_path, **extra_kwargs)
        else:
            model = model_class(model_info["num_classes"], **extra_kwargs)
            model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        if hasattr(model, "eval"):
            model.eval()
        loaded_models[model_name] = model
        return model, None
    except Exception as exc:
        return None, str(exc)


def as_probabilities(output_tensor):
    tensor = output_tensor.detach().cpu()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)

    row_sums = tensor.sum(dim=1)
    if torch.all(tensor >= 0) and torch.all(tensor <= 1) and torch.allclose(
        row_sums, torch.ones_like(row_sums), atol=1e-3
    ):
        return tensor
    return torch.softmax(tensor, dim=1)


class FileUploadView(APIView):
    permission_classes = [IsAuthenticated, IsAdminRole]

    def post(self, request, format=None):
        file = request.FILES.get("file")
        if not file:
            return Response({"error": "No file provided"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            df = load_tabular_file(file)

            try:
                column_mapping = extract_column_mapping(request.data)
            except ValueError as exc:
                return Response({"error": str(exc)}, status=400)

            df = apply_column_mapping(df, column_mapping)
            df = normalize_columns(df)

            if not REQUIRED_UPLOAD_COLUMNS.issubset(df.columns):
                missing = REQUIRED_UPLOAD_COLUMNS - set(df.columns)
                return Response(build_upload_mapping_response(df, missing), status=400)

            labels_by_value = {lbl.value: lbl for lbl in ECGLabel.objects.all()}

            stored_file_name = build_unique_file_name(file.name)
            ecg_file = ECGFile.objects.create(file_name=stored_file_name, status="processing")

            records = []
            for _, row in df.iterrows():
                label_obj = resolve_uploaded_label(row["label"], labels_by_value)
                records.append(
                    ECGRecord(
                        file=ecg_file,
                        patient_id=str(row["patient_id"]),
                        ecg_wave=row["ecg_wave"],  # expected format string/list
                        heart_rate=float(row["heart_rate"]),
                        label_id=label_obj.id if label_obj else None,
                    )
                )

            with transaction.atomic():
                ECGRecord.objects.bulk_create(records, batch_size=5000)

            ecg_file.total_records = len(records)
            ecg_file.status = "completed"
            ecg_file.save()

            return Response(ECGFileSerializer(ecg_file).data, status=status.HTTP_201_CREATED)

        except Exception as e:
            if 'ecg_file' in locals():
                ecg_file.status = "failed"
                ecg_file.save()
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# ---------------------------
# Paginated Records (with optional wave caching)
# ---------------------------

class ECGRecordFilter(filters.FilterSet):
    file__file_name__in = filters.BaseInFilter(field_name='file__file_name', lookup_expr='in')
    conflict_only = filters.BooleanFilter(method='filter_conflict_only', label='Show only label conflicts')

    class Meta:
        model = ECGRecord
        fields = {
            'patient_id': ['exact'],
            'label__value': ['exact'],
            'is_verified': ['exact'],
        }

    def filter_conflict_only(self, queryset, name, value):
        if value:
            # Conflict exists if both present AND values differ
            from django.db.models import Q, F
            return queryset.filter(
                label__isnull=False,
                ai_label__isnull=False
            ).exclude(label__value=F('ai_label__value'))
        return queryset


class ECGRecordListView(generics.ListAPIView):
    serializer_class = ECGRecordSerializer
    filter_backends = [filters.DjangoFilterBackend]
    filterset_class = ECGRecordFilter
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        """
        Base queryset:
        - select_related: resolves label + ai_label in ONE JOIN query (no N+1)
        - defer ecg_wave: 2604 floats per row are skipped unless explicitly needed
        """
        return (
            scope_records_for_user(self.request.user, ECGRecord.objects)
            .select_related('label', 'ai_label')
            .defer('ecg_wave')           # heavy column, loaded only on demand
            .order_by('id')
        )

    def list(self, request, *args, **kwargs):
        include_wave = request.query_params.get("include_wave", "false").lower() == "true"
        qs = self.filter_queryset(self.get_queryset())

        if include_wave:
            # Need ecg_wave — lift the defer and load with wave
            qs = qs.select_related('label', 'ai_label')  # already set, just clearer

        page = self.paginate_queryset(qs)

        if not include_wave:
            serializer = self.get_serializer(page, many=True)
            return self.get_paginated_response(serializer.data)

        # include_wave=true path
        use_redis = request.query_params.get("use_redis", "false").lower() == "true"
        data_with_wave = []
        for record in page:
            ecg_wave = None
            if use_redis:
                ecg_wave = get_ecg_wave(record.id)

            if ecg_wave is None:
                # Have to re-fetch with wave since we deferred it
                from django.db import models as dj_models
                full = ECGRecord.objects.only('ecg_wave').get(pk=record.id)
                ecg_wave = full.ecg_wave
                if use_redis:
                    set_ecg_wave(record.id, ecg_wave, timeout=300)

            label_obj = record.label
            label_data = {
                "id": label_obj.id, "name": label_obj.name,
                "value": label_obj.value, "color": label_obj.color,
            } if label_obj else None

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
    serializer_class = ECGRecordDetailSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return scope_records_for_user(
            self.request.user,
            ECGRecord.objects.select_related('label', 'ai_label')
        )


# ---------------------------
# Fetch ECG Wave (per record, uses cache)
# ---------------------------
@api_view(["GET"])
@permission_classes([IsAuthenticated])
def get_ecg_wave_view(request, record_id):
    patient_id = request.query_params.get("patient_id")

    try:
        record = scope_records_for_user(request.user, ECGRecord.objects).get(id=record_id)
    except ECGRecord.DoesNotExist:
        return Response({"error": "Record not found"}, status=status.HTTP_404_NOT_FOUND)

    if patient_id and str(record.patient_id) != str(patient_id):
        return Response({"error": "Patient ID does not match record"}, status=status.HTTP_400_BAD_REQUEST)

    use_redis = request.query_params.get("use_redis", "false").lower() == "true"
    ecg_wave = None

    if use_redis:
        ecg_wave = get_ecg_wave(record_id)

    if ecg_wave is None:
        ecg_wave = record.ecg_wave
        if use_redis:
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
# Bulk Label Update  (human annotation)
# ---------------------------
@api_view(["POST"])
@permission_classes([IsAuthenticated, IsDoctorOrAdmin])
def bulk_label_update_view(request):
    """
    Bulk update HUMAN labels for multiple ECG records.
    Sets labeled_by='human' and is_verified=True automatically.
    POST: { "records": [{"id": N, "patient_id": "X", "label": value}, ...] }
    """
    records = request.data.get("records", [])
    if not isinstance(records, list):
        return Response({"error": "'records' must be a list."}, status=400)

    unique_labels = set(rec.get("label") for rec in records if rec.get("label") is not None)
    label_map = {lbl.value: lbl for lbl in ECGLabel.objects.filter(value__in=unique_labels)}

    updated, errors, to_update = [], [], []

    with transaction.atomic():
        for rec in records:
            record_id = rec.get("id")
            patient_id = rec.get("patient_id")
            label_value = rec.get("label")

            if None in (record_id, label_value, patient_id):
                errors.append(f"Missing field in: {rec}")
                continue

            try:
                ecg_record = scope_records_for_user(
                    request.user,
                    ECGRecord.objects
                ).get(id=record_id, patient_id=patient_id)
            except ECGRecord.DoesNotExist:
                errors.append(f"Record {record_id} / patient {patient_id} not found")
                continue

            label_obj = label_map.get(label_value)
            if not label_obj:
                errors.append(f"Label value {label_value} does not exist")
                continue

            ecg_record.label = label_obj
            ecg_record.labeled_by = 'human'
            ecg_record.is_verified = True   # human annotation = verified
            to_update.append(ecg_record)
            updated.append(record_id)

        if to_update:
            ECGRecord.objects.bulk_update(to_update, ['label', 'labeled_by', 'is_verified'])

    if errors:
        return Response({"updated": updated, "errors": errors}, status=207)
    return Response({"updated": updated}, status=200)


# ---------------------------
# ECGFile ViewSet
# ---------------------------
class ECGFileViewSet(viewsets.ModelViewSet):
    serializer_class = ECGFileSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        role = get_effective_role(self.request.user)
        queryset = ECGFile.objects.all().order_by('-uploaded_at')
        if role == Profile.ROLE_PATIENT:
            visible_file_ids = scope_records_for_user(self.request.user, ECGRecord.objects).values_list('file_id', flat=True).distinct()
            queryset = queryset.filter(id__in=visible_file_ids)
        return queryset

    def get_permissions(self):
        if self.action in {'create', 'destroy', 'update', 'partial_update'}:
            permission_classes = [IsAuthenticated, IsAdminRole]
        else:
            permission_classes = [IsAuthenticated]
        return [permission() for permission in permission_classes]

    @action(detail=True, methods=['get'])
    def records(self, request, pk=None):
        try:
            ecg_file = self.get_object()
        except ECGFile.DoesNotExist:
            return Response({"detail": "File not found."}, status=status.HTTP_404_NOT_FOUND)

        records = scope_records_for_user(request.user, ECGRecord.objects.filter(file=ecg_file))
        serializer = ECGRecordSerializer(records, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['get'])
    def records_detail(self, request, pk=None):
        try:
            ecg_file = self.get_object()
        except ECGFile.DoesNotExist:
            return Response({"detail": "File not found."}, status=status.HTTP_404_NOT_FOUND)

        records = scope_records_for_user(request.user, ECGRecord.objects.filter(file=ecg_file))
        serializer = ECGRecordDetailSerializer(records, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['get'], url_path='download_records/csv',
            permission_classes=[IsAuthenticated], authentication_classes=[])
    def download_records_csv(self, request, pk=None):
        ecg_file = self.get_object()
        records_qs = scope_records_for_user(request.user, ECGRecord.objects.filter(file=ecg_file)).order_by('created_at')

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

    @action(detail=True, methods=['get'], url_path='download_records/xlsx',
            permission_classes=[IsAuthenticated], authentication_classes=[])
    def download_records_xlsx(self, request, pk=None):
        ecg_file = self.get_object()
        records_qs = scope_records_for_user(request.user, ECGRecord.objects.filter(file=ecg_file)).order_by('created_at')

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
    permission_classes = [IsAuthenticated]

    def get(self, request):
        files = self._get_files_for_user(request.user)
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

    def _get_files_for_user(self, user):
        role = get_effective_role(user)
        queryset = ECGFile.objects.all()
        if role == Profile.ROLE_PATIENT:
            visible_file_ids = scope_records_for_user(user, ECGRecord.objects).values_list('file_id', flat=True).distinct()
            queryset = queryset.filter(id__in=visible_file_ids)
        return queryset.annotate(total_records=Count('records'))

class LabelsPatientsByFilesView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        file_names = request.data.get('files', [])
        if not file_names:
            return Response({"error": "No files specified."}, status=status.HTTP_400_BAD_REQUEST)

        records = scope_records_for_user(request.user, ECGRecord.objects).filter(file__file_name__in=file_names) \
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
    """
    POST with optional verification_filter:
      'all'          - all labeled records (human + ai)
      'verified'     - only is_verified=True (human-confirmed)
      'unverified'   - is_verified=False but has a human label
      'ai_only'      - records that only have an ai_label, no human label
    """

    permission_classes = [IsAuthenticated]

    def post(self, request):
        files = request.data.get('files', [])
        patients = request.data.get('patients', [])
        labels = request.data.get('labels', [])
        file_format = request.data.get('format', 'csv').lower()
        verification_filter = request.data.get('verification_filter', 'all')

        if not files:
            return Response({"error": "Must specify at least one file"}, status=400)

        patient_ids = [p['patient_id'] for p in patients if p.get('numRecords', 0) > 0]
        label_ids = [l['id'] for l in labels if l.get('numRecords', 0) > 0]
        file_names = [f['file_name'] for f in files if f.get('numRecords', 0) > 0]

        queryset = scope_records_for_user(
            request.user,
            ECGRecord.objects.filter(file__file_name__in=file_names).select_related('label', 'ai_label')
        )

        # Apply verification filter
        if verification_filter == 'verified':
            queryset = queryset.filter(is_verified=True)
        elif verification_filter == 'unverified':
            queryset = queryset.filter(is_verified=False, label__isnull=False)
        elif verification_filter == 'ai_only':
            queryset = queryset.filter(label__isnull=True, ai_label__isnull=False)
        else:  # 'all' — any that has at least one label
            queryset = queryset.filter(Q(label__isnull=False) | Q(ai_label__isnull=False))

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

        for rec in queryset.order_by('file__file_name', 'patient_id', 'created_at'):
            if file_limits.get(rec.file.file_name) and file_counts[rec.file.file_name] >= file_limits[rec.file.file_name]:
                continue
            if patient_limits and patient_counts[rec.patient_id] >= patient_limits.get(rec.patient_id, float('inf')):
                continue
            label_id = rec.label_id if rec.label_id is not None else -1
            if label_limits and label_counts[label_id] >= label_limits.get(label_id, float('inf')):
                continue

            data.append({
                "ID": idx,
                "Patient ID": rec.patient_id,
                "Heart Rate": rec.heart_rate,
                "ECG Wave": rec.ecg_wave if rec.ecg_wave else '',
                "Label": rec.label.value if rec.label else '',
                "Label Name": rec.label.name if rec.label else '',
                "AI Label": rec.ai_label.value if rec.ai_label else '',
                "AI Label Name": rec.ai_label.name if rec.ai_label else '',
                "AI Model": rec.ai_model_name or '',
                "Verified": rec.is_verified,
                "Labeled By": rec.labeled_by,
            })
            idx += 1
            file_counts[rec.file.file_name] += 1
            patient_counts[rec.patient_id] += 1
            label_counts[label_id] += 1

        if not data:
            return Response({"error": "No records match the given filters."}, status=400)

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
    permission_classes = [IsAuthenticated]

    def get(self, request):
        patient_id = request.query_params.get("patient_id")
        source = request.query_params.get("source", "human").lower()

        if source == "ai":
            records = scope_records_for_user(
                request.user,
                ECGRecord.objects.exclude(ai_label__isnull=True).select_related("ai_label"),
            )
            if patient_id:
                records = records.filter(patient_id=str(patient_id))
            labels = list(
                records.values("ai_label_id", "ai_label__name")
                .annotate(count=Count("id"))
                .order_by("ai_label__name")
            )
            return Response([
                {"id": item["ai_label_id"], "name": item["ai_label__name"], "count": item["count"]}
                for item in labels
            ])

        records = scope_records_for_user(
            request.user,
            ECGRecord.objects.exclude(label__isnull=True).select_related("label"),
        )
        if patient_id:
            records = records.filter(patient_id=str(patient_id))
        labels = list(
            records.values("label_id", "label__name")
            .annotate(count=Count("id"))
            .order_by("label__name")
        )
        return Response([
            {"id": item["label_id"], "name": item["label__name"], "count": item["count"]}
            for item in labels
        ])


class PatientCountView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        patients = scope_records_for_user(request.user, ECGRecord.objects).values('patient_id').annotate(record_count=Count('id')).order_by('patient_id')
        response = [{'patient_id': p['patient_id'], 'record_count': p['record_count']} for p in patients]
        return Response(response)

class ECGFileSummaryView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        try:
            files_qs = ECGFile.objects.annotate(record_count=Count('records'))

            file_summaries = []
            for f in files_qs:
                # Get unique patients for this file
                file_records = scope_records_for_user(request.user, ECGRecord.objects.filter(file=f))
                unique_patients = file_records.values("patient_id").distinct().count()

                # Get label counts for this file (ignore "No Label")
                label_counts = (
                    file_records
                    .exclude(label__name="No Label")
                    .values("label_id", "label__name")
                    .annotate(count=Count("id"))
                )

                visible_records = file_records.count()
                if visible_records == 0 and get_effective_role(request.user) == Profile.ROLE_PATIENT:
                    continue

                file_summaries.append({
                    "file_name": f.file_name,
                    "uploaded_at": f.uploaded_at,
                    "status": f.status,
                    "total_records": visible_records,
                    "label_counts": list(label_counts),
                    "unique_patient_count": unique_patients,
                })

            return Response(file_summaries)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class DashboardSummaryView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        try:
            scoped_records = scope_records_for_user(request.user, ECGRecord.objects)
            total_files = ECGFile.objects.count() if get_effective_role(request.user) != Profile.ROLE_PATIENT else scoped_records.values('file_id').distinct().count()
            total_records = scoped_records.count()
            
            # Count records with human labels
            labelled_records = scoped_records.exclude(label=None).count()
            # Count records with AI labels
            ai_labelled_records = scoped_records.exclude(ai_label=None).count()
            
            # Verified count
            verified_records = scoped_records.filter(is_verified=True).count()

            # Label distributions (Human)
            labels_data_qs = scoped_records.exclude(label__isnull=True).values('label__name').annotate(count=Count('id'))
            labels_data = {item['label__name']: item['count'] for item in labels_data_qs}
            
            # Label distributions (AI)
            ai_labels_data_qs = scoped_records.exclude(ai_label__isnull=True).values('ai_label__name').annotate(count=Count('id'))
            ai_labels_data = {item['ai_label__name']: item['count'] for item in ai_labels_data_qs}

            # Compatibility field
            normal_records = labels_data.get("Normal", 0)

            return Response({
                "total_files": total_files,
                "total_records": total_records,
                "labelled_records": labelled_records,
                "ai_labelled_records": ai_labelled_records,
                "verified_records": verified_records,
                "normal_records": normal_records,
                "labels_data": labels_data,
                "ai_labels_data": ai_labels_data,
            }, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"DashboardSummary error: {str(e)}", exc_info=True)
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class AuthorizeUserView(APIView):
    permission_classes = [IsAuthenticated, IsAdminRole]

    def post(self, request):
        username = request.data.get('username')
        try:
            user = User.objects.get(username=username)
            profile = get_or_create_profile(user)
            profile.is_authorized = True
            profile.save()
            return Response({"authorized": True})
        except User.DoesNotExist:
            return Response({"error": "User does not exist"}, status=status.HTTP_404_NOT_FOUND)


class RegisterView(APIView):
    permission_classes = [AllowAny]  # Anyone can register

    def post(self, request):
        try:
            serializer = RegisterSerializer(data=request.data)
            if serializer.is_valid():
                user = serializer.save()
                return Response({
                    'msg': 'User registered successfully!',
                    'user': UserSerializer(user).data,
                }, status=status.HTTP_201_CREATED)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        except OperationalError:
            return Response({
                'error': 'Database write failed. Your local SQLite database is unavailable or corrupted.'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        except Exception as exc:
            return Response({
                'error': f'Registration failed: {exc}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class LoginView(APIView):
    permission_classes = [AllowAny]  # Anyone can login

    def post(self, request):
        username = request.data.get('username')
        password = request.data.get('password')
        user = authenticate(username=username, password=password)
        if user is not None:
            profile = get_or_create_profile(user)
            refresh = RefreshToken.for_user(user)
            return Response({
                'refresh': str(refresh),
                'access': str(refresh.access_token),
                'user': UserSerializer(user).data,
                'role': profile.effective_role,
                'patient_id': profile.patient_id,
            })
        return Response({'error': 'Invalid credentials'}, status=status.HTTP_401_UNAUTHORIZED)


class CurrentUserView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        get_or_create_profile(user)
        return Response(UserSerializer(user).data, status=status.HTTP_200_OK)


class UserSettingsView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        profile = get_or_create_profile(request.user)
        return Response(ProfileSerializer(profile).data, status=status.HTTP_200_OK)

    def patch(self, request):
        profile = get_or_create_profile(request.user)
        serializer = ProfileSerializer(profile, data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(serializer.data, status=status.HTTP_200_OK)


loaded_models = {}
class PredictECGView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        data = request.data
        model_name = data.get('model_name')
        input_data = data.get('input')
        visible_catalog = get_visible_model_runtime_catalog_for_user(request.user)

        if model_name not in visible_catalog:
            return Response({'error': f'Model "{model_name}" not found.'}, status=status.HTTP_400_BAD_REQUEST)

        model_info = visible_catalog[model_name]
        expected_size = model_info['input_size']

        if input_data is None or len(input_data) != expected_size:
            return Response({'error': f'Input must be a list of {expected_size} floats.'}, status=status.HTTP_400_BAD_REQUEST)

        model, err = load_inference_model(model_name)
        if err:
            return Response({'error': err}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        # Preprocess input — CNN models expect (batch, 1, L), LSTM expects (batch, L)
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)  # (1, 2604)
        model_type = model_info['class'].__name__
        if model_type in ('ECG1DCNN', 'ECGCNNLSTMHybrid', 'ECGResNet1D'):
            input_tensor = input_tensor.unsqueeze(0)  # (1, 1, 2604)

        with torch.no_grad():
            output = model(input_tensor)
            probs_tensor = as_probabilities(output)
            pred_class = torch.argmax(probs_tensor, dim=1).item()
            probs = probs_tensor.numpy().tolist()[0]

        # Fetch the friendly class name from the ECGLabel table
        try:
            pred_class_obj = ECGLabel.objects.get(value=pred_class)
            pred_class_name = pred_class_obj.name
        except ECGLabel.DoesNotExist:
            pred_class_name = f"Unknown (class {pred_class})"
        return Response({
            'predicted_class': pred_class,
            'predicted_class_name': pred_class_name,
            'probabilities': probs,
            'confidence': max(probs) if probs else 0
        })

class ModelListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        visible_catalog = get_visible_model_runtime_catalog_for_user(request.user)
        assignment = None
        if get_effective_role(request.user) == Profile.ROLE_PATIENT:
            profile = get_or_create_profile(request.user)
            assignment = get_patient_assignment(getattr(profile, "patient_id", None))

        models_info = {
            name: {
                "input_size": info["input_size"],
                "num_classes": info["num_classes"],
                "label": info.get("label", name),
                "available": os.path.exists(resolve_model_path(info)),
                "trainable": info.get("trainable", info["class"] not in SKLEARN_WRAPPERS),
                "source_type": info.get("source_type", DeployedModel.SOURCE_BUILTIN),
                "base_model_key": info.get("base_model_key", name),
            }
            for name, info in visible_catalog.items()
        }
        return Response({
            "models": models_info,
            "patient_assignment": (
                {
                    "patient_id": assignment.patient_id,
                    "model_key": assignment.model.key,
                    "model_label": assignment.model.label,
                }
                if assignment else None
            ),
        }, status=status.HTTP_200_OK)


class DeployedModelRegistryView(APIView):
    permission_classes = [IsAuthenticated, IsAdminRole]

    def get(self, request):
        ensure_builtin_models()
        models_qs = DeployedModel.objects.filter(is_active=True).order_by("label", "key")
        assignments_qs = PatientModelAssignment.objects.select_related("model").order_by("patient_id")
        return Response({
            "models": DeployedModelSerializer(models_qs, many=True).data,
            "assignments": PatientModelAssignmentSerializer(assignments_qs, many=True).data,
        })

    def post(self, request):
        ensure_builtin_models()
        uploaded_file = request.FILES.get("file")
        label = str(request.data.get("label", "")).strip()
        base_model_key = str(request.data.get("base_model_key", "")).strip()
        key = str(request.data.get("key", "")).strip()
        overwrite = str(request.data.get("overwrite", "")).lower() in ("1", "true", "yes", "on")
        input_size = request.data.get("input_size")
        num_classes = request.data.get("num_classes")

        if not uploaded_file:
            return Response({"error": "Model file is required."}, status=400)
        if not label:
            return Response({"error": "label is required."}, status=400)
        if base_model_key not in MODEL_MAP:
            return Response({"error": f'base_model_key must be one of: {list(MODEL_MAP.keys())}'}, status=400)

        try:
            input_size = int(input_size or MODEL_MAP[base_model_key]["input_size"])
            num_classes = int(num_classes or MODEL_MAP[base_model_key]["num_classes"])
        except (TypeError, ValueError):
            return Response({"error": "input_size and num_classes must be integers."}, status=400)

        ext = ospath.splitext(uploaded_file.name)[1].lower()
        expected_ext = ".pkl" if MODEL_MAP[base_model_key]["class"] in SKLEARN_WRAPPERS else ".pth"
        if ext != expected_ext:
            return Response({"error": f'Expected a {expected_ext} file for {base_model_key}.'}, status=400)

        safe_key_base = re.sub(r"[^a-zA-Z0-9_]+", "_", key or label).strip("_") or uuid.uuid4().hex[:10]
        safe_key = safe_key_base

        existing_model = None
        if key:
            existing_model = DeployedModel.objects.filter(key=safe_key).first()
            if existing_model and not overwrite:
                return Response(
                    {"error": "Model key already exists. Set overwrite=true to replace the existing uploaded model."},
                    status=400,
                )
            if existing_model and existing_model.source_type != DeployedModel.SOURCE_UPLOADED:
                return Response(
                    {"error": "Built-in models cannot be overwritten. Use a new key for uploaded models."},
                    status=400,
                )
        else:
            counter = 2
            while DeployedModel.objects.filter(key=safe_key).exists():
                safe_key = f"{safe_key_base}_{counter}"
                counter += 1

        relative_path = f"{UPLOADED_MODELS_FOLDER}/{safe_key}_{uuid.uuid4().hex[:8]}{ext}"
        absolute_path = Path(resolve_model_path({"path": relative_path}))
        absolute_path.parent.mkdir(parents=True, exist_ok=True)

        with absolute_path.open("wb") as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        if existing_model:
            existing_model.label = label
            existing_model.base_model_key = base_model_key
            existing_model.weights_path = relative_path
            existing_model.input_size = input_size
            existing_model.num_classes = num_classes
            existing_model.trainable = MODEL_MAP[base_model_key]["class"] not in SKLEARN_WRAPPERS
            existing_model.is_active = True
            existing_model.save()
            loaded_models.pop(existing_model.key, None)
            return Response(DeployedModelSerializer(existing_model).data, status=200)

        model_obj = DeployedModel.objects.create(
            key=safe_key,
            label=label,
            base_model_key=base_model_key,
            source_type=DeployedModel.SOURCE_UPLOADED,
            weights_path=relative_path,
            input_size=input_size,
            num_classes=num_classes,
            trainable=MODEL_MAP[base_model_key]["class"] not in SKLEARN_WRAPPERS,
            is_active=True,
        )
        loaded_models.pop(model_obj.key, None)

        return Response(DeployedModelSerializer(model_obj).data, status=201)


class PatientModelAssignmentView(APIView):
    permission_classes = [IsAuthenticated, IsAdminRole]

    def post(self, request):
        ensure_builtin_models()
        patient_id = str(request.data.get("patient_id", "")).strip()
        model_key = str(request.data.get("model_key", "")).strip()
        if not patient_id or not model_key:
            return Response({"error": "patient_id and model_key are required."}, status=400)

        try:
            model_obj = DeployedModel.objects.get(key=model_key, is_active=True)
        except DeployedModel.DoesNotExist:
            return Response({"error": f'Model "{model_key}" not found.'}, status=404)

        assignment, _ = PatientModelAssignment.objects.update_or_create(
            patient_id=patient_id,
            defaults={"model": model_obj},
        )
        return Response(PatientModelAssignmentSerializer(assignment).data, status=200)

    def delete(self, request):
        patient_id = str(request.data.get("patient_id", "")).strip()
        if not patient_id:
            return Response({"error": "patient_id is required."}, status=400)
        deleted, _ = PatientModelAssignment.objects.filter(patient_id=patient_id).delete()
        if not deleted:
            return Response({"error": "Assignment not found."}, status=404)
        return Response(status=204)


# ---------------------------
# Auto-Label: bulk AI prediction on a file's unlabeled records
# ---------------------------
class AutoLabelView(APIView):
    """
    POST { "file_name": "...", "model_name": "ECG1DCNN", "overwrite": false }
    Runs the AI model over every record in the file and writes predictions
    to the `ai_label` column ONLY.  The human `label` column is NEVER touched.
    If overwrite=false (default), only records with no ai_label are processed.
    Returns counts of labeled/skipped/errors.
    """

    permission_classes = [IsAuthenticated, IsAdminRole]

    def _load_model(self, model_name):
        return load_inference_model(model_name)

    def post(self, request):
        file_name = request.data.get('file_name')
        model_name = request.data.get('model_name', 'ECG1DCNN')
        overwrite = request.data.get('overwrite', False)

        if not file_name:
            return Response({'error': 'file_name is required.'}, status=400)

        try:
            ecg_file = ECGFile.objects.get(file_name=file_name)
        except ECGFile.DoesNotExist:
            return Response({'error': f'File "{file_name}" not found.'}, status=404)

        model, err = self._load_model(model_name)
        if err:
            return Response({'error': err}, status=503)

        model_info = get_model_runtime_info(model_name)
        if not model_info:
            return Response({'error': f'Model "{model_name}" not found.'}, status=400)
        model_type = model_info['class'].__name__

        # Only filter on ai_label — never touch human label
        qs = ECGRecord.objects.filter(file=ecg_file).select_related('ai_label')
        if not overwrite:
            qs = qs.filter(ai_label__isnull=True)

        all_labels = {lbl.value: lbl for lbl in ECGLabel.objects.all()}

        labeled_count = 0
        skipped_count = 0
        error_count = 0
        to_update = []

        with torch.no_grad():
            for record in qs:
                try:
                    wave = record.ecg_wave
                    if isinstance(wave, str):
                        wave = [float(v) for v in wave.split(',')]
                    if len(wave) != model_info['input_size']:
                        skipped_count += 1
                        continue

                    t = torch.tensor(wave, dtype=torch.float32).unsqueeze(0)
                    if model_type in ('ECG1DCNN', 'ECGCNNLSTMHybrid', 'ECGResNet1D'):
                        t = t.unsqueeze(0)

                    output = model(t)
                    probs_tensor = as_probabilities(output)
                    probs = probs_tensor.numpy().tolist()[0]
                    pred_class = torch.argmax(probs_tensor, dim=1).item()
                    label_obj = all_labels.get(pred_class)
                    if label_obj:
                        # Write to ai_label ONLY — human label stays safe
                        record.ai_label = label_obj
                        record.ai_model_name = model_name
                        record.ai_confidence = max(probs)
                        record.ai_probabilities = probs
                        to_update.append(record)
                        labeled_count += 1
                    else:
                        skipped_count += 1
                except Exception as e:
                    logger.warning('AutoLabel error record %s: %s', record.id, e)
                    error_count += 1

        if to_update:
            ECGRecord.objects.bulk_update(to_update, ['ai_label', 'ai_model_name', 'ai_confidence', 'ai_probabilities'])

        return Response({
            'file': file_name,
            'model': model_name,
            'ai_labeled': labeled_count,
            'skipped': skipped_count,
            'errors': error_count,
        }, status=200)


# ---------------------------
# Model Comparison: run multiple models on one record
# ---------------------------
class ModelCompareView(APIView):
    """
    POST { "record_id": 123, "models": ["ECG1DCNN", "ECGCNNLSTMHybrid"] }
    Returns predictions from all requested models for side-by-side comparison.
    """

    permission_classes = [IsAuthenticated]

    def post(self, request):
        record_id = request.data.get('record_id')
        visible_catalog = get_visible_model_runtime_catalog_for_user(request.user)
        model_names = request.data.get('models', list(visible_catalog.keys()))

        try:
            record = scope_records_for_user(
                request.user,
                ECGRecord.objects.select_related('label', 'ai_label')
            ).get(id=record_id)
        except ECGRecord.DoesNotExist:
            return Response({'error': 'Record not found.'}, status=404)

        wave = record.ecg_wave
        if isinstance(wave, str):
            wave = [float(v) for v in wave.split(',')]

        all_labels = {lbl.value: lbl for lbl in ECGLabel.objects.all()}
        results = {}

        for model_name in model_names:
            model_info = visible_catalog.get(model_name)
            if not model_info:
                results[model_name] = {'error': 'Model is not available for this user'}
                continue
            if not os.path.exists(resolve_model_path(model_info)):
                results[model_name] = {'error': 'Weights not found — train the model first'}
                continue
            try:
                model, err = load_inference_model(model_name)
                if err:
                    results[model_name] = {'error': err}
                    continue

                model_type = model_info['class'].__name__
                t = torch.tensor(wave, dtype=torch.float32).unsqueeze(0)
                if model_type in ('ECG1DCNN', 'ECGCNNLSTMHybrid', 'ECGResNet1D'):
                    t = t.unsqueeze(0)

                with torch.no_grad():
                    output = model(t)
                    probs_tensor = as_probabilities(output)
                    pred = torch.argmax(probs_tensor, dim=1).item()
                    probs = probs_tensor.numpy().tolist()[0]

                label_name = all_labels[pred].name if pred in all_labels else f'Class {pred}'
                results[model_name] = {
                    'label': model_info.get('label', model_name),
                    'predicted_class': pred,
                    'predicted_class_name': label_name,
                    'probabilities': probs,
                }
            except Exception as e:
                results[model_name] = {'error': str(e)}

        return Response({
            'record_id': record_id,
            'patient_id': record.patient_id,
            'human_label': record.label.name if record.label else None,
            'human_verified': record.is_verified,
            'ai_label': record.ai_label.name if record.ai_label else None,
            'model_results': results,
        })



class ModelAnalysisView(APIView):
    """
    GET /api/model-analysis/
    Calculates Accuracy and Confusion Matrix for all available models
    using Human Labeled records as the Ground Truth.
    """
    permission_classes = [IsAuthenticated, IsDoctorOrAdmin]

    def get(self, request):
        from django.conf import settings
        BASE_DIR = settings.BASE_DIR  # e.g.  .../backend/

        # 1. Get all records with human labels
        labeled_records = scope_records_for_user(
            request.user,
            ECGRecord.objects.exclude(label=None).select_related('label')
        )
        if not labeled_records.exists():
            return Response({'error': 'No human-labeled records found for analysis.'}, status=400)

        # 2. Get all available labels for the matrix axis
        all_labels = list(ECGLabel.objects.all().order_by('value'))
        label_map = {lbl.value: i for i, lbl in enumerate(all_labels)}
        label_names = [lbl.name for lbl in all_labels]
        num_labels = len(all_labels)

        # 3. Preparation
        analysis_results = {}
        ground_truth = [rec.label.value for rec in labeled_records]
        
        # Cache waves to avoid repeated DB hits
        waves_data = []
        for rec in labeled_records:
            w = rec.ecg_wave
            if isinstance(w, str):
                w = [float(v) for v in w.split(',')]
            waves_data.append(w)

        # 4. Loop through each registered model
        for model_id, model_info in get_visible_model_runtime_catalog_for_user(request.user).items():
            abs_path = resolve_model_path(model_info)
            if not os.path.exists(abs_path):
                continue
            
            try:
                # Load model if not in memory
                if model_id not in loaded_models:
                    mc = model_info['class']
                    kw = model_info.get('kwargs', {})
                    if mc in SKLEARN_WRAPPERS:
                        m = mc(model_path=abs_path, **kw)
                    else:
                        m = mc(model_info['num_classes'], **kw)
                        m.load_state_dict(torch.load(abs_path, map_location='cpu'))
                    if hasattr(m, 'eval'):
                        m.eval()
                    loaded_models[model_id] = m
                
                model = loaded_models[model_id]
                model_type = model_info['class'].__name__

                confusion_matrix = [[0] * num_labels for _ in range(num_labels)]
                correct = 0
                evaluated_total = 0

                for i, wave in enumerate(waves_data):
                    if len(wave) != model_info['input_size']:
                        continue
                    evaluated_total += 1
                    t = torch.tensor(wave, dtype=torch.float32).unsqueeze(0)
                    if model_type in ('ECG1DCNN', 'ECGCNNLSTMHybrid', 'ECGResNet1D'):
                        t = t.unsqueeze(0)
                    
                    with torch.no_grad():
                        output = model(t)
                        probs_tensor = as_probabilities(output)
                        pred = torch.argmax(probs_tensor, dim=1).item()
                    
                    actual = ground_truth[i]
                    if pred == actual:
                        correct += 1
                    
                    if actual in label_map and pred in label_map:
                        confusion_matrix[label_map[actual]][label_map[pred]] += 1

                if evaluated_total == 0:
                    analysis_results[model_id] = {'error': 'No records matched this model input size.'}
                    continue

                accuracy = correct / evaluated_total if evaluated_total else 0
                
                analysis_results[model_id] = {
                    'label': model_info.get('label', model_id),
                    'accuracy': round(accuracy, 4),
                    'correct': correct,
                    'total': evaluated_total,
                    'confusion_matrix': confusion_matrix,
                }

            except Exception as e:
                logger.error(f"Analysis failed for {model_id}: {str(e)}")
                analysis_results[model_id] = {'error': str(e)}

        return Response({
            'sample_size': len(ground_truth),
            'label_names': label_names,
            'results': analysis_results
        })


# ---------------------------
# Verify Label
# ---------------------------
class VerifyLabelView(APIView):
    """
    POST { "record_id": 123, "verified": true }
    Mark a record as human-verified. Optionally override the label.
    POST { "record_id": 123, "verified": true, "label": 0 }  -- also set the human label
    """

    permission_classes = [IsAuthenticated, IsDoctorOrAdmin]

    def post(self, request):
        record_id = request.data.get('record_id')
        verified = request.data.get('verified', True)
        label_value = request.data.get('label')  # optional

        try:
            record = scope_records_for_user(request.user, ECGRecord.objects).get(id=record_id)
        except ECGRecord.DoesNotExist:
            return Response({'error': 'Record not found.'}, status=404)

        fields_to_update = ['is_verified']
        record.is_verified = bool(verified)

        if label_value is not None:
            try:
                label_obj = ECGLabel.objects.get(value=label_value)
                record.label = label_obj
                record.labeled_by = 'human'
                fields_to_update += ['label', 'labeled_by']
            except ECGLabel.DoesNotExist:
                return Response({'error': f'Label value {label_value} not found.'}, status=400)

        record.save(update_fields=fields_to_update)

        return Response({
            'record_id': record_id,
            'is_verified': record.is_verified,
            'label': record.label.name if record.label else None,
            'labeled_by': record.labeled_by,
        })


# ─── Training job store (in-process, good for single-worker dev server) ──────
# { job_id: { 'status': 'running'|'done'|'error', 'logs': [...], 'model': '...' } }
_TRAINING_JOBS: dict = {}
_JOBS_LOCK = threading.Lock()


class TrainModelView(APIView):
    """
    POST /api/train/
      Body: { "model": "ECG1DCNN", "epochs": 30, "lr": 0.001, "batch": 64,
              "use_ai_labels": false }
    Launches training in a background thread, returns job_id immediately.
    Weights saved to api/models/best_<modelname>.pth (same path MODEL_MAP uses).

    GET /api/train/<job_id>/status/
    Returns: { "status": "running"|"done"|"error", "logs": [...] }
    """

    permission_classes = [IsAuthenticated, IsAdminRole]

    def post(self, request):
        model_name = request.data.get('model', 'ECG1DCNN')
        patient_id = request.data.get('patient_id')
        label_filters = request.data.get('labels', [])
        max_records = request.data.get('max_records')
        epochs = int(request.data.get('epochs', 30))
        lr = float(request.data.get('lr', 1e-3))
        batch = int(request.data.get('batch', 64))
        use_ai_labels = bool(request.data.get('use_ai_labels', False))

        try:
            max_records = int(max_records) if max_records not in (None, "", 0, "0") else None
        except (TypeError, ValueError):
            return Response({'error': 'max_records must be an integer.'}, status=400)
        if label_filters is not None and not isinstance(label_filters, list):
            return Response({'error': 'labels must be a list.'}, status=400)
        if isinstance(label_filters, list) and len(label_filters) == 0:
            return Response({'error': 'Choose at least one label for training.'}, status=400)

        model_info = get_model_runtime_info(model_name)
        if not model_info:
            return Response({'error': f'Unknown model "{model_name}".'}, status=400)
        if model_info['class'] in SKLEARN_WRAPPERS:
            return Response({'error': f'Model "{model_name}" is inference-only and cannot be trained with this endpoint.'}, status=400)
        if epochs < 1 or epochs > 500:
            return Response({'error': 'epochs must be 1-500'}, status=400)
        if max_records is not None and max_records < 2:
            return Response({'error': 'max_records must be at least 2 when provided.'}, status=400)

        job_id = str(uuid.uuid4())[:8]
        with _JOBS_LOCK:
            _TRAINING_JOBS[job_id] = {
                'status': 'running',
                'logs': [],
                'model': model_name,
                'patient_id': patient_id,
                'labels': label_filters,
                'max_records': max_records,
            }

        thread = threading.Thread(
            target=self._run_training,
            args=(job_id, model_name, patient_id, label_filters, max_records, epochs, lr, batch, use_ai_labels),
            daemon=True,
        )
        thread.start()

        return Response({
            'job_id': job_id,
            'model': model_name,
            'patient_id': patient_id,
            'labels': label_filters,
            'max_records': max_records,
            'message': 'Training started in background. Poll /api/train/{job_id}/status/ for progress.',
        }, status=202)

    @staticmethod
    def _run_training(job_id, model_name, patient_id, label_filters, max_records, epochs, lr, batch_size, use_ai_labels):
        """Heavy work — runs in a daemon thread, never touches the HTTP cycle."""
        import numpy as np
        import torch
        import torch.nn as nn
        from sklearn.model_selection import train_test_split
        from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

        def log(msg):
            with _JOBS_LOCK:
                _TRAINING_JOBS[job_id]['logs'].append(msg)
            logger.info('[Train %s] %s', job_id, msg)

        try:
            model_info = get_model_runtime_info(model_name)
            if not model_info:
                raise ValueError(f'Model "{model_name}" is no longer registered.')
            input_size = model_info['input_size']
            num_classes = model_info['num_classes']
            model_type = model_info['class'].__name__
            save_path = resolve_model_path(model_info)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # ── Load data from DB ─────────────────────────────────────────────
            log(
                'Loading records from DB '
                f'(use_ai_labels={use_ai_labels}, patient_id={patient_id or "ALL"}, '
                f'label_filters={label_filters or "ALL"}, max_records={max_records or "ALL"}) ...'
            )
            if use_ai_labels:
                qs = ECGRecord.objects.exclude(ai_label__isnull=True).select_related('ai_label').only('ecg_wave', 'ai_label__value', 'ai_label_id', 'created_at', 'patient_id')
                get_label_val = lambda r: r.ai_label.value
                get_label_db_id = lambda r: r.ai_label_id
            else:
                qs = ECGRecord.objects.exclude(label__isnull=True).select_related('label').only('ecg_wave', 'label__value', 'label_id', 'created_at', 'patient_id')
                get_label_val = lambda r: r.label.value
                get_label_db_id = lambda r: r.label_id
            if patient_id:
                qs = qs.filter(patient_id=str(patient_id))
            selected_label_filters = [
                {"id": int(item["id"]), "numRecords": int(item.get("numRecords", 0))}
                for item in (label_filters or [])
                if item.get("id") is not None and int(item.get("numRecords", 0)) > 0
            ]
            selected_label_ids = [item["id"] for item in selected_label_filters]
            if selected_label_ids:
                filter_field = "ai_label_id__in" if use_ai_labels else "label_id__in"
                qs = qs.filter(**{filter_field: selected_label_ids})
            label_limits = {item["id"]: item["numRecords"] for item in selected_label_filters}
            label_counts = defaultdict(int)

            waves, label_vals = [], []
            skipped = 0
            total_selected = 0
            for rec in qs.order_by('created_at', 'id'):
                label_db_id = get_label_db_id(rec)
                if label_limits and label_counts[label_db_id] >= label_limits.get(label_db_id, float("inf")):
                    continue
                if max_records is not None and total_selected >= max_records:
                    break
                wave = rec.ecg_wave
                if isinstance(wave, str):
                    wave = [float(v) for v in wave.split(',')]
                if len(wave) != input_size:
                    skipped += 1
                    continue
                waves.append(wave)
                label_vals.append(get_label_val(rec))
                label_counts[label_db_id] += 1
                total_selected += 1

            log(f'Loaded {len(waves)} records | Skipped {skipped} (wrong length)')
            if len(waves) < 20:
                raise ValueError(f'Not enough labeled records to train ({len(waves)} found, need at least 20).')

            # Remap labels to 0-based
            unique_labels = sorted(set(label_vals))
            if len(unique_labels) < 2:
                raise ValueError('Training requires at least 2 distinct labels after applying patient/label filters.')
            raw_label_counts = {label: label_vals.count(label) for label in unique_labels}
            too_small = [str(label) for label, count in raw_label_counts.items() if count < 2]
            if too_small:
                raise ValueError(
                    'Each selected class needs at least 2 records for train/validation split. '
                    f'Classes too small: {", ".join(too_small)}.'
                )
            remap = {v: i for i, v in enumerate(unique_labels)}
            labels_arr = np.array([remap[v] for v in label_vals], dtype=np.int64)
            waves_arr = np.array(waves, dtype=np.float32)
            log(f'Classes: {unique_labels} -> 0..{len(unique_labels)-1}')

            X_tr, X_val, y_tr, y_val = train_test_split(
                waves_arr, labels_arr, test_size=0.2, random_state=42, stratify=labels_arr
            )

            class _DS(Dataset):
                def __init__(self, X, y):
                    self.X, self.y = X, y
                def __len__(self): return len(self.X)
                def __getitem__(self, i):
                    w = torch.tensor(self.X[i], dtype=torch.float32)
                    if model_type in ('ECG1DCNN', 'ECGCNNLSTMHybrid', 'ECGResNet1D'):
                        w = w.unsqueeze(0)
                    return w, torch.tensor(self.y[i], dtype=torch.long)

            counts = np.bincount(y_tr)
            sample_weights = 1.0 / counts[y_tr]
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
            tr_loader = DataLoader(_DS(X_tr, y_tr), batch_size=batch_size, sampler=sampler, num_workers=0)
            val_loader = DataLoader(_DS(X_val, y_val), batch_size=batch_size, shuffle=False, num_workers=0)

            # ── Model & optimizer ─────────────────────────────────────────────
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            log(f'Device: {device} | Model: {model_name} | Epochs: {epochs} | LR: {lr} | Batch: {batch_size}')

            mc = model_info['class']
            kw = model_info.get('kwargs', {})
            actual_classes = len(unique_labels)
            model = mc(actual_classes, **kw).to(device)
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            log(f'Parameters: {param_count:,}')

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

            best_val_acc = 0.0
            best_epoch = 0

            # ── Training loop ─────────────────────────────────────────────────
            for epoch in range(1, epochs + 1):
                model.train()
                tr_correct = tr_total = 0
                tr_loss = 0.0
                for X, y in tr_loader:
                    X, y = X.to(device), y.to(device)
                    optimizer.zero_grad()
                    out = model(X)
                    loss = criterion(out, y)
                    loss.backward()
                    optimizer.step()
                    tr_loss += loss.item() * X.size(0)
                    tr_correct += (out.argmax(1) == y).sum().item()
                    tr_total += X.size(0)
                scheduler.step()

                model.eval()
                val_correct = val_total = 0
                with torch.no_grad():
                    for X, y in val_loader:
                        X, y = X.to(device), y.to(device)
                        out = model(X)
                        val_correct += (out.argmax(1) == y).sum().item()
                        val_total += X.size(0)

                val_acc = val_correct / val_total if val_total else 0
                tr_acc = tr_correct / tr_total if tr_total else 0
                is_best = val_acc > best_val_acc
                if is_best:
                    best_val_acc = val_acc
                    best_epoch = epoch
                    torch.save(model.state_dict(), save_path)

                log(f'Epoch {epoch:3d}/{epochs} | Loss {tr_loss/tr_total:.4f} | '
                    f'TrainAcc {tr_acc:.4f} | ValAcc {val_acc:.4f}'
                    + (' [BEST]' if is_best else ''))

            log(f'Training complete. Best val acc: {best_val_acc:.4f} at epoch {best_epoch}')
            log(f'Saved to: {save_path}')

            if actual_classes != num_classes:
                try:
                    model_obj = DeployedModel.objects.filter(key=model_name, is_active=True).first()
                    if model_obj and model_obj.num_classes != actual_classes:
                        model_obj.num_classes = actual_classes
                        model_obj.save(update_fields=['num_classes'])
                        log(f'Updated deployed model num_classes to {actual_classes}.')
                except Exception as e:
                    log(f'Warning: unable to persist updated num_classes: {e}')

            # Reload the newly trained model into loaded_models cache
            kw = model_info.get('kwargs', {})
            trained_model = mc(actual_classes, **kw)
            trained_model.load_state_dict(torch.load(save_path, map_location='cpu'))
            trained_model.eval()
            loaded_models[model_name] = trained_model
            log('Model reloaded into cache, ready for prediction.')

            with _JOBS_LOCK:
                _TRAINING_JOBS[job_id]['status'] = 'done'

        except Exception as e:
            logger.exception('Training job %s failed', job_id)
            with _JOBS_LOCK:
                _TRAINING_JOBS[job_id]['status'] = 'error'
                _TRAINING_JOBS[job_id]['logs'].append(f'ERROR: {e}')


class TrainJobListView(APIView):
    """GET /api/train/jobs/ — returns list of recent job IDs and models."""
    permission_classes = [IsAuthenticated, IsAdminRole]

    def get(self, request):
        with _JOBS_LOCK:
            jobs = [
                {
                    'job_id': jid,
                    'model': info['model'],
                    'status': info['status'],
                    'patient_id': info.get('patient_id'),
                    'labels': info.get('labels', []),
                    'max_records': info.get('max_records'),
                }
                for jid, info in _TRAINING_JOBS.items()
            ]
        return Response(jobs)


class TrainStatusView(APIView):
    """GET /api/train/<job_id>/status/"""
    permission_classes = [IsAuthenticated, IsAdminRole]

    def get(self, request, job_id):
        with _JOBS_LOCK:
            job = _TRAINING_JOBS.get(job_id)
        if not job:
            return Response({'error': 'Job not found.'}, status=404)
        return Response({
            'job_id': job_id,
            'model': job['model'],
            'status': job['status'],
            'logs': job['logs'],
            'patient_id': job.get('patient_id'),
            'labels': job.get('labels', []),
            'max_records': job.get('max_records'),
        })


# Config
DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1U7dPo0R20KonIRP46NsBRFREbAJ__O6A"
class DriveModelListView(APIView):
    permission_classes = [IsAuthenticated, IsAdminRole]

    def get(self, request):
        target_folder = request.query_params.get("target_folder", LOCAL_MODELS_FOLDER)
        os.makedirs(target_folder, exist_ok=True)

        # Download Drive folder contents quietly
        gdown.download_folder(DRIVE_FOLDER_URL, output=target_folder, quiet=True, use_cookies=False)

        # Find all .pth files recursively
        pth_files = glob.glob(os.path.join(target_folder, "**", "*.pth"), recursive=True)
        file_names = [os.path.basename(f) for f in pth_files]

        if not file_names:
            return Response({"error": "No .pth files found."}, status=status.HTTP_404_NOT_FOUND)

        return Response({"models": file_names, "folder": target_folder}, status=status.HTTP_200_OK)

    def post(self, request):
        filename = request.data.get("filename")
        target_folder = request.data.get("target_folder", LOCAL_MODELS_FOLDER)
        os.makedirs(target_folder, exist_ok=True)

        if not filename:
            return Response({"error": "Filename not provided"}, status=status.HTTP_400_BAD_REQUEST)

        local_file_path = os.path.join(target_folder, filename)

        # Always download/overwrite the file
        gdown.download_folder(DRIVE_FOLDER_URL, output=target_folder, quiet=True, use_cookies=False)

        # Verify the file exists now
        if not os.path.exists(local_file_path):
            return Response({"error": f"{filename} not found in Drive folder."}, status=status.HTTP_404_NOT_FOUND)

        return Response({
            "file_path": local_file_path,
            "folder": target_folder,
            "message": f"{filename} is ready and updated."
        }, status=status.HTTP_200_OK)


@csrf_exempt
def dialogflow_webhook(request):
    if request.method == "POST":
        body = json.loads(request.body)
        
        intent = body["queryResult"]["intent"]["displayName"]
        params = body["queryResult"]["parameters"]

        # ------------------------------
        # Intent: Get_All_Models
        # ------------------------------
        if intent == "Get_All_Models":
            models = ["CNN_Model", "LSTM_Model", "HybridNet"]
            return JsonResponse({
                "fulfillmentText": f"Available models: {', '.join(models)}."
            })

        # ------------------------------
        # Intent: Get_Model_Description
        # ------------------------------
        if intent == "Get_Model_Description":
            model_name = params.get("model_name")

            # Dummy — replace with actual DB or logic
            descriptions = {
                "CNN_Model": "CNN_Model is a 12-layer CNN trained on 20k ECG samples.",
                "LSTM_Model": "LSTM_Model uses recurrent layers and classifies 7 classes.",
                "HybridNet": "HybridNet combines CNN + LSTM for arrhythmia analysis."
            }

            desc = descriptions.get(model_name, "No description found.")
            return JsonResponse({"fulfillmentText": desc})

        # ------------------------------
        # Intent: Get_Model_Classes
        # ------------------------------
        if intent == "Get_Model_Classes":
            model_name = params.get("model_name")

            # Dummy classes
            classes = {
                "CNN_Model": ["Normal", "AF", "PVC", "LBBB", "RBBB"],
                "LSTM_Model": ["Normal", "AF", "VT", "VF"],
                "HybridNet": ["Normal", "AF", "PVC", "PAC", "LBBB"]
            }

            class_list = classes.get(model_name, [])
            return JsonResponse({
                "fulfillmentText": f"{model_name} predicts: {', '.join(class_list)}"
            })

        # Fallback
        return JsonResponse({
            "fulfillmentText": "Sorry, I couldn't process that request."
        })

    return JsonResponse({"message": "Only POST allowed"})
