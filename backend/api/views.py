from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, generics
from django.db import transaction
import pandas as pd
from .models import ECGFile, ECGRecord
from .serializers import ECGRecordSerializer, ECGFileSerializer

class FileUploadView(APIView):
    def post(self, request, format=None):
        file = request.FILES.get("file")
        if not file:
            return Response({"error": "No file provided"}, status=status.HTTP_400_BAD_REQUEST)

        # Save file record
        ecg_file = ECGFile.objects.create(file_name=file.name, status="processing")

        try:
            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
            elif file.name.endswith((".xls", ".xlsx")):
                df = pd.read_excel(file)
            else:
                return Response({"error": "Invalid file format"}, status=status.HTTP_400_BAD_REQUEST)

            # Expected columns: patient_id, ecgWave, Heartrate, Label
            required_cols = {"patient_id", "ecgWave", "Heartrate", "Label"}
            if not required_cols.issubset(df.columns):
                return Response({"error": f"Missing columns. Required: {required_cols}"}, status=400)

            records = []
            for _, row in df.iterrows():
                records.append(ECGRecord(
                    file=ecg_file,
                    patient_id=row["patient_id"],
                    ecg_wave=row["ecgWave"],   # Assume already JSON/list-like
                    heart_rate=row["Heartrate"],
                    label=int(row["Label"]) if not pd.isna(row["Label"]) else None,
                ))

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


class ECGRecordListView(generics.ListAPIView):
    queryset = ECGRecord.objects.all().order_by("id")
    serializer_class = ECGRecordSerializer
