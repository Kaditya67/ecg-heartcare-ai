import json

from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework.test import APITestCase

from api.models import ECGFile, ECGLabel, ECGRecord, Profile


class FileUploadBackendTests(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="admin_test",
            password="pass1234",
            is_staff=True,
            is_superuser=True,
        )
        Profile.objects.update_or_create(
            user=self.user,
            defaults={"role": Profile.ROLE_ADMIN, "is_authorized": True},
        )
        for value, name in [
            (-1, "Noise"),
            (0, "Normal"),
            (1, "PAC"),
            (2, "PVC"),
            (3, "Sinus Bradycardia"),
        ]:
            ECGLabel.objects.create(value=value, name=name, color="#000000")

        self.client.force_authenticate(user=self.user)

    def upload_file(self, name, content, extra_data=None):
        payload = {
            "file": SimpleUploadedFile(name, content.encode("utf-8"), content_type="text/csv"),
        }
        if extra_data:
            payload.update(extra_data)
        return self.client.post("/api/upload/", payload, format="multipart")

    def test_upload_accepts_known_alias_headers_without_manual_mapping(self):
        csv_content = "\n".join([
            "Patient ID,Heart Rate,ECG Wave,Label",
            '723,84,"1,2,3,4",0',
            '724,90,"5,6,7,8",1',
        ])

        response = self.upload_file("alias_headers.csv", csv_content)

        self.assertEqual(response.status_code, 201, response.data)
        self.assertEqual(ECGFile.objects.count(), 1)
        self.assertEqual(ECGRecord.objects.count(), 2)
        first = ECGRecord.objects.order_by("id").first()
        self.assertEqual(first.patient_id, "723")
        self.assertEqual(first.heart_rate, 84.0)
        self.assertEqual(first.label.value, 0)

    def test_upload_accepts_manual_mapping_and_numeric_label_strings(self):
        csv_content = "\n".join([
            "PID,HR_BPM,Wave Values,Diagnosis",
            '723,84,"1,2,3,4",0.0',
            '724,90,"5,6,7,8",2',
        ])
        mapping = {
            "patient_id": "PID",
            "heart_rate": "HR_BPM",
            "ecg_wave": "Wave Values",
            "label": "Diagnosis",
        }

        response = self.upload_file(
            "manual_mapping.csv",
            csv_content,
            {
                "column_mapping": json.dumps(mapping),
                "column_mapping_patient_id": "PID",
                "column_mapping_heart_rate": "HR_BPM",
                "column_mapping_ecg_wave": "Wave Values",
                "column_mapping_label": "Diagnosis",
            },
        )

        self.assertEqual(response.status_code, 201, response.data)
        records = list(ECGRecord.objects.order_by("id"))
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].label.value, 0)
        self.assertEqual(records[1].label.value, 2)

    def test_upload_returns_mapping_help_for_unknown_headers(self):
        csv_content = "\n".join([
            "PID,HR_BPM,Wave Values,Diagnosis",
            '723,84,"1,2,3,4",0',
        ])

        response = self.upload_file("needs_mapping.csv", csv_content)

        self.assertEqual(response.status_code, 400)
        self.assertIn("missing_required_columns", response.data)
        self.assertEqual(set(response.data["missing_required_columns"]), {"patient_id", "heart_rate", "ecg_wave"})

    def test_patient_model_list_shows_only_assigned_model(self):
        from api.models import DeployedModel, PatientModelAssignment
        from api.views import ensure_builtin_models

        ensure_builtin_models()
        patient_user = User.objects.create_user(username="patient_view", password="pass1234")
        Profile.objects.update_or_create(
            user=patient_user,
            defaults={"role": Profile.ROLE_PATIENT, "patient_id": "723", "is_authorized": True},
        )
        assigned_model = DeployedModel.objects.get(key="ECG1DCNN")
        PatientModelAssignment.objects.create(patient_id="723", model=assigned_model)

        self.client.force_authenticate(user=patient_user)
        response = self.client.get("/api/model_list/")

        self.assertEqual(response.status_code, 200, response.data)
        self.assertEqual(set(response.data["models"].keys()), {"ECG1DCNN"})
        self.assertEqual(response.data["patient_assignment"]["patient_id"], "723")
