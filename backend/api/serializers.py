from django.contrib.auth.models import User
from rest_framework import serializers
from rest_framework.reverse import reverse
from .models import ECGRecord, ECGFile, ECGLabel, Profile, DeployedModel, PatientModelAssignment


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


class DeployedModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = DeployedModel
        fields = [
            "id", "key", "label", "base_model_key", "source_type",
            "weights_path", "input_size", "num_classes", "trainable",
            "is_active", "uploaded_at", "updated_at",
        ]


class PatientModelAssignmentSerializer(serializers.ModelSerializer):
    model = DeployedModelSerializer(read_only=True)

    class Meta:
        model = PatientModelAssignment
        fields = ["id", "patient_id", "model", "assigned_at", "updated_at"]


class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)
    role = serializers.ChoiceField(
        choices=[Profile.ROLE_DOCTOR, Profile.ROLE_PATIENT],
        required=False,
        default=Profile.ROLE_DOCTOR,
    )
    patient_id = serializers.CharField(required=False, allow_blank=False)

    class Meta:
        model = User
        fields = ['username', 'password', 'role', 'patient_id']

    def validate(self, attrs):
        role = attrs.get('role', Profile.ROLE_DOCTOR)
        patient_id = attrs.get('patient_id')

        if role == Profile.ROLE_PATIENT and not patient_id:
            raise serializers.ValidationError({'patient_id': 'Patient ID is required for patient accounts.'})

        if role != Profile.ROLE_PATIENT:
            attrs.pop('patient_id', None)

        return attrs

    def create(self, validated_data):
        role = validated_data.pop('role', Profile.ROLE_DOCTOR)
        patient_id = validated_data.pop('patient_id', None)
        user = User.objects.create_user(
            username=validated_data['username'],
            password=validated_data['password']
        )
        profile = user.profile
        profile.role = role
        profile.patient_id = patient_id if role == Profile.ROLE_PATIENT else None
        profile.save()
        return user


class ProfileSerializer(serializers.ModelSerializer):
    role = serializers.SerializerMethodField()

    class Meta:
        model = Profile
        fields = [
            'role',
            'patient_id',
            'is_authorized',
            'show_missing_weights',
            'default_plot_library',
        ]
        extra_kwargs = {
            'role': {'read_only': True},
            'patient_id': {'read_only': True},
            'is_authorized': {'read_only': True},
        }

    def get_role(self, obj):
        return obj.effective_role


class UserSerializer(serializers.ModelSerializer):
    profile = ProfileSerializer(read_only=True)

    class Meta:
        model = User
        fields = ['id', 'username', 'is_staff', 'is_superuser', 'profile']
