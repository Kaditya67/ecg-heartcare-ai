from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("api", "0007_profile_role_patient_id"),
    ]

    operations = [
        migrations.CreateModel(
            name="DeployedModel",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("key", models.CharField(max_length=100, unique=True)),
                ("label", models.CharField(max_length=150)),
                ("base_model_key", models.CharField(max_length=100)),
                ("source_type", models.CharField(choices=[("builtin", "Built In"), ("uploaded", "Uploaded")], default="builtin", max_length=20)),
                ("weights_path", models.CharField(max_length=255)),
                ("input_size", models.IntegerField()),
                ("num_classes", models.IntegerField()),
                ("trainable", models.BooleanField(default=True)),
                ("is_active", models.BooleanField(default=True)),
                ("uploaded_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
        ),
        migrations.CreateModel(
            name="PatientModelAssignment",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("patient_id", models.CharField(db_index=True, max_length=100, unique=True)),
                ("assigned_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("model", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="patient_assignments", to="api.deployedmodel")),
            ],
        ),
        migrations.AddIndex(
            model_name="deployedmodel",
            index=models.Index(fields=["key"], name="api_deploye_key_bfef24_idx"),
        ),
        migrations.AddIndex(
            model_name="deployedmodel",
            index=models.Index(fields=["base_model_key"], name="api_deploye_base_m_9ae810_idx"),
        ),
        migrations.AddIndex(
            model_name="deployedmodel",
            index=models.Index(fields=["source_type", "is_active"], name="api_deploye_source__f8f7aa_idx"),
        ),
        migrations.AddIndex(
            model_name="patientmodelassignment",
            index=models.Index(fields=["patient_id"], name="api_patien_patient_9b9a34_idx"),
        ),
    ]
