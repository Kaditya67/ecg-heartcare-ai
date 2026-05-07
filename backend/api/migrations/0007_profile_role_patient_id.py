from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("api", "0006_ecgrecord_ai_confidence_ecgrecord_ai_probabilities"),
    ]

    operations = [
        migrations.AddField(
            model_name="profile",
            name="patient_id",
            field=models.CharField(
                blank=True,
                help_text="Required when the user is a patient. Limits access to this patient's records.",
                max_length=100,
                null=True,
            ),
        ),
        migrations.AddField(
            model_name="profile",
            name="role",
            field=models.CharField(
                choices=[("admin", "Admin"), ("doctor", "Doctor"), ("patient", "Patient")],
                default="doctor",
                max_length=20,
            ),
        ),
    ]
