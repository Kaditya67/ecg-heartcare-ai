import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
django.setup()

import pandas as pd
import re
import io
from api.models import ECGLabel

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

def normalize_columns(df):
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

# Test with normal CSV
csv_content = """Patient ID,Heart Rate,ECG Wave,Label
723,84,"1,2,3,4",0
724,90,"5,6,7,8",1"""

df = pd.read_csv(io.StringIO(csv_content))
print("Original columns:", list(df.columns))
df = normalize_columns(df)
print("After normalize:", list(df.columns))
print("Missing:", REQUIRED_UPLOAD_COLUMNS - set(df.columns))

# Test with BOM
csv_bom = "\ufeffPatient ID,Heart Rate,ECG Wave,Label\n723,84,\"1,2,3,4\",0"
df2 = pd.read_csv(io.StringIO(csv_bom))
print("BOM columns:", list(df2.columns))
df2 = normalize_columns(df2)
print("BOM after normalize:", list(df2.columns))
print("BOM Missing:", REQUIRED_UPLOAD_COLUMNS - set(df2.columns))

# Test with trailing spaces
csv_spaces = """Patient ID , Heart Rate , ECG Wave , Label 
723,84,"1,2,3,4",0"""
df3 = pd.read_csv(io.StringIO(csv_spaces))
print("Spaces columns:", list(df3.columns))
df3 = normalize_columns(df3)
print("Spaces after normalize:", list(df3.columns))
print("Spaces Missing:", REQUIRED_UPLOAD_COLUMNS - set(df3.columns))