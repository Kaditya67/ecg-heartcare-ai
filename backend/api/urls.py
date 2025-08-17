from django.urls import path
from .views import FileUploadView, ECGRecordListView, ECGRecordDetailView, get_ecg_wave_view, bulk_label_update_view

urlpatterns = [
    path("upload/", FileUploadView.as_view(), name="file-upload"),
    path("records/", ECGRecordListView.as_view(), name="record-list"),
    path("records/<int:pk>/", ECGRecordDetailView.as_view(), name="record-detail"),
    path("records/<int:record_id>/wave/", get_ecg_wave_view, name="get_ecg_wave"),
    path("records/bulk-label/", bulk_label_update_view, name="bulk-label"),
]
    