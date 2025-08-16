from django.urls import path
from .views import FileUploadView, ECGRecordListView, ECGRecordDetailView

urlpatterns = [
    path("upload/", FileUploadView.as_view(), name="file-upload"),
    path("records/", ECGRecordListView.as_view(), name="record-list"),
    path("records/<int:pk>/", ECGRecordDetailView.as_view(), name="record-detail"),
]
