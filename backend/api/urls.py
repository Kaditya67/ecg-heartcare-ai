from django.urls import path
from .views import FileUploadView, ECGRecordListView

urlpatterns = [
    path("upload/", FileUploadView.as_view(), name="file-upload"),
    path("records/", ECGRecordListView.as_view(), name="record-list"),
]
