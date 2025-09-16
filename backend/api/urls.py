from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    FileUploadView,
    ECGRecordListView,
    ECGRecordDetailView,
    get_ecg_wave_view,
    bulk_label_update_view,
    ECGFileViewSet,
    LabelCountView,
    PatientCountView,
    ECGCustomExportView,
    ECGFileSummaryView,
    LabelsPatientsByFilesView,
    DashboardSummaryView,
    PredictECGView,
    ModelListView
)
from .views import RegisterView, LoginView

router = DefaultRouter()
router.register(r'ecgfiles', ECGFileViewSet, basename='ecgfile')

urlpatterns = [
    path('', include(router.urls)),
    path('upload/', FileUploadView.as_view(), name='file-upload'),
    path('records/', ECGRecordListView.as_view(), name='record-list'),
    path('records/<int:pk>/', ECGRecordDetailView.as_view(), name='record-detail'),
    path('records/<int:record_id>/wave/', get_ecg_wave_view, name='get_ecg_wave'),
    path('records/bulk-label/', bulk_label_update_view, name='bulk-label'),
    path('labels/count/', LabelCountView.as_view(), name='labels-count'),
    path('patients/count/', PatientCountView.as_view(), name='patients-count'),
    path('ecgrecords/export/', ECGCustomExportView.as_view(), name='ecg-custom-export'),
    path('files/summary/', ECGFileSummaryView.as_view(), name='ecgfile-summary'),  # Use this to fetch file list and summary
    path('ecgrecords/labels-patients-by-files/', LabelsPatientsByFilesView.as_view(), name='labels-patients-by-files'),
    path('dashboard/summary/', DashboardSummaryView.as_view(), name='dashboard-summary'),
    path('register/', RegisterView.as_view(), name='register'),
    path('login/', LoginView.as_view(), name='login'),
    path('predict-ecg/', PredictECGView.as_view(), name='predict-ecg'),
    path('model_list/', ModelListView.as_view(), name='model-list')
]