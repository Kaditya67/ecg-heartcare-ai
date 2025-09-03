from .model_architecture import ECG1DCNN 

MODEL_MAP = {
    'ECG1DCNN': {
        'class': ECG1DCNN,
        'path': 'api/models/best_ecg_1dcnn.pth', 
        'input_size': 2604,
        'num_classes': 4,
    },
    # Add other models here
}
