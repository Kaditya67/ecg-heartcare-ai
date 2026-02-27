from api.Model_architechture.ECG1DCNN import ECG1DCNN
from api.Model_architechture.ECGLSTM import ECGLSTM
from api.Model_architechture.ECGCNNLSTMHybrid import ECGCNNLSTMHybrid
from api.Model_architechture.ECGResNet1D import ECGResNet1D
from api.Model_architechture.ECGBERT import ECGBERT
from api.Model_architechture.MLModels import ECGXGBoostWrapper, AdaBoostECGWrapper, RandomForestECGWrapper

# ─── Model Registry ─────────────────────────────────────────────────────────
# Each entry specifies the class, path to .pth weights, input size, and
# number of output classes.  The 'label' field is what shows in the frontend.
MODEL_MAP = {
    'ECG1DCNN': {
        'class': ECG1DCNN,
        'path': 'api/models/best_ecg_1dcnn.pth',
        'input_size': 2604,
        'num_classes': 4,
        'label': '1D-CNN (Baseline)',
    },
    'ECGLSTM': {
        'class': ECGLSTM,
        'path': 'api/models/best_ecg_lstm.pth',
        'input_size': 2604,
        'num_classes': 4,
        'label': 'BiLSTM + Attention',
        # Extra kwargs forwarded to model __init__
        'kwargs': {'hidden_dim': 128, 'num_layers': 2, 'seq_len': 52, 'feat_dim': 50},
    },
    'ECGCNNLSTMHybrid': {
        'class': ECGCNNLSTMHybrid,
        'path': 'api/models/best_ecg_cnn_lstm.pth',
        'input_size': 2604,
        'num_classes': 4,
        'label': 'CNN-LSTM Hybrid (Best)',
        'kwargs': {'lstm_hidden': 128, 'lstm_layers': 2},
    },
    'ECGResNet1D': {
        'class': ECGResNet1D,
        'path': 'api/models/best_ecg_resnet.pth',
        'input_size': 2604,
        'num_classes': 4,
        'label': '1D-ResNet',
    },
    'ECGBERT': {
        'class': ECGBERT,
        'path': 'api/models/best_bert_ecg.pth',
        'input_size': 2000,
        'num_classes': 5,
        'label': 'ECG-BERT (Transformer)',
    },
    'ECGXGBoost': {
        'class': ECGXGBoostWrapper,
        'path': 'api/models/ecg_xgboost.pth',
        'input_size': 2604,
        'num_classes': 4,
        'label': 'XGBoost Optimizer',
    },
    'ECGAdaBoost': {
        'class': AdaBoostECGWrapper,
        'path': 'api/models/ecg_adaboost.pkl',
        'input_size': 2604,
        'num_classes': 4,
        'label': 'AdaBoost Classifier',
    },
    'ECGRandomForest': {
        'class': RandomForestECGWrapper,
        'path': 'api/models/ecg_randomforest.pkl',
        'input_size': 2604,
        'num_classes': 2,  # Binary: 0=Normal, 1=Abnormal
        'label': 'Random Forest (SMOTE-Balanced)',
    },
}
