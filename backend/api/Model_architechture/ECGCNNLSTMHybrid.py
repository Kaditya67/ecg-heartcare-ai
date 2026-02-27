"""
ECG CNN-LSTM Hybrid Model — combines spatial and temporal feature extraction.
Architecture: 3× Conv1D blocks → reshape → BiLSTM → FC
This is typically the best performer on raw ECG waveforms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ECGCNNLSTMHybrid(nn.Module):
    """
    CNN extracts local morphological features from the ECG waveform,
    then LSTM captures temporal dynamics across the extracted feature maps.

    Input: (batch, 1, 2604) — 1D time series
    """

    def __init__(self, num_classes: int, lstm_hidden: int = 128,
                 lstm_layers: int = 2, dropout: float = 0.3):
        super().__init__()

        # ─── CNN Feature Extractor ────────────────────────────────────────
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)

        # After 3× pool(2): 2604 → 1302 → 651 → 325 time steps, 128 channels
        # LSTM sees 325 time steps of 128 features each

        # ─── BiLSTM Temporal Modelling ────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # ─── Classifier ──────────────────────────────────────────────────
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_hidden * 2, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (batch, 1, 2604)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))  # (batch, 32, 1302)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))  # (batch, 64, 651)
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))  # (batch, 128, 325)

        # Transpose for LSTM: (batch, time, features)
        x = x.permute(0, 2, 1)                           # (batch, 325, 128)

        lstm_out, (h_n, _) = self.lstm(x)

        # Use last hidden state from both directions
        h_fwd = h_n[-2]                                   # forward last layer
        h_bwd = h_n[-1]                                   # backward last layer
        out = torch.cat([h_fwd, h_bwd], dim=1)           # (batch, lstm_hidden*2)

        out = self.dropout(F.relu(self.fc1(out)))
        return self.fc2(out)
