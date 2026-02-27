"""
ECG LSTM Model — sequence-based classification
Architecture: Bidirectional LSTM → Attention → FC
Input shape: (batch, seq_len=2604) → reshaped to (batch, 52, 50) time steps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_out):
        # lstm_out: (batch, seq_len, hidden*2)
        scores = self.attn(lstm_out).squeeze(-1)          # (batch, seq_len)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (batch, seq_len, 1)
        context = (lstm_out * weights).sum(dim=1)         # (batch, hidden*2)
        return context


class ECGLSTM(nn.Module):
    """
    Bidirectional LSTM with attention for ECG classification.
    Input: flat list of 2604 floats per sample.
    Internally reshaped to (batch, 52 steps, 50 features).
    """

    def __init__(self, num_classes: int, hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.3, seq_len: int = 52, feat_dim: int = 50):
        super().__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim

        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = Attention(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (batch, 2604)
        batch = x.size(0)
        x = x.view(batch, self.seq_len, self.feat_dim)   # (batch, 52, 50)
        lstm_out, _ = self.lstm(x)                        # (batch, 52, hidden*2)
        context = self.attention(lstm_out)                # (batch, hidden*2)
        out = self.dropout(F.relu(self.fc1(context)))
        return self.fc2(out)
