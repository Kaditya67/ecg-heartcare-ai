"""
ECG ResNet-1D Model — residual connections for very deep 1D signal networks.
Architecture: Conv stem → 3× ResidualBlocks → Global Average Pool → FC
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, dropout=0.2):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

        # Shortcut connection to match dimensions if needed
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        out = self.dropout(F.relu(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ECGResNet1D(nn.Module):
    """
    1D ResNet for ECG classification.
    Input: (batch, 1, 2604) — flat 1D signal with channel dim.
    Uses Global Average Pooling (no fixed flatten dimension),
    so it handles any input length automatically.
    """

    def __init__(self, num_classes: int, dropout: float = 0.2):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # Residual blocks — progressively wider channels
        self.layer1 = self._make_layer(32, 64, blocks=2, stride=2, dropout=dropout)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2, dropout=dropout)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2, dropout=dropout)

        # Classifier
        self.gap = nn.AdaptiveAvgPool1d(1)  # Global average pooling
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)

    @staticmethod
    def _make_layer(in_ch, out_ch, blocks, stride, dropout):
        layers = [ResidualBlock1D(in_ch, out_ch, stride=stride, dropout=dropout)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_ch, out_ch, dropout=dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, 1, 2604)
        x = self.stem(x)       # (batch, 32, ~651)
        x = self.layer1(x)     # (batch, 64, ~326)
        x = self.layer2(x)     # (batch, 128, ~163)
        x = self.layer3(x)     # (batch, 256, ~82)
        x = self.gap(x)        # (batch, 256, 1)
        x = x.squeeze(-1)      # (batch, 256)
        x = self.dropout(x)
        return self.fc(x)
