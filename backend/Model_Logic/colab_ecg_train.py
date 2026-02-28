"""
ECG training script for Colab using preprocessed outputs from colab_ecg_preprocess.py.

BLOCK 1 (Install):
    !pip install -q numpy pandas scikit-learn torch

BLOCK 2 (Train):
    !python colab_ecg_train.py \
      --data_dirs /content/ecg_preprocessed \
      --epochs 20 \
      --batch_size 128 \
      --out_dir /content/ecg_trained

BLOCK 3 (Notebook-safe direct run):
    # Run this file directly in Colab cell without CLI flags.
    # Defaults in __main__ will be used.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class ECGContextDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)  # (N, W, L)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class ECGContextNet(nn.Module):
    """
    CNN over each beat + BiLSTM across consecutive beats in window.
    Input: (batch, window, beat_len)
    """
    def __init__(self, num_classes: int, beat_len: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 24, kernel_size=7, padding=3),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(24, 48, kernel_size=5, padding=2),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(48, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=96,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(192, 96),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(96, num_classes),
        )

    def forward(self, x):
        # x: (B, W, L)
        b, w, l = x.shape
        x = x.reshape(b * w, 1, l)
        f = self.cnn(x).squeeze(-1)      # (B*W, 64)
        f = f.reshape(b, w, 64)          # (B, W, 64)
        o, _ = self.lstm(f)              # (B, W, 192)
        center = o[:, w // 2, :]         # center beat context representation
        return self.head(center)


def load_npz_dirs(data_dirs: List[str]):
    xs, ys = [], []
    for d in data_dirs:
        p = Path(d) / "context_windows.npz"
        if not p.exists():
            continue
        data = np.load(p)
        x = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64)
        valid = y >= 0  # ignore noise class if already set as -1
        xs.append(x[valid])
        ys.append(y[valid])
    if not xs:
        raise RuntimeError("No context_windows.npz found in provided --data_dirs")
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    # remap to 0..K-1
    uniq = sorted(np.unique(y).tolist())
    remap = {old: i for i, old in enumerate(uniq)}
    y_new = np.array([remap[int(v)] for v in y], dtype=np.int64)
    return x, y_new, uniq


def train(args):
    x, y, original_labels = load_npz_dirs(args.data_dirs)
    beat_len = x.shape[2]
    num_classes = len(np.unique(y))

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    train_ds = ECGContextDataset(x_train, y_train)
    val_ds = ECGContextDataset(x_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGContextNet(num_classes=num_classes, beat_len=beat_len).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = 0.0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best_context_model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss, tr_ok, tr_n = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * xb.size(0)
            tr_ok += (logits.argmax(1) == yb).sum().item()
            tr_n += xb.size(0)

        model.eval()
        va_ok, va_n = 0, 0
        all_p, all_t = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                pred = logits.argmax(1)
                va_ok += (pred == yb).sum().item()
                va_n += xb.size(0)
                all_p.extend(pred.cpu().numpy().tolist())
                all_t.extend(yb.cpu().numpy().tolist())

        tr_acc = tr_ok / max(1, tr_n)
        va_acc = va_ok / max(1, va_n)
        print(f"Epoch {epoch:02d} | loss {tr_loss/max(1,tr_n):.4f} | train {tr_acc:.4f} | val {va_acc:.4f}")
        if va_acc > best_val:
            best_val = va_acc
            torch.save(model.state_dict(), best_path)

    report = classification_report(all_t, all_p, output_dict=True)
    summary = {
        "best_val_acc": float(best_val),
        "num_samples": int(len(y)),
        "num_classes": int(num_classes),
        "original_label_values": original_labels,
        "model_path": str(best_path),
        "classification_report": report,
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Training complete.")
    print(json.dumps({k: v for k, v in summary.items() if k != "classification_report"}, indent=2))


def parse_args():
    p = argparse.ArgumentParser(description="Train ECG context model from preprocessed outputs")
    p.add_argument("--data_dirs", nargs="+", required=True, help="One or more preprocessing output dirs")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


if __name__ == "__main__":
    # Colab/Jupyter-safe:
    # only use argparse when required flags are explicitly present
    cli_has_required = ("--data_dirs" in sys.argv) and ("--out_dir" in sys.argv)
    if cli_has_required:
        train(parse_args())
    else:
        class Args:
            def __init__(self):
                self.data_dirs = ["/content/ecg_preprocessed"]
                self.epochs = 20
                self.batch_size = 128
                self.lr = 1e-3
                self.out_dir = "/content/ecg_trained"

        train(Args())
