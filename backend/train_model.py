"""
ECG Model Training Script
=========================
Train any model from the MODEL_MAP on labeled records from the DB.

Usage:
    cd backend
    python train_model.py --model ECG1DCNN --epochs 30 --lr 1e-3 --batch 64
    python train_model.py --model ECGCNNLSTMHybrid --epochs 50 --lr 5e-4
    python train_model.py --model ECGResNet1D --epochs 40 --lr 1e-3
    python train_model.py --model ECGLSTM --epochs 40 --lr 1e-3

Output:
    api/models/best_<model_name_lower>.pth   (best val-accuracy checkpoint)

Requirements in requirements.txt are sufficient.
"""

import argparse
import os
import sys

import django
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# ── Django setup ─────────────────────────────────────────────────────────────
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
django.setup()

from api.models import ECGLabel, ECGRecord
from api.models_loader import MODEL_MAP


# ─────────────────────────────────────────────────────────────────────────────
class ECGDataset(Dataset):
    def __init__(self, samples, labels, model_type):
        self.samples = samples
        self.labels = labels
        self.model_type = model_type

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wave = torch.tensor(self.samples[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        # CNN / ResNet models need a channel dim: (1, L)
        if self.model_type in ('ECG1DCNN', 'ECGCNNLSTMHybrid', 'ECGResNet1D'):
            wave = wave.unsqueeze(0)
        return wave, label


# ─────────────────────────────────────────────────────────────────────────────
def load_data(input_size):
    """Pull labeled records from the DB, return (waves, labels)."""
    print("Loading records from DB …")
    qs = (
        ECGRecord.objects
        .exclude(label__isnull=True)
        .select_related('label')
        .only('ecg_wave', 'label__value')
    )

    waves, labels = [], []
    label_counts = {}
    skipped = 0

    for rec in qs:
        wave = rec.ecg_wave
        if isinstance(wave, str):
            wave = [float(v) for v in wave.split(',')]
        if len(wave) != input_size:
            skipped += 1
            continue
        label_val = rec.label.value
        waves.append(wave)
        labels.append(label_val)
        label_counts[label_val] = label_counts.get(label_val, 0) + 1

    if skipped:
        print(f"  Skipped {skipped} records with wrong wave length.")

    print(f"  Loaded {len(waves)} records.")
    print("  Label distribution:", label_counts)
    return np.array(waves, dtype=np.float32), np.array(labels, dtype=np.int64)


def make_sampler(labels):
    """WeightedRandomSampler to handle class imbalance."""
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts[labels]
    return WeightedRandomSampler(weights, len(weights), replacement=True)


# ─────────────────────────────────────────────────────────────────────────────
def train(args):
    if args.model not in MODEL_MAP:
        print(f"ERROR: model '{args.model}' not in MODEL_MAP. Available: {list(MODEL_MAP.keys())}")
        sys.exit(1)

    model_info = MODEL_MAP[args.model]
    input_size = model_info['input_size']
    num_classes = model_info['num_classes']
    model_type = model_info['class'].__name__
    save_path = model_info['path']
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    waves, labels = load_data(input_size)

    # Normalize labels to 0-based if needed
    unique_labels = sorted(set(labels))
    label_remap = {v: i for i, v in enumerate(unique_labels)}
    labels = np.array([label_remap[l] for l in labels])
    actual_classes = len(unique_labels)
    print(f"  Classes after remap: {unique_labels} → 0..{actual_classes-1}")

    X_train, X_val, y_train, y_val = train_test_split(
        waves, labels, test_size=0.2, random_state=42, stratify=labels
    )

    extra_kwargs = model_info.get('kwargs', {})
    train_ds = ECGDataset(X_train, y_train, model_type)
    val_ds = ECGDataset(X_val, y_val, model_type)

    sampler = make_sampler(y_train)
    train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    model_class = model_info['class']
    model = model_class(actual_classes, **extra_kwargs).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model}  |  Params: {param_count:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    best_epoch = 0

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nTraining for {args.epochs} epochs …\n")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)
            preds = out.argmax(dim=1)
            train_correct += (preds == y).sum().item()
            train_total += X.size(0)

        scheduler.step()

        # ── Validation ───────────────────────────────────────────────────────
        model.eval()
        val_correct, val_total = 0, 0
        all_preds, all_true = [], []

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                preds = out.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += X.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(y.cpu().numpy())

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        avg_loss = train_loss / train_total

        print(f"Epoch {epoch:3d}/{args.epochs}  |  "
              f"Loss: {avg_loss:.4f}  |  "
              f"Train Acc: {train_acc:.4f}  |  "
              f"Val Acc: {val_acc:.4f}"
              + (" ← BEST" if val_acc > best_val_acc else ""))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)

    # ── Final report ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Best Val Accuracy: {best_val_acc:.4f}  (epoch {best_epoch})")
    print(f"Saved to: {save_path}")
    print(f"\nClassification Report (last epoch):")
    label_names = [ECGLabel.objects.filter(value=uid).first() for uid in unique_labels]
    label_name_strs = [l.name if l else str(uid) for l, uid in zip(label_names, unique_labels)]
    print(classification_report(all_true, all_preds, target_names=label_name_strs))
    print("Confusion Matrix:")
    print(confusion_matrix(all_true, all_preds))


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train ECG classification models")
    parser.add_argument('--model', required=True, choices=list(MODEL_MAP.keys()),
                        help='Model name from MODEL_MAP')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch', type=int, default=64, help='Batch size')
    args = parser.parse_args()
    train(args)
