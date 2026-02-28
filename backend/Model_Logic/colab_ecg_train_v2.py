"""
ECG training V2 (Colab): morphology + context + prototype similarity + noise gate.

BLOCK 1 (Install):
    !pip install -q numpy pandas scikit-learn torch

BLOCK 2 (Train with CLI):
    !python colab_ecg_train_v2.py \
      --data_dirs /content/ecg_preprocessed \
      --out_dir /content/ecg_trained_v2 \
      --epochs 25 \
      --batch_size 128

BLOCK 3 (Notebook-safe direct run):
    # Run file directly in notebook cell; __main__ defaults will be used.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


# =========================
# BLOCK A: Data Utilities
# =========================
def _mix_purity(label_mix_top3: str) -> float:
    # format like "2:3090;0:1893;-1:158"
    if not isinstance(label_mix_top3, str) or ":" not in label_mix_top3:
        return 0.0
    pairs = []
    for part in label_mix_top3.split(";"):
        part = part.strip()
        if ":" not in part:
            continue
        a, b = part.split(":", 1)
        try:
            pairs.append((int(a), int(b)))
        except ValueError:
            continue
    if not pairs:
        return 0.0
    counts = [c for _, c in pairs]
    return float(max(counts)) / max(1.0, float(sum(counts)))


def load_prototype_bank(
    data_dirs: List[str],
    min_proto_size: int,
    min_proto_purity: float,
    max_prototypes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    waves_all = []
    labels_all = []

    for d in data_dirs:
        dpath = Path(d)
        csv_path = dpath / "prototype_library.csv"
        npz_path = dpath / "prototype_library_waves.npz"
        if not (csv_path.exists() and npz_path.exists()):
            continue

        df = pd.read_csv(csv_path)
        waves = np.load(npz_path)
        if "prototype_id" not in df.columns:
            continue

        for _, r in df.iterrows():
            size = int(r.get("size", 0))
            purity = _mix_purity(str(r.get("label_mix_top3", "")))
            if size < min_proto_size or purity < min_proto_purity:
                continue
            pid = int(r["prototype_id"])
            key = f"prototype_{pid}"
            if key not in waves.files:
                continue
            w = waves[key].astype(np.float32)
            lbl = int(r.get("dominant_label", -1))
            waves_all.append(w)
            labels_all.append(lbl)

    if not waves_all:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)

    # keep largest subset if too many
    waves_np = np.asarray(waves_all, dtype=np.float32)
    labels_np = np.asarray(labels_all, dtype=np.int64)
    if len(waves_np) > max_prototypes:
        idx = np.arange(len(waves_np))
        np.random.seed(42)
        np.random.shuffle(idx)
        idx = idx[:max_prototypes]
        waves_np = waves_np[idx]
        labels_np = labels_np[idx]
    return waves_np, labels_np


def cosine_proto_features(
    center_beats: np.ndarray,
    proto_waves: np.ndarray,
    proto_labels: np.ndarray,
    top_k_sim: int = 8,
) -> np.ndarray:
    """
    Build prototype feature vector:
      [top-k cosine similarities] + [max similarity per prototype label]
    """
    n = len(center_beats)
    if proto_waves.size == 0:
        return np.zeros((n, top_k_sim), dtype=np.float32)

    x = center_beats.astype(np.float32)
    p = proto_waves.astype(np.float32)

    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    p = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-8)

    sims = x @ p.T  # (N, P)
    k = min(top_k_sim, sims.shape[1])
    topk = -np.sort(-sims, axis=1)[:, :k]

    uniq_labels = sorted(np.unique(proto_labels).tolist())
    label_max = []
    for lbl in uniq_labels:
        idx = np.where(proto_labels == lbl)[0]
        if idx.size == 0:
            label_max.append(np.zeros((n, 1), dtype=np.float32))
        else:
            label_max.append(np.max(sims[:, idx], axis=1, keepdims=True).astype(np.float32))
    label_max_np = np.concatenate(label_max, axis=1) if label_max else np.zeros((n, 0), dtype=np.float32)
    return np.concatenate([topk.astype(np.float32), label_max_np], axis=1).astype(np.float32)


def load_training_data(
    data_dirs: List[str],
    min_proto_size: int,
    min_proto_purity: float,
    max_prototypes: int,
    top_k_sim: int,
):
    xs, ys = [], []
    for d in data_dirs:
        p = Path(d) / "context_windows.npz"
        if not p.exists():
            continue
        npz = np.load(p)
        xs.append(npz["X"].astype(np.float32))  # (N, W, L)
        ys.append(npz["y"].astype(np.int64))    # raw label values, may include -1
    if not xs:
        raise RuntimeError("No context_windows.npz found in --data_dirs")

    X = np.concatenate(xs, axis=0)
    y_raw = np.concatenate(ys, axis=0)
    center = X[:, X.shape[1] // 2, :]

    proto_waves, proto_labels = load_prototype_bank(
        data_dirs=data_dirs,
        min_proto_size=min_proto_size,
        min_proto_purity=min_proto_purity,
        max_prototypes=max_prototypes,
    )
    X_proto = cosine_proto_features(center, proto_waves, proto_labels, top_k_sim=top_k_sim)

    # class labels only for non-noise
    valid_class = y_raw >= 0
    uniq_cls = sorted(np.unique(y_raw[valid_class]).tolist()) if np.any(valid_class) else []
    remap = {old: i for i, old in enumerate(uniq_cls)}
    y_class = np.full_like(y_raw, fill_value=-1, dtype=np.int64)
    for old, new in remap.items():
        y_class[y_raw == old] = new

    # noise gate target
    y_gate = (y_raw < 0).astype(np.float32)

    meta = {
        "raw_class_values": uniq_cls,
        "num_prototypes": int(len(proto_waves)),
        "proto_feature_dim": int(X_proto.shape[1]),
    }
    return X, X_proto, y_class, y_gate, meta


# =========================
# BLOCK B: Model
# =========================
class ECGHybridDataset(Dataset):
    def __init__(self, x_ctx: np.ndarray, x_proto: np.ndarray, y_cls: np.ndarray, y_gate: np.ndarray):
        self.x_ctx = torch.tensor(x_ctx, dtype=torch.float32)
        self.x_proto = torch.tensor(x_proto, dtype=torch.float32)
        self.y_cls = torch.tensor(y_cls, dtype=torch.long)
        self.y_gate = torch.tensor(y_gate, dtype=torch.float32)

    def __len__(self):
        return len(self.y_cls)

    def __getitem__(self, idx):
        return self.x_ctx[idx], self.x_proto[idx], self.y_cls[idx], self.y_gate[idx]


class ECGHybridNet(nn.Module):
    """
    3-branch model:
      1) beat morphology feature
      2) temporal context feature
      3) prototype similarity feature
    outputs:
      - class logits
      - noise gate logit
    """
    def __init__(self, num_classes: int, proto_dim: int):
        super().__init__()
        self.beat_encoder = nn.Sequential(
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
        self.context_lstm = nn.LSTM(
            input_size=64,
            hidden_size=96,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.proto_mlp = nn.Sequential(
            nn.Linear(max(1, proto_dim), 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(64 + 192 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 96),
            nn.ReLU(),
        )
        self.class_head = nn.Linear(96, num_classes)
        self.gate_head = nn.Linear(96, 1)

    def forward(self, x_ctx, x_proto):
        # x_ctx: (B, W, L), x_proto: (B, D)
        b, w, l = x_ctx.shape
        x = x_ctx.reshape(b * w, 1, l)
        beat_feat = self.beat_encoder(x).squeeze(-1)   # (B*W,64)
        beat_feat = beat_feat.reshape(b, w, 64)        # (B,W,64)

        morph_feat = beat_feat[:, w // 2, :]           # center beat morphology
        ctx_out, _ = self.context_lstm(beat_feat)      # (B,W,192)
        ctx_feat = ctx_out[:, w // 2, :]               # center with temporal context

        proto_feat = self.proto_mlp(x_proto)
        fused = self.fusion(torch.cat([morph_feat, ctx_feat, proto_feat], dim=1))
        class_logits = self.class_head(fused)
        gate_logit = self.gate_head(fused).squeeze(1)
        return class_logits, gate_logit


# =========================
# BLOCK C: Train / Eval
# =========================
def split_data(X, XP, YC, YG, test_size=0.2):
    idx = np.arange(len(YC))
    # stratify by raw behavior proxy: class id where valid else special id
    strat = YC.copy()
    strat[strat < 0] = np.max(strat) + 1 if np.any(strat >= 0) else 0
    tr_idx, va_idx = train_test_split(idx, test_size=test_size, random_state=42, stratify=strat)
    return (
        X[tr_idx], XP[tr_idx], YC[tr_idx], YG[tr_idx],
        X[va_idx], XP[va_idx], YC[va_idx], YG[va_idx],
    )


def train_v2(args):
    X, XP, YC, YG, meta = load_training_data(
        data_dirs=args.data_dirs,
        min_proto_size=args.min_proto_size,
        min_proto_purity=args.min_proto_purity,
        max_prototypes=args.max_prototypes,
        top_k_sim=args.top_k_sim,
    )
    num_classes = int(np.max(YC) + 1) if np.any(YC >= 0) else 1

    Xtr, XPtr, YCtr, YGtr, Xva, XPva, YCva, YGva = split_data(X, XP, YC, YG, test_size=args.val_split)
    tr_ds = ECGHybridDataset(Xtr, XPtr, YCtr, YGtr)
    va_ds = ECGHybridDataset(Xva, XPva, YCva, YGva)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGHybridNet(num_classes=num_classes, proto_dim=XP.shape[1]).to(device)

    # Class imbalance handling for non-noise classes.
    if args.use_class_weights:
        valid_tr = YCtr >= 0
        counts = np.bincount(YCtr[valid_tr], minlength=num_classes).astype(np.float32)
        counts[counts == 0] = 1.0
        inv = 1.0 / counts
        w = inv / inv.sum() * num_classes
        class_w = torch.tensor(w, dtype=torch.float32, device=device)
        cls_loss_fn = nn.CrossEntropyLoss(weight=class_w)
    else:
        cls_loss_fn = nn.CrossEntropyLoss()

    gate_loss_fn = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best_hybrid_v2.pt"
    best_val = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss, tr_cls_ok, tr_cls_n = 0.0, 0, 0
        tr_gate_ok, tr_gate_n = 0, 0

        for x_ctx, x_proto, y_cls, y_gate in tr_loader:
            x_ctx = x_ctx.to(device)
            x_proto = x_proto.to(device)
            y_cls = y_cls.to(device)
            y_gate = y_gate.to(device)

            optim.zero_grad()
            cls_logits, gate_logit = model(x_ctx, x_proto)

            valid = y_cls >= 0
            if torch.any(valid):
                l_cls = cls_loss_fn(cls_logits[valid], y_cls[valid])
            else:
                l_cls = torch.tensor(0.0, device=device)
            l_gate = gate_loss_fn(gate_logit, y_gate)
            loss = l_cls + args.gate_loss_weight * l_gate
            loss.backward()
            optim.step()

            tr_loss += float(loss.item()) * x_ctx.size(0)
            if torch.any(valid):
                pred = cls_logits.argmax(1)
                tr_cls_ok += int((pred[valid] == y_cls[valid]).sum().item())
                tr_cls_n += int(valid.sum().item())
            gate_pred = (torch.sigmoid(gate_logit) >= 0.5).float()
            tr_gate_ok += int((gate_pred == y_gate).sum().item())
            tr_gate_n += int(y_gate.numel())

        # validation
        model.eval()
        va_cls_ok, va_cls_n = 0, 0
        va_gate_ok, va_gate_n = 0, 0
        all_p, all_t = [], []
        with torch.no_grad():
            for x_ctx, x_proto, y_cls, y_gate in va_loader:
                x_ctx = x_ctx.to(device)
                x_proto = x_proto.to(device)
                y_cls = y_cls.to(device)
                y_gate = y_gate.to(device)

                cls_logits, gate_logit = model(x_ctx, x_proto)
                valid = y_cls >= 0
                if torch.any(valid):
                    pred = cls_logits.argmax(1)
                    va_cls_ok += int((pred[valid] == y_cls[valid]).sum().item())
                    va_cls_n += int(valid.sum().item())
                    all_p.extend(pred[valid].cpu().numpy().tolist())
                    all_t.extend(y_cls[valid].cpu().numpy().tolist())
                gate_pred = (torch.sigmoid(gate_logit) >= 0.5).float()
                va_gate_ok += int((gate_pred == y_gate).sum().item())
                va_gate_n += int(y_gate.numel())

        tr_cls_acc = tr_cls_ok / max(1, tr_cls_n)
        va_cls_acc = va_cls_ok / max(1, va_cls_n)
        tr_gate_acc = tr_gate_ok / max(1, tr_gate_n)
        va_gate_acc = va_gate_ok / max(1, va_gate_n)
        print(
            f"Epoch {epoch:02d} | loss {tr_loss/max(1,len(tr_ds)):.4f} "
            f"| cls(train/val) {tr_cls_acc:.4f}/{va_cls_acc:.4f} "
            f"| gate(train/val) {tr_gate_acc:.4f}/{va_gate_acc:.4f}"
        )

        # select by class accuracy primarily; small tie-break on gate
        score = va_cls_acc + 0.05 * va_gate_acc
        if score > best_val:
            best_val = score
            torch.save(model.state_dict(), best_path)

    if all_t and all_p:
        report = classification_report(all_t, all_p, output_dict=True, zero_division=0)
        cm = confusion_matrix(all_t, all_p).tolist()
    else:
        report = {}
        cm = []

    summary = {
        "best_score": float(best_val),
        "num_samples": int(len(YC)),
        "num_classes_non_noise": int(num_classes),
        "raw_class_values": meta["raw_class_values"],
        "num_prototypes_used": meta["num_prototypes"],
        "proto_feature_dim": meta["proto_feature_dim"],
        "model_path": str(best_path),
        "use_class_weights": bool(args.use_class_weights),
        "confusion_matrix_non_noise": cm,
        "classification_report_non_noise": report,
    }
    (out_dir / "training_summary_v2.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Training V2 complete.")
    print(json.dumps({k: v for k, v in summary.items() if "report" not in k}, indent=2))


# =========================
# BLOCK D: Runtime
# =========================
def parse_args():
    p = argparse.ArgumentParser(description="Hybrid ECG training V2 (morphology + context + prototypes + gate)")
    p.add_argument("--data_dirs", nargs="+", required=True, help="One or more preprocess output dirs")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--gate_loss_weight", type=float, default=0.5)
    p.add_argument("--min_proto_size", type=int, default=120)
    p.add_argument("--min_proto_purity", type=float, default=0.70)
    p.add_argument("--max_prototypes", type=int, default=256)
    p.add_argument("--top_k_sim", type=int, default=8)
    p.add_argument("--use_class_weights", type=int, default=1, help="1=enable inverse-frequency CE class weights")
    return p.parse_args()


if __name__ == "__main__":
    cli_has_required = ("--data_dirs" in sys.argv) and ("--out_dir" in sys.argv)
    if cli_has_required:
        train_v2(parse_args())
    else:
        class Args:
            def __init__(self):
                self.data_dirs = ["/content/ecg_preprocessed"]
                self.out_dir = "/content/ecg_trained_v2"
                self.epochs = 25
                self.batch_size = 128
                self.lr = 1e-3
                self.val_split = 0.2
                self.gate_loss_weight = 0.5
                self.min_proto_size = 120
                self.min_proto_purity = 0.70
                self.max_prototypes = 256
                self.top_k_sim = 8
                self.use_class_weights = 1

        train_v2(Args())
