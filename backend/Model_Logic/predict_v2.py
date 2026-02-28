"""
Inference for hybrid V2 model with unknown/review gating.

BLOCK 1:
    !pip install -q numpy pandas torch

BLOCK 2:
    !python predict_v2.py \
      --model_ckpt /content/ecg_trained_v2/best_hybrid_v2.pt \
      --preprocessed_dir /content/ecg_preprocessed \
      --prototype_dir /content/ecg_preprocessed \
      --out_dir /content/ecg_predictions_v2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

LABEL_NAMES = {
    -1: "Noise",
    0: "Normal",
    1: "PAC",
    2: "PVC",
    3: "Sinus Bradycardia",
    4: "Sinus Tachycardia",
    5: "AFib",
    6: "Atrial Flutter",
    7: "VTach",
    8: "VFib",
    9: "Ventricular Ectopic",
    10: "Couplets",
    11: "Triplets",
    12: "PVC Subtypes",
}


def parse_mix(mix: str):
    out = []
    if not isinstance(mix, str):
        return out
    for p in mix.split(";"):
        if ":" not in p:
            continue
        a, b = p.split(":", 1)
        try:
            out.append((int(a), int(b)))
        except ValueError:
            pass
    return out


def purity(mix: str):
    pairs = parse_mix(mix)
    if not pairs:
        return 0.0
    counts = [c for _, c in pairs]
    return max(counts) / max(1, sum(counts))


def load_prototypes(proto_dir: Path, min_size: int, min_purity: float):
    csv_path = proto_dir / "prototype_library.csv"
    npz_path = proto_dir / "prototype_library_waves.npz"
    if not csv_path.exists() or not npz_path.exists():
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)
    df = pd.read_csv(csv_path)
    waves = np.load(npz_path)
    w_all, l_all = [], []
    for _, r in df.iterrows():
        if int(r.get("size", 0)) < min_size:
            continue
        if purity(str(r.get("label_mix_top3", ""))) < min_purity:
            continue
        pid = int(r["prototype_id"])
        key = f"prototype_{pid}"
        if key not in waves.files:
            continue
        w_all.append(waves[key].astype(np.float32))
        l_all.append(int(r.get("dominant_label", -1)))
    if not w_all:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.asarray(w_all, dtype=np.float32), np.asarray(l_all, dtype=np.int64)


def cosine_proto_features(center_beats: np.ndarray, proto_waves: np.ndarray, proto_labels: np.ndarray, top_k_sim: int):
    n = len(center_beats)
    if proto_waves.size == 0:
        return np.zeros((n, top_k_sim), dtype=np.float32), np.zeros((n,), dtype=np.float32)
    x = center_beats / (np.linalg.norm(center_beats, axis=1, keepdims=True) + 1e-8)
    p = proto_waves / (np.linalg.norm(proto_waves, axis=1, keepdims=True) + 1e-8)
    sims = x @ p.T
    k = min(top_k_sim, sims.shape[1])
    topk = -np.sort(-sims, axis=1)[:, :k]
    uniq = sorted(np.unique(proto_labels).tolist())
    per_lbl = []
    for lbl in uniq:
        idx = np.where(proto_labels == lbl)[0]
        per_lbl.append(np.max(sims[:, idx], axis=1, keepdims=True))
    feat = np.concatenate([topk] + per_lbl, axis=1).astype(np.float32) if per_lbl else topk.astype(np.float32)
    max_sim = np.max(sims, axis=1).astype(np.float32)
    return feat, max_sim


class ECGHybridNet(nn.Module):
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
        self.context_lstm = nn.LSTM(64, 96, num_layers=1, batch_first=True, bidirectional=True)
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
        b, w, l = x_ctx.shape
        x = x_ctx.reshape(b * w, 1, l)
        beat = self.beat_encoder(x).squeeze(-1).reshape(b, w, 64)
        morph = beat[:, w // 2, :]
        ctx, _ = self.context_lstm(beat)
        ctx = ctx[:, w // 2, :]
        proto = self.proto_mlp(x_proto)
        z = self.fusion(torch.cat([morph, ctx, proto], dim=1))
        return self.class_head(z), self.gate_head(z).squeeze(1)


def main(args):
    pre = Path(args.preprocessed_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ctx = np.load(pre / "context_windows.npz")
    X = ctx["X"].astype(np.float32)
    y = ctx["y"].astype(np.int64) if "y" in ctx else np.full((len(X),), -999, dtype=np.int64)

    beat_meta = pd.read_csv(pre / "beat_metadata.csv")
    raw_class_values = sorted(set([int(v) for v in y.tolist() if int(v) >= 0]))
    remap = {i: raw_class_values[i] for i in range(len(raw_class_values))}
    if not remap:
        # fallback if y unavailable
        remap = {0: 0}

    proto_waves, proto_labels = load_prototypes(
        Path(args.prototype_dir),
        min_size=args.min_proto_size,
        min_purity=args.min_proto_purity,
    )
    center = X[:, X.shape[1] // 2, :]
    Xp, max_proto_sim = cosine_proto_features(center, proto_waves, proto_labels, args.top_k_sim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGHybridNet(num_classes=len(remap), proto_dim=Xp.shape[1]).to(device)
    state = torch.load(args.model_ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    xb = torch.tensor(X, dtype=torch.float32, device=device)
    xp = torch.tensor(Xp, dtype=torch.float32, device=device)
    with torch.no_grad():
        cls_logits, gate_logit = model(xb, xp)
        cls_prob = torch.softmax(cls_logits, dim=1).cpu().numpy()
        cls_idx = np.argmax(cls_prob, axis=1)
        cls_conf = np.max(cls_prob, axis=1)
        gate_prob = torch.sigmoid(gate_logit).cpu().numpy()

    pred_raw = np.array([remap.get(int(i), -999) for i in cls_idx], dtype=np.int64)

    # do-not-force label rule
    review = (
        (cls_conf < args.min_class_conf)
        | (gate_prob > args.max_noise_prob)
        | (max_proto_sim < args.min_proto_sim)
    )
    final_label = pred_raw.copy()
    final_label[review] = -99  # unknown/review

    pred_df = beat_meta.copy()
    n = min(len(pred_df), len(final_label))
    pred_df = pred_df.iloc[:n].copy()
    pred_df["pred_label_raw"] = pred_raw[:n]
    pred_df["pred_label_name"] = pred_df["pred_label_raw"].map(lambda z: LABEL_NAMES.get(int(z), str(int(z))))
    pred_df["pred_conf"] = cls_conf[:n]
    pred_df["noise_prob"] = gate_prob[:n]
    pred_df["max_proto_sim"] = max_proto_sim[:n]
    pred_df["review_flag"] = review[:n].astype(np.int32)
    pred_df["final_label"] = final_label[:n]
    pred_df["final_label_name"] = pred_df["final_label"].map(
        lambda z: "Unknown/Review" if int(z) == -99 else LABEL_NAMES.get(int(z), str(int(z)))
    )
    pred_df.to_csv(out / "beat_predictions_v2.csv", index=False)

    # strip-level aggregation by source_row
    strip_rows = []
    for sid, g in pred_df.groupby("source_row"):
        usable = g[g["final_label"] >= 0]
        if usable.empty:
            strip_lbl = -99
            strip_conf = 0.0
        else:
            vc = usable["final_label"].value_counts()
            strip_lbl = int(vc.index[0])
            strip_conf = float(vc.iloc[0] / len(g))
        strip_rows.append(
            {
                "source_row": int(sid),
                "strip_pred_label": strip_lbl,
                "strip_pred_name": "Unknown/Review" if strip_lbl == -99 else LABEL_NAMES.get(strip_lbl, str(strip_lbl)),
                "strip_pred_ratio": strip_conf,
                "num_beats": int(len(g)),
                "num_review": int((g["review_flag"] == 1).sum()),
            }
        )
    strip_df = pd.DataFrame(strip_rows).sort_values("source_row")
    strip_df.to_csv(out / "strip_predictions_v2.csv", index=False)

    summary = {
        "num_beats_predicted": int(len(pred_df)),
        "num_review_flagged": int((pred_df["review_flag"] == 1).sum()),
        "review_rate": float((pred_df["review_flag"] == 1).mean()),
        "num_strips": int(len(strip_df)),
        "unknown_strips": int((strip_df["strip_pred_label"] == -99).sum()),
        "thresholds": {
            "min_class_conf": args.min_class_conf,
            "max_noise_prob": args.max_noise_prob,
            "min_proto_sim": args.min_proto_sim,
        },
    }
    (out / "prediction_summary_v2.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved predictions to: {out}")


def parse_args():
    p = argparse.ArgumentParser(description="Predict with hybrid V2 + unknown/review gating")
    p.add_argument("--model_ckpt", required=True)
    p.add_argument("--preprocessed_dir", required=True)
    p.add_argument("--prototype_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--min_proto_size", type=int, default=120)
    p.add_argument("--min_proto_purity", type=float, default=0.70)
    p.add_argument("--top_k_sim", type=int, default=8)
    p.add_argument("--min_class_conf", type=float, default=0.60)
    p.add_argument("--max_noise_prob", type=float, default=0.55)
    p.add_argument("--min_proto_sim", type=float, default=0.20)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())

