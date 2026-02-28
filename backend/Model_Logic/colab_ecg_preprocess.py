"""
ECG preprocessing pipeline for Colab.

Step-1 only: clean and segment ECG into beats, score noise, and export
beat-level + context-window datasets for downstream training.

BLOCK 1 (Colab install):
    !pip install -q pandas numpy scipy scikit-learn matplotlib openpyxl neurokit2

BLOCK 2 (Run script with CLI args):
    !pip install -q pandas numpy scipy scikit-learn matplotlib openpyxl neurokit2
    !python colab_ecg_preprocess.py \
        --input /content/723_labelled_multiclass.xlsx \
        --output_dir /content/ecg_preprocessed \
        --duration_sec 10

BLOCK 3 (View outputs):
    from IPython.display import Image, display
    display(Image('/content/ecg_preprocessed/class_templates.png'))
    display(Image('/content/ecg_preprocessed/global_clusters_pca2d.png'))
    display(Image('/content/ecg_preprocessed/prototype_gallery.png'))

BLOCK 4 (Inspect prototype map):
    import pandas as pd
    pd.read_csv('/content/ecg_preprocessed/prototype_library.csv').head(20)

BLOCK 5 (Inspect key-value prototypes anytime):
    from colab_ecg_preprocess import inspect_prototype_keys
    inspect_prototype_keys('/content/ecg_preprocessed', top_n=20)

BLOCK 6 (Plot prototypes for one label):
    from colab_ecg_preprocess import plot_label_prototypes
    plot_label_prototypes('/content/ecg_preprocessed', label_id=2)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

try:
    import neurokit2 as nk  # type: ignore
except Exception:
    nk = None


DEFAULT_LABEL_NAMES = {
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


# =========================
# BLOCK A: Core Utilities
# =========================
@dataclass
class BeatRecord:
    source_row: int
    patient_id: Optional[int]
    record_label: int
    beat_index_in_record: int
    r_index: int
    rr_prev_sec: float
    rr_next_sec: float
    quality_score: float
    noise_flag: bool
    cluster_id: int
    cluster_size: int
    beat: np.ndarray


def robust_zscore(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-9
    return 0.6745 * (x - med) / mad


def parse_wave(text: object) -> np.ndarray:
    if not isinstance(text, str):
        return np.array([], dtype=np.float32)
    vals = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            vals.append(float(token))
        except ValueError:
            return np.array([], dtype=np.float32)
    return np.asarray(vals, dtype=np.float32)


def infer_fs(wave_len: int, duration_sec: float) -> float:
    return float(wave_len) / duration_sec


def bandpass_ecg(wave: np.ndarray, fs: float, low_hz: float = 0.5, high_hz: float = 45.0) -> np.ndarray:
    nyq = fs * 0.5
    high_hz = min(high_hz, nyq - 1e-3)
    if high_hz <= low_hz:
        return wave.astype(np.float32, copy=False)
    b, a = signal.butter(4, [low_hz / nyq, high_hz / nyq], btype="band")
    return signal.filtfilt(b, a, wave).astype(np.float32)


def normalize_wave(wave: np.ndarray) -> np.ndarray:
    mu = float(np.mean(wave))
    sigma = float(np.std(wave)) + 1e-8
    return ((wave - mu) / sigma).astype(np.float32)


def detect_rpeaks(ecg: np.ndarray, fs: float) -> np.ndarray:
    if nk is not None:
        try:
            _, out = nk.ecg_peaks(ecg, sampling_rate=fs, method="neurokit")
            peaks = np.asarray(out["ECG_R_Peaks"], dtype=np.int32)
            if peaks.size > 0:
                return peaks
        except Exception:
            pass

    # Fallback: derivative-energy envelope + peak picking.
    diff = np.diff(ecg, prepend=ecg[0])
    energy = diff * diff
    win = max(3, int(0.12 * fs))
    kernel = np.ones(win, dtype=np.float32) / win
    smooth = np.convolve(energy, kernel, mode="same")
    thr = float(np.mean(smooth) + 0.5 * np.std(smooth))
    min_dist = int(0.24 * fs)  # allow faster rhythms as well
    peaks, _ = signal.find_peaks(smooth, height=thr, distance=max(1, min_dist))
    return peaks.astype(np.int32)


def extract_beats(ecg: np.ndarray, peaks: np.ndarray, fs: float, pre_ms: float, post_ms: float) -> Tuple[np.ndarray, np.ndarray]:
    pre = int(round((pre_ms / 1000.0) * fs))
    post = int(round((post_ms / 1000.0) * fs))
    beats = []
    kept_peaks = []
    for p in peaks:
        left = p - pre
        right = p + post
        if left < 0 or right >= len(ecg):
            continue
        beats.append(ecg[left:right + 1])
        kept_peaks.append(p)
    if not beats:
        return np.empty((0, pre + post + 1), dtype=np.float32), np.empty((0,), dtype=np.int32)
    return np.asarray(beats, dtype=np.float32), np.asarray(kept_peaks, dtype=np.int32)


def beat_quality_features(beats: np.ndarray) -> Dict[str, np.ndarray]:
    ptp = np.ptp(beats, axis=1)
    slope = np.max(np.abs(np.diff(beats, axis=1)), axis=1)
    start = beats[:, : max(4, beats.shape[1] // 10)]
    end = beats[:, -max(4, beats.shape[1] // 10):]
    edge_noise = np.std(np.concatenate([start, end], axis=1), axis=1)
    return {"ptp": ptp, "slope": slope, "edge_noise": edge_noise}


def score_noise(beats: np.ndarray, rr_prev: np.ndarray, rr_next: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    feats = beat_quality_features(beats)
    z_ptp = robust_zscore(feats["ptp"])
    z_slope = robust_zscore(feats["slope"])
    z_edge = robust_zscore(feats["edge_noise"])

    rr_bad = (rr_prev < 0.2) | (rr_prev > 2.5) | (rr_next < 0.2) | (rr_next > 2.5)
    morphology_bad = (z_ptp < -3.0) | (z_slope < -3.0) | (z_edge > 3.5) | (z_slope > 6.0)

    noise = rr_bad | morphology_bad
    score = (np.abs(z_ptp) + np.abs(z_slope) + np.abs(z_edge)) / 3.0
    return score.astype(np.float32), noise.astype(bool)


def dominant_cluster_ids(beats: np.ndarray, max_k: int = 4) -> Tuple[np.ndarray, Dict[int, int]]:
    n = len(beats)
    if n < 12:
        ids = np.zeros((n,), dtype=np.int32)
        return ids, {0: n}
    k = int(np.clip(round(math.sqrt(n / 60.0) + 1), 2, max_k))
    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    ids = model.fit_predict(beats)
    sizes = {cid: int(np.sum(ids == cid)) for cid in np.unique(ids)}
    return ids.astype(np.int32), sizes


def resolve_column(df: pd.DataFrame, aliases: List[str]) -> str:
    lower_map = {c.lower().strip(): c for c in df.columns}
    for alias in aliases:
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    raise KeyError(f"Missing required column. Tried aliases: {aliases}")


def build_templates(beats: np.ndarray, labels: np.ndarray, noise: np.ndarray) -> Dict[int, np.ndarray]:
    templates: Dict[int, np.ndarray] = {}
    for cls in sorted(np.unique(labels)):
        idx = np.where((labels == cls) & (~noise))[0]
        if idx.size == 0:
            continue
        templates[int(cls)] = np.median(beats[idx], axis=0).astype(np.float32)
    return templates


def build_global_feature_space(
    beats: np.ndarray,
    rr_prev: np.ndarray,
    rr_next: np.ndarray,
    n_pca: int = 24,
) -> np.ndarray:
    """
    Build compact feature vectors for label-free grouping/retrieval.
    Uses beat waveform + RR context ratios.
    """
    rr_prev_col = rr_prev.reshape(-1, 1)
    rr_next_col = rr_next.reshape(-1, 1)
    rr_ratio = (rr_prev_col + 1e-6) / (rr_next_col + 1e-6)
    rr_feat = np.hstack([rr_prev_col, rr_next_col, rr_ratio]).astype(np.float32)

    pca_dim = int(min(n_pca, beats.shape[1], max(2, beats.shape[0] - 1)))
    pca = PCA(n_components=pca_dim, random_state=42)
    beat_emb = pca.fit_transform(beats).astype(np.float32)

    feat = np.hstack([beat_emb, rr_feat]).astype(np.float32)
    feat_mu = feat.mean(axis=0, keepdims=True)
    feat_std = feat.std(axis=0, keepdims=True) + 1e-6
    feat = (feat - feat_mu) / feat_std
    return feat.astype(np.float32)


def cluster_global_beats(
    features: np.ndarray,
    usable_mask: np.ndarray,
    n_clusters: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cluster beats without using labels.
    Returns:
      cluster_id_all: cluster id for each beat (-1 for unusable beats)
      cluster_size_all: cluster size for each beat (0 for unusable beats)
    """
    n = len(features)
    cluster_id_all = np.full((n,), -1, dtype=np.int32)
    cluster_size_all = np.zeros((n,), dtype=np.int32)

    usable_idx = np.where(usable_mask)[0]
    if usable_idx.size < 2:
        return cluster_id_all, cluster_size_all

    k = int(min(n_clusters, usable_idx.size))
    k = max(2, k)
    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    ids = model.fit_predict(features[usable_idx]).astype(np.int32)
    cluster_id_all[usable_idx] = ids

    unique_ids, counts = np.unique(ids, return_counts=True)
    size_map = {int(cid): int(c) for cid, c in zip(unique_ids, counts)}
    for idx_local, gidx in enumerate(usable_idx):
        cluster_size_all[gidx] = size_map[int(ids[idx_local])]
    return cluster_id_all, cluster_size_all


def save_global_cluster_plot(
    features: np.ndarray,
    cluster_ids: np.ndarray,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    usable = cluster_ids >= 0
    if usable.sum() < 3:
        return
    p2 = PCA(n_components=2, random_state=42).fit_transform(features[usable])
    c = cluster_ids[usable]
    plt.figure(figsize=(8, 6))
    plt.scatter(p2[:, 0], p2[:, 1], c=c, s=7, alpha=0.75, cmap="tab20")
    plt.title("Label-Free Beat Clusters (PCA-2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def build_prototype_library(
    beats: np.ndarray,
    labels: np.ndarray,
    noise: np.ndarray,
    global_cluster_ids: np.ndarray,
    rr_prev: np.ndarray,
    rr_next: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Build key-value prototype map:
      key   -> prototype_id (global cluster id)
      value -> centroid/median beat waveform + stats
    """
    rows = []
    wave_map: Dict[str, np.ndarray] = {}
    usable_idx = np.where((~noise) & (global_cluster_ids >= 0))[0]
    if usable_idx.size == 0:
        return pd.DataFrame(), wave_map

    cids = np.unique(global_cluster_ids[usable_idx])
    for cid in cids:
        idx = np.where((global_cluster_ids == cid) & (~noise))[0]
        if idx.size == 0:
            continue

        cluster_beats = beats[idx]
        prototype = np.median(cluster_beats, axis=0).astype(np.float32)
        wave_map[f"prototype_{int(cid)}"] = prototype

        cls_vals, cls_counts = np.unique(labels[idx], return_counts=True)
        order = np.argsort(-cls_counts)
        top_pairs = [(int(cls_vals[i]), int(cls_counts[i])) for i in order[:3]]
        top_text = ";".join([f"{k}:{v}" for k, v in top_pairs])

        rows.append(
            {
                "prototype_id": int(cid),
                "size": int(idx.size),
                "dominant_label": int(top_pairs[0][0]) if top_pairs else -999,
                "label_mix_top3": top_text,
                "rr_prev_mean": float(np.mean(rr_prev[idx])),
                "rr_next_mean": float(np.mean(rr_next[idx])),
                "example_beat_ids": ",".join(map(str, idx[:10].tolist())),
            }
        )

    proto_df = pd.DataFrame(rows).sort_values("size", ascending=False)
    return proto_df, wave_map


def save_prototype_gallery(
    proto_df: pd.DataFrame,
    wave_map: Dict[str, np.ndarray],
    out_path: Path,
    max_items: int = 25,
) -> None:
    import matplotlib.pyplot as plt

    if proto_df.empty:
        return
    show_df = proto_df.head(max_items)
    n = len(show_df)
    cols = 5
    rows = int(math.ceil(n / cols))
    plt.figure(figsize=(3.2 * cols, 2.2 * rows))
    for i, (_, r) in enumerate(show_df.iterrows(), start=1):
        pid = int(r["prototype_id"])
        w = wave_map.get(f"prototype_{pid}")
        if w is None:
            continue
        ax = plt.subplot(rows, cols, i)
        ax.plot(w, linewidth=1.4)
        ax.set_title(f"P{pid} n={int(r['size'])}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def seed_similarity_mining(
    features: np.ndarray,
    beat_df: pd.DataFrame,
    seed_csv: str,
    top_k: int,
    search_rows: int,
) -> pd.DataFrame:
    """
    seed_csv format:
      beat_global_id,label_name
      123,PVC
      454,PVC
      700,Normal
    """
    seeds = pd.read_csv(seed_csv)
    required = {"beat_global_id", "label_name"}
    if not required.issubset(set(seeds.columns)):
        raise ValueError(f"seed_csv must contain columns: {required}")

    seeds = seeds.dropna(subset=["beat_global_id", "label_name"]).copy()
    seeds["beat_global_id"] = seeds["beat_global_id"].astype(int)
    valid_seed_ids = [x for x in seeds["beat_global_id"].tolist() if 0 <= x < len(features)]
    if not valid_seed_ids:
        return pd.DataFrame(
            columns=[
                "seed_label",
                "seed_beat_global_id",
                "candidate_beat_global_id",
                "similarity",
                "source_row",
                "patient_id",
                "rr_prev_sec",
                "rr_next_sec",
            ]
        )

    max_row = int(beat_df["source_row"].min()) + int(search_rows) - 1
    candidate_mask = (beat_df["noise_flag"].values == 0) & (beat_df["source_row"].values <= max_row)
    candidate_ids = np.where(candidate_mask)[0]
    if candidate_ids.size == 0:
        return pd.DataFrame()

    rows = []
    for _, srow in seeds.iterrows():
        sid = int(srow["beat_global_id"])
        if sid < 0 or sid >= len(features):
            continue
        seed_label = str(srow["label_name"])
        sim = cosine_similarity(features[sid:sid + 1], features[candidate_ids])[0]
        order = np.argsort(-sim)[:top_k]
        for oid in order:
            gid = int(candidate_ids[oid])
            rows.append(
                {
                    "seed_label": seed_label,
                    "seed_beat_global_id": sid,
                    "candidate_beat_global_id": gid,
                    "similarity": float(sim[oid]),
                    "source_row": int(beat_df.iloc[gid]["source_row"]),
                    "patient_id": int(beat_df.iloc[gid]["patient_id"]),
                    "rr_prev_sec": float(beat_df.iloc[gid]["rr_prev_sec"]),
                    "rr_next_sec": float(beat_df.iloc[gid]["rr_next_sec"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["seed_label", "similarity"], ascending=[True, False])


def save_template_plot(templates: Dict[int, np.ndarray], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    if not templates:
        return
    plt.figure(figsize=(11, 6))
    for cls, tpl in templates.items():
        name = DEFAULT_LABEL_NAMES.get(cls, str(cls))
        plt.plot(tpl, label=f"{cls}: {name}", linewidth=1.6)
    plt.title("Representative Beat Template Per Class")
    plt.xlabel("Sample")
    plt.ylabel("Normalized Amplitude")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def make_context_windows(beats: np.ndarray, labels: np.ndarray, source_row: np.ndarray, noise: np.ndarray, window_radius: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_rows = []
    y_rows = []
    mask_rows = []
    beat_len = beats.shape[1]
    group_ids = np.unique(source_row)

    for gid in group_ids:
        idx = np.where((source_row == gid) & (~noise))[0]
        if idx.size == 0:
            continue
        ordered = idx  # already in record order
        for j, center_idx in enumerate(ordered):
            window = []
            mask = []
            for shift in range(-window_radius, window_radius + 1):
                pos = j + shift
                if 0 <= pos < len(ordered):
                    window.append(beats[ordered[pos]])
                    mask.append(1.0)
                else:
                    window.append(np.zeros((beat_len,), dtype=np.float32))
                    mask.append(0.0)
            x_rows.append(np.stack(window, axis=0))
            y_rows.append(labels[center_idx])
            mask_rows.append(np.asarray(mask, dtype=np.float32))

    if not x_rows:
        return (
            np.empty((0, 2 * window_radius + 1, beat_len), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0, 2 * window_radius + 1), dtype=np.float32),
        )
    return np.asarray(x_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.int32), np.asarray(mask_rows, dtype=np.float32)


def preprocess(args: argparse.Namespace) -> None:
    # =========================
    # BLOCK B: Load Input Data
    # =========================
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(args.input)
    wave_col = resolve_column(df, ["ECG Wave", "ecg_wave", "wave"])
    label_col = resolve_column(df, ["Label", "label"])
    patient_col = resolve_column(df, ["Patient ID", "patient_id", "patient"])

    all_beats: List[np.ndarray] = []
    all_labels: List[int] = []
    all_source_row: List[int] = []
    all_patient_id: List[int] = []
    all_r: List[int] = []
    all_rr_prev: List[float] = []
    all_rr_next: List[float] = []
    all_local_index: List[int] = []

    row_stats = []
    skipped_rows = 0

    # ============================================
    # BLOCK C: Per-record signal -> beat extraction
    # ============================================
    for ridx, row in df.iterrows():
        wave = parse_wave(row.get(wave_col))
        if wave.size < 200:
            skipped_rows += 1
            continue

        fs = args.fs if args.fs > 0 else infer_fs(len(wave), args.duration_sec)
        ecg = bandpass_ecg(wave, fs=fs, low_hz=args.low_hz, high_hz=args.high_hz)
        ecg = normalize_wave(ecg)
        peaks = detect_rpeaks(ecg, fs=fs)
        beats, kept = extract_beats(ecg, peaks, fs=fs, pre_ms=args.pre_ms, post_ms=args.post_ms)
        if beats.shape[0] == 0:
            skipped_rows += 1
            continue

        label = int(row.get(label_col))
        patient = int(row.get(patient_col)) if pd.notna(row.get(patient_col)) else -1

        rr = np.diff(kept) / fs
        rr_prev = np.concatenate([[rr[0] if rr.size else 1.0], rr]).astype(np.float32)
        rr_next = np.concatenate([rr, [rr[-1] if rr.size else 1.0]]).astype(np.float32)

        for i in range(len(beats)):
            all_beats.append(beats[i])
            all_labels.append(label)
            all_source_row.append(int(ridx))
            all_patient_id.append(patient)
            all_r.append(int(kept[i]))
            all_rr_prev.append(float(rr_prev[i]))
            all_rr_next.append(float(rr_next[i]))
            all_local_index.append(i)

        row_stats.append((int(ridx), int(len(wave)), float(fs), int(len(peaks)), int(len(beats))))

    if not all_beats:
        raise RuntimeError("No beats extracted. Check `duration_sec`, `fs`, or wave format.")

    beats = np.asarray(all_beats, dtype=np.float32)
    labels = np.asarray(all_labels, dtype=np.int32)
    source_row = np.asarray(all_source_row, dtype=np.int32)
    patient_ids = np.asarray(all_patient_id, dtype=np.int32)
    r_idx = np.asarray(all_r, dtype=np.int32)
    rr_prev = np.asarray(all_rr_prev, dtype=np.float32)
    rr_next = np.asarray(all_rr_next, dtype=np.float32)
    local_idx = np.asarray(all_local_index, dtype=np.int32)

    # ============================
    # BLOCK D: Noise score + filter
    # ============================
    quality_score, noise = score_noise(beats, rr_prev, rr_next)

    # label-aware clustering (optional) to split minor outlier morphologies
    local_cluster_ids = np.full((len(beats),), -1, dtype=np.int32)
    local_cluster_size = np.zeros((len(beats),), dtype=np.int32)
    for cls in sorted(np.unique(labels)):
        idx = np.where((labels == cls) & (~noise))[0]
        if idx.size == 0:
            continue
        ids, sizes = dominant_cluster_ids(beats[idx], max_k=args.max_k_per_class)
        for j, global_idx in enumerate(idx):
            cid = int(ids[j])
            local_cluster_ids[global_idx] = cid
            local_cluster_size[global_idx] = sizes.get(cid, 0)
        tiny_cluster = np.array([sizes[int(ids[j])] < args.min_cluster_size for j in range(len(idx))], dtype=bool)
        noise[idx[tiny_cluster]] = True

    # ==================================
    # BLOCK E: Build pattern templates/map
    # ==================================
    templates = build_templates(beats, labels, noise)
    save_template_plot(templates, out_dir / "class_templates.png")

    # label-free representation map
    features = build_global_feature_space(
        beats=beats,
        rr_prev=rr_prev,
        rr_next=rr_next,
        n_pca=args.feature_pca_dim,
    )
    global_cluster_ids, global_cluster_sizes = cluster_global_beats(
        features=features,
        usable_mask=(~noise),
        n_clusters=args.global_clusters,
    )
    save_global_cluster_plot(features, global_cluster_ids, out_dir / "global_clusters_pca2d.png")

    # =========================================
    # BLOCK F: Build consecutive context windows
    # =========================================
    x_ctx, y_ctx, m_ctx = make_context_windows(
        beats=beats,
        labels=labels,
        source_row=source_row,
        noise=noise,
        window_radius=args.context_radius,
    )

    beat_df = pd.DataFrame(
        {
            "beat_global_id": np.arange(len(beats), dtype=np.int32),
            "source_row": source_row,
            "patient_id": patient_ids,
            "record_label": labels,
            "record_label_name": [DEFAULT_LABEL_NAMES.get(int(x), str(int(x))) for x in labels],
            "beat_index_in_record": local_idx,
            "r_index": r_idx,
            "rr_prev_sec": rr_prev,
            "rr_next_sec": rr_next,
            "quality_score": quality_score,
            "noise_flag": noise.astype(np.int32),
            "local_cluster_id": local_cluster_ids,
            "local_cluster_size": local_cluster_size,
            "global_cluster_id": global_cluster_ids,
            "global_cluster_size": global_cluster_sizes,
        }
    )

    # =============================
    # BLOCK G: Save core artefacts
    # =============================
    beat_df.to_csv(out_dir / "beat_metadata.csv", index=False)
    np.save(out_dir / "beats.npy", beats)
    np.save(out_dir / "labels.npy", labels)
    np.save(out_dir / "noise_flags.npy", noise.astype(np.int32))
    np.save(out_dir / "features.npy", features)
    np.save(out_dir / "global_cluster_ids.npy", global_cluster_ids)
    np.savez_compressed(out_dir / "context_windows.npz", X=x_ctx, y=y_ctx, mask=m_ctx)
    np.savez_compressed(out_dir / "class_templates.npz", **{f"class_{k}": v for k, v in templates.items()})

    # =======================================
    # BLOCK H: Save prototype key-value library
    # =======================================
    # Key-value prototype map (label-free)
    proto_df, proto_wave_map = build_prototype_library(
        beats=beats,
        labels=labels,
        noise=noise,
        global_cluster_ids=global_cluster_ids,
        rr_prev=rr_prev,
        rr_next=rr_next,
    )
    if not proto_df.empty:
        proto_df.to_csv(out_dir / "prototype_library.csv", index=False)
        np.savez_compressed(out_dir / "prototype_library_waves.npz", **proto_wave_map)
        save_prototype_gallery(proto_df, proto_wave_map, out_dir / "prototype_gallery.png")

    # ==========================================
    # BLOCK I: Optional seed-based similar mining
    # ==========================================
    # optional: from 5-10 manually selected seeds, find similar beats in early records
    if args.seed_csv:
        mined = seed_similarity_mining(
            features=features,
            beat_df=beat_df,
            seed_csv=args.seed_csv,
            top_k=args.seed_top_k,
            search_rows=args.seed_search_rows,
        )
        mined.to_csv(out_dir / "seed_mined_candidates.csv", index=False)

    # =========================
    # BLOCK J: Summary outputs
    # =========================
    summary = {
        "input_rows": int(len(df)),
        "rows_skipped": int(skipped_rows),
        "beats_extracted": int(len(beats)),
        "beat_length": int(beats.shape[1]),
        "noise_beats": int(np.sum(noise)),
        "usable_beats": int(np.sum(~noise)),
        "class_counts_raw": {str(int(k)): int(np.sum(labels == k)) for k in np.unique(labels)},
        "class_counts_usable": {str(int(k)): int(np.sum((labels == k) & (~noise))) for k in np.unique(labels)},
        "context_samples": int(len(y_ctx)),
        "context_window_size": int(2 * args.context_radius + 1),
        "global_clusters": int(len(np.unique(global_cluster_ids[global_cluster_ids >= 0]))),
        "prototype_count": int(len(proto_df)),
        "seed_mining_enabled": bool(args.seed_csv),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Preprocessing complete.")
    print(json.dumps(summary, indent=2))
    print(f"Saved outputs in: {out_dir}")


# =========================
# BLOCK K: Runtime / CLI
# =========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Colab ECG preprocessing for beat-level + context datasets")
    p.add_argument("--input", required=True, help="Path to xlsx file")
    p.add_argument("--output_dir", required=True, help="Directory to save outputs")
    p.add_argument("--duration_sec", type=float, default=10.0, help="Record duration used for fs inference")
    p.add_argument("--fs", type=float, default=-1.0, help="Sampling rate; if <=0 infer from wave length / duration")
    p.add_argument("--low_hz", type=float, default=0.5, help="Bandpass low cutoff")
    p.add_argument("--high_hz", type=float, default=45.0, help="Bandpass high cutoff")
    p.add_argument("--pre_ms", type=float, default=250.0, help="Beat window before R peak")
    p.add_argument("--post_ms", type=float, default=400.0, help="Beat window after R peak")
    p.add_argument("--max_k_per_class", type=int, default=4, help="Max clusters per class for grouping")
    p.add_argument("--min_cluster_size", type=int, default=12, help="Clusters smaller than this are flagged as noise")
    p.add_argument("--context_radius", type=int, default=2, help="Consecutive beat radius (2 -> window of 5)")
    p.add_argument("--feature_pca_dim", type=int, default=24, help="PCA dimensions for label-free beat map")
    p.add_argument("--global_clusters", type=int, default=20, help="Number of global label-free beat clusters")
    p.add_argument("--seed_csv", type=str, default="", help="Optional CSV with columns beat_global_id,label_name")
    p.add_argument("--seed_top_k", type=int, default=50, help="Top similar beats per seed")
    p.add_argument("--seed_search_rows", type=int, default=100, help="Search similar beats in first N records")
    return p.parse_args()


def visualize_preprocessed_data(output_dir: str) -> None:
    """
    Utility function to visualize the extracted beat templates and global clusters
    directly inside a Colab Notebook cell block.

    Example to run in Colab after preprocessing:
        from colab_ecg_preprocess import visualize_preprocessed_data
        visualize_preprocessed_data("/content/ecg_preprocessed")
    """
    import matplotlib.pyplot as plt
    try:
        from IPython.display import display
        is_notebook = True
    except ImportError:
        is_notebook = False

    out_dir = Path(output_dir)

    print("=== Loading Class Templates ===")
    npz_path = out_dir / "class_templates.npz"
    if npz_path.exists():
        data = np.load(npz_path)
        classes = sorted(data.files, key=lambda x: int(x.split('_')[1]))
        
        n_classes = len(classes)
        cols = min(4, n_classes) if n_classes > 0 else 1
        rows = (n_classes + cols - 1) // cols
        
        if n_classes > 0:
            plt.figure(figsize=(15, 3 * rows))
            for i, cls_key in enumerate(classes):
                plt.subplot(rows, cols, i + 1)
                plt.plot(data[cls_key], color='#1f77b4', linewidth=1.5)
                plt.title(f"Template: {cls_key.replace('_', ' ')}")
                plt.xlabel("Sample")
                plt.ylabel("Normalized Amplitude")
                plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        else:
            print("No class templates found in file.")
    else:
        print(f"File not found: {npz_path}")

    print("\n=== Loading Label-Free Beat Clusters ===")
    feats_path = out_dir / "features.npy"
    cluster_ids_path = out_dir / "global_cluster_ids.npy"
    
    if feats_path.exists() and cluster_ids_path.exists():
        features = np.load(feats_path)
        cluster_ids = np.load(cluster_ids_path)
        usable = cluster_ids >= 0
        if usable.sum() > 2:
            p2 = PCA(n_components=2, random_state=42).fit_transform(features[usable])
            c = cluster_ids[usable]
            
            plt.figure(figsize=(8, 6))
            sc = plt.scatter(p2[:, 0], p2[:, 1], c=c, s=15, alpha=0.7, cmap="tab20")
            plt.title("Label-Free Beat Clusters (PCA-2D)")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.colorbar(sc, label="Global Cluster ID")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        else:
            print("Not enough usable beats to plot clusters.")
    else:
        print(f"Cluster/Feature files not found in {out_dir}")


def inspect_prototype_keys(output_dir: str, top_n: int = 20) -> None:
    """
    Quick key-value inspector for prototype library.
    Shows:
      - key = prototype_id
      - value = waveform array name in prototype_library_waves.npz
      - stats = size, dominant label, label mix, example beat ids
    """
    out_dir = Path(output_dir)
    csv_path = out_dir / "prototype_library.csv"
    npz_path = out_dir / "prototype_library_waves.npz"

    if not csv_path.exists():
        print(f"Not found: {csv_path}")
        return
    if not npz_path.exists():
        print(f"Not found: {npz_path}")
        return

    df = pd.read_csv(csv_path).sort_values("size", ascending=False).head(top_n)
    waves = np.load(npz_path)

    print(f"Showing top {len(df)} prototype keys from: {out_dir}")
    for _, r in df.iterrows():
        pid = int(r["prototype_id"])
        key = f"prototype_{pid}"
        has_wave = key in waves.files
        shape = tuple(waves[key].shape) if has_wave else None
        print(
            f"key={key:<14} "
            f"size={int(r['size']):<5} "
            f"dominant_label={int(r['dominant_label']):<3} "
            f"mix={r['label_mix_top3']} "
            f"wave_shape={shape}"
        )


def plot_label_prototypes(output_dir: str, label_id: int) -> None:
    """
    Plot prototype waveforms for one dominant label.
    Example:
        plot_label_prototypes('/content/ecg_preprocessed', label_id=2)
    """
    import matplotlib.pyplot as plt

    out_dir = Path(output_dir)
    csv_path = out_dir / "prototype_library.csv"
    npz_path = out_dir / "prototype_library_waves.npz"

    if not csv_path.exists():
        print(f"Not found: {csv_path}")
        return
    if not npz_path.exists():
        print(f"Not found: {npz_path}")
        return

    proto = pd.read_csv(csv_path)
    waves = np.load(npz_path)
    rows = proto[proto["dominant_label"] == int(label_id)].sort_values("size", ascending=False)

    print(f"Found {len(rows)} prototypes for label {label_id} ({DEFAULT_LABEL_NAMES.get(label_id, 'Unknown')})")
    if rows.empty:
        return

    plt.figure(figsize=(12, 5))
    for _, r in rows.iterrows():
        pid = int(r["prototype_id"])
        key = f"prototype_{pid}"
        if key not in waves.files:
            continue
        w = waves[key]
        plt.plot(w, label=f"P{pid} n={int(r['size'])}", alpha=0.9)
    plt.title(f"Prototype Patterns for Label {label_id} - {DEFAULT_LABEL_NAMES.get(label_id, 'Unknown')}")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude (normalized)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Supports both styles:
    # 1) CLI: python colab_ecg_preprocess.py --input ... --output_dir ...
    # 2) Direct notebook run with no CLI args (uses editable defaults below)
    cli_has_required = ("--input" in sys.argv) and ("--output_dir" in sys.argv)
    if cli_has_required:
        preprocess(parse_args())
    else:
        class Args:
            def __init__(self):
                self.input = "/content/drive/MyDrive/HeartcareAI/Models/LabelledData/723_labelled_multiclass.xlsx"
                self.output_dir = "/content/ecg_preprocessed"
                self.duration_sec = 10.0
                self.fs = -1.0
                self.low_hz = 0.5
                self.high_hz = 45.0
                self.pre_ms = 250.0
                self.post_ms = 400.0
                self.max_k_per_class = 4
                self.min_cluster_size = 12
                self.context_radius = 2
                self.feature_pca_dim = 24
                self.global_clusters = 20
                self.seed_csv = ""
                self.seed_top_k = 50
                self.seed_search_rows = 100

        preprocess(Args())
