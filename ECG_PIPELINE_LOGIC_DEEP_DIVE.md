# ECG Pipeline Logic Deep Dive

This document provides a highly technical, line-by-line understanding of the modernized ECG classification pipeline (`colab_ecg_preprocess.py` and `colab_ecg_train.py`). 

It details exactly how we transitioned from a naive "full-strip" 1D-CNN (which memorizes wave positions and fails to generalize) to a **Beat-by-Beat Hybrid CNN-BiLSTM architecture** that treats ECG signals as a true temporal sequence of localized events, mimicking how a human cardiologist reads an ECG.

## 1. Core Architecture Shift & Rationale

**The Version 1 Problem:** 
The old pipeline fed an entire 10-second ECG string (2604 data points) directly into a basic CNN. Because pathological waveforms (like a PAC or PVC) can occur anywhere in that 10-second window, the CNN was forced to act as a positional memorizer rather than a shape feature extractor. If a PVC occurred at second 2 in the training set and second 8 in the validation set, the model failed. Furthermore, passing raw, unnormalized integer streams (baseline shifts from breathing, sensor drift) directly to the network caused severe overfitting (Validation loss spiking to 2.6).

**The Modernized Solution (`ECGContextNet`):**
1. **Clinical Signal Conditioning:** We enforce a strict 0.5–45 Hz bandpass filter to eliminate baseline wander and high-frequency noise, followed by Z-score normalization so the model evaluates *shape*, not sensor voltage offsets.
2. **Beat Extraction (R-Peak Centering):** Using `neurokit2` (or a fallback energy-envelope derivative algorithm), we isolate the R-peaks. Every single heartbeat is clipped into an equal-length vector centered exactly on the R-peak. The CNN now only looks at a centered heartbeat, instantly solving the positional-variance problem.
3. **Sequential Context Windows:** We group beats into moving windows of 5 ([T-2, T-1, Target, T+1, T+2]).
4. **Hybrid Network (CNN -> BiLSTM -> Head):** A 1D-CNN extracts the morphological features of each individual beat. Those features are passed into a Bidirectional LSTM which evaluates the *sequence* of those beats. This is critical for diagnosing rhythm-based conditions like AFib or Tachycardia which cannot be diagnosed from a single beat's shape alone.

---

## 2. Preprocessing Pipeline (`colab_ecg_preprocess.py`)

### 2.1 Dynamic Sampling Rate Validation
The pipeline infers the sampling frequency ($F_s$) dynamically: 
`infer_fs = wave_len / duration_sec`. Assuming a 10-second strip with ~2604 points, the pipeline derives $F_s \approx 260.4\text{ Hz}$, adjusting calculations automatically rather than hardcoding. 

### 2.2 Beat Slicing (`extract_beats`)
When an R-peak is found, the signal is clipped using strict pre-R and post-R time windows. 
- `pre_ms=250` (captures the P-wave)
- `post_ms=400` (captures the T-wave)
This converts a 10-second continuous wave into $N$ discrete matrices of shape `(Beat Length,)`. 

### 2.3 RR-Intervals & Clinical Noise Scoring
For each localized beat, we calculate its clinical context:
- `rr_prev_sec` (Time elapsed since the previous R-peak)
- `rr_next_sec` (Time until the next R-peak)

If RR intervals are physically impossible (`<0.2s` or `>2.5s`), or if morphological checks (like extreme slopes or edge noise) trigger our robust Z-score limits, the beat is dynamically flagged with `noise = True` and safely partitioned out of the training targets.

### 2.4 Contextual Window Generation (`make_context_windows`)
The pipeline iterates over the valid beats in a patient's strip. For every target beat, it gathers the $R$ beats before it and the $R$ beats after it. 
- Using `window_radius=2`, each target beat becomes a matrix of shape `(5, Beat Length)`. 
- If a beat is near the edge of a strip, padding with zero-vectors ensures dimensionality is preserved without corrupting the BiLSTM state.

### 2.5 Automated Analytics Outputs
The script naturally writes `.npy` and `.npz` arrays for immediate high-throughput GPU training in Colab, alongside generated PNG plots (e.g., `class_templates.png`, `global_clusters_pca2d.png`) to visually guarantee R-peak alignment before training starts.

---

## 3. Training Pipeline (`colab_ecg_train.py`)

### 3.1 Network Topology: `ECGContextNet`

The `ECGContextNet` class in PyTorch leverages a powerful `(Batch, Sequence, Features)` dual-pass architecture:

1. **Morphological Extractor (CNN Pass):**
   - The input tensor shapes are `(B, W, L)` where `W`=5 (the context window) and `L` is the length of a single beat array.
   - The tensor is flattened to `(B*W, 1, L)` and pushed through three deep 1D Convolutional blocks (`Conv1d -> BatchNorm -> ReLU -> MaxPool`).
   - This output compresses the raw wave data into a condensed 64-dimensional feature vector, resulting in shape `(B, W, 64)`.

2. **Rhythm Evaluator (LSTM Pass):**
   - The morphology features pass into a Bidirectional LSTM (`nn.LSTM(input_size=64, hidden_size=96, bidirectional=True)`).
   - The BiLSTM evaluates the sequence forwards and backwards, outputting a tensor of shape `(B, W, 192)` (96 hidden units * 2 directions).

3. **Classification Head:**
   - We extract exactly the hidden state output corresponding to the *center beat* of the window (`o[:, w // 2, :]`).
   - Standard linear combinations, dropout (`p=0.2`), and ReLU activations reduce the 192-dimensional vector down to the final `num_classes` logit predictions.

### 3.2 Loss, Overfitting Mitigation, & Export
By shifting away from training on entire strips to highly specific, normalized beat windows, the network is forced to learn medical pathology rather than dataset noise. 
- Validation Accuracy and Training Loss are monitored continually.
- The minimum validation loss triggers a best-model checkpoint save `best_context_model.pt`.
- `classification_report` logic is baked directly into the training completion step, calculating true precision and recall on the minority subsets (e.g., PACs/AFibs) automatically.

---

## 4. Why This Architecture Excels

1. **Interpretability:** A cardiologist looks at individual beat shapes and RR intervals. This network explicitly explicitly mirrors that process.
2. **Generalization:** By normalizing waves and centering on R-peaks, a PVC from Patient A looks morphologically identical to a PVC from Patient B, regardless of their resting heart rate or sensor calibration. 
3. **No Padding Hallucinations:** Because the inputs are fixed-length cutouts of beats rather than variable-length 10-second strips, we entirely avoid the dying gradients associated with massive zero-padded strings.

## Appendix A: Full Source with Line Numbers

The following appendices include exact source lines for traceability.


### File: `colab_ecg_preprocess.py`

```python
0001: """
0002: ECG preprocessing pipeline for Colab.
0003: 
0004: Step-1 only: clean and segment ECG into beats, score noise, and export
0005: beat-level + context-window datasets for downstream training.
0006: 
0007: BLOCK 1 (Colab install):
0008:     !pip install -q pandas numpy scipy scikit-learn matplotlib openpyxl neurokit2
0009: 
0010: BLOCK 2 (Run script with CLI args):
0011:     !pip install -q pandas numpy scipy scikit-learn matplotlib openpyxl neurokit2
0012:     !python colab_ecg_preprocess.py \
0013:         --input /content/723_labelled_multiclass.xlsx \
0014:         --output_dir /content/ecg_preprocessed \
0015:         --duration_sec 10
0016: 
0017: BLOCK 3 (View outputs):
0018:     from IPython.display import Image, display
0019:     display(Image('/content/ecg_preprocessed/class_templates.png'))
0020:     display(Image('/content/ecg_preprocessed/global_clusters_pca2d.png'))
0021:     display(Image('/content/ecg_preprocessed/prototype_gallery.png'))
0022: 
0023: BLOCK 4 (Inspect prototype map):
0024:     import pandas as pd
0025:     pd.read_csv('/content/ecg_preprocessed/prototype_library.csv').head(20)
0026: 
0027: BLOCK 5 (Inspect key-value prototypes anytime):
0028:     from colab_ecg_preprocess import inspect_prototype_keys
0029:     inspect_prototype_keys('/content/ecg_preprocessed', top_n=20)
0030: 
0031: BLOCK 6 (Plot prototypes for one label):
0032:     from colab_ecg_preprocess import plot_label_prototypes
0033:     plot_label_prototypes('/content/ecg_preprocessed', label_id=2)
0034: """
0035: 
0036: from __future__ import annotations
0037: 
0038: import argparse
0039: import json
0040: import math
0041: import sys
0042: from dataclasses import dataclass
0043: from pathlib import Path
0044: from typing import Dict, List, Optional, Tuple
0045: 
0046: import numpy as np
0047: import pandas as pd
0048: from scipy import signal
0049: from sklearn.decomposition import PCA
0050: from sklearn.cluster import KMeans
0051: from sklearn.metrics.pairwise import cosine_similarity
0052: 
0053: try:
0054:     import neurokit2 as nk  # type: ignore
0055: except Exception:
0056:     nk = None
0057: 
0058: 
0059: DEFAULT_LABEL_NAMES = {
0060:     -1: "Noise",
0061:     0: "Normal",
0062:     1: "PAC",
0063:     2: "PVC",
0064:     3: "Sinus Bradycardia",
0065:     4: "Sinus Tachycardia",
0066:     5: "AFib",
0067:     6: "Atrial Flutter",
0068:     7: "VTach",
0069:     8: "VFib",
0070:     9: "Ventricular Ectopic",
0071:     10: "Couplets",
0072:     11: "Triplets",
0073:     12: "PVC Subtypes",
0074: }
0075: 
0076: 
0077: # =========================
0078: # BLOCK A: Core Utilities
0079: # =========================
0080: @dataclass
0081: class BeatRecord:
0082:     source_row: int
0083:     patient_id: Optional[int]
0084:     record_label: int
0085:     beat_index_in_record: int
0086:     r_index: int
0087:     rr_prev_sec: float
0088:     rr_next_sec: float
0089:     quality_score: float
0090:     noise_flag: bool
0091:     cluster_id: int
0092:     cluster_size: int
0093:     beat: np.ndarray
0094: 
0095: 
0096: def robust_zscore(x: np.ndarray) -> np.ndarray:
0097:     med = np.median(x)
0098:     mad = np.median(np.abs(x - med)) + 1e-9
0099:     return 0.6745 * (x - med) / mad
0100: 
0101: 
0102: def parse_wave(text: object) -> np.ndarray:
0103:     if not isinstance(text, str):
0104:         return np.array([], dtype=np.float32)
0105:     vals = []
0106:     for token in text.split(","):
0107:         token = token.strip()
0108:         if not token:
0109:             continue
0110:         try:
0111:             vals.append(float(token))
0112:         except ValueError:
0113:             return np.array([], dtype=np.float32)
0114:     return np.asarray(vals, dtype=np.float32)
0115: 
0116: 
0117: def infer_fs(wave_len: int, duration_sec: float) -> float:
0118:     return float(wave_len) / duration_sec
0119: 
0120: 
0121: def bandpass_ecg(wave: np.ndarray, fs: float, low_hz: float = 0.5, high_hz: float = 45.0) -> np.ndarray:
0122:     nyq = fs * 0.5
0123:     high_hz = min(high_hz, nyq - 1e-3)
0124:     if high_hz <= low_hz:
0125:         return wave.astype(np.float32, copy=False)
0126:     b, a = signal.butter(4, [low_hz / nyq, high_hz / nyq], btype="band")
0127:     return signal.filtfilt(b, a, wave).astype(np.float32)
0128: 
0129: 
0130: def normalize_wave(wave: np.ndarray) -> np.ndarray:
0131:     mu = float(np.mean(wave))
0132:     sigma = float(np.std(wave)) + 1e-8
0133:     return ((wave - mu) / sigma).astype(np.float32)
0134: 
0135: 
0136: def detect_rpeaks(ecg: np.ndarray, fs: float) -> np.ndarray:
0137:     if nk is not None:
0138:         try:
0139:             _, out = nk.ecg_peaks(ecg, sampling_rate=fs, method="neurokit")
0140:             peaks = np.asarray(out["ECG_R_Peaks"], dtype=np.int32)
0141:             if peaks.size > 0:
0142:                 return peaks
0143:         except Exception:
0144:             pass
0145: 
0146:     # Fallback: derivative-energy envelope + peak picking.
0147:     diff = np.diff(ecg, prepend=ecg[0])
0148:     energy = diff * diff
0149:     win = max(3, int(0.12 * fs))
0150:     kernel = np.ones(win, dtype=np.float32) / win
0151:     smooth = np.convolve(energy, kernel, mode="same")
0152:     thr = float(np.mean(smooth) + 0.5 * np.std(smooth))
0153:     min_dist = int(0.24 * fs)  # allow faster rhythms as well
0154:     peaks, _ = signal.find_peaks(smooth, height=thr, distance=max(1, min_dist))
0155:     return peaks.astype(np.int32)
0156: 
0157: 
0158: def extract_beats(ecg: np.ndarray, peaks: np.ndarray, fs: float, pre_ms: float, post_ms: float) -> Tuple[np.ndarray, np.ndarray]:
0159:     pre = int(round((pre_ms / 1000.0) * fs))
0160:     post = int(round((post_ms / 1000.0) * fs))
0161:     beats = []
0162:     kept_peaks = []
0163:     for p in peaks:
0164:         left = p - pre
0165:         right = p + post
0166:         if left < 0 or right >= len(ecg):
0167:             continue
0168:         beats.append(ecg[left:right + 1])
0169:         kept_peaks.append(p)
0170:     if not beats:
0171:         return np.empty((0, pre + post + 1), dtype=np.float32), np.empty((0,), dtype=np.int32)
0172:     return np.asarray(beats, dtype=np.float32), np.asarray(kept_peaks, dtype=np.int32)
0173: 
0174: 
0175: def beat_quality_features(beats: np.ndarray) -> Dict[str, np.ndarray]:
0176:     ptp = np.ptp(beats, axis=1)
0177:     slope = np.max(np.abs(np.diff(beats, axis=1)), axis=1)
0178:     start = beats[:, : max(4, beats.shape[1] // 10)]
0179:     end = beats[:, -max(4, beats.shape[1] // 10):]
0180:     edge_noise = np.std(np.concatenate([start, end], axis=1), axis=1)
0181:     return {"ptp": ptp, "slope": slope, "edge_noise": edge_noise}
0182: 
0183: 
0184: def score_noise(beats: np.ndarray, rr_prev: np.ndarray, rr_next: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
0185:     feats = beat_quality_features(beats)
0186:     z_ptp = robust_zscore(feats["ptp"])
0187:     z_slope = robust_zscore(feats["slope"])
0188:     z_edge = robust_zscore(feats["edge_noise"])
0189: 
0190:     rr_bad = (rr_prev < 0.2) | (rr_prev > 2.5) | (rr_next < 0.2) | (rr_next > 2.5)
0191:     morphology_bad = (z_ptp < -3.0) | (z_slope < -3.0) | (z_edge > 3.5) | (z_slope > 6.0)
0192: 
0193:     noise = rr_bad | morphology_bad
0194:     score = (np.abs(z_ptp) + np.abs(z_slope) + np.abs(z_edge)) / 3.0
0195:     return score.astype(np.float32), noise.astype(bool)
0196: 
0197: 
0198: def dominant_cluster_ids(beats: np.ndarray, max_k: int = 4) -> Tuple[np.ndarray, Dict[int, int]]:
0199:     n = len(beats)
0200:     if n < 12:
0201:         ids = np.zeros((n,), dtype=np.int32)
0202:         return ids, {0: n}
0203:     k = int(np.clip(round(math.sqrt(n / 60.0) + 1), 2, max_k))
0204:     model = KMeans(n_clusters=k, random_state=42, n_init=20)
0205:     ids = model.fit_predict(beats)
0206:     sizes = {cid: int(np.sum(ids == cid)) for cid in np.unique(ids)}
0207:     return ids.astype(np.int32), sizes
0208: 
0209: 
0210: def resolve_column(df: pd.DataFrame, aliases: List[str]) -> str:
0211:     lower_map = {c.lower().strip(): c for c in df.columns}
0212:     for alias in aliases:
0213:         if alias.lower() in lower_map:
0214:             return lower_map[alias.lower()]
0215:     raise KeyError(f"Missing required column. Tried aliases: {aliases}")
0216: 
0217: 
0218: def build_templates(beats: np.ndarray, labels: np.ndarray, noise: np.ndarray) -> Dict[int, np.ndarray]:
0219:     templates: Dict[int, np.ndarray] = {}
0220:     for cls in sorted(np.unique(labels)):
0221:         idx = np.where((labels == cls) & (~noise))[0]
0222:         if idx.size == 0:
0223:             continue
0224:         templates[int(cls)] = np.median(beats[idx], axis=0).astype(np.float32)
0225:     return templates
0226: 
0227: 
0228: def build_global_feature_space(
0229:     beats: np.ndarray,
0230:     rr_prev: np.ndarray,
0231:     rr_next: np.ndarray,
0232:     n_pca: int = 24,
0233: ) -> np.ndarray:
0234:     """
0235:     Build compact feature vectors for label-free grouping/retrieval.
0236:     Uses beat waveform + RR context ratios.
0237:     """
0238:     rr_prev_col = rr_prev.reshape(-1, 1)
0239:     rr_next_col = rr_next.reshape(-1, 1)
0240:     rr_ratio = (rr_prev_col + 1e-6) / (rr_next_col + 1e-6)
0241:     rr_feat = np.hstack([rr_prev_col, rr_next_col, rr_ratio]).astype(np.float32)
0242: 
0243:     pca_dim = int(min(n_pca, beats.shape[1], max(2, beats.shape[0] - 1)))
0244:     pca = PCA(n_components=pca_dim, random_state=42)
0245:     beat_emb = pca.fit_transform(beats).astype(np.float32)
0246: 
0247:     feat = np.hstack([beat_emb, rr_feat]).astype(np.float32)
0248:     feat_mu = feat.mean(axis=0, keepdims=True)
0249:     feat_std = feat.std(axis=0, keepdims=True) + 1e-6
0250:     feat = (feat - feat_mu) / feat_std
0251:     return feat.astype(np.float32)
0252: 
0253: 
0254: def cluster_global_beats(
0255:     features: np.ndarray,
0256:     usable_mask: np.ndarray,
0257:     n_clusters: int,
0258: ) -> Tuple[np.ndarray, np.ndarray]:
0259:     """
0260:     Cluster beats without using labels.
0261:     Returns:
0262:       cluster_id_all: cluster id for each beat (-1 for unusable beats)
0263:       cluster_size_all: cluster size for each beat (0 for unusable beats)
0264:     """
0265:     n = len(features)
0266:     cluster_id_all = np.full((n,), -1, dtype=np.int32)
0267:     cluster_size_all = np.zeros((n,), dtype=np.int32)
0268: 
0269:     usable_idx = np.where(usable_mask)[0]
0270:     if usable_idx.size < 2:
0271:         return cluster_id_all, cluster_size_all
0272: 
0273:     k = int(min(n_clusters, usable_idx.size))
0274:     k = max(2, k)
0275:     model = KMeans(n_clusters=k, random_state=42, n_init=20)
0276:     ids = model.fit_predict(features[usable_idx]).astype(np.int32)
0277:     cluster_id_all[usable_idx] = ids
0278: 
0279:     unique_ids, counts = np.unique(ids, return_counts=True)
0280:     size_map = {int(cid): int(c) for cid, c in zip(unique_ids, counts)}
0281:     for idx_local, gidx in enumerate(usable_idx):
0282:         cluster_size_all[gidx] = size_map[int(ids[idx_local])]
0283:     return cluster_id_all, cluster_size_all
0284: 
0285: 
0286: def save_global_cluster_plot(
0287:     features: np.ndarray,
0288:     cluster_ids: np.ndarray,
0289:     out_path: Path,
0290: ) -> None:
0291:     import matplotlib.pyplot as plt
0292: 
0293:     usable = cluster_ids >= 0
0294:     if usable.sum() < 3:
0295:         return
0296:     p2 = PCA(n_components=2, random_state=42).fit_transform(features[usable])
0297:     c = cluster_ids[usable]
0298:     plt.figure(figsize=(8, 6))
0299:     plt.scatter(p2[:, 0], p2[:, 1], c=c, s=7, alpha=0.75, cmap="tab20")
0300:     plt.title("Label-Free Beat Clusters (PCA-2D)")
0301:     plt.xlabel("PC1")
0302:     plt.ylabel("PC2")
0303:     plt.tight_layout()
0304:     plt.savefig(out_path, dpi=170)
0305:     plt.close()
0306: 
0307: 
0308: def build_prototype_library(
0309:     beats: np.ndarray,
0310:     labels: np.ndarray,
0311:     noise: np.ndarray,
0312:     global_cluster_ids: np.ndarray,
0313:     rr_prev: np.ndarray,
0314:     rr_next: np.ndarray,
0315: ) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
0316:     """
0317:     Build key-value prototype map:
0318:       key   -> prototype_id (global cluster id)
0319:       value -> centroid/median beat waveform + stats
0320:     """
0321:     rows = []
0322:     wave_map: Dict[str, np.ndarray] = {}
0323:     usable_idx = np.where((~noise) & (global_cluster_ids >= 0))[0]
0324:     if usable_idx.size == 0:
0325:         return pd.DataFrame(), wave_map
0326: 
0327:     cids = np.unique(global_cluster_ids[usable_idx])
0328:     for cid in cids:
0329:         idx = np.where((global_cluster_ids == cid) & (~noise))[0]
0330:         if idx.size == 0:
0331:             continue
0332: 
0333:         cluster_beats = beats[idx]
0334:         prototype = np.median(cluster_beats, axis=0).astype(np.float32)
0335:         wave_map[f"prototype_{int(cid)}"] = prototype
0336: 
0337:         cls_vals, cls_counts = np.unique(labels[idx], return_counts=True)
0338:         order = np.argsort(-cls_counts)
0339:         top_pairs = [(int(cls_vals[i]), int(cls_counts[i])) for i in order[:3]]
0340:         top_text = ";".join([f"{k}:{v}" for k, v in top_pairs])
0341: 
0342:         rows.append(
0343:             {
0344:                 "prototype_id": int(cid),
0345:                 "size": int(idx.size),
0346:                 "dominant_label": int(top_pairs[0][0]) if top_pairs else -999,
0347:                 "label_mix_top3": top_text,
0348:                 "rr_prev_mean": float(np.mean(rr_prev[idx])),
0349:                 "rr_next_mean": float(np.mean(rr_next[idx])),
0350:                 "example_beat_ids": ",".join(map(str, idx[:10].tolist())),
0351:             }
0352:         )
0353: 
0354:     proto_df = pd.DataFrame(rows).sort_values("size", ascending=False)
0355:     return proto_df, wave_map
0356: 
0357: 
0358: def save_prototype_gallery(
0359:     proto_df: pd.DataFrame,
0360:     wave_map: Dict[str, np.ndarray],
0361:     out_path: Path,
0362:     max_items: int = 25,
0363: ) -> None:
0364:     import matplotlib.pyplot as plt
0365: 
0366:     if proto_df.empty:
0367:         return
0368:     show_df = proto_df.head(max_items)
0369:     n = len(show_df)
0370:     cols = 5
0371:     rows = int(math.ceil(n / cols))
0372:     plt.figure(figsize=(3.2 * cols, 2.2 * rows))
0373:     for i, (_, r) in enumerate(show_df.iterrows(), start=1):
0374:         pid = int(r["prototype_id"])
0375:         w = wave_map.get(f"prototype_{pid}")
0376:         if w is None:
0377:             continue
0378:         ax = plt.subplot(rows, cols, i)
0379:         ax.plot(w, linewidth=1.4)
0380:         ax.set_title(f"P{pid} n={int(r['size'])}", fontsize=8)
0381:         ax.set_xticks([])
0382:         ax.set_yticks([])
0383:     plt.tight_layout()
0384:     plt.savefig(out_path, dpi=170)
0385:     plt.close()
0386: 
0387: 
0388: def seed_similarity_mining(
0389:     features: np.ndarray,
0390:     beat_df: pd.DataFrame,
0391:     seed_csv: str,
0392:     top_k: int,
0393:     search_rows: int,
0394: ) -> pd.DataFrame:
0395:     """
0396:     seed_csv format:
0397:       beat_global_id,label_name
0398:       123,PVC
0399:       454,PVC
0400:       700,Normal
0401:     """
0402:     seeds = pd.read_csv(seed_csv)
0403:     required = {"beat_global_id", "label_name"}
0404:     if not required.issubset(set(seeds.columns)):
0405:         raise ValueError(f"seed_csv must contain columns: {required}")
0406: 
0407:     seeds = seeds.dropna(subset=["beat_global_id", "label_name"]).copy()
0408:     seeds["beat_global_id"] = seeds["beat_global_id"].astype(int)
0409:     valid_seed_ids = [x for x in seeds["beat_global_id"].tolist() if 0 <= x < len(features)]
0410:     if not valid_seed_ids:
0411:         return pd.DataFrame(
0412:             columns=[
0413:                 "seed_label",
0414:                 "seed_beat_global_id",
0415:                 "candidate_beat_global_id",
0416:                 "similarity",
0417:                 "source_row",
0418:                 "patient_id",
0419:                 "rr_prev_sec",
0420:                 "rr_next_sec",
0421:             ]
0422:         )
0423: 
0424:     max_row = int(beat_df["source_row"].min()) + int(search_rows) - 1
0425:     candidate_mask = (beat_df["noise_flag"].values == 0) & (beat_df["source_row"].values <= max_row)
0426:     candidate_ids = np.where(candidate_mask)[0]
0427:     if candidate_ids.size == 0:
0428:         return pd.DataFrame()
0429: 
0430:     rows = []
0431:     for _, srow in seeds.iterrows():
0432:         sid = int(srow["beat_global_id"])
0433:         if sid < 0 or sid >= len(features):
0434:             continue
0435:         seed_label = str(srow["label_name"])
0436:         sim = cosine_similarity(features[sid:sid + 1], features[candidate_ids])[0]
0437:         order = np.argsort(-sim)[:top_k]
0438:         for oid in order:
0439:             gid = int(candidate_ids[oid])
0440:             rows.append(
0441:                 {
0442:                     "seed_label": seed_label,
0443:                     "seed_beat_global_id": sid,
0444:                     "candidate_beat_global_id": gid,
0445:                     "similarity": float(sim[oid]),
0446:                     "source_row": int(beat_df.iloc[gid]["source_row"]),
0447:                     "patient_id": int(beat_df.iloc[gid]["patient_id"]),
0448:                     "rr_prev_sec": float(beat_df.iloc[gid]["rr_prev_sec"]),
0449:                     "rr_next_sec": float(beat_df.iloc[gid]["rr_next_sec"]),
0450:                 }
0451:             )
0452:     return pd.DataFrame(rows).sort_values(["seed_label", "similarity"], ascending=[True, False])
0453: 
0454: 
0455: def save_template_plot(templates: Dict[int, np.ndarray], out_path: Path) -> None:
0456:     import matplotlib.pyplot as plt
0457: 
0458:     if not templates:
0459:         return
0460:     plt.figure(figsize=(11, 6))
0461:     for cls, tpl in templates.items():
0462:         name = DEFAULT_LABEL_NAMES.get(cls, str(cls))
0463:         plt.plot(tpl, label=f"{cls}: {name}", linewidth=1.6)
0464:     plt.title("Representative Beat Template Per Class")
0465:     plt.xlabel("Sample")
0466:     plt.ylabel("Normalized Amplitude")
0467:     plt.legend(loc="best", fontsize=8)
0468:     plt.tight_layout()
0469:     plt.savefig(out_path, dpi=160)
0470:     plt.close()
0471: 
0472: 
0473: def make_context_windows(beats: np.ndarray, labels: np.ndarray, source_row: np.ndarray, noise: np.ndarray, window_radius: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
0474:     x_rows = []
0475:     y_rows = []
0476:     mask_rows = []
0477:     beat_len = beats.shape[1]
0478:     group_ids = np.unique(source_row)
0479: 
0480:     for gid in group_ids:
0481:         idx = np.where((source_row == gid) & (~noise))[0]
0482:         if idx.size == 0:
0483:             continue
0484:         ordered = idx  # already in record order
0485:         for j, center_idx in enumerate(ordered):
0486:             window = []
0487:             mask = []
0488:             for shift in range(-window_radius, window_radius + 1):
0489:                 pos = j + shift
0490:                 if 0 <= pos < len(ordered):
0491:                     window.append(beats[ordered[pos]])
0492:                     mask.append(1.0)
0493:                 else:
0494:                     window.append(np.zeros((beat_len,), dtype=np.float32))
0495:                     mask.append(0.0)
0496:             x_rows.append(np.stack(window, axis=0))
0497:             y_rows.append(labels[center_idx])
0498:             mask_rows.append(np.asarray(mask, dtype=np.float32))
0499: 
0500:     if not x_rows:
0501:         return (
0502:             np.empty((0, 2 * window_radius + 1, beat_len), dtype=np.float32),
0503:             np.empty((0,), dtype=np.int32),
0504:             np.empty((0, 2 * window_radius + 1), dtype=np.float32),
0505:         )
0506:     return np.asarray(x_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.int32), np.asarray(mask_rows, dtype=np.float32)
0507: 
0508: 
0509: def preprocess(args: argparse.Namespace) -> None:
0510:     # =========================
0511:     # BLOCK B: Load Input Data
0512:     # =========================
0513:     out_dir = Path(args.output_dir)
0514:     out_dir.mkdir(parents=True, exist_ok=True)
0515: 
0516:     df = pd.read_excel(args.input)
0517:     wave_col = resolve_column(df, ["ECG Wave", "ecg_wave", "wave"])
0518:     label_col = resolve_column(df, ["Label", "label"])
0519:     patient_col = resolve_column(df, ["Patient ID", "patient_id", "patient"])
0520: 
0521:     all_beats: List[np.ndarray] = []
0522:     all_labels: List[int] = []
0523:     all_source_row: List[int] = []
0524:     all_patient_id: List[int] = []
0525:     all_r: List[int] = []
0526:     all_rr_prev: List[float] = []
0527:     all_rr_next: List[float] = []
0528:     all_local_index: List[int] = []
0529: 
0530:     row_stats = []
0531:     skipped_rows = 0
0532: 
0533:     # ============================================
0534:     # BLOCK C: Per-record signal -> beat extraction
0535:     # ============================================
0536:     for ridx, row in df.iterrows():
0537:         wave = parse_wave(row.get(wave_col))
0538:         if wave.size < 200:
0539:             skipped_rows += 1
0540:             continue
0541: 
0542:         fs = args.fs if args.fs > 0 else infer_fs(len(wave), args.duration_sec)
0543:         ecg = bandpass_ecg(wave, fs=fs, low_hz=args.low_hz, high_hz=args.high_hz)
0544:         ecg = normalize_wave(ecg)
0545:         peaks = detect_rpeaks(ecg, fs=fs)
0546:         beats, kept = extract_beats(ecg, peaks, fs=fs, pre_ms=args.pre_ms, post_ms=args.post_ms)
0547:         if beats.shape[0] == 0:
0548:             skipped_rows += 1
0549:             continue
0550: 
0551:         label = int(row.get(label_col))
0552:         patient = int(row.get(patient_col)) if pd.notna(row.get(patient_col)) else -1
0553: 
0554:         rr = np.diff(kept) / fs
0555:         rr_prev = np.concatenate([[rr[0] if rr.size else 1.0], rr]).astype(np.float32)
0556:         rr_next = np.concatenate([rr, [rr[-1] if rr.size else 1.0]]).astype(np.float32)
0557: 
0558:         for i in range(len(beats)):
0559:             all_beats.append(beats[i])
0560:             all_labels.append(label)
0561:             all_source_row.append(int(ridx))
0562:             all_patient_id.append(patient)
0563:             all_r.append(int(kept[i]))
0564:             all_rr_prev.append(float(rr_prev[i]))
0565:             all_rr_next.append(float(rr_next[i]))
0566:             all_local_index.append(i)
0567: 
0568:         row_stats.append((int(ridx), int(len(wave)), float(fs), int(len(peaks)), int(len(beats))))
0569: 
0570:     if not all_beats:
0571:         raise RuntimeError("No beats extracted. Check `duration_sec`, `fs`, or wave format.")
0572: 
0573:     beats = np.asarray(all_beats, dtype=np.float32)
0574:     labels = np.asarray(all_labels, dtype=np.int32)
0575:     source_row = np.asarray(all_source_row, dtype=np.int32)
0576:     patient_ids = np.asarray(all_patient_id, dtype=np.int32)
0577:     r_idx = np.asarray(all_r, dtype=np.int32)
0578:     rr_prev = np.asarray(all_rr_prev, dtype=np.float32)
0579:     rr_next = np.asarray(all_rr_next, dtype=np.float32)
0580:     local_idx = np.asarray(all_local_index, dtype=np.int32)
0581: 
0582:     # ============================
0583:     # BLOCK D: Noise score + filter
0584:     # ============================
0585:     quality_score, noise = score_noise(beats, rr_prev, rr_next)
0586: 
0587:     # label-aware clustering (optional) to split minor outlier morphologies
0588:     local_cluster_ids = np.full((len(beats),), -1, dtype=np.int32)
0589:     local_cluster_size = np.zeros((len(beats),), dtype=np.int32)
0590:     for cls in sorted(np.unique(labels)):
0591:         idx = np.where((labels == cls) & (~noise))[0]
0592:         if idx.size == 0:
0593:             continue
0594:         ids, sizes = dominant_cluster_ids(beats[idx], max_k=args.max_k_per_class)
0595:         for j, global_idx in enumerate(idx):
0596:             cid = int(ids[j])
0597:             local_cluster_ids[global_idx] = cid
0598:             local_cluster_size[global_idx] = sizes.get(cid, 0)
0599:         tiny_cluster = np.array([sizes[int(ids[j])] < args.min_cluster_size for j in range(len(idx))], dtype=bool)
0600:         noise[idx[tiny_cluster]] = True
0601: 
0602:     # ==================================
0603:     # BLOCK E: Build pattern templates/map
0604:     # ==================================
0605:     templates = build_templates(beats, labels, noise)
0606:     save_template_plot(templates, out_dir / "class_templates.png")
0607: 
0608:     # label-free representation map
0609:     features = build_global_feature_space(
0610:         beats=beats,
0611:         rr_prev=rr_prev,
0612:         rr_next=rr_next,
0613:         n_pca=args.feature_pca_dim,
0614:     )
0615:     global_cluster_ids, global_cluster_sizes = cluster_global_beats(
0616:         features=features,
0617:         usable_mask=(~noise),
0618:         n_clusters=args.global_clusters,
0619:     )
0620:     save_global_cluster_plot(features, global_cluster_ids, out_dir / "global_clusters_pca2d.png")
0621: 
0622:     # =========================================
0623:     # BLOCK F: Build consecutive context windows
0624:     # =========================================
0625:     x_ctx, y_ctx, m_ctx = make_context_windows(
0626:         beats=beats,
0627:         labels=labels,
0628:         source_row=source_row,
0629:         noise=noise,
0630:         window_radius=args.context_radius,
0631:     )
0632: 
0633:     beat_df = pd.DataFrame(
0634:         {
0635:             "beat_global_id": np.arange(len(beats), dtype=np.int32),
0636:             "source_row": source_row,
0637:             "patient_id": patient_ids,
0638:             "record_label": labels,
0639:             "record_label_name": [DEFAULT_LABEL_NAMES.get(int(x), str(int(x))) for x in labels],
0640:             "beat_index_in_record": local_idx,
0641:             "r_index": r_idx,
0642:             "rr_prev_sec": rr_prev,
0643:             "rr_next_sec": rr_next,
0644:             "quality_score": quality_score,
0645:             "noise_flag": noise.astype(np.int32),
0646:             "local_cluster_id": local_cluster_ids,
0647:             "local_cluster_size": local_cluster_size,
0648:             "global_cluster_id": global_cluster_ids,
0649:             "global_cluster_size": global_cluster_sizes,
0650:         }
0651:     )
0652: 
0653:     # =============================
0654:     # BLOCK G: Save core artefacts
0655:     # =============================
0656:     beat_df.to_csv(out_dir / "beat_metadata.csv", index=False)
0657:     np.save(out_dir / "beats.npy", beats)
0658:     np.save(out_dir / "labels.npy", labels)
0659:     np.save(out_dir / "noise_flags.npy", noise.astype(np.int32))
0660:     np.save(out_dir / "features.npy", features)
0661:     np.save(out_dir / "global_cluster_ids.npy", global_cluster_ids)
0662:     np.savez_compressed(out_dir / "context_windows.npz", X=x_ctx, y=y_ctx, mask=m_ctx)
0663:     np.savez_compressed(out_dir / "class_templates.npz", **{f"class_{k}": v for k, v in templates.items()})
0664: 
0665:     # =======================================
0666:     # BLOCK H: Save prototype key-value library
0667:     # =======================================
0668:     # Key-value prototype map (label-free)
0669:     proto_df, proto_wave_map = build_prototype_library(
0670:         beats=beats,
0671:         labels=labels,
0672:         noise=noise,
0673:         global_cluster_ids=global_cluster_ids,
0674:         rr_prev=rr_prev,
0675:         rr_next=rr_next,
0676:     )
0677:     if not proto_df.empty:
0678:         proto_df.to_csv(out_dir / "prototype_library.csv", index=False)
0679:         np.savez_compressed(out_dir / "prototype_library_waves.npz", **proto_wave_map)
0680:         save_prototype_gallery(proto_df, proto_wave_map, out_dir / "prototype_gallery.png")
0681: 
0682:     # ==========================================
0683:     # BLOCK I: Optional seed-based similar mining
0684:     # ==========================================
0685:     # optional: from 5-10 manually selected seeds, find similar beats in early records
0686:     if args.seed_csv:
0687:         mined = seed_similarity_mining(
0688:             features=features,
0689:             beat_df=beat_df,
0690:             seed_csv=args.seed_csv,
0691:             top_k=args.seed_top_k,
0692:             search_rows=args.seed_search_rows,
0693:         )
0694:         mined.to_csv(out_dir / "seed_mined_candidates.csv", index=False)
0695: 
0696:     # =========================
0697:     # BLOCK J: Summary outputs
0698:     # =========================
0699:     summary = {
0700:         "input_rows": int(len(df)),
0701:         "rows_skipped": int(skipped_rows),
0702:         "beats_extracted": int(len(beats)),
0703:         "beat_length": int(beats.shape[1]),
0704:         "noise_beats": int(np.sum(noise)),
0705:         "usable_beats": int(np.sum(~noise)),
0706:         "class_counts_raw": {str(int(k)): int(np.sum(labels == k)) for k in np.unique(labels)},
0707:         "class_counts_usable": {str(int(k)): int(np.sum((labels == k) & (~noise))) for k in np.unique(labels)},
0708:         "context_samples": int(len(y_ctx)),
0709:         "context_window_size": int(2 * args.context_radius + 1),
0710:         "global_clusters": int(len(np.unique(global_cluster_ids[global_cluster_ids >= 0]))),
0711:         "prototype_count": int(len(proto_df)),
0712:         "seed_mining_enabled": bool(args.seed_csv),
0713:     }
0714:     (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
0715: 
0716:     print("Preprocessing complete.")
0717:     print(json.dumps(summary, indent=2))
0718:     print(f"Saved outputs in: {out_dir}")
0719: 
0720: 
0721: # =========================
0722: # BLOCK K: Runtime / CLI
0723: # =========================
0724: def parse_args() -> argparse.Namespace:
0725:     p = argparse.ArgumentParser(description="Colab ECG preprocessing for beat-level + context datasets")
0726:     p.add_argument("--input", required=True, help="Path to xlsx file")
0727:     p.add_argument("--output_dir", required=True, help="Directory to save outputs")
0728:     p.add_argument("--duration_sec", type=float, default=10.0, help="Record duration used for fs inference")
0729:     p.add_argument("--fs", type=float, default=-1.0, help="Sampling rate; if <=0 infer from wave length / duration")
0730:     p.add_argument("--low_hz", type=float, default=0.5, help="Bandpass low cutoff")
0731:     p.add_argument("--high_hz", type=float, default=45.0, help="Bandpass high cutoff")
0732:     p.add_argument("--pre_ms", type=float, default=250.0, help="Beat window before R peak")
0733:     p.add_argument("--post_ms", type=float, default=400.0, help="Beat window after R peak")
0734:     p.add_argument("--max_k_per_class", type=int, default=4, help="Max clusters per class for grouping")
0735:     p.add_argument("--min_cluster_size", type=int, default=12, help="Clusters smaller than this are flagged as noise")
0736:     p.add_argument("--context_radius", type=int, default=2, help="Consecutive beat radius (2 -> window of 5)")
0737:     p.add_argument("--feature_pca_dim", type=int, default=24, help="PCA dimensions for label-free beat map")
0738:     p.add_argument("--global_clusters", type=int, default=20, help="Number of global label-free beat clusters")
0739:     p.add_argument("--seed_csv", type=str, default="", help="Optional CSV with columns beat_global_id,label_name")
0740:     p.add_argument("--seed_top_k", type=int, default=50, help="Top similar beats per seed")
0741:     p.add_argument("--seed_search_rows", type=int, default=100, help="Search similar beats in first N records")
0742:     return p.parse_args()
0743: 
0744: 
0745: def visualize_preprocessed_data(output_dir: str) -> None:
0746:     """
0747:     Utility function to visualize the extracted beat templates and global clusters
0748:     directly inside a Colab Notebook cell block.
0749: 
0750:     Example to run in Colab after preprocessing:
0751:         from colab_ecg_preprocess import visualize_preprocessed_data
0752:         visualize_preprocessed_data("/content/ecg_preprocessed")
0753:     """
0754:     import matplotlib.pyplot as plt
0755:     try:
0756:         from IPython.display import display
0757:         is_notebook = True
0758:     except ImportError:
0759:         is_notebook = False
0760: 
0761:     out_dir = Path(output_dir)
0762: 
0763:     print("=== Loading Class Templates ===")
0764:     npz_path = out_dir / "class_templates.npz"
0765:     if npz_path.exists():
0766:         data = np.load(npz_path)
0767:         classes = sorted(data.files, key=lambda x: int(x.split('_')[1]))
0768:         
0769:         n_classes = len(classes)
0770:         cols = min(4, n_classes) if n_classes > 0 else 1
0771:         rows = (n_classes + cols - 1) // cols
0772:         
0773:         if n_classes > 0:
0774:             plt.figure(figsize=(15, 3 * rows))
0775:             for i, cls_key in enumerate(classes):
0776:                 plt.subplot(rows, cols, i + 1)
0777:                 plt.plot(data[cls_key], color='#1f77b4', linewidth=1.5)
0778:                 plt.title(f"Template: {cls_key.replace('_', ' ')}")
0779:                 plt.xlabel("Sample")
0780:                 plt.ylabel("Normalized Amplitude")
0781:                 plt.grid(True, alpha=0.3)
0782:             plt.tight_layout()
0783:             plt.show()
0784:         else:
0785:             print("No class templates found in file.")
0786:     else:
0787:         print(f"File not found: {npz_path}")
0788: 
0789:     print("\n=== Loading Label-Free Beat Clusters ===")
0790:     feats_path = out_dir / "features.npy"
0791:     cluster_ids_path = out_dir / "global_cluster_ids.npy"
0792:     
0793:     if feats_path.exists() and cluster_ids_path.exists():
0794:         features = np.load(feats_path)
0795:         cluster_ids = np.load(cluster_ids_path)
0796:         usable = cluster_ids >= 0
0797:         if usable.sum() > 2:
0798:             p2 = PCA(n_components=2, random_state=42).fit_transform(features[usable])
0799:             c = cluster_ids[usable]
0800:             
0801:             plt.figure(figsize=(8, 6))
0802:             sc = plt.scatter(p2[:, 0], p2[:, 1], c=c, s=15, alpha=0.7, cmap="tab20")
0803:             plt.title("Label-Free Beat Clusters (PCA-2D)")
0804:             plt.xlabel("PC1")
0805:             plt.ylabel("PC2")
0806:             plt.colorbar(sc, label="Global Cluster ID")
0807:             plt.grid(True, alpha=0.3)
0808:             plt.tight_layout()
0809:             plt.show()
0810:         else:
0811:             print("Not enough usable beats to plot clusters.")
0812:     else:
0813:         print(f"Cluster/Feature files not found in {out_dir}")
0814: 
0815: 
0816: def inspect_prototype_keys(output_dir: str, top_n: int = 20) -> None:
0817:     """
0818:     Quick key-value inspector for prototype library.
0819:     Shows:
0820:       - key = prototype_id
0821:       - value = waveform array name in prototype_library_waves.npz
0822:       - stats = size, dominant label, label mix, example beat ids
0823:     """
0824:     out_dir = Path(output_dir)
0825:     csv_path = out_dir / "prototype_library.csv"
0826:     npz_path = out_dir / "prototype_library_waves.npz"
0827: 
0828:     if not csv_path.exists():
0829:         print(f"Not found: {csv_path}")
0830:         return
0831:     if not npz_path.exists():
0832:         print(f"Not found: {npz_path}")
0833:         return
0834: 
0835:     df = pd.read_csv(csv_path).sort_values("size", ascending=False).head(top_n)
0836:     waves = np.load(npz_path)
0837: 
0838:     print(f"Showing top {len(df)} prototype keys from: {out_dir}")
0839:     for _, r in df.iterrows():
0840:         pid = int(r["prototype_id"])
0841:         key = f"prototype_{pid}"
0842:         has_wave = key in waves.files
0843:         shape = tuple(waves[key].shape) if has_wave else None
0844:         print(
0845:             f"key={key:<14} "
0846:             f"size={int(r['size']):<5} "
0847:             f"dominant_label={int(r['dominant_label']):<3} "
0848:             f"mix={r['label_mix_top3']} "
0849:             f"wave_shape={shape}"
0850:         )
0851: 
0852: 
0853: def plot_label_prototypes(output_dir: str, label_id: int) -> None:
0854:     """
0855:     Plot prototype waveforms for one dominant label.
0856:     Example:
0857:         plot_label_prototypes('/content/ecg_preprocessed', label_id=2)
0858:     """
0859:     import matplotlib.pyplot as plt
0860: 
0861:     out_dir = Path(output_dir)
0862:     csv_path = out_dir / "prototype_library.csv"
0863:     npz_path = out_dir / "prototype_library_waves.npz"
0864: 
0865:     if not csv_path.exists():
0866:         print(f"Not found: {csv_path}")
0867:         return
0868:     if not npz_path.exists():
0869:         print(f"Not found: {npz_path}")
0870:         return
0871: 
0872:     proto = pd.read_csv(csv_path)
0873:     waves = np.load(npz_path)
0874:     rows = proto[proto["dominant_label"] == int(label_id)].sort_values("size", ascending=False)
0875: 
0876:     print(f"Found {len(rows)} prototypes for label {label_id} ({DEFAULT_LABEL_NAMES.get(label_id, 'Unknown')})")
0877:     if rows.empty:
0878:         return
0879: 
0880:     plt.figure(figsize=(12, 5))
0881:     for _, r in rows.iterrows():
0882:         pid = int(r["prototype_id"])
0883:         key = f"prototype_{pid}"
0884:         if key not in waves.files:
0885:             continue
0886:         w = waves[key]
0887:         plt.plot(w, label=f"P{pid} n={int(r['size'])}", alpha=0.9)
0888:     plt.title(f"Prototype Patterns for Label {label_id} - {DEFAULT_LABEL_NAMES.get(label_id, 'Unknown')}")
0889:     plt.xlabel("Sample")
0890:     plt.ylabel("Amplitude (normalized)")
0891:     plt.legend(fontsize=8)
0892:     plt.grid(alpha=0.2)
0893:     plt.tight_layout()
0894:     plt.show()
0895: 
0896: 
0897: if __name__ == "__main__":
0898:     # Supports both styles:
0899:     # 1) CLI: python colab_ecg_preprocess.py --input ... --output_dir ...
0900:     # 2) Direct notebook run with no CLI args (uses editable defaults below)
0901:     cli_has_required = ("--input" in sys.argv) and ("--output_dir" in sys.argv)
0902:     if cli_has_required:
0903:         preprocess(parse_args())
0904:     else:
0905:         class Args:
0906:             def __init__(self):
0907:                 self.input = "/content/drive/MyDrive/HeartcareAI/Models/LabelledData/723_labelled_multiclass.xlsx"
0908:                 self.output_dir = "/content/ecg_preprocessed"
0909:                 self.duration_sec = 10.0
0910:                 self.fs = -1.0
0911:                 self.low_hz = 0.5
0912:                 self.high_hz = 45.0
0913:                 self.pre_ms = 250.0
0914:                 self.post_ms = 400.0
0915:                 self.max_k_per_class = 4
0916:                 self.min_cluster_size = 12
0917:                 self.context_radius = 2
0918:                 self.feature_pca_dim = 24
0919:                 self.global_clusters = 20
0920:                 self.seed_csv = ""
0921:                 self.seed_top_k = 50
0922:                 self.seed_search_rows = 100
0923: 
0924:         preprocess(Args())
```

### File: `colab_ecg_train_v2.py`

```python
0001: """
0002: ECG training V2 (Colab): morphology + context + prototype similarity + noise gate.
0003: 
0004: BLOCK 1 (Install):
0005:     !pip install -q numpy pandas scikit-learn torch
0006: 
0007: BLOCK 2 (Train with CLI):
0008:     !python colab_ecg_train_v2.py \
0009:       --data_dirs /content/ecg_preprocessed \
0010:       --out_dir /content/ecg_trained_v2 \
0011:       --epochs 25 \
0012:       --batch_size 128
0013: 
0014: BLOCK 3 (Notebook-safe direct run):
0015:     # Run file directly in notebook cell; __main__ defaults will be used.
0016: """
0017: 
0018: from __future__ import annotations
0019: 
0020: import argparse
0021: import json
0022: import math
0023: import sys
0024: from pathlib import Path
0025: from typing import Dict, List, Tuple
0026: 
0027: import numpy as np
0028: import pandas as pd
0029: import torch
0030: import torch.nn as nn
0031: from sklearn.metrics import classification_report, confusion_matrix
0032: from sklearn.model_selection import train_test_split
0033: from torch.utils.data import DataLoader, Dataset
0034: 
0035: 
0036: # =========================
0037: # BLOCK A: Data Utilities
0038: # =========================
0039: def _mix_purity(label_mix_top3: str) -> float:
0040:     # format like "2:3090;0:1893;-1:158"
0041:     if not isinstance(label_mix_top3, str) or ":" not in label_mix_top3:
0042:         return 0.0
0043:     pairs = []
0044:     for part in label_mix_top3.split(";"):
0045:         part = part.strip()
0046:         if ":" not in part:
0047:             continue
0048:         a, b = part.split(":", 1)
0049:         try:
0050:             pairs.append((int(a), int(b)))
0051:         except ValueError:
0052:             continue
0053:     if not pairs:
0054:         return 0.0
0055:     counts = [c for _, c in pairs]
0056:     return float(max(counts)) / max(1.0, float(sum(counts)))
0057: 
0058: 
0059: def load_prototype_bank(
0060:     data_dirs: List[str],
0061:     min_proto_size: int,
0062:     min_proto_purity: float,
0063:     max_prototypes: int,
0064: ) -> Tuple[np.ndarray, np.ndarray]:
0065:     waves_all = []
0066:     labels_all = []
0067: 
0068:     for d in data_dirs:
0069:         dpath = Path(d)
0070:         csv_path = dpath / "prototype_library.csv"
0071:         npz_path = dpath / "prototype_library_waves.npz"
0072:         if not (csv_path.exists() and npz_path.exists()):
0073:             continue
0074: 
0075:         df = pd.read_csv(csv_path)
0076:         waves = np.load(npz_path)
0077:         if "prototype_id" not in df.columns:
0078:             continue
0079: 
0080:         for _, r in df.iterrows():
0081:             size = int(r.get("size", 0))
0082:             purity = _mix_purity(str(r.get("label_mix_top3", "")))
0083:             if size < min_proto_size or purity < min_proto_purity:
0084:                 continue
0085:             pid = int(r["prototype_id"])
0086:             key = f"prototype_{pid}"
0087:             if key not in waves.files:
0088:                 continue
0089:             w = waves[key].astype(np.float32)
0090:             lbl = int(r.get("dominant_label", -1))
0091:             waves_all.append(w)
0092:             labels_all.append(lbl)
0093: 
0094:     if not waves_all:
0095:         return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)
0096: 
0097:     # keep largest subset if too many
0098:     waves_np = np.asarray(waves_all, dtype=np.float32)
0099:     labels_np = np.asarray(labels_all, dtype=np.int64)
0100:     if len(waves_np) > max_prototypes:
0101:         idx = np.arange(len(waves_np))
0102:         np.random.seed(42)
0103:         np.random.shuffle(idx)
0104:         idx = idx[:max_prototypes]
0105:         waves_np = waves_np[idx]
0106:         labels_np = labels_np[idx]
0107:     return waves_np, labels_np
0108: 
0109: 
0110: def cosine_proto_features(
0111:     center_beats: np.ndarray,
0112:     proto_waves: np.ndarray,
0113:     proto_labels: np.ndarray,
0114:     top_k_sim: int = 8,
0115: ) -> np.ndarray:
0116:     """
0117:     Build prototype feature vector:
0118:       [top-k cosine similarities] + [max similarity per prototype label]
0119:     """
0120:     n = len(center_beats)
0121:     if proto_waves.size == 0:
0122:         return np.zeros((n, top_k_sim), dtype=np.float32)
0123: 
0124:     x = center_beats.astype(np.float32)
0125:     p = proto_waves.astype(np.float32)
0126: 
0127:     x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
0128:     p = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-8)
0129: 
0130:     sims = x @ p.T  # (N, P)
0131:     k = min(top_k_sim, sims.shape[1])
0132:     topk = -np.sort(-sims, axis=1)[:, :k]
0133: 
0134:     uniq_labels = sorted(np.unique(proto_labels).tolist())
0135:     label_max = []
0136:     for lbl in uniq_labels:
0137:         idx = np.where(proto_labels == lbl)[0]
0138:         if idx.size == 0:
0139:             label_max.append(np.zeros((n, 1), dtype=np.float32))
0140:         else:
0141:             label_max.append(np.max(sims[:, idx], axis=1, keepdims=True).astype(np.float32))
0142:     label_max_np = np.concatenate(label_max, axis=1) if label_max else np.zeros((n, 0), dtype=np.float32)
0143:     return np.concatenate([topk.astype(np.float32), label_max_np], axis=1).astype(np.float32)
0144: 
0145: 
0146: def load_training_data(
0147:     data_dirs: List[str],
0148:     min_proto_size: int,
0149:     min_proto_purity: float,
0150:     max_prototypes: int,
0151:     top_k_sim: int,
0152: ):
0153:     xs, ys = [], []
0154:     for d in data_dirs:
0155:         p = Path(d) / "context_windows.npz"
0156:         if not p.exists():
0157:             continue
0158:         npz = np.load(p)
0159:         xs.append(npz["X"].astype(np.float32))  # (N, W, L)
0160:         ys.append(npz["y"].astype(np.int64))    # raw label values, may include -1
0161:     if not xs:
0162:         raise RuntimeError("No context_windows.npz found in --data_dirs")
0163: 
0164:     X = np.concatenate(xs, axis=0)
0165:     y_raw = np.concatenate(ys, axis=0)
0166:     center = X[:, X.shape[1] // 2, :]
0167: 
0168:     proto_waves, proto_labels = load_prototype_bank(
0169:         data_dirs=data_dirs,
0170:         min_proto_size=min_proto_size,
0171:         min_proto_purity=min_proto_purity,
0172:         max_prototypes=max_prototypes,
0173:     )
0174:     X_proto = cosine_proto_features(center, proto_waves, proto_labels, top_k_sim=top_k_sim)
0175: 
0176:     # class labels only for non-noise
0177:     valid_class = y_raw >= 0
0178:     uniq_cls = sorted(np.unique(y_raw[valid_class]).tolist()) if np.any(valid_class) else []
0179:     remap = {old: i for i, old in enumerate(uniq_cls)}
0180:     y_class = np.full_like(y_raw, fill_value=-1, dtype=np.int64)
0181:     for old, new in remap.items():
0182:         y_class[y_raw == old] = new
0183: 
0184:     # noise gate target
0185:     y_gate = (y_raw < 0).astype(np.float32)
0186: 
0187:     meta = {
0188:         "raw_class_values": uniq_cls,
0189:         "num_prototypes": int(len(proto_waves)),
0190:         "proto_feature_dim": int(X_proto.shape[1]),
0191:     }
0192:     return X, X_proto, y_class, y_gate, meta
0193: 
0194: 
0195: # =========================
0196: # BLOCK B: Model
0197: # =========================
0198: class ECGHybridDataset(Dataset):
0199:     def __init__(self, x_ctx: np.ndarray, x_proto: np.ndarray, y_cls: np.ndarray, y_gate: np.ndarray):
0200:         self.x_ctx = torch.tensor(x_ctx, dtype=torch.float32)
0201:         self.x_proto = torch.tensor(x_proto, dtype=torch.float32)
0202:         self.y_cls = torch.tensor(y_cls, dtype=torch.long)
0203:         self.y_gate = torch.tensor(y_gate, dtype=torch.float32)
0204: 
0205:     def __len__(self):
0206:         return len(self.y_cls)
0207: 
0208:     def __getitem__(self, idx):
0209:         return self.x_ctx[idx], self.x_proto[idx], self.y_cls[idx], self.y_gate[idx]
0210: 
0211: 
0212: class ECGHybridNet(nn.Module):
0213:     """
0214:     3-branch model:
0215:       1) beat morphology feature
0216:       2) temporal context feature
0217:       3) prototype similarity feature
0218:     outputs:
0219:       - class logits
0220:       - noise gate logit
0221:     """
0222:     def __init__(self, num_classes: int, proto_dim: int):
0223:         super().__init__()
0224:         self.beat_encoder = nn.Sequential(
0225:             nn.Conv1d(1, 24, kernel_size=7, padding=3),
0226:             nn.BatchNorm1d(24),
0227:             nn.ReLU(),
0228:             nn.MaxPool1d(2),
0229:             nn.Conv1d(24, 48, kernel_size=5, padding=2),
0230:             nn.BatchNorm1d(48),
0231:             nn.ReLU(),
0232:             nn.MaxPool1d(2),
0233:             nn.Conv1d(48, 64, kernel_size=3, padding=1),
0234:             nn.ReLU(),
0235:             nn.AdaptiveAvgPool1d(1),
0236:         )
0237:         self.context_lstm = nn.LSTM(
0238:             input_size=64,
0239:             hidden_size=96,
0240:             num_layers=1,
0241:             batch_first=True,
0242:             bidirectional=True,
0243:         )
0244:         self.proto_mlp = nn.Sequential(
0245:             nn.Linear(max(1, proto_dim), 64),
0246:             nn.ReLU(),
0247:             nn.Dropout(0.1),
0248:             nn.Linear(64, 32),
0249:             nn.ReLU(),
0250:         )
0251:         self.fusion = nn.Sequential(
0252:             nn.Linear(64 + 192 + 32, 128),
0253:             nn.ReLU(),
0254:             nn.Dropout(0.2),
0255:             nn.Linear(128, 96),
0256:             nn.ReLU(),
0257:         )
0258:         self.class_head = nn.Linear(96, num_classes)
0259:         self.gate_head = nn.Linear(96, 1)
0260: 
0261:     def forward(self, x_ctx, x_proto):
0262:         # x_ctx: (B, W, L), x_proto: (B, D)
0263:         b, w, l = x_ctx.shape
0264:         x = x_ctx.reshape(b * w, 1, l)
0265:         beat_feat = self.beat_encoder(x).squeeze(-1)   # (B*W,64)
0266:         beat_feat = beat_feat.reshape(b, w, 64)        # (B,W,64)
0267: 
0268:         morph_feat = beat_feat[:, w // 2, :]           # center beat morphology
0269:         ctx_out, _ = self.context_lstm(beat_feat)      # (B,W,192)
0270:         ctx_feat = ctx_out[:, w // 2, :]               # center with temporal context
0271: 
0272:         proto_feat = self.proto_mlp(x_proto)
0273:         fused = self.fusion(torch.cat([morph_feat, ctx_feat, proto_feat], dim=1))
0274:         class_logits = self.class_head(fused)
0275:         gate_logit = self.gate_head(fused).squeeze(1)
0276:         return class_logits, gate_logit
0277: 
0278: 
0279: # =========================
0280: # BLOCK C: Train / Eval
0281: # =========================
0282: def split_data(X, XP, YC, YG, test_size=0.2):
0283:     idx = np.arange(len(YC))
0284:     # stratify by raw behavior proxy: class id where valid else special id
0285:     strat = YC.copy()
0286:     strat[strat < 0] = np.max(strat) + 1 if np.any(strat >= 0) else 0
0287:     tr_idx, va_idx = train_test_split(idx, test_size=test_size, random_state=42, stratify=strat)
0288:     return (
0289:         X[tr_idx], XP[tr_idx], YC[tr_idx], YG[tr_idx],
0290:         X[va_idx], XP[va_idx], YC[va_idx], YG[va_idx],
0291:     )
0292: 
0293: 
0294: def train_v2(args):
0295:     X, XP, YC, YG, meta = load_training_data(
0296:         data_dirs=args.data_dirs,
0297:         min_proto_size=args.min_proto_size,
0298:         min_proto_purity=args.min_proto_purity,
0299:         max_prototypes=args.max_prototypes,
0300:         top_k_sim=args.top_k_sim,
0301:     )
0302:     num_classes = int(np.max(YC) + 1) if np.any(YC >= 0) else 1
0303: 
0304:     Xtr, XPtr, YCtr, YGtr, Xva, XPva, YCva, YGva = split_data(X, XP, YC, YG, test_size=args.val_split)
0305:     tr_ds = ECGHybridDataset(Xtr, XPtr, YCtr, YGtr)
0306:     va_ds = ECGHybridDataset(Xva, XPva, YCva, YGva)
0307:     tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
0308:     va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
0309: 
0310:     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
0311:     model = ECGHybridNet(num_classes=num_classes, proto_dim=XP.shape[1]).to(device)
0312: 
0313:     # Class imbalance handling for non-noise classes.
0314:     if args.use_class_weights:
0315:         valid_tr = YCtr >= 0
0316:         counts = np.bincount(YCtr[valid_tr], minlength=num_classes).astype(np.float32)
0317:         counts[counts == 0] = 1.0
0318:         inv = 1.0 / counts
0319:         w = inv / inv.sum() * num_classes
0320:         class_w = torch.tensor(w, dtype=torch.float32, device=device)
0321:         cls_loss_fn = nn.CrossEntropyLoss(weight=class_w)
0322:     else:
0323:         cls_loss_fn = nn.CrossEntropyLoss()
0324: 
0325:     gate_loss_fn = nn.BCEWithLogitsLoss()
0326:     optim = torch.optim.Adam(model.parameters(), lr=args.lr)
0327: 
0328:     out_dir = Path(args.out_dir)
0329:     out_dir.mkdir(parents=True, exist_ok=True)
0330:     best_path = out_dir / "best_hybrid_v2.pt"
0331:     best_val = 0.0
0332: 
0333:     for epoch in range(1, args.epochs + 1):
0334:         model.train()
0335:         tr_loss, tr_cls_ok, tr_cls_n = 0.0, 0, 0
0336:         tr_gate_ok, tr_gate_n = 0, 0
0337: 
0338:         for x_ctx, x_proto, y_cls, y_gate in tr_loader:
0339:             x_ctx = x_ctx.to(device)
0340:             x_proto = x_proto.to(device)
0341:             y_cls = y_cls.to(device)
0342:             y_gate = y_gate.to(device)
0343: 
0344:             optim.zero_grad()
0345:             cls_logits, gate_logit = model(x_ctx, x_proto)
0346: 
0347:             valid = y_cls >= 0
0348:             if torch.any(valid):
0349:                 l_cls = cls_loss_fn(cls_logits[valid], y_cls[valid])
0350:             else:
0351:                 l_cls = torch.tensor(0.0, device=device)
0352:             l_gate = gate_loss_fn(gate_logit, y_gate)
0353:             loss = l_cls + args.gate_loss_weight * l_gate
0354:             loss.backward()
0355:             optim.step()
0356: 
0357:             tr_loss += float(loss.item()) * x_ctx.size(0)
0358:             if torch.any(valid):
0359:                 pred = cls_logits.argmax(1)
0360:                 tr_cls_ok += int((pred[valid] == y_cls[valid]).sum().item())
0361:                 tr_cls_n += int(valid.sum().item())
0362:             gate_pred = (torch.sigmoid(gate_logit) >= 0.5).float()
0363:             tr_gate_ok += int((gate_pred == y_gate).sum().item())
0364:             tr_gate_n += int(y_gate.numel())
0365: 
0366:         # validation
0367:         model.eval()
0368:         va_cls_ok, va_cls_n = 0, 0
0369:         va_gate_ok, va_gate_n = 0, 0
0370:         all_p, all_t = [], []
0371:         with torch.no_grad():
0372:             for x_ctx, x_proto, y_cls, y_gate in va_loader:
0373:                 x_ctx = x_ctx.to(device)
0374:                 x_proto = x_proto.to(device)
0375:                 y_cls = y_cls.to(device)
0376:                 y_gate = y_gate.to(device)
0377: 
0378:                 cls_logits, gate_logit = model(x_ctx, x_proto)
0379:                 valid = y_cls >= 0
0380:                 if torch.any(valid):
0381:                     pred = cls_logits.argmax(1)
0382:                     va_cls_ok += int((pred[valid] == y_cls[valid]).sum().item())
0383:                     va_cls_n += int(valid.sum().item())
0384:                     all_p.extend(pred[valid].cpu().numpy().tolist())
0385:                     all_t.extend(y_cls[valid].cpu().numpy().tolist())
0386:                 gate_pred = (torch.sigmoid(gate_logit) >= 0.5).float()
0387:                 va_gate_ok += int((gate_pred == y_gate).sum().item())
0388:                 va_gate_n += int(y_gate.numel())
0389: 
0390:         tr_cls_acc = tr_cls_ok / max(1, tr_cls_n)
0391:         va_cls_acc = va_cls_ok / max(1, va_cls_n)
0392:         tr_gate_acc = tr_gate_ok / max(1, tr_gate_n)
0393:         va_gate_acc = va_gate_ok / max(1, va_gate_n)
0394:         print(
0395:             f"Epoch {epoch:02d} | loss {tr_loss/max(1,len(tr_ds)):.4f} "
0396:             f"| cls(train/val) {tr_cls_acc:.4f}/{va_cls_acc:.4f} "
0397:             f"| gate(train/val) {tr_gate_acc:.4f}/{va_gate_acc:.4f}"
0398:         )
0399: 
0400:         # select by class accuracy primarily; small tie-break on gate
0401:         score = va_cls_acc + 0.05 * va_gate_acc
0402:         if score > best_val:
0403:             best_val = score
0404:             torch.save(model.state_dict(), best_path)
0405: 
0406:     if all_t and all_p:
0407:         report = classification_report(all_t, all_p, output_dict=True, zero_division=0)
0408:         cm = confusion_matrix(all_t, all_p).tolist()
0409:     else:
0410:         report = {}
0411:         cm = []
0412: 
0413:     summary = {
0414:         "best_score": float(best_val),
0415:         "num_samples": int(len(YC)),
0416:         "num_classes_non_noise": int(num_classes),
0417:         "raw_class_values": meta["raw_class_values"],
0418:         "num_prototypes_used": meta["num_prototypes"],
0419:         "proto_feature_dim": meta["proto_feature_dim"],
0420:         "model_path": str(best_path),
0421:         "use_class_weights": bool(args.use_class_weights),
0422:         "confusion_matrix_non_noise": cm,
0423:         "classification_report_non_noise": report,
0424:     }
0425:     (out_dir / "training_summary_v2.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
0426:     print("Training V2 complete.")
0427:     print(json.dumps({k: v for k, v in summary.items() if "report" not in k}, indent=2))
0428: 
0429: 
0430: # =========================
0431: # BLOCK D: Runtime
0432: # =========================
0433: def parse_args():
0434:     p = argparse.ArgumentParser(description="Hybrid ECG training V2 (morphology + context + prototypes + gate)")
0435:     p.add_argument("--data_dirs", nargs="+", required=True, help="One or more preprocess output dirs")
0436:     p.add_argument("--out_dir", required=True)
0437:     p.add_argument("--epochs", type=int, default=25)
0438:     p.add_argument("--batch_size", type=int, default=128)
0439:     p.add_argument("--lr", type=float, default=1e-3)
0440:     p.add_argument("--val_split", type=float, default=0.2)
0441:     p.add_argument("--gate_loss_weight", type=float, default=0.5)
0442:     p.add_argument("--min_proto_size", type=int, default=120)
0443:     p.add_argument("--min_proto_purity", type=float, default=0.70)
0444:     p.add_argument("--max_prototypes", type=int, default=256)
0445:     p.add_argument("--top_k_sim", type=int, default=8)
0446:     p.add_argument("--use_class_weights", type=int, default=1, help="1=enable inverse-frequency CE class weights")
0447:     return p.parse_args()
0448: 
0449: 
0450: if __name__ == "__main__":
0451:     cli_has_required = ("--data_dirs" in sys.argv) and ("--out_dir" in sys.argv)
0452:     if cli_has_required:
0453:         train_v2(parse_args())
0454:     else:
0455:         class Args:
0456:             def __init__(self):
0457:                 self.data_dirs = ["/content/ecg_preprocessed"]
0458:                 self.out_dir = "/content/ecg_trained_v2"
0459:                 self.epochs = 25
0460:                 self.batch_size = 128
0461:                 self.lr = 1e-3
0462:                 self.val_split = 0.2
0463:                 self.gate_loss_weight = 0.5
0464:                 self.min_proto_size = 120
0465:                 self.min_proto_purity = 0.70
0466:                 self.max_prototypes = 256
0467:                 self.top_k_sim = 8
0468:                 self.use_class_weights = 1
0469: 
0470:         train_v2(Args())
```

### File: `predict_v2.py`

```python
0001: """
0002: Inference for hybrid V2 model with unknown/review gating.
0003: 
0004: BLOCK 1:
0005:     !pip install -q numpy pandas torch
0006: 
0007: BLOCK 2:
0008:     !python predict_v2.py \
0009:       --model_ckpt /content/ecg_trained_v2/best_hybrid_v2.pt \
0010:       --preprocessed_dir /content/ecg_preprocessed \
0011:       --prototype_dir /content/ecg_preprocessed \
0012:       --out_dir /content/ecg_predictions_v2
0013: """
0014: 
0015: from __future__ import annotations
0016: 
0017: import argparse
0018: import json
0019: from pathlib import Path
0020: from typing import Dict, List, Tuple
0021: 
0022: import numpy as np
0023: import pandas as pd
0024: import torch
0025: import torch.nn as nn
0026: 
0027: LABEL_NAMES = {
0028:     -1: "Noise",
0029:     0: "Normal",
0030:     1: "PAC",
0031:     2: "PVC",
0032:     3: "Sinus Bradycardia",
0033:     4: "Sinus Tachycardia",
0034:     5: "AFib",
0035:     6: "Atrial Flutter",
0036:     7: "VTach",
0037:     8: "VFib",
0038:     9: "Ventricular Ectopic",
0039:     10: "Couplets",
0040:     11: "Triplets",
0041:     12: "PVC Subtypes",
0042: }
0043: 
0044: 
0045: def parse_mix(mix: str):
0046:     out = []
0047:     if not isinstance(mix, str):
0048:         return out
0049:     for p in mix.split(";"):
0050:         if ":" not in p:
0051:             continue
0052:         a, b = p.split(":", 1)
0053:         try:
0054:             out.append((int(a), int(b)))
0055:         except ValueError:
0056:             pass
0057:     return out
0058: 
0059: 
0060: def purity(mix: str):
0061:     pairs = parse_mix(mix)
0062:     if not pairs:
0063:         return 0.0
0064:     counts = [c for _, c in pairs]
0065:     return max(counts) / max(1, sum(counts))
0066: 
0067: 
0068: def load_prototypes(proto_dir: Path, min_size: int, min_purity: float):
0069:     csv_path = proto_dir / "prototype_library.csv"
0070:     npz_path = proto_dir / "prototype_library_waves.npz"
0071:     if not csv_path.exists() or not npz_path.exists():
0072:         return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)
0073:     df = pd.read_csv(csv_path)
0074:     waves = np.load(npz_path)
0075:     w_all, l_all = [], []
0076:     for _, r in df.iterrows():
0077:         if int(r.get("size", 0)) < min_size:
0078:             continue
0079:         if purity(str(r.get("label_mix_top3", ""))) < min_purity:
0080:             continue
0081:         pid = int(r["prototype_id"])
0082:         key = f"prototype_{pid}"
0083:         if key not in waves.files:
0084:             continue
0085:         w_all.append(waves[key].astype(np.float32))
0086:         l_all.append(int(r.get("dominant_label", -1)))
0087:     if not w_all:
0088:         return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)
0089:     return np.asarray(w_all, dtype=np.float32), np.asarray(l_all, dtype=np.int64)
0090: 
0091: 
0092: def cosine_proto_features(center_beats: np.ndarray, proto_waves: np.ndarray, proto_labels: np.ndarray, top_k_sim: int):
0093:     n = len(center_beats)
0094:     if proto_waves.size == 0:
0095:         return np.zeros((n, top_k_sim), dtype=np.float32), np.zeros((n,), dtype=np.float32)
0096:     x = center_beats / (np.linalg.norm(center_beats, axis=1, keepdims=True) + 1e-8)
0097:     p = proto_waves / (np.linalg.norm(proto_waves, axis=1, keepdims=True) + 1e-8)
0098:     sims = x @ p.T
0099:     k = min(top_k_sim, sims.shape[1])
0100:     topk = -np.sort(-sims, axis=1)[:, :k]
0101:     uniq = sorted(np.unique(proto_labels).tolist())
0102:     per_lbl = []
0103:     for lbl in uniq:
0104:         idx = np.where(proto_labels == lbl)[0]
0105:         per_lbl.append(np.max(sims[:, idx], axis=1, keepdims=True))
0106:     feat = np.concatenate([topk] + per_lbl, axis=1).astype(np.float32) if per_lbl else topk.astype(np.float32)
0107:     max_sim = np.max(sims, axis=1).astype(np.float32)
0108:     return feat, max_sim
0109: 
0110: 
0111: class ECGHybridNet(nn.Module):
0112:     def __init__(self, num_classes: int, proto_dim: int):
0113:         super().__init__()
0114:         self.beat_encoder = nn.Sequential(
0115:             nn.Conv1d(1, 24, kernel_size=7, padding=3),
0116:             nn.BatchNorm1d(24),
0117:             nn.ReLU(),
0118:             nn.MaxPool1d(2),
0119:             nn.Conv1d(24, 48, kernel_size=5, padding=2),
0120:             nn.BatchNorm1d(48),
0121:             nn.ReLU(),
0122:             nn.MaxPool1d(2),
0123:             nn.Conv1d(48, 64, kernel_size=3, padding=1),
0124:             nn.ReLU(),
0125:             nn.AdaptiveAvgPool1d(1),
0126:         )
0127:         self.context_lstm = nn.LSTM(64, 96, num_layers=1, batch_first=True, bidirectional=True)
0128:         self.proto_mlp = nn.Sequential(
0129:             nn.Linear(max(1, proto_dim), 64),
0130:             nn.ReLU(),
0131:             nn.Dropout(0.1),
0132:             nn.Linear(64, 32),
0133:             nn.ReLU(),
0134:         )
0135:         self.fusion = nn.Sequential(
0136:             nn.Linear(64 + 192 + 32, 128),
0137:             nn.ReLU(),
0138:             nn.Dropout(0.2),
0139:             nn.Linear(128, 96),
0140:             nn.ReLU(),
0141:         )
0142:         self.class_head = nn.Linear(96, num_classes)
0143:         self.gate_head = nn.Linear(96, 1)
0144: 
0145:     def forward(self, x_ctx, x_proto):
0146:         b, w, l = x_ctx.shape
0147:         x = x_ctx.reshape(b * w, 1, l)
0148:         beat = self.beat_encoder(x).squeeze(-1).reshape(b, w, 64)
0149:         morph = beat[:, w // 2, :]
0150:         ctx, _ = self.context_lstm(beat)
0151:         ctx = ctx[:, w // 2, :]
0152:         proto = self.proto_mlp(x_proto)
0153:         z = self.fusion(torch.cat([morph, ctx, proto], dim=1))
0154:         return self.class_head(z), self.gate_head(z).squeeze(1)
0155: 
0156: 
0157: def main(args):
0158:     pre = Path(args.preprocessed_dir)
0159:     out = Path(args.out_dir)
0160:     out.mkdir(parents=True, exist_ok=True)
0161: 
0162:     ctx = np.load(pre / "context_windows.npz")
0163:     X = ctx["X"].astype(np.float32)
0164:     y = ctx["y"].astype(np.int64) if "y" in ctx else np.full((len(X),), -999, dtype=np.int64)
0165: 
0166:     beat_meta = pd.read_csv(pre / "beat_metadata.csv")
0167:     raw_class_values = sorted(set([int(v) for v in y.tolist() if int(v) >= 0]))
0168:     remap = {i: raw_class_values[i] for i in range(len(raw_class_values))}
0169:     if not remap:
0170:         # fallback if y unavailable
0171:         remap = {0: 0}
0172: 
0173:     proto_waves, proto_labels = load_prototypes(
0174:         Path(args.prototype_dir),
0175:         min_size=args.min_proto_size,
0176:         min_purity=args.min_proto_purity,
0177:     )
0178:     center = X[:, X.shape[1] // 2, :]
0179:     Xp, max_proto_sim = cosine_proto_features(center, proto_waves, proto_labels, args.top_k_sim)
0180: 
0181:     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
0182:     model = ECGHybridNet(num_classes=len(remap), proto_dim=Xp.shape[1]).to(device)
0183:     state = torch.load(args.model_ckpt, map_location=device)
0184:     model.load_state_dict(state)
0185:     model.eval()
0186: 
0187:     xb = torch.tensor(X, dtype=torch.float32, device=device)
0188:     xp = torch.tensor(Xp, dtype=torch.float32, device=device)
0189:     with torch.no_grad():
0190:         cls_logits, gate_logit = model(xb, xp)
0191:         cls_prob = torch.softmax(cls_logits, dim=1).cpu().numpy()
0192:         cls_idx = np.argmax(cls_prob, axis=1)
0193:         cls_conf = np.max(cls_prob, axis=1)
0194:         gate_prob = torch.sigmoid(gate_logit).cpu().numpy()
0195: 
0196:     pred_raw = np.array([remap.get(int(i), -999) for i in cls_idx], dtype=np.int64)
0197: 
0198:     # do-not-force label rule
0199:     review = (
0200:         (cls_conf < args.min_class_conf)
0201:         | (gate_prob > args.max_noise_prob)
0202:         | (max_proto_sim < args.min_proto_sim)
0203:     )
0204:     final_label = pred_raw.copy()
0205:     final_label[review] = -99  # unknown/review
0206: 
0207:     pred_df = beat_meta.copy()
0208:     n = min(len(pred_df), len(final_label))
0209:     pred_df = pred_df.iloc[:n].copy()
0210:     pred_df["pred_label_raw"] = pred_raw[:n]
0211:     pred_df["pred_label_name"] = pred_df["pred_label_raw"].map(lambda z: LABEL_NAMES.get(int(z), str(int(z))))
0212:     pred_df["pred_conf"] = cls_conf[:n]
0213:     pred_df["noise_prob"] = gate_prob[:n]
0214:     pred_df["max_proto_sim"] = max_proto_sim[:n]
0215:     pred_df["review_flag"] = review[:n].astype(np.int32)
0216:     pred_df["final_label"] = final_label[:n]
0217:     pred_df["final_label_name"] = pred_df["final_label"].map(
0218:         lambda z: "Unknown/Review" if int(z) == -99 else LABEL_NAMES.get(int(z), str(int(z)))
0219:     )
0220:     pred_df.to_csv(out / "beat_predictions_v2.csv", index=False)
0221: 
0222:     # strip-level aggregation by source_row
0223:     strip_rows = []
0224:     for sid, g in pred_df.groupby("source_row"):
0225:         usable = g[g["final_label"] >= 0]
0226:         if usable.empty:
0227:             strip_lbl = -99
0228:             strip_conf = 0.0
0229:         else:
0230:             vc = usable["final_label"].value_counts()
0231:             strip_lbl = int(vc.index[0])
0232:             strip_conf = float(vc.iloc[0] / len(g))
0233:         strip_rows.append(
0234:             {
0235:                 "source_row": int(sid),
0236:                 "strip_pred_label": strip_lbl,
0237:                 "strip_pred_name": "Unknown/Review" if strip_lbl == -99 else LABEL_NAMES.get(strip_lbl, str(strip_lbl)),
0238:                 "strip_pred_ratio": strip_conf,
0239:                 "num_beats": int(len(g)),
0240:                 "num_review": int((g["review_flag"] == 1).sum()),
0241:             }
0242:         )
0243:     strip_df = pd.DataFrame(strip_rows).sort_values("source_row")
0244:     strip_df.to_csv(out / "strip_predictions_v2.csv", index=False)
0245: 
0246:     summary = {
0247:         "num_beats_predicted": int(len(pred_df)),
0248:         "num_review_flagged": int((pred_df["review_flag"] == 1).sum()),
0249:         "review_rate": float((pred_df["review_flag"] == 1).mean()),
0250:         "num_strips": int(len(strip_df)),
0251:         "unknown_strips": int((strip_df["strip_pred_label"] == -99).sum()),
0252:         "thresholds": {
0253:             "min_class_conf": args.min_class_conf,
0254:             "max_noise_prob": args.max_noise_prob,
0255:             "min_proto_sim": args.min_proto_sim,
0256:         },
0257:     }
0258:     (out / "prediction_summary_v2.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
0259:     print(json.dumps(summary, indent=2))
0260:     print(f"Saved predictions to: {out}")
0261: 
0262: 
0263: def parse_args():
0264:     p = argparse.ArgumentParser(description="Predict with hybrid V2 + unknown/review gating")
0265:     p.add_argument("--model_ckpt", required=True)
0266:     p.add_argument("--preprocessed_dir", required=True)
0267:     p.add_argument("--prototype_dir", required=True)
0268:     p.add_argument("--out_dir", required=True)
0269:     p.add_argument("--min_proto_size", type=int, default=120)
0270:     p.add_argument("--min_proto_purity", type=float, default=0.70)
0271:     p.add_argument("--top_k_sim", type=int, default=8)
0272:     p.add_argument("--min_class_conf", type=float, default=0.60)
0273:     p.add_argument("--max_noise_prob", type=float, default=0.55)
0274:     p.add_argument("--min_proto_sim", type=float, default=0.20)
0275:     return p.parse_args()
0276: 
0277: 
0278: if __name__ == "__main__":
0279:     main(parse_args())
0280: 
```

### File: `prototype_validation_report.py`

```python
0001: """
0002: Prototype validation report (Colab-ready).
0003: 
0004: BLOCK 1:
0005:     !pip install -q numpy pandas matplotlib scipy
0006: 
0007: BLOCK 2:
0008:     !python prototype_validation_report.py \
0009:       --input_dir /content/ecg_preprocessed \
0010:       --out_dir /content/ecg_validation
0011: """
0012: 
0013: from __future__ import annotations
0014: 
0015: import argparse
0016: import json
0017: from pathlib import Path
0018: from typing import Dict, List
0019: 
0020: import matplotlib.pyplot as plt
0021: import numpy as np
0022: import pandas as pd
0023: 
0024: LABEL_NAMES = {
0025:     -1: "Noise",
0026:     0: "Normal",
0027:     1: "PAC",
0028:     2: "PVC",
0029:     3: "Sinus Bradycardia",
0030:     4: "Sinus Tachycardia",
0031:     5: "AFib",
0032:     6: "Atrial Flutter",
0033:     7: "VTach",
0034:     8: "VFib",
0035:     9: "Ventricular Ectopic",
0036:     10: "Couplets",
0037:     11: "Triplets",
0038:     12: "PVC Subtypes",
0039: }
0040: 
0041: 
0042: def parse_mix(mix: str) -> List[tuple]:
0043:     if not isinstance(mix, str) or ":" not in mix:
0044:         return []
0045:     out = []
0046:     for p in mix.split(";"):
0047:         if ":" not in p:
0048:             continue
0049:         a, b = p.split(":", 1)
0050:         try:
0051:             out.append((int(a), int(b)))
0052:         except ValueError:
0053:             pass
0054:     return out
0055: 
0056: 
0057: def purity(mix: str) -> float:
0058:     pairs = parse_mix(mix)
0059:     if not pairs:
0060:         return 0.0
0061:     counts = [c for _, c in pairs]
0062:     return max(counts) / max(1, sum(counts))
0063: 
0064: 
0065: def noise_ratio(mix: str) -> float:
0066:     pairs = parse_mix(mix)
0067:     total = sum(c for _, c in pairs)
0068:     n = sum(c for lbl, c in pairs if lbl < 0)
0069:     return n / max(1, total)
0070: 
0071: 
0072: def flag_suspicious(row, min_size: int, min_purity: float, max_noise_ratio: float) -> bool:
0073:     if int(row["size"]) < min_size:
0074:         return True
0075:     if float(row["purity"]) < min_purity:
0076:         return True
0077:     if float(row["noise_ratio"]) > max_noise_ratio:
0078:         return True
0079:     return False
0080: 
0081: 
0082: def save_label_galleries(df: pd.DataFrame, waves: np.lib.npyio.NpzFile, out_dir: Path, max_per_label: int = 12) -> None:
0083:     for lbl in sorted(df["dominant_label"].unique().tolist()):
0084:         sub = df[df["dominant_label"] == lbl].sort_values("size", ascending=False).head(max_per_label)
0085:         if sub.empty:
0086:             continue
0087:         cols = 4
0088:         rows = int(np.ceil(len(sub) / cols))
0089:         plt.figure(figsize=(4.2 * cols, 2.4 * rows))
0090:         for i, (_, r) in enumerate(sub.iterrows(), 1):
0091:             pid = int(r["prototype_id"])
0092:             key = f"prototype_{pid}"
0093:             if key not in waves.files:
0094:                 continue
0095:             ax = plt.subplot(rows, cols, i)
0096:             ax.plot(waves[key], linewidth=1.4)
0097:             ax.set_title(f"P{pid} n={int(r['size'])} p={float(r['purity']):.2f}", fontsize=8)
0098:             ax.set_xticks([])
0099:             ax.set_yticks([])
0100:         name = LABEL_NAMES.get(int(lbl), str(int(lbl)))
0101:         plt.suptitle(f"Label {lbl} - {name}", fontsize=11)
0102:         plt.tight_layout()
0103:         plt.savefig(out_dir / f"label_{lbl}_gallery.png", dpi=170)
0104:         plt.close()
0105: 
0106: 
0107: def main(args):
0108:     inp = Path(args.input_dir)
0109:     out = Path(args.out_dir)
0110:     out.mkdir(parents=True, exist_ok=True)
0111: 
0112:     csv_path = inp / "prototype_library.csv"
0113:     npz_path = inp / "prototype_library_waves.npz"
0114:     if not csv_path.exists() or not npz_path.exists():
0115:         raise RuntimeError("Missing prototype_library.csv or prototype_library_waves.npz in input_dir.")
0116: 
0117:     df = pd.read_csv(csv_path)
0118:     waves = np.load(npz_path)
0119:     if df.empty:
0120:         raise RuntimeError("prototype_library.csv is empty.")
0121: 
0122:     df["purity"] = df["label_mix_top3"].apply(purity)
0123:     df["noise_ratio"] = df["label_mix_top3"].apply(noise_ratio)
0124:     df["dominant_label_name"] = df["dominant_label"].map(lambda x: LABEL_NAMES.get(int(x), str(int(x))))
0125:     df["suspicious"] = df.apply(
0126:         lambda r: flag_suspicious(r, args.min_size, args.min_purity, args.max_noise_ratio),
0127:         axis=1,
0128:     )
0129: 
0130:     # per-label summary
0131:     rows = []
0132:     for lbl, g in df.groupby("dominant_label"):
0133:         rows.append(
0134:             {
0135:                 "label": int(lbl),
0136:                 "label_name": LABEL_NAMES.get(int(lbl), str(int(lbl))),
0137:                 "num_prototypes": int(len(g)),
0138:                 "total_members": int(g["size"].sum()),
0139:                 "mean_purity": float(g["purity"].mean()),
0140:                 "num_suspicious": int(g["suspicious"].sum()),
0141:             }
0142:         )
0143:     label_summary = pd.DataFrame(rows).sort_values("total_members", ascending=False)
0144: 
0145:     # save tables
0146:     df.sort_values(["suspicious", "size"], ascending=[False, False]).to_csv(out / "prototype_validation_detail.csv", index=False)
0147:     label_summary.to_csv(out / "prototype_validation_by_label.csv", index=False)
0148:     df[df["suspicious"]].to_csv(out / "prototype_suspicious.csv", index=False)
0149: 
0150:     # plot overall purity vs size
0151:     plt.figure(figsize=(8, 5))
0152:     c = np.where(df["suspicious"].values, "red", "tab:blue")
0153:     plt.scatter(df["size"], df["purity"], c=c, alpha=0.75, s=35)
0154:     plt.xscale("log")
0155:     plt.xlabel("Prototype Size (log scale)")
0156:     plt.ylabel("Purity")
0157:     plt.title("Prototype Purity vs Size")
0158:     plt.grid(alpha=0.25)
0159:     plt.tight_layout()
0160:     plt.savefig(out / "prototype_purity_vs_size.png", dpi=170)
0161:     plt.close()
0162: 
0163:     save_label_galleries(df, waves, out, max_per_label=args.max_per_label)
0164: 
0165:     summary = {
0166:         "num_prototypes": int(len(df)),
0167:         "num_suspicious": int(df["suspicious"].sum()),
0168:         "labels_seen": sorted(df["dominant_label"].astype(int).unique().tolist()),
0169:         "thresholds": {
0170:             "min_size": int(args.min_size),
0171:             "min_purity": float(args.min_purity),
0172:             "max_noise_ratio": float(args.max_noise_ratio),
0173:         },
0174:     }
0175:     (out / "prototype_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
0176:     print(json.dumps(summary, indent=2))
0177:     print(f"Saved validation report to: {out}")
0178: 
0179: 
0180: def parse_args():
0181:     p = argparse.ArgumentParser(description="Validate prototype-label mapping quality")
0182:     p.add_argument("--input_dir", required=True, help="Preprocess output dir containing prototype files")
0183:     p.add_argument("--out_dir", required=True, help="Output report dir")
0184:     p.add_argument("--min_size", type=int, default=80)
0185:     p.add_argument("--min_purity", type=float, default=0.70)
0186:     p.add_argument("--max_noise_ratio", type=float, default=0.20)
0187:     p.add_argument("--max_per_label", type=int, default=12)
0188:     return p.parse_args()
0189: 
0190: 
0191: if __name__ == "__main__":
0192:     main(parse_args())
0193: 
```

### File: `new_pattern_alerts.py`

```python
0001: """
0002: Discover and alert on frequent unknown/new ECG patterns.
0003: 
0004: BLOCK 1:
0005:     !pip install -q numpy pandas scikit-learn matplotlib
0006: 
0007: BLOCK 2:
0008:     !python new_pattern_alerts.py \
0009:       --preprocessed_dir /content/ecg_preprocessed \
0010:       --predictions_csv /content/ecg_predictions_v2/beat_predictions_v2.csv \
0011:       --out_dir /content/ecg_new_pattern_alerts
0012: """
0013: 
0014: from __future__ import annotations
0015: 
0016: import argparse
0017: import json
0018: from pathlib import Path
0019: 
0020: import matplotlib.pyplot as plt
0021: import numpy as np
0022: import pandas as pd
0023: from sklearn.cluster import KMeans
0024: from sklearn.decomposition import PCA
0025: 
0026: 
0027: def main(args):
0028:     pre = Path(args.preprocessed_dir)
0029:     out = Path(args.out_dir)
0030:     out.mkdir(parents=True, exist_ok=True)
0031: 
0032:     beats = np.load(pre / "beats.npy")
0033:     feats = np.load(pre / "features.npy") if (pre / "features.npy").exists() else None
0034:     if feats is None:
0035:         raise RuntimeError("features.npy not found in preprocessed_dir.")
0036: 
0037:     pred = pd.read_csv(args.predictions_csv)
0038:     if "review_flag" not in pred.columns:
0039:         raise RuntimeError("predictions_csv must contain review_flag column from predict_v2.py")
0040: 
0041:     unknown_idx = np.where(pred["review_flag"].values == 1)[0]
0042:     if unknown_idx.size == 0:
0043:         summary = {"unknown_count": 0, "alerts": 0}
0044:         (out / "new_pattern_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
0045:         print(json.dumps(summary, indent=2))
0046:         return
0047: 
0048:     u_feat = feats[unknown_idx]
0049:     u_beat = beats[unknown_idx]
0050: 
0051:     k = max(2, min(args.max_clusters, int(np.sqrt(len(unknown_idx) / max(1, args.cluster_scale)))))
0052:     model = KMeans(n_clusters=k, random_state=42, n_init=20)
0053:     cid = model.fit_predict(u_feat)
0054: 
0055:     rows = []
0056:     for c in sorted(np.unique(cid).tolist()):
0057:         idx_local = np.where(cid == c)[0]
0058:         idx_global = unknown_idx[idx_local]
0059:         size = len(idx_local)
0060:         centroid = np.median(u_beat[idx_local], axis=0)
0061:         np.save(out / f"cluster_{c}_centroid.npy", centroid.astype(np.float32))
0062:         rows.append(
0063:             {
0064:                 "cluster_id": int(c),
0065:                 "size": int(size),
0066:                 "size_ratio": float(size / max(1, len(unknown_idx))),
0067:                 "example_beat_ids": ",".join(map(str, idx_global[:15].tolist())),
0068:                 "alert_flag": int(size >= args.min_alert_size),
0069:             }
0070:         )
0071: 
0072:     rep = pd.DataFrame(rows).sort_values("size", ascending=False)
0073:     rep.to_csv(out / "new_pattern_clusters.csv", index=False)
0074:     rep[rep["alert_flag"] == 1].to_csv(out / "new_pattern_alerts.csv", index=False)
0075: 
0076:     # PCA plot
0077:     p2 = PCA(n_components=2, random_state=42).fit_transform(u_feat)
0078:     plt.figure(figsize=(8, 6))
0079:     plt.scatter(p2[:, 0], p2[:, 1], c=cid, cmap="tab20", s=10, alpha=0.75)
0080:     plt.title("Unknown/Review Beat Clusters")
0081:     plt.xlabel("PC1")
0082:     plt.ylabel("PC2")
0083:     plt.tight_layout()
0084:     plt.savefig(out / "unknown_clusters_pca2d.png", dpi=170)
0085:     plt.close()
0086: 
0087:     # centroid gallery
0088:     show = rep.head(min(args.max_plot_clusters, len(rep)))
0089:     cols = 4
0090:     rows_n = int(np.ceil(len(show) / cols))
0091:     plt.figure(figsize=(4 * cols, 2.5 * rows_n))
0092:     for i, (_, r) in enumerate(show.iterrows(), 1):
0093:         c = int(r["cluster_id"])
0094:         w = np.load(out / f"cluster_{c}_centroid.npy")
0095:         ax = plt.subplot(rows_n, cols, i)
0096:         ax.plot(w, linewidth=1.4)
0097:         ax.set_title(f"C{c} n={int(r['size'])}", fontsize=8)
0098:         ax.set_xticks([])
0099:         ax.set_yticks([])
0100:     plt.tight_layout()
0101:     plt.savefig(out / "unknown_cluster_centroids.png", dpi=170)
0102:     plt.close()
0103: 
0104:     summary = {
0105:         "unknown_count": int(len(unknown_idx)),
0106:         "clusters_found": int(len(rep)),
0107:         "alerts": int((rep["alert_flag"] == 1).sum()),
0108:         "min_alert_size": int(args.min_alert_size),
0109:     }
0110:     (out / "new_pattern_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
0111:     print(json.dumps(summary, indent=2))
0112:     print(f"Saved alerts to: {out}")
0113: 
0114: 
0115: def parse_args():
0116:     p = argparse.ArgumentParser(description="Detect frequent unknown/new ECG patterns")
0117:     p.add_argument("--preprocessed_dir", required=True)
0118:     p.add_argument("--predictions_csv", required=True, help="beat_predictions_v2.csv from predict_v2.py")
0119:     p.add_argument("--out_dir", required=True)
0120:     p.add_argument("--min_alert_size", type=int, default=80)
0121:     p.add_argument("--max_clusters", type=int, default=20)
0122:     p.add_argument("--cluster_scale", type=int, default=40)
0123:     p.add_argument("--max_plot_clusters", type=int, default=16)
0124:     return p.parse_args()
0125: 
0126: 
0127: if __name__ == "__main__":
0128:     main(parse_args())
0129: 
```
