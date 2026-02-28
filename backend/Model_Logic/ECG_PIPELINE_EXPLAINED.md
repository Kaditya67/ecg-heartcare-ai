# ECG Pipeline Explained (Line-by-Line Style)

This file explains what each major line/block in your latest pipeline does.

## 1) `colab_ecg_preprocess.py`

### Header + imports
- Docstring blocks: tells how to run in Colab.
- `numpy/pandas/scipy/sklearn`: used for parsing, filtering, clustering, and feature building.
- `neurokit2` optional import: preferred R-peak detector if available.

### `DEFAULT_LABEL_NAMES`
- Maps numeric label values to readable ECG class names.

### Core utility functions
- `parse_wave(...)`: converts comma-separated ECG string into numeric array.
- `infer_fs(...)`: infers sampling rate from wave length and strip duration.
- `bandpass_ecg(...)`: removes low/high-frequency noise (ECG-relevant band).
- `normalize_wave(...)`: standardizes each strip for model stability.
- `detect_rpeaks(...)`: finds R-peaks (first with NeuroKit, fallback custom logic).
- `extract_beats(...)`: cuts each beat around R-peak (`pre_ms`, `post_ms`).
- `score_noise(...)`: computes beat quality and flags noisy/outlier beats.

### Grouping + templates
- `dominant_cluster_ids(...)`: local clustering within class to find outliers.
- `build_global_feature_space(...)`: creates label-free feature map for beats.
- `cluster_global_beats(...)`: global clustering without labels.
- `build_templates(...)`: one median template per class.
- `build_prototype_library(...)`: creates key-value prototype map:
  - key: `prototype_id`
  - value: median waveform + metadata.

### Context windows
- `make_context_windows(...)`: creates consecutive beat windows (e.g., 5 beats) for rhythm-level learning.

### `preprocess(args)`
- Loads Excel and resolves columns.
- Loops records: filter -> detect R -> extract beats -> RR intervals.
- Builds arrays: beats, labels, source rows, RR features.
- Flags noise + local and global clusters.
- Saves outputs:
  - `beats.npy`, `labels.npy`, `noise_flags.npy`
  - `context_windows.npz`
  - `beat_metadata.csv`
  - `prototype_library.csv`, `prototype_library_waves.npz`
  - plots (`class_templates.png`, `prototype_gallery.png`, cluster plot)
  - `summary.json`

### `__main__`
- Works in both modes:
  - CLI args mode
  - Notebook default Args mode

---

## 2) `colab_ecg_train_v2.py`

### Goal
Hybrid training with:
1. Beat morphology branch
2. Consecutive-context branch
3. Prototype similarity branch
4. Noise gate output

### Data preparation
- `load_prototype_bank(...)`: reads prototype waveforms and keeps only good ones (`min_proto_size`, `min_proto_purity`).
- `cosine_proto_features(...)`: computes similarity features between each beat and prototype bank.
- `load_training_data(...)`: loads context windows and builds:
  - `X` context input
  - `X_proto` prototype-similarity input
  - `y_class` class label target (`-1` excluded from class head)
  - `y_gate` noise/non-noise target

### Model
- `ECGHybridNet`:
  - `beat_encoder`: CNN per beat morphology.
  - `context_lstm`: sequence modeling of neighboring beats.
  - `proto_mlp`: transforms prototype similarity features.
  - `fusion`: combines all three features.
  - `class_head`: class prediction logits.
  - `gate_head`: noise/review logit.

### Training loop
- Uses class-weighted CE (if enabled) to reduce minority collapse.
- Uses BCE loss for gate head.
- Total loss = class loss + `gate_loss_weight * gate_loss`.
- Tracks train/val class accuracy and gate accuracy each epoch.
- Saves best checkpoint.

### Outputs
- `best_hybrid_v2.pt`
- `training_summary_v2.json` with:
  - best score
  - confusion matrix
  - per-class report
  - prototype usage info

---

## 3) `predict_v2.py`

### Purpose
Inference with **do-not-force labeling**.

### Steps
- Loads trained V2 model checkpoint.
- Loads context windows and prototype bank.
- Rebuilds prototype similarity features.
- Runs model to get:
  - class probabilities
  - noise probability
- Applies review/unknown logic:
  - low class confidence OR high noise probability OR low prototype similarity => `Unknown/Review`.

### Outputs
- `beat_predictions_v2.csv`
- `strip_predictions_v2.csv` (aggregated by source strip)
- `prediction_summary_v2.json`

---

## 4) `prototype_validation_report.py`

### Purpose
Verify if prototype-label mapping is clinically consistent.

### What it computes
- purity per prototype from `label_mix_top3`
- noise ratio per prototype
- suspicious flags based on thresholds:
  - small cluster size
  - low purity
  - high noise ratio

### Outputs
- `prototype_validation_detail.csv`
- `prototype_validation_by_label.csv`
- `prototype_suspicious.csv`
- `prototype_purity_vs_size.png`
- per-label galleries (e.g., `label_2_gallery.png`)

---

## 5) `new_pattern_alerts.py`

### Purpose
Find frequent unknown/review patterns (possible new rhythm classes).

### Steps
- Loads review-flagged beats from `beat_predictions_v2.csv`.
- Clusters unknown beats.
- Builds cluster centroids.
- Flags clusters above alert size threshold.

### Outputs
- `new_pattern_clusters.csv`
- `new_pattern_alerts.csv`
- `unknown_clusters_pca2d.png`
- `unknown_cluster_centroids.png`
- `new_pattern_summary.json`

---

## End-to-end flow (recommended run order)
1. `colab_ecg_preprocess.py`
2. `colab_ecg_train_v2.py`
3. `prototype_validation_report.py`
4. `predict_v2.py`
5. `new_pattern_alerts.py`

This sequence ensures: preprocess -> train -> validate mapping -> predict safely -> discover new patterns.

