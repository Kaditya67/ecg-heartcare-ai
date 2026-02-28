# ECG Method Justification

This file explains what techniques are used in the current ECG pipeline and why each one was selected.

## 1) Used Models/Components

### Preprocessing / Feature Stage
- Bandpass filtering (`0.5–45 Hz`)
- R-peak detection (`neurokit2` + fallback detector)
- Beat segmentation around R-peaks
- Noise scoring (RR + morphology)
- Local clustering + global label-free clustering
- Prototype library (key-value waveform map)
- Context window creation (consecutive beats)

### Training Models
- V1: `ECGContextNet` (CNN per beat + BiLSTM over window)
- V2: `ECGHybridNet` with 3 branches:
  - morphology branch (CNN)
  - context branch (BiLSTM)
  - prototype-similarity branch (MLP)
  - plus noise gate head

### Losses used in V2
- CrossEntropy for class head
- BCEWithLogits for noise gate
- class-weighted CE (latest run)

### Inference
- `predict_v2.py`:
  - class probabilities
  - noise probability
  - prototype similarity
  - do-not-force rule -> `Unknown/Review` when uncertain
  - beat-level and strip-level outputs

### Validation / Monitoring
- `prototype_validation_report.py` (purity, suspicious prototypes, per-label galleries)
- `new_pattern_alerts.py` (clusters frequent unknown patterns, raises alerts)

---

## 2) Why Each Technique Was Chosen

## 2.1 Bandpass Filtering (0.5–45 Hz)
Why:
- ECG information of interest is mostly in this range.
- Removes baseline wander and high-frequency noise before model sees data.

Improvement:
- Better signal-to-noise ratio.
- Reduces overfitting to artifacts and sensor drift.

## 2.2 R-Peak Detection + Beat-Centered Segmentation
Why:
- Cardiac morphology is best compared when beats are aligned at R-peak.
- Full-strip training causes position dependence (same pathology at different time offsets).

Improvement:
- Model learns morphology, not absolute strip position.
- Better transfer across patients and recordings.

## 2.3 Noise Scoring (RR + Morphology)
Why:
- Some extracted beats are artifacts or physically implausible.
- RR interval sanity and morphology checks catch many bad beats automatically.

Improvement:
- Cleaner training data.
- Lower label pollution from noisy segments.

## 2.4 Local + Global Clustering
Why:
- Local clustering helps remove tiny outlier groups within labels.
- Global clustering discovers morphology families independent of labels.

Improvement:
- Better curation quality.
- Enables discovery of hidden subtypes and potential mislabeled regions.

## 2.5 Prototype Library (Key-Value Waveform Map)
Why:
- Needed for explainability and retrieval.
- Makes it possible to compare new beats to known representative patterns.

Improvement:
- Human-verifiable mapping between waveform families and assigned labels.
- Supports prototype-based confidence and unknown detection.

## 2.6 Context Windows (Consecutive Beats)
Why:
- Many abnormalities are rhythm patterns, not single-beat shape events.
- Need neighboring beats for robust rhythm interpretation.

Improvement:
- Captures temporal structure like repeated ectopy or spacing irregularity.
- Improves classification of sequence-dependent conditions.

---

## 3) Why V1 -> V2 Upgrade Was Necessary

## 3.1 V1 Strength
- Learned beat morphology + short context reasonably well.

## 3.2 V1 Gap
- No explicit prototype similarity path.
- Weaker reject-option behavior for uncertain/novel patterns.
- Minority behavior could collapse without weighting.

## 3.3 V2 Design Response
- Adds prototype branch (MLP on similarity features).
- Adds noise gate head.
- Supports class-weighted training for minority classes.

Expected improvement:
- Better balance of morphology + sequence + known-pattern similarity.
- Better handling of uncertainty and new patterns.

---

## 4) Why These Loss Functions

## 4.1 CrossEntropy for Class Head
Why:
- Standard, stable multiclass objective for class logits.

## 4.2 BCEWithLogits for Gate Head
Why:
- Gate is binary (noise/review tendency vs clean).
- BCEWithLogits is numerically stable for binary logits.

## 4.3 Class-Weighted CE
Why:
- Dataset has class imbalance.
- Without weighting, rare class recall can collapse.

Tradeoff:
- Can increase false positives for minority class if too strong.
- Needs threshold tuning/calibration after training.

---

## 5) Why Do-Not-Force Inference

Why:
- For medical-adjacent use, forced wrong labels are worse than review flags.
- Unknown or low-confidence beats should be escalated, not guessed.

How:
- Combine 3 checks:
  - low class confidence
  - high noise probability
  - low prototype similarity
- If any fails -> `Unknown/Review`.

Improvement:
- Reduces unsafe overconfident misclassification.
- Supports human-in-loop curation loop.

---

## 6) Why Validation + Alert Tools Were Added

## 6.1 Prototype Validation Report
Why:
- Cluster-to-label mapping can still be wrong/mixed.
- Need explicit purity and suspicious prototype tracking.

Improvement:
- Measurable mapping quality.
- Fast review of problematic prototype groups.

## 6.2 New Pattern Alerts
Why:
- Real-world incoming data may contain unseen morphology patterns.
- Need mechanism to detect recurring unknown clusters.

Improvement:
- System can evolve with data drift.
- Prioritizes high-impact annotation targets.

---

## 7) Net Result

This pipeline was chosen because it combines:
- signal quality control,
- morphology understanding,
- rhythm context modeling,
- prototype-based explainability,
- and safe uncertainty handling.

That combination is stronger than any single-method approach for your ECG use case.

