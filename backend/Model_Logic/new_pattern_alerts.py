"""
Discover and alert on frequent unknown/new ECG patterns.

BLOCK 1:
    !pip install -q numpy pandas scikit-learn matplotlib

BLOCK 2:
    !python new_pattern_alerts.py \
      --preprocessed_dir /content/ecg_preprocessed \
      --predictions_csv /content/ecg_predictions_v2/beat_predictions_v2.csv \
      --out_dir /content/ecg_new_pattern_alerts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def main(args):
    pre = Path(args.preprocessed_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    beats = np.load(pre / "beats.npy")
    feats = np.load(pre / "features.npy") if (pre / "features.npy").exists() else None
    if feats is None:
        raise RuntimeError("features.npy not found in preprocessed_dir.")

    pred = pd.read_csv(args.predictions_csv)
    if "review_flag" not in pred.columns:
        raise RuntimeError("predictions_csv must contain review_flag column from predict_v2.py")

    unknown_idx = np.where(pred["review_flag"].values == 1)[0]
    if unknown_idx.size == 0:
        summary = {"unknown_count": 0, "alerts": 0}
        (out / "new_pattern_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return

    u_feat = feats[unknown_idx]
    u_beat = beats[unknown_idx]

    k = max(2, min(args.max_clusters, int(np.sqrt(len(unknown_idx) / max(1, args.cluster_scale)))))
    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    cid = model.fit_predict(u_feat)

    rows = []
    for c in sorted(np.unique(cid).tolist()):
        idx_local = np.where(cid == c)[0]
        idx_global = unknown_idx[idx_local]
        size = len(idx_local)
        centroid = np.median(u_beat[idx_local], axis=0)
        np.save(out / f"cluster_{c}_centroid.npy", centroid.astype(np.float32))
        rows.append(
            {
                "cluster_id": int(c),
                "size": int(size),
                "size_ratio": float(size / max(1, len(unknown_idx))),
                "example_beat_ids": ",".join(map(str, idx_global[:15].tolist())),
                "alert_flag": int(size >= args.min_alert_size),
            }
        )

    rep = pd.DataFrame(rows).sort_values("size", ascending=False)
    rep.to_csv(out / "new_pattern_clusters.csv", index=False)
    rep[rep["alert_flag"] == 1].to_csv(out / "new_pattern_alerts.csv", index=False)

    # PCA plot
    p2 = PCA(n_components=2, random_state=42).fit_transform(u_feat)
    plt.figure(figsize=(8, 6))
    plt.scatter(p2[:, 0], p2[:, 1], c=cid, cmap="tab20", s=10, alpha=0.75)
    plt.title("Unknown/Review Beat Clusters")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(out / "unknown_clusters_pca2d.png", dpi=170)
    plt.close()

    # centroid gallery
    show = rep.head(min(args.max_plot_clusters, len(rep)))
    cols = 4
    rows_n = int(np.ceil(len(show) / cols))
    plt.figure(figsize=(4 * cols, 2.5 * rows_n))
    for i, (_, r) in enumerate(show.iterrows(), 1):
        c = int(r["cluster_id"])
        w = np.load(out / f"cluster_{c}_centroid.npy")
        ax = plt.subplot(rows_n, cols, i)
        ax.plot(w, linewidth=1.4)
        ax.set_title(f"C{c} n={int(r['size'])}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out / "unknown_cluster_centroids.png", dpi=170)
    plt.close()

    summary = {
        "unknown_count": int(len(unknown_idx)),
        "clusters_found": int(len(rep)),
        "alerts": int((rep["alert_flag"] == 1).sum()),
        "min_alert_size": int(args.min_alert_size),
    }
    (out / "new_pattern_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved alerts to: {out}")


def parse_args():
    p = argparse.ArgumentParser(description="Detect frequent unknown/new ECG patterns")
    p.add_argument("--preprocessed_dir", required=True)
    p.add_argument("--predictions_csv", required=True, help="beat_predictions_v2.csv from predict_v2.py")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--min_alert_size", type=int, default=80)
    p.add_argument("--max_clusters", type=int, default=20)
    p.add_argument("--cluster_scale", type=int, default=40)
    p.add_argument("--max_plot_clusters", type=int, default=16)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())

