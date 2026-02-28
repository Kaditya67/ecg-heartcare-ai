"""
Prototype validation report (Colab-ready).

BLOCK 1:
    !pip install -q numpy pandas matplotlib scipy

BLOCK 2:
    !python prototype_validation_report.py \
      --input_dir /content/ecg_preprocessed \
      --out_dir /content/ecg_validation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def parse_mix(mix: str) -> List[tuple]:
    if not isinstance(mix, str) or ":" not in mix:
        return []
    out = []
    for p in mix.split(";"):
        if ":" not in p:
            continue
        a, b = p.split(":", 1)
        try:
            out.append((int(a), int(b)))
        except ValueError:
            pass
    return out


def purity(mix: str) -> float:
    pairs = parse_mix(mix)
    if not pairs:
        return 0.0
    counts = [c for _, c in pairs]
    return max(counts) / max(1, sum(counts))


def noise_ratio(mix: str) -> float:
    pairs = parse_mix(mix)
    total = sum(c for _, c in pairs)
    n = sum(c for lbl, c in pairs if lbl < 0)
    return n / max(1, total)


def flag_suspicious(row, min_size: int, min_purity: float, max_noise_ratio: float) -> bool:
    if int(row["size"]) < min_size:
        return True
    if float(row["purity"]) < min_purity:
        return True
    if float(row["noise_ratio"]) > max_noise_ratio:
        return True
    return False


def save_label_galleries(df: pd.DataFrame, waves: np.lib.npyio.NpzFile, out_dir: Path, max_per_label: int = 12) -> None:
    for lbl in sorted(df["dominant_label"].unique().tolist()):
        sub = df[df["dominant_label"] == lbl].sort_values("size", ascending=False).head(max_per_label)
        if sub.empty:
            continue
        cols = 4
        rows = int(np.ceil(len(sub) / cols))
        plt.figure(figsize=(4.2 * cols, 2.4 * rows))
        for i, (_, r) in enumerate(sub.iterrows(), 1):
            pid = int(r["prototype_id"])
            key = f"prototype_{pid}"
            if key not in waves.files:
                continue
            ax = plt.subplot(rows, cols, i)
            ax.plot(waves[key], linewidth=1.4)
            ax.set_title(f"P{pid} n={int(r['size'])} p={float(r['purity']):.2f}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        name = LABEL_NAMES.get(int(lbl), str(int(lbl)))
        plt.suptitle(f"Label {lbl} - {name}", fontsize=11)
        plt.tight_layout()
        plt.savefig(out_dir / f"label_{lbl}_gallery.png", dpi=170)
        plt.close()


def main(args):
    inp = Path(args.input_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    csv_path = inp / "prototype_library.csv"
    npz_path = inp / "prototype_library_waves.npz"
    if not csv_path.exists() or not npz_path.exists():
        raise RuntimeError("Missing prototype_library.csv or prototype_library_waves.npz in input_dir.")

    df = pd.read_csv(csv_path)
    waves = np.load(npz_path)
    if df.empty:
        raise RuntimeError("prototype_library.csv is empty.")

    df["purity"] = df["label_mix_top3"].apply(purity)
    df["noise_ratio"] = df["label_mix_top3"].apply(noise_ratio)
    df["dominant_label_name"] = df["dominant_label"].map(lambda x: LABEL_NAMES.get(int(x), str(int(x))))
    df["suspicious"] = df.apply(
        lambda r: flag_suspicious(r, args.min_size, args.min_purity, args.max_noise_ratio),
        axis=1,
    )

    # per-label summary
    rows = []
    for lbl, g in df.groupby("dominant_label"):
        rows.append(
            {
                "label": int(lbl),
                "label_name": LABEL_NAMES.get(int(lbl), str(int(lbl))),
                "num_prototypes": int(len(g)),
                "total_members": int(g["size"].sum()),
                "mean_purity": float(g["purity"].mean()),
                "num_suspicious": int(g["suspicious"].sum()),
            }
        )
    label_summary = pd.DataFrame(rows).sort_values("total_members", ascending=False)

    # save tables
    df.sort_values(["suspicious", "size"], ascending=[False, False]).to_csv(out / "prototype_validation_detail.csv", index=False)
    label_summary.to_csv(out / "prototype_validation_by_label.csv", index=False)
    df[df["suspicious"]].to_csv(out / "prototype_suspicious.csv", index=False)

    # plot overall purity vs size
    plt.figure(figsize=(8, 5))
    c = np.where(df["suspicious"].values, "red", "tab:blue")
    plt.scatter(df["size"], df["purity"], c=c, alpha=0.75, s=35)
    plt.xscale("log")
    plt.xlabel("Prototype Size (log scale)")
    plt.ylabel("Purity")
    plt.title("Prototype Purity vs Size")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out / "prototype_purity_vs_size.png", dpi=170)
    plt.close()

    save_label_galleries(df, waves, out, max_per_label=args.max_per_label)

    summary = {
        "num_prototypes": int(len(df)),
        "num_suspicious": int(df["suspicious"].sum()),
        "labels_seen": sorted(df["dominant_label"].astype(int).unique().tolist()),
        "thresholds": {
            "min_size": int(args.min_size),
            "min_purity": float(args.min_purity),
            "max_noise_ratio": float(args.max_noise_ratio),
        },
    }
    (out / "prototype_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved validation report to: {out}")


def parse_args():
    p = argparse.ArgumentParser(description="Validate prototype-label mapping quality")
    p.add_argument("--input_dir", required=True, help="Preprocess output dir containing prototype files")
    p.add_argument("--out_dir", required=True, help="Output report dir")
    p.add_argument("--min_size", type=int, default=80)
    p.add_argument("--min_purity", type=float, default=0.70)
    p.add_argument("--max_noise_ratio", type=float, default=0.20)
    p.add_argument("--max_per_label", type=int, default=12)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())

