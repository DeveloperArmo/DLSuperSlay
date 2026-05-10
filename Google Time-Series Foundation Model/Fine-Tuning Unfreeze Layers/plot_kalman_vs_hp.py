"""Compare HP vs Kalman fine-tuning results across v1/v2/v3.

Reads the per-segment metrics .npz files for all six variants and writes
a 2x2 figure (RMSE / MAPE / Pearson R^2 / Directional Accuracy) plus a
small CSV of summary statistics.

Outputs:
  - kalman_vs_hp_comparison.png  (saved next to this script)
  - kalman_vs_hp_summary.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
GOOGLE_DIR = SCRIPT_DIR.parent  # "Google Time-Series Foundation Model"

# (label, decomposition family, version, npz_path)
VARIANTS = [
    ("HP v1",     "HP",     "v1", GOOGLE_DIR / "TimesFM_SSMI_FineTuned_Decomposed_Metrics.npz"),
    ("HP v2",     "HP",     "v2", SCRIPT_DIR / "TimesFM_SSMI_FineTuned_Decomposed_v2_Metrics.npz"),
    ("HP v3",     "HP",     "v3", SCRIPT_DIR / "TimesFM_SSMI_FineTuned_Decomposed_v3_Metrics.npz"),
    ("Kalman v1", "Kalman", "v1", SCRIPT_DIR / "TimesFM_SSMI_FineTuned_Decomposed_Kalman_v1_Metrics.npz"),
    ("Kalman v2", "Kalman", "v2", SCRIPT_DIR / "TimesFM_SSMI_FineTuned_Decomposed_Kalman_v2_Metrics.npz"),
    ("Kalman v3", "Kalman", "v3", SCRIPT_DIR / "TimesFM_SSMI_FineTuned_Decomposed_Kalman_v3_Metrics.npz"),
]

# Color per family (consistent within a family across versions, alpha varies by version)
FAMILY_COLORS = {"HP": "#1f77b4", "Kalman": "#d62728"}
VERSION_ALPHA = {"v1": 0.45, "v2": 0.70, "v3": 0.95}

FORECAST_HORIZON = 30  # to convert directional_hits into per-segment accuracy


def load_variant(npz_path: Path) -> dict:
    z = np.load(npz_path)
    rmse = z["rmse"]
    mape = z["mape"]
    r2 = z["pearson_coefficients"]
    dh = np.asarray(z["directional_hits"], dtype=float)

    # per-segment directional accuracy (%) for boxplot
    n_full = len(dh) // FORECAST_HORIZON
    if n_full > 0:
        per_seg_dir = dh[: n_full * FORECAST_HORIZON].reshape(n_full, FORECAST_HORIZON).mean(axis=1) * 100.0
    else:
        per_seg_dir = np.array([])

    return {
        "rmse": rmse,
        "mape": mape,
        "r2": r2,
        "dir_acc": per_seg_dir,
        "overall_dir_acc": float(dh.mean() * 100.0) if len(dh) else float("nan"),
    }


def main():
    data = {label: load_variant(path) for (label, _fam, _ver, path) in VARIANTS}

    # ----- Summary CSV -----
    rows = []
    for label, family, version, _ in VARIANTS:
        d = data[label]
        rows.append({
            "variant": label,
            "family": family,
            "version": version,
            "median_rmse": float(np.median(d["rmse"])),
            "median_mape": float(np.median(d["mape"])),
            "median_r2": float(np.median(d["r2"])),
            "directional_accuracy_pct": d["overall_dir_acc"],
            "num_segments": len(d["rmse"]),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(SCRIPT_DIR / "kalman_vs_hp_summary.csv", index=False)
    print(summary.to_string(index=False))

    # ----- Plot -----
    metrics = [
        ("rmse",    "RMSE",        False),
        ("mape",    "MAPE (%)",    False),
        ("r2",      "Pearson R²", False),
        ("dir_acc", "Directional Accuracy (%)", False),
    ]

    labels = [v[0] for v in VARIANTS]
    families = [v[1] for v in VARIANTS]
    versions = [v[2] for v in VARIANTS]
    colors = [FAMILY_COLORS[f] for f in families]
    alphas = [VERSION_ALPHA[v] for v in versions]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    for ax, (key, title, _) in zip(axes, metrics):
        plot_data = [data[lbl][key] for lbl in labels]
        bp = ax.boxplot(
            plot_data,
            labels=labels,
            showfliers=False,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.8),
        )
        for patch, color, alpha in zip(bp["boxes"], colors, alphas):
            patch.set_facecolor(color)
            patch.set_alpha(alpha)
            patch.set_edgecolor("black")

        # overlay the median value as text above each box
        for i, vals in enumerate(plot_data):
            med = float(np.median(vals)) if len(vals) else float("nan")
            y_pos = med
            ax.text(i + 1, y_pos, f"{med:.2f}", ha="center", va="bottom", fontsize=8)

        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_ylabel(title)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle(
        "HP vs Kalman fine-tuning — SSMI post-2018 test (24 segments)",
        fontsize=14, weight="bold", y=1.00,
    )

    # Legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=FAMILY_COLORS["HP"],     alpha=0.7, label="HP filter"),
        Patch(facecolor=FAMILY_COLORS["Kalman"], alpha=0.7, label="Kalman filter (q=0.1)"),
    ]
    fig.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(0.99, 0.98), frameon=True)

    plt.tight_layout()
    out_png = SCRIPT_DIR / "kalman_vs_hp_comparison.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"\nSaved figure: {out_png}")

    # ----- Also: a focused bar chart for overall directional accuracy -----
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    overall_dir = [data[lbl]["overall_dir_acc"] for lbl in labels]
    bars = ax2.bar(labels, overall_dir, color=colors, alpha=0.8, edgecolor="black")
    for bar, val in zip(bars, overall_dir):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.1,
                 f"{val:.2f}%", ha="center", va="bottom", fontsize=9)

    # Reference line for zero-shot baseline directional accuracy (from FINETUNING.md: 51.25%)
    ax2.axhline(51.25, color="gray", linestyle="--", linewidth=1.2, label="Zero-shot raw (51.25%)")
    ax2.set_title("Overall Directional Accuracy (full 720-day test)", fontsize=12, weight="bold")
    ax2.set_ylabel("Directional Accuracy (%)")
    ax2.set_ylim(45, 56)
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend(loc="lower right", fontsize=9)
    ax2.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    out_png2 = SCRIPT_DIR / "kalman_vs_hp_directional_accuracy.png"
    fig2.savefig(out_png2, dpi=200, bbox_inches="tight")
    print(f"Saved figure: {out_png2}")


if __name__ == "__main__":
    main()
