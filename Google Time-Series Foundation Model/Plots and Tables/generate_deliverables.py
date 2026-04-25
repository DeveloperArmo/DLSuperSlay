"""Generate final report deliverables: tables (markdown + CSV) and plots.

Outputs (all written next to this script):
  - finetune_180_15_rmse.png
  - finetune_180_15_mape.png
  - finetune_180_15_r2.png
  - finetune_180_15_diracc.png
  - tables.md
  - results_combined.csv

Style matches Ismael's Chronos plot script (cream background, orange medians,
DejaVu Sans, hidden top/right spines).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
TFM_DIR = PROJECT_ROOT / "Google Time-Series Foundation Model"
SWEEP_DIR = TFM_DIR / "Hyperparams Sweep"
FT_DIR = SWEEP_DIR / "Fine-Tuning"
OUTPUT_DIR = SCRIPT_DIR

# Palette for boxplots (matching Ismael's)
PALETTE = ["#1F77B4", "#E15759", "#2CA02C", "#9467BD", "#D4A72C"]


def configure_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.facecolor": "#F8F6F1",
        "axes.facecolor": "#FCFBF8",
        "axes.edgecolor": "#4A4A4A",
        "axes.labelcolor": "#222222",
        "xtick.color": "#2C2C2C",
        "ytick.color": "#2C2C2C",
        "grid.color": "#D9D4CA",
        "grid.alpha": 0.55,
    })


def style_axes(ax: plt.Axes, *, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=22, fontweight="bold", pad=18)
    ax.set_ylabel(ylabel, fontsize=14, fontweight="semibold")
    ax.set_xlabel("")
    ax.grid(axis="y", linewidth=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelrotation=0, labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    for label in ax.get_xticklabels():
        label.set_fontweight("semibold")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)


def load_metric_arrays(npz_path: Path) -> dict:
    payload = np.load(npz_path)
    rmse = np.asarray(payload["rmse"], dtype=float)
    mape = np.asarray(payload["mape"], dtype=float)
    r2 = np.asarray(payload["pearson_coefficients"], dtype=float)
    hits = np.asarray(payload["directional_hits"], dtype=float).reshape(-1)
    horizon = int(payload["forecast_horizon"])
    n_full = (len(hits) // horizon) * horizon
    diracc_per_seg = hits[:n_full].reshape(-1, horizon).mean(axis=1) * 100.0
    return {
        "rmse": rmse[np.isfinite(rmse)],
        "mape": mape[np.isfinite(mape)],
        "r2": r2[np.isfinite(r2)],
        "diracc": diracc_per_seg[np.isfinite(diracc_per_seg)],
        "hits_total": int(hits.sum()),
        "hits_count": int(len(hits)),
        "num_segments": int(payload["num_segments"]),
    }


def render_boxplot_180_15(metric_key: str, ylabel: str, title: str, output_name: str,
                          ylim: tuple | None = None) -> Path:
    """Boxplot: baseline vs v1 vs v2 vs v3 at 180/15 setting."""
    configs = [
        ("Baseline", SWEEP_DIR / "TimesFM_SSMI_Baseline_180_15_Metrics.npz"),
        ("Fine-tuned v1", FT_DIR / "TimesFM_SSMI_FineTuned_v1_180_15_Metrics.npz"),
        ("Fine-tuned v2", FT_DIR / "TimesFM_SSMI_FineTuned_v2_180_15_Metrics.npz"),
        ("Fine-tuned v3", FT_DIR / "TimesFM_SSMI_FineTuned_v3_180_15_Metrics.npz"),
    ]

    distributions = {}
    for label, path in configs:
        m = load_metric_arrays(path)
        distributions[label] = m[metric_key]

    labels = list(distributions.keys())
    values = [distributions[label] for label in labels]

    fig, ax = plt.subplots(figsize=(14.5, 8.5))
    box = ax.boxplot(
        values,
        tick_labels=labels,
        patch_artist=True,
        widths=0.62,
        showfliers=False,
        medianprops={"color": "#F28E2B", "linewidth": 2.4},
        whiskerprops={"color": "#505050", "linewidth": 1.7},
        capprops={"color": "#505050", "linewidth": 1.7},
        boxprops={"edgecolor": "#505050", "linewidth": 1.7},
    )
    for patch, color in zip(box["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.38)

    style_axes(ax, title=title, ylabel=ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)

    fig.tight_layout(pad=1.2)
    output_path = OUTPUT_DIR / output_name
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_tables() -> str:
    """Build all markdown tables. Returns the full markdown string."""
    md_lines = ["# TimesFM Final Results", ""]

    # === Table 1: hyperparameter sweep ===
    md_lines += ["## Table 1 — Hyperparameter Sweep at context=180, horizon=15", ""]
    sweep_csv = SWEEP_DIR / "timesfm_hyperparams_results.csv"
    if sweep_csv.exists():
        df = pd.read_csv(sweep_csv)
        df = df.copy()
        df["rmse"] = df["rmse"].round(2)
        df["mape"] = df["mape"].round(2)
        df["pearson_r2"] = df["pearson_r2"].round(3)
        df["directional_accuracy"] = df["directional_accuracy"].round(2)
        md_lines.append(df.to_markdown(index=False))
        md_lines.append("")
    else:
        md_lines.append("_(timesfm_hyperparams_results.csv not found)_")
        md_lines.append("")

    # === Table 2: best filter per metric ===
    md_lines += ["## Table 2 — Best Filter per Metric (Hyperparameter Sweep)", ""]
    best_csv = SWEEP_DIR / "timesfm_best_global.csv"
    if best_csv.exists():
        df = pd.read_csv(best_csv)
        if "metric_value" in df.columns:
            df["metric_value"] = df["metric_value"].round(3)
        md_lines.append(df.to_markdown(index=False))
        md_lines.append("")
    else:
        md_lines.append("_(timesfm_best_global.csv not found)_")
        md_lines.append("")

    # === Table 3: fine-tuning comparison at 180/15 ===
    md_lines += ["## Table 3 — Fine-Tuning Comparison at context=180, horizon=15", ""]
    rows_180 = []
    configs_180 = [
        ("Baseline", SWEEP_DIR / "TimesFM_SSMI_Baseline_180_15_Metrics.npz"),
        ("Fine-tuned v1 (head only)", FT_DIR / "TimesFM_SSMI_FineTuned_v1_180_15_Metrics.npz"),
        ("Fine-tuned v2 (13% + norm)", FT_DIR / "TimesFM_SSMI_FineTuned_v2_180_15_Metrics.npz"),
        ("Fine-tuned v3 (23% + reg)", FT_DIR / "TimesFM_SSMI_FineTuned_v3_180_15_Metrics.npz"),
    ]
    for label, path in configs_180:
        m = load_metric_arrays(path)
        diracc_overall = (m["hits_total"] / m["hits_count"]) * 100.0
        rows_180.append({
            "Config": label,
            "Segments": m["num_segments"],
            "RMSE": round(float(np.median(m["rmse"])), 2),
            "MAPE %": round(float(np.median(m["mape"])), 2),
            "R²": round(float(np.median(m["r2"])), 3),
            "DirAcc %": round(diracc_overall, 2),
        })
    df3 = pd.DataFrame(rows_180)
    md_lines.append(df3.to_markdown(index=False))
    md_lines.append("")

    # === Table 4: cross-setting comparison ===
    md_lines += ["## Table 4 — Cross-Setting Comparison", "",
                 "Comparing baseline + best fine-tuning at two sliding-window settings:",
                 ""]
    rows_cross = []
    # 120/30 setting
    configs_120 = [
        ("120/30 Baseline", TFM_DIR / "TimesFM_SSMI_PostTest_Baseline_Metrics.npz", 120, 30),
        ("120/30 Fine-tuned v1", TFM_DIR / "TimesFM_SSMI_FineTuned_Decomposed_Metrics.npz", 120, 30),
        ("120/30 Fine-tuned v2", TFM_DIR / "Fine-Tuning Unfreeze Layers" /
                                  "TimesFM_SSMI_FineTuned_Decomposed_v2_Metrics.npz", 120, 30),
        ("120/30 Fine-tuned v3", TFM_DIR / "Fine-Tuning Unfreeze Layers" /
                                  "TimesFM_SSMI_FineTuned_Decomposed_v3_Metrics.npz", 120, 30),
    ]
    configs_180_v2 = [
        ("180/15 Baseline", SWEEP_DIR / "TimesFM_SSMI_Baseline_180_15_Metrics.npz", 180, 15),
        ("180/15 Fine-tuned v1", FT_DIR / "TimesFM_SSMI_FineTuned_v1_180_15_Metrics.npz", 180, 15),
        ("180/15 Fine-tuned v2", FT_DIR / "TimesFM_SSMI_FineTuned_v2_180_15_Metrics.npz", 180, 15),
        ("180/15 Fine-tuned v3", FT_DIR / "TimesFM_SSMI_FineTuned_v3_180_15_Metrics.npz", 180, 15),
    ]
    for label, path, ctx, hor in configs_120 + configs_180_v2:
        if not path.exists():
            rows_cross.append({"Config": label, "Context": ctx, "Horizon": hor,
                               "RMSE": "—", "MAPE %": "—", "R²": "—", "DirAcc %": "—"})
            continue
        m = load_metric_arrays(path)
        diracc_overall = (m["hits_total"] / m["hits_count"]) * 100.0
        rows_cross.append({
            "Config": label,
            "Context": ctx,
            "Horizon": hor,
            "RMSE": round(float(np.median(m["rmse"])), 2),
            "MAPE %": round(float(np.median(m["mape"])), 2),
            "R²": round(float(np.median(m["r2"])), 3),
            "DirAcc %": round(diracc_overall, 2),
        })
    df4 = pd.DataFrame(rows_cross)
    md_lines.append(df4.to_markdown(index=False))
    md_lines.append("")

    return "\n".join(md_lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_plot_style()

    # Plots: 4 separate boxplots for fine-tuning comparison at 180/15
    print("Generating fine-tuning comparison plots (180/15)...")
    paths = []
    paths.append(render_boxplot_180_15(
        "rmse", "RMSE", "Fine-Tuning Variants — RMSE (180/15)",
        "finetune_180_15_rmse.png",
    ))
    paths.append(render_boxplot_180_15(
        "mape", "MAPE (%)", "Fine-Tuning Variants — MAPE (180/15)",
        "finetune_180_15_mape.png",
    ))
    paths.append(render_boxplot_180_15(
        "r2", "Pearson R²", "Fine-Tuning Variants — Pearson R² (180/15)",
        "finetune_180_15_r2.png",
        ylim=(-0.02, 1.02),
    ))
    paths.append(render_boxplot_180_15(
        "diracc", "Directional Accuracy (%)",
        "Fine-Tuning Variants — Directional Accuracy (180/15)",
        "finetune_180_15_diracc.png",
    ))
    for p in paths:
        print(f"  saved {p.name}")

    # Tables
    print("\nGenerating tables...")
    md = build_tables()
    md_path = OUTPUT_DIR / "tables.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"  saved {md_path.name}")

    print("\nAll deliverables in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()