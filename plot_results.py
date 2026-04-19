"""Boxplot comparison of TimesFM configurations on SSMI.

Generates four 300-DPI PNGs in ./plots/:
    plot_rmse.png, plot_mape.png, plot_r2.png, plot_diracc.png

Each figure shows five boxplots (MA, HP, Butterworth, Kalman, Fine-tuned+HP)
plus a dashed horizontal line for Filipp's raw zero-shot baseline. The
DirAcc figure additionally shows a 50% random-guess reference line.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
NPZ_DIR = ROOT / "Google Time-Series Foundation Model"
OUT_DIR = ROOT / "plots"
OUT_DIR.mkdir(exist_ok=True)

CONFIGS = [
    ("MA",            "TimesFM_SSMI_Filtered_Metrics.npz"),
    ("HP",            "TimesFM_SSMI_HP_Metrics.npz"),
    ("Butterworth",   "TimesFM_SSMI_Butterworth_Metrics.npz"),
    ("Kalman",        "TimesFM_SSMI_Kalman_Metrics.npz"),
    ("Fine-tuned+HP", "TimesFM_SSMI_FineTuned_HP_Metrics.npz"),
]

PALETTE = [
    "#1F77B4",  # MA
    "#E15759",  # HP
    "#2CA02C",  # Butterworth
    "#9467BD",  # Kalman
    "#D4A72C",  # Fine-tuned+HP
]
RAW_COLOR = "#d62728"      # red (raw zero-shot baseline)
REF_COLOR = "#555555"      # grey (random-guess reference)

# Filipp's zero-shot raw baseline (no per-segment data available).
RAW_BASELINE = {
    "rmse":   165.93,
    "mape":   2.25,
    "r2":     0.356,
    "diracc": 51.89,
}

RMSE_KEYS    = ("rmse", "rmse_list", "rmses")
MAPE_KEYS    = ("mape", "mape_list", "mapes", "mape_pct")
R2_KEYS      = ("r2", "pearson_coefficients", "pearson_r2", "r_squared")
HITS_KEYS    = ("directional_hits", "dir_hits")
HORIZON_KEYS = ("forecast_horizon", "horizon")


def pick(npz, candidates, required: bool = True):
    """Return the first key in ``candidates`` present in ``npz``."""
    for k in candidates:
        if k in npz.files:
            return np.asarray(npz[k])
    if required:
        raise KeyError(
            f"None of {candidates} found in npz. Available keys: {list(npz.files)}"
        )
    return None


def load_metrics(path: Path) -> dict[str, np.ndarray]:
    d = np.load(path)
    rmse = pick(d, RMSE_KEYS)
    mape = pick(d, MAPE_KEYS)
    r2 = pick(d, R2_KEYS)
    hits = pick(d, HITS_KEYS)
    horizon_arr = pick(d, HORIZON_KEYS, required=False)
    horizon = int(horizon_arr) if horizon_arr is not None else 30
    n_full = (len(hits) // horizon) * horizon
    diracc_per_seg = hits[:n_full].reshape(-1, horizon).mean(axis=1) * 100.0
    return {"rmse": rmse, "mape": mape, "r2": r2, "diracc": diracc_per_seg}


def make_plot(
    metric_key: str,
    ylabel: str,
    title: str,
    outpath: Path,
    extra_hline: tuple[float, str] | None = None,
) -> None:
    data: list[np.ndarray] = []
    labels: list[str] = []
    for label, fname in CONFIGS:
        m = load_metrics(NPZ_DIR / fname)
        data.append(m[metric_key])
        labels.append(label)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.6)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(PALETTE[i % len(PALETTE)])
        patch.set_alpha(0.75)
        patch.set_edgecolor("black")
    for element in ("whiskers", "caps", "medians"):
        for line in bp[element]:
            line.set_color("black")
            line.set_linewidth(1.2)

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=11)

    baseline = RAW_BASELINE[metric_key]
    ax.axhline(
        baseline,
        linestyle="--",
        color=RAW_COLOR,
        linewidth=1.6,
        label=f"Raw zero-shot ({baseline:.2f})",
    )
    if extra_hline is not None:
        y, lbl = extra_hline
        ax.axhline(y, linestyle="--", color=REF_COLOR, linewidth=1.3, label=lbl)

    ax.set_title(title, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel("Configuration", fontsize=12)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=10, frameon=True)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {outpath.relative_to(ROOT)}")


def main() -> None:
    make_plot(
        "rmse", "RMSE", "RMSE \u2014 TimesFM on SSMI", OUT_DIR / "plot_rmse.png"
    )
    make_plot(
        "mape", "MAPE (%)", "MAPE \u2014 TimesFM on SSMI", OUT_DIR / "plot_mape.png"
    )
    make_plot(
        "r2",
        "Pearson R\u00b2",
        "Pearson R\u00b2 \u2014 TimesFM on SSMI",
        OUT_DIR / "plot_r2.png",
    )
    make_plot(
        "diracc",
        "Directional Accuracy (%)",
        "Directional Accuracy \u2014 TimesFM on SSMI",
        OUT_DIR / "plot_diracc.png",
        extra_hline=(50.0, "Random guess (50%)"),
    )


if __name__ == "__main__":
    main()
