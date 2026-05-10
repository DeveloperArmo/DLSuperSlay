"""Compare three conditions on the same post-2018 test period:
  1. Baseline      - Chronos zero-shot, no filter
  2. ZeroShot + MA - Chronos zero-shot, MA-filtered (window=30)
  3. FineTuned + MA - Chronos fine-tuned on MA components
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_CSV     = PROJECT_ROOT / "DataSets" / "trimmed" / "SSMI.csv"

BASE_MODEL_ID  = "amazon/chronos-t5-base"
CKPT_LOW       = SCRIPT_DIR / "chronos_ssmi_low_v1.pt"
CKPT_HIGH      = SCRIPT_DIR / "chronos_ssmi_high_v1.pt"
TEST_START     = "2018-01-01"
CONTEXT_WINDOW = 120
HORIZON        = 30
STEP           = 30
MA_WINDOW      = 30

COLORS = {
    "Baseline":       "#4C72B0",
    "ZeroShot + MA":  "#DD8452",
    "FineTuned + MA": "#55A868",
}


def ma_decompose(y: np.ndarray, window: int = MA_WINDOW):
    low  = np.convolve(y, np.ones(window) / window, mode="same").astype(np.float32)
    high = (y - low).astype(np.float32)
    return low, high


def predict_median(pipeline, context_1d: np.ndarray, horizon: int) -> np.ndarray:
    ctx = torch.tensor(context_1d, dtype=torch.float32)
    with torch.no_grad():
        samples = pipeline.predict(ctx, prediction_length=horizon, num_samples=20)
    return samples[0].median(dim=0).values.cpu().numpy().astype(float)[:horizon]


def load_finetuned(ckpt_path: Path, device: torch.device) -> ChronosPipeline:
    pipeline = ChronosPipeline.from_pretrained(
        BASE_MODEL_ID, device_map=None,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    state = torch.load(str(ckpt_path), map_location=device)
    pipeline.model.model.load_state_dict(state)
    pipeline.model.model.eval()
    return pipeline


def eval_segments(y, pred_fn):
    """Sliding-window eval. pred_fn(s, e) -> forecast array of length HORIZON."""
    total = len(y)
    n_seg = (total - CONTEXT_WINDOW) // STEP
    rmse_list, mape_list, r2_list, dacc_list = [], [], [], []
    for seg in range(n_seg):
        s = seg * STEP
        e = s + CONTEXT_WINDOW
        if e + HORIZON > total:
            break
        y_true = y[e:e + HORIZON]
        pred   = pred_fn(s, e)
        anchor = np.concatenate([[y[e - 1]], y_true[:-1]])
        hits   = (np.sign(y_true - anchor) == np.sign(pred - anchor)).astype(float)
        rmse_list.append(float(np.sqrt(mean_squared_error(y_true, pred))))
        mape_list.append(float(mean_absolute_percentage_error(y_true, pred) * 100))
        try:
            r2_list.append(float(pearsonr(y_true, pred).statistic ** 2))
        except Exception:
            r2_list.append(float("nan"))
        dacc_list.append(float(hits.mean() * 100))
    return tuple(map(np.array, [rmse_list, mape_list, r2_list, dacc_list]))


def main():
    df   = pd.read_csv(DATA_CSV, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    test = df[df["Date"] >= pd.Timestamp(TEST_START)].reset_index(drop=True)
    y    = test["Adj Close"].ffill().bfill().values.astype(float)
    print(f"Test period: {test['Date'].iloc[0].date()} -> {test['Date'].iloc[-1].date()}  ({len(y)} days)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    y_low, y_high = ma_decompose(y)

    # 1. Baseline
    print("\n[1/3] Baseline (zero-shot, no filter)...")
    base = ChronosPipeline.from_pretrained(
        BASE_MODEL_ID, device_map=None,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    base.model.eval()

    r_base, m_base, p_base, d_base = eval_segments(
        y, lambda s, e: predict_median(base, y[s:e], HORIZON),
    )
    print(f"  Median RMSE={np.median(r_base):.2f}  MAPE={np.median(m_base):.2f}%  DirAcc={np.nanmean(d_base):.1f}%")

    # 2. ZeroShot + MA
    print("\n[2/3] ZeroShot + MA filter...")
    r_zma, m_zma, p_zma, d_zma = eval_segments(
        y, lambda s, e: (predict_median(base, y_low[s:e],  HORIZON) +
                         predict_median(base, y_high[s:e], HORIZON)),
    )
    print(f"  Median RMSE={np.median(r_zma):.2f}  MAPE={np.median(m_zma):.2f}%  DirAcc={np.nanmean(d_zma):.1f}%")

    # 3. FineTuned + MA
    print("\n[3/3] FineTuned + MA...")
    ft_low  = load_finetuned(CKPT_LOW,  device)
    ft_high = load_finetuned(CKPT_HIGH, device)
    r_ft, m_ft, p_ft, d_ft = eval_segments(
        y, lambda s, e: (predict_median(ft_low,  y_low[s:e],  HORIZON) +
                         predict_median(ft_high, y_high[s:e], HORIZON)),
    )
    print(f"  Median RMSE={np.median(r_ft):.2f}  MAPE={np.median(m_ft):.2f}%  DirAcc={np.nanmean(d_ft):.1f}%")

    # Plot
    labels    = ["Baseline", "ZeroShot + MA", "FineTuned + MA"]
    colors    = [COLORS[l] for l in labels]
    data_sets = [
        (r_base, m_base, p_base, d_base),
        (r_zma,  m_zma,  p_zma,  d_zma),
        (r_ft,   m_ft,   p_ft,   d_ft),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Chronos on SSMI - Post-2018 Test Period Comparison", fontsize=15, fontweight="bold")

    box_metrics = [
        (0, "RMSE per Segment",                 "RMSE"),
        (1, "MAPE per Segment (%)",              "MAPE (%)"),
        (3, "Directional Accuracy per Segment (%)", "Dir. Acc. (%)"),
    ]

    for ax, (mi, title, ylabel) in zip(axes.flat[:3], box_metrics):
        bp = ax.boxplot(
            [ds[mi] for ds in data_sets],
            labels=labels,
            patch_artist=True,
            medianprops=dict(color="black", lw=2),
            showfliers=True,
            flierprops=dict(marker="o", markersize=3, alpha=0.5),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelsize=9)
        if mi == 3:
            ax.axhline(50, color="gray", ls="--", lw=1, label="Random (50%)")
            ax.legend(fontsize=8)
        ymax = ax.get_ylim()[1]
        for i, ds in enumerate(data_sets):
            med = np.median(ds[mi])
            ax.text(i + 1, ymax * 0.97, f"{med:.1f}",
                    ha="center", va="top", fontsize=8, fontweight="bold", color=colors[i])

    ax4 = axes[1, 1]
    x = np.arange(3)
    w = 0.25
    vals = [
        ([np.median(ds[0]) for ds in data_sets], "Median RMSE",     "#4C72B0"),
        ([np.median(ds[1]) for ds in data_sets], "Median MAPE (%)", "#DD8452"),
        ([np.nanmean(ds[3]) for ds in data_sets], "Dir. Acc. (%)",  "#55A868"),
    ]
    for i, (v, lbl, col) in enumerate(vals):
        bars = ax4.bar(x + (i - 1) * w, v, w, label=lbl, color=col, alpha=0.85)
        for bar in bars:
            ax4.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.5,
                     f"{bar.get_height():.1f}",
                     ha="center", va="bottom", fontsize=7)
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, fontsize=9)
    ax4.set_title("Summary - Median Metrics", fontsize=11)
    ax4.set_ylabel("Value")
    ax4.legend(fontsize=8)

    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, -0.02), frameon=True)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out = SCRIPT_DIR / "plots_finetuned" / "comparison_all.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
