from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = REPO_ROOT / "Chronos"
OUTPUT_DIR = REPO_ROOT / "chronos_analysis" / "plots"
PALETTE = ["#1F77B4", "#E15759", "#2CA02C", "#9467BD", "#D4A72C"]


@dataclass(frozen=True)
class VariantSpec:
    filename: str
    display_name: str


VARIANTS = [
    VariantSpec("ChronosBase_SSMI_Metrics.npz", "Baseline Chronos"),
    VariantSpec("ChronosBase_SSMI_Filtered_Metrics.npz", "Moving Average"),
    VariantSpec("ChronosBase_SSMI_HP_Metrics.npz", "Hodrick-Prescott"),
    VariantSpec("ChronosBase_SSMI_Kalman_Metrics.npz", "Kalman"),
    VariantSpec("ChronosBase_SSMI_Butterworth_Metrics.npz", "Butterworth"),
]


def load_metric_values(npz_path: Path, metric: str) -> np.ndarray:
    payload = np.load(npz_path, allow_pickle=True)
    if metric == "r2":
        if "r2" in payload.files:
            values = payload["r2"]
        elif "pearson_coefficients" in payload.files:
            values = payload["pearson_coefficients"]
        else:
            raise KeyError(f"{npz_path.name} does not contain r2-compatible values.")
    else:
        if metric not in payload.files:
            raise KeyError(f"{npz_path.name} does not contain {metric}.")
        values = payload[metric]

    values = np.asarray(values, dtype=float).reshape(-1)
    return values[np.isfinite(values)]


def load_directional_accuracy_values(npz_path: Path) -> np.ndarray:
    payload = np.load(npz_path, allow_pickle=True)
    hits = np.asarray(payload["directional_hits"], dtype=float).reshape(-1)
    forecast_horizon = int(np.asarray(payload["forecast_horizon"]).item())
    num_segments = int(np.asarray(payload["num_segments"]).item())
    if hits.size != forecast_horizon * num_segments:
        raise ValueError(f"{npz_path.name} has inconsistent directional_hits size.")
    segment_hits = hits.reshape(num_segments, forecast_horizon)
    return segment_hits.mean(axis=1) * 100.0


def load_directional_accuracy_summary(npz_path: Path) -> tuple[int, int, float]:
    payload = np.load(npz_path, allow_pickle=True)
    hits = np.asarray(payload["directional_hits"], dtype=int).reshape(-1)
    total_days = int(hits.size)
    hit_days = int(hits.sum())
    percentage = (hit_days / total_days) * 100.0
    return hit_days, total_days, percentage


def collect_metric_distributions(metric: str) -> Dict[str, np.ndarray]:
    distributions: Dict[str, np.ndarray] = {}
    for variant in VARIANTS:
        npz_path = SOURCE_DIR / variant.filename
        distributions[variant.display_name] = load_metric_values(npz_path, metric)
    return distributions


def collect_directional_accuracy_distributions() -> Dict[str, np.ndarray]:
    distributions: Dict[str, np.ndarray] = {}
    for variant in VARIANTS:
        npz_path = SOURCE_DIR / variant.filename
        distributions[variant.display_name] = load_directional_accuracy_values(npz_path)
    return distributions


def collect_directional_accuracy_summary() -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for variant in VARIANTS:
        hit_days, total_days, percentage = load_directional_accuracy_summary(SOURCE_DIR / variant.filename)
        rows.append(
            {
                "label": variant.display_name,
                "hit_days": hit_days,
                "total_days": total_days,
                "percentage": percentage,
            }
        )
    return rows


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "#F8F6F1",
            "axes.facecolor": "#FCFBF8",
            "axes.edgecolor": "#4A4A4A",
            "axes.labelcolor": "#222222",
            "xtick.color": "#2C2C2C",
            "ytick.color": "#2C2C2C",
            "grid.color": "#D9D4CA",
            "grid.alpha": 0.55,
        }
    )


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


def render_boxplot(
    distributions: Dict[str, np.ndarray],
    *,
    title: str,
    ylabel: str,
    output_name: str,
    ylim: tuple[float, float] | None = None,
) -> Path:
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / output_name
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_directional_accuracy_summary_chart(
    rows: list[dict[str, float | int | str]],
    *,
    title: str,
    output_name: str,
) -> Path:
    sorted_rows = sorted(rows, key=lambda row: float(row["percentage"]), reverse=True)
    labels = [str(row["label"]) for row in sorted_rows]
    percentages = np.array([float(row["percentage"]) for row in sorted_rows], dtype=float)
    hit_days = [int(row["hit_days"]) for row in sorted_rows]
    total_days = [int(row["total_days"]) for row in sorted_rows]

    fig, ax = plt.subplots(figsize=(14.8, 8.8))
    y_positions = np.arange(len(labels))
    bars = ax.barh(
        y_positions,
        percentages,
        color=PALETTE,
        edgecolor="#4E4E4E",
        linewidth=1.3,
        alpha=0.9,
        height=0.72,
    )

    ax.axvline(50.0, color="#222222", linestyle="--", linewidth=2.2, alpha=0.9)
    ax.set_title(title, fontsize=22, fontweight="bold", pad=18)
    ax.set_xlabel("Directional Accuracy (%)", fontsize=14, fontweight="semibold")
    ax.set_ylabel("Chronos Variant", fontsize=14, fontweight="semibold")
    ax.set_xlim(0, max(60, float(percentages.max()) + 6.0))
    ax.set_yticks(y_positions, labels=labels, fontsize=12)
    ax.invert_yaxis()
    ax.grid(axis="x", linewidth=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    for label in ax.get_yticklabels():
        label.set_fontweight("semibold")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    for bar, pct, hits, total in zip(bars, percentages, hit_days, total_days):
        y = bar.get_y() + bar.get_height() / 2
        x = bar.get_width()
        ax.text(
            x + 0.9,
            y,
            f"{hits}/{total} days ({pct:.2f}%)",
            va="center",
            ha="left",
            fontsize=11,
            color="#222222",
        )

    ax.text(
        50.6,
        -0.6,
        "50% reference",
        fontsize=11,
        color="#222222",
        fontweight="semibold",
    )

    fig.tight_layout(pad=1.2)
    output_path = OUTPUT_DIR / output_name
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    configure_plot_style()
    r2_distributions = collect_metric_distributions("r2")
    mape_distributions = collect_metric_distributions("mape")
    rmse_distributions = collect_metric_distributions("rmse")
    directional_accuracy_rows = collect_directional_accuracy_summary()

    generated_paths: Iterable[Path] = [
        render_boxplot(
            r2_distributions,
            title="Chronos Frequency-Filtering Variants: R^2 Distribution",
            ylabel="R^2",
            output_name="chronos_r2_distribution.png",
            ylim=(-0.02, 1.02),
        ),
        render_boxplot(
            mape_distributions,
            title="Chronos Frequency-Filtering Variants: MAPE Distribution",
            ylabel="MAPE (%)",
            output_name="chronos_mape_distribution.png",
        ),
        render_boxplot(
            rmse_distributions,
            title="Chronos Frequency-Filtering Variants: RMSE Distribution",
            ylabel="RMSE",
            output_name="chronos_rmse_distribution.png",
        ),
        render_directional_accuracy_summary_chart(
            directional_accuracy_rows,
            title="Chronos Frequency-Filtering Variants: Directional Accuracy",
            output_name="chronos_directional_accuracy_histogram.png",
        ),
    ]

    for path in generated_paths:
        print(path)


if __name__ == "__main__":
    main()
