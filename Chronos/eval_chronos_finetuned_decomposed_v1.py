import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from statsmodels.tsa.statespace.structural import UnobservedComponents


def kalman_decompose_context(y_context: np.ndarray):
    """Fit on context only (causal) — no look-ahead leakage."""
    model  = UnobservedComponents(y_context, level="local linear trend")
    result = model.fit(disp=False)
    low  = np.asarray(result.level.filtered, dtype=np.float32)
    high = (y_context - low).astype(np.float32)
    return low, high

logging.basicConfig(
    filename="Chronos_SSMI_FineTuned_Decomposed_v1_Metrics.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    force=True,
)

CHECKPOINT_LOW = Path("./chronos_ssmi_low_v1.pt")
CHECKPOINT_HIGH = Path("./chronos_ssmi_high_v1.pt")
BASE_MODEL_ID = "amazon/chronos-t5-base"




def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Pearson can fail on constant arrays; return NaN in that corner case.
    try:
        return float(pearsonr(y_true, y_pred).statistic ** 2)
    except Exception:
        return float("nan")


def load_pipeline(checkpoint_path: Path, device: torch.device) -> ChronosPipeline:
    pipeline = ChronosPipeline.from_pretrained(
        BASE_MODEL_ID,
        device_map=None,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    state = torch.load(str(checkpoint_path), map_location=device)
    pipeline.model.model.load_state_dict(state)
    pipeline.model.model.eval()
    return pipeline


def predict_median(pipeline: ChronosPipeline, context_1d: np.ndarray, horizon: int) -> np.ndarray:
    ctx_t = torch.tensor(context_1d, dtype=torch.float32)
    with torch.no_grad():
        # Shape: (batch=1, num_samples, horizon)
        samples = pipeline.predict(
            ctx_t,
            prediction_length=horizon,
            num_samples=20,
            temperature=1.0,
            top_k=50,
            top_p=1.0,
        )
    pred = samples[0].median(dim=0).values.detach().cpu().numpy().astype(float)
    return pred[:horizon]


def generate_plots(
    y, rmse_list, mape_list, pearson_list, directional_hits, dir_acc_list,
    context_window, forecast_horizon, step_size, num_segments,
    test_df, chronos_low, chronos_high, device, script_dir,
):
    plots_dir = script_dir / "plots_finetuned"
    plots_dir.mkdir(exist_ok=True)

    segments = np.arange(1, len(rmse_list) + 1)

    # --- 1. Per-segment RMSE ---
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(segments, rmse_list, marker="o", ms=4, color="steelblue")
    ax.axhline(np.median(rmse_list), color="crimson", ls="--", label=f"Median={np.median(rmse_list):.2f}")
    ax.set_title("Chronos Fine-Tuned (MA Decomposed (window=30)) — RMSE per Segment")
    ax.set_xlabel("Segment")
    ax.set_ylabel("RMSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "rmse_per_segment.png", dpi=150)
    plt.close(fig)

    # --- 2. Per-segment MAPE ---
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(segments, mape_list, marker="o", ms=4, color="darkorange")
    ax.axhline(np.median(mape_list), color="crimson", ls="--", label=f"Median={np.median(mape_list):.2f}%")
    ax.set_title("Chronos Fine-Tuned (MA Decomposed (window=30)) — MAPE per Segment")
    ax.set_xlabel("Segment")
    ax.set_ylabel("MAPE (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "mape_per_segment.png", dpi=150)
    plt.close(fig)

    # --- 3. Rolling directional accuracy (30-window) ---
    hits_arr = np.array(directional_hits, dtype=float)
    window_size = forecast_horizon
    rolling_acc = np.convolve(hits_arr, np.ones(window_size) / window_size, mode="valid") * 100
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(rolling_acc, color="seagreen")
    ax.axhline(50, color="gray", ls="--", lw=1, label="Random baseline (50%)")
    ax.axhline(hits_arr.mean() * 100, color="crimson", ls="--",
               label=f"Overall={hits_arr.mean() * 100:.1f}%")
    ax.set_title("Chronos Fine-Tuned — Rolling Directional Accuracy (30-day window)")
    ax.set_xlabel("Day index")
    ax.set_ylabel("Directional Accuracy (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "directional_accuracy_rolling.png", dpi=150)
    plt.close(fig)

    # --- 4. Predicted vs Actual (segment midpoint forecast) ---
    dates = test_df["Date"].values
    pred_indices, pred_vals, actual_vals = [], [], []
    for seg in range(num_segments):
        start_ctx = seg * step_size
        end_ctx = start_ctx + context_window
        if end_ctx + forecast_horizon > len(y):
            break
        mid_offset = forecast_horizon // 2
        ctx_low, ctx_high = kalman_decompose_context(y[start_ctx:end_ctx])
        fl = predict_median(chronos_low,  ctx_low,  forecast_horizon)
        fh = predict_median(chronos_high, ctx_high, forecast_horizon)
        combined = fl + fh
        pred_indices.append(end_ctx + mid_offset)
        pred_vals.append(float(combined[mid_offset]))
        actual_vals.append(float(y[end_ctx + mid_offset]))

    pred_indices = np.array(pred_indices)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(range(len(y)), y, color="black", lw=1, label="Actual SSMI")
    ax.scatter(pred_indices, pred_vals, s=12, color="steelblue",
               label="Predicted (segment mid-horizon)", zorder=3)
    ax.set_title("Chronos Fine-Tuned (MA Decomposed (window=30)) — Predicted vs Actual SSMI (2018–2021)")
    ax.set_xlabel("Test day index")
    ax.set_ylabel("Adj Close")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "predicted_vs_actual.png", dpi=150)
    plt.close(fig)

    # --- 5. Summary metrics bar chart ---
    hits_arr2 = np.array(directional_hits, dtype=float)
    metrics = {
        "Median RMSE": np.median(rmse_list),
        "Median MAPE (%)": np.median(mape_list),
        "Median R²": np.median(pearson_list),
        "Dir. Acc. (%)": hits_arr2.mean() * 100,
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(list(metrics.keys()), list(metrics.values()),
                  color=["steelblue", "darkorange", "seagreen", "mediumpurple"])
    max_val = max(metrics.values())
    for bar, val in zip(bars, metrics.values()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01 * max_val,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_title("Chronos Fine-Tuned (MA Decomposed (window=30)) — Summary Metrics")
    ax.set_ylabel("Value")
    fig.tight_layout()
    fig.savefig(plots_dir / "summary_metrics.png", dpi=150)
    plt.close(fig)

    # --- 6. Box plots: RMSE, MAPE, Dir Acc per segment ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))

    bp0 = axes[0].boxplot(rmse_list, patch_artist=True,
                          boxprops=dict(facecolor="steelblue", alpha=0.7),
                          medianprops=dict(color="crimson", lw=2))
    axes[0].set_title("RMSE per Segment")
    axes[0].set_ylabel("RMSE")
    axes[0].set_xticks([])
    axes[0].text(1, np.median(rmse_list), f"  med={np.median(rmse_list):.1f}",
                 va="center", color="crimson", fontsize=9)

    bp1 = axes[1].boxplot(mape_list, patch_artist=True,
                          boxprops=dict(facecolor="darkorange", alpha=0.7),
                          medianprops=dict(color="crimson", lw=2))
    axes[1].set_title("MAPE per Segment (%)")
    axes[1].set_ylabel("MAPE (%)")
    axes[1].set_xticks([])
    axes[1].text(1, np.median(mape_list), f"  med={np.median(mape_list):.2f}%",
                 va="center", color="crimson", fontsize=9)

    bp2 = axes[2].boxplot(dir_acc_list, patch_artist=True,
                          boxprops=dict(facecolor="seagreen", alpha=0.7),
                          medianprops=dict(color="crimson", lw=2))
    axes[2].axhline(50, color="gray", ls="--", lw=1, label="Random (50%)")
    axes[2].set_title("Directional Accuracy per Segment (%)")
    axes[2].set_ylabel("Dir. Acc. (%)")
    axes[2].set_xticks([])
    axes[2].legend(fontsize=8)
    axes[2].text(1, np.median(dir_acc_list), f"  med={np.median(dir_acc_list):.1f}%",
                 va="center", color="crimson", fontsize=9)

    fig.suptitle("Chronos Fine-Tuned (MA Decomposed (window=30)) — Per-Segment Box Plots", fontsize=13)
    fig.tight_layout()
    fig.savefig(plots_dir / "boxplots_per_segment.png", dpi=150)
    plt.close(fig)

    # --- 7. Box plot: Low vs High raw forecast ranges (1st segment sanity check) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    seg0_low, seg0_high = kalman_decompose_context(y[:context_window])
    ctx_t_low  = torch.tensor(seg0_low,  dtype=torch.float32)
    ctx_t_high = torch.tensor(seg0_high, dtype=torch.float32)
    with torch.no_grad():
        samples_low = chronos_low.predict(ctx_t_low, prediction_length=forecast_horizon,
                                          num_samples=20).squeeze(0).cpu().numpy()
        samples_high = chronos_high.predict(ctx_t_high, prediction_length=forecast_horizon,
                                            num_samples=20).squeeze(0).cpu().numpy()

    samples_combined = samples_low + samples_high

    ax.boxplot(
        [samples_low.flatten(), samples_high.flatten(), samples_combined.flatten()],
        labels=["Low-pass\n(trend)", "High-pass\n(cycle)", "Combined\n(low+high)"],
        patch_artist=True,
        boxprops=dict(alpha=0.7),
    )
    for patch, color in zip(ax.patches, ["steelblue", "darkorange", "seagreen"]):
        patch.set_facecolor(color)
    ax.axhline(float(y[context_window]), color="crimson", ls="--", lw=1.5,
               label=f"True next price={y[context_window]:.0f}")
    ax.set_title("Sample Distribution: Low / High / Combined (Segment 1)")
    ax.set_ylabel("Price (CHF)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "boxplot_low_high_combined_seg1.png", dpi=150)
    plt.close(fig)

    print(f"\nPlotlar kaydedildi -> {plots_dir}")
    logging.info(f"Plots saved to {plots_dir}")


def main():
    rmse_list = []
    mape_list = []
    pearson_list = []
    directional_hits = []
    dir_acc_list = []

    try:
        # ========================
        # 1) Load SSMI test slice (post-2018)
        # ========================
        script_dir = Path(__file__).resolve().parent
        data_csv = script_dir.parent / "DataSets" / "trimmed" / "SSMI.csv"

        df = pd.read_csv(data_csv, parse_dates=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        test_df = df[df["Date"] >= pd.Timestamp("2018-01-01")].reset_index(drop=True)

        nan_count = test_df["Adj Close"].isna().sum()
        if nan_count > 0:
            test_df = test_df.copy()
            test_df["Adj Close"] = test_df["Adj Close"].ffill().bfill()
            logging.info(f"Forward-filled {nan_count} NaN values in test data")
            print(f"Forward-filled {nan_count} NaN values in test data")

        y = test_df["Adj Close"].values.astype(float)
        total_samples = len(y)
        logging.info(
            f"Test range: {test_df['Date'].iloc[0].date()} -> {test_df['Date'].iloc[-1].date()} ({total_samples} days)"
        )
        print(
            f"Test range: {test_df['Date'].iloc[0].date()} -> {test_df['Date'].iloc[-1].date()} ({total_samples} days)"
        )

        # ========================
        # 2) Sliding-window + MA filter (global, consistent with ChronosBase_Filtering.ipynb)
        # ========================
        context_window = 120
        forecast_horizon = 30
        step_size = 30
        num_segments = (total_samples - context_window) // step_size
        logging.info(f"Segments to evaluate: {num_segments}")


        # ========================
        # 3) Load two fine-tuned Chronos models (low-pass / high-pass)
        # ========================
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        low_ckpt = (script_dir / CHECKPOINT_LOW).resolve()
        high_ckpt = (script_dir / CHECKPOINT_HIGH).resolve()

        chronos_low = load_pipeline(low_ckpt, device)
        logging.info(f"Fine-tuned low-pass weights loaded from {low_ckpt}")
        print(f"Fine-tuned low-pass weights loaded from {low_ckpt}")

        chronos_high = load_pipeline(high_ckpt, device)
        logging.info(f"Fine-tuned high-pass weights loaded from {high_ckpt}")
        print(f"Fine-tuned high-pass weights loaded from {high_ckpt}")

        # ========================
        # 4) Sliding-window evaluation with per-window normalization
        # ========================
        for segment in range(num_segments):
            start_context = segment * step_size
            end_context = start_context + context_window
            if end_context + forecast_horizon > total_samples:
                break

            y_context = y[start_context:end_context]
            y_true    = y[end_context : end_context + forecast_horizon]

            context_low, context_high = kalman_decompose_context(y_context)

            forecast_low  = predict_median(chronos_low,  context_low,  forecast_horizon)
            forecast_high = predict_median(chronos_high, context_high, forecast_horizon)
            combined_pred = forecast_low + forecast_high

            prev_anchor = np.concatenate([[y[end_context - 1]], y_true[:-1]])
            actual_direction = np.sign(y_true - prev_anchor)
            pred_direction = np.sign(combined_pred - prev_anchor)
            hits = (actual_direction == pred_direction).astype(int)
            directional_hits.extend(hits.tolist())

            rmse = np.sqrt(mean_squared_error(y_true, combined_pred))
            mape = mean_absolute_percentage_error(y_true, combined_pred) * 100
            r2 = safe_r2(y_true, combined_pred)

            rmse_list.append(rmse)
            mape_list.append(mape)
            pearson_list.append(r2)

            segment_dir_acc = hits.mean() * 100
            dir_acc_list.append(segment_dir_acc)
            logging.info(
                f"Segment {segment+1}/{num_segments}: RMSE={rmse:.4f}, MAPE={mape:.4f}%, R²={r2:.4f}, DirAcc={segment_dir_acc:.1f}%"
            )
            print(
                f"Segment {segment+1}/{num_segments} — RMSE: {rmse:.2f} | MAPE: {mape:.2f}% | R²: {r2:.4f} | Dir Acc: {segment_dir_acc:.1f}%"
            )

        # ========================
        # 5) Save results
        # ========================
        np.savez_compressed(
            "Chronos_SSMI_FineTuned_Decomposed_v1_Metrics.npz",
            rmse=np.array(rmse_list),
            mape=np.array(mape_list),
            pearson_coefficients=np.array(pearson_list),
            directional_hits=np.array(directional_hits),
            context_window=context_window,
            forecast_horizon=forecast_horizon,
            filter="kalman",
            num_segments=num_segments,
        )
        logging.info("Results saved to Chronos_SSMI_FineTuned_Decomposed_v1_Metrics.npz")

        # ========================
        # 6) Summary
        # ========================
        total_days = len(directional_hits)
        total_hits = sum(directional_hits)
        dir_acc_pct = (total_hits / total_days) * 100 if total_days else 0.0

        print("\n--- Median Metrics for Fine-Tuned Decomposed Chronos v1 on SSMI (post-2018 test) ---")
        print(f"Median RMSE:          {np.median(rmse_list):.4f}")
        print(f"Median MAPE:          {np.median(mape_list):.4f}%")
        print(f"Median Pearson R²:    {np.median(pearson_list):.4f}")
        print(f"Directional Accuracy: {total_hits}/{total_days} days ({dir_acc_pct:.2f}%)")

        generate_plots(
            y=y,
            rmse_list=rmse_list,
            mape_list=mape_list,
            pearson_list=pearson_list,
            directional_hits=directional_hits,
            dir_acc_list=dir_acc_list,
            context_window=context_window,
            forecast_horizon=forecast_horizon,
            step_size=step_size,
            num_segments=num_segments,
            test_df=test_df,
            chronos_low=chronos_low,
            chronos_high=chronos_high,
            device=device,
            script_dir=script_dir,
        )

    except Exception:
        logging.error("An error occurred.", exc_info=True)
        print("An error occurred. Check Chronos_SSMI_FineTuned_Decomposed_v1_Metrics.log for details.")
        try:
            np.savez_compressed(
                "partial_Chronos_SSMI_FineTuned_Decomposed_v1_Metrics.npz",
                rmse=np.array(rmse_list),
                mape=np.array(mape_list),
                pearson_coefficients=np.array(pearson_list),
                directional_hits=np.array(directional_hits),
            )
        except Exception:
            logging.error("Failed to save partial results.", exc_info=True)
    finally:
        logging.info("Forecasting run completed.")


if __name__ == "__main__":
    main()
