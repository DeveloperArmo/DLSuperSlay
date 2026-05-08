import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from statsmodels.tsa.filters.hp_filter import hpfilter

logging.basicConfig(
    filename="Chronos_SSMI_FineTuned_Decomposed_v1_Metrics.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    force=True,
)

CHECKPOINT_LOW = Path("./chronos_ssmi_low_v1.pt")
CHECKPOINT_HIGH = Path("./chronos_ssmi_high_v1.pt")
BASE_MODEL_ID = "amazon/chronos-t5-base"


def hp_decompose_context(y_context, lamb):
    """HP decomposition on the context only — no look-ahead."""
    cycle, trend = hpfilter(y_context, lamb=lamb)
    return np.asarray(trend), np.asarray(cycle)


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


def main():
    rmse_list = []
    mape_list = []
    pearson_list = []
    directional_hits = []

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
        # 2) Sliding-window + HP config
        # ========================
        context_window = 120
        forecast_horizon = 30
        step_size = 30
        lamb = 129600
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
            y_true = y[end_context : end_context + forecast_horizon]

            # Per-window stats from the RAW context (matches training).
            ctx_mean = float(y_context.mean())
            ctx_std = float(y_context.std())
            if ctx_std < 1e-6:
                ctx_std = 1.0

            # HP-decompose the RAW context, then normalize each component
            # with the RAW context stats so that low_norm + high_norm == (raw - mean)/std.
            context_low_raw, context_high_raw = hp_decompose_context(y_context, lamb=lamb)
            context_low_norm = (context_low_raw - ctx_mean) / ctx_std
            context_high_norm = (context_high_raw - ctx_mean) / ctx_std

            forecast_low_norm = predict_median(chronos_low, context_low_norm, forecast_horizon)
            forecast_high_norm = predict_median(chronos_high, context_high_norm, forecast_horizon)

            # Denormalize each component back to price space, then recombine.
            forecast_low = forecast_low_norm * ctx_std + ctx_mean
            forecast_high = forecast_high_norm * ctx_std + ctx_mean
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
            lamb=lamb,
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
