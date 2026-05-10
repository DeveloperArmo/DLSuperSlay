"""Evaluate the Kalman-v1 fine-tuned TimesFM models on the post-2018 SSMI test set.

Mirrors TimesFM_FineTuned_Decomposed_Eval.ipynb (v1 HP filter eval), only swapping
the HP decomposition for the same scalar Kalman filter used during fine-tuning
(process_noise=0.1, measurement_noise=1.0).

Outputs (saved next to this script):
  - TimesFM_SSMI_FineTuned_Decomposed_Kalman_v1_Metrics.npz
  - TimesFM_SSMI_FineTuned_Decomposed_Kalman_v1_Metrics.log
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import timesfm
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from scipy.stats import pearsonr


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DATA_CSV = PROJECT_ROOT / "DataSets" / "trimmed" / "SSMI.csv"
CHECKPOINT_LOW = SCRIPT_DIR / "timesfm_ssmi_low_kalman_v1.pt"
CHECKPOINT_HIGH = SCRIPT_DIR / "timesfm_ssmi_high_kalman_v1.pt"

METRICS_NPZ = SCRIPT_DIR / "TimesFM_SSMI_FineTuned_Decomposed_Kalman_v1_Metrics.npz"
METRICS_LOG = SCRIPT_DIR / "TimesFM_SSMI_FineTuned_Decomposed_Kalman_v1_Metrics.log"

# Must match training config (finetune_timesfm_decomposed_kalman_v1.py)
KALMAN_PROCESS_NOISE = 0.1
KALMAN_MEASUREMENT_NOISE = 1.0

logging.basicConfig(
    filename=str(METRICS_LOG),
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    force=True,
)


def kalman_decompose_context(y_context, process_noise=KALMAN_PROCESS_NOISE,
                              measurement_noise=KALMAN_MEASUREMENT_NOISE):
    """One-pass causal scalar Kalman filter — identical to training-time decomposition."""
    y = np.asarray(y_context, dtype=float)
    q = float(process_noise)
    r = float(measurement_noise)

    x = float(y[0])
    p = 1.0
    low = np.zeros_like(y, dtype=float)

    for i, z in enumerate(y):
        p = p + q
        k = p / (p + r)
        x = x + k * (float(z) - x)
        p = (1.0 - k) * p
        low[i] = x

    high = y - low
    return low, high


def main():
    rmse_list, mape_list, pearson_list = [], [], []
    directional_hits = []

    try:
        df = pd.read_csv(DATA_CSV, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
        test_df = df[df["Date"] >= pd.Timestamp("2018-01-01")].reset_index(drop=True)

        nan_count = int(test_df["Adj Close"].isna().sum())
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

        context_window = 120
        forecast_horizon = 30
        step_size = 30
        num_segments = (total_samples - context_window) // step_size
        logging.info(f"Segments to evaluate: {num_segments}")

        tfm_low = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="cpu",
                per_core_batch_size=32,
                horizon_len=forecast_horizon,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-1.0-200m-pytorch",
            ),
        )
        tfm_low._model.load_state_dict(torch.load(str(CHECKPOINT_LOW), map_location="cpu"))
        tfm_low._model.eval()
        logging.info(f"Fine-tuned low-pass weights loaded from {CHECKPOINT_LOW.name}")
        print(f"Fine-tuned low-pass weights loaded from {CHECKPOINT_LOW.name}")

        tfm_high = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="cpu",
                per_core_batch_size=32,
                horizon_len=forecast_horizon,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-1.0-200m-pytorch",
            ),
        )
        tfm_high._model.load_state_dict(torch.load(str(CHECKPOINT_HIGH), map_location="cpu"))
        tfm_high._model.eval()
        logging.info(f"Fine-tuned high-pass weights loaded from {CHECKPOINT_HIGH.name}")
        print(f"Fine-tuned high-pass weights loaded from {CHECKPOINT_HIGH.name}")

        for segment in range(num_segments):
            start_context = segment * step_size
            end_context = start_context + context_window
            if end_context + forecast_horizon > total_samples:
                break

            y_context = y[start_context:end_context]
            y_true = y[end_context:end_context + forecast_horizon]

            context_low, context_high = kalman_decompose_context(y_context)

            point_forecast_low, _ = tfm_low.forecast([context_low], freq=[0])
            forecast_low = point_forecast_low[0][:forecast_horizon]

            point_forecast_high, _ = tfm_high.forecast([context_high], freq=[0])
            forecast_high = point_forecast_high[0][:forecast_horizon]

            combined_pred = forecast_low + forecast_high

            prev_anchor = np.concatenate([[y[end_context - 1]], y_true[:-1]])
            actual_direction = np.sign(y_true - prev_anchor)
            pred_direction = np.sign(combined_pred - prev_anchor)
            hits = (actual_direction == pred_direction).astype(int)
            directional_hits.extend(hits.tolist())

            rmse = np.sqrt(mean_squared_error(y_true, combined_pred))
            mape = mean_absolute_percentage_error(y_true, combined_pred) * 100
            r2 = pearsonr(y_true, combined_pred).statistic ** 2

            rmse_list.append(rmse)
            mape_list.append(mape)
            pearson_list.append(r2)

            segment_dir_acc = hits.mean() * 100
            logging.info(
                f"Segment {segment+1}/{num_segments}: RMSE={rmse:.4f}, MAPE={mape:.4f}%, R2={r2:.4f}, DirAcc={segment_dir_acc:.1f}%"
            )
            print(
                f"Segment {segment+1}/{num_segments} - RMSE: {rmse:.2f} | MAPE: {mape:.2f}% | R2: {r2:.4f} | Dir Acc: {segment_dir_acc:.1f}%"
            )

        np.savez_compressed(
            METRICS_NPZ,
            rmse=np.array(rmse_list),
            mape=np.array(mape_list),
            pearson_coefficients=np.array(pearson_list),
            directional_hits=np.array(directional_hits),
            context_window=context_window,
            forecast_horizon=forecast_horizon,
            process_noise=KALMAN_PROCESS_NOISE,
            measurement_noise=KALMAN_MEASUREMENT_NOISE,
            num_segments=num_segments,
        )
        logging.info(f"Results saved to {METRICS_NPZ.name}")

        total_days = len(directional_hits)
        total_hits = sum(directional_hits)
        dir_acc_pct = (total_hits / total_days) * 100 if total_days else 0.0

        print("\n--- Median Metrics for Kalman-v1 Fine-Tuned TimesFM on SSMI (post-2018 test) ---")
        print(f"Median RMSE:          {np.median(rmse_list):.4f}")
        print(f"Median MAPE:          {np.median(mape_list):.4f}%")
        print(f"Median Pearson R2:    {np.median(pearson_list):.4f}")
        print(f"Directional Accuracy: {total_hits}/{total_days} days ({dir_acc_pct:.2f}%)")

    except Exception:
        logging.error("An error occurred.", exc_info=True)
        print(f"An error occurred. Check {METRICS_LOG.name} for details.")
        try:
            np.savez_compressed(
                SCRIPT_DIR / "partial_TimesFM_SSMI_FineTuned_Decomposed_Kalman_v1_Metrics.npz",
                rmse=np.array(rmse_list),
                mape=np.array(mape_list),
                pearson_coefficients=np.array(pearson_list),
                directional_hits=np.array(directional_hits),
            )
        except Exception:
            logging.error("Failed to save partial results.", exc_info=True)
    finally:
        logging.info("Forecasting run completed.")


if __name__ == '__main__':
    main()
