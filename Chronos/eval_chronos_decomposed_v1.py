"""Evaluate decomposed Chronos fine-tuned models on SSMI test period.

Flow:
1) Load SSMI series.
2) HP-filter into low/high.
3) Forecast both components with their fine-tuned Chronos models.
4) Reconstruct final forecast = low + high.
5) Log per-segment and aggregate metrics.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline
from scipy.stats import pearsonr
from statsmodels.tsa.filters.hp_filter import hpfilter


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_DATA_CSV = PROJECT_ROOT / "DataSets" / "trimmed" / "SSMI.csv"
DEFAULT_TRAIN_END_DATE = "2017-12-31"


def configure_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("chronos_decomposed_eval")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def load_ssmi(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    if df["Adj Close"].isna().any():
        df["Adj Close"] = df["Adj Close"].ffill().bfill()
    return df[["Date", "Adj Close"]]


def hp_decompose(values: np.ndarray, lamb: float) -> tuple[np.ndarray, np.ndarray]:
    cycle, trend = hpfilter(values, lamb=lamb)
    return np.asarray(trend, dtype=np.float32), np.asarray(cycle, dtype=np.float32)


def rolling_segments(
    values: np.ndarray, context_length: int, horizon: int, stride: int
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    out: list[tuple[int, np.ndarray, np.ndarray]] = []
    last_start = len(values) - context_length - horizon
    if last_start < 0:
        return out
    for s in range(0, last_start + 1, stride):
        ctx = values[s : s + context_length]
        tgt = values[s + context_length : s + context_length + horizon]
        out.append((s, ctx, tgt))
    return out


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), 1e-8)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray, last_context: float) -> float:
    true_diff = np.diff(np.concatenate([[last_context], y_true]))
    pred_diff = np.diff(np.concatenate([[last_context], y_pred]))
    return float(np.mean(np.sign(true_diff) == np.sign(pred_diff)) * 100.0)


def r2_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return 0.0
    try:
        r, _ = pearsonr(y_true, y_pred)
        return float(r**2)
    except Exception:
        return 0.0


def predict_one_segment(
    pipeline: ChronosPipeline, ctx: np.ndarray, horizon: int, quantile: float = 0.5
) -> np.ndarray:
    ctx_t = torch.tensor(ctx, dtype=torch.float32)
    samples = pipeline.predict(ctx_t, prediction_length=horizon)  # [1, n_samples, horizon]
    return np.quantile(samples[0].cpu().numpy(), quantile, axis=0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=Path, default=DEFAULT_DATA_CSV)
    parser.add_argument("--train_end_date", type=str, default=DEFAULT_TRAIN_END_DATE)
    parser.add_argument("--context_length", type=int, default=120)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--hp_lambda", type=float, default=129600.0)
    parser.add_argument("--quantile", type=float, default=0.5)
    parser.add_argument("--low_model_dir", type=Path, required=True)
    parser.add_argument("--high_model_dir", type=Path, required=True)
    parser.add_argument("--log_file", type=Path, default=SCRIPT_DIR / "ChronosDecomposed_SSMI_Metrics.log")
    parser.add_argument("--npz_file", type=Path, default=SCRIPT_DIR / "ChronosDecomposed_SSMI_Metrics.npz")
    args = parser.parse_args()

    logger = configure_logger(args.log_file)
    logger.info("Loading SSMI from %s", args.data_csv)

    df = load_ssmi(args.data_csv)
    split_ts = pd.Timestamp(args.train_end_date)

    train_mask = df["Date"] <= split_ts
    test_mask = df["Date"] > split_ts
    test_df = df[test_mask].copy()

    logger.info("Total SSMI samples loaded: %d", len(df))
    logger.info("Train end date: %s", split_ts.date())
    logger.info("Test samples: %d", len(test_df))

    if len(test_df) < args.context_length + args.horizon:
        raise RuntimeError("Not enough test data for the requested context/horizon.")

    # Use train tail + test so first test segment has full context.
    start_idx = max(0, int(np.where(train_mask.values)[0][-1]) - args.context_length + 1)
    eval_values = df["Adj Close"].to_numpy(dtype=np.float32)[start_idx:]
    eval_dates = df["Date"].to_numpy()[start_idx:]

    low_all, high_all = hp_decompose(eval_values, lamb=args.hp_lambda)
    seg_low = rolling_segments(low_all, args.context_length, args.horizon, args.stride)
    seg_high = rolling_segments(high_all, args.context_length, args.horizon, args.stride)
    seg_raw = rolling_segments(eval_values, args.context_length, args.horizon, args.stride)

    # Keep only segments whose target starts after train_end_date.
    kept = []
    for i, ((s0, c_low, t_low), (_s1, c_high, t_high), (_s2, c_raw, t_raw)) in enumerate(
        zip(seg_low, seg_high, seg_raw)
    ):
        target_start_date = pd.Timestamp(eval_dates[s0 + args.context_length])
        if target_start_date > split_ts:
            kept.append((s0, c_low, t_low, c_high, t_high, c_raw, t_raw))

    logger.info("Segments to evaluate: %d", len(kept))

    device_map = "cuda" if torch.cuda.is_available() else "mps"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    low_pipe = ChronosPipeline.from_pretrained(
        str(args.low_model_dir),
        device_map=device_map,
        torch_dtype=dtype,
    )
    high_pipe = ChronosPipeline.from_pretrained(
        str(args.high_model_dir),
        device_map=device_map,
        torch_dtype=dtype,
    )
    logger.info("Loaded low/high Chronos fine-tuned models.")

    rmse_list, mape_list, r2_list, da_list = [], [], [], []
    preds_all, trues_all = [], []

    for i, (_s0, c_low, _t_low, c_high, _t_high, c_raw, t_raw) in enumerate(kept, start=1):
        pred_low = predict_one_segment(low_pipe, c_low, args.horizon, args.quantile)
        pred_high = predict_one_segment(high_pipe, c_high, args.horizon, args.quantile)
        pred_final = pred_low + pred_high

        rmse = float(np.sqrt(np.mean((pred_final - t_raw) ** 2)))
        mape = safe_mape(t_raw, pred_final)
        r2 = r2_pearson(t_raw, pred_final)
        da = directional_accuracy(t_raw, pred_final, last_context=float(c_raw[-1]))

        rmse_list.append(rmse)
        mape_list.append(mape)
        r2_list.append(r2)
        da_list.append(da)
        preds_all.append(pred_final)
        trues_all.append(t_raw)

        logger.info(
            "Segment %d/%d: RMSE=%.4f, MAPE=%.4f%%, Pearson R²=%.4f, DirAcc=%.1f%%",
            i,
            len(kept),
            rmse,
            mape,
            r2,
            da,
        )

    logger.info("==== Aggregate Results ====")
    logger.info("Mean RMSE: %.4f", float(np.mean(rmse_list)))
    logger.info("Mean MAPE: %.4f%%", float(np.mean(mape_list)))
    logger.info("Mean Pearson R²: %.4f", float(np.mean(r2_list)))
    logger.info("Mean DirAcc: %.2f%%", float(np.mean(da_list)))

    np.savez(
        args.npz_file,
        rmse=np.asarray(rmse_list, dtype=np.float32),
        mape=np.asarray(mape_list, dtype=np.float32),
        r2=np.asarray(r2_list, dtype=np.float32),
        directional_accuracy=np.asarray(da_list, dtype=np.float32),
        preds=np.asarray(preds_all, dtype=np.float32),
        trues=np.asarray(trues_all, dtype=np.float32),
    )
    logger.info("Saved metrics to %s", args.npz_file)


if __name__ == "__main__":
    main()

