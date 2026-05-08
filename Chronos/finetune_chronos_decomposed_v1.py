"""Fine-tune two Chronos models on HP-decomposed SSMI (+ optional extra tickers).

Pipeline:
1) Load all training series up to TRAIN_END_DATE.
2) HP-filter each series into low-pass and high-pass components.
3) Fine-tune one Chronos model on low-pass panel, one on high-pass panel.

Defaults mirror the TimeFM setup in this repo.
"""
from __future__ import annotations

import argparse
import os
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gluonts.dataset.pandas import PandasDataset
from gluonts.itertools import Filter
from statsmodels.tsa.filters.hp_filter import hpfilter
from transformers import Trainer, TrainingArguments, set_seed

from chronos.scripts.training.train import (
    ChronosDataset,
    has_enough_observations,
    load_model,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_DATA_DIR = PROJECT_ROOT / "DataSets" / "trimmed"
DEFAULT_TARGET_TICKER = "SSMI"
DEFAULT_TRAIN_END_DATE = "2017-12-31"


def set_cpu_threads() -> int:
    try:
        import psutil

        n = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1
    except ImportError:
        n = os.cpu_count() or 1
    torch.set_num_threads(int(n))
    return int(n)


def load_one_series(csv_path: Path, train_end_date: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    df = df[df["Date"] <= pd.Timestamp(train_end_date)].copy()
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "target"])

    if df["Adj Close"].isna().any():
        df["Adj Close"] = df["Adj Close"].ffill().bfill()

    out = df[["Date", "Adj Close"]].rename(columns={"Date": "timestamp", "Adj Close": "target"})
    out["target"] = out["target"].astype(np.float32)
    return out


def hp_decompose(values: np.ndarray, lamb: float) -> tuple[np.ndarray, np.ndarray]:
    cycle, trend = hpfilter(values, lamb=lamb)
    return np.asarray(trend, dtype=np.float32), np.asarray(cycle, dtype=np.float32)


def build_component_long_df(
    data_dir: Path,
    target_ticker: str,
    train_end_date: str,
    hp_lambda: float,
    include_extra_tickers: bool,
    component: str,
) -> pd.DataFrame:
    csv_paths = sorted(data_dir.glob("*.csv"))
    selected_paths: list[Path] = []

    for p in csv_paths:
        ticker = p.stem
        if ticker == target_ticker or include_extra_tickers:
            selected_paths.append(p)

    rows: list[pd.DataFrame] = []
    for p in selected_paths:
        ticker = p.stem
        one = load_one_series(p, train_end_date=train_end_date)
        if one.empty or len(one) < 300:
            continue
        low, high = hp_decompose(one["target"].to_numpy(), lamb=hp_lambda)
        comp = low if component == "low" else high
        comp_df = pd.DataFrame(
            {"item_id": ticker, "timestamp": one["timestamp"].values, "target": comp}
        )
        rows.append(comp_df)

    if not rows:
        raise RuntimeError("No usable time series found for training.")
    return pd.concat(rows, ignore_index=True)


def build_chronos_dataset(
    long_df: pd.DataFrame,
    context_length: int,
    prediction_length: int,
    min_past: int,
    mode: str = "training",
) -> ChronosDataset:
    ds = PandasDataset.from_long_dataframe(
        long_df,
        item_id="item_id",
        timestamp="timestamp",
        target="target",
        freq="B",
    )
    ds = Filter(
        partial(
            has_enough_observations,
            min_length=max(context_length + prediction_length, min_past),
        ),
        ds,
    )

    return ChronosDataset(
        datasets=[ds],
        probabilities=[1.0],
        tokenizer_class="MeanScaleUniformBins",
        tokenizer_kwargs={"low_limit": -15.0, "high_limit": 15.0},
        context_length=context_length,
        prediction_length=prediction_length,
        min_past=min_past,
        model_type="seq2seq",
        mode=mode,
    )


def finetune_component(
    component: str,
    train_df: pd.DataFrame,
    model_id: str,
    out_dir: Path,
    context_length: int,
    prediction_length: int,
    min_past: int,
    lr: float,
    batch_size: int,
    max_steps: int,
    save_steps: int,
    log_steps: int,
) -> Path:
    train_dataset = build_chronos_dataset(
        long_df=train_df,
        context_length=context_length,
        prediction_length=prediction_length,
        min_past=min_past,
        mode="training",
    )

    model = load_model(
        model_id=model_id,
        model_type="seq2seq",
        random_init=False,
        tie_embeddings=False,
        resume_from_checkpoint=None,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    train_args = TrainingArguments(
        output_dir=str(out_dir),
        overwrite_output_dir=True,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        max_steps=max_steps,
        logging_steps=log_steps,
        save_steps=save_steps,
        save_total_limit=2,
        dataloader_num_workers=2,
        report_to="none",
        remove_unused_columns=False,
        bf16=torch.cuda.is_available(),
        fp16=False,
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
    )
    print(f"\n=== Fine-tuning Chronos ({component}) ===")
    trainer.train()
    trainer.save_model(str(out_dir))
    print(f"Saved {component} model -> {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--target_ticker", type=str, default=DEFAULT_TARGET_TICKER)
    parser.add_argument("--train_end_date", type=str, default=DEFAULT_TRAIN_END_DATE)
    parser.add_argument("--include_extra_tickers", action="store_true")
    parser.add_argument("--model_id", type=str, default="amazon/chronos-t5-mini")
    parser.add_argument("--context_length", type=int, default=120)
    parser.add_argument("--prediction_length", type=int, default=30)
    parser.add_argument("--min_past", type=int, default=64)
    parser.add_argument("--hp_lambda", type=float, default=129600.0)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--log_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_root", type=Path, default=SCRIPT_DIR / "checkpoints")
    args = parser.parse_args()

    set_seed(args.seed)
    n_threads = set_cpu_threads()
    print(f"CPU threads: {n_threads}")
    print(f"Data dir: {args.data_dir}")
    print(f"Target ticker: {args.target_ticker}")
    print(f"Train end date: {args.train_end_date}")
    print(f"Include extra tickers: {args.include_extra_tickers}")

    low_df = build_component_long_df(
        data_dir=args.data_dir,
        target_ticker=args.target_ticker,
        train_end_date=args.train_end_date,
        hp_lambda=args.hp_lambda,
        include_extra_tickers=args.include_extra_tickers,
        component="low",
    )
    high_df = build_component_long_df(
        data_dir=args.data_dir,
        target_ticker=args.target_ticker,
        train_end_date=args.train_end_date,
        hp_lambda=args.hp_lambda,
        include_extra_tickers=args.include_extra_tickers,
        component="high",
    )
    print(f"Low-pass rows: {len(low_df):,}")
    print(f"High-pass rows: {len(high_df):,}")

    run_name = f"{args.target_ticker}_end_{args.train_end_date.replace('-', '')}"
    low_out = args.output_root / f"chronos_{run_name}_low"
    high_out = args.output_root / f"chronos_{run_name}_high"

    finetune_component(
        component="low",
        train_df=low_df,
        model_id=args.model_id,
        out_dir=low_out,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        min_past=args.min_past,
        lr=args.learning_rate,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        log_steps=args.log_steps,
    )
    finetune_component(
        component="high",
        train_df=high_df,
        model_id=args.model_id,
        out_dir=high_out,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        min_past=args.min_past,
        lr=args.learning_rate,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        log_steps=args.log_steps,
    )
    print("\nDone.")
    print(f"Low model:  {low_out}")
    print(f"High model: {high_out}")


if __name__ == "__main__":
    main()

