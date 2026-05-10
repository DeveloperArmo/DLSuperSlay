"""Fine-tune TWO Chronos models (low-pass / high-pass) on trimmed SSMI.

Usage:
  python finetune_chronos_decomposed_v1.py --filter hp
  python finetune_chronos_decomposed_v1.py --filter ma
  python finetune_chronos_decomposed_v1.py --filter kalman
  python finetune_chronos_decomposed_v1.py --filter butterworth

Decompositions:
  hp          - Hodrick-Prescott filter (lamb=10000)
  ma          - Moving Average (window=30)
  kalman      - Kalman filter (local linear trend)
  butterworth - Butterworth low-pass filter (order=4, cutoff=0.05)

Train range: 2001-10-01 -> 2017-12-31
Saves checkpoints:
  chronos_ssmi_low_{filter}.pt
  chronos_ssmi_high_{filter}.pt
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, filtfilt
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.statespace.structural import UnobservedComponents
from torch.utils.data import DataLoader, Dataset

from chronos import ChronosPipeline


SCRIPT_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT   = SCRIPT_DIR.parent
DATA_CSV       = PROJECT_ROOT / "DataSets" / "trimmed" / "SSMI.csv"
TRAIN_END_DATE = "2017-12-31"
BASE_MODEL_ID  = "amazon/chronos-t5-base"

CONTEXT_LEN  = 180
HORIZON      = 15
STRIDE       = 30
BATCH_SIZE   = 8
LR           = 5e-5
WEIGHT_DECAY = 0.01
MAX_EPOCHS   = 3
SEED         = 42
VAL_FRACTION = 0.10

HP_LAMBDA         = 10000
MA_WINDOW         = 30
BW_ORDER          = 4
BW_CUTOFF         = 0.05


# ── decompositions ────────────────────────────────────────────────────────────

def hp_decompose(series: np.ndarray):
    cycle, trend = hpfilter(series, lamb=HP_LAMBDA)
    return np.asarray(trend, dtype=np.float32), np.asarray(cycle, dtype=np.float32)


def ma_decompose(series: np.ndarray):
    s   = pd.Series(series)
    low = s.rolling(MA_WINDOW, min_periods=1).mean().values.astype(np.float32)
    return low, (series - low).astype(np.float32)


def kalman_decompose(series: np.ndarray):
    print("  Fitting Kalman filter...")
    model  = UnobservedComponents(series, level="local linear trend")
    result = model.fit(disp=False)
    low    = np.asarray(result.level.filtered, dtype=np.float32)
    return low, (series - low).astype(np.float32)


def butterworth_decompose(series: np.ndarray):
    b, a = butter(BW_ORDER, BW_CUTOFF, btype="low", analog=False)
    low  = filtfilt(b, a, series).astype(np.float32)
    return low, (series - low).astype(np.float32)


DECOMPOSE = {
    "hp":          hp_decompose,
    "ma":          ma_decompose,
    "kalman":      kalman_decompose,
    "butterworth": butterworth_decompose,
}


# ── dataset ───────────────────────────────────────────────────────────────────

class ComponentWindows(Dataset):
    def __init__(self, series: np.ndarray, ctx: int, hor: int, stride: int):
        assert series.ndim == 1
        self.series = series.astype(np.float32)
        self.ctx    = ctx
        self.hor    = hor
        last_start  = len(series) - ctx - hor
        self.starts = list(range(0, last_start + 1, stride)) if last_start >= 0 else []

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = self.starts[i]
        return (torch.from_numpy(self.series[s:s + self.ctx]),
                torch.from_numpy(self.series[s + self.ctx:s + self.ctx + self.hor]))


def collate_windows(batch):
    return torch.stack([x[0] for x in batch]), torch.stack([x[1] for x in batch])


# ── data loading ──────────────────────────────────────────────────────────────

def load_train_series() -> np.ndarray:
    df = pd.read_csv(DATA_CSV, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    mask     = df["Date"] <= pd.Timestamp(TRAIN_END_DATE)
    train_df = df[mask].copy()
    nan_count = int(train_df["Adj Close"].isna().sum())
    if nan_count:
        train_df["Adj Close"] = train_df["Adj Close"].ffill().bfill()
        print(f"Forward-filled {nan_count} NaN values in training data")
    y = train_df["Adj Close"].to_numpy().astype(np.float32)
    print(f"Train range: {train_df['Date'].iloc[0].date()} -> {train_df['Date'].iloc[-1].date()} ({len(y)} days)")
    return y


# ── tokenizer helpers ─────────────────────────────────────────────────────────

def build_batch_tokens(tokenizer, ctx, tgt, device):
    input_ids, attention_mask, tok_state = tokenizer.context_input_transform(ctx)
    model_pred_len = int(tokenizer.config.prediction_length)
    target_len     = int(tgt.shape[-1])
    if target_len < model_pred_len:
        pad = torch.full((tgt.shape[0], model_pred_len - target_len), float("nan"), dtype=tgt.dtype)
        tgt_for_labels = torch.cat([tgt, pad], dim=-1)
    elif target_len > model_pred_len:
        tgt_for_labels = tgt[:, :model_pred_len]
    else:
        tgt_for_labels = tgt
    label_ids, label_mask = tokenizer.label_input_transform(tgt_for_labels, tok_state)
    labels = label_ids.clone()
    labels[~label_mask] = -100
    return input_ids.to(device), attention_mask.to(device), labels.to(device)


# ── fine-tuning ───────────────────────────────────────────────────────────────

def finetune_component(name: str, series: np.ndarray, ckpt_path: Path) -> None:
    print(f"\n{'=' * 60}\nFine-tuning Chronos model_{name}\n{'=' * 60}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    n         = len(series)
    val_start = int(n * (1 - VAL_FRACTION))
    train_ds  = ComponentWindows(series[:val_start], CONTEXT_LEN, HORIZON, STRIDE)
    val_ds    = ComponentWindows(series[val_start:], CONTEXT_LEN, HORIZON, STRIDE)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=False, collate_fn=collate_windows)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              drop_last=False, collate_fn=collate_windows)
    print(f"Train windows: {len(train_ds)}, Val windows: {len(val_ds)}")

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype    = torch.bfloat16 if device.type == "cuda" else torch.float32
    pipeline = ChronosPipeline.from_pretrained(BASE_MODEL_ID, device_map=None, dtype=dtype)
    tokenizer = pipeline.tokenizer
    model     = pipeline.model.model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val, best_state = float("inf"), None
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses = []
        t0 = time.perf_counter()
        for ctx, tgt in train_loader:
            in_ids, attn, labels = build_batch_tokens(tokenizer, ctx, tgt, device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=in_ids, attention_mask=attn, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for ctx, tgt in val_loader:
                in_ids, attn, labels = build_batch_tokens(tokenizer, ctx, tgt, device)
                val_losses.append(float(model(input_ids=in_ids, attention_mask=attn, labels=labels).loss.detach().cpu()))

        tr = float(np.mean(train_losses)) if train_losses else float("nan")
        va = float(np.mean(val_losses))   if val_losses   else float("nan")
        print(f"Epoch {epoch}/{MAX_EPOCHS}  train={tr:.6f}  val={va:.6f}  ({time.perf_counter()-t0:.1f}s)")
        if va < best_val:
            best_val  = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(best_state, ckpt_path)
    print(f"Saved model_{name} -> {ckpt_path.name}  (val loss {best_val:.6f})")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Chronos with a chosen filter decomposition.")
    parser.add_argument("--filter", choices=["hp", "ma", "kalman", "butterworth"], required=True,
                        help="Decomposition filter: hp | ma | kalman | butterworth")
    args = parser.parse_args()
    filter_name = args.filter

    ckpt_low  = SCRIPT_DIR / f"chronos_ssmi_low_{filter_name}.pt"
    ckpt_high = SCRIPT_DIR / f"chronos_ssmi_high_{filter_name}.pt"

    n_threads = os.cpu_count() or 1
    torch.set_num_threads(n_threads)
    print(f"Filter:       {filter_name.upper()}")
    print(f"CPU threads:  {n_threads}")
    print(f"Data CSV:     {DATA_CSV}")
    print(f"Low  ckpt:    {ckpt_low}")
    print(f"High ckpt:    {ckpt_high}")

    y = load_train_series()
    print(f"Decomposing with {filter_name.upper()}...")
    train_low, train_high = DECOMPOSE[filter_name](y)

    finetune_component("low",  train_low,  ckpt_low)
    finetune_component("high", train_high, ckpt_high)


if __name__ == "__main__":
    main()
