"""Fine-tune TWO Chronos models (low-pass / high-pass) on trimmed SSMI.

Train range: 2001-10-01 -> 2017-12-31 (from trimmed SSMI)
Saves checkpoints next to this script:
  - chronos_ssmi_low_v1.pt
  - chronos_ssmi_high_v1.pt

Notes:
- Uses Chronos tokenizer transforms directly, so training labels are in Chronos token space.
- Keeps pipeline/settings close to existing TimesFM decomposed setup.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from statsmodels.tsa.filters.hp_filter import hpfilter
from torch.utils.data import DataLoader, Dataset

from chronos import ChronosPipeline


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_CSV = PROJECT_ROOT / "DataSets" / "trimmed" / "SSMI.csv"
TRAIN_END_DATE = "2017-12-31"

BASE_MODEL_ID = "amazon/chronos-t5-base"

CONTEXT_LEN = 120
HORIZON = 30
STRIDE = 30

BATCH_SIZE = 8
LR = 5e-5
WEIGHT_DECAY = 0.01
MAX_EPOCHS = 3
SEED = 42
VAL_FRACTION = 0.10
HP_LAMBDA = 129600

CHECKPOINT_LOW = SCRIPT_DIR / "chronos_ssmi_low_v1.pt"
CHECKPOINT_HIGH = SCRIPT_DIR / "chronos_ssmi_high_v1.pt"


def set_cpu_threads() -> int:
    try:
        import psutil

        n = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1
    except ImportError:
        n = os.cpu_count() or 1
    torch.set_num_threads(int(n))
    return int(n)


class ComponentWindows(Dataset):
    """Simple sliding windows returning raw context/target values."""

    def __init__(self, series: np.ndarray, ctx: int, hor: int, stride: int):
        assert series.ndim == 1
        self.series = series.astype(np.float32)
        self.ctx = ctx
        self.hor = hor
        last_start = len(series) - ctx - hor
        self.starts = list(range(0, last_start + 1, stride)) if last_start >= 0 else []

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, i: int):
        s = self.starts[i]
        ctx = self.series[s : s + self.ctx]
        tgt = self.series[s + self.ctx : s + self.ctx + self.hor]
        return torch.from_numpy(ctx), torch.from_numpy(tgt)


def collate_windows(batch):
    ctxs = torch.stack([x[0] for x in batch], dim=0)
    tgts = torch.stack([x[1] for x in batch], dim=0)
    return ctxs, tgts


def load_train_series() -> np.ndarray:
    df = pd.read_csv(DATA_CSV, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    mask = df["Date"] <= pd.Timestamp(TRAIN_END_DATE)
    train_df = df[mask].copy()

    nan_count = int(train_df["Adj Close"].isna().sum())
    if nan_count > 0:
        train_df["Adj Close"] = train_df["Adj Close"].ffill().bfill()
        print(f"Forward-filled {nan_count} NaN values in training data")

    y = train_df["Adj Close"].to_numpy().astype(np.float32)
    print(
        f"Train range: {train_df['Date'].iloc[0].date()} -> "
        f"{train_df['Date'].iloc[-1].date()} ({len(y)} days)"
    )
    return y


def hp_decompose_full(series: np.ndarray, lamb: float = HP_LAMBDA):
    cycle, trend = hpfilter(series, lamb=lamb)
    return np.asarray(trend, dtype=np.float32), np.asarray(cycle, dtype=np.float32)


def build_batch_tokens(tokenizer, ctx: torch.Tensor, tgt: torch.Tensor, device: torch.device):
    input_ids, attention_mask, tok_state = tokenizer.context_input_transform(ctx)
    model_pred_len = int(tokenizer.config.prediction_length)
    target_len = int(tgt.shape[-1])
    if target_len < model_pred_len:
        pad_width = model_pred_len - target_len
        pad = torch.full((tgt.shape[0], pad_width), float("nan"), dtype=tgt.dtype)
        tgt_for_labels = torch.cat([tgt, pad], dim=-1)
    elif target_len > model_pred_len:
        tgt_for_labels = tgt[:, :model_pred_len]
    else:
        tgt_for_labels = tgt

    label_ids, label_mask = tokenizer.label_input_transform(tgt_for_labels, tok_state)

    labels = label_ids.clone()
    labels[~label_mask] = -100

    return (
        input_ids.to(device),
        attention_mask.to(device),
        labels.to(device),
    )


def finetune_component(name: str, series: np.ndarray, ckpt_path: Path) -> None:
    print(f"\n{'=' * 60}\nFine-tuning Chronos model_{name} (v1)\n{'=' * 60}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    n = len(series)
    val_start = int(n * (1 - VAL_FRACTION))
    train_series = series[:val_start]
    val_series = series[val_start:]

    train_ds = ComponentWindows(train_series, CONTEXT_LEN, HORIZON, STRIDE)
    val_ds = ComponentWindows(val_series, CONTEXT_LEN, HORIZON, STRIDE)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_windows,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_windows,
    )
    print(f"Train windows: {len(train_ds)}, Val windows: {len(val_ds)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    pipeline = ChronosPipeline.from_pretrained(
        BASE_MODEL_ID,
        device_map=None,
        dtype=dtype,
    )
    tokenizer = pipeline.tokenizer
    model = pipeline.model.model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val = float("inf")
    best_state = None

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses = []
        t0_epoch = time.perf_counter()

        for ctx, tgt in train_loader:
            in_ids, attn, labels = build_batch_tokens(tokenizer, ctx, tgt, device)
            optimizer.zero_grad(set_to_none=True)
            out = model(input_ids=in_ids, attention_mask=attn, labels=labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for ctx, tgt in val_loader:
                in_ids, attn, labels = build_batch_tokens(tokenizer, ctx, tgt, device)
                out = model(input_ids=in_ids, attention_mask=attn, labels=labels)
                val_losses.append(float(out.loss.detach().cpu().item()))

        tr = float(np.mean(train_losses)) if train_losses else float("nan")
        va = float(np.mean(val_losses)) if val_losses else float("nan")
        dt = time.perf_counter() - t0_epoch
        print(f"Epoch {epoch}/{MAX_EPOCHS}  train={tr:.6f}  val={va:.6f}  ({dt:.1f}s)")

        if va < best_val:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(best_state, ckpt_path)
    print(f"Saved best model_{name} -> {ckpt_path.name} (val loss {best_val:.6f})")


def main() -> None:
    n_threads = set_cpu_threads()
    print(f"CPU threads: {n_threads}")
    print(f"Data CSV: {DATA_CSV}")
    print(f"Base model: {BASE_MODEL_ID}")
    print(f"Low checkpoint: {CHECKPOINT_LOW}")
    print(f"High checkpoint: {CHECKPOINT_HIGH}")

    y = load_train_series()
    train_low, train_high = hp_decompose_full(y)

    finetune_component("low", train_low, CHECKPOINT_LOW)
    finetune_component("high", train_high, CHECKPOINT_HIGH)


if __name__ == "__main__":
    main()
