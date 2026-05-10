"""Fine-tune two TimesFM models on Kalman-decomposed SSMI train set (v1 hyperparams).

Mirrors the v1 setup from finetune_timesfm_decomposed.py exactly, only
swapping the HP filter for the scalar Kalman filter that won the
hyperparameter sweep (Hyperparams Sweep/sim_Hyperparams.ipynb).

v1 hyperparams (same as the HP-based v1):
  - Only `horizon_ff_layer` trainable (~2.4% of params)
  - AdamW, lr=1e-4, 3 epochs, MSE loss
  - Batch size 4, 90/10 chronological train/val split
  - No per-window normalization

Decomposition (matches decompose_kalman in sim_Hyperparams.ipynb):
  - Scalar Kalman filter, one-pass forward (causal — no look-ahead)
  - process_noise q = 0.1   (best RMSE/MAPE in the sweep)
  - measurement_noise r = 1.0  (sweep default)
  - low  component = filtered state estimate
  - high component = y_train - low

Train range: 2001-10-01 -> 2017-12-31 (from trimmed SSMI)
Saves checkpoints next to this script:
  - timesfm_ssmi_low_kalman_v1.pt
  - timesfm_ssmi_high_kalman_v1.pt
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import timesfm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # up two levels to project root
DATA_CSV = PROJECT_ROOT / "DataSets" / "trimmed" / "SSMI.csv"
TRAIN_END_DATE = "2017-12-31"

CONTEXT_LEN = 120
MODEL_CTX_LEN = 128
HORIZON = 30
STRIDE = 30

PATCH_LEN = 32
MODEL_OUTPUT_PATCH = 128

BATCH_SIZE = 4
LR = 1e-4
MAX_EPOCHS = 3
FREQ_ID = 0
SEED = 42
VAL_FRACTION = 0.10

# Best Kalman config from the hyperparameter sweep (RMSE/MAPE-optimal)
KALMAN_PROCESS_NOISE = 0.1
KALMAN_MEASUREMENT_NOISE = 1.0

CHECKPOINT_LOW = SCRIPT_DIR / "timesfm_ssmi_low_kalman_v1.pt"
CHECKPOINT_HIGH = SCRIPT_DIR / "timesfm_ssmi_high_kalman_v1.pt"


def set_cpu_threads() -> int:
    try:
        import psutil
        n = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1
    except ImportError:
        n = os.cpu_count() or 1
    torch.set_num_threads(int(n))
    return int(n)


class ComponentWindows(Dataset):
    """Sliding windows over a 1D component series (low-pass or high-pass)."""

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
        pad = MODEL_CTX_LEN - self.ctx
        ctx_padded = np.concatenate([np.zeros(pad, dtype=np.float32), ctx])
        padding = np.concatenate(
            [np.ones(pad, dtype=np.float32), np.zeros(self.ctx, dtype=np.float32)]
        )
        return (
            torch.from_numpy(ctx_padded),
            torch.from_numpy(padding),
            torch.from_numpy(tgt),
        )


def load_timesfm() -> "timesfm.TimesFm":
    hparams = timesfm.TimesFmHparams(
        backend="cpu",
        per_core_batch_size=BATCH_SIZE,
        horizon_len=HORIZON,
        context_len=MODEL_CTX_LEN,
    )
    ckpt = timesfm.TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
    )
    tfm = timesfm.TimesFm(hparams=hparams, checkpoint=ckpt)
    if getattr(tfm, "_model", None) is None:
        tfm.load_from_checkpoint(ckpt)
    return tfm


def freeze_backbone(model: nn.Module) -> tuple[int, int]:
    trainable, total = 0, 0
    for name, p in model.named_parameters():
        total += p.numel()
        if name.startswith("horizon_ff_layer"):
            p.requires_grad = True
            trainable += p.numel()
        else:
            p.requires_grad = False
    return trainable, total


def horizon_predictions(out: torch.Tensor, hor: int) -> torch.Tensor:
    return out[:, -1, :hor, 0]


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


def kalman_decompose_full(
    series: np.ndarray,
    process_noise: float = KALMAN_PROCESS_NOISE,
    measurement_noise: float = KALMAN_MEASUREMENT_NOISE,
):
    """Scalar Kalman filter decomposition (one-pass forward, causal).

    Identical to `decompose_kalman` in Hyperparams Sweep/sim_Hyperparams.ipynb.
    Each output sample depends only on past+current observations, so applying
    this to the full training series produces no look-ahead leakage when later
    sliced into sliding windows.

    State: scalar level x with variance p.
      predict:  p <- p + q
      gain:     k = p / (p + r)
      update:   x <- x + k * (z - x); p <- (1 - k) * p
    """
    y = series.astype(float)
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

    low = low.astype(np.float32)
    high = (y.astype(np.float32) - low).astype(np.float32)
    return low, high


def finetune_component(name: str, series: np.ndarray, ckpt_path: Path) -> None:
    print(f"\n{'=' * 60}\nFine-tuning model_{name} (kalman v1)\n{'=' * 60}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Chronological 90/10 split
    n = len(series)
    val_start = int(n * (1 - VAL_FRACTION))
    train_series = series[:val_start]
    val_series = series[val_start:]

    train_ds = ComponentWindows(train_series, CONTEXT_LEN, HORIZON, STRIDE)
    val_ds = ComponentWindows(val_series, CONTEXT_LEN, HORIZON, STRIDE)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    print(f"Train windows: {len(train_ds)}, Val windows: {len(val_ds)}")

    tfm = load_timesfm()
    model: nn.Module = tfm._model
    model.to("cpu")

    trainable, total = freeze_backbone(model)
    pct = 100.0 * trainable / total
    print(f"Trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")

    params_to_train = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params_to_train, lr=LR)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses = []
        t0_epoch = time.perf_counter()
        for ctx, pad, tgt in train_loader:
            freq = torch.full((ctx.shape[0], 1), FREQ_ID, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            out = model(ctx, pad, freq)
            pred = horizon_predictions(out, HORIZON)
            loss = loss_fn(pred, tgt)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for ctx, pad, tgt in val_loader:
                freq = torch.full((ctx.shape[0], 1), FREQ_ID, dtype=torch.long)
                out = model(ctx, pad, freq)
                pred = horizon_predictions(out, HORIZON)
                val_losses.append(loss_fn(pred, tgt).item())

        tr = float(np.mean(train_losses)) if train_losses else float("nan")
        va = float(np.mean(val_losses)) if val_losses else float("nan")
        dt = time.perf_counter() - t0_epoch
        print(f"Epoch {epoch}/{MAX_EPOCHS}  train={tr:.4f}  val={va:.4f}  ({dt:.1f}s)")

        if va < best_val:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(best_state, ckpt_path)
    print(f"Saved best model_{name} -> {ckpt_path.name} (val loss {best_val:.4f})")


def main() -> None:
    n_threads = set_cpu_threads()
    print(f"CPU threads: {n_threads}")
    print(f"Data CSV: {DATA_CSV}")
    print(f"Low checkpoint: {CHECKPOINT_LOW}")
    print(f"High checkpoint: {CHECKPOINT_HIGH}")

    y_train = load_train_series()
    train_low, train_high = kalman_decompose_full(y_train)
    print(
        f"Kalman-decomposed train (q={KALMAN_PROCESS_NOISE}, r={KALMAN_MEASUREMENT_NOISE}) "
        f"into low + high (len={len(train_low)})"
    )

    finetune_component("low", train_low, CHECKPOINT_LOW)
    finetune_component("high", train_high, CHECKPOINT_HIGH)

    print("\nDone. Checkpoints saved:")
    print(f"  {CHECKPOINT_LOW}")
    print(f"  {CHECKPOINT_HIGH}")


if __name__ == "__main__":
    main()
