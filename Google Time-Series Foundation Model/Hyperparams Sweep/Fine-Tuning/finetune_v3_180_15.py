"""Fine-tune TWO TimesFM models — v3 at context=180, horizon=15, HP lambda=10000.

v3 design vs. v2:
  1. UNFREEZE EVEN MORE LAYERS: input_ff_layer + horizon_ff_layer + last 4
     transformer blocks (layers 16-19) — ~23% of params trainable vs. 13% in v2.
  2. REGULARIZATION: AdamW weight_decay=0.01 + gradient clipping at norm 1.0,
     to keep the larger trainable surface from drifting too far from the
     pretrained weights.
  3. EARLY STOPPING: train up to 5 epochs but stop after `EARLY_STOP_PATIENCE`
     consecutive non-improving val epochs. Best checkpoint is restored.
  4. Learning rate dropped further to 3e-5 (more trainable params, gentler
     steps under regularization).

Per-window z-score normalization carries over from v2 (context and target
both normalized by the context's mean/std; predictions denormalized at eval).
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
from statsmodels.tsa.filters.hp_filter import hpfilter

import timesfm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent  # up 3 levels: Fine-Tuning -> Hyperparams Sweep -> Google.../ -> root
DATA_CSV = PROJECT_ROOT / "DataSets" / "trimmed" / "SSMI.csv"
TRAIN_END_DATE = "2017-12-31"

CONTEXT_LEN = 180
MODEL_CTX_LEN = 192   # next multiple of 32 above 180
HORIZON = 15
STRIDE = 30

BATCH_SIZE = 4
LR = 3e-5
MAX_EPOCHS = 5
FREQ_ID = 0
SEED = 42
VAL_FRACTION = 0.10
HP_LAMBDA = 10000  # best from sweep

WEIGHT_DECAY = 0.01
EARLY_STOP_PATIENCE = 2
GRAD_CLIP_NORM = 1.0

TRAINABLE_PREFIXES = (
    "input_ff_layer.",
    "horizon_ff_layer.",
    "stacked_transformer.layers.16.",
    "stacked_transformer.layers.17.",
    "stacked_transformer.layers.18.",
    "stacked_transformer.layers.19.",
)

CHECKPOINT_LOW = SCRIPT_DIR / "timesfm_ssmi_low_v3_180_15.pt"
CHECKPOINT_HIGH = SCRIPT_DIR / "timesfm_ssmi_high_v3_180_15.pt"


def set_cpu_threads() -> int:
    try:
        import psutil
        n = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1
    except ImportError:
        n = os.cpu_count() or 1
    torch.set_num_threads(int(n))
    return int(n)


class ComponentWindows(Dataset):
    """Sliding windows with per-window z-score normalization (v2/v3 design).

    Returns (ctx_padded_norm, padding_mask, target_norm, ctx_mean, ctx_std).
    Context and target are BOTH normalized using the context's mean/std.
    Padding positions in the context are zero (which is the mean in normalized
    space, so they don't distort anything).
    """

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
        raw_ctx = self.series[s : s + self.ctx]
        raw_tgt = self.series[s + self.ctx : s + self.ctx + self.hor]

        ctx_mean = float(raw_ctx.mean())
        ctx_std = float(raw_ctx.std())
        if ctx_std < 1e-6:
            ctx_std = 1.0

        ctx_norm = (raw_ctx - ctx_mean) / ctx_std
        tgt_norm = (raw_tgt - ctx_mean) / ctx_std

        pad = MODEL_CTX_LEN - self.ctx
        ctx_padded = np.concatenate([np.zeros(pad, dtype=np.float32), ctx_norm.astype(np.float32)])
        padding = np.concatenate(
            [np.ones(pad, dtype=np.float32), np.zeros(self.ctx, dtype=np.float32)]
        )

        return (
            torch.from_numpy(ctx_padded),
            torch.from_numpy(padding),
            torch.from_numpy(tgt_norm.astype(np.float32)),
            torch.tensor(ctx_mean, dtype=torch.float32),
            torch.tensor(ctx_std, dtype=torch.float32),
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


def freeze_with_prefixes(model: nn.Module, trainable_prefixes: tuple[str, ...]) -> tuple[int, int, list[str]]:
    trainable, total = 0, 0
    trainable_names: list[str] = []
    for name, p in model.named_parameters():
        total += p.numel()
        is_trainable = any(name.startswith(pref) for pref in trainable_prefixes)
        if is_trainable:
            p.requires_grad = True
            trainable += p.numel()
            trainable_names.append(name)
        else:
            p.requires_grad = False
    return trainable, total, trainable_names


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


def hp_decompose_full(series: np.ndarray, lamb: float = HP_LAMBDA):
    cycle, trend = hpfilter(series, lamb=lamb)
    return np.asarray(trend, dtype=np.float32), np.asarray(cycle, dtype=np.float32)


def finetune_component(name: str, series: np.ndarray, ckpt_path: Path) -> None:
    print(f"\n{'=' * 60}\nFine-tuning model_{name} (v3 / 180-15 / HP=10000 / norm + reg)\n{'=' * 60}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

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

    trainable, total, trainable_names = freeze_with_prefixes(model, TRAINABLE_PREFIXES)
    pct = 100.0 * trainable / total
    print(f"Trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")

    params_to_train = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params_to_train, lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    epochs_since_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses = []
        t0_epoch = time.perf_counter()
        for ctx, pad, tgt, _mean, _std in train_loader:
            freq = torch.full((ctx.shape[0], 1), FREQ_ID, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            out = model(ctx, pad, freq)
            pred = horizon_predictions(out, HORIZON)
            loss = loss_fn(pred, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_to_train, GRAD_CLIP_NORM)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for ctx, pad, tgt, _mean, _std in val_loader:
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
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch} (best was epoch {best_epoch})")
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = best_epoch or MAX_EPOCHS
    torch.save(best_state, ckpt_path)
    print(f"Saved best model_{name} -> {ckpt_path.name} (val loss {best_val:.4f} @ epoch {best_epoch})")


def main() -> None:
    n_threads = set_cpu_threads()
    print(f"CPU threads: {n_threads}")
    print(f"Data CSV: {DATA_CSV}")

    y_train = load_train_series()
    train_low, train_high = hp_decompose_full(y_train)
    print(f"Decomposed train into low-pass + high-pass with HP lambda={HP_LAMBDA} (len={len(train_low)})")

    finetune_component("low", train_low, CHECKPOINT_LOW)
    finetune_component("high", train_high, CHECKPOINT_HIGH)

    print("\nDone. Checkpoints saved:")
    print(f"  {CHECKPOINT_LOW}")
    print(f"  {CHECKPOINT_HIGH}")


if __name__ == "__main__":
    main()
