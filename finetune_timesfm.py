"""Fine-tune TimesFM 1.0 PyTorch on SSMI daily Adj Close.

CPU-only. The transformer backbone is frozen; only the output head
(``horizon_ff_layer`` on ``PatchedTimeSeriesDecoder``) is trained.

Run from the project root:

    python finetune_timesfm.py

After 10 batches the script prints an estimated wall-clock for the full run so
you can Ctrl+C early if it's impractical.
"""
from __future__ import annotations

import argparse
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
DATA_CSV = SCRIPT_DIR / "DataSets" / "SSMI cleaned" / "SSMI_cleaned.csv"

CONTEXT_LEN = 120          # supervised window context (as specified)
MODEL_CTX_LEN = 128        # model-facing context; left-pad 120 -> 128 (multiple of 32)
HORIZON = 30               # supervised horizon
STRIDE = 30

PATCH_LEN = 32             # TimesFM 1.0 input patch length
MODEL_OUTPUT_PATCH = 128   # TimesFM 1.0 output patch length (model internal)

TRAIN_OBS = 6142           # chronological 80/20 split (first 6142 train, remainder val)
BATCH_SIZE = 4
LR = 1e-4
MAX_EPOCHS = 3
PATIENCE = 1               # stop after 1 non-improving val epoch
FREQ_ID = 0                # matches Filipp's zero-shot baseline (Google_time_series_zeroShot.ipynb)
SEED = 42

LOG_FILE = SCRIPT_DIR / "finetune_log.txt"
CKPT_EPOCH_FMT = "timesfm_ssmi_finetuned_epoch{}.pt"
CKPT_BEST = SCRIPT_DIR / "timesfm_ssmi_finetuned_best.pt"


def set_cpu_threads() -> int:
    try:
        import psutil
        n = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1
    except ImportError:
        n = os.cpu_count() or 1
    torch.set_num_threads(int(n))
    return int(n)


class SSMIWindows(Dataset):
    """Sliding windows: context (120) -> horizon (30), stride 30.

    Returns left-padded context of length MODEL_CTX_LEN and a matching padding
    mask (1 where padded, 0 where real) so the model sees a 32-multiple input.
    """

    def __init__(self, series: np.ndarray, ctx: int, hor: int, stride: int):
        assert series.ndim == 1
        self.series = series.astype(np.float32)
        self.ctx = ctx
        self.hor = hor
        last_start = len(series) - ctx - hor
        if last_start < 0:
            self.starts: list[int] = []
        else:
            self.starts = list(range(0, last_start + 1, stride))

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
    """Freeze everything except ``horizon_ff_layer`` (the output head)."""
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
    """Mean forecast from the last patch, first ``hor`` steps.

    ``out`` shape: [B, N_patches, MODEL_OUTPUT_PATCH, 1 + n_quantiles]. Index 0
    of the last axis is the mean forecast.
    """
    return out[:, -1, :hor, 0]


def log_line(fh, msg: str) -> None:
    fh.write(msg + "\n")
    fh.flush()


def run(smoke_test: bool = False) -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    n_threads = set_cpu_threads()

    df = pd.read_csv(DATA_CSV)
    series = df["Adj Close"].to_numpy()
    assert len(series) >= TRAIN_OBS + CONTEXT_LEN + HORIZON, (
        f"Series too short: {len(series)}"
    )
    train_series = series[:TRAIN_OBS]
    val_series = series[TRAIN_OBS:]

    train_ds = SSMIWindows(train_series, CONTEXT_LEN, HORIZON, STRIDE)
    val_ds = SSMIWindows(val_series, CONTEXT_LEN, HORIZON, STRIDE)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=False
    )

    print("Loading TimesFM 1.0 200M (PyTorch) from Hugging Face...")
    tfm = load_timesfm()
    model: nn.Module = tfm._model
    model.to("cpu")

    trainable, total = freeze_backbone(model)
    pct = 100.0 * trainable / total
    print(f"Trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")

    params_to_train = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params_to_train, lr=LR)
    loss_fn = nn.MSELoss()

    if smoke_test:
        model.train()
        ctx, pad, tgt = next(iter(train_loader))
        freq = torch.full((ctx.shape[0], 1), FREQ_ID, dtype=torch.long)
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        out = model(ctx, pad, freq)
        pred = horizon_predictions(out, HORIZON)
        loss = loss_fn(pred, tgt)
        loss.backward()
        optimizer.step()
        dt = time.perf_counter() - t0
        print(
            f"[smoke-test] OK: first batch forward+backward+step in {dt:.2f}s "
            f"| loss={loss.item():.6f} | batch_size={ctx.shape[0]}"
        )
        print("[smoke-test] no checkpoints saved, no log file written; exiting.")
        return

    log_fh = open(LOG_FILE, "w", encoding="utf-8")
    log_line(
        log_fh,
        f"CPU threads: {n_threads} | train windows: {len(train_ds)}"
        f" | val windows: {len(val_ds)} | trainable/total: {trainable}/{total}"
        f" ({pct:.2f}%)",
    )

    best_val = float("inf")
    epochs_since_improve = 0
    batch_times: list[float] = []
    train_mean = float("nan")
    val_mean = float("nan")

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses: list[float] = []
        for bi, (ctx, pad, tgt) in enumerate(train_loader):
            t0 = time.perf_counter()
            freq = torch.full((ctx.shape[0], 1), FREQ_ID, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            out = model(ctx, pad, freq)
            pred = horizon_predictions(out, HORIZON)
            loss = loss_fn(pred, tgt)
            loss.backward()
            optimizer.step()
            dt = time.perf_counter() - t0
            batch_times.append(dt)
            train_losses.append(loss.item())

            log_line(
                log_fh,
                f"epoch={epoch} split=train batch={bi} loss={loss.item():.6f} sec={dt:.2f}",
            )

            if len(batch_times) == 10:
                avg = sum(batch_times) / 10.0
                total_batches = len(train_loader) * MAX_EPOCHS
                remaining = (total_batches - (bi + 1)) * avg
                print(
                    f"[timing] avg/batch over first 10: {avg:.2f}s "
                    f"| est. remaining full run: {remaining/60:.1f} min "
                    f"({remaining/3600:.2f} h). Abort now if too slow."
                )
                log_line(
                    log_fh,
                    f"timing avg_batch_sec={avg:.2f} est_remaining_sec={remaining:.0f}",
                )

            if (bi + 1) % 50 == 0:
                recent = float(np.mean(train_losses[-50:]))
                print(
                    f"epoch {epoch} batch {bi+1}/{len(train_loader)} "
                    f"train_loss(last 50)={recent:.6f}"
                )

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for bi, (ctx, pad, tgt) in enumerate(val_loader):
                freq = torch.full((ctx.shape[0], 1), FREQ_ID, dtype=torch.long)
                out = model(ctx, pad, freq)
                pred = horizon_predictions(out, HORIZON)
                loss = loss_fn(pred, tgt)
                val_losses.append(loss.item())
                log_line(
                    log_fh,
                    f"epoch={epoch} split=val batch={bi} loss={loss.item():.6f}",
                )

        train_mean = float(np.mean(train_losses)) if train_losses else float("nan")
        val_mean = float(np.mean(val_losses)) if val_losses else float("nan")
        print(f"epoch {epoch}: train={train_mean:.6f}  val={val_mean:.6f}")
        log_line(
            log_fh,
            f"epoch={epoch} summary train={train_mean:.6f} val={val_mean:.6f}",
        )

        ep_path = SCRIPT_DIR / CKPT_EPOCH_FMT.format(epoch)
        torch.save(model.state_dict(), ep_path)
        print(f"saved {ep_path.name}")

        if val_mean < best_val - 1e-9:
            best_val = val_mean
            epochs_since_improve = 0
            torch.save(model.state_dict(), CKPT_BEST)
            print(f"new best val={val_mean:.6f} -> saved {CKPT_BEST.name}")
            log_line(log_fh, f"epoch={epoch} new_best_val={val_mean:.6f}")
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= PATIENCE:
                print(
                    f"early stop at epoch {epoch} "
                    f"(no val improvement for {epochs_since_improve} epoch(s))"
                )
                log_line(log_fh, f"epoch={epoch} early_stop")
                break

    print(
        f"\nFINAL: best_val={best_val:.6f}  last_train={train_mean:.6f}  "
        f"last_val={val_mean:.6f}"
    )
    log_fh.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one training batch end-to-end, print loss, exit (no checkpoints, no log).",
    )
    args = parser.parse_args()
    run(smoke_test=args.smoke_test)
