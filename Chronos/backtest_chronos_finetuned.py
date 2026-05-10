"""Backtesting for Fine-Tuned Chronos (HP filter decomposition) on SSMI post-2018.

Strategy: per-day long/short directional signal via the bt library.
  - +1 (long)  if predicted price > previous price
  - -1 (short) otherwise
  - Signal deduplication: last forecast wins for overlapping windows.
  - Signal offset by -1 day: rebalance at open of the following day.

Benchmark: Buy-and-Hold SSMI.

Outputs (Chronos/plots_finetuned/):
  backtest_signals.png
  backtest_equity_curve.png
  backtest_monthly_heatmap.png
"""
from pathlib import Path

import bt
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from chronos import ChronosPipeline
from matplotlib.patches import Patch
from statsmodels.tsa.filters.hp_filter import hpfilter

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_CSV     = PROJECT_ROOT / "DataSets" / "trimmed" / "SSMI.csv"
PLOTS_DIR    = SCRIPT_DIR / "plots_finetuned"

BASE_MODEL_ID  = "amazon/chronos-t5-base"
CKPT_LOW       = SCRIPT_DIR / "chronos_ssmi_low_v1.pt"
CKPT_HIGH      = SCRIPT_DIR / "chronos_ssmi_high_v1.pt"

TEST_START     = "2018-01-01"
CONTEXT_WINDOW = 180
HORIZON        = 15
STEP           = 30


# ── helpers ───────────────────────────────────────────────────────────────────

HP_LAMBDA = 10000


def hp_decompose(y: np.ndarray):
    cycle, trend = hpfilter(y, lamb=HP_LAMBDA)
    return np.asarray(trend, dtype=np.float32), np.asarray(cycle, dtype=np.float32)


def predict_median(pipeline, context_1d: np.ndarray, horizon: int) -> np.ndarray:
    ctx = torch.tensor(context_1d, dtype=torch.float32)
    with torch.no_grad():
        samples = pipeline.predict(ctx, prediction_length=horizon, num_samples=20)
    return samples[0].median(dim=0).values.cpu().numpy().astype(float)[:horizon]


def load_finetuned(ckpt_path: Path, device: torch.device) -> ChronosPipeline:
    pipeline = ChronosPipeline.from_pretrained(
        BASE_MODEL_ID, device_map=None,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    state = torch.load(str(ckpt_path), map_location=device)
    pipeline.model.model.load_state_dict(state)
    pipeline.model.model.eval()
    return pipeline


# ── signal collection ─────────────────────────────────────────────────────────

def collect_signals(y, dates, ft_low, ft_high):
    total = len(y)
    n_seg = (total - CONTEXT_WINDOW) // STEP

    signal_dates  = []
    signal_values = []

    for seg in range(n_seg):
        s = seg * STEP
        e = s + CONTEXT_WINDOW
        if e + HORIZON > total:
            break

        # HP filter applied to context window only — no look-ahead
        ctx_low, ctx_high = hp_decompose(y[s:e])

        pred = (predict_median(ft_low,  ctx_low,  HORIZON) +
                predict_median(ft_high, ctx_high, HORIZON))

        prev = np.concatenate([[y[e - 1]], pred[:-1]])
        direction = np.where(pred > prev, 1.0, -1.0)

        seg_dates = pd.to_datetime(dates.iloc[e:e + HORIZON].values)
        signal_dates.extend(seg_dates.tolist())
        signal_values.extend(direction.tolist())

        print(f"  Segment {seg + 1}/{n_seg} done")

    return signal_dates, signal_values


# ── bt strategy ───────────────────────────────────────────────────────────────

class ChronosSignalStrategy(bt.Algo):
    """Applies precomputed daily directional signal as long/short weight."""

    def __init__(self, signals):
        self.signals = signals

    def __call__(self, target):
        if target.now in self.signals.index:
            signal = float(self.signals.loc[target.now, "SIGNAL"])
            target.temp["weights"] = {"SSMI": signal}
        return True


# ── plots ──────────────────────────────────────────────────────────────────────

def make_signal_plot(test_df, y, signal_df_raw):
    PLOTS_DIR.mkdir(exist_ok=True)
    plt.rcParams["axes.spines.top"]   = False
    plt.rcParams["axes.spines.right"] = False

    dates       = pd.to_datetime(test_df["Date"])
    long_dates  = signal_df_raw[signal_df_raw["SIGNAL"] ==  1.0].index
    short_dates = signal_df_raw[signal_df_raw["SIGNAL"] == -1.0].index

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, y, color="black", lw=1, label="SSMI Adj Close")

    for d in long_dates:
        ax.axvspan(d, d + pd.Timedelta(days=1), alpha=0.15, color="seagreen", lw=0)
    for d in short_dates:
        ax.axvspan(d, d + pd.Timedelta(days=1), alpha=0.15, color="crimson", lw=0)

    legend_els = [
        ax.lines[0],
        Patch(facecolor="seagreen", alpha=0.6, label="LONG"),
        Patch(facecolor="crimson",  alpha=0.6, label="SHORT"),
    ]
    ax.legend(handles=legend_els, fontsize=9)
    ax.set_title(
        "Fine-Tuned Chronos (HP) - Long/Short Signals on SSMI (2018-2021)",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylabel("Price (CHF)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "backtest_signals.png", dpi=150)
    plt.close(fig)


def make_equity_plot(backtest_results):
    PLOTS_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    backtest_results.plot(ax=ax)
    ax.set_title(
        "Equity Curve - Fine-Tuned Chronos (HP) vs Buy & Hold (SSMI, 2018-2021)",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylabel("Portfolio Value (rebased to 100)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "backtest_equity_curve.png", dpi=150)
    plt.close(fig)


def make_monthly_heatmap(backtest_results):
    PLOTS_DIR.mkdir(exist_ok=True)
    monthly_rets = backtest_results.prices.resample("ME").last().pct_change().dropna()

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    for ax, col in zip(axes, monthly_rets.columns):
        pivot = monthly_rets[[col]].copy()
        pivot.index = pd.MultiIndex.from_arrays(
            [pivot.index.year, pivot.index.month], names=["Year", "Month"]
        )
        pivot = pivot[col].unstack(level="Month")
        pivot.columns = month_labels[:pivot.shape[1]]
        sns.heatmap(
            pivot * 100, annot=True, fmt=".1f", center=0,
            cmap="RdYlGn", linewidths=0.5, ax=ax,
            cbar_kws={"label": "Monthly Return (%)"},
        )
        ax.set_title(f"Monthly Returns - {col}")

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "backtest_monthly_heatmap.png", dpi=150)
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    df   = pd.read_csv(DATA_CSV, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    test = df[df["Date"] >= pd.Timestamp(TEST_START)].reset_index(drop=True)
    test["Adj Close"] = test["Adj Close"].ffill().bfill()
    y    = test["Adj Close"].values.astype(float)
    dates = pd.to_datetime(test["Date"])
    print(f"Test period: {dates.iloc[0].date()} -> {dates.iloc[-1].date()}  ({len(y)} days)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading fine-tuned checkpoints...")
    ft_low  = load_finetuned(CKPT_LOW,  device)
    ft_high = load_finetuned(CKPT_HIGH, device)

    print("Collecting signals (HP applied per context window)...")
    signal_dates, signal_values = collect_signals(y, dates, ft_low, ft_high)

    # Deduplicate overlapping windows (keep last forecast per day)
    signal_df_raw = pd.DataFrame(
        {"SIGNAL": signal_values},
        index=pd.DatetimeIndex(signal_dates),
    )
    signal_df_raw = signal_df_raw[~signal_df_raw.index.duplicated(keep="last")].sort_index()

    # Offset by -1 day: signal at close of day t -> rebalance at open of t+1
    signal_df = signal_df_raw.copy()
    signal_df = signal_df.set_index(signal_df.index - pd.DateOffset(1))

    print(f"\nSignal range: {signal_df.index[0].date()} -> {signal_df.index[-1].date()}")
    print(f"Long days :  {(signal_df['SIGNAL'] ==  1.0).sum()}")
    print(f"Short days:  {(signal_df['SIGNAL'] == -1.0).sum()}")

    # Price DataFrame for bt
    price_df = test[["Date", "Adj Close"]].copy()
    price_df = price_df.set_index("Date").rename(columns={"Adj Close": "SSMI"})
    price_df.index = pd.to_datetime(price_df.index)
    price_df = price_df.sort_index()
    price_df = price_df.loc[signal_df.index[0]:signal_df.index[-1]]
    print(f"Price range for bt: {price_df.index[0].date()} -> {price_df.index[-1].date()}  ({len(price_df)} rows)")

    # Strategy: fine-tuned Chronos long/short
    chronos_strategy = bt.Strategy(
        "Chronos FT+HP (ctx=180, hor=15)",
        [bt.algos.SelectAll(), ChronosSignalStrategy(signal_df), bt.algos.Rebalance()],
    )
    backtest_chronos = bt.Backtest(
        strategy=chronos_strategy, data=price_df, name="Chronos_FT_HP"
    )

    # Strategy: buy-and-hold baseline
    signal_bnh = signal_df.copy(deep=True)
    signal_bnh["SIGNAL"] = 1.0
    bnh_strategy = bt.Strategy(
        "Buy & Hold",
        [bt.algos.SelectAll(), ChronosSignalStrategy(signal_bnh), bt.algos.Rebalance()],
    )
    backtest_bnh = bt.Backtest(strategy=bnh_strategy, data=price_df, name="BuyAndHold")

    print("\nRunning bt backtest...")
    backtest_results = bt.run(backtest_chronos, backtest_bnh)

    print("\n")
    backtest_results.display()

    make_signal_plot(test, y, signal_df_raw)
    make_equity_plot(backtest_results)
    make_monthly_heatmap(backtest_results)
    print(f"\nPlots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
