"""Multi-filter backtesting for Fine-Tuned Chronos on SSMI post-2018.

Compares HP / MA / Kalman fine-tuned checkpoints against Buy & Hold using
the bt library. Skips any filter whose checkpoint files are missing.

Signal: long/cash (1.0 = long SSMI, 0.0 = hold cash).
  - HP/MA applied per context window (causal, no look-ahead).
  - Kalman filtered state is inherently causal.
  - Signal offset by -1 day: rebalance at open of t+1.

Usage:
  python backtest_chronos_filters.py

Outputs (Chronos/plots_backtest/):
  equity_curves.png
  monthly_heatmap_{filter}.png   (one per available filter)
  signals_{filter}.png           (one per available filter)
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
from scipy.signal import butter, filtfilt
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.statespace.structural import UnobservedComponents

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_CSV     = PROJECT_ROOT / "DataSets" / "trimmed" / "SSMI.csv"
PLOTS_DIR    = SCRIPT_DIR / "plots_backtest"

BASE_MODEL_ID  = "amazon/chronos-t5-base"
TEST_START     = "2018-01-01"
CONTEXT_WINDOW = 180
HORIZON        = 15
STEP           = 30
HP_LAMBDA      = 10000
MA_WINDOW      = 30
BW_ORDER       = 4
BW_CUTOFF      = 0.05

FILTER_CONFIGS = {
    "hp":          {"label": "Chronos FT+HP",          "color": "steelblue"},
    "ma":          {"label": "Chronos FT+MA",          "color": "seagreen"},
    "kalman":      {"label": "Chronos FT+Kalman",      "color": "darkorange"},
    "butterworth": {"label": "Chronos FT+Butterworth", "color": "mediumpurple"},
}


# ── decompositions (all causal / per-window safe) ─────────────────────────────

def hp_decompose(y: np.ndarray):
    cycle, trend = hpfilter(y, lamb=HP_LAMBDA)
    return np.asarray(trend, dtype=np.float32), np.asarray(cycle, dtype=np.float32)


def ma_decompose(y: np.ndarray):
    s   = pd.Series(y)
    low = s.rolling(MA_WINDOW, min_periods=1).mean().values.astype(np.float32)
    return low, (y - low).astype(np.float32)


def kalman_decompose(y: np.ndarray):
    model  = UnobservedComponents(y, level="local linear trend")
    result = model.fit(disp=False)
    low    = np.asarray(result.level.filtered, dtype=np.float32)
    return low, (y - low).astype(np.float32)


def butterworth_decompose(y: np.ndarray):
    b, a = butter(BW_ORDER, BW_CUTOFF, btype="low", analog=False)
    low  = filtfilt(b, a, y).astype(np.float32)
    return low, (y - low).astype(np.float32)


DECOMPOSE = {
    "hp":          hp_decompose,
    "ma":          ma_decompose,
    "kalman":      kalman_decompose,
    "butterworth": butterworth_decompose,
}


# ── model helpers ─────────────────────────────────────────────────────────────

def load_finetuned(ckpt_path: Path, device: torch.device) -> ChronosPipeline:
    pipeline = ChronosPipeline.from_pretrained(
        BASE_MODEL_ID, device_map=None,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    state = torch.load(str(ckpt_path), map_location=device)
    pipeline.model.model.load_state_dict(state)
    pipeline.model.model.eval()
    return pipeline


def predict_median(pipeline, context_1d: np.ndarray, horizon: int) -> np.ndarray:
    ctx = torch.tensor(context_1d, dtype=torch.float32)
    with torch.no_grad():
        samples = pipeline.predict(ctx, prediction_length=horizon, num_samples=200)
    return samples[0].median(dim=0).values.cpu().numpy().astype(float)[:horizon]


# ── signal collection ─────────────────────────────────────────────────────────

def collect_signals(y: np.ndarray, dates: pd.DatetimeIndex,
                    ft_low: ChronosPipeline, ft_high: ChronosPipeline,
                    decompose_fn) -> pd.DataFrame:
    total = len(y)
    n_seg = (total - CONTEXT_WINDOW) // STEP
    signal_dates, signal_values = [], []

    for seg in range(n_seg):
        s = seg * STEP
        e = s + CONTEXT_WINDOW
        if e + HORIZON > total:
            break

        ctx_low, ctx_high = decompose_fn(y[s:e])
        pred = (predict_median(ft_low,  ctx_low,  HORIZON) +
                predict_median(ft_high, ctx_high, HORIZON))

        prev      = np.concatenate([[y[e - 1]], pred[:-1]])
        direction = np.where(pred > prev, 1.0, 0.0)   # long / cash

        seg_dates = pd.to_datetime(dates.iloc[e:e + HORIZON].values)
        signal_dates.extend(seg_dates.tolist())
        signal_values.extend(direction.tolist())

    raw = pd.DataFrame({"SIGNAL": signal_values}, index=pd.DatetimeIndex(signal_dates))
    raw = raw[~raw.index.duplicated(keep="last")].sort_index()
    # Offset by -1 day: signal at close of t -> rebalance at open of t+1
    shifted = raw.copy()
    shifted = shifted.set_index(shifted.index - pd.DateOffset(1))
    return shifted, raw


# ── bt strategy class ─────────────────────────────────────────────────────────

class SignalStrategy(bt.Algo):
    def __init__(self, signals: pd.DataFrame):
        self.signals = signals

    def __call__(self, target):
        if target.now in self.signals.index:
            signal = float(self.signals.loc[target.now, "SIGNAL"])
            target.temp["weights"] = {"SSMI": signal}
        return True


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_equity_curves(results, available_filters):
    PLOTS_DIR.mkdir(exist_ok=True)
    prices = results.prices
    prices = prices / prices.iloc[0] * 100  # rebase to 100

    fig, ax = plt.subplots(figsize=(14, 6))
    for f in available_filters:
        col = f"Chronos_FT_{f.upper()}"
        if col in prices.columns:
            cfg = FILTER_CONFIGS[f]
            ax.plot(prices.index, prices[col], color=cfg["color"], lw=1.5, label=cfg["label"])
    if "BuyAndHold" in prices.columns:
        ax.plot(prices.index, prices["BuyAndHold"], color="crimson",
                lw=2, ls="--", label="Buy & Hold")

    ax.axhline(100, color="gray", ls=":", lw=0.8)
    ax.set_title(
        "Equity Curves — Chronos Fine-Tuned Filter Comparison vs Buy & Hold (SSMI, 2018-2021)",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylabel("Portfolio Value (rebased to 100)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "equity_curves.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: equity_curves.png")


def plot_monthly_heatmap(results, filter_name):
    PLOTS_DIR.mkdir(exist_ok=True)
    bt_name  = f"Chronos_FT_{filter_name.upper()}"
    bnh_name = "BuyAndHold"
    available = [c for c in results.prices.columns if c in (bt_name, bnh_name)]
    monthly_rets = results.prices[available].resample("ME").last().pct_change().dropna()
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, monthly_rets.columns):
        pivot = monthly_rets[[col]].copy()
        pivot.index = pd.MultiIndex.from_arrays(
            [pivot.index.year, pivot.index.month], names=["Year", "Month"]
        )
        pivot = pivot[col].unstack(level="Month")
        pivot.columns = month_labels[:pivot.shape[1]]
        sns.heatmap(pivot * 100, annot=True, fmt=".1f", center=0,
                    cmap="RdYlGn", linewidths=0.5, ax=ax,
                    cbar_kws={"label": "Monthly Return (%)"})
        ax.set_title(f"Monthly Returns — {col}")
    plt.tight_layout()
    fname = f"monthly_heatmap_{filter_name}.png"
    fig.savefig(PLOTS_DIR / fname, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_signals(y, test_df, raw_signal_df, filter_name, cfg):
    PLOTS_DIR.mkdir(exist_ok=True)
    plt.rcParams["axes.spines.top"]   = False
    plt.rcParams["axes.spines.right"] = False

    dates      = pd.to_datetime(test_df["Date"])
    long_dates = raw_signal_df[raw_signal_df["SIGNAL"] == 1.0].index
    cash_dates = raw_signal_df[raw_signal_df["SIGNAL"] == 0.0].index

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, y, color="black", lw=1)
    for d in long_dates:
        ax.axvspan(d, d + pd.Timedelta(days=1), alpha=0.18, color=cfg["color"], lw=0)
    for d in cash_dates:
        ax.axvspan(d, d + pd.Timedelta(days=1), alpha=0.08, color="gray", lw=0)

    legend_els = [
        ax.lines[0],
        Patch(facecolor=cfg["color"], alpha=0.6, label="LONG"),
        Patch(facecolor="gray",       alpha=0.4, label="CASH"),
    ]
    ax.legend(handles=legend_els, fontsize=9)
    ax.set_title(
        f"{cfg['label']} - Long/Cash Signals on SSMI (2018-2021)",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylabel("Price (CHF)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fname = f"signals_{filter_name}.png"
    fig.savefig(PLOTS_DIR / fname, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")


# ── summary table ─────────────────────────────────────────────────────────────

def print_comparison_table(results, available_filters):
    stats = results.stats
    print("\n" + "=" * 70)
    print("  FILTER COMPARISON  |  Chronos Fine-Tuned  vs  Buy & Hold")
    print("=" * 70)
    header = f"  {'Metric':<28}"
    for f in available_filters:
        header += f"  {('FT+' + f.upper()):>12}"
    header += f"  {'B&H':>8}"
    print(header)
    print("-" * 70)

    bt_names  = [f"Chronos_FT_{f.upper()}" for f in available_filters]
    bnh_name  = "BuyAndHold"
    all_names = bt_names + [bnh_name]

    metrics = [
        ("Total Return",  "total_return",  "{:>+.2f}%"),
        ("CAGR",          "cagr",          "{:>+.2f}%"),
        ("Daily Sharpe",  "daily_sharpe",  "{:>+.3f}"),
        ("Max Drawdown",  "max_drawdown",  "{:>+.2f}%"),
    ]
    for label, key, fmt in metrics:
        row = f"  {label:<28}"
        for name in all_names:
            if name in stats.columns:
                val = stats.loc[key, name] * (100 if "%" in fmt else 1)
                row += f"  {fmt.format(val):>12}"
            else:
                row += f"  {'N/A':>12}"
        print(row)
    print("=" * 70)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    df   = pd.read_csv(DATA_CSV, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    test = df[df["Date"] >= pd.Timestamp(TEST_START)].reset_index(drop=True)
    test["Adj Close"] = test["Adj Close"].ffill().bfill()
    y     = test["Adj Close"].values.astype(float)
    dates = pd.to_datetime(test["Date"])
    print(f"Test period: {dates.iloc[0].date()} -> {dates.iloc[-1].date()}  ({len(y)} days)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Discover available filter checkpoints
    available_filters = []
    for f in ["hp", "ma", "kalman", "butterworth"]:
        low  = SCRIPT_DIR / f"chronos_ssmi_low_{f}.pt"
        high = SCRIPT_DIR / f"chronos_ssmi_high_{f}.pt"
        if low.exists() and high.exists():
            available_filters.append(f)
        else:
            print(f"  [{f.upper()}] checkpoints missing — skipping")

    if not available_filters:
        print("No checkpoints found. Run finetune_chronos_decomposed_v1.py --filter <hp|ma|kalman> first.")
        return

    print(f"Running backtest for filters: {available_filters}")

    # Collect signals for each filter
    all_signals     = {}
    all_signals_raw = {}
    price_start     = None
    price_end       = None

    for f in available_filters:
        print(f"\n[{f.upper()}] Loading checkpoints...")
        ft_low  = load_finetuned(SCRIPT_DIR / f"chronos_ssmi_low_{f}.pt",  device)
        ft_high = load_finetuned(SCRIPT_DIR / f"chronos_ssmi_high_{f}.pt", device)

        print(f"[{f.upper()}] Collecting signals (causal, per-window)...")
        sig, sig_raw = collect_signals(y, dates, ft_low, ft_high, DECOMPOSE[f])
        all_signals[f]     = sig
        all_signals_raw[f] = sig_raw

        if price_start is None or sig.index[0] > price_start:
            price_start = sig.index[0]
        if price_end is None or sig.index[-1] < price_end:
            price_end = sig.index[-1]

        long_days = (sig["SIGNAL"] == 1.0).sum()
        cash_days = (sig["SIGNAL"] == 0.0).sum()
        print(f"  Signal range: {sig.index[0].date()} -> {sig.index[-1].date()}")
        print(f"  LONG: {long_days}  CASH: {cash_days}")

    # Build price DataFrame for bt (common window)
    price_df = test[["Date", "Adj Close"]].copy()
    price_df = price_df.set_index("Date").rename(columns={"Adj Close": "SSMI"})
    price_df.index = pd.to_datetime(price_df.index)
    price_df = price_df.sort_index().loc[price_start:price_end]
    print(f"\nPrice range for bt: {price_df.index[0].date()} -> {price_df.index[-1].date()}  ({len(price_df)} rows)")

    # Build bt backtests
    backtests = []
    for f in available_filters:
        cfg = FILTER_CONFIGS[f]
        strat = bt.Strategy(
            cfg["label"],
            [bt.algos.SelectAll(), SignalStrategy(all_signals[f]), bt.algos.Rebalance()],
        )
        backtests.append(bt.Backtest(strat, price_df, name=f"Chronos_FT_{f.upper()}"))

    # Buy & Hold (use first filter's signal range, all 1.0)
    first_sig     = all_signals[available_filters[0]]
    signal_bnh    = first_sig.copy(deep=True)
    signal_bnh["SIGNAL"] = 1.0
    bnh_strat = bt.Strategy(
        "Buy & Hold",
        [bt.algos.SelectAll(), SignalStrategy(signal_bnh), bt.algos.Rebalance()],
    )
    backtests.append(bt.Backtest(bnh_strat, price_df, name="BuyAndHold"))

    print("\nRunning bt backtest...")
    results = bt.run(*backtests)

    print("\n")
    results.display()

    print_comparison_table(results, available_filters)

    # Plots
    print("\nSaving plots...")
    plot_equity_curves(results, available_filters)
    for f in available_filters:
        plot_monthly_heatmap(results, f)
        plot_signals(y, test, all_signals_raw[f], f, FILTER_CONFIGS[f])

    print(f"\nAll plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
