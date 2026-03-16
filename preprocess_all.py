import pandas as pd
import os
from dateutil.easter import easter

# ──────────────────────────────────────────────
# Schweizer Feiertage generieren
# ──────────────────────────────────────────────

def get_swiss_holidays(year_range):
    """Gibt ein Set aller Schweizer Feiertage (fix + beweglich) zurück."""
    holidays = set()
    for year in year_range:
        # Feste Feiertage
        holidays.add(pd.Timestamp(year, 1, 1))    # Neujahr
        holidays.add(pd.Timestamp(year, 1, 2))    # Berchtoldstag
        holidays.add(pd.Timestamp(year, 5, 1))    # Tag der Arbeit
        holidays.add(pd.Timestamp(year, 8, 1))    # Bundesfeiertag
        holidays.add(pd.Timestamp(year, 12, 24))  # Heiligabend
        holidays.add(pd.Timestamp(year, 12, 25))  # Weihnachten
        holidays.add(pd.Timestamp(year, 12, 26))  # Stephanstag
        holidays.add(pd.Timestamp(year, 12, 31))  # Silvester

        # Bewegliche Feiertage (Oster-basiert)
        e = pd.Timestamp(easter(year))
        holidays.add(e - pd.Timedelta(days=2))    # Karfreitag
        holidays.add(e + pd.Timedelta(days=1))    # Ostermontag
        holidays.add(e + pd.Timedelta(days=39))   # Auffahrt
        holidays.add(e + pd.Timedelta(days=50))   # Pfingstmontag
    return holidays


# ──────────────────────────────────────────────
# Alle DataSets ausser SSMI verarbeiten
# ──────────────────────────────────────────────

DATA_DIR = "DataSets"
EXCLUDE = {"SSMI.csv"}
COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

csv_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".csv") and f not in EXCLUDE)

print(f"Verarbeite {len(csv_files)} Dateien (SSMI ausgeschlossen)\n")
print("=" * 80)

summary_rows = []

for filename in csv_files:
    ticker = filename.replace(".csv", "")
    path = os.path.join(DATA_DIR, filename)

    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    total_rows = len(df)
    nan_mask = df.isna().any(axis=1)
    nan_count = nan_mask.sum()

    if nan_count == 0:
        summary_rows.append({
            "Ticker": ticker,
            "Zeilen": total_rows,
            "NaN gesamt": 0,
            "Feiertage": 0,
            "Unerklärte NaN": 0,
            "NaN nach ffill": 0,
        })
        continue

    # Feiertage für den relevanten Zeitraum generieren
    years = df.index.year.unique()
    holidays = get_swiss_holidays(years)

    # Feste Feiertage Maske
    fixed_mask = nan_mask & (
        ((df.index.month == 1) & (df.index.day.isin([1, 2])))
        | ((df.index.month == 5) & (df.index.day == 1))
        | ((df.index.month == 8) & (df.index.day == 1))
        | ((df.index.month == 12) & (df.index.day.isin([24, 25, 26, 31])))
    )

    # Bewegliche Feiertage Maske
    moving_mask = nan_mask & df.index.isin(holidays)

    holiday_mask = fixed_mask | moving_mask
    holiday_count = holiday_mask.sum()
    unexplained_count = nan_count - holiday_count

    # ── Details zu unerklärten NaN-Zeilen ausgeben ──
    unexplained_dates = df[nan_mask & ~holiday_mask].index
    if len(unexplained_dates) > 0:
        print(f"\n{ticker}: {nan_count} NaN-Zeilen ({holiday_count} Feiertage, "
              f"{unexplained_count} unerklärte)")
        print(f"  Unerklärte NaN-Daten:")
        for d in unexplained_dates:
            loc = df.index.get_loc(d)
            prev_close = df.iloc[loc - 1]["Close"] if loc > 0 else None
            next_close = df.iloc[loc + 1]["Close"] if loc < len(df) - 1 else None
            print(f"    {d.strftime('%Y-%m-%d')} ({d.strftime('%A')})  "
                  f"prev_close={prev_close}  next_close={next_close}")

    # ── Feiertags-Zeilen entfernen ──
    df = df.drop(df[holiday_mask].index)

    # ── Verbleibende NaN-Zeilen per forward-fill auffüllen ──
    remaining_nan = df.isna().any(axis=1).sum()
    df[COLUMNS] = df[COLUMNS].ffill()
    nan_after_ffill = df.isna().any(axis=1).sum()

    # ── Bereinigtes CSV speichern ──
    df.to_csv(path)

    summary_rows.append({
        "Ticker": ticker,
        "Zeilen": total_rows,
        "NaN gesamt": nan_count,
        "Feiertage": holiday_count,
        "Unerklärte NaN": unexplained_count,
        "NaN nach ffill": nan_after_ffill,
    })

# ──────────────────────────────────────────────
# Zusammenfassung
# ──────────────────────────────────────────────

print("\n" + "=" * 80)
print("ZUSAMMENFASSUNG")
print("=" * 80)

summary = pd.DataFrame(summary_rows)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 120)
print(summary.to_string(index=False))

total_holidays = summary["Feiertage"].sum()
total_unexplained = summary["Unerklärte NaN"].sum()
total_remaining = summary["NaN nach ffill"].sum()
print(f"\nTotal Feiertage entfernt: {total_holidays}")
print(f"Total unerklärte NaN (forward-filled): {total_unexplained}")
print(f"Total verbleibende NaN nach ffill: {total_remaining}")
