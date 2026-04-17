# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

University group project (MCS Deep Learning, Spring 2026) investigating whether frequency filtering combined with fine-tuning a time-series foundation model improves $SMI stock market forecasting vs. standard zero-shot forecasting. Models under consideration: Chronos (Amazon), TimesFM (Google), TiDeCast.

## Data Pipeline

The data pipeline has three stages, each building on the previous:

1. **Raw CSVs** (`DataSets/*.csv`) - 30 SMI constituent stocks + SSMI index from Kaggle (1990-2021), standard OHLCV format with Date index.
2. **Missing value analysis & cleaning**:
   - `check_missing.py` - Analyzes SSMI.csv specifically: identifies NaN rows, classifies them as Swiss holidays (fixed + Easter-based movable) vs. unexplained.
   - `SMI_preprocessing.ipynb` - Interactive SSMI cleaning: drops holiday rows, forward-fills remaining NaNs, outputs `DataSets/SSMI cleaned/SSMI_cleaned.csv`.
   - `preprocess_all.py` - Batch-cleans all stock CSVs except SSMI: drops Swiss holiday rows, forward-fills unexplained NaNs, **overwrites originals in place**.
3. **Temporal alignment** (`trim_datasets.ipynb`) - Trims all datasets to common window 2001-10-01 through 2021-05-17. Excludes ALC, AMS, PGHN (too short). Outputs to `DataSets/trimmed/`. Originals are preserved.

## Commands

```bash
# Run batch preprocessing (cleans all stock CSVs, overwrites originals)
python preprocess_all.py

# Run SSMI-only missing value check
python check_missing.py
```

Notebooks are meant to be run interactively in Jupyter.

## Key Dependencies

- pandas, dateutil (for Easter calculation in holiday detection)
- Standard Jupyter environment for notebooks

## Important Notes

- `preprocess_all.py` modifies CSV files **in place** - the original raw data is only preserved in git history.
- Swiss holiday detection covers both fixed holidays (Neujahr, Berchtoldstag, Tag der Arbeit, Bundesfeiertag, Heiligabend, Weihnachten, Stephanstag, Silvester) and Easter-based movable holidays (Karfreitag, Ostermontag, Auffahrt, Pfingstmontag).
- Code comments and output labels are in German.
- The `DataSets/trimmed/` folder contains the analysis-ready data (26 stocks + SSMI, aligned time range, cleaned).
