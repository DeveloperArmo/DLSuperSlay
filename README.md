# $SMI Stock Market Forecasting Using Frequency Filtering and Fine-Tuned Foundation Time-Series Model

## MCS Deep Learning Project - Spring 2026

### Group: Super Slay
**Group Members:** Ismael B., Arman Gezerer, Arabella Jane Todd, Filipp Rechsteiner

---

## Research Question
Does frequency filtering combined with fine-tuning a time-series foundation model improve $SMI stock market forecasting accuracy and address noisiness compared to standard zero-shot forecasting?

## Objective
This project aims to improve forecast accuracy and stability on $SMI financial time series by reducing the impact of noisy frequency filtering. The approach involves leveraging time-series foundation models to forecast and then eventually fine-tune for the target $SMI financial data domain.

## Data
- **Dataset:** Swiss Market Index ($SMI) time series data
- **Source:** [Kaggle - Swiss Market Index Dataset](https://www.kaggle.com/datasets/kaminraminang/swiss-market-index-biggest-20-shares-nse-csv)
- **Composition:** Time series data on the value of the Swiss Market Index, comprising the biggest 20 securities (~80% of Switzerland's entire market capitalization)
- **Format:** CSV with standard market fields (Date, Open, High, Low, Close, Adj Close, Volume)

## Key Papers
1. **Background:** Shah & Thaker - "A review of time series forecasting methods" (INTERNATIONAL JOURNAL OF RESEARCH PUBLICATION AND REVIEWS, 2024)
2. **Amazon's Chronos Model:** Ansari et al. - "Chronos: Learning the language of time series" (2024)
3. **Google's TimesFM Model:** Das, Sen, et al. - "A decoder-only foundation model for time-series forecasting" (2023)
4. **TiDE Model:** Zhu et al. - "TiDeCast: A Foundation Model for Financial Time-Series Forecasting" (2025)

---

## Project Structure
```
DL2026/
├── data/               # Data files (ignored in git)
├── notebooks/          # Jupyter notebooks for experiments
├── src/                # Source code
├── models/             # Saved models (ignored in git)
├── results/            # Results and visualizations
└── README.md
```

## Setup
Instructions for environment setup and dependencies will be added.

## License
This project is for academic purposes as part of the MCS Deep Learning course.
