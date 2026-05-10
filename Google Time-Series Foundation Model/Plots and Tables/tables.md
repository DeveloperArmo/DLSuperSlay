# TimesFM Final Results

## Table 1 — Hyperparameter Sweep at context=180, horizon=15

| filter         |   config_id | config                       |   rmse |   mape |   pearson_r2 |   directional_accuracy |
|:---------------|------------:|:-----------------------------|-------:|-------:|-------------:|-----------------------:|
| Butterworth    |           1 | {'cutoff': 0.05, 'order': 2} | 189.76 |   2.48 |        0.161 |                  50.55 |
| Butterworth    |           2 | {'cutoff': 0.1, 'order': 2}  | 205.66 |   2.78 |        0.218 |                  47.6  |
| Butterworth    |           3 | {'cutoff': 0.2, 'order': 2}  | 183.67 |   2.52 |        0.241 |                  49.05 |
| HP Filter      |           1 | {'lambda': 100}              | 146.31 |   2.13 |        0.345 |                  48.73 |
| HP Filter      |           2 | {'lambda': 1600}             | 140.8  |   2    |        0.322 |                  50.09 |
| HP Filter      |           3 | {'lambda': 10000}            | 138.88 |   1.93 |        0.371 |                  51.38 |
| Kalman Filter  |           1 | {'process_noise': 0.001}     | 130.69 |   1.8  |        0.409 |                  51.38 |
| Kalman Filter  |           2 | {'process_noise': 0.01}      | 134.41 |   1.71 |        0.321 |                  50.31 |
| Kalman Filter  |           3 | {'process_noise': 0.1}       | 125.99 |   1.67 |        0.238 |                  49.64 |
| Moving Average |           1 | {'window': 10}               | 129.77 |   1.76 |        0.311 |                  50.04 |
| Moving Average |           2 | {'window': 30}               | 124.45 |   1.76 |        0.391 |                  51    |
| Moving Average |           3 | {'window': 60}               | 133.51 |   1.72 |        0.393 |                  51.81 |

## Table 2 — Best Filter per Metric (Hyperparameter Sweep)

| metric               | filter         |   config_id | config                   |   metric_value |
|:---------------------|:---------------|------------:|:-------------------------|---------------:|
| rmse                 | Moving Average |           2 | {'window': 30}           |        124.446 |
| mape                 | Kalman Filter  |           3 | {'process_noise': 0.1}   |          1.671 |
| pearson_r2           | Kalman Filter  |           1 | {'process_noise': 0.001} |          0.409 |
| directional_accuracy | Moving Average |           3 | {'window': 60}           |         51.807 |

## Table 3 — Fine-Tuning Comparison at context=180, horizon=15

| Config                     |   Segments |   RMSE |   MAPE % |    R² |   DirAcc % |
|:---------------------------|-----------:|-------:|---------:|------:|-----------:|
| Baseline                   |        249 | 121.18 |     1.63 | 0.413 |      50.47 |
| Fine-tuned v1 (head only)  |         22 | 162.9  |     1.39 | 0.37  |      52.12 |
| Fine-tuned v2 (13% + norm) |         22 | 239.73 |     2.1  | 0.32  |      49.09 |
| Fine-tuned v3 (23% + reg)  |         22 | 250.59 |     2.2  | 0.244 |      51.21 |

## Table 4 — Cross-Setting Comparison

Comparing baseline + best fine-tuning at two sliding-window settings:

| Config               |   Context |   Horizon |   RMSE |   MAPE % |    R² |   DirAcc % |
|:---------------------|----------:|----------:|-------:|---------:|------:|-----------:|
| 120/30 Baseline      |       120 |        30 | 207.98 |     1.72 | 0.182 |      51.25 |
| 120/30 Fine-tuned v1 |       120 |        30 | 271.82 |     2.35 | 0.252 |      50.69 |
| 120/30 Fine-tuned v2 |       120 |        30 | 239.36 |     2.08 | 0.179 |      51.94 |
| 120/30 Fine-tuned v3 |       120 |        30 | 274.6  |     2.45 | 0.14  |      50.97 |
| 180/15 Baseline      |       180 |        15 | 121.18 |     1.63 | 0.413 |      50.47 |
| 180/15 Fine-tuned v1 |       180 |        15 | 162.9  |     1.39 | 0.37  |      52.12 |
| 180/15 Fine-tuned v2 |       180 |        15 | 239.73 |     2.1  | 0.32  |      49.09 |
| 180/15 Fine-tuned v3 |       180 |        15 | 250.59 |     2.2  | 0.244 |      51.21 |
