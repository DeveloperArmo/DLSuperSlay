# TimesFM Fine-Tuning

## Common Setup (applies to all versions)

- **Model:** TimesFM 1.0 (200M parameters, Google's time-series foundation model)
- **Dataset split:** trimmed SSMI (2001-2021)
  - Training: 2001-10-01 to 2017-12-31 (4,137 days)
  - Test: 2018-01-01 to 2021-05-17 (843 days = 24 forecast segments)
- **Approach:** two-model decomposition
  - Apply HP filter to training data → low-pass (smooth trend) and high-pass (cyclic residual) components
  - Fine-tune `model_low` on the low-pass component
  - Fine-tune `model_high` on the high-pass component
  - At test time: decompose each test segment, forecast each component with its specialized model, recombine
- **Method:** parameter-efficient fine-tuning — freeze most of the model, train only selected parts
- **Shared hyperparameters:** AdamW optimizer, MSE loss, batch size 4, 90/10 chronological train/val split, HP filter λ=129600

---

## Version 1 — baseline approach

**Trainable parameters:** 2.4% of the model (~5M params)

- Only the output head (`horizon_ff_layer`) trainable
- Everything else frozen — transformer backbone, input projections, etc.

**Hyperparameters:** learning rate 1e-4, 3 epochs, no normalization

**Result on post-2018 test:** RMSE 271.82, DirAcc 50.69%

**Interpretation:** Mild learning happened but not enough capacity to adapt to SSMI-specific patterns. Underfit.

---

## Version 2 — capacity + normalization

**Two changes from v1:**

1. **More unfrozen parameters** — 13% of the model (~26M params)
   - Added `input_ff_layer` (input projection)
   - Added last 2 transformer blocks (layers 18 and 19)
   - Kept `horizon_ff_layer` (output head)

2. **Per-window z-score normalization**
   - For each training window, compute mean and standard deviation of the 120-day context
   - Normalize both context and target using those statistics before feeding the model
   - At test time: normalize the input the same way, forecast, then denormalize prediction back to raw price scale
   - Why: stabilizes training across samples at different price levels (SSMI was ~7,000 in 2001 and ~11,000 in 2017)

**Hyperparameters:** learning rate 5e-5 (halved from v1), 3 epochs

**Result on post-2018 test:** RMSE 239.36, DirAcc 51.94%

**Interpretation:** Meaningful improvement — 12% better RMSE than v1. Closed roughly half the gap to zero-shot baseline. Best DirAcc among all variants. Healthy training curves, no overfitting signs.

---

## Version 3 — more capacity

**Changes from v2:**

1. **Even more unfrozen parameters** — 23% of the model (~46M params)
   - Same as v2 plus last 4 transformer blocks instead of 2 (layers 16-19)

2. **Regularization to fight overfitting**
   - Weight decay 0.01 in AdamW
   - Gradient clipping (norm 1.0)
   - Early stopping with patience=2 (max 5 epochs, stop if val loss increases 2 in a row)

3. **Lower learning rate** — 3e-5 (more params need gentler updates)

**Result on post-2018 test:** RMSE 274.60, DirAcc 50.97%

**Interpretation:** Worse than v2. V3 overfit. With only 120 training windows + 9 validation windows, 46M trainable parameters was too many. Early stopping kicked in, but with only 9 val windows, picking the "best" checkpoint became unreliable.

---

## Capacity-generalization tradeoff

| Version | Trainable % | Test RMSE | Interpretation |
|---------|-------------|-----------|----------------|
| v1      | 2.4%        | 272       | underfit (too restrictive) |
| **v2**  | **13%**     | **239**   | **sweet spot**             |
| v3      | 23%         | 275       | overfit — too much capacity for the data |

Classic capacity curve. V2 is our best fine-tuning.

---

## How fine-tuning compares to zero-shot

Same post-2018 test set (24 segments):

| Config                  | RMSE   | DirAcc |
|-------------------------|--------|--------|
| Zero-shot raw TimesFM   | **208** | 51.25% |
| Zero-shot HP filter     | 277    | 50.28% |
| **Fine-tuned v2**       | 239    | **51.94%** |

**Bottom line:**

- Zero-shot raw TimesFM still wins on RMSE (208 vs 239)
- Fine-tuned v2 beats zero-shot raw on directional accuracy (51.94% vs 51.25%)
- HP filtering alone hurts performance (doesn't improve over raw)
- Fine-tuning on HP-decomposed components partially recovers what HP filtering lost, but doesn't beat doing nothing

---

## Honest research story

**What we set out to show:** "Foundation models + HP filtering + fine-tuning is better than just using the foundation model."

**What we actually found:**

1. HP filtering alone hurts (both zero-shot and fine-tuned).
2. Output-head-only fine-tuning (v1) is too restrictive to help.
3. Moderate fine-tuning with normalization (v2) recovers most of what HP costs.
4. More fine-tuning capacity (v3) overfits on this small dataset.
5. **Nothing we tried beats zero-shot raw TimesFM on RMSE.** Fine-tuned v2 does beat it on directional accuracy by a small margin.

---

## Methodological caveat

Our test set is small — 24 segments covering 2018-2021. Differences of ±30 RMSE between configurations might not be statistically significant. The COVID crash in early 2020 is included, which likely inflates all RMSE values. A longer test period would give cleaner conclusions.

---

## What's still open

- **Path D (not done):** expand training data from just SSMI (~120 windows) to SSMI + 25 constituent stocks (~3,000 windows). This addresses the small-sample problem that killed v3. With proper training data, v3-sized capacity might genuinely beat zero-shot raw.
- **Ablation (not done):** we combined two changes in v2 (more layers + normalization). We don't know which did the heavy lifting. A principled writeup would include an ablation.