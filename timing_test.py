import time
import numpy as np
import timesfm

print("Loading model...", flush=True)
start_load = time.time()

tfm = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(
        backend="cpu",
        per_core_batch_size=32,
        horizon_len=30,
    ),
    checkpoint=timesfm.TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
    ),
)

print(f"Model loaded in {time.time() - start_load:.1f}s", flush=True)

# Warm-up
print("Warm-up forecast...", flush=True)
dummy = np.random.randn(120)
tfm.forecast([dummy], freq=[0])
print("Warm-up done", flush=True)

# Measured runs
print("Running 6 forecasts (= 3 segments × 2 components)...", flush=True)
start = time.time()
for i in range(6):
    tfm.forecast([dummy], freq=[0])
    print(f"  forecast {i+1}/6 done at {time.time() - start:.1f}s", flush=True)
elapsed = time.time() - start

print()
print(f"=== RESULTS ===")
print(f"6 forecasts in {elapsed:.1f}s ({elapsed/6:.1f}s per forecast)")
print(f"Per filter (251 segments × 2 components): {elapsed * 251 / 6 / 60:.1f} min")
print(f"Full 4-filter run: {elapsed * 251 * 4 / 6 / 3600:.1f} hours")