import timesfm
import numpy as np

print("Loading TimesFM...")

tfm = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(
        backend="cpu",
        per_core_batch_size=32,
        horizon_len=60,
    ),
    checkpoint=timesfm.TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-1.0-200m-pytorch"
    ),
)

print("Model loaded. Running dummy forecast with context=120 (Filipp's filtering window size)...")
dummy = np.sin(np.linspace(0, 20, 120))
forecast, _ = tfm.forecast([dummy], freq=[0])

print(f"Forecast shape: {forecast.shape}")
print(f"First 5 values: {forecast[0][:5]}")
print("SUCCESS")