"""
Generate a synthetic RF IQ dataset for software-pipeline testing.
IMPORTANT: These are simulated classes, NOT real cellular phone measurements.
"""
import numpy as np
import pandas as pd
from scipy.signal import welch

RNG = np.random.default_rng(42)
N_SAMPLES = 4096
N_PER_CLASS = 300

def make_iq(simulated_activity: bool) -> np.ndarray:
    # Complex background noise.
    noise_scale = RNG.uniform(0.35, 0.8)
    iq = noise_scale * (
        RNG.normal(size=N_SAMPLES) + 1j * RNG.normal(size=N_SAMPLES)
    ) / np.sqrt(2)

    if simulated_activity:
        # Add a synthetic bursty, modulated-looking component.
        # This is deliberately a simple toy signal, not an LTE/5G waveform.
        t = np.arange(N_SAMPLES)
        freq = RNG.uniform(0.04, 0.22)  # normalized cycles/sample
        phase = RNG.uniform(0, 2 * np.pi)
        carrier = np.exp(1j * (2 * np.pi * freq * t + phase))
        envelope = np.zeros(N_SAMPLES)
        for _ in range(RNG.integers(2, 7)):
            start = RNG.integers(0, N_SAMPLES - 300)
            width = RNG.integers(100, 500)
            envelope[start:min(start + width, N_SAMPLES)] = RNG.uniform(0.8, 2.2)
        iq += envelope * carrier * RNG.uniform(0.5, 1.5)
    return iq.astype(np.complex64)

def features(iq: np.ndarray) -> dict:
    power = np.abs(iq) ** 2
    freqs, psd = welch(iq, nperseg=512, return_onesided=False, scaling="density")
    psd = np.fft.fftshift(psd)
    psd = np.maximum(psd, 1e-12)
    prob = psd / psd.sum()
    entropy = -np.sum(prob * np.log2(prob))
    return {
        "mean_power": float(np.mean(power)),
        "peak_power": float(np.max(power)),
        "power_variance": float(np.var(power)),
        "spectral_entropy": float(entropy),
        "spectral_peak_fraction": float(np.max(psd) / np.sum(psd)),
        "label": "simulated_activity" if False else None,
    }

rows = []
for label, active in [("simulated_activity", True), ("simulated_background", False)]:
    for i in range(N_PER_CLASS):
        iq = make_iq(active)
        row = features(iq)
        row["label"] = label
        row["sample_id"] = f"{label}_{i:04d}"
        rows.append(row)

df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("dataset.csv", index=False)
print(f"Created dataset.csv with {len(df)} rows.")
print("Reminder: labels describe synthetic signal conditions, not actual phone presence.")
