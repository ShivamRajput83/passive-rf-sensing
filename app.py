"""Streamlit demo dashboard for the synthetic RF pipeline."""
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import welch

st.set_page_config(page_title="RF Simulation Dashboard", page_icon="📡", layout="wide")
st.title("📡 Passive RF Sensing — Simulation Dashboard")
st.warning(
    "Simulation only: this dashboard classifies synthetic signal patterns. "
    "It does not detect real smartphones or prove cellular RF detection."
)

@st.cache_resource
def load_model():
    return joblib.load("rf_simulated_model.joblib")

def simulate_iq(activity: bool, n=4096):
    rng = np.random.default_rng()
    scale = rng.uniform(0.35, 0.8)
    iq = scale * (rng.normal(size=n) + 1j*rng.normal(size=n)) / np.sqrt(2)
    if activity:
        t = np.arange(n)
        f = rng.uniform(0.04, 0.22)
        carrier = np.exp(1j * (2*np.pi*f*t + rng.uniform(0, 2*np.pi)))
        env = np.zeros(n)
        for _ in range(rng.integers(2, 7)):
            start = rng.integers(0, n-300)
            width = rng.integers(100, 500)
            env[start:min(start+width, n)] = rng.uniform(0.8, 2.2)
        iq += env * carrier * rng.uniform(0.5, 1.5)
    return iq.astype(np.complex64)

def extract(iq):
    p = np.abs(iq)**2
    _, psd = welch(iq, nperseg=512, return_onesided=False, scaling="density")
    psd = np.fft.fftshift(np.maximum(psd, 1e-12))
    prob = psd / psd.sum()
    return {
        "mean_power": float(np.mean(p)),
        "peak_power": float(np.max(p)),
        "power_variance": float(np.var(p)),
        "spectral_entropy": float(-np.sum(prob*np.log2(prob))),
        "spectral_peak_fraction": float(np.max(psd)/np.sum(psd)),
    }, psd

bundle = load_model()
model, feature_names = bundle["model"], bundle["features"]

left, right = st.columns(2)
with left:
    st.subheader("Generate a test window")
    choice = st.radio(
        "Choose simulated condition",
        ["Simulated activity", "Simulated background"],
        help="This selects a synthetic signal generator, not a real phone."
    )
    if st.button("Generate and classify", type="primary"):
        iq = simulate_iq(choice == "Simulated activity")
        feat, psd = extract(iq)
        X = pd.DataFrame([feat])[feature_names]
        pred = model.predict(X)[0]
        probs = model.predict_proba(X)[0]
        confidence = float(np.max(probs))
        st.session_state["result"] = (choice, feat, pred, confidence, psd)

with right:
    st.subheader("Model output")
    if "result" in st.session_state:
        choice, feat, pred, confidence, psd = st.session_state["result"]
        st.metric("Predicted simulated class", pred.replace("_", " ").title())
        st.metric("Model score (not calibrated probability)", f"{confidence:.1%}")
        st.caption(f"Generator setting: {choice}")
        st.write("Extracted features")
        st.dataframe(pd.DataFrame([feat]), use_container_width=True)
        st.line_chart(pd.DataFrame({"Shifted spectrum": psd}))
    else:
        st.info("Choose a simulated condition and click Generate and classify.")

st.divider()
st.subheader("Pipeline")
st.code("Synthetic IQ → feature extraction → trained Random Forest → dashboard")
st.caption(
    "To claim real-world smartphone presence detection, collect labelled measurements "
    "with appropriate RF hardware and validate on unseen sessions and locations."
)
