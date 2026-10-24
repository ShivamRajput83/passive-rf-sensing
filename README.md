# Passive RF Sensing — Simulation Starter Project

## What this prototype does
This project demonstrates a software pipeline using synthetic complex IQ samples:
1. Generate synthetic signal windows and background-noise windows.
2. Extract numerical features into `dataset.csv`.
3. Train and evaluate a Random Forest classifier.
4. View simulated predictions in a Streamlit dashboard.

**Important limitation:** the generated classes are synthetic signal patterns. They are not measured cellular transmissions and do not establish that a smartphone is present. Do not present the synthetic accuracy as real-world detection accuracy.

## Setup (Windows)
Install Python 3.10 or newer. Open Command Prompt in this folder.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python generate_dataset.py
python train_model.py
streamlit run app.py
```

If `python` is not recognized, try `py` instead.

## Files
- `generate_dataset.py` — creates `dataset.csv`
- `train_model.py` — trains and saves `rf_simulated_model.joblib`
- `app.py` — interactive demo dashboard
- `requirements.txt` — required Python packages

## How GNU Radio fits later
GNU Radio Companion can be used to build a flowgraph that generates or reads complex samples and visualizes them. First get this Python-only prototype running. Then replace the synthetic sample source with GNU Radio-generated samples, and later—if hardware becomes available—with recorded IQ data. Keep feature extraction identical between training and inference.

## Suggested next milestone
1. Run the pipeline successfully.
2. Record screenshots of the dataset, evaluation report, and dashboard.
3. Document this as a simulation proof-of-concept.
4. For real-world validation, collect properly labelled IQ recordings with a suitable receiver and compare against a threshold baseline.
