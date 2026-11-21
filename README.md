# Diabetes Progression Predictor

A Streamlit app that predicts diabetes disease progression from patient measurements, using a Ridge Regression model trained on the scikit-learn Diabetes dataset.

> **Educational project only. Not medical advice or a diagnostic tool.**

## What it does

- Takes patient values in real units (age, BMI, blood pressure, six blood serum measurements)
- Converts them to the dataset's scaled form and predicts a **progression score** (dataset range: 25–346; higher = more progression)
- Shows the score against the dataset average, its percentile, and where it falls on the target distribution

This is a **regression** task, not classification.

## Model

| Item | Value |
|---|---|
| Pipeline | `StandardScaler` → `Ridge(alpha=1)` |
| Data | scikit-learn Diabetes dataset (442 patients, 10 features) |
| CV R² | 0.4808 |
| Test R² | 0.4541 |
| Test RMSE | 53.78 |
| Test MAE | 42.81 |

The model explains under half of the variance, so treat predictions as rough estimates.

## Files

| File | Purpose |
|---|---|
| `main.py` | Streamlit app |
| `requirements.txt` | Dependencies |
| `best_diabetes_model.joblib` | Trained pipeline saved from the notebook |
| `diabetes_regression_project.ipynb` | Training and evaluation notebook |

## Run locally

```bash
pip install -r requirements.txt
streamlit run main.py
```

> Use the same `scikit-learn` version that was used to train the model, otherwise loading the `.joblib` file may fail.

## Deploy

Push the files to GitHub, then create the app on [Streamlit Community Cloud](https://share.streamlit.io) with `main.py` as the main file.
