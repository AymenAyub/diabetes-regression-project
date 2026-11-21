"""
Diabetes Progression Predictor — Streamlit app.

Deploys the Ridge regression model trained in diabetes_regression_project.ipynb.

Model file : best_diabetes_model.joblib
            = Pipeline([StandardScaler(), Ridge(alpha=1, random_state=42)])
Dataset    : sklearn Diabetes (REGRESSION; target = disease progression, range 25–346)
Features   : age, sex, bmi, bp, s1, s2, s3, s4, s5, s6  (order matters)

The model was trained on sklearn's *pre-scaled* features, so inputs given in real
units (raw values from load_diabetes(scaled=False)) must be converted to that
scaled space before prediction:

    scaled = (raw - mean) / (std * sqrt(n))

where mean and std (ddof=0) are computed per feature from load_diabetes(scaled=False)
and n = 442 (number of samples).
"""

import pathlib

import numpy as np
import pandas as pd
import altair as alt
import joblib
import streamlit as st
from sklearn.datasets import load_diabetes

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
# Feature order must match the columns the model was trained on.
FEATURE_ORDER = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
MODEL_PATH = pathlib.Path(__file__).parent / "best_diabetes_model.joblib"

# Metrics printed by the notebook (shown for context only — not recomputed here).
CV_R2 = 0.4808
TEST_R2 = 0.4541
TEST_RMSE = 53.7775
TEST_MAE = 42.8120

# ─────────────────────────────────────────────────────────────────────────────
# Cached loaders
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_reference_data():
    """Raw (unscaled) diabetes data + per-feature scaling statistics.

    Returns mins/maxs/means/scale per feature plus the target vector.
    scale = std * sqrt(n) is the denominator of the sklearn scaling formula.
    """
    raw = load_diabetes(scaled=False)
    X, y = raw.data, raw.target
    n = X.shape[0]
    means = X.mean(axis=0)                 # per-feature mean
    stds = X.std(axis=0, ddof=0)           # population std (ddof=0)
    scale = stds * np.sqrt(n)              # sklearn divides by std * sqrt(n)
    return {
        "mins": dict(zip(FEATURE_ORDER, X.min(axis=0))),
        "maxs": dict(zip(FEATURE_ORDER, X.max(axis=0))),
        "means": dict(zip(FEATURE_ORDER, means)),
        "scale": dict(zip(FEATURE_ORDER, scale)),
        "target": y,
        "target_mean": float(y.mean()),
    }


@st.cache_resource
def load_model():
    """Load the saved sklearn pipeline. Returns (model, error_msg)."""
    if not MODEL_PATH.exists():
        return None, f"Model file not found: {MODEL_PATH.name}"
    try:
        return joblib.load(MODEL_PATH), None
    except Exception as exc:
        return None, f"Failed to load model: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# Page config + load data/model, fail fast on errors
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Progression Predictor",
    page_icon="🩺",
    layout="wide",
)

ref = load_reference_data()
model, model_error = load_model()

if model_error:
    st.error(f"⚠️ {model_error}")
    st.info("Re-run `diabetes_regression_project.ipynb` to regenerate `best_diabetes_model.joblib`.")
    st.stop()

# Guard against a feature-name/order mismatch between model and app.
expected = list(getattr(model, "feature_names_in_", FEATURE_ORDER))
if expected != FEATURE_ORDER:
    st.error(f"Feature mismatch: model expects {expected}, app provides {FEATURE_ORDER}.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — model info + disclaimer
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🩺 Diabetes Progression Predictor")
    st.caption("Ridge Regression · sklearn Diabetes Dataset")
    st.markdown("**Model details**")
    st.markdown("- Algorithm: Ridge (α = 1)")
    st.markdown("- Pipeline: StandardScaler → Ridge")
    st.markdown(f"- CV R²: {CV_R2}")
    st.markdown(f"- Test R²: {TEST_R2}")
    st.markdown(f"- Test RMSE: {TEST_RMSE:.2f}")
    st.markdown(f"- Test MAE: {TEST_MAE:.2f}")
    st.warning("⚠️ For educational use only — not medical advice.")

# ─────────────────────────────────────────────────────────────────────────────
# Reset callback — restore every widget key to the dataset mean
# ─────────────────────────────────────────────────────────────────────────────
def reset_to_average():
    for f in FEATURE_ORDER:
        st.session_state[f] = float(ref["means"][f])
    # sex is a radio with codes {1, 2}; pick the code closest to the mean.
    st.session_state["sex"] = 1 if ref["means"]["sex"] < 1.5 else 2

# Initialise widget keys on first run (no value= argument on the widgets).
if "age" not in st.session_state:
    reset_to_average()

# ─────────────────────────────────────────────────────────────────────────────
# Layout — inputs (left) | live prediction (right)
# ─────────────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    st.button("↺ Reset to average", on_click=reset_to_average)

    # Patient card: age, sex, bmi, bp
    with st.container(border=True):
        st.subheader("Patient")
        c1, c2 = st.columns(2)
        with c1:
            st.slider(
                "Age (years)",
                min_value=float(ref["mins"]["age"]),
                max_value=float(ref["maxs"]["age"]),
                step=1.0,
                key="age",
                help="Age in years",
            )
            st.slider(
                "BMI",
                min_value=float(ref["mins"]["bmi"]),
                max_value=float(ref["maxs"]["bmi"]),
                step=0.1,
                key="bmi",
                help="Body mass index",
            )
        with c2:
            # Sex codes are 1 and 2; the dataset does not document which is which.
            st.radio(
                "Sex (code)",
                options=[1, 2],
                key="sex",
                help="Dataset code 1 or 2; meaning not documented",
            )
            st.slider(
                "Blood pressure",
                min_value=float(ref["mins"]["bp"]),
                max_value=float(ref["maxs"]["bp"]),
                step=1.0,
                key="bp",
                help="Average blood pressure (mm Hg)",
            )

    # Blood serum card: s1–s6
    with st.container(border=True):
        st.subheader("Blood serum")
        c1, c2 = st.columns(2)
        with c1:
            st.slider("S1 — Total cholesterol", min_value=float(ref["mins"]["s1"]), max_value=float(ref["maxs"]["s1"]), step=1.0, key="s1")
            st.slider("S2 — LDL", min_value=float(ref["mins"]["s2"]), max_value=float(ref["maxs"]["s2"]), step=0.1, key="s2")
            st.slider("S3 — HDL", min_value=float(ref["mins"]["s3"]), max_value=float(ref["maxs"]["s3"]), step=1.0, key="s3")
        with c2:
            st.slider("S4 — TCH/HDL ratio", min_value=float(ref["mins"]["s4"]), max_value=float(ref["maxs"]["s4"]), step=0.01, key="s4")
            st.slider("S5 — log triglycerides", min_value=float(ref["mins"]["s5"]), max_value=float(ref["maxs"]["s5"]), step=0.01, key="s5")
            st.slider("S6 — Blood sugar", min_value=float(ref["mins"]["s6"]), max_value=float(ref["maxs"]["s6"]), step=1.0, key="s6")

with col_right:
    # Build the raw input vector in training order, then scale to model space.
    raw_vals = np.array([st.session_state[f] for f in FEATURE_ORDER], dtype=float)
    means_arr = np.array([ref["means"][f] for f in FEATURE_ORDER])
    scale_arr = np.array([ref["scale"][f] for f in FEATURE_ORDER])
    # Convert real units → sklearn pre-scaled space: (raw - mean) / (std * sqrt(n))
    scaled = (raw_vals - means_arr) / scale_arr

    try:
        X_df = pd.DataFrame([scaled], columns=FEATURE_ORDER)
        score = float(model.predict(X_df)[0])
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    target = ref["target"]
    target_mean = ref["target_mean"]
    # Percentile = share of dataset targets strictly below the prediction.
    percentile = float(np.mean(target < score) * 100)

    st.subheader("Prediction")
    st.metric(
        "Progression score",
        f"{score:.1f}",
        delta=f"{score - target_mean:+.1f}",
        delta_color="inverse",
    )
    st.caption(f"Dataset target mean: {target_mean:.1f} · range 25–346")
    st.markdown(
        f"**Percentile:** {percentile:.1f}% of patients in the dataset had a lower score."
    )

    # Altair histogram of dataset targets with a red vertical rule at the prediction.
    hist_df = pd.DataFrame({"target": target})
    bars = (
        alt.Chart(hist_df)
        .mark_bar(color="#0D9488")
        .encode(
            alt.X("target:Q", bin=alt.Bin(maxbins=30), title="Disease progression"),
            alt.Y("count():Q", title=None),
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"score": [score]}))
        .mark_rule(color="#DC2626", strokeWidth=2)
        .encode(x="score:Q")
    )
    st.altair_chart((bars + rule).properties(height=260), width="stretch")

    st.info(
        f"Typical error on the test set: RMSE ≈ {TEST_RMSE:.2f}, "
        f"R² ≈ {TEST_R2} (MAE ≈ {TEST_MAE:.2f})."
    )
