"""Streamlit app for Loan Prediction.

This script is designed to be run with Streamlit:

    streamlit run app.py

It reuses the existing data pipeline and trained model logic from the
project to provide an interactive UI for making predictions and exploring
model / data information.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

import train_model
from data_pipeline import (
    clean_data,
    get_head,
    get_preprocessing_steps,
    load_raw_data,
    summarize_for_ui,
)

MODEL_FILE = Path(__file__).parent / "model.joblib"
DATA_FILE = Path(__file__).parent / "Loan_Prediction_Dataset.csv"


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Load the raw dataset from disk."""

    return load_raw_data(DATA_FILE)


@st.cache_resource(show_spinner=False)
def build_or_load_model() -> tuple:
    """Ensure a trained model exists and return (model, report)."""
    if not MODEL_FILE.exists():
        report = train_model.train_and_save(
            str(DATA_FILE), str(MODEL_FILE), return_report=True
        )
        model = joblib.load(MODEL_FILE)
        return model, report

    # Retrain the model to ensure it matches the current preprocessing/feature set.
    report = train_model.train_and_save(
        str(DATA_FILE), str(MODEL_FILE), return_report=True
    )
    model = joblib.load(MODEL_FILE)
    return model, report


def predict_from_inputs(model, inputs: dict) -> dict:
    """Run a prediction from a normalized input dict.

    The training pipeline expects log-transformed income/loan features and a derived
    `total_income` feature (matching `train_model.load_and_clean_data`).
    """
    inputs = inputs.copy()

    # Ensure numeric inputs are floats so downstream preprocessing behaves consistently.
    # (JSON may deserialize whole numbers as int; sklearn pipelines expect floats.)
    for col in ["total_income", "loan_amount", "loan_amount_term"]:
        if col in inputs:
            try:
                inputs[col] = float(inputs[col])
            except (TypeError, ValueError):
                pass

    # Create derived features to match training pipeline
    if "total_income" in inputs and "loan_amount" in inputs:
        # loan_amount is in thousands in the dataset.
        income = inputs.get("total_income", 0.0)
        loan = inputs.get("loan_amount", 0.0)
        inputs["debt_to_income_ratio"] = (loan * 1000) / income if income > 0 else 0.0

    # Apply the same log1p transformations used during training.
    for col in ["total_income", "loan_amount", "debt_to_income_ratio"]:
        if col in inputs:
            inputs[col] = np.log1p(inputs[col])

    df = pd.DataFrame([inputs])
    pred_proba = model.predict_proba(df)[0]
    pred = model.predict(df)[0]

    # Map probabilities to class labels (e.g. 0 = No, 1 = Yes)
    class_proba = dict(zip(model.classes_, pred_proba))
    approve_proba = float(class_proba.get(1, 0.0))
    reject_proba = float(class_proba.get(0, 0.0))

    # Use the model's confidence in the selected class as the primary confidence metric.
    selected_proba = float(class_proba.get(pred, 0.0))

    return {
        "prediction": int(pred),
        "prediction_label": "Y" if pred == 1 else "N",
        "selected_probability": selected_proba,
        "approve_probability": approve_proba,
        "reject_probability": reject_proba,
    }


def render_prediction_ui(model):
    """Render the prediction form and output."""

    st.header("Loan Approval Prediction")

    st.markdown(
        """
Provide the applicant profile below and click **Predict** to see the model's loan approval prediction.
"""
    )

    with st.form(key="prediction_form"):
        st.subheader("Applicant details")

        col1, col2 = st.columns(2)
        with col1:
            married = st.selectbox(
                "Married", options=["Yes", "No"], index=0
            )
            dependents = st.selectbox(
                "Number of Dependents", options=["0", "1", "2", "3"], index=0
            )
            education = st.selectbox(
                "Education", options=["Graduate", "Not Graduate"], index=0
            )
            self_employed = st.selectbox(
                "Self Employed", options=["Yes", "No"], index=1
            )
            property_area = st.selectbox(
                "Property Area",
                options=["Urban", "Semiurban", "Rural"],
                index=0,
            )

        with col2:
            total_income = st.number_input(
                "Total Income", min_value=0.0, value=0.0, step=100.0
            )
            loan_amount = st.number_input(
                "Loan Amount (in thousands)", min_value=0.0, value=100.0, step=10.0
            )
            loan_term = st.number_input(
                "Loan Amount Term (in months)", min_value=0.0, value=360.0, step=12.0
            )
            credit_history = st.selectbox(
                "Credit History", options=[0, 1], index=1
            )
            

        submitted = st.form_submit_button("Predict")

    if submitted:
        inputs = {
            "married": married,
            "dependents": int(dependents),
            "education": education,
            "self_employed": self_employed,
            "total_income": total_income,
            "loan_amount": loan_amount,
            "loan_amount_term": int(loan_term),
            "credit_history": credit_history,
            "property_area": property_area,
        }

        result = predict_from_inputs(model, inputs)

        # Metrics used to describe result
        approve_pct = result['approve_probability'] * 100
        reject_pct = result['reject_probability'] * 100
        confidence_pct = result['selected_probability'] * 100

        # A simple risk tier based on approval probability.
        if approve_pct >= 80:
            risk_tier = "Low"
        elif approve_pct >= 60:
            risk_tier = "Moderate"
        elif approve_pct >= 40:
            risk_tier = "High"
        else:
            risk_tier = "Very high"

        # Risk score: higher = more risky
        risk_score = 100 - approve_pct

        # Map risk score to a color gradient (green -> yellow -> red)
        def _risk_color(score: float) -> str:
            # Normalize to 0..1
            t = max(0.0, min(1.0, score / 100.0))
            # Use a simple green->yellow->red gradient
            if t < 0.5:
                # green to yellow
                r = int(255 * (t * 2))
                g = 255
            else:
                # yellow to red
                r = 255
                g = int(255 * (1 - (t - 0.5) * 2))
            b = 0
            return f"rgb({r},{g},{b})"

        risk_color = _risk_color(risk_score)

        approved = result['prediction_label'] == "Y"
        status_color = "#218838" if approved else "#C82333"  # green / red
        status_label = "APPROVED" if approved else "REJECTED"

        st.subheader("Prediction")

        # Layout: status badge + details
        col_status, col_details = st.columns([1, 2])

        with col_status:
            st.markdown(
                f"<div style='padding:18px 14px; border-radius:14px; background: {status_color}; color: #fff; text-align: center;'>"
                f"<div style='font-size:22px; font-weight:600; margin-bottom:6px;'>{status_label}</div>"
                f"<div style='font-size:14px; opacity:.9;'>Confidence: {confidence_pct:.1f}%</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Risk bar (green -> red)
            st.markdown(
                f"<div style='margin-top:12px; padding:12px; border-radius:12px; background: #f5f5f5; color: #0b0c0d;'>"
                f"<div style='font-weight:600; margin-bottom:6px;'>Risk estimate (if accepted)</div>"
                f"<div style='position: relative; height: 16px; border-radius: 10px; background: #ddd;'>"
                f"<div style='width: {risk_score:.1f}%; height: 100%; border-radius: 10px; background: {risk_color};'></div>"
                f"</div>"
                f"<div style='margin-top:6px; font-size:12px; opacity:.85;'>{risk_score:.1f}% ({risk_tier} risk)</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        with col_details:
            st.markdown(
                "<div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px; color: #0b0c0d;'>"
                f"<div style='padding:12px; border-radius:12px; background:#f2f8ff; color: #0b0c0d;'>"
                f"<div style='font-size:12px; color:#0b0c0d; font-weight:600;'>Approval probability</div>"
                f"<div style='font-size:20px; font-weight:700; margin-top:6px;'>{approve_pct:.1f}%</div>"
                f"</div>"
                f"<div style='padding:12px; border-radius:12px; background:#fff1f1; color: #0b0c0d;'>"
                f"<div style='font-size:12px; color:#0b0c0d; font-weight:600;'>Rejection probability</div>"
                f"<div style='font-size:20px; font-weight:700; margin-top:6px;'>{reject_pct:.1f}%</div>"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )


def render_feature_importance(model):
    """Render a bar chart showing feature importance from the trained model."""

    st.header("Feature Importance")
    st.markdown(
        """
This chart shows how much each feature contributes to the model's decision.

Higher importance means the model relies more heavily on that feature when making predictions.
"""
    )

    try:
        importances = model.named_steps["classifier"].feature_importances_
    except Exception:
        st.warning("Unable to extract feature importances from the trained model.")
        return

    # Attempt to get feature names from the preprocessing pipeline
    feature_names = None
    try:
        preprocessor = model.named_steps["preprocessor"]
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = None

    if feature_names is None or len(feature_names) != len(importances):
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False)

    fig, ax = plt.subplots(figsize=(10, max(4, len(imp_df) * 0.25)))
    sns.barplot(data=imp_df, x="importance", y="feature", palette="viridis", ax=ax)
    ax.set_title("Model feature importances")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Interpretation of feature importances")
    st.markdown(
        """
Below are some general interpretations of the features shown in the chart. The model uses these to decide whether a loan will be approved.

- **Total income**: Higher income often makes approval more likely, since it indicates greater ability to repay.
- **Loan amount**: Larger loans typically make approval harder, since they increase repayment burden.
- **Loan term**: Longer terms can reduce monthly payment pressure, but may also increase risk.
- **Credit history**: A positive credit history is usually the strongest indicator of approval.
- **Debt-to-income ratio**: Higher ratios mean the applicant may be over-leveraged, reducing approval chances.
- **Dependents**: More dependents can reduce approval likelihood because it increases living expenses.
- **Married**: Married applicants often have slightly higher approval rates due to more stable household finances.
- **Education**: Graduates may be seen as lower risk compared to non-graduates.
- **Self employed**: Self-employed applicants can be seen as riskier due to income variability.
- **Property area**: Applicants from urban/semiurban areas may have different risk profiles than rural applicants.

Keep in mind that the importance values are specific to this model and dataset; they show what the model learned as useful signals, not causal relationships.
"""
    )


def render_correlation_heatmap(clean_df: pd.DataFrame):
    """Render a correlation heatmap showing relationships between numeric inputs."""

    st.header("Feature Correlations")
    st.markdown(
        """
This heatmap shows linear correlations (Pearson r) between the numeric features used by the model.

A value close to +1 indicates a strong positive relationship; a value close to -1 indicates a strong negative relationship.
"""
    )

    numeric_features = [
        "dependents",
        "total_income",
        "loan_amount",
        "loan_amount_term",
        "credit_history",
        "debt_to_income_ratio",
    ]
    available = [c for c in numeric_features if c in clean_df.columns]

    if not available:
        st.warning("No numeric features available for correlation analysis.")
        return

    corr = clean_df[available].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Correlation matrix (numeric inputs)")
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("How to interpret these correlations")
    st.markdown(
        """
- Strong positive values (close to +1) mean the two features tend to increase together.
- Strong negative values (close to -1) mean the features move in opposite directions.
- Values near 0 indicate little to no linear relationship.

Some specific points to watch for:

- A strong positive correlation between **total_income** and **loan_amount** may indicate that higher earners request larger loans.
- A strong positive correlation between **total_income** and **debt_to_income_ratio** (if present) can mean that higher incomes are accompanied by higher debt obligations, which could affect approval.
- A strong negative correlation between **credit_history** and **loan_amount** could indicate that applicants with stronger credit tend to take smaller loans (or vice versa).

These correlations do not imply causation, but they help you understand which features tend to move together in the dataset.
"""
    )


def main():
    st.set_page_config(page_title="Loan Prediction", layout="wide")

    raw_df = load_data()
    clean_df = clean_data(raw_df)
    model, model_report = build_or_load_model()

    st.title("Loan Prediction Explorer")

    tabs = st.tabs(["Predict", "Feature importance", "Correlations"])

    with tabs[0]:
        render_prediction_ui(model)

    with tabs[1]:
        render_feature_importance(model)

    with tabs[2]:
        render_correlation_heatmap(clean_df)


if __name__ == "__main__":
    main()
