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
import pandas as pd
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

    model = joblib.load(MODEL_FILE)
    # Recompute report to keep it consistent with the data
    report = train_model.train_and_save(
        str(DATA_FILE), str(MODEL_FILE), return_report=True
    )
    return model, report


def predict_from_inputs(model, inputs: dict) -> dict:
    """Run a prediction from a normalized input dict."""
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
            gender = st.selectbox("Gender", options=["Male", "Female"], index=0)
            married = st.selectbox(
                "Married", options=["Yes", "No"], index=0
            )
            dependents = st.selectbox(
                "Dependents", options=["0", "1", "2", "3"], index=0
            )
            education = st.selectbox(
                "Education", options=["Graduate", "Not Graduate"], index=0
            )
            self_employed = st.selectbox(
                "Self Employed", options=["Yes", "No"], index=1
            )

        with col2:
            applicant_income = st.number_input(
                "Applicant Income", min_value=0.0, value=0.0, step=100.0
            )
            coapplicant_income = st.number_input(
                "Coapplicant Income", min_value=0.0, value=0.0, step=100.0
            )
            loan_amount = st.number_input(
                "Loan Amount (in thousands)", min_value=0.0, value=100.0, step=10.0
            )
            loan_term = st.number_input(
                "Loan Amount Term (in days)", min_value=0.0, value=360.0, step=12.0
            )
            credit_history = st.selectbox(
                "Credit History", options=[0, 1], index=1
            )
            property_area = st.selectbox(
                "Property Area",
                options=["Urban", "Semiurban", "Rural"],
                index=0,
            )

        submitted = st.form_submit_button("Predict")

    if submitted:
        inputs = {
            "gender": gender,
            "married": married,
            "dependents": int(dependents),
            "education": education,
            "self_employed": self_employed,
            "applicant_income": applicant_income,
            "coapplicant_income": coapplicant_income,
            "loan_amount": loan_amount,
            "loan_amount_term": int(loan_term),
            "credit_history": credit_history,
            "property_area": property_area,
        }

        result = predict_from_inputs(model, inputs)

        st.subheader("Prediction")
        st.write(f"**Loan Approved:** {result['prediction_label']}")

        # Confidence / risk breakdown
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

        st.write(f"**Approval confidence:** {confidence_pct:.1f}%")
        st.write(f"**Approval probability:** {approve_pct:.1f}%")
        st.write(f"**Rejection probability:** {reject_pct:.1f}%")
        st.write(
            f"**Risk estimate (if accepted):** {100 - approve_pct:.1f}% ({risk_tier} risk)"
        )


def main():
    st.set_page_config(page_title="Loan Prediction", layout="wide")

    raw_df = load_data()
    clean_df = clean_data(raw_df)
    model, model_report = build_or_load_model()

    st.title("Loan Prediction Explorer")

    tabs = st.tabs(["Predict", "Data Summary", "EDA", "Preprocessing", "Model"])

    with tabs[0]:
        render_prediction_ui(model)

    with tabs[1]:
        st.subheader("Dataset Summary")
        st.write(summarize_for_ui(clean_df))

    with tabs[2]:
        st.subheader("Exploratory Data Analysis (Sample)")
        st.write("### Cleaned Data Head")
        st.dataframe(get_head(clean_df, n=10))
        st.write("### Raw Data Head")
        st.dataframe(get_head(raw_df, n=10))

    with tabs[3]:
        st.subheader("Preprocessing Steps")
        steps = get_preprocessing_steps()
        for step in steps:
            st.write(f"- {step}")

    with tabs[4]:
        st.subheader("Model Information")
        st.write("**Model type:** Logistic Regression")
        st.write("**Features used:**")
        st.write(model_report.get("features", []))
        st.write("**Evaluation metrics (test set)**")
        st.json(model_report.get("report", {}))


if __name__ == "__main__":
    main()
