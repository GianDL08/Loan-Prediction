import json
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
import joblib

import train_model
from data_pipeline import clean_data, load_raw_data, summarize_for_ui, get_head, get_preprocessing_steps


MODEL_FILE = Path(__file__).parent / "model.joblib"
DATA_FILE = Path(__file__).parent / "Loan_Prediction_Dataset.csv"

app = Flask(
    __name__, static_folder=".", static_url_path="",
)


def build_or_load_model() -> tuple:
    """Ensure a trained model exists and return (model, report)."""
    if not MODEL_FILE.exists():
        report = train_model.train_and_save(str(DATA_FILE), str(MODEL_FILE), return_report=True)
        model = joblib.load(MODEL_FILE)
        return model, report

    model = joblib.load(MODEL_FILE)
    # Recompute report to keep it consistent with data
    report = train_model.train_and_save(str(DATA_FILE), str(MODEL_FILE), return_report=True)
    return model, report


# Load resources once at startup
raw_df = load_raw_data(DATA_FILE)
clean_df = clean_data(raw_df)
model, model_report = build_or_load_model()


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Request body must be JSON."}), 400

    required_fields = [
        "gender",
        "married",
        "dependents",
        "education",
        "selfEmployed",
        "applicantIncome",
        "coapplicantIncome",
        "loanAmount",
        "loanTerm",
        "credit_history",
        "propertyArea",
    ]

    missing = [f for f in required_fields if f not in payload]
    if missing:
        return (
            jsonify({"error": "Missing required fields", "missing": missing}),
            400,
        )

    # Normalize the record to match training schema
    record = {
        "gender": payload["gender"],
        "married": payload["married"],
        "dependents": payload["dependents"],
        "education": payload["education"],
        "self_employed": payload["selfEmployed"],
        "applicant_income": payload["applicantIncome"],
        "coapplicant_income": payload["coapplicantIncome"],
        "loan_amount": payload["loanAmount"],
        "loan_amount_term": payload["loanTerm"],
        "credit_history": payload["credit_history"],
        "property_area": payload["propertyArea"],
    }

    # Keep as DataFrame for pipeline compatibility
    try:
        import pandas as pd

        df = pd.DataFrame([record])
    except Exception:
        return jsonify({"error": "Failed to build input data frame."}), 400

    pred_proba = model.predict_proba(df)[0]
    pred = model.predict(df)[0]

    return jsonify(
        {
            "prediction": int(pred),
            "probability": float(pred_proba[pred]),
            "prediction_label": "Y" if pred == 1 else "N",
        }
    )


@app.route("/api/summary", methods=["GET"])
def summary():
    """Return data cleaning / EDA summary information."""
    return jsonify(summarize_for_ui(clean_df))


@app.route("/api/eda", methods=["GET"])
def eda():
    """Return EDA tables and data for UI rendering."""
    return jsonify(
        {
            "head": get_head(clean_df, n=10),
            "summary": summarize_for_ui(clean_df),
        }
    )


@app.route("/api/preprocessing", methods=["GET"])
def preprocessing():
    """Return preprocessing steps and sample of cleaned data."""
    return jsonify(
        {
            "steps": get_preprocessing_steps(),
            "raw_head": get_head(raw_df, n=5),
            "clean_head": get_head(clean_df, n=5),
        }
    )


@app.route("/api/model", methods=["GET"])
def model_info():
    """Return model details and evaluation metrics."""
    return jsonify(
        {
            "model_type": "Logistic Regression",
            "metrics": model_report.get("report", {}),
            "classes": model_report.get("classes", []),
            "features": model_report.get("features", []),
        }
    )


@app.route("/", methods=["GET"])
def index():
    return send_from_directory(app.static_folder, "index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
