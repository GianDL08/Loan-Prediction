import json
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
import joblib

from data_pipeline import clean_data, load_raw_data, summarize_for_ui


MODEL_FILE = Path(__file__).parent / "model.joblib"
DATA_FILE = Path(__file__).parent / "Loan_Prediction_Dataset.csv"

app = Flask(
    __name__, static_folder=".", static_url_path="",
)


def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "Model checkpoint not found. Run `python train_model.py` to create model.joblib.`"
        )
    return joblib.load(MODEL_FILE)


# Load resources once at startup
model = load_model()
raw_df = load_raw_data(DATA_FILE)
clean_df = clean_data(raw_df)


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


@app.route("/", methods=["GET"])
def index():
    return send_from_directory(app.static_folder, "index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
