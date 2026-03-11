import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib


def load_and_clean_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Normalize column names to match the notebook
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    df.rename(
        columns={
            "coapplicantincome": "coapplicant_income",
            "loanamount": "loan_amount",
            "applicantincome": "applicant_income",
        },
        inplace=True,
    )

    # Standardize dependents
    if "dependents" in df.columns:
        df["dependents"] = df["dependents"].replace("3+", "3")

    # Fill missing values using sensible defaults (mode/median)
    fill_modes = [
        "gender",
        "married",
        "self_employed",
        "credit_history",
    ]
    for col in fill_modes:
        if col in df.columns:
            df.loc[:, col] = df[col].fillna(df[col].mode()[0])

    if "dependents" in df.columns:
        df["dependents"] = pd.to_numeric(df["dependents"], errors="coerce")
        df["dependents"] = df["dependents"].fillna(df["dependents"].median())

    # Ensure we don't have missing income values
    for col in ["applicant_income", "coapplicant_income"]:
        if col in df.columns:
            df.loc[:, col] = df[col].fillna(df[col].median())

    num_median_cols = ["loan_amount", "loan_amount_term"]
    for col in num_median_cols:
        if col in df.columns:
            df.loc[:, col] = df[col].fillna(df[col].median())

    # Create derived features and apply transformations that match the notebook
    if "applicant_income" in df.columns and "coapplicant_income" in df.columns:
        df["total_income"] = df["applicant_income"] + df["coapplicant_income"]

    # New feature: debt-to-income ratio right before we drop individual income columns.
    # Note: loan_amount is stored in thousands, so convert to unit values for ratio.
    if "total_income" in df.columns and "loan_amount" in df.columns:
        df["debt_to_income_ratio"] = np.where(
            df["total_income"] > 0,
            (df["loan_amount"] * 1000) / df["total_income"],
            0,
        )

    # The model only uses total income (not applicant/coapplicant separately).
    for col in ["applicant_income", "coapplicant_income"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Apply log transforms to the numeric features used by the model.
    for col in ["total_income", "loan_amount", "debt_to_income_ratio"]:
        if col in df.columns:
            df.loc[:, col] = np.log1p(df[col])

    # Force types
    if "credit_history" in df.columns:
        df["credit_history"] = df["credit_history"].astype("Int64")

    # Drop columns that should not be used in modelling
    for col in ["loan_id", "gender"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df


def build_pipeline() -> Pipeline:
    numeric_features = [
        "dependents",
        "total_income",
        "loan_amount",
        "loan_amount_term",
        "credit_history",
        "debt_to_income_ratio",
    ]

    categorical_features = [
        "married",
        "education",
        "self_employed",
        "property_area",
    ]

    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=500)),
        ]
    )

    return pipeline


def train_and_save(data_path: str, model_path: str, return_report: bool = False):
    df = load_and_clean_data(data_path)

    # Map target
    df = df.drop(columns=[c for c in ["loan_id"] if c in df.columns])
    df = df.dropna(subset=["loan_status"])
    df["loan_status"] = df["loan_status"].map({"Y": 1, "N": 0})

    X = df.drop(columns=["loan_status"])
    y = df["loan_status"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    report = classification_report(y_test, predictions, output_dict=True)

    print("Test set classification report:\n")
    print(classification_report(y_test, predictions))

    joblib.dump(pipeline, model_path)
    print(f"Saved trained pipeline to {model_path}")

    if return_report:
        return {
            "report": report,
            "classes": list(pipeline.classes_),
            "features": list(X.columns),
        }


if __name__ == "__main__":
    train_and_save("Loan_Prediction_Dataset.csv", "model.joblib")
