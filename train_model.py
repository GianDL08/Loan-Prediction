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
        df.loc[:, "dependents"] = pd.to_numeric(df["dependents"], errors="coerce")
        df.loc[:, "dependents"] = df["dependents"].fillna(df["dependents"].median())

    num_median_cols = ["loan_amount", "loan_amount_term"]
    for col in num_median_cols:
        if col in df.columns:
            df.loc[:, col] = df[col].fillna(df[col].median())

    # Force types
    if "credit_history" in df.columns:
        df["credit_history"] = df["credit_history"].astype("Int64")

    return df


def build_pipeline() -> Pipeline:
    numeric_features = [
        "applicant_income",
        "coapplicant_income",
        "loan_amount",
        "loan_amount_term",
        "dependents",
        "credit_history",
    ]

    categorical_features = [
        "gender",
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
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
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
