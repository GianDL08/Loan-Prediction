"""Data pipeline utilities for the Loan Prediction project.

This module captures the data cleaning and EDA steps from the original
`dss_grp8_co3_project.py` notebook and exposes them as reusable functions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load the raw dataset from disk."""
    df = pd.read_csv(path)
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names (strip/lowers/replace spaces/hyphens)."""
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    df.rename(
        columns={
            "coapplicantincome": "coapplicant_income",
            "loanamount": "loan_amount",
            "applicantincome": "applicant_income",
        },
        inplace=True,
    )
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform the data cleaning steps used in the notebook."""
    df = normalize_columns(df)

    # Standardize dependents
    if "dependents" in df.columns:
        df["dependents"] = df["dependents"].replace("3+", "3")

    # Impute common columns with mode/median
    for col in ["gender", "married", "self_employed", "credit_history"]:
        if col in df.columns:
            df.loc[:, col] = df[col].fillna(df[col].mode()[0])

    if "dependents" in df.columns:
        df.loc[:, "dependents"] = pd.to_numeric(df["dependents"], errors="coerce")
        df.loc[:, "dependents"] = df["dependents"].fillna(df["dependents"].median())

    for col in ["loan_amount", "loan_amount_term"]:
        if col in df.columns:
            df.loc[:, col] = df[col].fillna(df[col].median())

    if "credit_history" in df.columns:
        df["credit_history"] = df["credit_history"].astype("Int64")

    return df


def get_missing_summary(df: pd.DataFrame) -> Dict[str, int]:
    """Return missing value counts per column."""
    return df.isnull().sum().to_dict()


def get_duplicate_count(df: pd.DataFrame) -> int:
    """Return number of exact duplicate rows."""
    return int(df.duplicated().sum())


def get_dtype_summary(df: pd.DataFrame) -> Dict[str, str]:
    """Return types for each column."""
    return {col: str(dtype) for col, dtype in df.dtypes.items()}


def get_value_counts(df: pd.DataFrame, columns: List[str]) -> Dict[str, Dict[str, int]]:
    """Return value counts for a set of columns (categorical)."""
    output: Dict[str, Dict[str, int]] = {}
    for col in columns:
        if col in df.columns:
            counts = df[col].fillna("<NA>").astype(str).value_counts(dropna=False)
            output[col] = {str(k): int(v) for k, v in counts.items()}
    return output


def get_numeric_stats(df: pd.DataFrame, columns: List[str]) -> Dict[str, Dict[str, float]]:
    """Return descriptive statistics for numeric columns."""
    output: Dict[str, Dict[str, float]] = {}
    stats = df[columns].describe().to_dict()
    for col, values in stats.items():
        output[col] = {k: float(v) for k, v in values.items()}
    return output


def summarize_for_ui(df: pd.DataFrame) -> Dict[str, Any]:
    """Return a JSON-friendly summary for rendering in the UI."""
    summary: Dict[str, Any] = {
        "shape": df.shape,
        "missing": get_missing_summary(df),
        "duplicate_count": get_duplicate_count(df),
        "dtypes": get_dtype_summary(df),
    }

    # Add common categorical summaries if present
    categorical = [
        "gender",
        "married",
        "education",
        "self_employed",
        "property_area",
        "loan_status",
    ]
    summary["categorical_counts"] = get_value_counts(df, categorical)

    numeric_cols = [
        "applicant_income",
        "coapplicant_income",
        "loan_amount",
        "loan_amount_term",
        "dependents",
        "credit_history",
    ]
    summary["numeric_stats"] = get_numeric_stats(df, [c for c in numeric_cols if c in df.columns])

    return summary


def get_head(df: pd.DataFrame, n: int = 10) -> List[Dict[str, Any]]:
    """Return the first n rows as JSON-friendly dicts."""
    return df.head(n).to_dict(orient="records")


def get_preprocessing_steps() -> List[str]:
    """Return a description of the preprocessing steps applied to the dataset."""
    return [
        "Normalize column names: strip whitespace, lowercase, replace spaces/hyphens with underscores.",
        "Standardize 'Dependents' by converting '3+' to '3' and casting to numeric.",
        "Impute missing values: use mode for categorical features, median for numeric features.",
        "Convert appropriate columns to numeric types (ApplicantIncome, LoanAmount, Credit_History, etc.).",
        "Drop irrelevant columns (Loan_ID) before modeling.",
        "Prepare features and target for model training (Loan_Status mapped to 0/1).",
    ]
