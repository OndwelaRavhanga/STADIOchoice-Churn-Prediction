"""
preprocessing.py

Preprocessing script for the PowerCo SME Customer Churn dataset.

What this script does:
1. Loads the raw dataset (datasets/public_churn_dataset_clean.csv)
2. Converts date columns to proper datetime type
3. Encodes categorical columns (one-hot encoding)
4. Caps outliers on key numeric columns using the IQR method
5. Saves the cleaned dataset to datasets/preprocessed_data.csv

Usage:
    python scripts/preprocessing.py
"""

import pandas as pd
import numpy as np
import os

RAW_DATA_PATH = os.path.join("datasets", "public_churn_dataset_clean.csv")
OUTPUT_PATH = os.path.join("datasets", "preprocessed_data.csv")

# Columns where extreme outliers can distort the model (based on EDA:
# these are the consumption, forecast, and margin columns with the
# largest spread / most extreme max values).
OUTLIER_COLUMNS = [
    "cons_12m", "cons_gas_12m", "cons_last_month",
    "forecast_cons_12m", "forecast_cons_year", "forecast_meter_rent_12m",
    "imp_cons", "margin_gross_pow_ele", "margin_net_pow_ele", "net_margin",
    "pow_max",
]

CATEGORICAL_COLUMNS = ["channel_sales", "has_gas", "origin_up"]

DATE_COLUMNS = ["date_activ", "date_end", "date_modif_prod", "date_renewal"]


def cap_outliers_iqr(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Cap values below Q1-1.5*IQR or above Q3+1.5*IQR to those bounds."""
    df = df.copy()
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[col] = df[col].clip(lower=lower, upper=upper)
    return df


def main():
    print("Loading raw data...")
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Raw shape: {df.shape}")

    # --- 1. Parse dates ---
    for col in DATE_COLUMNS:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # --- 2. Encode categorical columns ---
    # has_gas is binary (t/f) -> map to 1/0
    df["has_gas"] = df["has_gas"].map({"t": 1, "f": 0})

    # channel_sales and origin_up are hashed, low-cardinality categories,
    # including an explicit "MISSING" category already present in the data.
    # One-hot encoding keeps this information without implying any order.
    df = pd.get_dummies(
        df, columns=["channel_sales", "origin_up"],
        prefix=["channel", "origin"], dummy_na=False
    )

    # --- 3. Cap outliers on key numeric columns ---
    print("Capping outliers using IQR method...")
    df = cap_outliers_iqr(df, OUTLIER_COLUMNS)

    # --- 4. Save ---
    os.makedirs("datasets", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Preprocessed data saved to {OUTPUT_PATH}")
    print(f"Final shape: {df.shape}")


if __name__ == "__main__":
    main()
