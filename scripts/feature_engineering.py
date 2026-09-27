"""
feature_engineering.py

Feature engineering script for the PowerCo SME Customer Churn dataset.
Builds on the output of preprocessing.py.

What this script does:
1. Loads datasets/preprocessed_data.csv
2. Engineers new features from the date columns (contract length,
   time since last modification, time to renewal) - these mirror the
   top churn predictors identified in Related Literature 1
   (Al Mamun et al., 2026), which found contract_interval and
   contract_modification_interval to be the leading churn drivers.
3. Engineers price-change features (combined yearly/6-month price
   movement across all three pricing periods)
4. Engineers a forecast-accuracy ratio (forecast vs actual consumption)
5. Drops columns no longer needed for modelling (id, raw dates)
6. Saves the final model-ready dataset to datasets/model_ready_data.csv

Usage:
    python scripts/feature_engineering.py
"""

import pandas as pd
import numpy as np
import os

INPUT_PATH = os.path.join("datasets", "preprocessed_data.csv")
OUTPUT_PATH = os.path.join("datasets", "model_ready_data.csv")

# A fixed reference date is used (rather than "today") so that this
# script produces the same features every time it is run, regardless
# of the actual run date.
REFERENCE_DATE = pd.Timestamp("2016-01-01")


def main():
    print("Loading preprocessed data...")
    df = pd.read_csv(INPUT_PATH, parse_dates=[
        "date_activ", "date_end", "date_modif_prod", "date_renewal"
    ])
    print(f"Input shape: {df.shape}")

    # --- 1. Date-derived features ---
    df["contract_length_days"] = (df["date_end"] - df["date_activ"]).dt.days
    df["days_since_modification"] = (REFERENCE_DATE - df["date_modif_prod"]).dt.days
    df["days_to_renewal"] = (df["date_renewal"] - REFERENCE_DATE).dt.days

    # --- 2. Price-change features ---
    # Combine the three pricing periods (off-peak, peak, mid-peak) into
    # a single "total price movement" signal for the year and the
    # last 6 months.
    df["total_price_change_year"] = (
        df["var_year_price_off_peak"]
        + df["var_year_price_peak"]
        + df["var_year_price_mid_peak"]
    )
    df["total_price_change_6m"] = (
        df["var_6m_price_off_peak"]
        + df["var_6m_price_peak"]
        + df["var_6m_price_mid_peak"]
    )

    # --- 3. Forecast-accuracy ratio ---
    # How far off the company's own forecast was from actual consumption.
    # +1 avoids division by zero for customers with zero consumption.
    df["forecast_accuracy_ratio"] = (
        df["forecast_cons_12m"] / (df["cons_12m"] + 1)
    )

    # --- 4. Drop columns no longer needed for modelling ---
    df = df.drop(columns=[
        "id", "date_activ", "date_end", "date_modif_prod", "date_renewal"
    ])

    # --- 5. Save ---
    os.makedirs("datasets", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Model-ready data saved to {OUTPUT_PATH}")
    print(f"Final shape: {df.shape}")
    print(f"New engineered columns: contract_length_days, "
          f"days_since_modification, days_to_renewal, "
          f"total_price_change_year, total_price_change_6m, "
          f"forecast_accuracy_ratio")


if __name__ == "__main__":
    main()
