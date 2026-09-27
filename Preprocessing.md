# Preprocessing

This file describes the data preprocessing step for the STADIOchoice
churn prediction project, using the PowerCo SME Customer Churn dataset
described in Part A.

## Which file does this?

**Script:** [`scripts/preprocessing.py`](./scripts/preprocessing.py)

## What this script does

1. **Loads the raw dataset** from `datasets/public_churn_dataset_clean.csv` (the
   PowerCo dataset described in Part A's dataset table).
2. **Parses date columns** (`date_activ`, `date_end`,
   `date_modif_prod`, `date_renewal`) into proper datetime format, so
   they can be used to build date-based features later in
   [`FeatureEngineering.md`](./FeatureEngineering.md).
3. **Encodes categorical columns:**
   - `has_gas` (binary "t"/"f") is mapped to 1/0.
   - `channel_sales` and `origin_up` (hashed, low-cardinality
     categories, including an explicit `"MISSING"` category already
     present in the raw data) are one-hot encoded. One-hot encoding
     was chosen over label encoding because these categories are not
     ordinal (a channel code has no natural order), so label encoding
     would wrongly imply a ranking between them.
4. **Caps outliers** on key numeric columns (consumption, forecast,
   and margin columns) using the IQR method: any value below
   `Q1 − 1.5×IQR` or above `Q3 + 1.5×IQR` is capped to that boundary.
   This follows the same approach used in Related Literature 1
   (Al Mamun et al., 2026), which used this exact dataset.
5. **Saves the cleaned dataset** to `datasets/preprocessed_data.csv`.

No missing-value imputation was required, since the PowerCo dataset
supplied for this project (`clean_data_after_eda.csv`) had already had
missing values resolved (the `"MISSING"` category is an explicit,
pre-existing label in the categorical columns rather than a null
value).

## How to run it

1. Make sure `datasets/public_churn_dataset_clean.csv` exists (the raw PowerCo
   dataset).
2. Install dependencies: `pip install -r requirements.txt`
3. From the repository root, run:
   ```
   python scripts/preprocessing.py
   ```
4. Output: `datasets/preprocessed_data.csv` (14,606 rows × 56 columns).

## Next step

The output of this script is used directly by
[`scripts/feature_engineering.py`](./scripts/feature_engineering.py) —
see [`FeatureEngineering.md`](./FeatureEngineering.md).
