# Feature Engineering

This file describes the feature engineering step for the STADIOchoice
churn prediction project, using the PowerCo SME Customer Churn dataset
described in Part A.

## Which file does this?

**Script:** [`scripts/feature_engineering.py`](./scripts/feature_engineering.py)

## What this script does

This script builds on the cleaned data produced by
[`scripts/preprocessing.py`](./scripts/preprocessing.py) (see
[`Preprocessing.md`](./Preprocessing.md)) and engineers six new
features:

1. **`contract_length_days`** — the number of days between
   `date_activ` and `date_end` (total contract length).
2. **`days_since_modification`** — days between `date_modif_prod` and
   a fixed reference date, showing how recently the customer's product
   was last changed.
3. **`days_to_renewal`** — days between the fixed reference date and
   `date_renewal`, showing how soon the customer's contract is next up
   for renewal.
4. **`total_price_change_year`** — the combined yearly price movement
   across all three pricing periods (off-peak, peak, mid-peak).
5. **`total_price_change_6m`** — the same, but over the last 6 months.
6. **`forecast_accuracy_ratio`** — the company's forecasted 12-month
   consumption divided by actual 12-month consumption, showing how
   accurate (or inaccurate) PowerCo's own forecasting was for that
   customer.

**Why these features:** Related Literature 1 (Al Mamun et al., 2026),
which used this same PowerCo dataset, found that `contract_interval`
and `contract_modification_interval` were the two leading predictors
of churn. The `contract_length_days` and `days_since_modification`
features engineered here are built directly from the same underlying
date information, so this project can test whether the same pattern
holds. (This was confirmed when training Model 2 — see
[`Model2.md`](./Model2.md).)

After engineering these features, the script drops columns no longer
needed for modelling: `id` (not predictive) and the four raw date
columns (now represented by the engineered features above).

The final dataset is saved to `datasets/model_ready_data.csv`.

## How to run it

1. Make sure `datasets/preprocessed_data.csv` exists (see
   [`Preprocessing.md`](./Preprocessing.md)).
2. Install dependencies: `pip install -r requirements.txt`
3. From the repository root, run:
   ```
   python scripts/feature_engineering.py
   ```
4. Output: `datasets/model_ready_data.csv` (14,606 rows × 57 columns).

## Next step

The output of this script is used directly by both
[`scripts/model1_logistic_regression.py`](./scripts/model1_logistic_regression.py)
and
[`scripts/model2_random_forest.py`](./scripts/model2_random_forest.py) —
see [`Model1.md`](./Model1.md) and [`Model2.md`](./Model2.md).
