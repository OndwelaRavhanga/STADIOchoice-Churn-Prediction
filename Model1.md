# Model 1: Logistic Regression

This file describes Model 1 for the STADIOchoice churn prediction
project, trained on the PowerCo SME Customer Churn dataset described
in Part A.

## Which file does this?

**Script:** [`scripts/model1_logistic_regression.py`](./scripts/model1_logistic_regression.py)

## Why Logistic Regression?

Logistic Regression is used as an **interpretable baseline model**. It
is fast to train, and each feature gets a clear, explainable
coefficient (positive or negative influence on churn), which is
important for eventually explaining predictions to non-technical
STADIOchoice stakeholders (see Related Literature 3, Chang et al.,
2024, which highlights explainability as a key requirement for
business-facing churn models). It also gives a sensible reference
point to judge whether the more complex Model 2 (Random Forest) is
actually worth its added complexity.

## What this script does

1. Loads `datasets/model_ready_data.csv` (the output of
   [`scripts/feature_engineering.py`](./scripts/feature_engineering.py) —
   see [`FeatureEngineering.md`](./FeatureEngineering.md)).
2. Splits the data into training (80%) and test (20%) sets, using
   stratified sampling on the `churn` column so both sets keep the
   same ~9.72% churn rate as the full dataset.
3. Scales all features using `StandardScaler`. This step is required
   for Logistic Regression specifically, since it is sensitive to the
   scale of input features (unlike tree-based models such as Model 2).
4. Trains a `LogisticRegression` model with the following
   hyperparameters:

   | Hyperparameter | Value | Reason |
   |---|---|---|
   | `C` | 1.0 | Default regularisation strength; balances fitting the data against overfitting. |
   | `max_iter` | 1000 | Increased from scikit-learn's default of 100, to ensure the model fully converges given the number of features. |
   | `class_weight` | `"balanced"` | Automatically up-weights the minority (churn) class, to counter the dataset's ~9:1 class imbalance (9.72% churn) rather than letting the model default to always predicting "no churn". |
   | `random_state` | 42 | Fixed seed, so results are reproducible on every run. |

5. Prints sanity-check metrics (accuracy, precision, recall, F1,
   ROC-AUC) on the test set, to confirm the model trained correctly.
   **Full performance reporting and comparison against Model 2 is
   covered separately in Part C**, not in this script.
6. Saves the trained model (together with the fitted scaler, since
   both are needed to make predictions on new data) to
   `models/logistic_regression.pkl`.

## How to run it

1. Make sure `datasets/model_ready_data.csv` exists (see
   [`FeatureEngineering.md`](./FeatureEngineering.md)).
2. Install dependencies: `pip install -r requirements.txt`
3. From the repository root, run:
   ```
   python scripts/model1_logistic_regression.py
   ```
4. Output: `models/logistic_regression.pkl`, plus printed metrics in
   the terminal.

## Compare with

[`Model2.md`](./Model2.md) — Random Forest, the second model trained
on the same data split for a fair comparison.
