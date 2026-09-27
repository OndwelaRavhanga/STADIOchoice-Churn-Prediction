# Model 2: Random Forest

This file describes Model 2 for the STADIOchoice churn prediction
project, trained on the PowerCo SME Customer Churn dataset described
in Part A.

## Which file does this?

**Script:** [`scripts/model2_random_forest.py`](./scripts/model2_random_forest.py)

## Why Random Forest?

Random Forest is used as a **stronger ensemble model** to compare
against the Logistic Regression baseline (Model 1). Unlike Logistic
Regression, it can capture non-linear relationships and interactions
between features (e.g. how contract length and pricing changes might
combine to affect churn risk). Related Literature 1 (Al Mamun et al.,
2026), which used this exact PowerCo dataset, found Random Forest and
other ensemble tree models achieved strong results (94% accuracy) with
this configuration, so the same hyperparameters were adopted here as a
justified, evidence-based starting point.

## What this script does

1. Loads `datasets/model_ready_data.csv` (the output of
   [`scripts/feature_engineering.py`](./scripts/feature_engineering.py) —
   see [`FeatureEngineering.md`](./FeatureEngineering.md)).
2. Splits the data into training (80%) and test (20%) sets, using the
   **same random seed and stratified split** as Model 1, so both
   models are trained and tested on identical data for a fair
   comparison in Part C.
3. Trains a `RandomForestClassifier` model. Unlike Model 1, no feature
   scaling is needed, since tree-based models split on raw feature
   values rather than distances or weighted sums.

   | Hyperparameter | Value | Reason |
   |---|---|---|
   | `n_estimators` | 300 | Number of trees in the forest; follows Al Mamun et al. (2026), who used this value on the same dataset. |
   | `max_depth` | 10 | Limits how deep each tree can grow, reducing the risk of overfitting to the training data. |
   | `criterion` | `"gini"` | Standard splitting criterion for classification trees. |
   | `class_weight` | `"balanced"` | Automatically up-weights the minority (churn) class, to counter the dataset's ~9:1 class imbalance. |
   | `random_state` | 42 | Fixed seed, so results are reproducible on every run. |

4. Prints sanity-check metrics (accuracy, precision, recall, F1,
   ROC-AUC) on the test set, plus the top 10 most important features
   according to the model. **Full performance reporting and comparison
   against Model 1 is covered separately in Part C**, not in this
   script.
5. Saves the trained model to `models/random_forest.pkl`.

## A useful finding from running this script

When tested, four of the six features engineered in
[`FeatureEngineering.md`](./FeatureEngineering.md) —
`forecast_accuracy_ratio`, `days_since_modification`,
`days_to_renewal`, and `contract_length_days` — appeared in the
model's top 10 most important features. This directly supports the
finding in Related Literature 1, which identified contract-related and
modification-related timing features as leading churn predictors on
this same dataset.

## How to run it

1. Make sure `datasets/model_ready_data.csv` exists (see
   [`FeatureEngineering.md`](./FeatureEngineering.md)).
2. Install dependencies: `pip install -r requirements.txt`
3. From the repository root, run:
   ```
   python scripts/model2_random_forest.py
   ```
4. Output: `models/random_forest.pkl`, plus printed metrics and
   feature importances in the terminal.

## Compare with

[`Model1.md`](./Model1.md) — Logistic Regression, the baseline model
trained on the same data split for a fair comparison.
