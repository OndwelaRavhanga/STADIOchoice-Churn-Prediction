# Model 2 Performance: Random Forest

This file reports the performance of Model 2 (Random Forest) on the
held-out test set, using the PowerCo SME Customer Churn dataset
described in Part A.

## Which file does this?

**Script:** [`scripts/model2_performance.py`](./scripts/model2_performance.py)

## What this script does

1. Loads `datasets/model_ready_data.csv` and re-creates the **exact
   same** train/test split used when training Model 2 (same random
   seed, same 80/20 split, same stratification on `churn`) — this is
   also identical to the split used in
   [`Model1Performance.md`](./Model1Performance.md), so both models
   are judged on precisely the same unseen data.
2. Loads the trained model from `models/random_forest.pkl`.
3. Computes accuracy, precision, recall, F1-score, ROC-AUC, and a full
   confusion matrix.
4. Saves all results (plus the raw predictions, for reuse by
   [`Comparison.md`](./Comparison.md)) to
   `experimental_results/model2_performance.json`.

## Results

| Metric | Value |
|---|---|
| Accuracy | 0.865 |
| Precision | 0.289 |
| Recall | 0.268 |
| F1-score | 0.278 |
| ROC-AUC | 0.670 |

**Confusion matrix** (test set, n = 2,922):

| | Predicted: No Churn | Predicted: Churn |
|---|---|---|
| **Actual: No Churn** | 2,451 (TN) | 187 (FP) |
| **Actual: Churn** | 208 (FN) | 76 (TP) |

## Why these numbers look the way they do

Model 2's accuracy (86.5%) is much higher than Model 1's, and its
precision (28.9%) is roughly double Model 1's — when Model 2 flags a
customer as "at risk," it is right about 3 times more often than
Model 1. This makes it a much more efficient model in terms of not
wasting retention budget on false alarms (only 187 false positives,
versus Model 1's 1,004).

However, Model 2's **recall (26.8%) is much lower** than Model 1's
(59.5%) — it only catches about 1 in 4 customers who actually churn,
missing 208 of the 284 real churners in the test set. This happened
despite also using `class_weight="balanced"` (see
[`Model2.md`](./Model2.md)), because Random Forest's tree-based
splitting still tends to favour the majority class more strongly than
Logistic Regression does once trees are limited to `max_depth=10`.

As with Model 1, a useful sanity check comes from the feature
importances printed during training (see [`Model2.md`](./Model2.md)):
several of the engineered features from
[`FeatureEngineering.md`](./FeatureEngineering.md) rank in the top 10,
confirming the model is genuinely using the new signals rather than
ignoring them.

## How to run it

1. Make sure `models/random_forest.pkl` exists (see
   [`Model2.md`](./Model2.md)).
2. Install dependencies: `pip install -r requirements.txt`
3. From the repository root, run:
   ```
   python scripts/model2_performance.py
   ```
4. Output: `experimental_results/model2_performance.json`, plus a
   printed classification report in the terminal.

## Compare with

[`Model1Performance.md`](./Model1Performance.md) and
[`Comparison.md`](./Comparison.md) for the full statistical comparison
between both models.
