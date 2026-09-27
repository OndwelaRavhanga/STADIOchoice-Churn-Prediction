# Model 1 Performance: Logistic Regression

This file reports the performance of Model 1 (Logistic Regression) on
the held-out test set, using the PowerCo SME Customer Churn dataset
described in Part A.

## Which file does this?

**Script:** [`scripts/model1_performance.py`](./scripts/model1_performance.py)

## What this script does

1. Loads `datasets/model_ready_data.csv` and re-creates the **exact
   same** train/test split used when training Model 1 (same random
   seed, same 80/20 split, same stratification on `churn`), so the
   test set here is data the model has genuinely never seen.
2. Loads the trained model from `models/logistic_regression.pkl`.
3. Computes accuracy, precision, recall, F1-score, ROC-AUC, and a full
   confusion matrix.
4. Saves all results (plus the raw predictions, for reuse by
   [`Comparison.md`](./Comparison.md)) to
   `experimental_results/model1_performance.json`.

## Results

| Metric | Value |
|---|---|
| Accuracy | 0.617 |
| Precision | 0.144 |
| Recall | 0.595 |
| F1-score | 0.232 |
| ROC-AUC | 0.636 |

**Confusion matrix** (test set, n = 2,922):

| | Predicted: No Churn | Predicted: Churn |
|---|---|---|
| **Actual: No Churn** | 1,634 (TN) | 1,004 (FP) |
| **Actual: Churn** | 115 (FN) | 169 (TP) |

## Why these numbers look the way they do

At first glance, an accuracy of 61.7% looks weak. But **accuracy alone
is misleading on this dataset**, because only 9.72% of customers
actually churn — a model that always predicted "no churn" would score
~90% accuracy while being completely useless for the business
question STADIOchoice actually cares about.

The metric that matters most here is **recall (59.5%)**: of all
customers who genuinely churned, Model 1 correctly identified 59.5% of
them (169 out of 284). This happened because the model was trained
with `class_weight="balanced"` (see [`Model1.md`](./Model1.md)),
which deliberately trades away some overall accuracy in exchange for
catching far more actual churners — exactly the trade-off a retention
team would want, since missing a churner (a false negative) is more
costly to the business than wrongly flagging a loyal customer (a false
positive, which just means an unnecessary retention offer).

The low precision (14.4%) means most customers flagged as "at risk"
are false alarms. This is a real limitation, discussed further in
[`Comparison.md`](./Comparison.md).

## How to run it

1. Make sure `models/logistic_regression.pkl` exists (see
   [`Model1.md`](./Model1.md)).
2. Install dependencies: `pip install -r requirements.txt`
3. From the repository root, run:
   ```
   python scripts/model1_performance.py
   ```
4. Output: `experimental_results/model1_performance.json`, plus a
   printed classification report in the terminal.

## Compare with

[`Model2Performance.md`](./Model2Performance.md) and
[`Comparison.md`](./Comparison.md) for the full statistical comparison
between both models.
