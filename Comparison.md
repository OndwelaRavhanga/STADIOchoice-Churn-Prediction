# Comparison: Model 1 vs Model 2

This file compares Model 1 (Logistic Regression) and Model 2 (Random
Forest), using both their performance metrics and a formal statistical
significance test, to determine which model is genuinely better for
this problem — and whether the difference between them could just be
due to chance.

## Which file does this?

**Script:** [`scripts/model_comparison.py`](./scripts/model_comparison.py)

## What this script does

1. Loads the saved results from
   `experimental_results/model1_performance.json` and
   `experimental_results/model2_performance.json` (both must be
   generated first — see [`Model1Performance.md`](./Model1Performance.md)
   and [`Model2Performance.md`](./Model2Performance.md)).
2. Builds a side-by-side metrics comparison table.
3. Runs **McNemar's test** on the two models' predictions. McNemar's
   test is the appropriate statistical test here (rather than, say, a
   t-test) because both models were evaluated on the exact same test
   customers — the two sets of predictions are not independent
   samples, and McNemar's test is specifically designed to compare
   paired classifier predictions like this.
4. Saves the comparison table to
   `experimental_results/model_comparison.csv` and the statistical
   test results to `experimental_results/mcnemar_test_results.json`.

## Metrics comparison

| Metric | Model 1 (Logistic Regression) | Model 2 (Random Forest) | Difference (M2 − M1) |
|---|---|---|---|
| Accuracy | 0.617 | 0.865 | +0.248 |
| Precision | 0.144 | 0.289 | +0.145 |
| Recall | 0.595 | 0.268 | **−0.328** |
| F1-score | 0.232 | 0.278 | +0.046 |
| ROC-AUC | 0.636 | 0.670 | +0.034 |

## Statistical significance: McNemar's Test

| | Value |
|---|---|
| Cases Model 1 got right, Model 2 got wrong (n10) | 104 |
| Cases Model 1 got wrong, Model 2 got right (n01) | 828 |
| McNemar's chi-square statistic | 560.87 |
| p-value | < 0.000001 |
| Significant at α = 0.05? | **Yes** |

Since the p-value is far below 0.05, the difference between Model 1
and Model 2's predictions is **statistically significant** — this is a
real, meaningful difference in behaviour between the two models, not
something that could plausibly have happened by chance on this test
set.

## Which model is actually better?

**Neither model is simply "better" overall — they make different
trade-offs, and the right choice depends on what STADIOchoice values
more:**

- **Model 2 (Random Forest) is better if the priority is efficient
  spending.** It is far more accurate overall (86.5% vs 61.7%) and
  precise (28.9% vs 14.4%) — when it flags a customer as at risk, it
  is right much more often, meaning fewer retention offers get wasted
  on customers who were never going to leave.

- **Model 1 (Logistic Regression) is better if the priority is not
  missing churners.** It catches 59.5% of all customers who actually
  churn, more than double Model 2's 26.8% recall. For a business
  problem where the whole point is to intervene before a subscriber
  cancels (as set out in this project's Motivation and Problem
  Statement), missing a real churner (a false negative) is arguably a
  bigger loss than wasting a retention offer on a loyal customer (a
  false positive).

Given STADIOchoice's stated goal of catching subscribers **before**
they cancel — including the "quiet leavers" who never contact support
— **recall is the more business-critical metric for this specific
problem**, which favours **Model 1** despite its lower precision and
accuracy. This trade-off, and how it could be improved, is explored
further in Part D's recommendations report.

## How to run it

1. Make sure both
   `experimental_results/model1_performance.json` and
   `experimental_results/model2_performance.json` exist (run
   `model1_performance.py` and `model2_performance.py` first).
2. Install dependencies: `pip install -r requirements.txt`
3. From the repository root, run:
   ```
   python scripts/model_comparison.py
   ```
4. Output: `experimental_results/model_comparison.csv` and
   `experimental_results/mcnemar_test_results.json`.
