"""
model_comparison.py

Compares Model 1 (Logistic Regression) and Model 2 (Random Forest)
using both their performance metrics and a formal statistical
significance test (McNemar's test), to determine whether the
difference in their predictions is statistically meaningful or could
plausibly be due to chance.

What this script does:
1. Loads the saved results from
   experimental_results/model1_performance.json and
   experimental_results/model2_performance.json (produced by
   model1_performance.py and model2_performance.py - both scripts
   must be run first).
2. Builds a side-by-side metrics comparison table.
3. Runs McNemar's test on the two models' predictions on the same
   test set. McNemar's test is the standard statistical test for
   comparing two classifiers evaluated on the SAME test set (unlike a
   t-test, it does not assume independent samples, which is important
   here since both models were tested on identical data points).
4. Saves the comparison table to
   experimental_results/model_comparison.csv

Usage:
    python scripts/model_comparison.py
    (requires model1_performance.py and model2_performance.py to have
    been run first)
"""

import json
import os
import numpy as np
import pandas as pd
from scipy.stats import chi2

MODEL1_RESULTS_PATH = os.path.join("experimental_results", "model1_performance.json")
MODEL2_RESULTS_PATH = os.path.join("experimental_results", "model2_performance.json")
OUTPUT_CSV_PATH = os.path.join("experimental_results", "model_comparison.csv")


def mcnemar_test(y_true, y_pred_1, y_pred_2):
    """
    Manual implementation of McNemar's test (with continuity
    correction), so no extra dependency beyond scipy is required.

    Builds a 2x2 table of:
      - n01: model 1 wrong, model 2 correct
      - n10: model 1 correct, model 2 wrong
    and tests whether n01 and n10 are significantly different.
    """
    y_true = np.array(y_true)
    y_pred_1 = np.array(y_pred_1)
    y_pred_2 = np.array(y_pred_2)

    correct_1 = (y_pred_1 == y_true)
    correct_2 = (y_pred_2 == y_true)

    n10 = int(np.sum(correct_1 & ~correct_2))  # model 1 right, model 2 wrong
    n01 = int(np.sum(~correct_1 & correct_2))  # model 1 wrong, model 2 right

    # Continuity-corrected McNemar statistic
    if (n10 + n01) == 0:
        return n10, n01, 0.0, 1.0
    statistic = (abs(n10 - n01) - 1) ** 2 / (n10 + n01)
    p_value = 1 - chi2.cdf(statistic, df=1)
    return n10, n01, statistic, p_value


def main():
    with open(MODEL1_RESULTS_PATH) as f:
        m1 = json.load(f)
    with open(MODEL2_RESULTS_PATH) as f:
        m2 = json.load(f)

    # --- Metrics comparison table ---
    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    comparison_df = pd.DataFrame({
        "Metric": metrics,
        "Model 1 (Logistic Regression)": [m1[m] for m in metrics],
        "Model 2 (Random Forest)": [m2[m] for m in metrics],
    })
    comparison_df["Difference (M2 - M1)"] = (
        comparison_df["Model 2 (Random Forest)"] - comparison_df["Model 1 (Logistic Regression)"]
    ).round(4)

    print("=== Metrics Comparison ===")
    print(comparison_df.to_string(index=False))

    # --- McNemar's test ---
    n10, n01, statistic, p_value = mcnemar_test(m1["y_true"], m1["y_pred"], m2["y_pred"])

    print("\n=== McNemar's Test (statistical significance of the difference) ===")
    print(f"Cases where Model 1 was right and Model 2 was wrong (n10): {n10}")
    print(f"Cases where Model 1 was wrong and Model 2 was right (n01): {n01}")
    print(f"McNemar's chi-square statistic: {statistic:.4f}")
    print(f"p-value: {p_value:.6f}")
    alpha = 0.05
    if p_value < alpha:
        interpretation = (
            f"p-value ({p_value:.6f}) < {alpha}, so the difference between "
            "Model 1 and Model 2's predictions IS statistically significant. "
            "The two models are not making the same kinds of errors by chance."
        )
    else:
        interpretation = (
            f"p-value ({p_value:.6f}) >= {alpha}, so the difference between "
            "Model 1 and Model 2's predictions is NOT statistically significant "
            "at the 5% level."
        )
    print(interpretation)

    # --- Save ---
    os.makedirs("experimental_results", exist_ok=True)
    comparison_df.to_csv(OUTPUT_CSV_PATH, index=False)

    summary = {
        "mcnemar_n10_model1_right_model2_wrong": n10,
        "mcnemar_n01_model1_wrong_model2_right": n01,
        "mcnemar_statistic": round(statistic, 4),
        "mcnemar_p_value": round(p_value, 6),
        "significant_at_0.05": bool(p_value < alpha),
    }
    with open(os.path.join("experimental_results", "mcnemar_test_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nComparison table saved to {OUTPUT_CSV_PATH}")
    print(f"McNemar's test results saved to experimental_results/mcnemar_test_results.json")


if __name__ == "__main__":
    main()
