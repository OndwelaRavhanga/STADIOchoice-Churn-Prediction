"""
model1_performance.py

Evaluates the trained Model 1 (Logistic Regression) on the held-out
test set, and reports appropriate metrics and statistical measures for
a binary classification problem with class imbalance.

What this script does:
1. Loads datasets/model_ready_data.csv and re-creates the exact same
   train/test split used during training (same random_state, same
   test_size, same stratify column) so the test set here is identical
   to the one the model has never seen.
2. Loads the trained Model 1 (models/logistic_regression.pkl)
3. Computes: accuracy, precision, recall, F1-score, ROC-AUC, and a
   full confusion matrix
4. Saves the results to experimental_results/model1_performance.json

Usage:
    python scripts/model1_performance.py
"""

import pandas as pd
import numpy as np
import joblib
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

DATA_PATH = os.path.join("datasets", "model_ready_data.csv")
MODEL_PATH = os.path.join("models", "logistic_regression.pkl")
OUTPUT_PATH = os.path.join("experimental_results", "model1_performance.json")

RANDOM_STATE = 42


def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["churn"])
    y = df["churn"]

    # Must exactly match the split used in model1_logistic_regression.py
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    scaler = saved["scaler"]

    X_test_scaled = scaler.transform(X_test)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    results = {
        "model": "Logistic Regression (Model 1)",
        "test_set_size": int(len(y_test)),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "confusion_matrix": {
            "true_negative": int(tn), "false_positive": int(fp),
            "false_negative": int(fn), "true_positive": int(tp),
        },
    }

    print("=== Model 1 (Logistic Regression) Performance ===")
    for k, v in results.items():
        if k != "confusion_matrix":
            print(f"{k}: {v}")
    print(f"Confusion matrix: {results['confusion_matrix']}")
    print()
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    os.makedirs("experimental_results", exist_ok=True)
    # Save predictions too, so Comparison.md's script can reuse them
    # without needing to reload/re-predict from the model file.
    results["y_true"] = y_test.tolist()
    results["y_pred"] = y_pred.tolist()
    results["y_proba"] = [round(float(p), 4) for p in y_proba]
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
