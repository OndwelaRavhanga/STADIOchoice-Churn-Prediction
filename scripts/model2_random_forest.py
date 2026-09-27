"""
model2_random_forest.py

Model 2: Random Forest classifier for predicting customer churn.

Random Forest is used as a stronger ensemble model to compare against
the Logistic Regression baseline (Model 1). It can capture non-linear
relationships and feature interactions that Logistic Regression cannot.
The hyperparameters below (300 trees, max depth 10, Gini criterion)
follow the configuration reported in Related Literature 1
(Al Mamun et al., 2026), which used the same PowerCo dataset and found
this configuration to perform well (94% accuracy in their study).

What this script does:
1. Loads datasets/model_ready_data.csv
2. Splits into train/test sets (80/20, stratified on churn) - same
   split logic as Model 1, so the two models are compared fairly
3. Trains a Random Forest with class_weight="balanced" to account for
   the dataset's ~9:1 class imbalance (9.72% churn)
4. Saves the trained model to models/random_forest.pkl
5. Prints basic performance metrics as a sanity check (full results
   reporting is covered separately in Part C)

Usage:
    python scripts/model2_random_forest.py
"""

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

DATA_PATH = os.path.join("datasets", "model_ready_data.csv")
MODEL_OUTPUT_PATH = os.path.join("models", "random_forest.pkl")

RANDOM_STATE = 42

# Hyperparameters (following Al Mamun et al., 2026, who used the same
# PowerCo dataset)
N_ESTIMATORS = 300
MAX_DEPTH = 10
CRITERION = "gini"
CLASS_WEIGHT = "balanced"


def main():
    print("Loading model-ready data...")
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["churn"])
    y = df["churn"]

    # Same split parameters as Model 1, so both models are trained
    # and tested on identical data for a fair comparison in Part C.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Random Forest does not require feature scaling, unlike Logistic
    # Regression, since it splits on raw feature values rather than
    # distances or weighted sums.
    print("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        criterion=CRITERION,
        class_weight=CLASS_WEIGHT,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # --- Quick sanity-check metrics (full reporting happens in Part C) ---
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n--- Sanity-check metrics on test set ---")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score:  {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")

    # --- Feature importance (top 10) - useful sanity check that the
    # engineered features from Part B are actually being used ---
    importances = pd.Series(model.feature_importances_, index=X.columns)
    print("\nTop 10 most important features:")
    print(importances.sort_values(ascending=False).head(10))

    # --- Save model ---
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"\nModel saved to {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
