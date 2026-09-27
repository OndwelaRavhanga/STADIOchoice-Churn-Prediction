"""
model1_logistic_regression.py

Model 1: Logistic Regression classifier for predicting customer churn.

Logistic Regression is used as an interpretable baseline model: it is
fast to train, easy to explain to non-technical stakeholders (each
feature gets a clear positive/negative weight), and gives a sensible
reference point to compare the more complex Model 2 (Random Forest)
against in Part C.

What this script does:
1. Loads datasets/model_ready_data.csv
2. Splits into train/test sets (80/20, stratified on churn)
3. Scales numeric features (required for Logistic Regression, since it
   is sensitive to feature scale)
4. Trains a Logistic Regression model with class_weight="balanced" to
   account for the dataset's ~9:1 class imbalance (9.72% churn)
5. Saves the trained model to models/logistic_regression.pkl
6. Prints basic performance metrics as a sanity check (full results
   reporting is covered separately in Part C)

Usage:
    python scripts/model1_logistic_regression.py
"""

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

DATA_PATH = os.path.join("datasets", "model_ready_data.csv")
MODEL_OUTPUT_PATH = os.path.join("models", "logistic_regression.pkl")

RANDOM_STATE = 42

# Hyperparameters
C_VALUE = 1.0          # inverse of regularisation strength (default; balances fit vs overfitting)
MAX_ITER = 1000        # increased from sklearn's default of 100 to ensure convergence
CLASS_WEIGHT = "balanced"  # up-weights the minority (churn) class automatically


def main():
    print("Loading model-ready data...")
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["churn"])
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Logistic Regression needs scaled features to converge reliably
    # and to make coefficient magnitudes comparable across features.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training Logistic Regression...")
    model = LogisticRegression(
        C=C_VALUE,
        max_iter=MAX_ITER,
        class_weight=CLASS_WEIGHT,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_scaled, y_train)

    # --- Quick sanity-check metrics (full reporting happens in Part C) ---
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    print("\n--- Sanity-check metrics on test set ---")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score:  {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")

    # --- Save model and scaler together ---
    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler}, MODEL_OUTPUT_PATH)
    print(f"\nModel saved to {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
