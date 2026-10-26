# SS3 Part B: Applying the final method to the STADIOchoice client data

This file explains `SS3_PartB.ipynb`, the notebook that applies the SS2 method to the real data supplied by STADIOchoice. The problem is the same as in SS1 and SS2: predict which streaming subscribers will cancel (churn) before they do, so retention offers can be targeted.

## Files

| File | What it is |
|---|---|
| `SS3_PartB.ipynb` | The notebook. It has five clearly marked sections: 1 Dataset loading, 2 Dataset preprocessing, 3 Model training, 4 Model validation, 5 Results visualisation. |
| `models/ss3_logistic_regression.joblib`, `models/ss3_random_forest.joblib` | The two trained model pipelines (preprocessing and model together). |
| `experimental_results/ss3_training_log.json` | Training record: settings, split sizes, features used, fit times, training and cross-validation scores, library versions. |
| `experimental_results/ss3_cv_summary.csv`, `ss3_cv_fold_scores.csv` | Cross-validation results on the training set (summary and every fold). |
| `experimental_results/ss3_training_feature_effects.csv` | What each model learned (Logistic Regression coefficients, Random Forest importances). |
| `experimental_results/ss3_validation_results.json` | Validation and test metrics with 95% confidence intervals, the model-selection decision and the decision thresholds. |
| `experimental_results/ss3_validation_metrics.csv`, `ss3_test_metrics.csv` | The same metrics as tables. |
| `experimental_results/ss3_leakage_illustration.csv` | How much leakage would have inflated the results (illustration only). |
| `experimental_results/ss3_fig1_...png` to `ss3_fig7_...png` | The seven figures from section 5. |

## The data

The client supplied three CSV tables, linked by `subscriber_id`: `table1_subscription_billing.csv`, `table2_streaming_viewing_behaviour.csv` and `table3_cancellation_retention.csv`. The data belongs to the client, so it is **not stored in this repository**. To re-run the notebook, put the three files in `datasets/client_data/` (or set the environment variable `STADIO_DATA_DIR` to the folder that holds them). All outputs of the last run are saved inside the notebook itself, so the results can be read without re-running it.

## How to run

1. Install the requirements: `pip install -r requirements.txt`
2. Put the three client CSV files in `datasets/client_data/`.
3. Open `SS3_PartB.ipynb` (Jupyter, VS Code or Codespaces) and choose **Run All**. It takes about three to four minutes.
4. Results are written to `experimental_results/` and `models/`.

## What the notebook does

1. **Dataset loading.** Loads the three tables, reports the number of instances (28,104 subscribers), the classes (4,591 churned and 23,513 stayed) and the imbalance (5.12 : 1), and explains why the CSV format is used and how the tables are joined.
2. **Dataset preprocessing.** Cleans the data (duplicates, text, types, missing values, logical checks), joins the tables, builds features that are known at a fixed prediction date, splits the data (60% training, 20% validation, 20% test, stratified), audits for leakage on the training data only, and defines the preprocessing pipeline that is fitted on the training set only.
3. **Model training.** Trains the two SS2 models (Logistic Regression and Random Forest) with the SS2 hyper-parameters and seed, explains each setting, runs five-fold cross-validation on the training set, and saves all training outputs.
4. **Model validation.** Evaluates on the validation set, chooses the final model with a rule fixed beforehand, then evaluates once on the test set. All metrics have 95% bootstrap confidence intervals. The results are tied back to the business problem.
5. **Results visualisation.** Seven figures for training results (cross-validation, learning curves) and for unseen data (ROC and precision-recall curves, gains and lift, confusion matrices, what drives the predictions, calibration).

## Key decisions

* **Leakage.** The client's data contains columns that reveal the outcome. `days_since_last_login` alone separates the classes almost perfectly, and `tenure_months` is counted to the cancellation date for churners. These columns, the other 30-day engagement columns and all columns of the cancellation table are excluded (evidence in section 2.5). Section 4.8 shows what including them would do: the validation ROC-AUC would jump from 0.693 to 0.999, which looks excellent and is useless in practice.
* **Features kept.** Plan, price, free-trial flag, payment status, payment failures, upgrade/downgrade history, device, and three features built from the subscription start date relative to 1 September 2025 (the start of the observation window).
* **Models and alignment with SS2.** The same two models, hyper-parameters and seed as SS2. The data, the features and the split are what differ (and a validation set is added).
* **Final model.** Chosen by a rule written down before the comparison: Random Forest replaces the SS2 choice only if its validation PR-AUC is better with a 95% paired-bootstrap interval above zero. The difference (Random Forest minus Logistic Regression) is -0.038 (interval -0.058 to -0.018), so the final model is **Logistic Regression (Model 1)**, the SS2 choice. The Random Forest also over-fits (training ROC-AUC 0.791 against 0.658 on held-out folds), and a more regularised forest does no better (section 3.3).
* **Decision threshold.** Set on the validation scores so that the top 20% highest-risk subscribers are contacted, then fixed for the test set.

## Results on the test set (used once)

Values are shown with 95% bootstrap confidence intervals. The "top 20%" rows use the decision threshold chosen on the validation set.

| Metric | Logistic Regression (Model 1) | Random Forest (Model 2) | No-skill baseline |
|---|---|---|---|
| ROC-AUC (ranking quality) | 0.671 [0.652, 0.690] | 0.649 [0.630, 0.668] | 0.5 |
| PR-AUC (ranking quality) | 0.286 [0.264, 0.315] | 0.265 [0.244, 0.291] | 0.163 |
| Precision, top 20% contacted | 0.314 [0.288, 0.341] | 0.292 [0.265, 0.320] | 0.163 |
| Recall (share of churners caught), top 20% contacted | 0.392 [0.362, 0.425] | 0.345 [0.314, 0.376] | 0.2 |
| F1-score, top 20% contacted | 0.348 [0.324, 0.375] | 0.316 [0.291, 0.342] | 0.18 |
| Lift vs random targeting, top 20% contacted | 1.920 [1.789, 2.061] | 1.786 [1.642, 1.932] | 1.0 |
| Accuracy, top 20% contacted (misleading) | 0.761 [0.749, 0.772] | 0.756 [0.744, 0.767] | 0.837 |

For the final model, contacting the top 20% of subscribers reaches **39.2%** of the subscribers who go on to cancel (interval 36.2% to 42.5%), which is **1.92 times better than contacting 20% at random**. Accuracy is shown only to demonstrate that it is misleading here (most subscribers stay).

## Limitations

* The model is **modest**: its test ROC-AUC is 0.671 (95% interval 0.652 to 0.690). That is expected for a leakage-free model that only has account and billing information.
* The client's data is one snapshot taken at the extract date. A real early-warning model needs the engagement data recorded as at a fixed date *before* the outcome. We would ask STADIOchoice for such an extract.
* The two search columns are excluded as a cautious choice. Adding them would raise the validation ROC-AUC from 0.693 to 0.729; they should be added if the client confirms they were recorded before the cancellation decision.
* The model scores are for ranking, not probabilities (`class_weight="balanced"` inflates them; see figure 7).
* The billing columns that are kept are assumed to be known before a subscriber decides to cancel. This should be confirmed with the client.

## Reproducibility

Seed 42 everywhere; stratified splits; all preprocessing fitted on the training set only. Running the notebook again reproduces the saved results exactly (checked by re-running all cells in a fresh folder). Library versions are stored in `experimental_results/ss3_training_log.json`.
