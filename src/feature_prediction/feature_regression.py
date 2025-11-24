# src/reconstruction/feature_regression.py

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from dataclasses import dataclass
import pandas as pd


# ---------------------------------------------------------
# Model Container
# ---------------------------------------------------------
@dataclass
class RFSeparationModel:
    model: RandomForestRegressor
    target_name: str   # "heart" or "lung"


# ---------------------------------------------------------
# Train a Multi-Output Random Forest
# ---------------------------------------------------------
def train_rf_regressor(X, y, n_estimators=500, max_depth=None, random_state=42, target_name="heart"):
    """
    Train a multi-output RandomForestRegressor for feature-space source separation.
    """
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )

    rf.fit(X, y)
    return RFSeparationModel(model=rf, target_name=target_name)


# ---------------------------------------------------------
# Predict
# ---------------------------------------------------------
def predict_features(rf_model: RFSeparationModel, X):
    """Predict target (heart or lung) feature vectors for mixed features."""
    return rf_model.model.predict(X)


# ---------------------------------------------------------
# Bootstrapped R² Confidence Interval
# ---------------------------------------------------------
def bootstrap_r2_ci(y_true, y_pred, n_boot=1000, ci=0.95, random_state=42):
    """
    Compute bootstrapped confidence intervals for R².
    Works for pandas DataFrames or numpy arrays.
    """
    # Convert to numpy
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    rng = np.random.default_rng(random_state)
    n = len(y_true)
    boot_r2 = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)  # bootstrap rows
        r2 = r2_score(y_true[idx], y_pred[idx], multioutput="variance_weighted")
        boot_r2.append(r2)

    lower = np.percentile(boot_r2, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_r2, (1 + ci) / 2 * 100)

    return np.mean(boot_r2), (lower, upper)



# ---------------------------------------------------------
# Full Model Evaluation
# ---------------------------------------------------------


def evaluate_rf_model(rf_model: RFSeparationModel, X_test, y_test, n_boot=1000, save_path=None):
    """
    Evaluates the RF model and returns:
      (1) a dictionary of results
      (2) a pandas DataFrame with a single row
    """
    y_pred = predict_features(rf_model, X_test)

    # Ensure numpy arrays
    y_test_np = np.asarray(y_test)
    y_pred_np = np.asarray(y_pred)

    # Per-feature R² scores
    r2_per_dim = r2_score(y_test_np, y_pred_np, multioutput="raw_values")

    # Overall R²
    r2_overall = r2_score(y_test_np, y_pred_np, multioutput="variance_weighted")

    # Bootstrap CI
    r2_mean, (lo, hi) = bootstrap_r2_ci(y_test_np, y_pred_np, n_boot=n_boot)

    # -------------------------------------
    # Build dictionary result
    # -------------------------------------
    result_dict = {
        "target": rf_model.target_name,
        "r2_overall": r2_overall,
        "r2_mean_boot": r2_mean,
        "r2_ci_lower": lo,
        "r2_ci_upper": hi,
        "y_pred": y_pred_np,
        "r2_per_feature": r2_per_dim,
    }

    # -------------------------------------
    # Build DataFrame (tidy)
    # -------------------------------------
    # turn per-feature R² into separate columns
    feature_cols = {f"r2_feature_{i}": r2 for i, r2 in enumerate(r2_per_dim)}

    df_row = {
        "target": rf_model.target_name,
        "r2_overall": r2_overall,
        "r2_mean_boot": r2_mean,
        "r2_ci_lower": lo,
        "r2_ci_upper": hi,
    }
    df_row.update(feature_cols)

    result_df = pd.DataFrame([df_row])
    result_df.to_csv(save_path)

    return result_dict, result_df
