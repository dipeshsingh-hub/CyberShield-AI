"""
baseline_module1.py
--------------------
Triviality diagnosis for Module 1's anomaly detection data.

Trains the full AnomalyEnsemble alongside single-feature logistic
regression baselines. If a single feature achieves AUC within ~0.05 of
the full model, the dataset is trivially separable and the model isn't
learning meaningful multi-feature patterns.

Usage:
    cd module1/src && python baseline_module1.py
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import set_seed, get_logger
from feature_engineering import engineer_features, FEATURE_COLUMNS
from anomaly_detector import AnomalyEnsemble
from generate_data import generate_synthetic_logs

logger = get_logger("baseline_module1")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "synthetic_network_logs.csv")


def load_or_generate_data() -> pd.DataFrame:
    if os.path.exists(DATA_PATH):
        logger.info(f"Loading existing dataset from {DATA_PATH}")
        return pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    logger.info("No dataset found — generating synthetic logs.")
    df = generate_synthetic_logs()
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    return df


def compute_roc_auc(y_true, scores):
    """Compute ROC AUC, handling edge cases."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    return auc(fpr, tpr)


def main():
    set_seed()

    # 1. Load data and engineer features
    df_raw = load_or_generate_data()
    df_features, processed_features, scaler = engineer_features(df_raw, fit=True)
    y = df_features["is_attack"].values

    # 2. Train/test split (same as train.py)
    idx_train, idx_test = train_test_split(
        np.arange(len(df_features)), test_size=0.25, stratify=y, random_state=42
    )
    X_train, X_test = processed_features[idx_train], processed_features[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]

    # 3. Full model AUC
    logger.info("Training full AnomalyEnsemble...")
    ensemble = AnomalyEnsemble(contamination=0.05, random_state=42)
    ensemble.fit(X_train)
    anomaly_score = ensemble.compute_anomaly_score(X_test)
    full_model_auc = compute_roc_auc(y_test, anomaly_score)
    logger.info(f"Full AnomalyEnsemble ROC AUC: {full_model_auc:.4f}")

    # 4. Single-feature logistic regression baselines
    print("\n" + "=" * 65)
    print("MODULE 1 TRIVIALITY DIAGNOSIS")
    print("=" * 65)
    print(f"\n  Full AnomalyEnsemble ROC AUC:  {full_model_auc:.4f}")
    print(f"\n  Single-feature Logistic Regression baselines:")
    print(f"  {'Feature':<30} {'AUC':>8}  {'Gap vs Full':>12}")
    print(f"  {'-'*30} {'-'*8}  {'-'*12}")

    best_single_auc = 0.0
    best_single_feature = ""
    results = {}

    for i, feat_name in enumerate(FEATURE_COLUMNS):
        X_single_train = X_train[:, i].reshape(-1, 1)
        X_single_test = X_test[:, i].reshape(-1, 1)

        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_single_train, y_train)
        lr_prob = lr.predict_proba(X_single_test)[:, 1]
        single_auc = compute_roc_auc(y_test, lr_prob)
        gap = full_model_auc - single_auc

        results[feat_name] = single_auc
        if single_auc > best_single_auc:
            best_single_auc = single_auc
            best_single_feature = feat_name

        print(f"  {feat_name:<30} {single_auc:>8.4f}  {gap:>+12.4f}")

    print(f"\n  " + "-" * 55)
    print(f"  Best single-feature:  {best_single_feature} (AUC={best_single_auc:.4f})")
    gap = full_model_auc - best_single_auc
    print(f"  Gap (full - best single):  {gap:+.4f}")

    if abs(gap) < 0.05:
        print(f"\n  [!] TRIVIAL: Single feature gets within {abs(gap):.4f} of full model.")
        print(f"      The full model is NOT learning meaningful multi-feature patterns.")
    else:
        print(f"\n  [+] NON-TRIVIAL: Full model has a {gap:.4f} AUC advantage.")
        print(f"      The model is learning something beyond single-feature separation.")

    print("=" * 65 + "\n")

    return {
        "full_model_auc": full_model_auc,
        "best_single_feature": best_single_feature,
        "best_single_auc": best_single_auc,
        "gap": gap,
        "all_features": results,
    }


if __name__ == "__main__":
    main()
