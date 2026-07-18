"""
cross_validate.py
------------------
5-fold cross-validation for Module 1's AnomalyEnsemble.

Purpose: show that the reported AUC/F1 aren't an artifact of one lucky
train/test split. For each of 5 stratified folds:
    - fit a fresh AnomalyEnsemble on the 4 training folds
    - score the held-out fold
    - compute ROC AUC and F1 (F1 uses the SAME per-fold threshold-search
      logic as train.py: sweep on the training folds, evaluate on the held-out fold)

Reports mean +/- std across folds. Uses the same synthetic_network_logs.csv
that train.py generates/consumes (generates it if missing, via the same
generate_synthetic_logs() call — no new data-generation logic is introduced).

Run:
    cd module1/src && python cross_validate.py
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import set_seed, get_logger
from feature_engineering import engineer_features
from anomaly_detector import AnomalyEnsemble
from generate_data import generate_synthetic_logs

logger = get_logger("cross_validate")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "synthetic_network_logs.csv")

N_FOLDS = 5


def load_or_generate_data() -> pd.DataFrame:
    if os.path.exists(DATA_PATH):
        logger.info(f"Loading existing dataset from {DATA_PATH}")
        return pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    logger.info("No dataset found — generating synthetic logs.")
    df = generate_synthetic_logs()
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    return df


def find_optimal_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    candidates = np.linspace(1, 99, 99)
    best_t, best_f1 = 50.0, -1.0
    for t in candidates:
        pred = (scores >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t


def main():
    set_seed()
    df_raw = load_or_generate_data()
    df_features, X, _ = engineer_features(df_raw, fit=True)
    y = df_features["is_attack"].values

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    fold_aucs, fold_f1s = [], []
    logger.info(f"Running {N_FOLDS}-fold stratified cross-validation on {len(y)} rows "
                f"({y.mean():.3%} attack rate)...")

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        ensemble = AnomalyEnsemble(contamination=0.05, random_state=42)
        ensemble.fit(X_train)

        train_scores = ensemble.compute_anomaly_score(X_train)
        threshold = find_optimal_threshold(y_train, train_scores)

        test_scores = ensemble.compute_anomaly_score(X_test)
        fpr, tpr, _ = roc_curve(y_test, test_scores)
        fold_auc = auc(fpr, tpr)

        y_pred = (test_scores >= threshold).astype(int)
        fold_f1 = f1_score(y_test, y_pred, zero_division=0)

        fold_aucs.append(fold_auc)
        fold_f1s.append(fold_f1)
        logger.info(f"Fold {fold_idx}/{N_FOLDS}: AUC={fold_auc:.4f}, "
                    f"F1={fold_f1:.4f} (threshold={threshold:.1f})")

    auc_mean, auc_std = float(np.mean(fold_aucs)), float(np.std(fold_aucs))
    f1_mean, f1_std = float(np.mean(fold_f1s)), float(np.std(fold_f1s))

    logger.info(f"5-fold AUC: {auc_mean:.4f} +/- {auc_std:.4f}")
    logger.info(f"5-fold F1:  {f1_mean:.4f} +/- {f1_std:.4f}")

    return {
        "fold_aucs": fold_aucs, "fold_f1s": fold_f1s,
        "auc_mean": auc_mean, "auc_std": auc_std,
        "f1_mean": f1_mean, "f1_std": f1_std,
    }


if __name__ == "__main__":
    main()
