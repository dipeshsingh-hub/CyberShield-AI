"""
cross_validate.py
------------------
5-fold cross-validation for Module 2's selected phishing classifier (LightGBM).

For each of 5 stratified folds:
    - fit a fresh TF-IDF vectorizer on the training folds only (avoids any
      vocabulary leakage from the held-out fold)
    - fit a fresh LightGBM classifier on the training folds
    - evaluate ROC AUC and F1 (threshold=0.5) on the held-out fold

Reports mean +/- std across folds, showing the near-perfect scores aren't an
artifact of one lucky split. Uses the same synthetic_phishing_dataset.csv
that train.py generates/consumes.

Run:
    cd module2/src && python cross_validate.py
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, f1_score
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import set_seed, get_logger
from text_features import fit_vectorizer, transform_texts

logger = get_logger("cross_validate")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "synthetic_phishing_dataset.csv")

N_FOLDS = 5


def load_or_generate_data() -> pd.DataFrame:
    if os.path.exists(DATA_PATH):
        logger.info(f"Loading existing dataset from {DATA_PATH}")
        return pd.read_csv(DATA_PATH)
    logger.info("No dataset found — generating synthetic phishing data.")
    from generate_data import generate_synthetic_phishing_data
    df = generate_synthetic_phishing_data()
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    return df


def main():
    set_seed()
    df = load_or_generate_data()
    texts = df["text"].astype(str).values
    y = df["is_phishing"].values

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    fold_aucs, fold_f1s = [], []
    logger.info(f"Running {N_FOLDS}-fold stratified cross-validation on {len(y)} rows "
                f"({y.mean():.3%} phishing rate)...")

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(texts, y), start=1):
        texts_train, texts_test = texts[train_idx], texts[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        vectorizer = fit_vectorizer(texts_train)
        X_train = transform_texts(vectorizer, texts_train)
        X_test = transform_texts(vectorizer, texts_test)

        model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fold_auc = auc(fpr, tpr)
        fold_f1 = f1_score(y_test, y_pred, zero_division=0)

        fold_aucs.append(fold_auc)
        fold_f1s.append(fold_f1)
        logger.info(f"Fold {fold_idx}/{N_FOLDS}: AUC={fold_auc:.4f}, F1={fold_f1:.4f}")

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
