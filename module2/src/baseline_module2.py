"""
baseline_module2.py
--------------------
Triviality diagnosis for Module 2's phishing detection data.

Trains the full LightGBM pipeline alongside simple baselines:
1. Single best TF-IDF feature logistic regression
2. Keyword-count logistic regression (counts of known phishing trigger words)

If a simple baseline achieves AUC within ~0.05 of the full model despite
the sophistication tiers, the data generation isn't creating real challenge.

Usage:
    cd module2/src && python baseline_module2.py
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import set_seed, get_logger
from text_features import fit_vectorizer, transform_texts

logger = get_logger("baseline_module2")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "synthetic_phishing_dataset.csv")

# Known phishing trigger words — a trivial "bag of red flags" baseline
PHISHING_KEYWORDS = [
    "verify", "suspend", "urgent", "click", "confirm", "password",
    "account", "locked", "immediately", "secure", "credentials",
    "billing", "failed", "alert", "unusual", "action required",
    "gift card", "won", "claim", "terminated", "deletion",
    "tor_exit_node=true", "impossible_travel=true", "country_mismatch=true",
    "mfa_bypassed=true", "sql_injection", "drop table", "union select",
    "script>", "passwd", "sqlmap", "curl attacker",
]


def compute_roc_auc(y_true, scores):
    fpr, tpr, _ = roc_curve(y_true, scores)
    return auc(fpr, tpr)


def keyword_count_features(texts):
    """Count occurrences of known phishing keywords per text — the simplest possible baseline."""
    counts = np.zeros((len(texts), len(PHISHING_KEYWORDS)))
    for i, text in enumerate(texts):
        text_lower = text.lower()
        for j, kw in enumerate(PHISHING_KEYWORDS):
            counts[i, j] = text_lower.count(kw.lower())
    return counts


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

    # 1. Load data
    df = load_or_generate_data()
    texts = df["text"].astype(str).values
    y = df["is_phishing"].values

    # 2. Same split as train.py
    idx_train, idx_temp = train_test_split(
        np.arange(len(df)), test_size=0.30, stratify=y, random_state=42
    )
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.5, stratify=y[idx_temp], random_state=42
    )

    # 3. TF-IDF features
    vectorizer = fit_vectorizer(texts[idx_train])
    X_train = transform_texts(vectorizer, texts[idx_train])
    X_val = transform_texts(vectorizer, texts[idx_val])
    X_test = transform_texts(vectorizer, texts[idx_test])
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

    # 4. Full LightGBM model (same config as train.py)
    logger.info("Training full LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        max_depth=-1, min_child_samples=20, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=1.0, random_state=42,
        n_jobs=-1, verbosity=-1,
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )
    lgb_prob = lgb_model.predict_proba(X_test)[:, 1]
    full_model_auc = compute_roc_auc(y_test, lgb_prob)

    # 5. Keyword-count baseline
    logger.info("Training keyword-count logistic regression baseline...")
    kw_train = keyword_count_features(texts[idx_train])
    kw_test = keyword_count_features(texts[idx_test])
    lr_kw = LogisticRegression(random_state=42, max_iter=1000)
    lr_kw.fit(kw_train, y_train)
    kw_prob = lr_kw.predict_proba(kw_test)[:, 1]
    kw_auc = compute_roc_auc(y_test, kw_prob)

    # 6. Best single TF-IDF feature baseline (top 20 by variance)
    logger.info("Training single-feature TF-IDF baselines (top 20 by variance)...")
    train_dense = X_train.toarray()
    test_dense = X_test.toarray()
    feature_vars = np.var(train_dense, axis=0)
    top_20_idx = np.argsort(feature_vars)[-20:][::-1]

    feature_names = vectorizer.get_feature_names_out()
    best_single_auc = 0.0
    best_single_name = ""
    single_results = {}

    for fi in top_20_idx:
        X_s_train = train_dense[:, fi].reshape(-1, 1)
        X_s_test = test_dense[:, fi].reshape(-1, 1)
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_s_train, y_train)
        lr_prob = lr.predict_proba(X_s_test)[:, 1]
        s_auc = compute_roc_auc(y_test, lr_prob)
        fname = feature_names[fi]
        single_results[fname] = s_auc
        if s_auc > best_single_auc:
            best_single_auc = s_auc
            best_single_name = fname

    # 7. Report
    print("\n" + "=" * 65)
    print("MODULE 2 TRIVIALITY DIAGNOSIS")
    print("=" * 65)
    print(f"\n  Full LightGBM ROC AUC:         {full_model_auc:.4f}")
    print(f"  Keyword-count baseline AUC:    {kw_auc:.4f}  (gap: {full_model_auc - kw_auc:+.4f})")
    print(f"  Best single TF-IDF feat AUC:   {best_single_auc:.4f}  (gap: {full_model_auc - best_single_auc:+.4f})")
    print(f"    -> feature: '{best_single_name}'")

    print(f"\n  Top-20 single TF-IDF feature AUCs:")
    print(f"  {'Feature':<30} {'AUC':>8}")
    print(f"  {'-'*30} {'-'*8}")
    for fname, fauc in sorted(single_results.items(), key=lambda x: -x[1]):
        print(f"  {fname:<30} {fauc:>8.4f}")

    gap_kw = full_model_auc - kw_auc
    gap_single = full_model_auc - best_single_auc

    print(f"\n  " + "-" * 55)
    if gap_kw < 0.05:
        print(f"  [!] KEYWORD BASELINE within {abs(gap_kw):.4f} of full model — data may be trivial.")
    else:
        print(f"  [+] Keyword baseline {gap_kw:.4f} below full model — good signal.")

    if gap_single < 0.05:
        print(f"  [!] SINGLE TF-IDF FEATURE within {abs(gap_single):.4f} of full model.")
    else:
        print(f"  [+] Single TF-IDF feature {gap_single:.4f} below full model — good signal.")

    print("=" * 65 + "\n")

    return {
        "full_model_auc": full_model_auc,
        "keyword_baseline_auc": kw_auc,
        "best_single_tfidf_auc": best_single_auc,
        "best_single_tfidf_name": best_single_name,
        "gap_keyword": gap_kw,
        "gap_single_tfidf": gap_single,
    }


if __name__ == "__main__":
    main()
