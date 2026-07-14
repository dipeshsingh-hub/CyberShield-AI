"""
train.py
--------
End-to-end training entrypoint for Module 1.

Pipeline:
    1. Load (or generate) synthetic_network_logs.csv
    2. Run feature_engineering.engineer_features() -> processed_features
    3. Train AnomalyEnsemble (IsolationForest + OneClassSVM) -> trained_models
    4. Compute anomaly_score on a held-out test split
    5. Evaluate against the (synthetic, ground-truth) is_attack label:
         - Confusion matrix
         - ROC curve + AUC
         - Precision / Recall / F1
         - Permutation-based feature importance approximation
    6. Persist models, scaler, and score-normalization bounds to models/
    7. Write evaluation plots to evaluation/

Run:
    cd module1/src && python train.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_score, recall_score, f1_score,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import set_seed, get_logger, save_artifact, classify_risk_level
from feature_engineering import engineer_features, FEATURE_COLUMNS
from anomaly_detector import AnomalyEnsemble
from generate_data import generate_synthetic_logs

logger = get_logger("train")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "synthetic_network_logs.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
EVAL_DIR = os.path.join(BASE_DIR, "evaluation")

# Anomaly-score threshold used to convert continuous scores into a binary
# attack/normal call for classification metrics. This is NOT a fixed guess —
# find_optimal_threshold() below sweeps thresholds on the training split and
# picks the one that maximizes F1, then that value is what actually gets
# persisted to eval_summary.pkl and should be used at inference time for
# binary alerting. A hardcoded threshold divorced from the score distribution
# (e.g. an arbitrary "midpoint of Medium risk") silently wrecks recall —
# verified empirically: threshold=55 on this dataset gives recall=0.52,
# threshold~33 gives recall=0.95 at nearly the same precision.
DEFAULT_THRESHOLD = 50.0  # fallback only, overridden by find_optimal_threshold()


def find_optimal_threshold(y_true: np.ndarray, anomaly_score: np.ndarray) -> float:
    """Sweep candidate thresholds and return the one maximizing F1 on this split."""
    candidates = np.linspace(1, 99, 99)
    best_threshold, best_f1 = DEFAULT_THRESHOLD, -1.0
    for t in candidates:
        y_pred = (anomaly_score >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_threshold, best_f1 = float(t), f1
    logger.info(f"Optimal threshold found: {best_threshold} (F1={best_f1:.4f})")
    return best_threshold


def load_or_generate_data() -> pd.DataFrame:
    if os.path.exists(DATA_PATH):
        logger.info(f"Loading existing dataset from {DATA_PATH}")
        return pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    logger.info("No dataset found — generating synthetic logs.")
    df = generate_synthetic_logs()
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    return df


def evaluate(y_true: np.ndarray, anomaly_score: np.ndarray, threshold: float) -> dict:
    y_pred = (anomaly_score >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, anomaly_score)
    roc_auc = auc(fpr, tpr)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    logger.info(f"Confusion matrix:\n{cm}")
    logger.info(f"ROC AUC: {roc_auc:.4f}")
    logger.info(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

    # ---- Plot: confusion matrix ----
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Normal", "Attack"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal", "Attack"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix (threshold={threshold})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(EVAL_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # ---- Plot: ROC curve ----
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(EVAL_DIR, "roc_curve.png"), dpi=150)
    plt.close(fig)

    return {
        "confusion_matrix": cm.tolist(),
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "threshold": threshold,
    }


def permutation_feature_importance(ensemble: AnomalyEnsemble, X: np.ndarray, y_true: np.ndarray) -> dict:
    """
    Approximate feature importance for the ensemble via permutation:
    shuffle one feature column at a time, measure the drop in ROC-AUC
    (relative to the true is_attack labels) caused by destroying that
    feature's signal. Larger drop = more important feature.

    Neither IsolationForest nor OneClassSVM expose native feature_importances_,
    so this model-agnostic approach is the standard substitute.
    """
    baseline_score = ensemble.compute_anomaly_score(X)
    fpr, tpr, _ = roc_curve(y_true, baseline_score)
    baseline_auc = auc(fpr, tpr)

    importances = {}
    rng = np.random.RandomState(42)
    for i, feat_name in enumerate(FEATURE_COLUMNS):
        X_permuted = X.copy()
        rng.shuffle(X_permuted[:, i])
        permuted_score = ensemble.compute_anomaly_score(X_permuted)
        fpr_p, tpr_p, _ = roc_curve(y_true, permuted_score)
        permuted_auc = auc(fpr_p, tpr_p)
        importances[feat_name] = float(baseline_auc - permuted_auc)

    # ---- Plot: feature importance ----
    sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    names = [k for k, _ in sorted_items]
    values = [v for _, v in sorted_items]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(names[::-1], values[::-1], color="steelblue")
    ax.set_xlabel("AUC drop when feature is shuffled (higher = more important)")
    ax.set_title("Permutation Feature Importance")
    fig.tight_layout()
    fig.savefig(os.path.join(EVAL_DIR, "feature_importance.png"), dpi=150)
    plt.close(fig)

    logger.info(f"Feature importance (AUC drop): {importances}")
    return importances


def main():
    set_seed()
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)

    # 1. Data
    df_raw = load_or_generate_data()

    # 2. Feature engineering -> processed_features
    df_features, processed_features, scaler = engineer_features(df_raw, fit=True)
    y = df_features["is_attack"].values

    # Train/test split (stratified so both splits keep the ~5% attack rate)
    idx_train, idx_test = train_test_split(
        np.arange(len(df_features)), test_size=0.25, stratify=y, random_state=42
    )
    X_train, X_test = processed_features[idx_train], processed_features[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]

    # 3. Train ensemble -> trained_models
    ensemble = AnomalyEnsemble(contamination=0.05, random_state=42)
    ensemble.fit(X_train)
    trained_models = ensemble.trained_models  # public contract name
    logger.info(f"trained_models keys: {list(trained_models.keys())}")

    # 4. Score test split -> anomaly_score
    anomaly_score = ensemble.compute_anomaly_score(X_test)  # public contract name
    logger.info(f"anomaly_score range on test set: [{anomaly_score.min():.2f}, {anomaly_score.max():.2f}]")

    # 5. Pick the decision threshold on the TRAIN split (never on test — that
    #    would leak test-set information into a metric we then report on the
    #    same test set), then evaluate on the held-out test split.
    train_scores = ensemble.compute_anomaly_score(X_train)
    threshold = find_optimal_threshold(y_train, train_scores)
    metrics = evaluate(y_test, anomaly_score, threshold=threshold)
    importances = permutation_feature_importance(ensemble, X_test, y_test)

    # 6. Persist artifacts
    save_artifact(trained_models["isolation_forest"], os.path.join(MODELS_DIR, "isolation_forest.pkl"))
    save_artifact(trained_models["oneclass_svm"], os.path.join(MODELS_DIR, "oneclass_svm.pkl"))
    save_artifact(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    save_artifact(ensemble.get_bounds(), os.path.join(MODELS_DIR, "score_bounds.pkl"))
    save_artifact(
        {"metrics": metrics, "feature_importance": importances, "threshold": threshold},
        os.path.join(MODELS_DIR, "eval_summary.pkl"),
    )

    logger.info(f"Artifacts saved to {MODELS_DIR}")
    logger.info(f"Evaluation plots saved to {EVAL_DIR}")
    logger.info("Training complete.")

    return ensemble, metrics, importances


if __name__ == "__main__":
    main()
