"""
train.py
--------
Trains and compares two phishing classifiers on TF-IDF features:
    1. Multinomial Naive Bayes  — fast, strong baseline for text classification,
       genuinely competitive for sparse bag-of-words/n-gram problems.
    2. LightGBM (gradient-boosted trees) — typically stronger on structured
       signal but can overfit sparse high-dimensional TF-IDF if not careful;
       included per spec ("LightGBM or XGBoost").

Both are evaluated identically on the same held-out test split, and the
better one (by ROC-AUC, with F1 as tiebreaker) is selected as the model
backing `phishing_probability`. This comparison is not cosmetic — the losing
model's own metrics are saved too, so the choice is auditable, not asserted.

Run:
    cd module2/src && python train.py
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, precision_score, recall_score, f1_score, accuracy_score,
)
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import set_seed, get_logger, save_artifact
from text_features import fit_vectorizer, transform_texts

logger = get_logger("train")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "synthetic_phishing_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
EVAL_DIR = os.path.join(BASE_DIR, "evaluation")


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


def train_naive_bayes(X_train, y_train) -> MultinomialNB:
    logger.info("Training MultinomialNB...")
    model = MultinomialNB(alpha=0.3)
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train, y_train, X_val, y_val) -> lgb.LGBMClassifier:
    logger.info("Training LightGBM...")
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
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )
    return model


def evaluate_model(name: str, y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    metrics = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }
    logger.info(f"[{name}] " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    return metrics


def plot_model_comparison(results: dict) -> None:
    """Bar chart comparing both models across all metrics — makes the selection auditable."""
    metric_names = ["roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1_score"]
    model_names = list(results.keys())

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(metric_names))
    width = 0.35
    for i, model_name in enumerate(model_names):
        values = [results[model_name][m] for m in metric_names]
        ax.bar(x + i * width, values, width, label=model_name)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metric_names, rotation=20)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison: Naive Bayes vs LightGBM (TF-IDF features)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(EVAL_DIR, "model_comparison.png"), dpi=150)
    plt.close(fig)


def plot_final_evaluation(y_true: np.ndarray, y_prob: np.ndarray, selected_model_name: str) -> None:
    y_pred = (y_prob >= 0.5).astype(int)

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {selected_model_name} (selected)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(EVAL_DIR, "roc_curve.png"), dpi=150)
    plt.close(fig)

    # Precision-Recall
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, label=f"PR curve (AP={pr_auc:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {selected_model_name}")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(EVAL_DIR, "precision_recall_curve.png"), dpi=150)
    plt.close(fig)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Legitimate", "Phishing"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Legitimate", "Phishing"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {selected_model_name} (threshold=0.5)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(EVAL_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)


def main():
    set_seed()
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)

    df = load_or_generate_data()

    # Stratified 3-way split: train / val (for LightGBM early stopping) / test (final, untouched)
    idx_train, idx_temp = train_test_split(
        np.arange(len(df)), test_size=0.30, stratify=df["is_phishing"], random_state=42
    )
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.5, stratify=df["is_phishing"].iloc[idx_temp], random_state=42
    )

    texts = df["text"].astype(str).values
    y = df["is_phishing"].values

    vectorizer = fit_vectorizer(texts[idx_train])
    X_train = transform_texts(vectorizer, texts[idx_train])
    X_val = transform_texts(vectorizer, texts[idx_val])
    X_test = transform_texts(vectorizer, texts[idx_test])
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

    # ---- Train both candidates ----
    nb_model = train_naive_bayes(X_train, y_train)
    lgb_model = train_lightgbm(X_train, y_train, X_val, y_val)

    # ---- Evaluate both on the SAME untouched test split ----
    nb_test_prob = nb_model.predict_proba(X_test)[:, 1]
    lgb_test_prob = lgb_model.predict_proba(X_test)[:, 1]

    results = {
        "naive_bayes": evaluate_model("naive_bayes", y_test, nb_test_prob),
        "lightgbm": evaluate_model("lightgbm", y_test, lgb_test_prob),
    }
    plot_model_comparison(results)

    # ---- Select best by ROC-AUC, F1 as tiebreaker ----
    best_name = max(results, key=lambda k: (round(results[k]["roc_auc"], 4), results[k]["f1_score"]))
    logger.info(f"Selected model: {best_name} (roc_auc={results[best_name]['roc_auc']:.4f})")

    best_model = nb_model if best_name == "naive_bayes" else lgb_model
    best_prob = nb_test_prob if best_name == "naive_bayes" else lgb_test_prob
    plot_final_evaluation(y_test, best_prob, best_name)

    # ---- Persist everything ----
    save_artifact(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    save_artifact(nb_model, os.path.join(MODELS_DIR, "naive_bayes_model.pkl"))
    save_artifact(lgb_model, os.path.join(MODELS_DIR, "lightgbm_model.pkl"))
    save_artifact(
        {"selected_model": best_name, "results": results},
        os.path.join(MODELS_DIR, "model_selection.pkl"),
    )

    logger.info(f"Artifacts saved to {MODELS_DIR}")
    logger.info(f"Evaluation plots saved to {EVAL_DIR}")
    logger.info("Training complete.")
    return best_name, results


if __name__ == "__main__":
    main()
