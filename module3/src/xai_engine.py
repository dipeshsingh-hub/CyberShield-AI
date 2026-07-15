"""
xai_engine.py
--------------
Core explainability engine backing the dashboard.

WHY A SURROGATE MODEL FOR SHAP (documented up front, not buried):
Module 1's actual anomaly detector is an ensemble of IsolationForest +
OneClassSVM. SHAP has no native support for OneClassSVM at all, and its
IsolationForest support (via TreeExplainer) is partial/version-fragile for
sklearn's implementation. Trying to force genuine SHAP values out of that
ensemble would mean either (a) shap.KernelExplainer, which is model-agnostic
but re-evaluates the model O(2^features) times per explanation — far too
slow for an interactive dashboard — or (b) pretending compatibility that
doesn't really exist.

Instead, Module 3 trains its OWN lightweight LightGBM surrogate classifier
on the same processed_features (+ a few raw fields for the "Packet Size /
Burst Frequency / DNS Requests Influence" panels) to predict the ground-truth
is_attack label. This is standard, well-documented XAI practice — explain a
fast, faithful surrogate rather than an unexplainable black box — and it's
labeled as exactly that everywhere it shows up in the dashboard, not
presented as "this is what Module 1's IsolationForest actually did
internally." Its own fidelity (agreement with Module 1's actual anomaly_score
/ risk categorization) is measured and reported, not assumed.

WHY LIME FOR THE PHISHING TEXT:
Module 2's actual selected model (LightGBM) operates on 5000-dim TF-IDF
features — SHAP TreeExplainer *could* run on it directly, but a 5000-token
SHAP summary is unreadable and mostly noise. LIME's local text explainer is
built for exactly this case: perturbing words in one specific message and
showing which ones pushed the prediction toward "phishing". Uses Module 2's
REAL trained model + vectorizer (copied via build_dataset.py, loaded here
with joblib — no source-code import, so no collision), not a redundant
retrain.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from lime.lime_text import LimeTextExplainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

warnings.filterwarnings("ignore", message="X does not have valid feature names")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import set_seed, get_logger, save_artifact, load_artifact

logger = get_logger("xai_engine")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "unified_threat_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# The 8 Module-1 processed_features plus 3 raw network fields, included
# specifically so the dashboard can render literal "Packet Size Influence",
# "Burst Frequency Influence", and "DNS Requests Influence" SHAP dependence
# plots against interpretable raw units, not just standardized z-scores.
PROCESSED_FEATURE_COLS = [
    "rolling_packet_mean", "packet_std", "packet_entropy", "traffic_ratio",
    "connection_density", "request_ratio", "burst_score", "failed_connection_rate",
]
RAW_FEATURE_COLS = ["packet_size", "burst_frequency", "dns_requests"]
SURROGATE_FEATURE_COLS = PROCESSED_FEATURE_COLS + RAW_FEATURE_COLS

_dataset = None
_surrogate_model = None
_explainer = None
_shap_values = None
_lime_vectorizer = None
_lime_model = None


# -----------------------------------------------------------------------------
# Data + surrogate model loading
# -----------------------------------------------------------------------------
def load_dataset() -> pd.DataFrame:
    global _dataset
    if _dataset is None:
        _dataset = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    return _dataset


def train_surrogate_model(df: pd.DataFrame = None, save: bool = True):
    """
    Trains the LightGBM surrogate used for all SHAP explanations. Reports
    its own fidelity against Module 1's ground truth so the dashboard never
    silently presents surrogate output as if it WERE Module 1's real model.
    """
    set_seed()
    if df is None:
        df = load_dataset()

    X = df[SURROGATE_FEATURE_COLS].values
    y = df["is_attack"].values

    idx_train, idx_test = train_test_split(
        np.arange(len(df)), test_size=0.25, stratify=y, random_state=42
    )

    model = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.08, num_leaves=15,
        min_child_samples=10, random_state=42, verbosity=-1,
    )
    model.fit(X[idx_train], y[idx_train])

    test_prob = model.predict_proba(X[idx_test])[:, 1]
    test_pred = (test_prob >= 0.5).astype(int)
    fidelity = {
        "auc_vs_is_attack": float(roc_auc_score(y[idx_test], test_prob)),
        "accuracy_vs_is_attack": float(accuracy_score(y[idx_test], test_pred)),
    }
    logger.info(f"Surrogate model fidelity (predicting Module 1's ground-truth is_attack): {fidelity}")

    if save:
        save_artifact(model, os.path.join(MODELS_DIR, "surrogate_model.pkl"))
        save_artifact(fidelity, os.path.join(MODELS_DIR, "surrogate_fidelity.pkl"))

    return model, fidelity


def _ensure_surrogate_loaded():
    global _surrogate_model, _explainer, _shap_values
    if _surrogate_model is not None:
        return

    model_path = os.path.join(MODELS_DIR, "surrogate_model.pkl")
    if os.path.exists(model_path):
        _surrogate_model = load_artifact(model_path)
    else:
        _surrogate_model, _ = train_surrogate_model()

    df = load_dataset()
    X = df[SURROGATE_FEATURE_COLS].values

    _explainer = shap.TreeExplainer(_surrogate_model)
    raw_shap = _explainer.shap_values(X)
    # LightGBM binary classifier via shap.TreeExplainer: newer shap/lightgbm
    # combinations return a single (n, features) array for the positive
    # class directly; older ones return a list [class0_shap, class1_shap].
    # Handle both without assuming which one we'll get at runtime.
    if isinstance(raw_shap, list):
        _shap_values = raw_shap[1]
    else:
        _shap_values = raw_shap
    logger.info(f"Computed SHAP values for {X.shape[0]} rows x {X.shape[1]} features.")


def _ensure_lime_loaded():
    global _lime_vectorizer, _lime_model
    if _lime_vectorizer is not None:
        return
    _lime_vectorizer = load_artifact(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    selection = load_artifact(os.path.join(MODELS_DIR, "model_selection.pkl"))
    model_name = selection["selected_model"]
    model_file = "naive_bayes_model.pkl" if model_name == "naive_bayes" else "lightgbm_model.pkl"
    _lime_model = load_artifact(os.path.join(MODELS_DIR, model_file))
    logger.info(f"Loaded Module 2's real '{model_name}' model + vectorizer for LIME.")


def _lime_predict_proba(texts):
    X = _lime_vectorizer.transform(texts)
    return _lime_model.predict_proba(X)


# -----------------------------------------------------------------------------
# Public explainability functions used by the dashboard
# -----------------------------------------------------------------------------
def get_global_feature_importance() -> pd.DataFrame:
    """Mean |SHAP value| per feature, across the whole dataset — for 'Top SHAP Features' / 'SHAP Summary'."""
    _ensure_surrogate_loaded()
    mean_abs_shap = np.abs(_shap_values).mean(axis=0)
    return pd.DataFrame({
        "feature": SURROGATE_FEATURE_COLS,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def get_local_shap_values(row_index: int) -> pd.DataFrame:
    """Per-feature SHAP values for one specific row — for the SHAP waterfall / 'Feature Contribution' panel."""
    _ensure_surrogate_loaded()
    df = load_dataset()
    row = df[SURROGATE_FEATURE_COLS].iloc[row_index]
    shap_row = _shap_values[row_index]
    return pd.DataFrame({
        "feature": SURROGATE_FEATURE_COLS,
        "value": row.values,
        "shap_value": shap_row,
    }).sort_values("shap_value", key=abs, ascending=False).reset_index(drop=True)


def get_shap_base_value() -> float:
    _ensure_surrogate_loaded()
    ev = _explainer.expected_value
    return float(ev[1] if isinstance(ev, (list, np.ndarray)) and len(np.atleast_1d(ev)) > 1 else np.atleast_1d(ev)[0])


def get_lime_explanation(row_index: int, num_features: int = 8):
    """Local LIME explanation for one row's text content — for the 'LIME Explanation' panel."""
    _ensure_lime_loaded()
    df = load_dataset()
    text = df["text"].iloc[row_index]

    explainer = LimeTextExplainer(class_names=["legitimate", "phishing"])
    exp = explainer.explain_instance(text, _lime_predict_proba, num_features=num_features)
    return {
        "text": text,
        "prediction_proba": float(_lime_predict_proba([text])[0][1]),
        "word_weights": exp.as_list(),  # [(word, weight), ...]
    }


def get_dependence_data(feature: str) -> pd.DataFrame:
    """Raw feature value + SHAP value pairs, for the SHAP dependence-style 'Influence' plots."""
    _ensure_surrogate_loaded()
    df = load_dataset()
    idx = SURROGATE_FEATURE_COLS.index(feature)
    return pd.DataFrame({
        "value": df[feature].values,
        "shap_value": _shap_values[:, idx],
        "is_attack": df["is_attack"].values,
    })


def explain_live_event(feature_vector: dict, anomaly_score: float, phishing_probability: float,
                        final_risk_probability: float, risk_category: str) -> dict:
    """
    NEW capability (added for Module 4's live orchestration — does not rename
    or alter any existing Module 3 export). Computes a local SHAP explanation
    for an ARBITRARY new event that is NOT a row in the static
    unified_threat_data.csv dataset — e.g. a live event just scored by
    Module 1 + Module 2 in main.py's real-time pipeline.

    No retraining needed: shap.TreeExplainer scores any new instance through
    the already-trained surrogate model instantly, exactly like calling
    predict_proba on a new row.

    Args:
        feature_vector: dict with the 8 Module-1 processed_features keys
            (rolling_packet_mean, packet_std, packet_entropy, traffic_ratio,
            connection_density, request_ratio, burst_score, failed_connection_rate).
            Raw packet_size/burst_frequency/dns_requests are optional — if
            absent, filled with the training dataset's median (documented
            degraded-precision fallback, mirroring Module 1's own approach
            to missing rolling-feature context).
        anomaly_score, phishing_probability, final_risk_probability, risk_category:
            already-computed upstream values, passed through into risk_summary.

    Returns: same shape as generate_xai_report()'s local-report branch.
    """
    _ensure_surrogate_loaded()
    df = load_dataset()

    row_values = []
    for col in SURROGATE_FEATURE_COLS:
        if col in feature_vector and feature_vector[col] is not None:
            row_values.append(float(feature_vector[col]))
        else:
            fallback = float(df[col].median())
            logger.warning(f"explain_live_event: '{col}' missing from feature_vector, using dataset median {fallback:.4f}")
            row_values.append(fallback)

    X_live = np.array(row_values).reshape(1, -1)
    shap_row = _explainer.shap_values(X_live)
    if isinstance(shap_row, list):
        shap_row = shap_row[1]
    shap_row = shap_row[0]

    local = pd.DataFrame({
        "feature": SURROGATE_FEATURE_COLS,
        "value": row_values,
        "shap_value": shap_row,
    }).sort_values("shap_value", key=abs, ascending=False).reset_index(drop=True)

    return {
        "important_features": local["feature"].tolist(),
        "feature_contributions": dict(zip(local["feature"], local["shap_value"].astype(float))),
        "risk_summary": {
            "anomaly_score": float(anomaly_score),
            "phishing_probability": float(phishing_probability),
            "final_risk_probability": float(final_risk_probability),
            "risk_category": risk_category,
        },
    }


def generate_xai_report(row_index: int = None, live_event: dict = None) -> dict:
    """
    PUBLIC CONTRACT (do not rename): generate_xai_report(row_index=None)
    EXTENDED (not renamed) with an optional `live_event` parameter so
    Module 4's real-time orchestrator can get an explanation for a brand new
    event that was never part of the static dashboard dataset.

    Returns:
        {
            "important_features": [...],
            "feature_contributions": {...},
            "risk_summary": {...},
        }

    Exactly one of row_index / live_event should be given, or neither for
    the global dataset-wide report. If both are given, live_event takes
    precedence (documented, not silent) since it's the more specific request.

    If live_event is given: a LOCAL report for a NEW event not in the
        static dataset — see explain_live_event() for the expected dict shape:
        {"feature_vector": {...}, "anomaly_score": ..., "phishing_probability": ...,
         "final_risk_probability": ..., "risk_category": ...}

    If row_index is given: a LOCAL report for that specific EXISTING row —
        important_features: that row's top features ranked by |SHAP value|
        feature_contributions: {feature: shap_value} for that row
        risk_summary: that row's anomaly_score / phishing_probability /
            final_risk_probability / risk_category

    If neither is given: a GLOBAL report over the whole dataset —
        important_features: dataset-wide top features ranked by mean |SHAP value|
        feature_contributions: {feature: mean_shap_value} across the dataset
            (signed mean, not mean absolute — shows each feature's average
            directional pull toward/away from "attack", not just magnitude)
        risk_summary: aggregate risk distribution stats
    """
    _ensure_surrogate_loaded()
    df = load_dataset()

    if live_event is not None:
        return explain_live_event(
            feature_vector=live_event.get("feature_vector", {}),
            anomaly_score=live_event.get("anomaly_score", 0.0),
            phishing_probability=live_event.get("phishing_probability", 0.0),
            final_risk_probability=live_event.get("final_risk_probability", 0.0),
            risk_category=live_event.get("risk_category", "Unknown"),
        )

    if row_index is not None:
        local = get_local_shap_values(row_index)
        row = df.iloc[row_index]
        return {
            "important_features": local["feature"].tolist(),
            "feature_contributions": dict(zip(local["feature"], local["shap_value"].astype(float))),
            "risk_summary": {
                "anomaly_score": float(row["anomaly_score"]),
                "phishing_probability": float(row["phishing_probability"]),
                "final_risk_probability": float(row["final_risk_probability"]),
                "risk_category": row["risk_category"],
                "module1_risk_level": row["module1_risk_level"],
            },
        }

    global_importance = get_global_feature_importance()
    mean_signed_shap = _shap_values.mean(axis=0)

    return {
        "important_features": global_importance["feature"].tolist(),
        "feature_contributions": dict(zip(SURROGATE_FEATURE_COLS, mean_signed_shap.astype(float))),
        "risk_summary": {
            "total_events": int(len(df)),
            "risk_category_counts": df["risk_category"].value_counts().to_dict(),
            "mean_final_risk_probability": float(df["final_risk_probability"].mean()),
            "mean_anomaly_score": float(df["anomaly_score"].mean()),
            "mean_phishing_probability": float(df["phishing_probability"].mean()),
            "actual_attack_rate": float(df["is_attack"].mean()),
            "actual_phishing_rate": float(df["is_phishing"].mean()),
        },
    }


if __name__ == "__main__":
    logger.info("Training surrogate model...")
    train_surrogate_model()

    logger.info("Testing generate_xai_report() — global report:")
    global_report = generate_xai_report()
    logger.info(f"important_features: {global_report['important_features']}")
    logger.info(f"feature_contributions: {global_report['feature_contributions']}")
    logger.info(f"risk_summary: {global_report['risk_summary']}")

    logger.info("Testing generate_xai_report(row_index=0) — local report:")
    local_report = generate_xai_report(row_index=0)
    logger.info(f"local report: {local_report}")

    logger.info("Testing LIME explanation on row 0:")
    lime_result = get_lime_explanation(0)
    logger.info(f"LIME result: {lime_result}")
