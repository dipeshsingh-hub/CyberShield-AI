"""
phishing_detector.py
----------------------
Loads the model selected by train.py (whichever of Naive Bayes / LightGBM
scored higher) plus the TF-IDF vectorizer, and exposes predict_phishing().

PUBLIC CONTRACT (do not rename):
    predict_phishing(text_or_texts) -> dict or list[dict], each containing
        "phishing_probability": float in [0, 1]
"""

import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore", message="X does not have valid feature names")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import get_logger, load_artifact

logger = get_logger("phishing_detector")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")

_vectorizer = None
_model = None
_model_name = None


def reload_artifacts() -> None:
    global _vectorizer, _model, _model_name

    _vectorizer = load_artifact(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    selection = load_artifact(os.path.join(MODELS_DIR, "model_selection.pkl"))
    _model_name = selection["selected_model"]

    model_file = "naive_bayes_model.pkl" if _model_name == "naive_bayes" else "lightgbm_model.pkl"
    _model = load_artifact(os.path.join(MODELS_DIR, model_file))

    logger.info(f"Loaded selected model: {_model_name} (test roc_auc={selection['results'][_model_name]['roc_auc']:.4f})")


def _ensure_loaded() -> None:
    if _model is None or _vectorizer is None:
        reload_artifacts()


def predict_phishing(text_or_texts):
    """
    Score one or more raw text/serialized-payload strings for phishing likelihood.

    Args:
        text_or_texts: a single string, or a list/array/Series of strings.
            Works uniformly across all 5 channels (email, sms, url,
            login_attempt, api_payload) since they all funnel through the
            same TF-IDF vectorizer fit during training.

    Returns:
        A single dict if given one string, otherwise a list of dicts:
            {"phishing_probability": float}   # public contract name
    """
    _ensure_loaded()

    single_input = isinstance(text_or_texts, str)
    texts = [text_or_texts] if single_input else list(text_or_texts)

    if len(texts) == 0:
        return []

    X = _vectorizer.transform(texts)
    probs = _model.predict_proba(X)[:, 1]

    results = [{"phishing_probability": float(p)} for p in probs]
    return results[0] if single_input else results


if __name__ == "__main__":
    samples = [
        "URGENT: Your Chase Bank account has your account will be suspended. Verify at chase-secure-verify471.com",
        "Hi Priya, following up regarding quarterly report attached. Let me know if you have any questions, thanks.",
        "endpoint=/api/v1/login method=POST param=\"' OR '1'='1\" requests_per_min=340 auth_header_present=False user_agent=sqlmap/1.6 anomalous_payload_size=True",
    ]
    for s, r in zip(samples, predict_phishing(samples)):
        logger.info(f"phishing_probability={r['phishing_probability']:.4f} <- {s[:70]}...")
