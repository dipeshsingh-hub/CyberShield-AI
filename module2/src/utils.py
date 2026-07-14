"""
utils.py
--------
Shared utilities for Module 2 (phishing detection + Bayesian risk fusion).
Mirrors module1/src/utils.py conventions so the two modules feel like one
codebase, but is kept independent (module2 does not import module1's utils,
to keep the modules genuinely standalone as required).
"""

import logging
import os
import random

import joblib
import numpy as np

RANDOM_SEED = 42


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def classify_risk_category(score: float) -> str:
    """
    Map a 0-100 final_risk_probability*100 (or any 0-100 risk score) to the
    three risk tiers specified for Module 2:
        0-40   -> Low
        40-70  -> Medium
        70-100 -> Critical
    Note: this is a DIFFERENT band structure from Module 1's Low/Medium/High
    (which used empirically-calibrated 25/55 cut points on the anomaly score
    distribution). Module 2's bands are fixed by spec, so they are NOT
    re-derived from the data the way Module 1's were — worth knowing if the
    two modules' risk labels ever get compared side by side.
    """
    if score < 40:
        return "Low"
    elif score < 70:
        return "Medium"
    return "Critical"


def save_artifact(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)


def load_artifact(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact not found at {path}. Run train.py first.")
    return joblib.load(path)


def safe_logit(p: np.ndarray, eps: float = 1e-6):
    """
    Log-odds transform with clipping to avoid -inf/+inf at p=0 or p=1.
    Central to bayesian_layer.py's evidence-fusion math.
    """
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid(x):
    """Inverse of safe_logit. Numerically stable for large |x|."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
