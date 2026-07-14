"""
utils.py
--------
Shared utility functions used across the anomaly detection pipeline.

Responsibilities:
    - Deterministic seeding for reproducibility
    - Risk-level classification from a 0-100 anomaly score
    - Joblib-based model persistence helpers
    - Lightweight logging setup

IMPORTANT: This module is imported by feature_engineering.py, anomaly_detector.py,
train.py, and predict.py. Do not rename any public function here without
updating all downstream imports.
"""

import logging
import os
import random

import joblib
import numpy as np

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Seed numpy/random for reproducible synthetic data + model training."""
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    """Return a configured logger. Avoids duplicate handlers on re-import."""
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


# ---------------------------------------------------------------------------
# Risk classification
# ---------------------------------------------------------------------------
def classify_risk_level(score: float) -> str:
    """
    Map a normalized 0-100 anomaly_score to a human-readable risk tier.

    Thresholds were derived empirically from the trained ensemble's actual
    score distribution on the synthetic dataset (not picked as round numbers):
    normal traffic clusters below ~25 (95th percentile ~34, with a long right
    tail from noisy-but-benign rows), the F1-optimal binary decision boundary
    sits around 33, and confirmed attacks median around 60+. So:
        0-25   -> Low     (consistent with normal traffic)
        25-55  -> Medium  (straddles the decision boundary — worth a human look)
        55-100 -> High    (well past the boundary, high-confidence attack)
    Re-validate these cut points against evaluation/eval_summary.pkl's
    threshold whenever the model is retrained on new data — a materially
    different attack mix will shift the score distribution.
    """
    if score < 25:
        return "Low"
    elif score < 55:
        return "Medium"
    return "High"


# ---------------------------------------------------------------------------
# Min-max scaling to a fixed 0-100 range (used for ensemble score normalization)
# ---------------------------------------------------------------------------
def minmax_to_100(raw_scores: np.ndarray, fit_min: float = None, fit_max: float = None):
    """
    Scale raw_scores to the 0-100 range.

    If fit_min/fit_max are provided, scales using those bounds (this is how
    inference-time scores are normalized using bounds learned at train time,
    preventing test-time min/max leakage or unstable single-row scaling).
    If not provided, fits bounds from raw_scores itself (used during training).

    Returns (scaled_scores, fit_min, fit_max) so callers can persist the bounds.
    """
    raw_scores = np.asarray(raw_scores, dtype=np.float64)

    if fit_min is None or fit_max is None:
        fit_min = float(np.min(raw_scores))
        fit_max = float(np.max(raw_scores))

    denom = (fit_max - fit_min) if (fit_max - fit_min) != 0 else 1e-9
    scaled = (raw_scores - fit_min) / denom * 100.0
    scaled = np.clip(scaled, 0.0, 100.0)
    return scaled, fit_min, fit_max


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------
def save_artifact(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)


def load_artifact(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact not found at {path}. Run train.py first.")
    return joblib.load(path)
