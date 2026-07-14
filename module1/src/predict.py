"""
predict.py
----------
Inference entrypoint for Module 1. This is the file later modules should
import from.

PUBLIC CONTRACT (do not rename):
    predict_network_anomaly(df) -> dict (single row) or list[dict] (multi-row)
        {
            "anomaly_score": float,      # 0-100
            "risk_level": str,           # "Low" | "Medium" | "High"
            "feature_vector": dict,      # engineered feature name -> scaled value
        }

Design choice: models/scaler/bounds are lazily loaded once at module import
time and cached in module-level globals, so repeated calls to
predict_network_anomaly() during a live-scoring loop don't pay disk I/O cost
per call. Call reload_artifacts() if models/ have been retrained mid-session.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import get_logger, load_artifact, classify_risk_level
from feature_engineering import engineer_features, FEATURE_COLUMNS
from anomaly_detector import AnomalyEnsemble

logger = get_logger("predict")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")

_ensemble = None
_scaler = None


def reload_artifacts() -> None:
    """Load trained_models, scaler, and score-normalization bounds from disk."""
    global _ensemble, _scaler

    iso_forest = load_artifact(os.path.join(MODELS_DIR, "isolation_forest.pkl"))
    oc_svm = load_artifact(os.path.join(MODELS_DIR, "oneclass_svm.pkl"))
    scaler = load_artifact(os.path.join(MODELS_DIR, "scaler.pkl"))
    bounds = load_artifact(os.path.join(MODELS_DIR, "score_bounds.pkl"))

    ensemble = AnomalyEnsemble()
    ensemble.trained_models = {"isolation_forest": iso_forest, "oneclass_svm": oc_svm}
    ensemble.set_bounds(bounds)

    _ensemble = ensemble
    _scaler = scaler
    logger.info("Loaded trained_models, scaler, and score bounds from disk.")


def _ensure_loaded() -> None:
    if _ensemble is None or _scaler is None:
        reload_artifacts()


def predict_network_anomaly(df: pd.DataFrame):
    """
    Score one or more raw network log rows for anomalous behavior.

    Args:
        df: DataFrame with the raw schema produced by generate_data.py
            (timestamp, source_ip, destination_ip, protocol, packet_size,
            packets_per_second, burst_frequency, failed_connections,
            dns_requests, http_requests, tcp_flags, bytes_sent,
            bytes_received, session_duration, country[, is_attack]).
            is_attack, if present, is ignored at inference time.

    Returns:
        A single dict if df has exactly one row, otherwise a list of dicts,
        each of the form:
            {
                "anomaly_score": float (0-100),
                "risk_level": "Low" | "Medium" | "High",
                "feature_vector": {feature_name: scaled_value, ...}
            }
    """
    _ensure_loaded()

    if df.empty:
        return []

    # NOTE: rolling features (rolling_packet_mean, packet_std, packet_entropy)
    # are computed per source_ip across the rows passed in. For best accuracy,
    # pass a batch of recent rows per source_ip rather than a single isolated
    # row where possible — a single-row call still works, it just falls back
    # to window size 1 for the rolling stats.
    _, processed_features, _ = engineer_features(df, scaler=_scaler, fit=False)

    anomaly_score = _ensemble.compute_anomaly_score(processed_features)
    risk_levels = [classify_risk_level(s) for s in anomaly_score]

    results = []
    for i in range(len(df)):
        feature_vector = {
            feat_name: float(processed_features[i, j])
            for j, feat_name in enumerate(FEATURE_COLUMNS)
        }
        results.append({
            "anomaly_score": float(anomaly_score[i]),
            "risk_level": risk_levels[i],
            "feature_vector": feature_vector,
        })

    return results[0] if len(results) == 1 else results


if __name__ == "__main__":
    # Quick smoke test against a handful of rows from the synthetic dataset
    data_path = os.path.join(BASE_DIR, "data", "synthetic_network_logs.csv")
    sample = pd.read_csv(data_path, parse_dates=["timestamp"]).sample(5, random_state=1)
    output = predict_network_anomaly(sample.drop(columns=["is_attack"]))
    for row, result in zip(sample.to_dict("records"), output):
        logger.info(f"true_label={row['is_attack']} -> {result['risk_level']} (score={result['anomaly_score']:.1f})")
