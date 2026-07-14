"""
feature_engineering.py
-----------------------
Transforms raw network log rows into the numerical feature space used by the
anomaly detection models.

PUBLIC CONTRACT (do not rename — later modules depend on these names):
    - engineer_features(df, scaler=None, fit=True) -> (df_out, processed_features, scaler)
      where `processed_features` is the normalized numpy feature matrix.

Engineered features (per spec):
    rolling_packet_mean   - rolling mean of packet_size per source_ip
    packet_std            - rolling std of packet_size per source_ip
    packet_entropy        - Shannon entropy of packet_size distribution in a
                             rolling window per source_ip (captures payload
                             regularity vs. erratic/scripted traffic)
    traffic_ratio          - bytes_sent / bytes_received (asymmetry signal;
                             exfiltration and DDoS skew this hard)
    connection_density    - packets_per_second / session_duration (rate of
                             connection activity per unit time)
    request_ratio          - http_requests / dns_requests (protocol-mix signal)
    burst_score            - burst_frequency * packets_per_second, scaled down
                             (captures spike-like behavior)
    failed_connection_rate - failed_connections normalized by traffic volume
                             (core brute-force / scan signal)
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy as shannon_entropy
from sklearn.preprocessing import StandardScaler

from utils import get_logger

logger = get_logger("feature_engineering")

FEATURE_COLUMNS = [
    "rolling_packet_mean",
    "packet_std",
    "packet_entropy",
    "traffic_ratio",
    "connection_density",
    "request_ratio",
    "burst_score",
    "failed_connection_rate",
]

ROLLING_WINDOW = 5
_ENTROPY_BINS = np.linspace(40, 1500, 12)  # fixed global bins so entropy is comparable across rows


def _windowed_entropy(values: np.ndarray) -> float:
    """Shannon entropy of packet sizes in a rolling window, using fixed bins."""
    if len(values) < 2:
        return 0.0
    counts, _ = np.histogram(values, bins=_ENTROPY_BINS)
    if counts.sum() == 0:
        return 0.0
    probs = counts / counts.sum()
    return float(shannon_entropy(probs + 1e-12, base=2))


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-source_ip rolling statistics.

    Internally sorts by (source_ip, timestamp) so each group's rolling window
    walks forward in time. Callers MUST restore the caller's original row
    order afterward (see engineer_features) — do not rely on this function's
    output order.
    """
    df = df.sort_values(["source_ip", "timestamp"])

    grouped = df.groupby("source_ip")["packet_size"]

    df["rolling_packet_mean"] = grouped.transform(
        lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).mean()
    )
    df["packet_std"] = grouped.transform(
        lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).std()
    ).fillna(0.0)

    df["packet_entropy"] = grouped.transform(
        lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).apply(_windowed_entropy, raw=True)
    ).fillna(0.0)

    return df


def _add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute non-rolling ratio/rate features. Vectorized, order-independent."""
    df["traffic_ratio"] = df["bytes_sent"] / (df["bytes_received"] + 1.0)
    df["connection_density"] = df["packets_per_second"] / (df["session_duration"] + 1.0)
    df["request_ratio"] = df["http_requests"] / (df["dns_requests"] + 1.0)
    df["burst_score"] = (df["burst_frequency"] * df["packets_per_second"]) / 100.0
    df["failed_connection_rate"] = df["failed_connections"] / (
        (df["packets_per_second"] * df["session_duration"]) + 1.0
    )
    return df


def engineer_features(df: pd.DataFrame, scaler: StandardScaler = None, fit: bool = True):
    """
    Full feature engineering pipeline.

    Args:
        df: raw network log DataFrame with the canonical schema (see generate_data.py).
        scaler: an already-fitted sklearn StandardScaler. Required when fit=False
                (i.e. at inference time, reuse the scaler learned during training).
        fit: if True, fits a new StandardScaler on this data (training mode).
             if False, transforms using the provided scaler (inference mode).

    Returns:
        df_out: original df + engineered feature columns (unscaled, for inspection)
        processed_features: numpy array of shape (n_rows, len(FEATURE_COLUMNS)),
                             standardized (zero mean, unit variance per training fit).
                             THIS VARIABLE NAME IS PART OF THE PUBLIC CONTRACT.
        scaler: the fitted (or reused) StandardScaler, for persistence/reuse.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Mark original row order BEFORE any internal sorting. _add_rolling_features
    # reorders rows by (source_ip, timestamp) to compute correct rolling
    # windows; we restore this exact input order before returning, so
    # processed_features[i] always corresponds to the caller's df.iloc[i] —
    # regardless of the input's index or whether it arrived pre-sorted.
    df["_orig_pos"] = np.arange(len(df))

    df = _add_rolling_features(df)
    df = _add_ratio_features(df)

    # Restore caller's original row order
    df = df.sort_values("_orig_pos").drop(columns=["_orig_pos"]).reset_index(drop=True)

    # Guard against inf/NaN from divide-by-near-zero edge cases before scaling
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(df[FEATURE_COLUMNS].median())

    raw_matrix = df[FEATURE_COLUMNS].values

    if fit or scaler is None:
        scaler = StandardScaler()
        processed_features = scaler.fit_transform(raw_matrix)
        logger.info(f"Fitted new StandardScaler on {raw_matrix.shape[0]} rows.")
    else:
        processed_features = scaler.transform(raw_matrix)

    logger.info(f"processed_features shape: {processed_features.shape}")
    return df, processed_features, scaler
