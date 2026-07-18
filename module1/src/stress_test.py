"""
stress_test.py
---------------
A separate, harder holdout evaluation for Module 1's trained AnomalyEnsemble.

The main train/test split in train.py draws from generate_synthetic_logs()'s
DEFAULT attack-tier mix: for each of the 3 attack archetypes (portscan, ddos,
bruteforce), that's 35% "obvious" (tier1) + 40% "moderate" (tier2) + 25%
"stealthy/low-and-slow" (tier3) attacks, and normal traffic split 70%
standard / 15% bursty / 15% "noisy" (elevated failed-connections, the
sub-population most likely to be confused with a real attack).

This script does NOT retrain anything and does NOT touch generate_data.py.
It builds an ADDITIONAL, separate, held-out sample that is deliberately
weighted toward the harder ends of those same, already-existing
distributions:
    - 100% tier3 (stealthy/low-and-slow) attacks instead of the 35/40/25 mix,
      for all three archetypes
    - 100% "noisy normal" traffic instead of the 70/15/15 mix (the sub-block
      the module1/README.md explicitly documents as "elevated failed
      connections, short duration — e.g. network drops/retries")

The tier3 / noisy-normal statistical parameters below are copied verbatim
from module1/src/generate_data.py (see _generate_portscan_block,
_generate_ddos_block, _generate_bruteforce_block, _generate_normal_block)
— nothing here is a new or invented distribution, it's the same generator
logic, just selecting only the hardest slice of it instead of the blended mix.

The already-trained models (models/isolation_forest.pkl, oneclass_svm.pkl,
scaler.pkl, score_bounds.pkl) are loaded as-is via predict.py's public
contract and scored against this harder sample — this is a pure evaluation,
not a retrain.

Run:
    cd module1/src && python train.py            # must run first, to produce models/
    cd module1/src && python stress_test.py
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import set_seed, get_logger
from generate_data import (
    _random_timestamps, _random_ip,
    _PORTSCAN_SOURCE_POOL, _DDOS_SOURCE_POOL, _BRUTEFORCE_SOURCE_POOL, _NORMAL_SOURCE_POOL,
)
from predict import predict_network_anomaly, reload_artifacts

logger = get_logger("stress_test")

N_STRESS_ROWS = 8000
ATTACK_FRACTION = 0.05


def _stealthy_portscan(n: int) -> pd.DataFrame:
    """Tier3-only portscan — parameters copied from generate_data._generate_portscan_block T3."""
    timestamps = _random_timestamps(n, start=datetime(2026, 6, 1))
    risk_countries = ["RU", "CN", "NG"]
    safe_countries = ["US", "GB", "DE"]
    country_choices = np.where(
        np.random.random(n) < 0.6,
        np.random.choice(risk_countries, n),
        np.random.choice(safe_countries, n),
    )
    failed_connections = np.random.poisson(2.0, n).clip(0, 6)
    packets_per_second = np.random.gamma(2.0, 6, n).clip(1, 60)
    packet_size = np.random.normal(500, 250, n).clip(40, 1500)
    burst_frequency = np.random.poisson(1.5, n).clip(0, 6)
    session_duration = np.random.exponential(30, n).clip(1, 200)

    return pd.DataFrame({
        "timestamp": timestamps,
        "source_ip": np.random.choice(_PORTSCAN_SOURCE_POOL, size=n),
        "destination_ip": [_random_ip() for _ in range(n)],
        "protocol": np.random.choice(["TCP", "UDP"], size=n, p=[0.8, 0.2]),
        "packet_size": packet_size,
        "packets_per_second": packets_per_second,
        "burst_frequency": burst_frequency,
        "failed_connections": failed_connections,
        "dns_requests": np.random.poisson(0.5, n).clip(0, 5),
        "http_requests": np.random.poisson(0.5, n).clip(0, 5),
        "tcp_flags": np.random.choice(["SYN", "RST"], size=n, p=[0.7, 0.3]),
        "bytes_sent": np.random.lognormal(mean=5.8, sigma=0.6, size=n).clip(40, 5000),
        "bytes_received": np.random.lognormal(mean=5.2, sigma=0.8, size=n).clip(10, 5000),
        "session_duration": session_duration,
        "country": country_choices,
        "is_attack": 1,
    })


def _stealthy_ddos(n: int) -> pd.DataFrame:
    """Tier3-only (low-rate) DDoS — parameters copied from generate_data._generate_ddos_block T3."""
    timestamps = _random_timestamps(n, start=datetime(2026, 6, 1))
    botnet_countries = ["CN", "RU", "NG", "BR"]
    country_choices = np.random.choice(botnet_countries, size=n, p=[0.35, 0.35, 0.2, 0.1])

    packets_per_second = np.random.gamma(2.5, 8, n).clip(2, 100)
    burst_frequency = np.random.poisson(4, n).clip(0, 12)
    bytes_sent = np.random.lognormal(mean=8.5, sigma=0.8, size=n).clip(500, 50000)
    bytes_received = np.random.lognormal(mean=7.5, sigma=1.0, size=n).clip(200, 30000)
    packet_size = np.random.normal(600, 300, n).clip(100, 1500)
    session_duration = np.random.exponential(15, n).clip(1, 100)

    return pd.DataFrame({
        "timestamp": timestamps,
        "source_ip": np.random.choice(_DDOS_SOURCE_POOL, size=n),
        "destination_ip": [_random_ip() for _ in range(n)],
        "protocol": np.random.choice(["UDP", "ICMP", "TCP"], size=n, p=[0.5, 0.3, 0.2]),
        "packet_size": packet_size,
        "packets_per_second": packets_per_second,
        "burst_frequency": burst_frequency,
        "failed_connections": np.random.poisson(2, n).clip(0, 10),
        "dns_requests": np.random.poisson(0.2, n).clip(0, 3),
        "http_requests": np.random.poisson(1, n).clip(0, 10),
        "tcp_flags": np.random.choice(["SYN", "ACK", "RST"], size=n, p=[0.6, 0.2, 0.2]),
        "bytes_sent": bytes_sent,
        "bytes_received": bytes_received,
        "session_duration": session_duration,
        "country": country_choices,
        "is_attack": 1,
    })


def _stealthy_bruteforce(n: int) -> pd.DataFrame:
    """Tier3-only (low-and-slow) brute-force — parameters copied from generate_data._generate_bruteforce_block T3."""
    timestamps = _random_timestamps(n, start=datetime(2026, 6, 1))
    brute_countries = ["CN", "RU", "NG", "IN", "BR"]
    country_choices = np.random.choice(brute_countries, size=n, p=[0.25, 0.25, 0.2, 0.2, 0.1])

    failed_connections = np.random.poisson(2.5, n).clip(0, 8)
    http_requests = np.random.poisson(3, n).clip(0, 10)
    packets_per_second = np.random.gamma(2.0, 5, n).clip(1, 50)

    return pd.DataFrame({
        "timestamp": timestamps,
        "source_ip": np.random.choice(_BRUTEFORCE_SOURCE_POOL, size=n),
        "destination_ip": [_random_ip() for _ in range(n)],
        "protocol": np.random.choice(["TCP", "HTTPS"], size=n, p=[0.6, 0.4]),
        "packet_size": np.random.normal(400, 180, n).clip(100, 1000),
        "packets_per_second": packets_per_second,
        "burst_frequency": np.random.poisson(8, n).clip(3, 25),
        "failed_connections": failed_connections,
        "dns_requests": np.random.poisson(1, n).clip(0, 5),
        "http_requests": http_requests,
        "tcp_flags": np.random.choice(["SYN-ACK", "RST", "PSH-ACK"], size=n, p=[0.4, 0.3, 0.3]),
        "bytes_sent": np.random.lognormal(mean=7.0, sigma=0.6, size=n).clip(200, 20000),
        "bytes_received": np.random.lognormal(mean=6.0, sigma=0.6, size=n).clip(100, 10000),
        "session_duration": np.random.exponential(scale=12, size=n).clip(0.5, 90),
        "country": country_choices,
        "is_attack": 1,
    })


def _noisy_normal(n: int) -> pd.DataFrame:
    """100% 'noisy normal' traffic — parameters copied from generate_data._generate_normal_block noisy sub-block."""
    timestamps = _random_timestamps(n, start=datetime(2026, 6, 1))
    countries = ["US", "IN", "DE", "BR", "CN", "RU", "GB", "NG", "FR", "JP"]
    protocols = ["TCP", "UDP", "ICMP", "HTTP", "HTTPS", "DNS"]

    return pd.DataFrame({
        "timestamp": timestamps,
        "source_ip": np.random.choice(_NORMAL_SOURCE_POOL, size=n),
        "destination_ip": [_random_ip() for _ in range(n)],
        "protocol": np.random.choice(protocols, size=n, p=[0.4, 0.2, 0.2, 0.1, 0.05, 0.05]),
        "packet_size": np.random.normal(150, 50, n).clip(64, 500),
        "packets_per_second": np.random.gamma(shape=2.0, scale=8, size=n).clip(0.1, 100),
        "burst_frequency": np.random.poisson(3.0, n).clip(0, 15),
        "failed_connections": np.random.poisson(6.0, n).clip(1, 20),
        "dns_requests": np.random.poisson(5, n).clip(0, 15),
        "http_requests": np.random.poisson(1, n).clip(0, 10),
        "tcp_flags": np.random.choice(["SYN", "RST", "ACK"], size=n, p=[0.4, 0.4, 0.2]),
        "bytes_sent": np.random.lognormal(mean=7.5, sigma=1.0, size=n).clip(50, 20000),
        "bytes_received": np.random.lognormal(mean=7.5, sigma=1.0, size=n).clip(50, 20000),
        "session_duration": np.random.exponential(scale=10, size=n).clip(0.2, 50),
        "country": np.random.choice(countries, size=n),
        "is_attack": 0,
    })


def build_stress_dataset(n_rows: int = N_STRESS_ROWS, attack_fraction: float = ATTACK_FRACTION) -> pd.DataFrame:
    set_seed(123)  # different seed from the main dataset — this must be a genuinely separate sample
    n_attack_total = int(n_rows * attack_fraction)
    n_normal = n_rows - n_attack_total
    n_portscan = n_attack_total // 3
    n_ddos = n_attack_total // 3
    n_bruteforce = n_attack_total - n_portscan - n_ddos

    logger.info(
        f"Building stress-test set: {n_normal} all-noisy-normal rows + "
        f"{n_attack_total} all-stealthy-tier attack rows "
        f"({n_portscan} portscan, {n_ddos} ddos, {n_bruteforce} bruteforce)"
    )

    blocks = [
        _noisy_normal(n_normal),
        _stealthy_portscan(n_portscan),
        _stealthy_ddos(n_ddos),
        _stealthy_bruteforce(n_bruteforce),
    ]
    df = pd.concat(blocks, ignore_index=True)
    df = df.sample(frac=1.0, random_state=123).reset_index(drop=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    col_order = [
        "timestamp", "source_ip", "destination_ip", "protocol", "packet_size",
        "packets_per_second", "burst_frequency", "failed_connections",
        "dns_requests", "http_requests", "tcp_flags", "bytes_sent",
        "bytes_received", "session_duration", "country", "is_attack",
    ]
    return df[col_order]


def main():
    reload_artifacts()
    df = build_stress_dataset()
    y_true = df["is_attack"].values

    results = predict_network_anomaly(df.drop(columns=["is_attack"]))
    scores = np.array([r["anomaly_score"] for r in results])

    fpr, tpr, _ = roc_curve(y_true, scores)
    stress_auc = auc(fpr, tpr)

    # Use the SAME F1-optimal threshold philosophy as train.py: sweep on this
    # stress set itself since it's a standalone evaluation set, not something
    # we're deploying a threshold from — report at both the model's normal
    # ~37 operating threshold (from eval_summary.pkl) AND at stress-set-optimal,
    # to be transparent about both "as deployed" and "best case" performance.
    import joblib
    eval_summary_path = os.path.join(
        os.path.dirname(__file__), "..", "models", "eval_summary.pkl"
    )
    deployed_threshold = joblib.load(eval_summary_path)["threshold"]

    y_pred_deployed = (scores >= deployed_threshold).astype(int)
    precision_deployed = precision_score(y_true, y_pred_deployed, zero_division=0)
    recall_deployed = recall_score(y_true, y_pred_deployed, zero_division=0)
    f1_deployed = f1_score(y_true, y_pred_deployed, zero_division=0)

    best_t, best_f1 = deployed_threshold, -1.0
    for t in np.linspace(1, 99, 99):
        f1 = f1_score(y_true, (scores >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1

    logger.info(f"STRESS-TEST ROC AUC: {stress_auc:.4f}")
    logger.info(
        f"At deployed threshold ({deployed_threshold}): "
        f"Precision={precision_deployed:.4f}, Recall={recall_deployed:.4f}, F1={f1_deployed:.4f}"
    )
    logger.info(f"Stress-set-optimal threshold={best_t}: F1={best_f1:.4f}")

    return {
        "stress_auc": stress_auc,
        "deployed_threshold": deployed_threshold,
        "precision_at_deployed": precision_deployed,
        "recall_at_deployed": recall_deployed,
        "f1_at_deployed": f1_deployed,
        "stress_optimal_threshold": best_t,
        "stress_optimal_f1": best_f1,
    }


if __name__ == "__main__":
    main()
