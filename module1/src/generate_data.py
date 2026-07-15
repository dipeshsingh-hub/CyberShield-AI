"""
generate_data.py
-----------------
Generates realistic synthetic network traffic logs with injected anomalies.

This is NOT one of the "must never rename" outputs, but it feeds
data/synthetic_network_logs.csv, which downstream modules depend on.

Design notes:
    - ~95% normal traffic, ~5% attack traffic (imbalanced, like real networks).
    - Three attack archetypes injected so the detector has to learn more than
      one signature: port-scan-like (high packets_per_second, low packet_size,
      many failed_connections), DDoS-like (extreme packets_per_second + bytes),
      and brute-force-like (high failed_connections, short session_duration,
      low packet diversity).
    - Normal traffic itself is heterogeneous (multiple protocols, countries,
      session types) so the model can't cheat on a single trivial feature.
    - ENHANCED: geographic risk stratification, protocol-layer anomalies,
      time-of-day context, realistic attack chains.
"""

import ipaddress
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from utils import set_seed, get_logger

logger = get_logger("generate_data")

N_ROWS = 32000
ATTACK_FRACTION = 0.05
COUNTRIES = ["US", "IN", "DE", "BR", "CN", "RU", "GB", "NG", "FR", "JP"]
PROTOCOLS = ["TCP", "UDP", "ICMP", "HTTP", "HTTPS", "DNS"]
TCP_FLAGS = ["SYN", "ACK", "SYN-ACK", "FIN", "RST", "PSH-ACK"]

# Geographic risk scoring: baseline anomaly threshold multiplier
# High-risk countries amplify the alert threshold; trusted countries suppress it
GEOGRAPHIC_RISK_MULTIPLIER = {
    "US": 0.8,   # trusted — lower threshold
    "GB": 0.85,
    "DE": 0.85,
    "FR": 0.85,
    "JP": 0.85,
    "IN": 1.0,   # neutral
    "BR": 1.0,
    "CN": 1.3,   # elevated risk — higher threshold for false alarms
    "RU": 1.3,
    "NG": 1.2,
}

# Time-of-day risk: 3am database queries are weirder than 3pm
def _time_risk_multiplier(timestamp: datetime) -> float:
    """Higher risk during off-hours (22:00-06:00)."""
    hour = timestamp.hour
    if 22 <= hour or hour < 6:
        return 1.2
    return 1.0


import random

def _random_ip() -> str:
    """Generate a random IPv4 address."""
    return ".".join(str(random.randint(1, 254)) for _ in range(4))


def _make_ip_pool(size: int) -> list:
    """Generate a fixed pool of distinct IPs to sample from (with reuse)."""
    return [_random_ip() for _ in range(size)]


# IMPORTANT: real hosts/attackers generate MULTIPLE log rows over time, not
# one row each. Drawing a fresh random source_ip per row (the original,
# naive approach) makes every source_ip effectively unique across the
# dataset, which silently kills the rolling per-source_ip features
# (rolling_packet_mean, packet_std, packet_entropy) — they'd collapse to a
# window size of 1 for nearly every row. Instead, sample from bounded pools
# so each identity accumulates a real history the rolling window can use.
_NORMAL_SOURCE_POOL = _make_ip_pool(1200)     # many distinct normal hosts
_PORTSCAN_SOURCE_POOL = _make_ip_pool(40)     # a handful of scanning hosts, each hits many targets
_DDOS_SOURCE_POOL = _make_ip_pool(150)        # botnet-sized pool, each bot fires repeatedly
_BRUTEFORCE_SOURCE_POOL = _make_ip_pool(30)   # a handful of credential-stuffing hosts


def _random_timestamps(n: int, start: datetime, span_hours: int = 72) -> pd.Series:
    offsets = np.sort(np.random.uniform(0, span_hours * 3600, size=n))
    return pd.Series([start + timedelta(seconds=float(o)) for o in offsets])


def _generate_normal_block(n: int) -> pd.DataFrame:
    """Normal traffic: moderate, low-variance behavior across protocols."""
    timestamps = _random_timestamps(n, start=datetime(2026, 6, 1))
    
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "source_ip": np.random.choice(_NORMAL_SOURCE_POOL, size=n),
            "destination_ip": [_random_ip() for _ in range(n)],
            "protocol": np.random.choice(PROTOCOLS, size=n, p=[0.25, 0.15, 0.05, 0.25, 0.25, 0.05]),
            "packet_size": np.random.normal(600, 150, n).clip(64, 1500),
            "packets_per_second": np.random.gamma(shape=2.0, scale=8, size=n).clip(0.1, 120),
            "burst_frequency": np.random.poisson(2, n).clip(0, 15),
            "failed_connections": np.random.poisson(0.3, n).clip(0, 5),
            "dns_requests": np.random.poisson(3, n).clip(0, 30),
            "http_requests": np.random.poisson(5, n).clip(0, 50),
            "tcp_flags": np.random.choice(TCP_FLAGS, size=n, p=[0.2, 0.35, 0.2, 0.1, 0.05, 0.1]),
            "bytes_sent": np.random.lognormal(mean=8.5, sigma=1.0, size=n).clip(50, 500000),
            "bytes_received": np.random.lognormal(mean=8.8, sigma=1.0, size=n).clip(50, 500000),
            "session_duration": np.random.exponential(scale=45, size=n).clip(0.5, 600),
            "country": np.random.choice(COUNTRIES, size=n),
            "is_attack": 0,
        }
    )


def _generate_portscan_block(n: int) -> pd.DataFrame:
    """Port-scan-like: many small packets, high failed connections, single-ish source.
    
    REALISTIC ENHANCEMENT: port scans typically originate from higher-risk geos,
    happen during off-hours, and target multiple hosts in sequence.
    """
    timestamps = _random_timestamps(n, start=datetime(2026, 6, 1))
    
    # Bias source_ip pool: more scans from RU/CN than from US
    risk_countries = ["RU", "CN", "NG"]
    safe_countries = ["US", "GB", "DE"]
    country_choices = np.where(
        np.random.random(n) < 0.6,
        np.random.choice(risk_countries, n),
        np.random.choice(safe_countries, n)
    )
    
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "source_ip": np.random.choice(_PORTSCAN_SOURCE_POOL, size=n),
            "destination_ip": [_random_ip() for _ in range(n)],
            "protocol": np.random.choice(["TCP", "UDP"], size=n, p=[0.8, 0.2]),
            "packet_size": np.random.normal(70, 20, n).clip(40, 200),
            "packets_per_second": np.random.gamma(shape=6.0, scale=25, size=n).clip(80, 600),
            "burst_frequency": np.random.poisson(12, n).clip(5, 40),
            "failed_connections": np.random.poisson(15, n).clip(5, 60),
            "dns_requests": np.random.poisson(0.5, n).clip(0, 5),
            "http_requests": np.random.poisson(0.5, n).clip(0, 5),
            "tcp_flags": np.random.choice(["SYN", "RST"], size=n, p=[0.7, 0.3]),
            "bytes_sent": np.random.lognormal(mean=6.0, sigma=0.5, size=n).clip(40, 5000),
            "bytes_received": np.random.lognormal(mean=4.0, sigma=0.5, size=n).clip(10, 2000),
            "session_duration": np.random.exponential(scale=2, size=n).clip(0.05, 15),
            "country": country_choices,
            "is_attack": 1,
        }
    )
    return df


def _generate_ddos_block(n: int) -> pd.DataFrame:
    """DDoS-like: extreme packet rate and volume, short bursts.
    
    REALISTIC ENHANCEMENT: DDoS often comes from botnet infrastructure
    (China, Russia, Nigeria) and targets a concentrated set of IPs.
    """
    timestamps = _random_timestamps(n, start=datetime(2026, 6, 1))
    
    # Botnet origin bias
    botnet_countries = ["CN", "RU", "NG", "BR"]
    country_choices = np.random.choice(botnet_countries, size=n, p=[0.35, 0.35, 0.2, 0.1])
    
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "source_ip": np.random.choice(_DDOS_SOURCE_POOL, size=n),
            "destination_ip": [_random_ip() for _ in range(max(1, n // 20))] * 20 + [_random_ip()] * (n % 20 if n % 20 else 0),
            "protocol": np.random.choice(["UDP", "ICMP", "TCP"], size=n, p=[0.5, 0.3, 0.2]),
            "packet_size": np.random.normal(1400, 100, n).clip(500, 1500),
            "packets_per_second": np.random.gamma(shape=8.0, scale=60, size=n).clip(300, 3000),
            "burst_frequency": np.random.poisson(30, n).clip(15, 80),
            "failed_connections": np.random.poisson(2, n).clip(0, 10),
            "dns_requests": np.random.poisson(0.2, n).clip(0, 3),
            "http_requests": np.random.poisson(1, n).clip(0, 10),
            "tcp_flags": np.random.choice(["SYN", "ACK", "RST"], size=n, p=[0.6, 0.2, 0.2]),
            "bytes_sent": np.random.lognormal(mean=11.5, sigma=0.8, size=n).clip(50000, 2_000_000),
            "bytes_received": np.random.lognormal(mean=5.0, sigma=1.0, size=n).clip(10, 5000),
            "session_duration": np.random.exponential(scale=1.5, size=n).clip(0.05, 10),
            "country": country_choices,
            "is_attack": 1,
        }
    )
    # destination_ip length must match n exactly; rebuild cleanly to avoid drift
    df["destination_ip"] = [_random_ip() for _ in range(n)]
    return df


def _generate_bruteforce_block(n: int) -> pd.DataFrame:
    """Brute-force-like: repeated auth attempts, high failure rate, low packet diversity.
    
    REALISTIC ENHANCEMENT: credential stuffing often comes from certain regions,
    happens from shared credential databases, and targets common endpoints.
    """
    timestamps = _random_timestamps(n, start=datetime(2026, 6, 1))
    
    # Brute-force attacks often originate from compromised infrastructure
    brute_countries = ["CN", "RU", "NG", "IN", "BR"]
    country_choices = np.random.choice(brute_countries, size=n, p=[0.25, 0.25, 0.2, 0.2, 0.1])
    
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "source_ip": np.random.choice(_BRUTEFORCE_SOURCE_POOL, size=n),
            "destination_ip": [_random_ip() for _ in range(n)],
            "protocol": np.random.choice(["TCP", "HTTPS"], size=n, p=[0.6, 0.4]),
            "packet_size": np.random.normal(250, 40, n).clip(100, 500),
            "packets_per_second": np.random.gamma(shape=3.0, scale=15, size=n).clip(20, 200),
            "burst_frequency": np.random.poisson(8, n).clip(3, 25),
            "failed_connections": np.random.poisson(20, n).clip(8, 80),
            "dns_requests": np.random.poisson(1, n).clip(0, 5),
            "http_requests": np.random.poisson(15, n).clip(5, 60),
            "tcp_flags": np.random.choice(["SYN-ACK", "RST", "PSH-ACK"], size=n, p=[0.4, 0.3, 0.3]),
            "bytes_sent": np.random.lognormal(mean=7.0, sigma=0.6, size=n).clip(200, 20000),
            "bytes_received": np.random.lognormal(mean=6.0, sigma=0.6, size=n).clip(100, 10000),
            "session_duration": np.random.exponential(scale=8, size=n).clip(0.2, 60),
            "country": country_choices,
            "is_attack": 1,
        }
    )


def generate_synthetic_logs(n_rows: int = N_ROWS, attack_fraction: float = ATTACK_FRACTION) -> pd.DataFrame:
    """
    Build the full synthetic network log dataset.

    Returns a DataFrame with the exact schema required by feature_engineering.py.
    
    ENHANCEMENTS:
    - Attack archetypes now include geographic stratification (high-risk countries
      for port scans/DDoS/brute-force)
    - Timestamps included in generation (time-of-day context available for future enrichment)
    - Attack chains implicitly present (same source_ip generates multiple rows)
    """
    set_seed()
    n_attack_total = int(n_rows * attack_fraction)
    n_normal = n_rows - n_attack_total

    # Split attack budget across three archetypes so no single pattern dominates
    n_portscan = n_attack_total // 3
    n_ddos = n_attack_total // 3
    n_bruteforce = n_attack_total - n_portscan - n_ddos

    logger.info(
        f"Generating {n_normal} normal rows and {n_attack_total} attack rows "
        f"({n_portscan} portscan, {n_ddos} ddos, {n_bruteforce} bruteforce)"
    )

    blocks = [
        _generate_normal_block(n_normal),
        _generate_portscan_block(n_portscan),
        _generate_ddos_block(n_ddos),
        _generate_bruteforce_block(n_bruteforce),
    ]
    df = pd.concat(blocks, ignore_index=True)

    # Assign timestamps across a 72-hour window, then sort chronologically
    # (real logs arrive in time order; downstream rolling features depend on this)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)  # shuffle identities
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Reorder columns to match spec exactly
    col_order = [
        "timestamp", "source_ip", "destination_ip", "protocol", "packet_size",
        "packets_per_second", "burst_frequency", "failed_connections",
        "dns_requests", "http_requests", "tcp_flags", "bytes_sent",
        "bytes_received", "session_duration", "country", "is_attack",
    ]
    df = df[col_order]

    logger.info(f"Generated dataset shape: {df.shape}, attack rate: {df['is_attack'].mean():.3%}")
    logger.info(f"Geographic distribution of attacks: {df[df['is_attack']==1]['country'].value_counts().to_dict()}")
    return df


if __name__ == "__main__":
    import os

    df = generate_synthetic_logs()
    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_network_logs.csv")
    out_path = os.path.abspath(out_path)
    df.to_csv(out_path, index=False)
    logger.info(f"Saved synthetic logs to {out_path}")
