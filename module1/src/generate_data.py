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
    # Split n into standard, bursty, and noisy normal traffic to inject realistic variation.
    n_standard = int(0.70 * n)
    n_bursty = int(0.15 * n)
    n_noisy = n - n_standard - n_bursty

    # standard normal sub-block
    ts_standard = _random_timestamps(n_standard, start=datetime(2026, 6, 1))
    df_standard = pd.DataFrame({
        "timestamp": ts_standard,
        "source_ip": np.random.choice(_NORMAL_SOURCE_POOL, size=n_standard),
        "destination_ip": [_random_ip() for _ in range(n_standard)],
        "protocol": np.random.choice(PROTOCOLS, size=n_standard, p=[0.25, 0.15, 0.05, 0.25, 0.25, 0.05]),
        "packet_size": np.random.normal(600, 150, n_standard).clip(64, 1500),
        "packets_per_second": np.random.gamma(shape=2.5, scale=8, size=n_standard).clip(0.5, 120),
        "burst_frequency": np.random.poisson(3.0, n_standard).clip(0, 12),
        "failed_connections": np.random.poisson(0.6, n_standard).clip(0, 5),
        "dns_requests": np.random.poisson(3, n_standard).clip(0, 30),
        "http_requests": np.random.poisson(5, n_standard).clip(0, 50),
        "tcp_flags": np.random.choice(TCP_FLAGS, size=n_standard, p=[0.2, 0.35, 0.2, 0.1, 0.05, 0.1]),
        "bytes_sent": np.random.lognormal(mean=8.2, sigma=0.8, size=n_standard).clip(50, 100000),
        "bytes_received": np.random.lognormal(mean=8.5, sigma=0.8, size=n_standard).clip(50, 100000),
        "session_duration": np.random.exponential(scale=15, size=n_standard).clip(0.5, 150),
        "country": np.random.choice(COUNTRIES, size=n_standard),
        "is_attack": 0,
    })

    # bursty normal sub-block (high pps/burst, low failed_connections; e.g. streaming or backups)
    ts_bursty = _random_timestamps(n_bursty, start=datetime(2026, 6, 1))
    df_bursty = pd.DataFrame({
        "timestamp": ts_bursty,
        "source_ip": np.random.choice(_NORMAL_SOURCE_POOL, size=n_bursty),
        "destination_ip": [_random_ip() for _ in range(n_bursty)],
        "protocol": np.random.choice(["TCP", "HTTPS"], size=n_bursty, p=[0.3, 0.7]),
        "packet_size": np.random.normal(1200, 100, n_bursty).clip(500, 1500),
        "packets_per_second": np.random.gamma(shape=6.0, scale=25, size=n_bursty).clip(40, 500),
        "burst_frequency": np.random.poisson(12.0, n_bursty).clip(4, 40),
        "failed_connections": np.random.poisson(0.4, n_bursty).clip(0, 4),
        "dns_requests": np.random.poisson(1, n_bursty).clip(0, 10),
        "http_requests": np.random.poisson(20, n_bursty).clip(0, 100),
        "tcp_flags": np.random.choice(TCP_FLAGS, size=n_bursty, p=[0.1, 0.5, 0.1, 0.1, 0.1, 0.1]),
        "bytes_sent": np.random.lognormal(mean=10.8, sigma=1.0, size=n_bursty).clip(1000, 800000),
        # Shift bytes_received lower for a sub-population of bursty normal (high traffic ratio, like file uploads)
        "bytes_received": np.random.lognormal(mean=8.2, sigma=1.2, size=n_bursty).clip(100, 800000),
        "session_duration": np.random.exponential(scale=40, size=n_bursty).clip(1, 300),
        "country": np.random.choice(COUNTRIES, size=n_bursty),
        "is_attack": 0,
    })

    # noisy normal sub-block (elevated failed connections, short duration; e.g. network drops/retries)
    ts_noisy = _random_timestamps(n_noisy, start=datetime(2026, 6, 1))
    df_noisy = pd.DataFrame({
        "timestamp": ts_noisy,
        "source_ip": np.random.choice(_NORMAL_SOURCE_POOL, size=n_noisy),
        "destination_ip": [_random_ip() for _ in range(n_noisy)],
        "protocol": np.random.choice(PROTOCOLS, size=n_noisy, p=[0.4, 0.2, 0.2, 0.1, 0.05, 0.05]),
        "packet_size": np.random.normal(150, 50, n_noisy).clip(64, 500),
        "packets_per_second": np.random.gamma(shape=2.0, scale=8, size=n_noisy).clip(0.1, 100),
        "burst_frequency": np.random.poisson(3.0, n_noisy).clip(0, 15),
        "failed_connections": np.random.poisson(6.0, n_noisy).clip(1, 20),
        "dns_requests": np.random.poisson(5, n_noisy).clip(0, 15),
        "http_requests": np.random.poisson(1, n_noisy).clip(0, 10),
        "tcp_flags": np.random.choice(["SYN", "RST", "ACK"], size=n_noisy, p=[0.4, 0.4, 0.2]),
        "bytes_sent": np.random.lognormal(mean=7.5, sigma=1.0, size=n_noisy).clip(50, 20000),
        "bytes_received": np.random.lognormal(mean=7.5, sigma=1.0, size=n_noisy).clip(50, 20000),
        "session_duration": np.random.exponential(scale=10, size=n_noisy).clip(0.2, 50),
        "country": np.random.choice(COUNTRIES, size=n_noisy),
        "is_attack": 0,
    })

    return pd.concat([df_standard, df_bursty, df_noisy], ignore_index=True)


def _generate_portscan_block(n: int) -> pd.DataFrame:
    """Port-scan-like: many small packets, high failed connections, single-ish source."""
    timestamps = _random_timestamps(n, start=datetime(2026, 6, 1))
    
    # Bias source_ip pool
    risk_countries = ["RU", "CN", "NG"]
    safe_countries = ["US", "GB", "DE"]
    country_choices = np.where(
        np.random.random(n) < 0.6,
        np.random.choice(risk_countries, n),
        np.random.choice(safe_countries, n)
    )

    # 3-tier sophistication splitting
    n_tier1 = int(0.35 * n)
    n_tier2 = int(0.40 * n)
    n_tier3 = n - n_tier1 - n_tier2

    # T1: Obvious
    fc_t1 = np.random.poisson(10, n_tier1).clip(1, 30)
    pps_t1 = np.random.gamma(5.0, 15, n_tier1).clip(20, 300)
    ps_t1 = np.random.normal(100, 60, n_tier1).clip(40, 600)
    bf_t1 = np.random.poisson(8, n_tier1).clip(1, 20)
    sd_t1 = np.random.exponential(6, n_tier1).clip(0.05, 30)

    # T2: Moderate
    fc_t2 = np.random.poisson(5, n_tier2).clip(0, 12)
    pps_t2 = np.random.gamma(3.0, 10, n_tier2).clip(10, 120)
    ps_t2 = np.random.normal(250, 120, n_tier2).clip(40, 1000)
    bf_t2 = np.random.poisson(4, n_tier2).clip(0, 12)
    sd_t2 = np.random.exponential(12, n_tier2).clip(0.1, 50)

    # T3: Low-and-slow (stealthy)
    fc_t3 = np.random.poisson(2.0, n_tier3).clip(0, 6)
    pps_t3 = np.random.gamma(2.0, 6, n_tier3).clip(1, 60)
    ps_t3 = np.random.normal(500, 250, n_tier3).clip(40, 1500)
    bf_t3 = np.random.poisson(1.5, n_tier3).clip(0, 6)
    sd_t3 = np.random.exponential(30, n_tier3).clip(1, 200)

    failed_connections = np.concatenate([fc_t1, fc_t2, fc_t3])
    packets_per_second = np.concatenate([pps_t1, pps_t2, pps_t3])
    packet_size = np.concatenate([ps_t1, ps_t2, ps_t3])
    burst_frequency = np.concatenate([bf_t1, bf_t2, bf_t3])
    session_duration = np.concatenate([sd_t1, sd_t2, sd_t3])

    df = pd.DataFrame(
        {
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
        }
    )
    return df


def _generate_ddos_block(n: int) -> pd.DataFrame:
    """DDoS-like: extreme packet rate and volume, short bursts."""
    timestamps = _random_timestamps(n, start=datetime(2026, 6, 1))
    
    # Botnet origin bias
    botnet_countries = ["CN", "RU", "NG", "BR"]
    country_choices = np.random.choice(botnet_countries, size=n, p=[0.35, 0.35, 0.2, 0.1])

    # 3-tier sophistication splitting
    n_tier1 = int(0.35 * n)
    n_tier2 = int(0.40 * n)
    n_tier3 = n - n_tier1 - n_tier2

    # T1: Loud DDoS
    pps_t1 = np.random.gamma(6.0, 30, n_tier1).clip(50, 800)
    bf_t1 = np.random.poisson(20, n_tier1).clip(3, 50)
    bytes_sent_t1 = np.random.lognormal(mean=10.0, sigma=0.8, size=n_tier1).clip(5000, 500000)
    bytes_received_t1 = np.random.lognormal(mean=5.5, sigma=1.0, size=n_tier1).clip(10, 10000)
    ps_t1 = np.random.normal(1000, 300, n_tier1).clip(200, 1500)

    # T2: Moderate DDoS
    pps_t2 = np.random.gamma(4.0, 15, n_tier2).clip(20, 300)
    bf_t2 = np.random.poisson(10, n_tier2).clip(1, 25)
    bytes_sent_t2 = np.random.lognormal(mean=9.2, sigma=0.8, size=n_tier2).clip(2000, 150000)
    bytes_received_t2 = np.random.lognormal(mean=6.5, sigma=1.0, size=n_tier2).clip(100, 20000)
    ps_t2 = np.random.normal(800, 300, n_tier2).clip(150, 1500)

    # T3: Low-rate DDoS (bursty traffic lookalike)
    pps_t3 = np.random.gamma(2.5, 8, n_tier3).clip(2, 100)
    bf_t3 = np.random.poisson(4, n_tier3).clip(0, 12)
    bytes_sent_t3 = np.random.lognormal(mean=8.5, sigma=0.8, size=n_tier3).clip(500, 50000)
    bytes_received_t3 = np.random.lognormal(mean=7.5, sigma=1.0, size=n_tier3).clip(200, 30000)
    ps_t3 = np.random.normal(600, 300, n_tier3).clip(100, 1500)

    packets_per_second = np.concatenate([pps_t1, pps_t2, pps_t3])
    burst_frequency = np.concatenate([bf_t1, bf_t2, bf_t3])
    bytes_sent = np.concatenate([bytes_sent_t1, bytes_sent_t2, bytes_sent_t3])
    bytes_received = np.concatenate([bytes_received_t1, bytes_received_t2, bytes_received_t3])
    packet_size = np.concatenate([ps_t1, ps_t2, ps_t3])

    sd_t1 = np.random.exponential(4, n_tier1).clip(0.1, 30)
    sd_t2 = np.random.exponential(8, n_tier2).clip(0.5, 60)
    sd_t3 = np.random.exponential(15, n_tier3).clip(1, 100)
    session_duration = np.concatenate([sd_t1, sd_t2, sd_t3])

    df = pd.DataFrame(
        {
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
        }
    )
    return df


def _generate_bruteforce_block(n: int) -> pd.DataFrame:
    """Brute-force-like: repeated auth attempts, high failure rate, low packet diversity."""
    timestamps = _random_timestamps(n, start=datetime(2026, 6, 1))
    
    # Brute-force attacks often originate from compromised infrastructure
    brute_countries = ["CN", "RU", "NG", "IN", "BR"]
    country_choices = np.random.choice(brute_countries, size=n, p=[0.25, 0.25, 0.2, 0.2, 0.1])

    # 3-tier sophistication splitting
    n_tier1 = int(0.35 * n)
    n_tier2 = int(0.40 * n)
    n_tier3 = n - n_tier1 - n_tier2

    # T1: Loud
    fc_t1 = np.random.poisson(12, n_tier1).clip(1, 40)
    http_t1 = np.random.poisson(15, n_tier1).clip(2, 50)
    pps_t1 = np.random.gamma(3.0, 12, n_tier1).clip(10, 150)

    # T2: Moderate
    fc_t2 = np.random.poisson(6, n_tier2).clip(0, 20)
    http_t2 = np.random.poisson(8, n_tier2).clip(1, 25)
    pps_t2 = np.random.gamma(2.5, 8, n_tier2).clip(5, 100)

    # T3: Low-and-slow (stealthy spraying / credential checks)
    fc_t3 = np.random.poisson(2.5, n_tier3).clip(0, 8)
    http_t3 = np.random.poisson(3, n_tier3).clip(0, 10)
    pps_t3 = np.random.gamma(2.0, 5, n_tier3).clip(1, 50)

    failed_connections = np.concatenate([fc_t1, fc_t2, fc_t3])
    http_requests = np.concatenate([http_t1, http_t2, http_t3])
    packets_per_second = np.concatenate([pps_t1, pps_t2, pps_t3])

    return pd.DataFrame(
        {
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
