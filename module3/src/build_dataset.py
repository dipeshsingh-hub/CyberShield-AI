"""
build_dataset.py
------------------
Builds module3/data/unified_threat_data.csv — the single dataset the
Streamlit dashboard reads from. This is a deliberate architectural choice:
rather than have the dashboard process import Module 1 and Module 2 source
code live (which collides — see module_bridge.py), we run a batch ETL job
that calls both modules' real, shipped functions via isolated subprocesses,
then persist the combined result. This is also how a real production
dashboard would work — reading from a data store populated by upstream
services, not importing their internals.

Pairing strategy: Module 1's network logs and Module 2's phishing content
are separate synthetic datasets with no natural join key (they were built
independently in Modules 1 and 2). To get a realistic, varied combined
dataset instead of a handful of repeated demo strings, we:
    1. Sample N rows from Module 1's synthetic_network_logs.csv
    2. Sample N rows from Module 2's synthetic_phishing_dataset.csv,
       stratified so the paired is_attack / is_phishing labels roughly
       agree (attack network rows get paired with phishing content more
       often than not, and vice versa for legitimate rows) — simulating a
       real-world correlation between suspicious network activity and
       accompanying malicious content, without claiming a false ground-truth
       join that doesn't actually exist between these two independent
       synthetic datasets.
    3. Run both through their real modules' prediction functions.
    4. Fuse via Module 2's actual bayesian_risk_adjustment().

Run:
    cd module3/src && python build_dataset.py
"""

import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import set_seed, get_logger
from module_bridge import (
    call_module1_predict_network_anomaly,
    call_module2_predict_phishing,
    call_module2_bayesian_risk_adjustment,
    MODULE1_SRC, MODULE2_SRC,
)

logger = get_logger("build_dataset")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_OUT = os.path.join(BASE_DIR, "data", "unified_threat_data.csv")
MODELS_OUT = os.path.join(BASE_DIR, "models")

MODULE1_DATA = os.path.abspath(os.path.join(MODULE1_SRC, "..", "data", "synthetic_network_logs.csv"))
MODULE2_DATA = os.path.abspath(os.path.join(MODULE2_SRC, "..", "data", "synthetic_phishing_dataset.csv"))
MODULE2_MODELS = os.path.abspath(os.path.join(MODULE2_SRC, "..", "models"))

N_SAMPLE = 3000
HISTORICAL_ATTACK_RATE = 0.10  # matches Module 2's own validation default; documented in README
BATCH_SIZE = 500  # subprocess calls in chunks — one giant call risks timeouts/memory spikes


def _batched_call(items_or_df, call_fn, batch_size=BATCH_SIZE):
    """Run a bridge call in chunks and concatenate results, with progress logging."""
    n = len(items_or_df)
    results = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = items_or_df.iloc[start:end] if isinstance(items_or_df, pd.DataFrame) else items_or_df[start:end]
        chunk_result = call_fn(chunk)
        results.extend(chunk_result)
        logger.info(f"  processed {end}/{n}")
    return results


def pair_datasets(net_df: pd.DataFrame, phish_df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """
    Sample n network rows and n phishing-dataset rows, paired so labels
    correlate (~75% same-label pairing, ~25% cross-paired) rather than
    perfectly or randomly matched — simulates realistic partial correlation
    between network anomalies and malicious content without fabricating a
    join key these two independently-generated datasets don't actually share.
    """
    rng = np.random.RandomState(seed)

    net_attack = net_df[net_df["is_attack"] == 1].reset_index(drop=True)
    net_legit = net_df[net_df["is_attack"] == 0].reset_index(drop=True)
    phish_pos = phish_df[phish_df["is_phishing"] == 1].reset_index(drop=True)
    phish_neg = phish_df[phish_df["is_phishing"] == 0].reset_index(drop=True)

    # Preserve Module 1's real ~5% attack rate in the sample
    n_attack = max(1, int(n * net_df["is_attack"].mean()))
    n_legit = n - n_attack

    net_sample = pd.concat([
        net_attack.sample(min(n_attack, len(net_attack)), random_state=seed, replace=len(net_attack) < n_attack),
        net_legit.sample(min(n_legit, len(net_legit)), random_state=seed, replace=len(net_legit) < n_legit),
    ]).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    paired_text, paired_channel, paired_is_phishing = [], [], []
    for is_attack in net_sample["is_attack"]:
        same_label = rng.random() < 0.75
        if (is_attack == 1) == same_label:
            row = phish_pos.sample(1, random_state=rng.randint(1_000_000)).iloc[0]
        else:
            row = phish_neg.sample(1, random_state=rng.randint(1_000_000)).iloc[0]
        paired_text.append(row["text"])
        paired_channel.append(row["channel"])
        paired_is_phishing.append(row["is_phishing"])

    net_sample["text"] = paired_text
    net_sample["channel"] = paired_channel
    net_sample["is_phishing"] = paired_is_phishing
    return net_sample.reset_index(drop=True)


def main():
    set_seed()
    os.makedirs(os.path.dirname(DATA_OUT), exist_ok=True)
    os.makedirs(MODELS_OUT, exist_ok=True)

    logger.info("Loading Module 1 and Module 2 source datasets...")
    net_df = pd.read_csv(MODULE1_DATA, parse_dates=["timestamp"])
    phish_df = pd.read_csv(MODULE2_DATA)

    logger.info(f"Pairing {N_SAMPLE} rows across both datasets...")
    combined = pair_datasets(net_df, phish_df, N_SAMPLE)

    logger.info("Calling Module 1's predict_network_anomaly() (isolated subprocess, batched)...")
    net_input = combined.drop(columns=["is_attack", "text", "channel", "is_phishing"])
    net_results = _batched_call(net_input, call_module1_predict_network_anomaly)

    combined["anomaly_score"] = [r["anomaly_score"] for r in net_results]
    combined["module1_risk_level"] = [r["risk_level"] for r in net_results]
    feature_vector_df = pd.DataFrame([r["feature_vector"] for r in net_results])
    combined = pd.concat([combined.reset_index(drop=True), feature_vector_df.reset_index(drop=True)], axis=1)

    logger.info("Calling Module 2's predict_phishing() (isolated subprocess, batched)...")
    phish_results = _batched_call(list(combined["text"]), call_module2_predict_phishing)
    combined["phishing_probability"] = [r["phishing_probability"] for r in phish_results]

    logger.info("Calling Module 2's bayesian_risk_adjustment() (isolated subprocess)...")
    bayes_result = call_module2_bayesian_risk_adjustment(
        anomaly_score=list(combined["anomaly_score"]),
        phishing_probability=list(combined["phishing_probability"]),
        historical_attack_rate=HISTORICAL_ATTACK_RATE,
        prior_probability=[HISTORICAL_ATTACK_RATE] * len(combined),
    )
    combined["final_risk_probability"] = bayes_result["final_risk_probability"]
    combined["risk_category"] = bayes_result["risk_category"]

    combined.to_csv(DATA_OUT, index=False)
    logger.info(f"Saved unified dataset: {combined.shape} -> {DATA_OUT}")
    logger.info(f"Risk category distribution:\n{combined['risk_category'].value_counts()}")

    # Copy Module 2's actual trained phishing model + vectorizer into module3/models/
    # for the dashboard's LIME explainer — loaded via joblib (no source-code import,
    # so no collision) rather than retraining a redundant copy.
    for fname in ["tfidf_vectorizer.pkl", "lightgbm_model.pkl", "naive_bayes_model.pkl", "model_selection.pkl"]:
        src = os.path.join(MODULE2_MODELS, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(MODELS_OUT, fname))
    logger.info(f"Copied Module 2's phishing model artifacts into {MODELS_OUT}")

    return combined


if __name__ == "__main__":
    main()
