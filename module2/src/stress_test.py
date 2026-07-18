"""
stress_test.py
---------------
A separate, harder holdout evaluation for Module 2's selected phishing
classifier (LightGBM, loaded via phishing_detector.predict_phishing()).

The main train/test split in train.py draws from
generate_synthetic_phishing_data()'s DEFAULT phishing sophistication mix:
50% "obvious" (tier1) + 35% "moderate" (tier2) + 15% "advanced" (tier3 —
brand impersonation, locale-aware, typosquat) phishing examples per channel,
per PHISHING_SOPHISTICATION in generate_data.py.

This script does NOT retrain anything and does NOT touch generate_data.py.
It builds an ADDITIONAL, separate, held-out sample using the SAME
CHANNEL_GENERATORS functions already defined in generate_data.py, but calls
them with sophistication_tier="tier3" for every single phishing example
(instead of the blended 50/35/15 mix) across all 5 channels, at the same
~35% phishing / 65% legitimate ratio. Legitimate examples are generated
exactly as train.py does (channel generators already bake in their own
~15-25% "hard negative" rate per generate_data.py's docstrings) — nothing
new is invented here, this only selects the hardest existing phishing tier.

Run:
    cd module2/src && python train.py         # must run first, to produce models/
    cd module2/src && python stress_test.py
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import set_seed, get_logger
from generate_data import CHANNEL_GENERATORS, N_PER_CHANNEL, PHISH_FRACTION
from phishing_detector import predict_phishing, reload_artifacts

logger = get_logger("stress_test")


def build_stress_dataset(n_per_channel: int = N_PER_CHANNEL, phish_fraction: float = PHISH_FRACTION) -> pd.DataFrame:
    set_seed(123)  # different seed from the main dataset — must be a genuinely separate sample
    rows = []
    for channel, generator in CHANNEL_GENERATORS.items():
        n_phish = int(n_per_channel * phish_fraction)
        n_legit = n_per_channel - n_phish

        # 100% tier3 (advanced/hardest) phishing, instead of the 50/35/15 mix.
        for _ in range(n_phish):
            rows.append({"channel": channel, "text": generator(True, "tier3"), "is_phishing": 1})
        # Legitimate examples generated exactly as in the default pipeline —
        # their own baked-in hard-negative rate applies unchanged.
        for _ in range(n_legit):
            rows.append({"channel": channel, "text": generator(False), "is_phishing": 0})

        logger.info(f"{channel}: generated {n_phish} tier3-only phishing + {n_legit} legitimate rows")

    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=123).reset_index(drop=True)
    logger.info(f"Stress-test dataset shape: {df.shape}, phishing rate: {df['is_phishing'].mean():.3%}")
    return df


def main():
    reload_artifacts()
    df = build_stress_dataset()
    y_true = df["is_phishing"].values

    results = predict_phishing(df["text"].tolist())
    probs = np.array([r["phishing_probability"] for r in results])

    fpr, tpr, _ = roc_curve(y_true, probs)
    stress_auc = auc(fpr, tpr)

    y_pred = (probs >= 0.5).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    logger.info(f"STRESS-TEST (tier3-only phishing) ROC AUC: {stress_auc:.4f}")
    logger.info(f"At threshold=0.5: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

    return {
        "stress_auc": stress_auc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


if __name__ == "__main__":
    main()
