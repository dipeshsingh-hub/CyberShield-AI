"""
predict.py
----------
Top-level entrypoint for Module 2. Re-exports the required public contract
so later modules can do:

    from predict import predict_phishing, bayesian_risk_adjustment

PUBLIC CONTRACT (do not rename):
    predict_phishing(text_or_texts) -> phishing_probability
    bayesian_risk_adjustment(...)    -> final_risk_probability

Also provides `assess_risk()`, a convenience wrapper that runs the full
Module 2 pipeline (text -> phishing_probability -> Bayesian fusion) in one
call. This is NOT part of the locked public contract — it's a thin
convenience layer on top of it, safe to change without breaking downstream
imports.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phishing_detector import predict_phishing  # noqa: F401 (re-exported)
from bayesian_layer import bayesian_risk_adjustment, risk_category_from_probability  # noqa: F401
from utils import get_logger

logger = get_logger("predict")


def assess_risk(text: str, anomaly_score: float, historical_attack_rate: float = 0.10,
                 prior_probability: float = None) -> dict:
    """
    Convenience wrapper: given a piece of content (email/sms/url/login/api
    text) and a Module-1-derived anomaly_score, run the full Module 2
    pipeline and return a single risk assessment.

    Args:
        text: raw content string for the phishing content classifier
        anomaly_score: 0-100, typically Module 1's predict_network_anomaly() output
        historical_attack_rate: base rate to calibrate the Bayesian fusion against
        prior_probability: case-specific prior; defaults to historical_attack_rate
            if not provided (i.e. "no extra context beyond the base rate")

    Returns:
        {
            "phishing_probability": float,
            "final_risk_probability": float,
            "risk_category": "Low" | "Medium" | "Critical",
        }
    """
    if prior_probability is None:
        prior_probability = historical_attack_rate

    phishing_result = predict_phishing(text)
    phishing_probability = phishing_result["phishing_probability"]

    final_risk_probability = bayesian_risk_adjustment(
        anomaly_score=anomaly_score,
        phishing_probability=phishing_probability,
        historical_attack_rate=historical_attack_rate,
        prior_probability=prior_probability,
    )

    return {
        "phishing_probability": phishing_probability,
        "final_risk_probability": final_risk_probability,
        "risk_category": risk_category_from_probability(final_risk_probability),
    }


if __name__ == "__main__":
    result = assess_risk(
        text="URGENT: Your Chase Bank account has your account will be suspended. "
             "Verify at chase-secure-verify471.com",
        anomaly_score=82.0,
        historical_attack_rate=0.10,
    )
    logger.info(f"assess_risk() smoke test: {result}")
