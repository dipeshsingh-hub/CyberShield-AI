"""
evaluate_bayesian_layer.py
----------------------------
The self-tests in bayesian_layer.py verify the formula is INTERNALLY
consistent (degenerate case collapses to prior, symmetry holds, etc). This
script verifies something different and more important: when fed noisy,
realistic evidence, does bayesian_risk_adjustment() actually produce
probabilities that (a) discriminate real attacks from legitimate cases, and
(b) are calibrated — i.e. among all cases scored "70% risk", is roughly 70%
of them actually an attack?

Method: simulate N scenarios with a known ground-truth label. For attacks,
draw anomaly_score and phishing_probability from distributions skewed high
(with realistic noise/overlap — some attacks evade behavioral detection,
some evade content detection). For legitimate cases, draw from distributions
skewed low, with occasional false-alarm-prone cases. Run everything through
bayesian_risk_adjustment() and check AUC + calibration against the known
ground truth.
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, brier_score_loss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import set_seed, get_logger
from bayesian_layer import bayesian_risk_adjustment, calibrate_shrinkage, DEFAULT_SHRINKAGE

logger = get_logger("evaluate_bayesian_layer")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVAL_DIR = os.path.join(BASE_DIR, "evaluation")

N_SIM = 8000
HISTORICAL_ATTACK_RATE = 0.10


def simulate_scenarios(n: int, historical_attack_rate: float, seed: int = 42, calibrated: bool = True):
    """
    Args:
        calibrated: if True (the fair test), each sub-detector's TYPICAL output
            for the negative class is centered near historical_attack_rate —
            i.e. the sub-detectors are properly calibrated to the stated base
            rate, which is exactly what bayesian_risk_adjustment()'s math
            assumes. If False, simulates miscalibrated sub-detectors whose
            "normal" output for legitimate cases is already elevated well
            above the base rate (a real failure mode worth documenting
            separately — see the __main__ block).
    """
    rng = np.random.RandomState(seed)
    y_true = rng.binomial(1, historical_attack_rate, size=n)

    anomaly_score = np.zeros(n)
    phishing_probability = np.zeros(n)

    # Beta distribution parameter pairs chosen so mean = a/(a+b) lands near
    # the intended target probability.
    if calibrated:
        legit_typical = (1, 9)    # mean ~0.10, matches historical_attack_rate
        legit_flare = (4, 2)      # mean ~0.67, occasional false-alarm-prone legit traffic
        attack_typical = (6, 2)   # mean ~0.75, clearly elevated
        attack_evasive = (1, 9)   # mean ~0.10, evasive attack — genuinely indistinguishable
                                   # from base rate on this evidence channel alone
    else:
        # Deliberately miscalibrated: legit "normal" output already sits well
        # above the stated base rate — simulates a sub-detector whose
        # probabilities were never actually validated against this population.
        legit_typical = (2, 5)    # mean ~0.286, already 2.9x the 0.10 base rate
        legit_flare = (5, 2)
        attack_typical = (5, 2)
        attack_evasive = (2, 5)

    for i in range(n):
        if y_true[i] == 1:
            a_params = attack_typical if rng.random() > 0.15 else attack_evasive
            p_params = attack_typical if rng.random() > 0.15 else attack_evasive
        else:
            a_params = legit_typical if rng.random() > 0.10 else legit_flare
            p_params = legit_typical if rng.random() > 0.10 else legit_flare
        anomaly_score[i] = np.clip(rng.beta(*a_params) * 100, 0, 100)
        phishing_probability[i] = np.clip(rng.beta(*p_params), 0, 1)

    return y_true, anomaly_score, phishing_probability


def reliability_diagram(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    observed_rates, mean_predicted, counts = [], [], []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        observed_rates.append(y_true[mask].mean())
        mean_predicted.append(y_prob[mask].mean())
        counts.append(mask.sum())
    return np.array(mean_predicted), np.array(observed_rates), np.array(counts)


def run_scenario(label: str, calibrated: bool):
    y_true, anomaly_score, phishing_probability = simulate_scenarios(
        N_SIM, HISTORICAL_ATTACK_RATE, calibrated=calibrated
    )
    prior_probability = np.full(N_SIM, HISTORICAL_ATTACK_RATE)

    final_risk_probability = bayesian_risk_adjustment(
        anomaly_score=anomaly_score,
        phishing_probability=phishing_probability,
        historical_attack_rate=HISTORICAL_ATTACK_RATE,
        prior_probability=prior_probability,
    )

    auc = roc_auc_score(y_true, final_risk_probability)
    brier = brier_score_loss(y_true, final_risk_probability)
    mean_legit = final_risk_probability[y_true == 0].mean()
    mean_attack = final_risk_probability[y_true == 1].mean()

    naive_avg = 0.5 * (anomaly_score / 100.0) + 0.5 * phishing_probability
    naive_auc = roc_auc_score(y_true, naive_avg)
    naive_brier = brier_score_loss(y_true, naive_avg)

    logger.info(f"--- {label} ---")
    logger.info(f"bayesian_risk_adjustment: AUC={auc:.4f}, Brier={brier:.4f}, "
                f"mean(legit)={mean_legit:.3f}, mean(attack)={mean_attack:.3f}")
    logger.info(f"naive average baseline:   AUC={naive_auc:.4f}, Brier={naive_brier:.4f}")

    return y_true, final_risk_probability, {
        "auc": auc, "brier": brier, "naive_auc": naive_auc, "naive_brier": naive_brier,
        "mean_legit": mean_legit, "mean_attack": mean_attack,
    }


def main():
    set_seed()
    os.makedirs(EVAL_DIR, exist_ok=True)

    # Fair test: sub-detectors properly calibrated to historical_attack_rate,
    # matching the assumption bayesian_risk_adjustment()'s math is built on.
    y_true, final_risk_probability, metrics_fair = run_scenario("CALIBRATED inputs (fair test)", calibrated=True)

    # Stress test: sub-detectors miscalibrated (their "normal" legit output
    # already sits well above the stated base rate). Documents a real
    # failure mode rather than hiding it.
    _, _, metrics_miscal = run_scenario("MISCALIBRATED inputs (stress test)", calibrated=False)

    if metrics_fair["brier"] > metrics_fair["naive_brier"]:
        logger.warning(
            "Bayesian fusion's Brier score is WORSE than a naive average even under "
            "calibrated inputs at shrinkage=default. Flagging honestly rather than "
            "shipping it quietly."
        )
    else:
        logger.info("Bayesian fusion beats the naive-average baseline on calibrated inputs "
                     "on Brier score at the default shrinkage — the correction is working.")

    # Demonstrate the proper production workflow: fit shrinkage against real
    # labeled outcomes rather than trusting the hardcoded default forever.
    y_cal, a_cal, p_cal = simulate_scenarios(N_SIM, HISTORICAL_ATTACK_RATE, calibrated=True, seed=99)
    prior_cal = np.full(N_SIM, HISTORICAL_ATTACK_RATE)
    fitted_shrinkage = calibrate_shrinkage(y_cal, a_cal, p_cal, HISTORICAL_ATTACK_RATE, prior_cal)
    logger.info(f"DEFAULT_SHRINKAGE={DEFAULT_SHRINKAGE} vs. data-fitted shrinkage={fitted_shrinkage:.4f} "
                f"(fit on an independent simulated sample — should land in a similar range if the "
                f"default is reasonable)")

    mean_pred, observed, counts = reliability_diagram(y_true, final_risk_probability)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    axes[0].plot(mean_pred, observed, marker="o", label="bayesian_risk_adjustment")
    axes[0].set_xlabel("Mean predicted final_risk_probability")
    axes[0].set_ylabel("Observed attack rate")
    axes[0].set_title(f"Reliability Diagram, calibrated inputs (Brier={metrics_fair['brier']:.4f})")
    axes[0].legend()

    axes[1].hist(final_risk_probability[y_true == 0], bins=30, alpha=0.6, label="Legitimate", density=True)
    axes[1].hist(final_risk_probability[y_true == 1], bins=30, alpha=0.6, label="Attack", density=True)
    axes[1].axvline(0.40, color="orange", linestyle="--", linewidth=1, label="Low/Medium cut")
    axes[1].axvline(0.70, color="red", linestyle="--", linewidth=1, label="Medium/Critical cut")
    axes[1].set_xlabel("final_risk_probability")
    axes[1].set_title(f"Score Distribution by True Label (AUC={metrics_fair['auc']:.4f})")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(EVAL_DIR, "bayesian_layer_validation.png"), dpi=150)
    plt.close(fig)

    logger.info(f"Saved validation plot to {EVAL_DIR}/bayesian_layer_validation.png")
    return {"calibrated": metrics_fair, "miscalibrated": metrics_miscal}


if __name__ == "__main__":
    main()
