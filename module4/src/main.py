"""
main.py
-------
Orchestration entrypoint for the SOC Assistant. Wires Modules 1, 2, 3, and 4
together into a single event-processing pipeline.

Workflow (matches spec exactly):
    Load Module 1
        v
    Predict anomaly            -> anomaly_score, feature_vector (Module 1)
        v
    Run phishing classifier    -> phishing_probability (Module 2)
        v
    Bayesian adjustment        -> final_risk_probability, risk_level (Module 2)
        v
    Generate SHAP explanation  -> important_features, feature_contributions (Module 3)
        v
    If risk > 70 -> Generate Playbook (Module 4, LLM gated behind this exact check)
    Else         -> Store event only

Every module is independently executable (run any of module1/2/3/4's
src/*.py files directly — each has been verified to work standalone in its
own module's test suite). This file is what CONNECTS them, via the isolated
subprocess bridges in module_bridge.py (see that file's docstring for why
direct in-process imports across modules would silently corrupt each
other's utils.py/predict.py).

Run a single event:
    cd module4/src && python main.py --source-ip 203.0.113.9 --channel login_attempt

Run a batch from Module 1's synthetic data + paired Module 2 content:
    cd module4/src && python main.py --batch 20
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import get_logger
import module_bridge as bridge
import event_store
from playbook_generator import generate_response_playbook, RISK_THRESHOLD_0_100

logger = get_logger("main")

HISTORICAL_ATTACK_RATE = 0.10  # same default used in Module 2's own validation + Module 3's ETL


def process_event(network_row: pd.DataFrame, content_text: str, channel: str = "unknown", mock_llm: bool = None) -> dict:
    """
    Runs one event through the full pipeline. network_row is a single-row
    DataFrame matching Module 1's raw schema; content_text is the
    accompanying email/SMS/URL/login/API content for Module 2; channel
    identifies which of the 5 content types content_text is (drives
    threat-category classification in the playbook — see threat_intel.py).

    Returns the FINAL OUTPUT dict exactly as specified:
        {anomaly_score, phishing_probability, final_risk_probability,
         risk_level, important_features, playbook}
    """
    source_ip = network_row["source_ip"].iloc[0]
    logger.info(f"Processing event: source_ip={source_ip}")

    # --- Step 1+2: Module 1 — predict anomaly ---
    try:
        net_result = bridge.call_module1_predict_network_anomaly(network_row)[0]
    except Exception as e:
        logger.exception(f"Module 1 prediction failed for {source_ip}")
        raise RuntimeError(f"Module 1 (anomaly detection) failed: {e}") from e
    anomaly_score = net_result["anomaly_score"]
    feature_vector = net_result["feature_vector"]
    logger.info(f"  anomaly_score={anomaly_score:.1f} ({net_result['risk_level']})")

    # --- Step 3: Module 2 — phishing classifier ---
    try:
        phishing_result = bridge.call_module2_predict_phishing([content_text])[0]
    except Exception as e:
        logger.exception(f"Module 2 phishing classification failed for {source_ip}")
        raise RuntimeError(f"Module 2 (phishing detection) failed: {e}") from e
    phishing_probability = phishing_result["phishing_probability"]
    logger.info(f"  phishing_probability={phishing_probability:.3f}")

    # --- Step 4: Module 2 — Bayesian adjustment ---
    try:
        bayes_result = bridge.call_module2_bayesian_risk_adjustment(
            anomaly_score=anomaly_score,
            phishing_probability=phishing_probability,
            historical_attack_rate=HISTORICAL_ATTACK_RATE,
            prior_probability=HISTORICAL_ATTACK_RATE,
        )
    except Exception as e:
        logger.exception(f"Module 2 Bayesian adjustment failed for {source_ip}")
        raise RuntimeError(f"Module 2 (Bayesian risk adjustment) failed: {e}") from e
    final_risk_probability = bayes_result["final_risk_probability"]
    risk_level = bayes_result["risk_category"]
    logger.info(f"  final_risk_probability={final_risk_probability:.3f} ({risk_level})")

    # --- Step 5: Module 3 — SHAP explanation ---
    try:
        xai_result = bridge.call_module3_generate_xai_report({
            "feature_vector": feature_vector,
            "anomaly_score": anomaly_score,
            "phishing_probability": phishing_probability,
            "final_risk_probability": final_risk_probability,
            "risk_category": risk_level,
        })
    except Exception as e:
        # SHAP explanation failing shouldn't take down the whole pipeline —
        # the risk score itself is still valid and actionable. Degrade
        # gracefully: log loudly, continue with empty explanation fields.
        logger.exception(f"Module 3 SHAP explanation failed for {source_ip} — continuing without it")
        xai_result = {"important_features": [], "feature_contributions": {}}
    important_features = xai_result["important_features"]
    feature_contributions = xai_result["feature_contributions"]
    logger.info(f"  top features: {important_features[:3]}")

    # --- Step 6: gate on final_risk_probability > 70 ---
    frp_0_100 = final_risk_probability * 100
    event_record = {
        "anomaly_score": anomaly_score,
        "phishing_probability": phishing_probability,
        "final_risk_probability": final_risk_probability,
        "risk_level": risk_level,
        "important_features": important_features,
        "feature_contributions": feature_contributions,
        "source_ip": source_ip,
        "destination_ip": network_row["destination_ip"].iloc[0],
        "channel": channel,
        "protocol": network_row["protocol"].iloc[0],
        "text": content_text,
    }

    if frp_0_100 > RISK_THRESHOLD_0_100:
        logger.warning(f"  RISK {frp_0_100:.1f} > {RISK_THRESHOLD_0_100} -> generating playbook (LLM will be invoked)")
        try:
            playbook = generate_response_playbook(event_record, mock_llm=mock_llm)
        except Exception as e:
            logger.exception(f"Playbook generation failed for {source_ip}")
            playbook = {"error": f"Playbook generation failed: {e}"}
    else:
        logger.info(f"  risk {frp_0_100:.1f} <= {RISK_THRESHOLD_0_100} -> storing event only, LLM NOT invoked")
        playbook = "No response required."

    event_store.store_event({**event_record, "playbook": playbook if isinstance(playbook, dict) else None})

    return {
        "anomaly_score": anomaly_score,
        "phishing_probability": phishing_probability,
        "final_risk_probability": final_risk_probability,
        "risk_level": risk_level,
        "important_features": important_features,
        "playbook": playbook,
    }


def run_single(source_ip_row: pd.Series, content_text: str, channel: str, mock_llm: bool = None) -> dict:
    row_df = pd.DataFrame([source_ip_row])
    result = process_event(row_df, content_text, channel=channel, mock_llm=mock_llm)
    return result


def run_batch(n: int, mock_llm: bool = None) -> list:
    """Pulls n rows from Module 1's + Module 2's real synthetic datasets, paired, and processes each."""
    module1_data = os.path.abspath(os.path.join(bridge.MODULE1_SRC, "..", "data", "synthetic_network_logs.csv"))
    module2_data = os.path.abspath(os.path.join(bridge.MODULE2_SRC, "..", "data", "synthetic_phishing_dataset.csv"))

    net_df = pd.read_csv(module1_data, parse_dates=["timestamp"])
    phish_df = pd.read_csv(module2_data)

    rng = np.random.RandomState(7)
    # Bias the sample toward attack rows so the demo actually exercises the
    # playbook-generation path, not just "store event only" every time.
    net_attack = net_df[net_df["is_attack"] == 1].sample(min(n // 2, (net_df["is_attack"] == 1).sum()), random_state=7)
    net_legit = net_df[net_df["is_attack"] == 0].sample(n - len(net_attack), random_state=7)
    net_sample = pd.concat([net_attack, net_legit]).sample(frac=1.0, random_state=7)

    results = []
    for _, row in net_sample.iterrows():
        is_attack = row["is_attack"] == 1
        pool = phish_df[phish_df["is_phishing"] == (1 if is_attack else 0)]
        content_row = pool.sample(1, random_state=rng.randint(1_000_000)).iloc[0]
        try:
            result = process_event(
                pd.DataFrame([row]).drop(columns=["is_attack"]),
                content_row["text"],
                channel=content_row["channel"],
                mock_llm=mock_llm,
            )
            results.append(result)
        except Exception:
            logger.exception(f"Skipping event for {row.get('source_ip')} due to pipeline failure")
    return results


def main():
    parser = argparse.ArgumentParser(description="SOC Assistant orchestrator (Module 4)")
    parser.add_argument("--batch", type=int, default=None, help="Process N sampled events from Module 1+2's real data")
    parser.add_argument("--source-ip", type=str, default=None, help="Run a single ad-hoc event (requires --content)")
    parser.add_argument("--content", type=str, default=None, help="Content text for the single-event mode")
    parser.add_argument("--mock-llm", action="store_true", help="Use mock LLM mode (no API key / API cost required)")
    args = parser.parse_args()

    mock_llm = True if args.mock_llm else None

    if args.batch:
        logger.info(f"=== Running batch of {args.batch} events ===")
        results = run_batch(args.batch, mock_llm=mock_llm)
        n_playbooks = sum(1 for r in results if isinstance(r["playbook"], dict))
        logger.info(f"=== Batch complete: {len(results)} events processed, {n_playbooks} playbooks generated ===")
        for r in results:
            logger.info(f"  risk={r['final_risk_probability']:.2f} ({r['risk_level']}) "
                        f"playbook={'YES' if isinstance(r['playbook'], dict) else 'no'}")
        return results

    if args.source_ip:
        module1_data = os.path.abspath(os.path.join(bridge.MODULE1_SRC, "..", "data", "synthetic_network_logs.csv"))
        net_df = pd.read_csv(module1_data, parse_dates=["timestamp"])
        row = net_df[net_df["source_ip"] == args.source_ip]
        if row.empty:
            logger.error(f"source_ip {args.source_ip} not found in Module 1's dataset — using a random attack row instead.")
            row = net_df[net_df["is_attack"] == 1].sample(1, random_state=1)
        content = args.content or "URGENT verify your account at suspicious-login-verify.tk"
        result = run_single(row.drop(columns=["is_attack"]).iloc[0], content, channel="url", mock_llm=mock_llm)
        logger.info(f"=== Result ===\nanomaly_score={result['anomaly_score']:.1f}, "
                    f"phishing_probability={result['phishing_probability']:.3f}, "
                    f"final_risk_probability={result['final_risk_probability']:.3f}, "
                    f"risk_level={result['risk_level']}, "
                    f"playbook={'generated' if isinstance(result['playbook'], dict) else result['playbook']}")
        return result

    parser.print_help()


if __name__ == "__main__":
    main()
