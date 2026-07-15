"""
module_bridge.py
------------------
Calls Module 1, Module 2, and Module 3's real, shipped functions in isolated
subprocesses. Same rationale as Module 3's module_bridge.py: module1/src,
module2/src, module3/src, and module4/src all define files named utils.py
(and 1/2 also share predict.py) — importing more than one into the same
interpreter silently corrupts whichever module's utils/predict "loses" the
sys.modules race. Subprocess isolation, with data exchanged via temp JSON
files, sidesteps this entirely.
"""

import json
import os
import subprocess
import sys
import tempfile

import pandas as pd

from utils import get_logger

logger = get_logger("module_bridge")

MODULE1_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "module1", "src"))
MODULE2_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "module2", "src"))
MODULE3_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "module3", "src"))


def _run_isolated(module_src: str, runner_body: str, args: list, timeout: int = 300):
    if not os.path.isdir(module_src):
        raise FileNotFoundError(
            f"{module_src} not found. This bridge expects module1/, module2/, module3/, and "
            f"module4/ to sit side by side in the same parent directory."
        )
    proc = subprocess.run(
        [sys.executable, "-c", runner_body, module_src] + args,
        capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Isolated subprocess ({module_src}) failed:\n{proc.stderr}")


# -----------------------------------------------------------------------------
# Module 1: predict_network_anomaly(df) -> anomaly_score, risk_level, feature_vector
# -----------------------------------------------------------------------------
_MODULE1_RUNNER = r"""
import sys, json, pandas as pd
sys.path.insert(0, sys.argv[1])
from predict import predict_network_anomaly
with open(sys.argv[2]) as f:
    rows = pd.read_json(f, orient="records")
rows["timestamp"] = pd.to_datetime(rows["timestamp"])
result = predict_network_anomaly(rows)
if isinstance(result, dict):
    result = [result]
with open(sys.argv[3], "w") as f:
    json.dump(result, f)
"""


def call_module1_predict_network_anomaly(rows_df: pd.DataFrame) -> list:
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, "rows.json")
        out_path = os.path.join(tmpdir, "result.json")
        rows_df.to_json(in_path, orient="records", date_format="iso")
        _run_isolated(MODULE1_SRC, _MODULE1_RUNNER, [in_path, out_path])
        with open(out_path) as f:
            return json.load(f)


# -----------------------------------------------------------------------------
# Module 2: predict_phishing(texts) -> phishing_probability
# -----------------------------------------------------------------------------
_MODULE2_PHISHING_RUNNER = r"""
import sys, json
sys.path.insert(0, sys.argv[1])
from phishing_detector import predict_phishing
with open(sys.argv[2]) as f:
    texts = json.load(f)
result = predict_phishing(texts)
if isinstance(result, dict):
    result = [result]
with open(sys.argv[3], "w") as f:
    json.dump(result, f)
"""


def call_module2_predict_phishing(texts: list) -> list:
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, "texts.json")
        out_path = os.path.join(tmpdir, "result.json")
        with open(in_path, "w") as f:
            json.dump(texts, f)
        _run_isolated(MODULE2_SRC, _MODULE2_PHISHING_RUNNER, [in_path, out_path])
        with open(out_path) as f:
            return json.load(f)


# -----------------------------------------------------------------------------
# Module 2: bayesian_risk_adjustment(...) -> final_risk_probability
# -----------------------------------------------------------------------------
_MODULE2_BAYES_RUNNER = r"""
import sys, json
sys.path.insert(0, sys.argv[1])
from bayesian_layer import bayesian_risk_adjustment, risk_category_from_probability
with open(sys.argv[2]) as f:
    payload = json.load(f)
final = bayesian_risk_adjustment(
    anomaly_score=payload["anomaly_score"],
    phishing_probability=payload["phishing_probability"],
    historical_attack_rate=payload["historical_attack_rate"],
    prior_probability=payload["prior_probability"],
)
category = risk_category_from_probability(final)
with open(sys.argv[3], "w") as f:
    json.dump({"final_risk_probability": final, "risk_category": category}, f)
"""


def call_module2_bayesian_risk_adjustment(anomaly_score: float, phishing_probability: float,
                                           historical_attack_rate: float, prior_probability: float) -> dict:
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, "payload.json")
        out_path = os.path.join(tmpdir, "result.json")
        with open(in_path, "w") as f:
            json.dump({
                "anomaly_score": anomaly_score,
                "phishing_probability": phishing_probability,
                "historical_attack_rate": historical_attack_rate,
                "prior_probability": prior_probability,
            }, f)
        _run_isolated(MODULE2_SRC, _MODULE2_BAYES_RUNNER, [in_path, out_path])
        with open(out_path) as f:
            return json.load(f)


# -----------------------------------------------------------------------------
# Module 3: generate_xai_report(live_event=...) -> important_features, feature_contributions, risk_summary
# -----------------------------------------------------------------------------
_MODULE3_XAI_RUNNER = r"""
import sys, json
sys.path.insert(0, sys.argv[1])
from xai_engine import generate_xai_report
with open(sys.argv[2]) as f:
    live_event = json.load(f)
result = generate_xai_report(live_event=live_event)
with open(sys.argv[3], "w") as f:
    json.dump(result, f)
"""


def call_module3_generate_xai_report(live_event: dict) -> dict:
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, "live_event.json")
        out_path = os.path.join(tmpdir, "result.json")
        with open(in_path, "w") as f:
            json.dump(live_event, f)
        _run_isolated(MODULE3_SRC, _MODULE3_XAI_RUNNER, [in_path, out_path], timeout=180)
        with open(out_path) as f:
            return json.load(f)


if __name__ == "__main__":
    logger.info("Testing all four cross-module bridges (requires module1/module2/module3 trained)...")

    net_df = pd.read_csv(
        os.path.abspath(os.path.join(MODULE1_SRC, "..", "data", "synthetic_network_logs.csv")),
        parse_dates=["timestamp"],
    )
    sample = net_df.iloc[300:301].drop(columns=["is_attack"])

    r1 = call_module1_predict_network_anomaly(sample)
    assert "anomaly_score" in r1[0]
    logger.info(f"Module 1 bridge OK: anomaly_score={r1[0]['anomaly_score']:.2f}")

    r2 = call_module2_predict_phishing(["URGENT verify your account now at fake-login.tk"])
    assert "phishing_probability" in r2[0]
    logger.info(f"Module 2 phishing bridge OK: phishing_probability={r2[0]['phishing_probability']:.4f}")

    r3 = call_module2_bayesian_risk_adjustment(r1[0]["anomaly_score"], r2[0]["phishing_probability"], 0.10, 0.10)
    assert "final_risk_probability" in r3
    logger.info(f"Module 2 Bayesian bridge OK: final_risk_probability={r3['final_risk_probability']:.4f}")

    r4 = call_module3_generate_xai_report({
        "feature_vector": r1[0]["feature_vector"],
        "anomaly_score": r1[0]["anomaly_score"],
        "phishing_probability": r2[0]["phishing_probability"],
        "final_risk_probability": r3["final_risk_probability"],
        "risk_category": r3["risk_category"],
    })
    assert "important_features" in r4
    logger.info(f"Module 3 bridge OK: top features={r4['important_features'][:3]}")

    logger.info("module_bridge.py self-test passed — all four bridges functional.")
