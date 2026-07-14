"""
integration_demo.py
----------------------
Demonstrates Module 2 actually consuming Module 1's live output — not just
matching a documented contract. Requires module1/ to be trained (its
models/*.pkl must exist) and sit alongside module2/ in the same parent
directory (../../module1 relative to this file), matching the platform's
folder layout.

IMPLEMENTATION NOTE: Module 1 and Module 2 both have files named utils.py,
predict.py, etc. A naive `sys.path.insert` + `import predict` from both
directories collides in Python's sys.modules cache — whichever module
imports first "wins" the name for both, silently feeding Module 2 code
Module 1's utils module or vice versa. Rather than rename Module 1's
already-shipped files (which the spec explicitly prohibits) or fight
Python's import system with fragile importlib gymnastics, this demo calls
Module 1's predict.py in an isolated subprocess and exchanges data as JSON.
This also happens to mirror how independently-deployed services would
actually talk to each other in a real multi-module platform — the modules
stay genuinely decoupled, not accidentally sharing interpreter state.

Pipeline:
    1. Take raw network log rows -> Module 1's predict_network_anomaly()
       (subprocess) -> anomaly_score
    2. Take associated content (e.g. the phishing email that triggered this
       traffic) -> Module 2's predict_phishing() -> phishing_probability
    3. Fuse both via bayesian_risk_adjustment() -> final_risk_probability

Run:
    cd module2/src && python integration_demo.py
"""

import json
import os
import subprocess
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODULE1_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "module1", "src"))
MODULE1_DATA = os.path.abspath(os.path.join(MODULE1_SRC, "..", "data", "synthetic_network_logs.csv"))

from utils import get_logger
from phishing_detector import predict_phishing
from bayesian_layer import bayesian_risk_adjustment, risk_category_from_probability

logger = get_logger("integration_demo")

# Historical attack rate used across this demo: matches Module 1's synthetic
# dataset's actual attack rate (5%), so the Bayesian fusion's calibration
# anchor reflects the same population Module 1 was trained on.
HISTORICAL_ATTACK_RATE = 0.05

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
    """
    Calls Module 1's predict_network_anomaly() in an isolated subprocess to
    avoid the utils.py/predict.py module-name collision described above.
    Data is exchanged via temp files rather than CLI args or stdout, since
    a full log batch as a JSON string is both too large for OS argv limits
    and gets misinterpreted by pandas as a filepath if passed inline.
    """
    if not os.path.isdir(MODULE1_SRC):
        raise FileNotFoundError(
            f"Module 1 not found at {MODULE1_SRC}. This demo expects module1/ and "
            f"module2/ to sit side by side in the same parent directory."
        )

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, "rows.json")
        out_path = os.path.join(tmpdir, "result.json")
        rows_df.to_json(in_path, orient="records", date_format="iso")

        proc = subprocess.run(
            [sys.executable, "-c", _MODULE1_RUNNER, MODULE1_SRC, in_path, out_path],
            capture_output=True, text=True, timeout=60,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Module 1 subprocess failed:\n{proc.stderr}")

        with open(out_path) as f:
            return json.load(f)


def run_integration_demo():
    net_df = pd.read_csv(MODULE1_DATA, parse_dates=["timestamp"])
    attack_rows = net_df[net_df["is_attack"] == 1].sample(3, random_state=7)
    legit_rows = net_df[net_df["is_attack"] == 0].sample(3, random_state=7)
    demo_df = pd.concat([attack_rows, legit_rows]).sort_values("timestamp")
    sample_rows = demo_df.drop(columns=["is_attack"])
    true_labels = demo_df["is_attack"].values

    network_results = call_module1_predict_network_anomaly(sample_rows)

    # Pair each network event with plausible accompanying content — in a real
    # system this would be the actual email/SMS/login attempt correlated to
    # that network session; here we pair suspicious traffic with a phishing
    # sample and normal traffic with legitimate content, to demonstrate the
    # full pipeline end to end.
    phishing_sample = ("URGENT: Your Apple account has your account will be suspended. "
                        "Verify at apple-secure-verify882.com")
    legit_sample = "Hi Emma, following up regarding quarterly report attached. Let me know if you have any questions, thanks."

    logger.info("=== Module 1 -> Module 2 integration demo ===")
    for i, net_result in enumerate(network_results):
        true_attack = bool(true_labels[i])
        content = phishing_sample if true_attack else legit_sample

        phishing_result = predict_phishing(content)
        phishing_probability = phishing_result["phishing_probability"]

        final_risk_probability = bayesian_risk_adjustment(
            anomaly_score=net_result["anomaly_score"],
            phishing_probability=phishing_probability,
            historical_attack_rate=HISTORICAL_ATTACK_RATE,
            prior_probability=HISTORICAL_ATTACK_RATE,
        )
        risk_category = risk_category_from_probability(final_risk_probability)

        logger.info(
            f"Row {i} [true_attack={true_attack}]: "
            f"Module1 anomaly_score={net_result['anomaly_score']:.1f} ({net_result['risk_level']}) | "
            f"Module2 phishing_probability={phishing_probability:.3f} | "
            f"-> final_risk_probability={final_risk_probability:.3f} ({risk_category})"
        )


if __name__ == "__main__":
    run_integration_demo()
