"""
module_bridge.py
------------------
Calls Module 1's and Module 2's real, shipped prediction functions in
isolated subprocesses. This is necessary, not cosmetic: module1/src,
module2/src, and module3/src each define files named utils.py and
predict.py. The moment more than one of those directories is added to
sys.path and imported in the same interpreter, Python's sys.modules cache
serves whichever "utils" or "predict" module happened to import first to
EVERY subsequent `from utils import ...` — silently feeding, say, Module 2's
code Module 1's utils functions. Subprocess isolation sidesteps this
entirely and, as a side effect, mirrors how independently-deployed services
in a real platform would actually talk to each other (decoupled processes,
not shared interpreter state).

Each bridge function exchanges data via temp JSON files (not stdout/CLI args
— stdout can carry logging noise, and large payloads exceed OS argv limits).
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


def _run_isolated(module_src: str, runner_body: str, args: list, timeout: int = 300):
    if not os.path.isdir(module_src):
        raise FileNotFoundError(
            f"{module_src} not found. This bridge expects module1/, module2/, and module3/ "
            f"to sit side by side in the same parent directory."
        )
    proc = subprocess.run(
        [sys.executable, "-c", runner_body, module_src] + args,
        capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Isolated subprocess failed:\n{proc.stderr}")


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
if isinstance(final, float):
    final = [final]
else:
    final = list(final)
categories = [risk_category_from_probability(f) for f in final]
with open(sys.argv[3], "w") as f:
    json.dump({"final_risk_probability": final, "risk_category": categories}, f)
"""


def call_module2_bayesian_risk_adjustment(anomaly_score: list, phishing_probability: list,
                                           historical_attack_rate: float, prior_probability: list) -> dict:
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
