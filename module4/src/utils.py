"""
utils.py
--------
Shared utilities for Module 4 (SOC orchestration + playbook generation).
Independent from module1/2/3's utils.py — see module_bridge.py for why all
cross-module calls go through subprocess isolation rather than direct import.
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone

RANDOM_SEED = 42


def get_logger(name: str, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_env(var_name: str, purpose: str) -> str:
    """
    Reads a required environment variable, raising a clear, actionable error
    if it's missing — never silently proceeding with a hardcoded fallback
    for something security-sensitive like an API key.
    """
    value = os.environ.get(var_name)
    if not value:
        raise EnvironmentError(
            f"Missing required environment variable '{var_name}' (needed for {purpose}). "
            f"Set it with: export {var_name}=<your-key>  (see README for details)."
        )
    return value


def safe_json_dumps(obj, **kwargs) -> str:
    """json.dumps that doesn't choke on numpy scalars or datetimes leaking in from upstream modules."""
    def default(o):
        if hasattr(o, "item"):  # numpy scalar
            return o.item()
        if isinstance(o, datetime):
            return o.isoformat()
        return str(o)
    return json.dumps(obj, default=default, **kwargs)


if __name__ == "__main__":
    logger = get_logger("utils_selftest")
    logger.info(f"utc_now_iso(): {utc_now_iso()}")
    logger.info(f"safe_json_dumps(): {safe_json_dumps({'a': 1, 'ts': datetime.now(timezone.utc)})}")
    try:
        require_env("DEFINITELY_NOT_SET_XYZ", "self-test")
        print("FAILED: should have raised")
    except EnvironmentError as e:
        logger.info(f"require_env() correctly raised: {e}")
    logger.info("utils.py self-test passed.")
