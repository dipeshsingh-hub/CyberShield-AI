"""
utils.py
--------
Shared utilities for Module 3 (XAI dashboard). Mirrors module1/module2
conventions but kept fully independent — Module 3 never imports module1 or
module2 source code directly in-process (see build_dataset.py for why:
all three modules define files named utils.py/predict.py, which collide in
Python's sys.modules cache the moment more than one is imported into the
same interpreter). Cross-module calls go through isolated subprocesses;
only self-contained artifacts (pickled sklearn/lightgbm models, no custom
classes) are loaded directly via joblib.
"""

import logging
import os

import joblib
import numpy as np

RANDOM_SEED = 42


def set_seed(seed: int = RANDOM_SEED) -> None:
    np.random.seed(seed)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def save_artifact(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)


def load_artifact(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact not found at {path}.")
    return joblib.load(path)
