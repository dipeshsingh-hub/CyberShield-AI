"""
anomaly_detector.py
--------------------
Core ensemble anomaly detection engine.

PUBLIC CONTRACT (do not rename):
    - AnomalyEnsemble.trained_models  -> dict {"isolation_forest": ..., "oneclass_svm": ...}
    - AnomalyEnsemble.compute_anomaly_score(processed_features) -> returns `anomaly_score`
      computed as: 0.6 * IsolationForestScore + 0.4 * OneClassSVMScore, normalized to 0-100.

Why an ensemble of these two specific models:
    - IsolationForest: cheap, tree-based, handles non-linear separability well,
      scales to large n with O(n log n) training. Good general-purpose anomaly
      catcher, but can be less precise on tightly-clustered normal traffic.
    - OneClassSVM (RBF kernel): learns a tighter boundary around the dense
      "normal" region in feature space, better at catching anomalies that sit
      just outside a compact normal cluster (which IsolationForest sometimes
      misses because its splits are axis-aligned).
    - Combining them (0.6/0.4, weighted toward the more scalable/robust
      IsolationForest) reduces variance from either model's blind spots
      without adding real inference latency (both are O(1)-ish per row).
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from utils import get_logger, minmax_to_100, classify_risk_level

logger = get_logger("anomaly_detector")

ISO_WEIGHT = 0.6
SVM_WEIGHT = 0.4


class AnomalyEnsemble:
    """
    Wraps IsolationForest + OneClassSVM into a single fit/score interface.

    Usage:
        ensemble = AnomalyEnsemble(contamination=0.05)
        ensemble.fit(processed_features)
        anomaly_score = ensemble.compute_anomaly_score(processed_features)
    """

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        # trained_models: public contract name, must not be renamed.
        self.trained_models = {}
        # Bounds learned at training time, used to normalize inference-time
        # scores onto the SAME 0-100 scale (prevents a single inference row
        # from trivially min-max-ing itself to 0 or 100).
        self._score_bounds = {}
        self._is_fitted = False

    def fit(self, processed_features: np.ndarray) -> "AnomalyEnsemble":
        n_estimators = 200
        logger.info(
            f"Training IsolationForest (n_estimators={n_estimators}, "
            f"contamination={self.contamination}) on {processed_features.shape[0]} rows..."
        )
        iso_forest = IsolationForest(
            n_estimators=n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        iso_forest.fit(processed_features)

        # OneClassSVM does not scale well past ~10-20k rows (O(n^2)-ish), so we
        # subsample for training only. Scoring at inference is still O(1) per row.
        svm_train_data = processed_features
        max_svm_rows = 8000
        if processed_features.shape[0] > max_svm_rows:
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(processed_features.shape[0], max_svm_rows, replace=False)
            svm_train_data = processed_features[idx]
            logger.info(f"Subsampled {max_svm_rows} rows for OneClassSVM training (perf).")

        logger.info(f"Training OneClassSVM (nu={self.contamination}, kernel=rbf)...")
        oc_svm = OneClassSVM(nu=self.contamination, kernel="rbf", gamma="scale")
        oc_svm.fit(svm_train_data)

        self.trained_models = {
            "isolation_forest": iso_forest,
            "oneclass_svm": oc_svm,
        }

        # Establish normalization bounds from the full training set
        iso_raw = self._raw_iso_score(processed_features)
        svm_raw = self._raw_svm_score(processed_features)
        _, iso_min, iso_max = minmax_to_100(iso_raw)
        _, svm_min, svm_max = minmax_to_100(svm_raw)
        self._score_bounds = {
            "iso_min": iso_min, "iso_max": iso_max,
            "svm_min": svm_min, "svm_max": svm_max,
        }
        self._is_fitted = True
        logger.info("AnomalyEnsemble training complete.")
        return self

    def _raw_iso_score(self, processed_features: np.ndarray) -> np.ndarray:
        # decision_function: higher = more normal. Flip sign so higher = more anomalous.
        return -self.trained_models["isolation_forest"].decision_function(processed_features)

    def _raw_svm_score(self, processed_features: np.ndarray) -> np.ndarray:
        return -self.trained_models["oneclass_svm"].decision_function(processed_features)

    def compute_anomaly_score(self, processed_features: np.ndarray) -> np.ndarray:
        """
        Compute the ensemble anomaly_score (0-100) for each row.

            anomaly_score = 0.6 * IsolationForestScore + 0.4 * OneClassSVMScore

        Each sub-score is independently min-max normalized to 0-100 using the
        bounds learned at training time, THEN combined. Since the weights sum
        to 1.0, the combined anomaly_score is guaranteed to stay within [0, 100].
        """
        if not self._is_fitted:
            raise RuntimeError("AnomalyEnsemble must be fit() before scoring.")

        iso_raw = self._raw_iso_score(processed_features)
        svm_raw = self._raw_svm_score(processed_features)

        iso_scaled, _, _ = minmax_to_100(
            iso_raw, self._score_bounds["iso_min"], self._score_bounds["iso_max"]
        )
        svm_scaled, _, _ = minmax_to_100(
            svm_raw, self._score_bounds["svm_min"], self._score_bounds["svm_max"]
        )

        # anomaly_score: public contract name, must not be renamed.
        anomaly_score = ISO_WEIGHT * iso_scaled + SVM_WEIGHT * svm_scaled
        anomaly_score = np.clip(anomaly_score, 0.0, 100.0)
        return anomaly_score

    def risk_levels(self, anomaly_score: np.ndarray) -> list:
        return [classify_risk_level(s) for s in anomaly_score]

    def get_bounds(self) -> dict:
        """Expose normalization bounds so they can be persisted alongside the models."""
        return dict(self._score_bounds)

    def set_bounds(self, bounds: dict) -> None:
        """Restore normalization bounds after loading a saved ensemble."""
        self._score_bounds = bounds
        self._is_fitted = True
