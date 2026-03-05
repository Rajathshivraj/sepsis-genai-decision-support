"""
risk_calibration.py — Platt-scaling calibration for XGBoost risk scores.

Wraps the existing XGBoost model with sklearn's CalibratedClassifierCV
(sigmoid/Platt scaling) to produce better-calibrated probability estimates.

This module does NOT modify the original model file.  It loads the saved
pickle, fits a thin calibration layer on held-out data, and exposes a
``calibrate_score`` function that maps a raw XGBoost probability to a
calibrated one.
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Optional, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV

from src.utils.logger import setup_logger

logger = setup_logger("risk_calibration")

# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------
_calibrator: Optional[CalibratedClassifierCV] = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_calibrator(
    X: np.ndarray,
    y: np.ndarray,
    model_path: str = "models/xgboost_model.pkl",
    method: str = "sigmoid",
    cv: int = 5,
) -> CalibratedClassifierCV:
    """Wrap the saved XGBoost model with Platt scaling.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (should be the same scale used during training).
    y : np.ndarray
        Binary target labels.
    model_path : str
        Path to the saved XGBoost pickle.
    method : str
        Calibration method — ``"sigmoid"`` (Platt) or ``"isotonic"``.
    cv : int
        Number of cross-validation folds for calibration fitting.

    Returns
    -------
    CalibratedClassifierCV
        Calibrated model wrapper.
    """
    global _calibrator

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"XGBoost model not found at {model_path}")

    with open(model_path, "rb") as f:
        base_model = pickle.load(f)

    logger.info(
        "Fitting Platt-scaling calibrator (method=%s, cv=%d) on %d samples …",
        method, cv, len(y),
    )

    calibrated = CalibratedClassifierCV(
        estimator=base_model,
        method=method,
        cv=cv,
    )
    calibrated.fit(X, y)

    _calibrator = calibrated
    logger.info("Calibrator ready.")
    return calibrated


def calibrate_score(raw_score: float) -> float:
    """Map a raw XGBoost probability to a calibrated probability.

    If no calibrator has been fitted, returns the raw score unchanged.

    Parameters
    ----------
    raw_score : float
        Raw predicted probability from XGBoost, in ``[0, 1]``.

    Returns
    -------
    float
        Calibrated probability, in ``[0, 1]``.
    """
    if _calibrator is None:
        logger.warning("No calibrator fitted — returning raw score.")
        return raw_score

    # CalibratedClassifierCV expects a 2D array
    # We create a minimal dummy input and use the calibrator's calibration
    # mapping. Since we only have the score (not raw features), we apply
    # a simple Platt sigmoid transform using the fitted parameters.
    #
    # Fallback: return raw score if calibrator internals aren't accessible.
    try:
        # Access the first calibrated classifier's calibrators
        calibrated_classifiers = _calibrator.calibrated_classifiers_
        calibrator_obj = calibrated_classifiers[0].calibrators[0]
        # Platt sigmoid: P = 1 / (1 + exp(A*f + B))
        a = calibrator_obj.a_
        b = calibrator_obj.b_
        calibrated = 1.0 / (1.0 + np.exp(a * raw_score + b))
        return float(np.clip(calibrated, 0.0, 1.0))
    except Exception:
        logger.debug("Calibrator parameter extraction failed — returning raw score.")
        return raw_score


def save_calibrator(path: str = "models/calibrated_xgboost.pkl") -> None:
    """Persist the fitted calibrator to disk."""
    if _calibrator is None:
        raise RuntimeError("No calibrator to save. Call build_calibrator() first.")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(_calibrator, f)
    logger.info("Calibrated model saved → %s", path)


def load_calibrator(path: str = "models/calibrated_xgboost.pkl") -> None:
    """Load a previously saved calibrator from disk."""
    global _calibrator

    if not os.path.exists(path):
        raise FileNotFoundError(f"Calibrator not found at {path}")

    with open(path, "rb") as f:
        _calibrator = pickle.load(f)
    logger.info("Calibrated model loaded ← %s", path)
