"""
shap_explainer.py — SHAP-based feature importance for XGBoost predictions.

Computes SHAP (SHapley Additive exPlanations) values for individual
patient predictions, identifying which features contribute most to the
predicted sepsis risk.

This module loads the existing XGBoost model (or accepts one at runtime)
and does NOT modify any model files.

Requires
--------
``pip install shap``
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger("shap_explainer")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_shap_values(
    patient_features: pd.DataFrame,
    model: Optional[Any] = None,
    model_path: str = "models/xgboost_model.pkl",
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute SHAP feature importance for a single patient prediction.

    Parameters
    ----------
    patient_features : pd.DataFrame
        A single-row DataFrame with the engineered ML features.
    model : object, optional
        A fitted XGBoost model.  If ``None``, loads from ``model_path``.
    model_path : str
        Fallback path to the saved XGBoost pickle.
    feature_names : list[str], optional
        Override column names for display purposes.

    Returns
    -------
    dict
        Keys:

        * ``shap_values`` — array of SHAP values per feature.
        * ``feature_names`` — corresponding feature names.
        * ``base_value`` — expected (average) model output.
        * ``top_features`` — list of dicts with ``feature``, ``shap_value``,
          ``direction`` (sorted by absolute SHAP importance).
    """
    import shap

    # Load model if not provided
    if model is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"XGBoost model not found at {model_path}")
        with open(model_path, "rb") as f:
            model = pickle.load(f)

    # Build SHAP explainer
    explainer = shap.TreeExplainer(model)

    # Compute SHAP values
    shap_result = explainer.shap_values(patient_features)

    # shap_values may be a list (multi-class) — take positive class
    if isinstance(shap_result, list):
        values = shap_result[1] if len(shap_result) > 1 else shap_result[0]
    else:
        values = shap_result

    # Flatten if single sample
    if values.ndim == 2:
        values = values[0]

    # Feature names
    if feature_names is None:
        feature_names = list(patient_features.columns)

    # Base value
    base_value = float(
        explainer.expected_value[1]
        if isinstance(explainer.expected_value, (list, np.ndarray))
        and len(explainer.expected_value) > 1
        else explainer.expected_value
    )

    # Build sorted top-features list
    top_features = _rank_features(feature_names, values)

    logger.info(
        "SHAP analysis complete — %d features, top driver: %s (%.4f)",
        len(feature_names),
        top_features[0]["feature"] if top_features else "N/A",
        top_features[0]["shap_value"] if top_features else 0.0,
    )

    return {
        "shap_values": values.tolist(),
        "feature_names": feature_names,
        "base_value": base_value,
        "top_features": top_features,
    }


def get_top_risk_drivers(
    shap_result: Dict[str, Any],
    n: int = 5,
) -> List[Dict[str, Any]]:
    """Return the top-N risk-driving features from a SHAP result.

    Parameters
    ----------
    shap_result : dict
        Output of :func:`compute_shap_values`.
    n : int
        Number of top features to return.

    Returns
    -------
    list[dict]
        Each entry has ``feature``, ``shap_value``, ``direction``.
    """
    return shap_result.get("top_features", [])[:n]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rank_features(
    feature_names: List[str],
    shap_values: np.ndarray,
) -> List[Dict[str, Any]]:
    """Rank features by absolute SHAP value."""
    ranked = []
    for name, val in zip(feature_names, shap_values):
        ranked.append({
            "feature": name,
            "shap_value": round(float(val), 4),
            "direction": "↑ increases risk" if val > 0 else "↓ decreases risk",
        })
    ranked.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    return ranked
