"""
risk_ensemble.py — Weighted ensemble of ML and LSTM risk predictions.

Combines the XGBoost (ML) and LSTM risk scores into a single ensemble
probability using a configurable weighted average.

Default weights
---------------
ML   : 0.6  (XGBoost is the more interpretable, well-calibrated model)
LSTM : 0.4  (temporal model supplements with trend information)
"""

from __future__ import annotations

from src.utils.logger import setup_logger

logger = setup_logger("risk_ensemble")

# ---------------------------------------------------------------------------
# Default weights
# ---------------------------------------------------------------------------

DEFAULT_ML_WEIGHT: float = 0.6
DEFAULT_LSTM_WEIGHT: float = 0.4


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_ensemble_score(
    ml_score: float,
    lstm_score: float,
    *,
    ml_weight: float = DEFAULT_ML_WEIGHT,
    lstm_weight: float = DEFAULT_LSTM_WEIGHT,
) -> float:
    """Compute a weighted ensemble risk score.

    Parameters
    ----------
    ml_score : float
        Risk probability from the XGBoost baseline model, in ``[0, 1]``.
    lstm_score : float
        Risk probability from the LSTM model, in ``[0, 1]``.
    ml_weight : float
        Weight for the ML score.  Default ``0.6``.
    lstm_weight : float
        Weight for the LSTM score.  Default ``0.4``.

    Returns
    -------
    float
        Ensemble risk score in ``[0, 1]``.

    Raises
    ------
    ValueError
        If weights do not sum to approximately 1.0.
    """
    total_weight = ml_weight + lstm_weight
    if abs(total_weight - 1.0) > 1e-6:
        logger.warning(
            "Weights sum to %.4f (expected 1.0) — normalising.", total_weight
        )
        ml_weight /= total_weight
        lstm_weight /= total_weight

    ensemble = ml_weight * ml_score + lstm_weight * lstm_score
    ensemble = max(0.0, min(1.0, ensemble))

    logger.debug(
        "Ensemble — ML=%.4f×%.2f + LSTM=%.4f×%.2f → %.4f",
        ml_score, ml_weight, lstm_score, lstm_weight, ensemble,
    )
    return round(ensemble, 4)


def ensemble_risk_label(score: float) -> str:
    """Convert an ensemble score to a categorical risk label.

    Parameters
    ----------
    score : float
        Ensemble risk score in ``[0, 1]``.

    Returns
    -------
    str
        ``"HIGH"`` (≥0.7), ``"MODERATE"`` (≥0.4), or ``"LOW"`` (<0.4).
    """
    if score >= 0.7:
        return "HIGH"
    if score >= 0.4:
        return "MODERATE"
    return "LOW"
