"""
uncertainty.py — Prediction uncertainty estimation and model agreement scoring.

Provides lightweight calibration utilities that quantify how much the ML
baseline and LSTM models agree with each other.  Large disagreements
indicate higher uncertainty and should lower clinical confidence in the
combined risk score.

Functions
---------
* :func:`compute_model_agreement` — classify agreement level (HIGH / MEDIUM / LOW).
* :func:`compute_uncertainty`     — return a scalar uncertainty score in [0, 1].
"""

from __future__ import annotations

from src.utils.logger import setup_logger

logger = setup_logger("uncertainty")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_model_agreement(ml_score: float, lstm_score: float) -> str:
    """Classify the level of agreement between the ML and LSTM risk scores.

    Agreement is based on the absolute difference between the two model
    outputs.  Scores that are close together indicate the models have
    converged on a similar assessment, increasing clinical reliability.

    Parameters
    ----------
    ml_score : float
        Risk probability from the baseline ML model, in ``[0, 1]``.
    lstm_score : float
        Risk probability from the LSTM model, in ``[0, 1]``.

    Returns
    -------
    str
        One of:

        * ``"HIGH"``   — absolute difference < 0.05 (models strongly agree)
        * ``"MEDIUM"`` — absolute difference < 0.15 (moderate agreement)
        * ``"LOW"``    — absolute difference ≥ 0.15 (significant disagreement)

    Examples
    --------
    >>> compute_model_agreement(0.83, 0.80)
    'HIGH'
    >>> compute_model_agreement(0.83, 0.72)
    'MEDIUM'
    >>> compute_model_agreement(0.83, 0.55)
    'LOW'
    """
    if not (0.0 <= ml_score <= 1.0 and 0.0 <= lstm_score <= 1.0):
        logger.warning(
            "Scores out of [0, 1] range — ml=%.4f, lstm=%.4f. "
            "Agreement computed on raw values.",
            ml_score, lstm_score,
        )

    diff = abs(ml_score - lstm_score)

    if diff < 0.05:
        agreement = "HIGH"
    elif diff < 0.15:
        agreement = "MEDIUM"
    else:
        agreement = "LOW"

    logger.debug(
        "Model agreement — diff=%.4f → %s", diff, agreement
    )
    return agreement


def compute_uncertainty(ml_score: float, lstm_score: float) -> float:
    """Estimate prediction uncertainty from the gap between model scores.

    Uses the absolute difference between the two risk probabilities as a
    simple variance proxy.  When the models disagree, uncertainty is high;
    when they agree, uncertainty is low.

    Parameters
    ----------
    ml_score : float
        Risk probability from the baseline ML model, in ``[0, 1]``.
    lstm_score : float
        Risk probability from the LSTM model, in ``[0, 1]``.

    Returns
    -------
    float
        Uncertainty score in ``[0, 1]``.  A value of ``0.0`` means the two
        models are in perfect agreement; ``1.0`` means they are maximally
        opposed (e.g., one predicts 0.0 and the other 1.0).

    Examples
    --------
    >>> compute_uncertainty(0.83, 0.79)
    0.04
    >>> compute_uncertainty(0.9, 0.4)
    0.5
    """
    uncertainty = abs(ml_score - lstm_score)
    # Clamp to [0, 1] as a safety guard for out-of-range inputs
    uncertainty = max(0.0, min(1.0, uncertainty))

    logger.debug(
        "Uncertainty — ml=%.4f, lstm=%.4f → uncertainty=%.4f",
        ml_score, lstm_score, uncertainty,
    )
    return round(uncertainty, 4)
