"""
uncertainty_estimator.py — Extended uncertainty estimation.

Builds on the existing ``src.utils.uncertainty`` module to provide
additional uncertainty metrics beyond simple model disagreement.

New metrics
-----------
* **Prediction entropy** — Shannon entropy of the ensemble prediction.
* **Confidence interval** — pseudo-interval derived from model spread.
* **Clinical reliability** — composite reliability indicator.
"""

from __future__ import annotations

import math
from typing import Dict

from src.utils.logger import setup_logger

logger = setup_logger("uncertainty_estimator")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_prediction_entropy(ensemble_score: float) -> float:
    """Compute Shannon binary entropy of the ensemble prediction.

    Parameters
    ----------
    ensemble_score : float
        Combined risk probability in ``[0, 1]``.

    Returns
    -------
    float
        Entropy value in ``[0, 1]`` (normalised).
        ``0.0`` = maximally certain, ``1.0`` = maximally uncertain.
    """
    p = max(1e-10, min(1.0 - 1e-10, ensemble_score))
    entropy = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
    return round(entropy, 4)


def compute_confidence_interval(
    ml_score: float,
    lstm_score: float,
) -> Dict[str, float]:
    """Compute a pseudo confidence interval from model disagreement.

    Parameters
    ----------
    ml_score : float
        ML baseline risk score.
    lstm_score : float
        LSTM risk score.

    Returns
    -------
    dict
        Keys: ``lower``, ``upper``, ``spread``.
    """
    lower = min(ml_score, lstm_score)
    upper = max(ml_score, lstm_score)
    spread = upper - lower

    return {
        "lower": round(lower, 4),
        "upper": round(upper, 4),
        "spread": round(spread, 4),
    }


def compute_clinical_reliability(
    ml_score: float,
    lstm_score: float,
    ensemble_score: float,
) -> Dict[str, object]:
    """Compute a composite clinical reliability indicator.

    Combines model disagreement, prediction entropy, and confidence
    spread into a single reliability assessment.

    Parameters
    ----------
    ml_score : float
        ML baseline risk score.
    lstm_score : float
        LSTM risk score.
    ensemble_score : float
        Weighted ensemble score.

    Returns
    -------
    dict
        Keys: ``disagreement``, ``entropy``, ``confidence_interval``,
        ``reliability_score``, ``reliability_label``.
    """
    disagreement = abs(ml_score - lstm_score)
    entropy = compute_prediction_entropy(ensemble_score)
    ci = compute_confidence_interval(ml_score, lstm_score)

    # Reliability = inverse of average uncertainty signals
    # Normalised to [0, 1] where 1 = maximally reliable
    unreliability = (disagreement + entropy + ci["spread"]) / 3.0
    reliability = max(0.0, min(1.0, 1.0 - unreliability))

    if reliability >= 0.8:
        label = "HIGH"
    elif reliability >= 0.5:
        label = "MODERATE"
    else:
        label = "LOW"

    logger.debug(
        "Clinical reliability — disagreement=%.4f, entropy=%.4f, "
        "spread=%.4f → reliability=%.4f (%s)",
        disagreement, entropy, ci["spread"], reliability, label,
    )

    return {
        "disagreement": round(disagreement, 4),
        "entropy": round(entropy, 4),
        "confidence_interval": ci,
        "reliability_score": round(reliability, 4),
        "reliability_label": label,
    }
