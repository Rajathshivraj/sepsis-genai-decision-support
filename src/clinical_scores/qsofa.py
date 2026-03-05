"""
qsofa.py — Quick SOFA (qSOFA) clinical scoring.

The qSOFA score is a bedside screening tool for sepsis that uses three
simple clinical criteria (no lab tests required):

1. Altered mentation (GCS ≤ 13)
2. Systolic blood pressure ≤ 100 mmHg
3. Respiratory rate ≥ 22 breaths/min

Score range: 0–3.  A score ≥ 2 suggests increased risk of poor outcome
and warrants further investigation for organ dysfunction.

Reference
---------
Singer M, et al.  *The Third International Consensus Definitions for
Sepsis and Septic Shock (Sepsis-3).*  JAMA. 2016;315(8):801-810.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.utils.logger import setup_logger

logger = setup_logger("qsofa")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_qsofa(
    sbp: Optional[float] = None,
    resp: Optional[float] = None,
    gcs: Optional[float] = None,
    *,
    altered_mentation: Optional[bool] = None,
) -> Dict[str, Any]:
    """Compute the qSOFA score from clinical parameters.

    Parameters
    ----------
    sbp : float, optional
        Systolic blood pressure (mmHg).
    resp : float, optional
        Respiratory rate (breaths/min).
    gcs : float, optional
        Glasgow Coma Scale score (3–15).  If not provided,
        ``altered_mentation`` can be used as a boolean substitute.
    altered_mentation : bool, optional
        True if the patient has altered mentation.  Used only if
        ``gcs`` is not provided.

    Returns
    -------
    dict
        Keys:

        * ``score`` — integer 0–3.
        * ``criteria_met`` — list of strings describing which criteria
          are positive.
        * ``risk_level`` — ``"LOW"`` (0–1) or ``"HIGH"`` (≥2).
        * ``interpretation`` — clinical interpretation text.
    """
    score = 0
    criteria_met = []

    # Criterion 1: Altered mentation
    if gcs is not None and gcs <= 13:
        score += 1
        criteria_met.append(f"Altered mentation (GCS = {gcs:.0f})")
    elif altered_mentation is True:
        score += 1
        criteria_met.append("Altered mentation (reported)")

    # Criterion 2: Systolic BP ≤ 100 mmHg
    if sbp is not None and sbp <= 100:
        score += 1
        criteria_met.append(f"Systolic BP ≤ 100 mmHg (SBP = {sbp:.0f})")

    # Criterion 3: Respiratory rate ≥ 22 /min
    if resp is not None and resp >= 22:
        score += 1
        criteria_met.append(f"Respiratory rate ≥ 22/min (Resp = {resp:.0f})")

    risk_level = "HIGH" if score >= 2 else "LOW"

    if score >= 2:
        interpretation = (
            f"qSOFA score {score}/3 — positive screen. "
            "This patient meets criteria for suspected sepsis and should be "
            "evaluated for organ dysfunction (consider full SOFA assessment)."
        )
    elif score == 1:
        interpretation = (
            f"qSOFA score {score}/3 — borderline. "
            "One criterion met. Monitor closely for clinical deterioration."
        )
    else:
        interpretation = (
            f"qSOFA score {score}/3 — negative screen. "
            "No qSOFA criteria currently met."
        )

    logger.debug("qSOFA — score=%d, risk=%s", score, risk_level)

    return {
        "score": score,
        "criteria_met": criteria_met,
        "risk_level": risk_level,
        "interpretation": interpretation,
    }
