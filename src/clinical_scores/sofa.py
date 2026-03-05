"""
sofa.py — Sequential Organ Failure Assessment (SOFA) scoring.

Computes the SOFA score used to assess the extent of organ dysfunction
in ICU patients.  The full SOFA evaluates six organ systems:

1. Respiration   — PaO2/FiO2 ratio
2. Coagulation   — Platelet count
3. Liver         — Bilirubin level
4. Cardiovascular — MAP and vasopressor use
5. CNS           — Glasgow Coma Scale
6. Renal         — Creatinine / urine output

Each subsystem scores 0–4, giving a total range of 0–24.  An acute
increase of ≥ 2 points in the context of infection is the Sepsis-3
definition of sepsis.

Reference
---------
Vincent JL, et al.  *The SOFA (Sepsis-related Organ Failure Assessment)
score to describe organ dysfunction/failure.*  Intensive Care Med.
1996;22(7):707-710.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.utils.logger import setup_logger

logger = setup_logger("sofa")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_sofa(
    pao2_fio2: Optional[float] = None,
    platelets: Optional[float] = None,
    bilirubin: Optional[float] = None,
    map_val: Optional[float] = None,
    vasopressors: bool = False,
    gcs: Optional[float] = None,
    creatinine: Optional[float] = None,
    urine_output_24h: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute the SOFA score from available clinical parameters.

    All parameters are optional — subsystems with missing data score 0
    and are flagged as ``"not assessed"``.

    Parameters
    ----------
    pao2_fio2 : float, optional
        PaO2/FiO2 ratio (mmHg).
    platelets : float, optional
        Platelet count (×10³/µL).
    bilirubin : float, optional
        Total bilirubin (mg/dL).
    map_val : float, optional
        Mean arterial pressure (mmHg).
    vasopressors : bool
        Whether the patient is on vasopressor support.
    gcs : float, optional
        Glasgow Coma Scale (3–15).
    creatinine : float, optional
        Serum creatinine (mg/dL).
    urine_output_24h : float, optional
        24-hour urine output (mL).

    Returns
    -------
    dict
        Keys:

        * ``total_score`` — integer 0–24.
        * ``subsystems`` — dict mapping system name → sub-score (0–4).
        * ``interpretation`` — clinical interpretation text.
        * ``organ_dysfunction`` — list of systems with score ≥ 2.
    """
    subsystems: Dict[str, int] = {}
    details: Dict[str, str] = {}

    # 1. Respiration
    subsystems["respiration"], details["respiration"] = _score_respiration(pao2_fio2)

    # 2. Coagulation
    subsystems["coagulation"], details["coagulation"] = _score_coagulation(platelets)

    # 3. Liver
    subsystems["liver"], details["liver"] = _score_liver(bilirubin)

    # 4. Cardiovascular
    subsystems["cardiovascular"], details["cardiovascular"] = _score_cardiovascular(
        map_val, vasopressors
    )

    # 5. CNS
    subsystems["cns"], details["cns"] = _score_cns(gcs)

    # 6. Renal
    subsystems["renal"], details["renal"] = _score_renal(creatinine, urine_output_24h)

    total = sum(subsystems.values())

    organ_dysfunction = [
        name for name, score in subsystems.items() if score >= 2
    ]

    if total >= 10:
        interpretation = (
            f"SOFA score {total}/24 — severe organ dysfunction. "
            "High mortality risk.  Aggressive ICU management required."
        )
    elif total >= 6:
        interpretation = (
            f"SOFA score {total}/24 — significant organ dysfunction. "
            "Close monitoring and organ-support interventions recommended."
        )
    elif total >= 2:
        interpretation = (
            f"SOFA score {total}/24 — mild-to-moderate organ dysfunction. "
            "In the context of infection, this meets Sepsis-3 criteria."
        )
    else:
        interpretation = (
            f"SOFA score {total}/24 — minimal organ dysfunction."
        )

    logger.debug(
        "SOFA — total=%d, dysfunction in: %s",
        total,
        organ_dysfunction or "none",
    )

    return {
        "total_score": total,
        "subsystems": subsystems,
        "details": details,
        "interpretation": interpretation,
        "organ_dysfunction": organ_dysfunction,
    }


# ---------------------------------------------------------------------------
# Subsystem scoring helpers
# ---------------------------------------------------------------------------

def _score_respiration(pao2_fio2: Optional[float]) -> tuple:
    if pao2_fio2 is None:
        return 0, "Not assessed"
    if pao2_fio2 < 100:
        return 4, f"PaO2/FiO2 = {pao2_fio2:.0f} (< 100)"
    if pao2_fio2 < 200:
        return 3, f"PaO2/FiO2 = {pao2_fio2:.0f} (< 200)"
    if pao2_fio2 < 300:
        return 2, f"PaO2/FiO2 = {pao2_fio2:.0f} (< 300)"
    if pao2_fio2 < 400:
        return 1, f"PaO2/FiO2 = {pao2_fio2:.0f} (< 400)"
    return 0, f"PaO2/FiO2 = {pao2_fio2:.0f} (≥ 400)"


def _score_coagulation(platelets: Optional[float]) -> tuple:
    if platelets is None:
        return 0, "Not assessed"
    if platelets < 20:
        return 4, f"Platelets = {platelets:.0f} (< 20)"
    if platelets < 50:
        return 3, f"Platelets = {platelets:.0f} (< 50)"
    if platelets < 100:
        return 2, f"Platelets = {platelets:.0f} (< 100)"
    if platelets < 150:
        return 1, f"Platelets = {platelets:.0f} (< 150)"
    return 0, f"Platelets = {platelets:.0f} (≥ 150)"


def _score_liver(bilirubin: Optional[float]) -> tuple:
    if bilirubin is None:
        return 0, "Not assessed"
    if bilirubin >= 12.0:
        return 4, f"Bilirubin = {bilirubin:.1f} (≥ 12.0)"
    if bilirubin >= 6.0:
        return 3, f"Bilirubin = {bilirubin:.1f} (≥ 6.0)"
    if bilirubin >= 2.0:
        return 2, f"Bilirubin = {bilirubin:.1f} (≥ 2.0)"
    if bilirubin >= 1.2:
        return 1, f"Bilirubin = {bilirubin:.1f} (≥ 1.2)"
    return 0, f"Bilirubin = {bilirubin:.1f} (< 1.2)"


def _score_cardiovascular(
    map_val: Optional[float],
    vasopressors: bool,
) -> tuple:
    if vasopressors:
        return 3, f"On vasopressors (MAP = {map_val or 'N/A'})"
    if map_val is None:
        return 0, "Not assessed"
    if map_val < 70:
        return 1, f"MAP = {map_val:.0f} (< 70)"
    return 0, f"MAP = {map_val:.0f} (≥ 70)"


def _score_cns(gcs: Optional[float]) -> tuple:
    if gcs is None:
        return 0, "Not assessed"
    if gcs < 6:
        return 4, f"GCS = {gcs:.0f} (< 6)"
    if gcs < 10:
        return 3, f"GCS = {gcs:.0f} (6–9)"
    if gcs < 13:
        return 2, f"GCS = {gcs:.0f} (10–12)"
    if gcs < 15:
        return 1, f"GCS = {gcs:.0f} (13–14)"
    return 0, f"GCS = {gcs:.0f} (15)"


def _score_renal(
    creatinine: Optional[float],
    urine_output_24h: Optional[float],
) -> tuple:
    score_cr = 0
    detail = "Not assessed"

    if creatinine is not None:
        if creatinine >= 5.0:
            score_cr = 4
            detail = f"Creatinine = {creatinine:.1f} (≥ 5.0)"
        elif creatinine >= 3.5:
            score_cr = 3
            detail = f"Creatinine = {creatinine:.1f} (3.5–4.9)"
        elif creatinine >= 2.0:
            score_cr = 2
            detail = f"Creatinine = {creatinine:.1f} (2.0–3.4)"
        elif creatinine >= 1.2:
            score_cr = 1
            detail = f"Creatinine = {creatinine:.1f} (1.2–1.9)"
        else:
            detail = f"Creatinine = {creatinine:.1f} (< 1.2)"

    # Urine output can upgrade score
    score_uo = 0
    if urine_output_24h is not None:
        if urine_output_24h < 200:
            score_uo = 4
        elif urine_output_24h < 500:
            score_uo = 3

    return max(score_cr, score_uo), detail
