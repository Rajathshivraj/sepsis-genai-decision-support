"""
risk_trajectory.py — Risk trajectory modeling from time-series data.

Uses per-hour heuristic scoring (or LSTM predictions when available) to
build a timeline of risk scores across the patient's ICU stay.

This module is standalone and does NOT modify existing model files.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger("risk_trajectory")

# Clinical thresholds for heuristic scoring
_THRESHOLDS = {
    "HR": {"warn": 100, "crit": 120},
    "MAP": {"warn": 70, "crit": 60},
    "Lactate": {"warn": 2.0, "crit": 4.0},
    "Resp": {"warn": 20, "crit": 25},
    "Temp": {"warn": 38.0, "crit": 39.0},
    "WBC": {"high_warn": 12.0, "low_warn": 4.0},
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_risk_trajectory(
    df: pd.DataFrame,
    window_size: int = 1,
) -> List[Dict[str, object]]:
    """Compute a per-hour risk score timeline from patient data.

    Parameters
    ----------
    df : pd.DataFrame
        Patient time-series DataFrame with hourly rows.  Expected columns
        include a subset of: HR, MAP, Lactate, Resp, Temp, WBC.
    window_size : int
        Rolling window size for smoothing scores.  ``1`` = no smoothing.

    Returns
    -------
    list[dict]
        Each entry has keys: ``hour``, ``risk_score``, ``risk_label``,
        ``contributing_factors``.
    """
    trajectory: List[Dict[str, object]] = []

    for idx in range(len(df)):
        row = df.iloc[idx]
        score, factors = _score_single_hour(row)
        trajectory.append({
            "hour": idx + 1,
            "risk_score": round(score, 4),
            "risk_label": _label(score),
            "contributing_factors": factors,
        })

    # Optional rolling smoothing
    if window_size > 1 and len(trajectory) >= window_size:
        scores = [t["risk_score"] for t in trajectory]
        smoothed = pd.Series(scores).rolling(window_size, min_periods=1).mean()
        for i, s in enumerate(smoothed):
            trajectory[i]["risk_score"] = round(float(s), 4)
            trajectory[i]["risk_label"] = _label(float(s))

    logger.info(
        "Risk trajectory computed — %d hours, final score=%.4f",
        len(trajectory),
        trajectory[-1]["risk_score"] if trajectory else 0.0,
    )
    return trajectory


def compute_risk_trend(trajectory: List[Dict[str, object]]) -> Dict[str, object]:
    """Summarise the risk trajectory with trend statistics.

    Parameters
    ----------
    trajectory : list[dict]
        Output of :func:`compute_risk_trajectory`.

    Returns
    -------
    dict
        Keys: ``trend_direction``, ``slope``, ``max_score``, ``min_score``,
        ``mean_score``, ``current_score``.
    """
    if not trajectory:
        return {
            "trend_direction": "UNKNOWN",
            "slope": 0.0,
            "max_score": 0.0,
            "min_score": 0.0,
            "mean_score": 0.0,
            "current_score": 0.0,
        }

    scores = [t["risk_score"] for t in trajectory]
    current = scores[-1]

    # Simple linear regression slope
    if len(scores) >= 2:
        x = np.arange(len(scores), dtype=float)
        slope = float(np.polyfit(x, scores, 1)[0])
    else:
        slope = 0.0

    if slope > 0.01:
        direction = "INCREASING"
    elif slope < -0.01:
        direction = "DECREASING"
    else:
        direction = "STABLE"

    return {
        "trend_direction": direction,
        "slope": round(slope, 6),
        "max_score": round(float(max(scores)), 4),
        "min_score": round(float(min(scores)), 4),
        "mean_score": round(float(np.mean(scores)), 4),
        "current_score": round(current, 4),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _score_single_hour(row: pd.Series) -> tuple:
    """Score a single hourly observation using clinical thresholds."""
    score = 0.0
    factors = []

    # Heart Rate
    hr = row.get("HR", np.nan)
    if not np.isnan(hr):
        if hr > _THRESHOLDS["HR"]["crit"]:
            score += 0.20
            factors.append(f"HR critically elevated ({hr:.0f})")
        elif hr > _THRESHOLDS["HR"]["warn"]:
            score += 0.10
            factors.append(f"HR elevated ({hr:.0f})")

    # MAP
    map_val = row.get("MAP", np.nan)
    if not np.isnan(map_val):
        if map_val < _THRESHOLDS["MAP"]["crit"]:
            score += 0.25
            factors.append(f"MAP critically low ({map_val:.0f})")
        elif map_val < _THRESHOLDS["MAP"]["warn"]:
            score += 0.15
            factors.append(f"MAP low ({map_val:.0f})")

    # Lactate
    lac = row.get("Lactate", np.nan)
    if not np.isnan(lac):
        if lac > _THRESHOLDS["Lactate"]["crit"]:
            score += 0.25
            factors.append(f"Lactate critically elevated ({lac:.1f})")
        elif lac > _THRESHOLDS["Lactate"]["warn"]:
            score += 0.15
            factors.append(f"Lactate elevated ({lac:.1f})")

    # Respiratory rate
    resp = row.get("Resp", np.nan)
    if not np.isnan(resp):
        if resp > _THRESHOLDS["Resp"]["crit"]:
            score += 0.15
            factors.append(f"Resp rate high ({resp:.0f})")
        elif resp > _THRESHOLDS["Resp"]["warn"]:
            score += 0.08
            factors.append(f"Resp rate elevated ({resp:.0f})")

    # Temperature
    temp = row.get("Temp", np.nan)
    if not np.isnan(temp):
        if temp > _THRESHOLDS["Temp"]["crit"]:
            score += 0.15
            factors.append(f"Fever ({temp:.1f}°C)")
        elif temp > _THRESHOLDS["Temp"]["warn"]:
            score += 0.08
            factors.append(f"Low-grade fever ({temp:.1f}°C)")

    # WBC
    wbc = row.get("WBC", np.nan)
    if not np.isnan(wbc):
        if wbc > _THRESHOLDS["WBC"]["high_warn"]:
            score += 0.10
            factors.append(f"WBC elevated ({wbc:.1f})")
        elif wbc < _THRESHOLDS["WBC"]["low_warn"]:
            score += 0.10
            factors.append(f"WBC low ({wbc:.1f})")

    return min(1.0, score), factors


def _label(score: float) -> str:
    if score >= 0.7:
        return "HIGH"
    if score >= 0.4:
        return "MODERATE"
    return "LOW"
