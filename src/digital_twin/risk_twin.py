"""
risk_twin.py — Digital twin risk simulation for treatment interventions.

Simulates how different clinical interventions (fluid resuscitation,
early antibiotics) would alter the patient's physiological trajectory
by perturbing the underlying vital-sign features and re-running the
risk forecasting model.

This module reuses:
  • src/forecasting/risk_forecast.py  — forward risk prediction
  • The existing LSTM model (read-only)

No models are modified or retrained.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger("risk_twin")


# ---------------------------------------------------------------------------
# Intervention definitions
# ---------------------------------------------------------------------------

# Each intervention specifies which vitals to modify and how.
# Modifications are applied as deltas per simulated hour.
# "factor" = multiplicative, "delta" = additive per hour.

INTERVENTION_PROFILES: Dict[str, Dict[str, Any]] = {
    "baseline": {
        "label": "No Intervention (Baseline)",
        "description": "Natural disease progression without treatment.",
        "modifications": {},
    },
    "fluid_resuscitation": {
        "label": "Fluid Resuscitation (30 mL/kg crystalloid)",
        "description": (
            "IV crystalloid bolus targeting MAP restoration and lactate clearance. "
            "Expected effects: MAP ↑, HR ↓ (improved preload), Lactate ↓ (improved perfusion)."
        ),
        "modifications": {
            "MAP": {"delta": 2.5, "min": 55, "max": 100},
            "HR": {"delta": -1.5, "min": 50, "max": 180},
            "Lactate": {"factor": 0.92, "min": 0.5, "max": 20},
            "SBP": {"delta": 2.0, "min": 70, "max": 180},
            "O2Sat": {"delta": 0.3, "min": 80, "max": 100},
        },
    },
    "early_antibiotics": {
        "label": "Early Broad-Spectrum Antibiotics",
        "description": (
            "Empiric antibiotic therapy within 1 hour of sepsis recognition. "
            "Expected effects: WBC normalisation, Temp normalisation, "
            "gradual Lactate reduction."
        ),
        "modifications": {
            "Temp": {"delta": -0.15, "min": 36.0, "max": 42.0},
            "WBC": {"factor": 0.96, "min": 3.0, "max": 30.0},
            "Lactate": {"factor": 0.95, "min": 0.5, "max": 20},
            "HR": {"delta": -0.8, "min": 50, "max": 180},
            "Resp": {"delta": -0.3, "min": 10, "max": 40},
        },
    },
}


# ---------------------------------------------------------------------------
# Internal: apply intervention perturbations to a DataFrame
# ---------------------------------------------------------------------------

def _apply_intervention(
    df: pd.DataFrame,
    intervention_key: str,
    n_hours: int,
) -> pd.DataFrame:
    """Create a perturbed copy of the patient DataFrame.

    Starting from the last available row, simulate ``n_hours`` additional
    rows with the intervention's physiological modifications applied
    cumulatively.

    Parameters
    ----------
    df : pd.DataFrame
        Original patient time-series data.
    intervention_key : str
        Key into INTERVENTION_PROFILES.
    n_hours : int
        Number of future hours to simulate.

    Returns
    -------
    pd.DataFrame
        Original rows + ``n_hours`` simulated rows.
    """
    profile = INTERVENTION_PROFILES.get(intervention_key, INTERVENTION_PROFILES["baseline"])
    mods = profile["modifications"]

    if not mods:
        # Baseline — return original data with small noise to model natural drift
        return _apply_natural_drift(df, n_hours)

    # Start from the last row's values
    last_row = df.iloc[-1].copy()
    new_rows = []

    for h in range(1, n_hours + 1):
        row = last_row.copy()

        for col, params in mods.items():
            if col not in df.columns:
                continue

            current_val = float(row.get(col, np.nan))
            if np.isnan(current_val):
                continue

            # Apply modification
            if "delta" in params:
                current_val += params["delta"]
            if "factor" in params:
                current_val *= params["factor"]

            # Clamp to physiological bounds
            current_val = max(params.get("min", 0), min(params.get("max", 999), current_val))

            row[col] = round(current_val, 2)

        # Update last_row for cumulative effect
        last_row = row.copy()
        new_rows.append(row)

    simulated = pd.DataFrame(new_rows)
    result = pd.concat([df, simulated], ignore_index=True)
    return result


def _apply_natural_drift(df: pd.DataFrame, n_hours: int) -> pd.DataFrame:
    """Simulate natural disease progression (baseline scenario)."""
    last_row = df.iloc[-1].copy()
    rng = np.random.RandomState(42)
    new_rows = []

    drift_cols = ["HR", "MAP", "Temp", "Resp", "Lactate", "O2Sat", "WBC"]

    for h in range(1, n_hours + 1):
        row = last_row.copy()
        for col in drift_cols:
            if col in df.columns:
                val = float(row.get(col, np.nan))
                if not np.isnan(val):
                    # Small random drift — disease tends to worsen slightly
                    noise = rng.normal(0, 0.005) * val
                    if col in ("HR", "Lactate", "Resp", "WBC"):
                        val += abs(noise) * 0.3  # slight worsening bias
                    elif col in ("MAP", "O2Sat"):
                        val -= abs(noise) * 0.3  # slight decline
                    else:
                        val += noise
                    row[col] = round(val, 2)
        last_row = row.copy()
        new_rows.append(row)

    simulated = pd.DataFrame(new_rows)
    return pd.concat([df, simulated], ignore_index=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def simulate_interventions(
    df: pd.DataFrame,
    interventions: Optional[List[str]] = None,
    horizon_hours: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Simulate treatment interventions and compare risk trajectories.

    Parameters
    ----------
    df : pd.DataFrame
        Patient time-series data.
    interventions : list[str], optional
        Intervention keys to simulate.
        Default: ["baseline", "fluid_resuscitation", "early_antibiotics"].
    horizon_hours : list[int], optional
        Future time points for risk forecasting (default: [1, 3, 6, 12]).

    Returns
    -------
    dict
        Keys are intervention names, values are forecast dicts from
        ``forecast_future_risk``.  Also includes ``summary`` with
        a comparison of endpoints.
    """
    if interventions is None:
        interventions = ["baseline", "fluid_resuscitation", "early_antibiotics"]
    if horizon_hours is None:
        horizon_hours = [1, 3, 6, 12]

    max_sim_hours = max(horizon_hours) + 2  # extra buffer

    results: Dict[str, Any] = {}

    for intervention_key in interventions:
        if intervention_key not in INTERVENTION_PROFILES:
            logger.warning("Unknown intervention: %s — skipping.", intervention_key)
            continue

        try:
            # Create perturbed patient timeline
            sim_df = _apply_intervention(df, intervention_key, max_sim_hours)

            # Forecast risk on the perturbed data
            from src.forecasting.risk_forecast import forecast_future_risk
            forecast = forecast_future_risk(sim_df, horizon_hours=horizon_hours)

            results[intervention_key] = {
                "label": INTERVENTION_PROFILES[intervention_key]["label"],
                "description": INTERVENTION_PROFILES[intervention_key]["description"],
                "forecast": forecast,
            }

            logger.info(
                "Digital twin [%s] — current=%.4f, endpoint=%.4f",
                intervention_key,
                forecast.get("current_risk", 0),
                forecast["forecast"][-1]["risk"] if forecast.get("forecast") else 0,
            )

        except Exception as exc:
            logger.error("Simulation failed for %s: %s", intervention_key, exc)
            results[intervention_key] = {
                "label": INTERVENTION_PROFILES[intervention_key]["label"],
                "description": INTERVENTION_PROFILES[intervention_key]["description"],
                "forecast": {"current_risk": 0, "forecast": [], "trend": "UNKNOWN"},
                "error": str(exc),
            }

    # Build summary comparison
    results["summary"] = _build_summary(results, horizon_hours)

    return results


def _build_summary(
    results: Dict[str, Any],
    horizon_hours: List[int],
) -> Dict[str, Any]:
    """Build a comparison summary across interventions."""
    summary: Dict[str, Any] = {"horizon_hours": horizon_hours, "comparisons": {}}

    baselines_keys = [k for k in results if k != "summary"]
    for key in baselines_keys:
        entry = results[key]
        forecast = entry.get("forecast", {})
        endpoint_risks = forecast.get("forecast", [])
        endpoint = endpoint_risks[-1]["risk"] if endpoint_risks else forecast.get("current_risk", 0)

        summary["comparisons"][key] = {
            "label": entry.get("label", key),
            "current_risk": forecast.get("current_risk", 0),
            "endpoint_risk": endpoint,
            "trend": forecast.get("trend", "UNKNOWN"),
        }

    # Identify best intervention
    comparisons = summary["comparisons"]
    if comparisons:
        best = min(comparisons.items(), key=lambda x: x[1].get("endpoint_risk", 1.0))
        summary["recommended_intervention"] = best[0]
        summary["recommended_label"] = best[1].get("label", best[0])

    return summary


# ---------------------------------------------------------------------------
# UI helper
# ---------------------------------------------------------------------------


def get_digital_twin_results(df: pd.DataFrame, horizon_hours: int = 12):
    """
    Run digital twin simulations for predefined interventions.
    """

    from src.forecasting.risk_forecast import forecast_future_risk

    results = {}

    for key in ["baseline", "fluid_resuscitation", "early_antibiotics"]:

        simulated_df = _apply_intervention(df, key, horizon_hours)

        forecast = forecast_future_risk(simulated_df, [1,3,6,12])

        results[key] = forecast

    return results

