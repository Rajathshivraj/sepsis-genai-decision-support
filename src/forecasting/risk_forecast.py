"""
risk_forecast.py — Future sepsis risk forecasting using the existing LSTM model.

Simulates forward predictions by iteratively feeding the model's own
output back as the next time-step input.  No retraining is performed;
the saved LSTM checkpoint is loaded read-only.

This module is standalone and does NOT modify any model files.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from configs.config import cfg
from src.utils.logger import setup_logger

logger = setup_logger("risk_forecast")

# LSTM features — must match training order exactly
_LSTM_FEATURE_COLUMNS: List[str] = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "Lactate", "Creatinine", "WBC", "Platelets",
]

# Default forecast horizons (hours into the future)
DEFAULT_HORIZONS: List[int] = [1, 3, 6, 12]

# Path to the saved LSTM state dict
_DEFAULT_MODEL_PATH = "models/lstm_model.pt"

# ---------------------------------------------------------------------------
# Internal: LSTM loader (mirrors src/models/lstm_model.SepsisLSTM)
# ---------------------------------------------------------------------------

_cached_model = None
_cached_input_size = None


def _load_lstm_model(
    model_path: str = _DEFAULT_MODEL_PATH,
    input_size: int = len(_LSTM_FEATURE_COLUMNS),
) -> Any:
    """Load the LSTM checkpoint (cached after first call)."""
    global _cached_model, _cached_input_size

    if _cached_model is not None and _cached_input_size == input_size:
        return _cached_model

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"LSTM model not found at {model_path}")

    # Import the model class from the existing codebase
    from src.models.lstm_model import SepsisLSTM

    model = SepsisLSTM(input_size=input_size)
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()

    _cached_model = model
    _cached_input_size = input_size
    logger.info("LSTM model loaded from %s (input_size=%d)", model_path, input_size)
    return model


# ---------------------------------------------------------------------------
# Internal: sequence preparation
# ---------------------------------------------------------------------------

def _df_to_sequence(
    df: pd.DataFrame,
    seq_len: int,
    feature_cols: List[str],
) -> np.ndarray:
    """Convert the tail of a DataFrame into a (1, seq_len, n_features) array.

    If the DataFrame has fewer rows than seq_len, the first row is repeated
    to pad the beginning — matching the training-time padding strategy.
    """
    available = [c for c in feature_cols if c in df.columns]
    values = df[available].values.astype(np.float32)

    # Fill NaNs with column medians (or 0)
    for col_idx in range(values.shape[1]):
        col = values[:, col_idx]
        mask = np.isnan(col)
        if mask.any():
            med = np.nanmedian(col) if not np.all(mask) else 0.0
            col[mask] = med

    # Pad if shorter than seq_len
    if len(values) < seq_len:
        pad_len = seq_len - len(values)
        pad = np.repeat(values[:1], pad_len, axis=0)
        values = np.vstack([pad, values])

    # Take the last seq_len rows
    seq = values[-seq_len:]

    # Ensure exactly n_features columns (pad missing cols with 0)
    if len(available) < len(feature_cols):
        full = np.zeros((seq_len, len(feature_cols)), dtype=np.float32)
        for i, col_name in enumerate(feature_cols):
            if col_name in available:
                src_idx = available.index(col_name)
                full[:, i] = seq[:, src_idx]
        seq = full

    return seq[np.newaxis, :, :]  # (1, seq_len, n_features)


def _predict_risk(model, sequence: np.ndarray) -> float:
    """Run a single forward pass and return sigmoid probability."""
    with torch.no_grad():
        x = torch.tensor(sequence, dtype=torch.float32)
        logits = model(x)  # (1, 1)
        prob = torch.sigmoid(logits).item()
    return round(prob, 4)


# ---------------------------------------------------------------------------
# Internal: autoregressive forward simulation
# ---------------------------------------------------------------------------

def _simulate_forward(
    model,
    initial_sequence: np.ndarray,
    horizon_hours: List[int],
    drift_rate: float = 0.005,
) -> List[Dict[str, Any]]:
    """Autoregressively simulate future risk scores.

    At each simulated hour:
      1. Predict the current risk.
      2. Apply a small physiological drift to the last time-step features.
      3. Shift the sequence window forward by one step.

    Parameters
    ----------
    model : SepsisLSTM
        The loaded LSTM model.
    initial_sequence : np.ndarray
        Shape (1, seq_len, n_features).
    horizon_hours : list[int]
        Hours into the future to report.
    drift_rate : float
        Per-step random drift magnitude applied to features.

    Returns
    -------
    list[dict]
        Each entry: {"hour": int, "risk": float}.
    """
    seq = initial_sequence.copy()  # (1, seq_len, n_features)
    seq_len = seq.shape[1]
    n_features = seq.shape[2]
    max_h = max(horizon_hours)

    hourly_risks: List[float] = []

    rng = np.random.RandomState(42)

    for h in range(max_h + 1):
        risk = _predict_risk(model, seq)
        hourly_risks.append(risk)

        if h < max_h:
            # Drift the last row slightly to simulate physiological evolution
            last_row = seq[0, -1, :].copy()
            noise = rng.normal(0, drift_rate, size=n_features).astype(np.float32)
            new_row = last_row + noise * last_row  # proportional noise
            new_row = np.clip(new_row, 0, None)  # vitals can't be negative

            # Shift window: drop first row, append new row
            seq = np.concatenate(
                [seq[:, 1:, :], new_row.reshape(1, 1, n_features)],
                axis=1,
            )

    forecast = []
    for h in horizon_hours:
        if h < len(hourly_risks):
            forecast.append({"hour": h, "risk": hourly_risks[h]})
        else:
            forecast.append({"hour": h, "risk": hourly_risks[-1]})

    return forecast


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def forecast_future_risk(
    df: pd.DataFrame,
    horizon_hours: Optional[List[int]] = None,
    model_path: str = _DEFAULT_MODEL_PATH,
) -> Dict[str, Any]:
    """Predict future sepsis risk using the existing LSTM model.

    Parameters
    ----------
    df : pd.DataFrame
        Patient time-series DataFrame with hourly rows.
    horizon_hours : list[int], optional
        Future time points to forecast (default: [1, 3, 6, 12]).
    model_path : str
        Path to the saved LSTM state dict.

    Returns
    -------
    dict
        Keys:
        * ``current_risk`` — risk at t=0.
        * ``forecast`` — list of {"hour": int, "risk": float}.
        * ``trend`` — "INCREASING", "DECREASING", or "STABLE".

    If the LSTM model is unavailable, falls back to a heuristic forecast
    based on the risk trajectory module.
    """
    if horizon_hours is None:
        horizon_hours = list(DEFAULT_HORIZONS)

    # Try LSTM-based forecasting
    try:
        model = _load_lstm_model(model_path)
        seq_len = cfg.SEQUENCE_LENGTH
        seq = _df_to_sequence(df, seq_len, _LSTM_FEATURE_COLUMNS)
        current_risk = _predict_risk(model, seq)
        forecast = _simulate_forward(model, seq, horizon_hours)

        # Determine trend
        risks = [current_risk] + [f["risk"] for f in forecast]
        if len(risks) >= 2:
            slope = risks[-1] - risks[0]
            trend = "INCREASING" if slope > 0.05 else ("DECREASING" if slope < -0.05 else "STABLE")
        else:
            trend = "STABLE"

        logger.info(
            "LSTM forecast — current=%.4f, %d-hour=%.4f, trend=%s",
            current_risk,
            horizon_hours[-1] if horizon_hours else 0,
            forecast[-1]["risk"] if forecast else current_risk,
            trend,
        )

        return {
            "current_risk": current_risk,
            "forecast": forecast,
            "trend": trend,
        }

    except Exception as exc:
        logger.warning("LSTM forecast unavailable (%s) — using heuristic fallback.", exc)
        return _heuristic_forecast(df, horizon_hours)


def _heuristic_forecast(
    df: pd.DataFrame,
    horizon_hours: List[int],
) -> Dict[str, Any]:
    """Rule-based fallback when the LSTM model is unavailable."""
    # Compute a simple current risk from vital trends
    score = 0.0
    hr = df["HR"].dropna()
    if len(hr) >= 2:
        trend = (float(hr.iloc[-1]) - float(hr.iloc[0])) / max(len(hr), 1)
        if trend > 2:
            score += 0.15
    map_vals = df["MAP"].dropna() if "MAP" in df.columns else pd.Series(dtype=float)
    if len(map_vals) >= 2:
        trend = (float(map_vals.iloc[-1]) - float(map_vals.iloc[0])) / max(len(map_vals), 1)
        if trend < -1:
            score += 0.15
    lac = df["Lactate"].dropna() if "Lactate" in df.columns else pd.Series(dtype=float)
    if len(lac) and float(lac.iloc[-1]) > 2.0:
        score += 0.20

    current_risk = min(1.0, round(score + 0.15, 4))

    # Simple linear extrapolation with slight increase
    forecast = []
    for h in horizon_hours:
        projected = min(1.0, round(current_risk + 0.02 * h, 4))
        forecast.append({"hour": h, "risk": projected})

    risks = [current_risk] + [f["risk"] for f in forecast]
    slope = risks[-1] - risks[0]
    trend = "INCREASING" if slope > 0.05 else ("DECREASING" if slope < -0.05 else "STABLE")

    return {
        "current_risk": current_risk,
        "forecast": forecast,
        "trend": trend,
    }


# ---------------------------------------------------------------------------
# UI helper
# ---------------------------------------------------------------------------

def get_forecast_chart_data(
    forecast_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Format forecast result for Plotly chart rendering.

    Returns
    -------
    dict
        Keys: ``hours``, ``risks``, ``current_risk``, ``trend``.
    """
    hours = [0] + [f["hour"] for f in forecast_result.get("forecast", [])]
    risks = [forecast_result.get("current_risk", 0)] + [
        f["risk"] for f in forecast_result.get("forecast", [])
    ]
    return {
        "hours": hours,
        "risks": risks,
        "current_risk": forecast_result.get("current_risk", 0),
        "trend": forecast_result.get("trend", "STABLE"),
    }
