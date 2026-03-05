"""
results_logger.py — Structured results logging for baseline and hybrid experiments.

Saves experiment outputs as timestamped JSON files into dedicated subdirectories:

* Baseline model metrics  →  ``results/baseline/<timestamp>.json``
* Hybrid pipeline output  →  ``results/hybrid/<timestamp>.json``

This module complements the existing :mod:`src.utils.results_manager` (which saves
to the flat ``results/`` root) by providing separated, purpose-specific directories
with a richer payload envelope (run metadata + results body).

Functions
---------
* :func:`save_baseline_results` — persist baseline ML metrics.
* :func:`save_hybrid_results`   — persist hybrid pipeline output.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from configs.config import PROJECT_ROOT
from src.utils.logger import setup_logger

logger = setup_logger("results_logger")

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

_RESULTS_ROOT = PROJECT_ROOT / "results"
_BASELINE_DIR = _RESULTS_ROOT / "baseline"
_HYBRID_DIR = _RESULTS_ROOT / "hybrid"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _iso_timestamp() -> str:
    """Return the current UTC time as a compact ISO-8601 string."""
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _save_json(directory: Path, data: Dict[str, Any], prefix: str) -> Path:
    """Serialise *data* to a timestamped JSON file under *directory*.

    Parameters
    ----------
    directory : Path
        Target directory (created automatically if it does not exist).
    data : dict
        Payload to serialise.
    prefix : str
        Filename prefix, e.g. ``"baseline"`` or ``"hybrid"``.

    Returns
    -------
    Path
        Absolute path of the written file.

    Raises
    ------
    OSError
        Propagated if the file cannot be written after logging the error.
    """
    directory.mkdir(parents=True, exist_ok=True)

    timestamp = _iso_timestamp()
    filename = f"{prefix}_{timestamp}.json"
    output_path = directory / filename

    try:
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=4, default=str)
        logger.info("Results saved → %s", output_path)
    except OSError as exc:
        logger.error("Failed to save results to %s: %s", output_path, exc)
        raise

    return output_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_baseline_results(
    metrics: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Persist baseline ML model metrics to ``results/baseline/``.

    Parameters
    ----------
    metrics : dict
        Nested mapping of ``model_name → {metric_name: value}``, as returned
        by :func:`src.models.ml_baseline.train_baseline_models`.
    extra : dict, optional
        Any additional metadata to include in the saved file (e.g., dataset
        size, training configuration).

    Returns
    -------
    Path
        Path of the saved JSON file.

    Examples
    --------
    >>> metrics = {"XGBoost": {"AUROC": 0.85, "F1": 0.76}}
    >>> path = save_baseline_results(metrics)
    >>> print(path.name)
    baseline_20260305T...Z.json
    """
    payload: Dict[str, Any] = {
        "experiment": "baseline",
        "timestamp": _iso_timestamp(),
        "metrics": metrics,
    }
    if extra:
        payload["extra"] = extra

    return _save_json(_BASELINE_DIR, payload, prefix="baseline")


def save_hybrid_results(
    output: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Persist hybrid pipeline output to ``results/hybrid/``.

    Parameters
    ----------
    output : dict
        Hybrid pipeline result containing at minimum:

        * ``patient_id``      — patient identifier
        * ``ml_score``        — ML risk probability
        * ``lstm_score``      — LSTM risk probability
        * ``num_retrieved``   — number of similar cases retrieved
        * ``reasoning``       — dict with ``sepsis_risk``, ``reasoning``,
                                 ``confidence`` from the LLM
    extra : dict, optional
        Any additional metadata (elapsed time, pipeline version, etc.).

    Returns
    -------
    Path
        Path of the saved JSON file.

    Examples
    --------
    >>> result = {
    ...     "patient_id": "p012345",
    ...     "ml_score": 0.81,
    ...     "lstm_score": 0.77,
    ...     "num_retrieved": 5,
    ...     "reasoning": {"sepsis_risk": "HIGH", "reasoning": "...", "confidence": "0.79"},
    ... }
    >>> path = save_hybrid_results(result)
    >>> print(path.name)
    hybrid_20260305T...Z.json
    """
    payload: Dict[str, Any] = {
        "experiment": "hybrid",
        "timestamp": _iso_timestamp(),
        "output": output,
    }
    if extra:
        payload["extra"] = extra

    return _save_json(_HYBRID_DIR, payload, prefix="hybrid")
