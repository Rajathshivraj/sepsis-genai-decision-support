"""
feature_engineering.py — Convert patient time-series ICU data into flat ML features.

Takes the multi-row-per-patient DataFrame produced by
:func:`src.preprocessing.dataset_loader.load_all_patients` and collapses each
patient into a single feature vector suitable for classical ML models.

Feature extraction strategy
---------------------------
For each patient (grouped by ``patient_id``):

* **Vital signs** — aggregate statistics (mean, max, min) capture the
  patient's overall physiological state and the severity of excursions.
* **Lab values** — worst-case (max) values for Lactate and Creatinine are
  strong sepsis indicators.
* **ICU length of stay** — ``ICULOS_max`` captures total exposure time.

Target variable
---------------
``SepsisLabel`` is reduced to a single binary label per patient using
``max`` — a patient is positive if *any* hour was labelled sepsis.
"""

from __future__ import annotations

import logging
from typing import Tuple

import pandas as pd

from src.preprocessing.dataset_loader import TARGET_COLUMN
from src.utils.logger import setup_logger

logger = setup_logger("feature_engineering")

# ---------------------------------------------------------------------------
# Feature specification
# ---------------------------------------------------------------------------

# Each entry: (source_column, aggregation_function, output_column_name)
_FEATURE_SPEC: list[Tuple[str, str, str]] = [
    # Heart rate
    ("HR", "mean", "HR_mean"),
    ("HR", "max",  "HR_max"),
    ("HR", "min",  "HR_min"),
    # Mean arterial pressure
    ("MAP", "mean", "MAP_mean"),
    ("MAP", "min",  "MAP_min"),
    # Temperature
    ("Temp", "mean", "Temp_mean"),
    # Respiratory rate
    ("Resp", "mean", "Resp_mean"),
    # Lab values — worst-case indicators
    ("Lactate",    "max", "Lactate_max"),
    ("Creatinine", "max", "Creatinine_max"),
    # ICU length-of-stay
    ("ICULOS", "max", "ICULOS_max"),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_ml_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Collapse hourly patient data into one feature row per patient.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset as returned by
        :func:`~src.preprocessing.dataset_loader.load_all_patients`.
        Must contain columns referenced in the feature spec plus
        ``patient_id`` and ``SepsisLabel``.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with shape ``(n_patients, n_features)``.
        Index is ``patient_id``.  Missing values are forward-filled with the
        column median so that no NaNs remain.
    y : pd.Series
        Binary target vector (1 = sepsis, 0 = no sepsis) indexed by
        ``patient_id``.

    Raises
    ------
    KeyError
        If ``patient_id`` or ``SepsisLabel`` columns are missing from *df*.

    Examples
    --------
    >>> from src.preprocessing.dataset_loader import load_all_patients
    >>> raw = load_all_patients("A", max_patients=100)
    >>> X, y = create_ml_features(raw)
    >>> X.shape
    (100, 10)
    """
    _validate_input(df)

    grouped = df.groupby("patient_id")

    # --- Build aggregation dict from the feature spec ---
    # pandas .agg() accepts {output_col: (source_col, agg_func)}
    agg_dict = {
        output_name: (source_col, agg_func)
        for source_col, agg_func, output_name in _FEATURE_SPEC
    }

    logger.info(
        "Engineering %d features for %d patients …",
        len(agg_dict),
        grouped.ngroups,
    )

    features = grouped.agg(**agg_dict)

    # --- Handle missing values ---
    # After aggregation some patients may still have NaN if *every* hourly
    # reading was missing for a feature.  Fill with column median.
    missing_before = int(features.isna().sum().sum())
    if missing_before > 0:
        medians = features.median()
        features = features.fillna(medians)
        logger.info(
            "Filled %d missing aggregated values with column medians.",
            missing_before,
        )

    # --- Target variable ---
    # A patient is sepsis-positive if any hour in their stay has label 1.
    target = grouped[TARGET_COLUMN].max().astype(int)
    target.name = TARGET_COLUMN

    logger.info(
        "Feature matrix ready — shape %s | sepsis prevalence %.2f%%",
        features.shape,
        target.mean() * 100,
    )

    return features, target


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_input(df: pd.DataFrame) -> None:
    """Raise early if required columns are missing."""
    required = {"patient_id", TARGET_COLUMN}
    source_cols = {src for src, _, _ in _FEATURE_SPEC}
    required |= source_cols

    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"Input DataFrame is missing required columns: {sorted(missing)}"
        )


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    )

    from src.preprocessing.dataset_loader import load_all_patients

    logger.info("Loading 100 patients from training_setA …")
    raw = load_all_patients("A", max_patients=100)

    X, y = create_ml_features(raw)

    logger.info("── Feature Matrix ──")
    logger.info("Shape : %s", X.shape)
    logger.info("Columns: %s", list(X.columns))
    logger.info("First 5 rows:\n%s", X.head())
    logger.info("── Target ──")
    logger.info("Sepsis positive: %d / %d", int(y.sum()), len(y))
