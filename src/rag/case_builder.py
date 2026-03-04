"""
case_builder.py — Convert patient ICU data into natural-language case summaries.

Each patient's hourly time-series is distilled into a concise clinical
narrative suitable for:

* Embedding into a vector store for retrieval (RAG).
* Passing as context to an LLM for clinical reasoning.

The summaries capture trend information (first → last values), peak /
trough values for key vitals and labs, and the patient's sepsis outcome.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd

from src.preprocessing.dataset_loader import TARGET_COLUMN
from src.utils.logger import setup_logger

logger = setup_logger("case_builder")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_patient_case(df_patient: pd.DataFrame) -> str:
    """Build a text case summary for a single patient.

    Parameters
    ----------
    df_patient : pd.DataFrame
        Subset of rows for **one** patient (all sharing the same
        ``patient_id``), sorted by ``ICULOS``.

    Returns
    -------
    str
        Multi-line clinical narrative.

    Examples
    --------
    >>> from src.preprocessing.dataset_loader import load_all_patients
    >>> raw = load_all_patients("A", max_patients=5)
    >>> pid = raw["patient_id"].unique()[0]
    >>> case = build_patient_case(raw[raw["patient_id"] == pid])
    >>> logger.info("\\n%s", case[:80])
    Patient summary:
    ...
    """
    df = df_patient.sort_values("ICULOS").reset_index(drop=True)
    patient_id = df["patient_id"].iloc[0] if "patient_id" in df.columns else "unknown"
    icu_hours = int(len(df))

    sections: List[str] = []

    # ── Header ────────────────────────────────────────────────────────────
    sections.append("Patient summary:")
    sections.append(f"Patient ID: {patient_id}")
    sections.append(f"ICU stay duration: {icu_hours} hours")

    # ── Demographics ──────────────────────────────────────────────────────
    age = _safe_first(df, "Age")
    gender = _safe_first(df, "Gender")
    if age is not None:
        sections.append(f"Age: {age:.0f} years")
    if gender is not None:
        sections.append(f"Gender: {'Male' if int(gender) == 1 else 'Female'}")

    # ── Vital sign trends ─────────────────────────────────────────────────
    sections.append("")
    sections.append("Vital signs:")
    sections.append(_trend_line(df, "HR", "Heart rate", "bpm"))
    sections.append(_trend_line(df, "MAP", "Mean arterial pressure", "mmHg"))
    sections.append(_trend_line(df, "Temp", "Temperature", "°C"))
    sections.append(_trend_line(df, "Resp", "Respiratory rate", "breaths/min"))
    sections.append(_trend_line(df, "O2Sat", "Oxygen saturation", "%"))
    sections.append(_trend_line(df, "SBP", "Systolic blood pressure", "mmHg"))

    # ── Lab values ────────────────────────────────────────────────────────
    sections.append("")
    sections.append("Key lab values:")
    sections.append(_peak_line(df, "Lactate", "Lactate", "mmol/L"))
    sections.append(_peak_line(df, "Creatinine", "Creatinine", "mg/dL"))
    sections.append(_peak_line(df, "WBC", "White blood cell count", "×10³/µL"))
    sections.append(_peak_line(df, "Platelets", "Platelets", "×10³/µL"))
    sections.append(_peak_line(df, "Bilirubin_total", "Total bilirubin", "mg/dL"))

    # ── Outcome ───────────────────────────────────────────────────────────
    sections.append("")
    if TARGET_COLUMN in df.columns:
        sepsis = int(df[TARGET_COLUMN].max())
        if sepsis == 1:
            # Find the hour at which sepsis was first flagged
            onset_row = df[df[TARGET_COLUMN] == 1].iloc[0]
            onset_hour = int(onset_row["ICULOS"]) if "ICULOS" in df.columns else "?"
            sections.append(f"Outcome: Sepsis diagnosed at hour {onset_hour}.")
        else:
            sections.append("Outcome: No sepsis diagnosed during ICU stay.")
    else:
        sections.append("Outcome: Unknown (label not available).")

    # Filter out empty / None lines
    return "\n".join(line for line in sections if line is not None)


def build_all_patient_cases(df: pd.DataFrame) -> Dict[str, str]:
    """Build case summaries for every patient in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with ``patient_id`` column.

    Returns
    -------
    dict[str, str]
        Mapping of ``patient_id`` → case summary string.
    """
    cases: Dict[str, str] = {}
    grouped = df.groupby("patient_id")

    logger.info("Building case summaries for %d patients …", grouped.ngroups)

    for patient_id, patient_df in grouped:
        try:
            cases[patient_id] = build_patient_case(patient_df)
        except Exception:
            logger.warning(
                "Failed to build case for patient %s", patient_id, exc_info=True
            )

    logger.info("Built %d patient case summaries.", len(cases))
    return cases


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_first(df: pd.DataFrame, col: str) -> Optional[float]:
    """Return the first non-NaN value of *col*, or None."""
    if col not in df.columns:
        return None
    vals = df[col].dropna()
    return float(vals.iloc[0]) if len(vals) > 0 else None


def _safe_last(df: pd.DataFrame, col: str) -> Optional[float]:
    """Return the last non-NaN value of *col*, or None."""
    if col not in df.columns:
        return None
    vals = df[col].dropna()
    return float(vals.iloc[-1]) if len(vals) > 0 else None


def _trend_line(
    df: pd.DataFrame, col: str, label: str, unit: str
) -> Optional[str]:
    """Produce a single trend description line, e.g.
    'Heart rate changed from 85 to 120 bpm (peak: 130 bpm).'
    """
    first = _safe_first(df, col)
    last = _safe_last(df, col)
    if first is None and last is None:
        return f"  {label}: No data recorded."

    vals = df[col].dropna()
    peak = float(vals.max()) if len(vals) else None
    trough = float(vals.min()) if len(vals) else None

    if first is not None and last is not None:
        direction = "increased" if last > first else "decreased" if last < first else "stable"
        line = f"  {label} {direction} from {first:.1f} to {last:.1f} {unit}"
    elif last is not None:
        line = f"  {label} last recorded at {last:.1f} {unit}"
    else:
        line = f"  {label} first recorded at {first:.1f} {unit}"

    extras = []
    if peak is not None:
        extras.append(f"peak: {peak:.1f} {unit}")
    if trough is not None:
        extras.append(f"low: {trough:.1f} {unit}")
    if extras:
        line += f" ({', '.join(extras)})"
    line += "."

    return line


def _peak_line(
    df: pd.DataFrame, col: str, label: str, unit: str
) -> Optional[str]:
    """Produce a peak-value description for a lab column."""
    if col not in df.columns:
        return f"  {label}: Not measured."
    vals = df[col].dropna()
    if len(vals) == 0:
        return f"  {label}: Not measured."
    peak = float(vals.max())
    mean = float(vals.mean())
    return f"  {label}: peak {peak:.2f} {unit}, mean {mean:.2f} {unit}."
