"""
dataset_loader.py — PhysioNet 2019 Sepsis Challenge dataset loader.

Provides utilities to:
  • Load individual patient .psv files into pandas DataFrames.
  • Batch-load all patients from one or both training sets (A / B).
  • Produce a concise dataset summary for downstream EDA.

Dataset layout expected under `data/raw/`:
  physionet.org/files/challenge-2019/1.0.0/training/training_setA/p*.psv
  physionet.org/files/challenge-2019/1.0.0/training/training_setB/p*.psv

Each .psv file is pipe-separated ("|"), one row per hour of ICU stay.
Target column: SepsisLabel (0 = no sepsis, 1 = sepsis).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import pandas as pd

from configs.config import cfg
from src.utils.logger import setup_logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

logger = setup_logger("dataset_loader")

# Resolve project root relative to this file:
#   src/preprocessing/dataset_loader.py  →  ../../
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

_RAW_DATA_DIR = (
    _PROJECT_ROOT
    / "data"
    / "raw"
    / "physionet.org"
    / "files"
    / "challenge-2019"
    / "1.0.0"
    / "training"
)

_SET_DIRS: Dict[str, Path] = {
    "A": _RAW_DATA_DIR / "training_setA",
    "B": _RAW_DATA_DIR / "training_setB",
}

TARGET_COLUMN = "SepsisLabel"

# Column groupings (useful downstream)
VITAL_SIGN_COLUMNS = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
]

LAB_VALUE_COLUMNS = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets",
]

DEMOGRAPHIC_COLUMNS = [
    "Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS",
]

FEATURE_COLUMNS = VITAL_SIGN_COLUMNS + LAB_VALUE_COLUMNS + DEMOGRAPHIC_COLUMNS


# ---------------------------------------------------------------------------
# Single-patient loader
# ---------------------------------------------------------------------------

def load_patient_file(filepath: Union[str, Path]) -> pd.DataFrame:
    """Load a single patient .psv file and return a DataFrame.

    Parameters
    ----------
    filepath : str | Path
        Absolute or relative path to a ``.psv`` file.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per ICU hour. A ``patient_id`` column is
        added, derived from the filename (e.g. ``"p012949"``).
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Patient file not found: {filepath}")

    df = pd.read_csv(filepath, sep="|", na_values=["NaN", "nan", ""])
    df["patient_id"] = filepath.stem  # e.g. "p012949"
    return df


# ---------------------------------------------------------------------------
# Batch loader
# ---------------------------------------------------------------------------

TrainingSet = Literal["A", "B", "both"]


def load_all_patients(
    training_set: TrainingSet = "both",
    *,
    max_patients: Optional[int] = None,
    data_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Load all patient files from the PhysioNet Sepsis dataset.

    Parameters
    ----------
    training_set : {"A", "B", "both"}
        Which training set(s) to load. ``"both"`` loads A and B.
    max_patients : int, optional
        If given, cap the number of patients loaded **per set** (useful for
        quick experimentation & debugging).
    data_dir : str | Path, optional
        Override the default raw-data root directory.  When ``None`` the
        loader resolves it relative to the project tree.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with all patient rows.  Contains an extra
        ``patient_id`` column and a ``training_set`` column (``"A"`` or
        ``"B"``).

    Raises
    ------
    FileNotFoundError
        If the resolved dataset directory does not exist.
    ValueError
        If ``training_set`` is not one of the accepted values.
    """
    sets_to_load = _resolve_sets(training_set)
    set_dirs = _resolve_data_dirs(data_dir)

    all_frames: List[pd.DataFrame] = []

    for set_key in sets_to_load:
        set_path = set_dirs[set_key]
        if not set_path.exists():
            raise FileNotFoundError(
                f"Training set directory not found: {set_path}"
            )

        psv_files = sorted(set_path.glob("*.psv"))
        if max_patients is not None:
            psv_files = psv_files[:max_patients]

        logger.info(
            "Loading %d patient files from training_set%s …",
            len(psv_files),
            set_key,
        )

        for psv_file in psv_files:
            try:
                df = load_patient_file(psv_file)
                df["training_set"] = set_key
                all_frames.append(df)
            except Exception:
                logger.warning(
                    "Skipping corrupt/unreadable file: %s", psv_file, exc_info=True
                )

    if not all_frames:
        logger.warning("No patient files were loaded — returning empty DataFrame.")
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)
    logger.info(
        "Loaded %d rows across %d patients.",
        len(combined),
        combined["patient_id"].nunique(),
    )
    return combined


# ---------------------------------------------------------------------------
# Dataset summary
# ---------------------------------------------------------------------------

def get_dataset_summary(df: pd.DataFrame) -> Dict[str, object]:
    """Return a quick summary dict of the loaded dataset.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame produced by :func:`load_all_patients`.

    Returns
    -------
    dict
        Keys: ``total_rows``, ``total_patients``, ``sepsis_positive_patients``,
        ``sepsis_negative_patients``, ``sepsis_prevalence``,
        ``features``, ``missing_rate``, ``hours_per_patient``.
    """
    if df.empty:
        return {"total_rows": 0, "total_patients": 0}

    total_patients = df["patient_id"].nunique()

    # A patient is sepsis-positive if they have *any* row with SepsisLabel == 1
    patient_labels = df.groupby("patient_id")[TARGET_COLUMN].max()
    sepsis_pos = int((patient_labels == 1).sum())
    sepsis_neg = int((patient_labels == 0).sum())

    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    missing_rate = (
        df[feature_cols].isna().mean().to_dict() if feature_cols else {}
    )

    hours_per_patient = df.groupby("patient_id").size()

    return {
        "total_rows": len(df),
        "total_patients": total_patients,
        "sepsis_positive_patients": sepsis_pos,
        "sepsis_negative_patients": sepsis_neg,
        "sepsis_prevalence": round(sepsis_pos / total_patients, 4)
        if total_patients
        else 0.0,
        "features": feature_cols,
        "missing_rate": missing_rate,
        "hours_per_patient": {
            "mean": round(hours_per_patient.mean(), 2),
            "median": float(hours_per_patient.median()),
            "min": int(hours_per_patient.min()),
            "max": int(hours_per_patient.max()),
        },
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_sets(training_set: TrainingSet) -> List[str]:
    """Validate and expand the *training_set* argument."""
    if training_set == "both":
        return ["A", "B"]
    if training_set in ("A", "B"):
        return [training_set]
    raise ValueError(
        f"training_set must be 'A', 'B', or 'both'; got {training_set!r}"
    )


def _resolve_data_dirs(
    data_dir: Optional[Union[str, Path]],
) -> Dict[str, Path]:
    """Return {set_key: directory_path} mapping."""
    if data_dir is not None:
        base = Path(data_dir)
        return {
            "A": base / "training_setA",
            "B": base / "training_setB",
        }
    return dict(_SET_DIRS)  # use module-level defaults


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    )

    logger.info("Loading a sample of 50 patients from training_setA …")
    sample_df = load_all_patients("A", max_patients=50)

    summary = get_dataset_summary(sample_df)
    logger.info("\n── Dataset Summary ──\n%s", json.dumps(summary, indent=2, default=str))
    logger.info("DataFrame shape : %s", sample_df.shape)
    logger.info("Columns         : %s", list(sample_df.columns))
    logger.info("\nFirst 5 rows:\n%s", sample_df.head())
