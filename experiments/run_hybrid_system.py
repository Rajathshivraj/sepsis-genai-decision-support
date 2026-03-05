"""
run_hybrid_system.py — Full hybrid AI pipeline for sepsis risk screening.

Orchestrates a 10-step pipeline that combines classical ML, LSTM time-series
modelling, FAISS-backed retrieval-augmented generation (RAG), and LLM clinical
reasoning (via Ollama/llama3) to produce an explainable sepsis risk decision
for a query ICU patient.

Pipeline steps
--------------
1.  Load the PhysioNet 2019 dataset (training set A, ≤ 200 patients).
2.  Select a representative subset by unique patient ID.
3.  Engineer per-patient ML features via :func:`create_ml_features`.
4.  Train a lightweight XGBoost model and generate ML risk scores.
5.  Build LSTM sequences and generate LSTM risk scores.
6.  Convert each patient's data into a natural-language case summary.
7.  Build a FAISS vector index over case-summary embeddings.
8.  Retrieve the top-k most similar historical cases for a query patient.
9.  Send the query patient state to the LLM for clinical reasoning.
10. Print the final structured decision report and persist results.

Usage
-----
::

    python experiments/run_hybrid_system.py

Dependencies
------------
    pip install sentence-transformers faiss-cpu torch xgboost scikit-learn pandas
    # Ollama must be running locally for LLM reasoning:
    #   ollama serve  &&  ollama pull llama3
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Ensure project root is importable (handles running as `python experiments/…`)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from configs.config import cfg
from src.preprocessing.dataset_loader import load_all_patients, TARGET_COLUMN
from src.preprocessing.feature_engineering import create_ml_features
from src.models.lstm_model import prepare_lstm_sequences, train_lstm_model
from src.rag.case_builder import build_all_patient_cases, build_patient_case
from src.rag.vector_store import build_vector_index, retrieve_similar_cases
from src.llm.reasoner import generate_clinical_reasoning
from src.utils.logger import setup_logger
from src.utils.results_logger import save_hybrid_results
from src.utils.uncertainty import compute_model_agreement, compute_uncertainty

logger = setup_logger("hybrid_system")

# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

_MAX_PATIENTS = 200       # cap for fast experiments; increase for full runs
_TRAINING_SET = "A"       # which PhysioNet training set to use
_TOP_K = cfg.TOP_K_RETRIEVAL  # number of similar cases to retrieve


# ---------------------------------------------------------------------------
# Step helpers
# ---------------------------------------------------------------------------

def _get_ml_scores(
    df: pd.DataFrame,
) -> Dict[str, float]:
    """Train a quick XGBoost model and return per-patient probability scores.

    Parameters
    ----------
    df : pd.DataFrame
        Raw multi-row patient dataset.

    Returns
    -------
    dict
        Mapping of ``patient_id → float`` ML risk probability in ``[0, 1]``.
    """
    logger.info("── Step 3 & 4: ML feature engineering + XGBoost scoring ──")
    X, y = create_ml_features(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=cfg.RANDOM_STATE,
        verbosity=0,
    )
    model.fit(X_scaled, y)
    probs = model.predict_proba(X_scaled)[:, 1]

    scores = {pid: float(p) for pid, p in zip(X.index, probs)}
    logger.info("ML scores computed for %d patients.", len(scores))
    return scores


def _get_lstm_scores(df: pd.DataFrame) -> Dict[str, float]:
    """Train a short-epoch LSTM and return per-sequence probability scores.

    Because the LSTM operates on sliding windows (not per-patient), this
    function maps each patient to the **mean probability** across all windows
    extracted from their stay.

    Parameters
    ----------
    df : pd.DataFrame
        Raw multi-row patient dataset.

    Returns
    -------
    dict
        Mapping of ``patient_id → float`` LSTM risk probability in ``[0, 1]``.
    """
    logger.info("── Step 5: LSTM sequence preparation + scoring ──")
    try:
        X_seq, y_seq = prepare_lstm_sequences(df, sequence_length=cfg.SEQUENCE_LENGTH)
        model, metrics = train_lstm_model(
            X_seq,
            y_seq,
            epochs=5,           # fast sweep; increase for production accuracy
            batch_size=cfg.BATCH_SIZE,
            random_state=cfg.RANDOM_STATE,
        )
        logger.info("LSTM training metrics: %s", metrics)

        # Generate a per-patient mean score from all their sliding windows
        import torch

        model.eval()
        all_probs: List[float] = []
        with torch.no_grad():
            tensor = torch.tensor(X_seq, dtype=torch.float32)
            logits = model(tensor).squeeze(1)
            probs = torch.sigmoid(logits).numpy()
            all_probs = probs.tolist()

        # Map window index back to patient_id (same ordering as sequences)
        # Build same sequence map to track patient_id per window
        patient_windows: Dict[str, List[float]] = {}
        seq_len = cfg.SEQUENCE_LENGTH
        idx = 0
        for patient_id, patient_df in df.groupby("patient_id"):
            patient_df = patient_df.sort_values("ICULOS")
            n_rows = len(patient_df)
            n_windows = max(n_rows - seq_len + 1, 1)
            window_probs = all_probs[idx: idx + n_windows]
            if window_probs:
                patient_windows[patient_id] = window_probs
            idx += n_windows

        scores = {pid: float(np.mean(v)) for pid, v in patient_windows.items()}
        logger.info("LSTM scores computed for %d patients.", len(scores))
        return scores

    except Exception as exc:
        logger.error("LSTM scoring failed: %s. Defaulting to 0.5 for all.", exc)
        patient_ids = df["patient_id"].unique()
        return {pid: 0.5 for pid in patient_ids}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Execute the full hybrid sepsis screening pipeline."""
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("  SEPSIS GENAI HYBRID PIPELINE  —  starting …")
    logger.info("=" * 60)

    # ── Step 1: Load dataset ──────────────────────────────────────────────
    logger.info("── Step 1: Loading dataset (max_patients=%d, set=%s) ──",
                _MAX_PATIENTS, _TRAINING_SET)
    try:
        df = load_all_patients(_TRAINING_SET, max_patients=_MAX_PATIENTS)
    except Exception as exc:
        logger.error("Dataset load failed: %s", exc)
        logger.warning("Pipeline cannot continue without data. Exiting.")
        sys.exit(1)

    if df.empty:
        logger.error("No patient data loaded. Check data directory.")
        sys.exit(1)

    # ── Step 2: Select patient subset ─────────────────────────────────────
    logger.info("── Step 2: Selecting patient subset ──")
    all_pids = df["patient_id"].unique()
    subset_pids = all_pids[:_MAX_PATIENTS]
    df_subset = df[df["patient_id"].isin(subset_pids)].copy()
    logger.info("Working with %d patients (%d rows).",
                len(subset_pids), len(df_subset))

    # ── Steps 3 & 4: ML features + scores ────────────────────────────────
    ml_scores: Dict[str, float] = {}
    try:
        ml_scores = _get_ml_scores(df_subset)
    except Exception as exc:
        logger.error("ML scoring pipeline failed: %s.", exc)

    # ── Step 5: LSTM scores ───────────────────────────────────────────────
    lstm_scores: Dict[str, float] = {}
    try:
        lstm_scores = _get_lstm_scores(df_subset)
    except Exception as exc:
        logger.error("LSTM scoring pipeline failed: %s.", exc)

    # ── Step 6: Build case summaries ──────────────────────────────────────
    logger.info("── Step 6: Building patient case summaries ──")
    try:
        cases = build_all_patient_cases(df_subset)
    except Exception as exc:
        logger.error("Case builder failed: %s", exc)
        cases = {}

    if not cases:
        logger.error("No case summaries generated. Exiting.")
        sys.exit(1)

    # ── Step 8: Build vector index ────────────────────────────────────────
    logger.info("── Step 8: Building FAISS vector index ──")
    try:
        store = build_vector_index(cases)
    except Exception as exc:
        logger.error("Vector index build failed: %s", exc)
        store = None

    # ── Step 9: Select query patient + retrieve similar cases ─────────────
    logger.info("── Step 9: Retrieving similar cases for query patient ──")

    # Select the patient with the highest ML risk score as a compelling demo
    if ml_scores:
        query_pid = max(ml_scores, key=ml_scores.get)
    else:
        query_pid = subset_pids[0]

    logger.info("Query patient: %s", query_pid)
    query_case: str = cases.get(query_pid, "Patient data unavailable.")

    retrieved: List[Dict[str, Any]] = []
    if store is not None:
        try:
            retrieved = retrieve_similar_cases(query_case, k=_TOP_K, store=store)
            logger.info("Retrieved %d similar cases.", len(retrieved))
        except Exception as exc:
            logger.error("Case retrieval failed: %s", exc)

    # Resolve per-patient risk scores for the query patient
    ml_risk = ml_scores.get(query_pid, 0.5)
    lstm_risk = lstm_scores.get(query_pid, 0.5)

    # ── Step 7: Uncertainty calibration ──────────────────────────────────
    logger.info("── Step 7: Computing model agreement and uncertainty ──")
    agreement = compute_model_agreement(ml_risk, lstm_risk)
    uncertainty = compute_uncertainty(ml_risk, lstm_risk)
    logger.info(
        "Model agreement: %s  |  Uncertainty: %.4f", agreement, uncertainty
    )

    # ── Step 10: LLM clinical reasoning ──────────────────────────────────
    logger.info("── Step 10: Generating LLM clinical reasoning ──")

    try:
        reasoning = generate_clinical_reasoning(
            patient_summary=query_case,
            retrieved_cases=retrieved,
            ml_score=ml_risk,
            lstm_score=lstm_risk,
        )
    except Exception as exc:
        logger.error("LLM reasoning failed: %s. Using fallback.", exc)
        reasoning = {
            "sepsis_risk": "MODERATE",
            "reasoning": (
                "LLM reasoning unavailable. Risk assessment is based on ML "
                f"score ({ml_risk:.2f}) and LSTM score ({lstm_risk:.2f}) only."
            ),
            "confidence": "LOW (FALLBACK)",
        }

    # ── Step 11: Print structured output + save results ──────────────────
    elapsed = time.time() - t0
    _print_decision_report(
        patient_id=query_pid,
        ml_score=ml_risk,
        lstm_score=lstm_risk,
        agreement=agreement,
        uncertainty=uncertainty,
        num_retrieved=len(retrieved),
        reasoning=reasoning,
        elapsed=elapsed,
    )

    result_payload = {
        "patient_id": query_pid,
        "ml_score": round(ml_risk, 4),
        "lstm_score": round(lstm_risk, 4),
        "agreement": agreement,
        "uncertainty": uncertainty,
        "num_retrieved_cases": len(retrieved),
        "reasoning": reasoning,
        "patient_case_summary": query_case,
        "elapsed_seconds": round(elapsed, 2),
    }

    try:
        save_hybrid_results(result_payload)
    except Exception as exc:
        logger.error("Failed to save results: %s", exc)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _print_decision_report(
    patient_id: str,
    ml_score: float,
    lstm_score: float,
    agreement: str,
    uncertainty: float,
    num_retrieved: int,
    reasoning: Dict[str, str],
    elapsed: float,
) -> None:
    """Log the final structured sepsis risk decision report.

    Parameters
    ----------
    patient_id : str
        The patient being assessed.
    ml_score : float
        XGBoost risk probability.
    lstm_score : float
        LSTM risk probability.
    agreement : str
        Model agreement level — ``"HIGH"``, ``"MEDIUM"``, or ``"LOW"``.
    uncertainty : float
        Scalar uncertainty score in ``[0, 1]``.
    num_retrieved : int
        Number of similar historical cases retrieved.
    reasoning : dict
        LLM output with keys ``sepsis_risk``, ``reasoning``, ``confidence``.
    elapsed : float
        Total pipeline wall-clock time in seconds.
    """
    sep = "─" * 50

    logger.info("\n%s", sep)
    logger.info("  ## Sepsis Decision Support")
    logger.info("  Patient ID : %s", patient_id)
    logger.info(sep)
    logger.info("  ML Risk Score   : %.2f", ml_score)
    logger.info("  LSTM Risk Score : %.2f", lstm_score)
    logger.info("")
    logger.info("  Model Agreement  : %s", agreement)
    logger.info("  Uncertainty Score: %.4f", uncertainty)
    logger.info("")
    logger.info("  Retrieved Similar Cases : %d", num_retrieved)
    logger.info("")
    logger.info("  Sepsis Risk Level : %s", reasoning.get("sepsis_risk", "N/A"))
    logger.info("")
    logger.info("  LLM Reasoning:")
    logger.info("  %s", reasoning.get("reasoning", "N/A"))
    logger.info("")
    logger.info("  Confidence : %s", reasoning.get("confidence", "N/A"))
    logger.info("%s", sep)
    logger.info("  Pipeline completed in %.2fs", elapsed)
    logger.info("%s\n", sep)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
