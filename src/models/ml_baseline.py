"""
ml_baseline.py — Train and evaluate baseline ML models for sepsis prediction.

Implements three classifiers:

1. **Logistic Regression** — linear baseline.
2. **Random Forest** — ensemble of decision trees.
3. **XGBoost** — gradient-boosted trees (state-of-the-art tabular baseline).

Each model is trained on the same 80 / 20 stratified split and evaluated with:

* AUROC  (primary metric)
* Accuracy
* Precision
* Recall
* F1 Score

A comparison table is printed to stdout after evaluation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from configs.config import cfg
from src.utils.logger import setup_logger

logger = setup_logger("ml_baseline")

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: Dict[str, Any] = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=5,       # helps with class imbalance
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_baseline_models(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: Optional[float] = None,
    random_state: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """Train all baseline models and return evaluation metrics.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (one row per patient).
    y : pd.Series
        Binary target vector (1 = sepsis, 0 = no sepsis).
    test_size : float, default 0.20
        Fraction of data reserved for testing.
    random_state : int, default 42
        Seed for reproducible train/test splits.

    Returns
    -------
    results : dict[str, dict[str, float]]
        Nested dict mapping model name → metric name → score.

    Notes
    -----
    * Features are standardised (zero mean, unit variance) before training
      to ensure Logistic Regression converges reliably.
    * The train/test split is **stratified** to preserve the sepsis
      prevalence ratio in both partitions.
    """
    if test_size is None:
        test_size = cfg.TRAIN_TEST_SPLIT
    if random_state is None:
        random_state = cfg.RANDOM_STATE

    # ── 1. Train / test split ─────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    logger.info(
        "Split: %d train / %d test  (sepsis prevalence: train=%.1f%%, test=%.1f%%)",
        len(X_train), len(X_test),
        y_train.mean() * 100, y_test.mean() * 100,
    )

    # ── 2. Standardise features ───────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ── 3. Train & evaluate each model ────────────────────────────────────
    results: Dict[str, Dict[str, float]] = {}

    for name, model in _MODEL_REGISTRY.items():
        logger.info("Training %s …", name)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)

        # Probability estimates for AUROC
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_prob = y_pred.astype(float)

        metrics = _evaluate(y_test, y_pred, y_prob)
        results[name] = metrics

        logger.info("  %s — AUROC=%.4f  F1=%.4f", name, metrics["AUROC"], metrics["F1"])

    # ── 4. Print comparison table ─────────────────────────────────────────
    _log_comparison_table(results)

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """Compute classification metrics for a single model."""
    return {
        "AUROC":     round(roc_auc_score(y_true, y_prob), 4),
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


def _log_comparison_table(results: Dict[str, Dict[str, float]]) -> None:
    """Log a formatted comparison table."""
    metric_names = ["AUROC", "Accuracy", "Precision", "Recall", "F1"]

    # Column widths
    name_w = max(len(n) for n in results) + 2
    col_w = 12

    # Header
    header = f"{'Model':<{name_w}}" + "".join(f"{m:>{col_w}}" for m in metric_names)
    separator = "─" * len(header)

    logger.info("\n%s", separator)
    logger.info("  BASELINE MODEL COMPARISON")
    logger.info(separator)
    logger.info(header)
    logger.info(separator)

    for model_name, metrics in results.items():
        row = f"{model_name:<{name_w}}"
        row += "".join(f"{metrics[m]:>{col_w}.4f}" for m in metric_names)
        logger.info(row)

    logger.info(separator)

    # Highlight best AUROC
    best_model = max(results, key=lambda m: results[m]["AUROC"])
    logger.info("  ★ Best AUROC: %s (%.4f)", best_model, results[best_model]['AUROC'])
    logger.info("%s\n", separator)
