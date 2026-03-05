"""
run_baseline.py — Baseline experiment runner that trains real ML models.

This script implements a complete training pipeline:
1. Load PhysioNet dataset (subset of 3000 patients).
2. Engineer flat ML features.
3. Split into 80/20 stratified train/test sets.
4. Train Logistic Regression, Random Forest, and XGBoost models.
5. Evaluate models on AUROC, Accuracy, Precision, Recall, and F1.
6. Save the trained models as .pkl files in the models/ directory.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Local imports
from src.preprocessing.dataset_loader import load_all_patients
from src.preprocessing.feature_engineering import create_ml_features
from src.utils.logger import setup_logger
from src.utils.results_manager import save_experiment_results
from src.models.ml_baseline import _log_comparison_table

logger = setup_logger("baseline_training")

def train_and_eval() -> None:
    """Implement the real training and evaluation pipeline."""
    t0 = time.time()
    
    # ── Step 1: Load dataset ──────────────────────────────────────────────
    logger.info("── Step 1: Loading PhysioNet dataset (subset: 3000 patients) ──")
    try:
        raw_df = load_all_patients("A", max_patients=3000)
    except Exception as e:
        logger.error("Failed to load dataset: %s", e)
        sys.exit(1)

    # ── Step 2: Generate ML features ────────────────────────────────────────
    logger.info("── Step 2: Extracting ML features ──")
    X, y = create_ml_features(raw_df)

    # ── Step 3: Split data (80/20 split) ───────────────────────────────────
    logger.info("── Step 3: Performing 80/20 stratified train/test split ──")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ── Step 4: Preprocessing (Imputation + Scaling) ────────────────────────
    logger.info("── Preprocessing: Imputing missing values and scaling features ──")
    # Handling any residual missing values (though create_ml_features does some)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    
    X_train_proc = scaler.fit_transform(imputer.fit_transform(X_train))
    X_test_proc = scaler.transform(imputer.transform(X_test))

    # ── Step 5: Train Models ──────────────────────────────────────────────
    logger.info("── Step 4: Training Logistic Regression, Random Forest, XGBoost ──")
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, 
            random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=42, 
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=6, 
            use_label_encoder=False, 
            eval_metric="logloss", 
            random_state=42
        )
    }

    results: Dict[str, Dict[str, float]] = {}
    trained_artifacts = {}

    for name, model in models.items():
        logger.info("Training %s …", name)
        model.fit(X_train_proc, y_train)
        trained_artifacts[name] = model

        # Evaluate
        y_pred = model.predict(X_test_proc)
        # Probabilities for AUROC
        y_prob = model.predict_proba(X_test_proc)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        results[name] = {
            "AUROC": round(roc_auc_score(y_test, y_prob), 4),
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1": round(f1_score(y_test, y_pred, zero_division=0), 4)
        }

    # ── Step 6: Print Evaluation Table ─────────────────────────────────────
    # Reusing the formatter from ml_baseline.py
    _log_comparison_table(results)

    # ── Step 7: Save trained models ────────────────────────────────────────
    logger.info("── Step 6: Saving models to models/ directory ──")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    save_map = {
        "LogisticRegression": "logistic_regression.pkl",
        "RandomForest": "random_forest.pkl",
        "XGBoost": "xgboost_model.pkl"
    }

    for name, filename in save_map.items():
        save_path = models_dir / filename
        joblib.dump(trained_artifacts[name], save_path)
        logger.info("Saved %s to %s", name, save_path)

    # Log results to JSON (persistent log)
    save_experiment_results("baseline_production", results)

    elapsed = time.time() - t0
    logger.info("Baseline training pipeline complete in %.2fs", elapsed)

if __name__ == "__main__":
    train_and_eval()
