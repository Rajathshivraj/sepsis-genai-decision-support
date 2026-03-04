"""
run_baseline.py — Baseline experiment runner using placeholder metrics.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.logger import setup_logger
from src.utils.results_manager import save_experiment_results
from src.models.ml_baseline import _log_comparison_table

logger = setup_logger("baseline_experiment")

def main() -> None:
    """Run the baseline experiment with placeholder results."""
    t0 = time.time()
    
    logger.info("Running baseline experiment with placeholder metrics...")

    # Placeholder metrics as requested
    results = {
        "LogisticRegression": {
            "AUROC": 0.72,
            "Accuracy": 0.68,
            "Precision": 0.64,
            "Recall": 0.61,
            "F1": 0.62
        },
        "RandomForest": {
            "AUROC": 0.81,
            "Accuracy": 0.77,
            "Precision": 0.74,
            "Recall": 0.71,
            "F1": 0.72
        },
        "XGBoost": {
            "AUROC": 0.85,
            "Accuracy": 0.80,
            "Precision": 0.77,
            "Recall": 0.75,
            "F1": 0.76
        }
    }

    # Log the summary table using the existing helper
    _log_comparison_table(results)

    # Save metrics using results_manager
    save_experiment_results("baseline", results)

    elapsed = time.time() - t0
    logger.info(f"Baseline experiment complete (simulated) in {elapsed:.2f}s")

if __name__ == "__main__":
    main()
