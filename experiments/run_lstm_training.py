"""
run_lstm_training.py — Train a real LSTM model on patient time-series data.

This script implements the "real" LSTM training pipeline:
1. Load PhysioNet dataset (subset of 3000 patients).
2. Prepare sliding-window sequences.
3. Train the PyTorch LSTM model for 20 epochs.
4. Evaluate on AUROC, Accuracy, Precision, Recall, and F1.
5. Save the trained model to models/lstm_model.pth.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import torch
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Local imports
from src.preprocessing.dataset_loader import load_all_patients
from src.models.lstm_model import prepare_lstm_sequences, train_lstm_model
from src.utils.logger import setup_logger
from src.utils.results_manager import save_experiment_results
from configs.config import cfg

logger = setup_logger("lstm_training")

def main() -> None:
    """Run the real LSTM training pipeline."""
    t0 = time.time()
    
    # ── Step 1: Load dataset ──────────────────────────────────────────────
    logger.info("── Step 1: Loading PhysioNet dataset (subset: 3000 patients) ──")
    try:
        # Load enough patients to get a good training signal
        df = load_all_patients("A", max_patients=3000)
    except Exception as e:
        logger.error("Failed to load dataset: %s", e)
        sys.exit(1)

    if df.empty:
        logger.error("No data loaded. Check data directory.")
        sys.exit(1)

    # ── Step 2: Prepare Sequences ─────────────────────────────────────────
    logger.info("── Step 2: Preparing time-series sequences ──")
    X_seq, y_seq = prepare_lstm_sequences(df)

    # ── Step 3: Train & Eval LSTM ─────────────────────────────────────────
    logger.info("── Step 3: Training LSTM model (%d epochs) ──", cfg.EPOCHS)
    model, metrics = train_lstm_model(
        X_seq,
        y_seq,
        epochs=cfg.EPOCHS,
        batch_size=cfg.BATCH_SIZE,
        learning_rate=cfg.LEARNING_RATE,
        test_size=0.2,
        random_state=42
    )

    # ── Step 4: Save Model ────────────────────────────────────────────────
    logger.info("── Step 4: Saving LSTM model to models/ directory ──")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    save_path = models_dir / "lstm_model.pth"
    torch.save(model.state_dict(), save_path)
    logger.info("Saved LSTM state_dict to %s", save_path)

    # Save metrics using results_manager
    save_experiment_results("lstm_production", {"LSTM": metrics})

    elapsed = time.time() - t0
    logger.info("LSTM training pipeline complete in %.2fs", elapsed)

if __name__ == "__main__":
    main()
