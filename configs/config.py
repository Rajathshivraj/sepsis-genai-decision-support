"""
config.py — Centralised configuration for the Sepsis AI project.

All tuneable parameters live here so that experiments can be adjusted
from a single location without touching individual module source files.

Usage
-----
::

    from configs.config import cfg

    df = load_all_patients("A", max_patients=cfg.MAX_PATIENTS)
    X_seq, y_seq = prepare_lstm_sequences(df, sequence_length=cfg.SEQUENCE_LENGTH)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# ---------------------------------------------------------------------------
# Project root — resolved relative to this config file
#   configs/config.py  →  ../
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Project-wide configuration container.

    Attributes are grouped by subsystem.  Modify values here or override
    them at runtime via ``cfg.<ATTR> = value``.
    """

    # ── Dataset settings ──────────────────────────────────────────────────
    DATA_PATH: Path = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "physionet.org"
        / "files"
        / "challenge-2019"
        / "1.0.0"
        / "training"
    )
    MAX_PATIENTS: int = 500
    """Maximum patients to load per training set (``None`` = all)."""

    TRAINING_SET: str = "A"
    """Which training set to use: ``"A"``, ``"B"``, or ``"both"``."""

    SEQUENCE_LENGTH: int = 12
    """Number of consecutive ICU hours per LSTM input window."""

    # ── ML model settings ─────────────────────────────────────────────────
    TRAIN_TEST_SPLIT: float = 0.20
    """Fraction of data reserved for the test set."""

    RANDOM_STATE: int = 42
    """Global random seed for reproducibility."""

    # ── LSTM settings ─────────────────────────────────────────────────────
    HIDDEN_SIZE: int = 64
    """LSTM hidden-state dimensionality."""

    NUM_LAYERS: int = 1
    """Number of stacked LSTM layers."""

    EPOCHS: int = 20
    """Training epochs for the LSTM model."""

    BATCH_SIZE: int = 64
    """Mini-batch size for LSTM training."""

    LEARNING_RATE: float = 1e-3
    """Adam optimiser learning rate."""

    DROPOUT: float = 0.3
    """Dropout probability after the LSTM layer."""

    # ── RAG settings ──────────────────────────────────────────────────────
    TOP_K_RETRIEVAL: int = 5
    """Number of similar cases to retrieve from the vector store."""

    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    """Sentence-transformers model for patient case embeddings."""

    # ── LLM settings ──────────────────────────────────────────────────────
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    """Base URL of the local Ollama server."""

    OLLAMA_MODEL_NAME: str = "llama3:latest"
    """Ollama model tag to use for clinical reasoning."""

    MAX_TOKENS: int = 512
    """Maximum tokens the LLM is allowed to generate per request."""

    LLM_TEMPERATURE: float = 0.3
    """LLM sampling temperature (lower = more deterministic)."""

    LLM_TIMEOUT: int = 120
    """HTTP request timeout (seconds) for Ollama API calls."""


# ---------------------------------------------------------------------------
# Singleton instance — import this from other modules
# ---------------------------------------------------------------------------

cfg = Config()
