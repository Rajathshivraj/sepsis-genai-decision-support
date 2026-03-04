"""
lstm_model.py — LSTM-based sepsis prediction on patient time-series data.

Converts the raw hourly ICU DataFrame into fixed-length sequences per
patient and trains a single-layer LSTM binary classifier using PyTorch.

Pipeline
--------
1. ``prepare_lstm_sequences`` — window the hourly rows into
   ``(samples, time_steps, features)`` tensors.
2. ``train_lstm_model`` — fit an LSTM + FC head and return the trained
   model along with evaluation metrics.

Architecture
------------
::

    Input  →  LSTM(hidden_size)  →  Dropout  →  Linear(1)  →  Sigmoid

All default hyper-parameters are read from :data:`configs.config.cfg`.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from configs.config import cfg
from src.preprocessing.dataset_loader import (
    TARGET_COLUMN,
    VITAL_SIGN_COLUMNS,
)
from src.utils.logger import setup_logger

logger = setup_logger("lstm_model")

# Features used for the LSTM input — vitals + key labs
LSTM_FEATURE_COLUMNS: List[str] = VITAL_SIGN_COLUMNS + [
    "Lactate",
    "Creatinine",
    "WBC",
    "Platelets",
]


# ---------------------------------------------------------------------------
# 1. Sequence preparation
# ---------------------------------------------------------------------------

def prepare_lstm_sequences(
    df: pd.DataFrame,
    sequence_length: Optional[int] = None,
    feature_columns: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert hourly patient data into fixed-length sequences.

    For each patient, a sliding window of ``sequence_length`` consecutive
    hours is extracted.  The label for each window is the ``SepsisLabel``
    at the **last** hour of the window.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset from :func:`~src.preprocessing.dataset_loader.load_all_patients`.
    sequence_length : int, default 12
        Number of consecutive hours per sequence.
    feature_columns : list[str], optional
        Columns to include as features.  Defaults to
        :data:`LSTM_FEATURE_COLUMNS`.

    Returns
    -------
    X_seq : np.ndarray
        Shape ``(n_samples, sequence_length, n_features)``.
    y_seq : np.ndarray
        Shape ``(n_samples,)`` — binary labels.
    """
    if sequence_length is None:
        sequence_length = cfg.SEQUENCE_LENGTH
    if feature_columns is None:
        feature_columns = LSTM_FEATURE_COLUMNS

    # Use only columns that actually exist in the DataFrame
    available_cols = [c for c in feature_columns if c in df.columns]
    if not available_cols:
        raise ValueError("None of the requested feature columns exist in the DataFrame.")

    logger.info(
        "Preparing LSTM sequences — %d features, window=%d hours",
        len(available_cols),
        sequence_length,
    )

    # --- Impute missing values per-column before sequencing ----------------
    df = df.copy()
    for col in available_cols:
        df[col] = df[col].fillna(df[col].median())

    # --- Sliding-window extraction per patient ----------------------------
    X_sequences: List[np.ndarray] = []
    y_labels: List[int] = []

    for patient_id, patient_df in df.groupby("patient_id"):
        # Ensure temporal ordering
        patient_df = patient_df.sort_values("ICULOS")
        values = patient_df[available_cols].values          # (T, F)
        labels = patient_df[TARGET_COLUMN].values           # (T,)

        if len(values) < sequence_length:
            # Pad short patients with the first row repeated
            pad_len = sequence_length - len(values)
            pad = np.repeat(values[:1], pad_len, axis=0)
            values = np.vstack([pad, values])
            labels = np.concatenate([np.zeros(pad_len, dtype=labels.dtype), labels])

        for i in range(len(values) - sequence_length + 1):
            X_sequences.append(values[i : i + sequence_length])
            y_labels.append(int(labels[i + sequence_length - 1]))

    X_seq = np.array(X_sequences, dtype=np.float32)
    y_seq = np.array(y_labels, dtype=np.float32)

    logger.info(
        "Sequences ready — X %s, y %s  (pos=%.1f%%)",
        X_seq.shape,
        y_seq.shape,
        y_seq.mean() * 100,
    )
    return X_seq, y_seq


# ---------------------------------------------------------------------------
# 2. LSTM model definition
# ---------------------------------------------------------------------------

class SepsisLSTM(nn.Module):
    """Single-layer LSTM binary classifier for sepsis prediction."""

    def __init__(
        self,
        input_size: int,
        hidden_size: Optional[int] = None,
        num_layers: Optional[int] = None,
        dropout: Optional[float] = None,
    ) -> None:
        super().__init__()
        hidden_size = hidden_size if hidden_size is not None else cfg.HIDDEN_SIZE
        num_layers = num_layers if num_layers is not None else cfg.NUM_LAYERS
        dropout_p = dropout if dropout is not None else cfg.DROPOUT
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Shape ``(batch, seq_len, input_size)``.

        Returns
        -------
        Tensor
            Raw logits of shape ``(batch, 1)``.
        """
        # lstm_out: (batch, seq_len, hidden)
        lstm_out, _ = self.lstm(x)
        # Use the hidden state from the last time-step
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden)
        dropped = self.dropout(last_hidden)
        logits = self.fc(dropped)
        return logits


# ---------------------------------------------------------------------------
# 3. Training loop
# ---------------------------------------------------------------------------

def train_lstm_model(
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    *,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    test_size: Optional[float] = None,
    random_state: Optional[int] = None,
    device: Optional[str] = None,
) -> Tuple[SepsisLSTM, Dict[str, float]]:
    """Train the LSTM model and return it with evaluation metrics.

    Parameters
    ----------
    X_seq : np.ndarray
        Shape ``(n_samples, seq_len, n_features)``.
    y_seq : np.ndarray
        Shape ``(n_samples,)`` — binary labels.
    epochs : int
        Number of training epochs.
    batch_size : int
        Mini-batch size.
    learning_rate : float
        Adam optimiser learning rate.
    test_size : float
        Fraction reserved for evaluation.
    random_state : int
        Seed for reproducibility.
    device : str, optional
        ``"cuda"`` or ``"cpu"``.  Auto-detected if ``None``.

    Returns
    -------
    model : SepsisLSTM
        Trained model (moved to CPU).
    metrics : dict[str, float]
        Evaluation results: AUROC, Accuracy, Precision, Recall, F1.
    """
    epochs = epochs if epochs is not None else cfg.EPOCHS
    batch_size = batch_size if batch_size is not None else cfg.BATCH_SIZE
    learning_rate = learning_rate if learning_rate is not None else cfg.LEARNING_RATE
    test_size = test_size if test_size is not None else cfg.TRAIN_TEST_SPLIT
    random_state = random_state if random_state is not None else cfg.RANDOM_STATE

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Training LSTM on device=%s", device)

    # ── Train / test split ────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq,
        test_size=test_size,
        stratify=y_seq.astype(int),
        random_state=random_state,
    )

    # ── Scale features (fit on train only) ────────────────────────────────
    n_samples_tr, seq_len, n_features = X_train.shape
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, n_features)
    X_test_flat = X_test.reshape(-1, n_features)
    scaler.fit(X_train_flat)
    X_train = scaler.transform(X_train_flat).reshape(n_samples_tr, seq_len, n_features)
    X_test = scaler.transform(X_test_flat).reshape(-1, seq_len, n_features)

    # ── DataLoaders ───────────────────────────────────────────────────────
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # ── Model, loss, optimiser ────────────────────────────────────────────
    model = SepsisLSTM(input_size=n_features).to(device)

    # Compute positive weight for class imbalance
    n_pos = float(y_train.sum())
    n_neg = float(len(y_train) - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            epoch_loss += loss.item() * len(X_batch)

        avg_loss = epoch_loss / len(train_ds)
        if epoch % 5 == 0 or epoch == 1:
            logger.info("  Epoch %2d/%d — loss=%.4f", epoch, epochs, avg_loss)

    # ── Evaluation ────────────────────────────────────────────────────────
    model.eval()
    all_logits: List[float] = []
    all_labels: List[float] = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch).squeeze(1)
            all_logits.extend(logits.cpu().numpy().tolist())
            all_labels.extend(y_batch.numpy().tolist())

    y_true = np.array(all_labels)
    y_prob = torch.sigmoid(torch.tensor(all_logits)).numpy()
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "AUROC":     round(float(roc_auc_score(y_true, y_prob)), 4),
        "Accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "Recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "F1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }

    logger.info(
        "LSTM evaluation — AUROC=%.4f  F1=%.4f  Recall=%.4f",
        metrics["AUROC"], metrics["F1"], metrics["Recall"],
    )

    model = model.cpu()
    return model, metrics
