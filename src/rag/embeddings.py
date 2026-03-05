"""
embeddings.py — Sentence-embedding utilities for patient case RAG retrieval.

Wraps ``sentence-transformers`` to produce dense vector representations of
natural-language patient case summaries.  A module-level singleton ensures the
model is loaded only once per process, avoiding repeated disk I/O.

Model
-----
``all-MiniLM-L6-v2`` — a compact, high-quality general-purpose embedding model
that produces 384-dimensional vectors.  Configurable via :data:`configs.config.cfg`.

Functions
---------
* :func:`generate_embedding`  — embed a single text string.
* :func:`generate_embeddings` — embed a list of text strings (batched).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from configs.config import cfg
from src.utils.logger import setup_logger

logger = setup_logger("embeddings")

# ---------------------------------------------------------------------------
# Module-level singleton — model loaded once and reused
# ---------------------------------------------------------------------------

_model = None          # SentenceTransformer instance
_loaded_model_name: Optional[str] = None


def _get_model(model_name: Optional[str] = None):
    """Return the cached SentenceTransformer, loading it on first call.

    Parameters
    ----------
    model_name : str, optional
        Sentence-transformers model identifier.  If ``None``, falls back to
        ``cfg.EMBEDDING_MODEL`` (``"all-MiniLM-L6-v2"``).

    Returns
    -------
    SentenceTransformer
        Loaded and cached model instance.
    """
    global _model, _loaded_model_name

    if model_name is None:
        model_name = cfg.EMBEDDING_MODEL

    if _model is None or _loaded_model_name != model_name:
        from sentence_transformers import SentenceTransformer  # lazy import

        logger.info("Loading embedding model: %s", model_name)
        _model = SentenceTransformer(model_name)
        _loaded_model_name = model_name
        logger.info("Embedding model loaded successfully.")

    return _model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_embedding(
    text: str,
    model_name: Optional[str] = None,
) -> np.ndarray:
    """Generate a dense vector embedding for a single text string.

    Parameters
    ----------
    text : str
        The input text to embed (e.g., a patient case summary).
    model_name : str, optional
        Sentence-transformers model name.  Defaults to ``cfg.EMBEDDING_MODEL``.

    Returns
    -------
    np.ndarray
        1-D float32 numpy array of shape ``(embedding_dim,)`` — typically
        ``(384,)`` for ``all-MiniLM-L6-v2``.

    Raises
    ------
    ValueError
        If ``text`` is empty or whitespace-only.
    RuntimeError
        If the embedding model fails to load or encode.

    Examples
    --------
    >>> vec = generate_embedding("Patient shows elevated lactate levels.")
    >>> vec.shape
    (384,)
    """
    if not text or not text.strip():
        raise ValueError("Input text must be a non-empty string.")

    try:
        model = _get_model(model_name)
        embedding: np.ndarray = model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embedding.astype(np.float32)
    except Exception as exc:
        logger.error("Failed to generate embedding: %s", exc)
        raise RuntimeError(f"Embedding generation failed: {exc}") from exc


def generate_embeddings(
    texts: List[str],
    model_name: Optional[str] = None,
    batch_size: int = 32,
    show_progress: bool = False,
) -> np.ndarray:
    """Generate dense vector embeddings for a list of text strings.

    Parameters
    ----------
    texts : list[str]
        Input texts to embed (e.g., patient case summaries).
    model_name : str, optional
        Sentence-transformers model name.  Defaults to ``cfg.EMBEDDING_MODEL``.
    batch_size : int, default 32
        Number of texts to encode per batch.  Larger values are faster on GPU.
    show_progress : bool, default False
        Whether to display a tqdm progress bar during encoding.

    Returns
    -------
    np.ndarray
        2-D float32 numpy array of shape ``(len(texts), embedding_dim)``.
        For ``all-MiniLM-L6-v2``, ``embedding_dim = 384``.

    Raises
    ------
    ValueError
        If ``texts`` is empty.
    RuntimeError
        If the embedding model fails to load or encode.

    Examples
    --------
    >>> cases = ["Patient A summary...", "Patient B summary..."]
    >>> vecs = generate_embeddings(cases)
    >>> vecs.shape
    (2, 384)
    """
    if not texts:
        raise ValueError("texts must be a non-empty list of strings.")

    try:
        model = _get_model(model_name)
        logger.info(
            "Generating embeddings for %d texts (batch_size=%d) …",
            len(texts),
            batch_size,
        )
        embeddings: np.ndarray = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        logger.info(
            "Embeddings ready — shape %s, dtype %s",
            embeddings.shape,
            embeddings.dtype,
        )
        return embeddings.astype(np.float32)
    except Exception as exc:
        logger.error("Failed to generate embeddings: %s", exc)
        raise RuntimeError(f"Batch embedding generation failed: {exc}") from exc
