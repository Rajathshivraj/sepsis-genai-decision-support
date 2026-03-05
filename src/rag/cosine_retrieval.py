"""
cosine_retrieval.py — Cosine-similarity RAG retrieval for ICU patient cases.

Provides an alternative retrieval function that normalizes FAISS embeddings
and computes cosine similarity scores instead of raw L2 distances.

This module is OPTIONAL and does NOT modify vector_store.py.
It reuses the same FAISS index and patient cases from disk.
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np

from src.utils.logger import setup_logger

logger = setup_logger("cosine_retrieval")

# Paths — same as vector_store.py
INDEX_PATH = "models/patient_index.faiss"
CASES_PATH = "models/patient_cases.pkl"

# Module-level cache
_embedding_model = None
_index = None
_cases = None
_normalized_vectors = None


# ---------------------------------------------------------------------------
# Internal: lazy loaders
# ---------------------------------------------------------------------------

def _get_embedding_model():
    """Load the sentence-transformer model (singleton)."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        logger.info("Loading embedding model: %s", model_name)
        _embedding_model = SentenceTransformer(model_name)
    return _embedding_model


def _load_index_and_cases():
    """Load the FAISS index and patient cases (singleton)."""
    global _index, _cases, _normalized_vectors

    if _index is not None and _cases is not None:
        return _index, _cases

    import faiss

    if not os.path.exists(INDEX_PATH) or not os.path.exists(CASES_PATH):
        raise FileNotFoundError(
            f"FAISS index ({INDEX_PATH}) or cases ({CASES_PATH}) not found. "
            "Run scripts/build_patient_index.py first."
        )

    logger.info("Loading FAISS index from %s", INDEX_PATH)
    _index = faiss.read_index(INDEX_PATH)

    with open(CASES_PATH, "rb") as f:
        _cases = pickle.load(f)

    logger.info("Loaded %d vectors, %d cases", _index.ntotal, len(_cases))

    # Pre-compute normalized vectors for cosine similarity
    _normalized_vectors = _extract_and_normalize(_index)

    return _index, _cases


def _extract_and_normalize(index) -> Optional[np.ndarray]:
    """Extract raw vectors from the FAISS index and L2-normalize them.

    Returns None if extraction fails (e.g. non-flat index type).
    """
    try:
        import faiss

        n = index.ntotal
        d = index.d

        # Reconstruct all vectors from the index
        vectors = np.zeros((n, d), dtype=np.float32)
        for i in range(n):
            vectors[i] = index.reconstruct(i)

        # L2-normalize each vector
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # avoid division by zero
        normalized = vectors / norms

        logger.info("Normalized %d vectors (dim=%d) for cosine retrieval", n, d)
        return normalized

    except Exception as exc:
        logger.warning("Vector extraction failed: %s — falling back to L2", exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve_similar_cases_cosine(
    query_text: str,
    k: int = 5,
) -> List[Dict[str, Any]]:
    """Retrieve the top-k most similar patient cases using cosine similarity.

    Parameters
    ----------
    query_text : str
        Natural-language patient case summary or clinical query.
    k : int
        Number of results to return.

    Returns
    -------
    list[dict]
        Each entry has the original case fields plus:
        * ``similarity`` — cosine similarity score in [0, 1].
        * ``retrieval_method`` — "cosine".
    """
    _, cases = _load_index_and_cases()
    model = _get_embedding_model()

    # Encode query
    query_vec = model.encode([query_text], convert_to_numpy=True).astype(np.float32)

    # Normalize query vector
    query_norm = np.linalg.norm(query_vec, axis=1, keepdims=True)
    query_norm = np.where(query_norm == 0, 1, query_norm)
    query_normalized = query_vec / query_norm

    if _normalized_vectors is not None:
        # Compute cosine similarity via dot product on normalized vectors
        similarities = np.dot(_normalized_vectors, query_normalized.T).flatten()

        # Get top-k indices
        top_k_indices = np.argsort(similarities)[::-1][:k]

        results = []
        for idx in top_k_indices:
            if idx < len(cases):
                case = cases[idx].copy()
                case["similarity"] = round(float(similarities[idx]), 4)
                case["retrieval_method"] = "cosine"
                results.append(case)
    else:
        # Fallback: use FAISS L2 search + convert distances to pseudo-similarity
        logger.info("Using L2 fallback with similarity conversion")
        results = _l2_fallback(query_vec, cases, k)

    logger.info(
        "Cosine retrieval — query length=%d, top-1 sim=%.4f",
        len(query_text),
        results[0]["similarity"] if results else 0,
    )

    return results


def _l2_fallback(
    query_vec: np.ndarray,
    cases: list,
    k: int,
) -> List[Dict[str, Any]]:
    """Fallback using the raw FAISS L2 index, converting distances to similarity."""
    distances, indices = _index.search(query_vec, k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(cases):
            case = cases[idx].copy()
            # Convert L2 distance to a pseudo-cosine similarity
            # sim ≈ 1 / (1 + d^2)  — maps to (0, 1]
            d = float(distances[0][i])
            case["similarity"] = round(1.0 / (1.0 + d), 4)
            case["retrieval_method"] = "l2_converted"
            results.append(case)

    return results


def retrieve_and_compare(
    query_text: str,
    k: int = 5,
) -> Dict[str, List[Dict[str, Any]]]:
    """Retrieve using both L2 and cosine methods for comparison.

    Parameters
    ----------
    query_text : str
        Query text.
    k : int
        Number of results per method.

    Returns
    -------
    dict
        Keys: ``cosine``, ``l2``.
    """
    cosine_results = retrieve_similar_cases_cosine(query_text, k=k)

    # L2 results from existing vector store
    try:
        from src.rag.vector_store import retrieve_similar_cases
        l2_results = retrieve_similar_cases(query_text, k=k)
    except Exception:
        l2_results = []

    return {
        "cosine": cosine_results,
        "l2": l2_results,
    }
