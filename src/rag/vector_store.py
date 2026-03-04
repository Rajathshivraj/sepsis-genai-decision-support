"""
vector_store.py — FAISS-backed vector index for patient case retrieval.

Uses ``sentence-transformers`` to embed patient case summaries and FAISS
for fast approximate nearest-neighbour search.

Workflow
--------
1. ``build_vector_index(cases)`` — embed all patient case summaries and
   build a FAISS ``IndexFlatIP`` (inner-product / cosine similarity on
   L2-normalised vectors).
2. ``retrieve_similar_cases(query_case, k)`` — embed a query case and
   return the top-*k* most similar historical cases.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from configs.config import cfg
from src.utils.logger import setup_logger

logger = setup_logger("vector_store")


# ---------------------------------------------------------------------------
# Data container — holds index + metadata together
# ---------------------------------------------------------------------------

@dataclass
class PatientVectorStore:
    """Container for the FAISS index and associated metadata.

    Attributes
    ----------
    index : faiss.Index
        FAISS index populated with normalised embeddings.
    patient_ids : list[str]
        Ordered patient IDs matching the index rows.
    cases : dict[str, str]
        Original case text keyed by patient_id.
    model_name : str
        Name of the sentence-transformer model used for embeddings.
    """
    index: object  # faiss.Index (typed loosely to avoid import at class level)
    patient_ids: List[str] = field(default_factory=list)
    cases: Dict[str, str] = field(default_factory=dict)
    model_name: str = ""


# Module-level store — populated by ``build_vector_index``
_store: Optional[PatientVectorStore] = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_vector_index(
    cases: Dict[str, str],
    model_name: Optional[str] = None,
) -> PatientVectorStore:
    """Embed patient case summaries and build a FAISS index.

    Parameters
    ----------
    cases : dict[str, str]
        Mapping of ``patient_id`` → case summary text produced by
        :func:`~src.rag.case_builder.build_all_patient_cases`.
    model_name : str
        Sentence-transformers model to use for encoding.

    Returns
    -------
    PatientVectorStore
        Populated store object.  Also cached at module level for use by
        :func:`retrieve_similar_cases`.
    """
    import faiss
    from sentence_transformers import SentenceTransformer

    global _store

    if model_name is None:
        model_name = cfg.EMBEDDING_MODEL

    logger.info(
        "Building vector index — %d cases, model=%s", len(cases), model_name
    )

    # ── Encode ────────────────────────────────────────────────────────────
    model = SentenceTransformer(model_name)
    patient_ids = list(cases.keys())
    texts = [cases[pid] for pid in patient_ids]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Normalise for cosine similarity via inner-product index
    faiss.normalize_L2(embeddings)

    # ── Build FAISS index ─────────────────────────────────────────────────
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)   # cosine sim on normalised vecs
    index.add(embeddings)

    logger.info(
        "FAISS index built — %d vectors, dim=%d", index.ntotal, dimension
    )

    _store = PatientVectorStore(
        index=index,
        patient_ids=patient_ids,
        cases=dict(cases),
        model_name=model_name,
    )
    return _store


def retrieve_similar_cases(
    query_case: str,
    k: Optional[int] = None,
    store: Optional[PatientVectorStore] = None,
) -> List[Dict[str, object]]:
    """Retrieve the top-*k* most similar patient cases.

    Parameters
    ----------
    query_case : str
        Natural-language patient case summary to query against.
    k : int
        Number of results to return.
    store : PatientVectorStore, optional
        Vector store to search.  If ``None``, uses the module-level store
        populated by the last call to :func:`build_vector_index`.

    Returns
    -------
    list[dict]
        Each dict contains:

        * ``patient_id`` — matching patient identifier.
        * ``score`` — cosine similarity score.
        * ``case`` — full case summary text.

    Raises
    ------
    RuntimeError
        If no vector store has been built yet.
    """
    import faiss
    from sentence_transformers import SentenceTransformer

    if store is None:
        store = _store
    if store is None:
        raise RuntimeError(
            "No vector store available.  Call build_vector_index() first."
        )

    if k is None:
        k = cfg.TOP_K_RETRIEVAL

    # Encode the query
    model = SentenceTransformer(store.model_name)
    query_emb = model.encode([query_case], convert_to_numpy=True)
    faiss.normalize_L2(query_emb)

    # Search
    k = min(k, store.index.ntotal)
    scores, indices = store.index.search(query_emb, k)

    results: List[Dict[str, object]] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue  # FAISS sentinel for missing results
        pid = store.patient_ids[idx]
        results.append({
            "patient_id": pid,
            "score": round(float(score), 4),
            "case": store.cases[pid],
        })

    logger.info("Retrieved %d similar cases (top score=%.4f)", len(results), results[0]["score"] if results else 0.0)
    return results
