"""
RAG module initialization.

Provides access to vector store and retrieval utilities.
"""

from .vector_store import load_vector_store, retrieve_similar_cases

# Optional modules
try:
    from .advanced_rag import retrieve_with_guidelines, get_relevant_guidelines
except Exception:
    pass

try:
    from .temporal_rag import retrieve_temporal_cases
except Exception:
    pass
