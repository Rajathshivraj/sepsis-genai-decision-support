"""
vector_store.py — FAISS vector database loader for ICU patient cases.
"""

import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

INDEX_PATH = "models/patient_index.faiss"
CASES_PATH = "models/patient_cases.pkl"

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

index = None
cases = None


def load_vector_store():
    global index, cases

    if index is not None and cases is not None:
        return index, cases

    if os.path.exists(INDEX_PATH) and os.path.exists(CASES_PATH):
        print("Loading FAISS index from disk...")

        index = faiss.read_index(INDEX_PATH)

        with open(CASES_PATH, "rb") as f:
            cases = pickle.load(f)

        print(f"Loaded FAISS index — {index.ntotal} vectors")

    else:
        raise RuntimeError(
            "FAISS index not found. Please run scripts/build_patient_index.py first."
        )

    return index, cases


def retrieve_similar_cases(query_text, k=5):
    index, cases = load_vector_store()

    query_embedding = model.encode([query_text])

    distances, indices = index.search(query_embedding, k)

    results = []

    for i, idx in enumerate(indices[0]):
        case = cases[idx].copy()
        case["similarity"] = float(1 - distances[0][i])
        results.append(case)

    return results
