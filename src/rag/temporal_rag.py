"""
temporal_rag.py

Temporal RAG retrieval for ICU patient deterioration patterns.
Uses time-series features instead of summary text.
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from .vector_store import load_vector_store

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def build_temporal_summary(df: pd.DataFrame) -> str:
    """
    Convert a patient time-series into a compact temporal description.
    """

    hr_mean = df["HR"].mean()
    hr_trend = df["HR"].iloc[-1] - df["HR"].iloc[0]

    map_mean = df["MAP"].mean()
    map_trend = df["MAP"].iloc[-1] - df["MAP"].iloc[0]

    lact_max = df["Lactate"].max()

    resp_mean = df["Resp"].mean()

    text = (
        f"Heart rate mean {hr_mean:.1f}, trend {hr_trend:.1f}. "
        f"MAP mean {map_mean:.1f}, trend {map_trend:.1f}. "
        f"Respiratory rate mean {resp_mean:.1f}. "
        f"Peak lactate {lact_max:.1f}."
    )

    return text


def retrieve_temporal_cases(df: pd.DataFrame, k: int = 5):
    """
    Retrieve ICU cases with similar temporal physiological patterns.
    """

    index, cases = load_vector_store()

    query_text = build_temporal_summary(df)

    embedding = model.encode([query_text])

    distances, indices = index.search(np.array(embedding).astype("float32"), k)

    results = []

    for i, idx in enumerate(indices[0]):
        case = cases[idx].copy()
        case["similarity"] = float(1 - distances[0][i])
        results.append(case)

    return results
