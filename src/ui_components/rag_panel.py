"""
rag_panel.py — RAG similar cases display with styled cards.
"""

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st


def render_rag_panel(results: Dict[str, Any]) -> None:
    """Render the RAG retrieved cases panel with styled cards.

    Displays top-k similar historical ICU cases and clinical guidelines.
    """
    from src.ui_components.icu_layout import icu_panel
    icu_panel("📚 Similar Historical Cases (RAG)")

    retrieved = results.get("retrieved", [])

    if not retrieved:
        st.info("No similar cases retrieved from the vector store.")
        return

    for i, case in enumerate(retrieved[:5], 1):
        _render_case_card(i, case)

    # Guidelines
    guidelines = results.get("guidelines", [])
    if guidelines:
        with st.expander("📋 Matched Clinical Guidelines"):
            for g in guidelines:
                st.markdown(f"**{g.get('title', '')}**")
                st.caption(g.get("text", ""))


def _render_case_card(index: int, case: Dict[str, Any]) -> None:
    """Render a single similar-case card."""
    pid = case.get("patient_id", "unknown")
    score = case.get("score", case.get("similarity", 0.0))
    text = case.get("case", "")
    method = case.get("retrieval_method", "")

    # Parse key details from the case text
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    snippet_lines = [
        ln for ln in lines
        if not ln.startswith("Patient summary") and not ln.startswith("---")
    ][:4]
    snippet = " · ".join(snippet_lines[:3])

    outcome_line = next(
        (ln for ln in lines if "Outcome" in ln), "Outcome: Unknown"
    )

    # Similarity bar colour
    if score >= 0.8:
        sim_color = "#68d391"
    elif score >= 0.5:
        sim_color = "#f6ad55"
    else:
        sim_color = "#a0aec0"

    method_badge = f"<span style='color:#4a5568;font-size:0.65rem'>[{method}]</span>" if method else ""

    st.markdown(
        f'<div class="case-card">'
        f'<strong>Case {index} — {pid}</strong> '
        f'<span style="color:{sim_color}">sim={score:.3f}</span> {method_badge}<br>'
        f'<span style="color:#a0aec0;font-size:0.82rem">{snippet[:160]}…</span><br>'
        f'<em style="color:#f6ad55">{outcome_line.strip()}</em>'
        f'</div>',
        unsafe_allow_html=True,
    )
