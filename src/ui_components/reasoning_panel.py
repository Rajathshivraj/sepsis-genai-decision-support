"""
reasoning_panel.py — AI reasoning container.
"""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st


def render_reasoning_panel(results: Dict[str, Any]) -> None:
    """Render the AI clinical reasoning panel."""
    from src.ui_components.icu_layout import icu_panel
    icu_panel("🧠 AI Clinical Reasoning")

    reasoning = results.get("reasoning", {})
    reasoning_text = reasoning.get("reasoning", "No reasoning generated.")

    st.markdown(
        f'<div class="reasoning-box">{reasoning_text}</div>',
        unsafe_allow_html=True,
    )

    cited = reasoning.get("cited_guidelines", [])
    if cited:
        st.write("")
        st.caption("📖 **Cited Guidelines:** " + ", ".join(cited))

    with st.expander("📄 Generated Patient Summary"):
        st.text(results.get("case_summary", "Not available."))
