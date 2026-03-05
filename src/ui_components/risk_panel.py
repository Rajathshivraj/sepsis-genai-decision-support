"""
risk_panel.py — ICU risk status panel with Plotly gauge.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import streamlit as st


def render_risk_panel(results: Dict[str, Any]) -> None:
    """Render the ICU risk status panel.

    Displays a large Plotly risk gauge alongside key model metrics.
    Falls back to simple metric cards if Plotly is unavailable.
    """
    ml_score = results.get("ml_score", 0)
    lstm_score = results.get("lstm_score", 0)
    ensemble_score = results.get("ensemble_score", (ml_score + lstm_score) / 2)
    agreement = results.get("agreement", "N/A")
    uncertainty = results.get("uncertainty", 0)
    reasoning = results.get("reasoning", {})
    risk_label = reasoning.get("sepsis_risk", _score_to_label(ensemble_score))
    confidence = reasoning.get("confidence", "N/A")

    # Risk badge
    risk_emoji = {"HIGH": "🔴", "MODERATE": "🟠", "LOW": "🟢"}.get(risk_label, "🟡")
    risk_css = {"HIGH": "risk-high", "MODERATE": "risk-medium", "LOW": "risk-low"}.get(risk_label, "risk-medium")

    # Try Plotly gauge
    try:
        from src.visualization.charts import create_risk_gauge
        gauge_col, info_col = st.columns([1, 2])
        with gauge_col:
            st.plotly_chart(create_risk_gauge(ensemble_score), use_container_width=True)
        with info_col:
            st.markdown(
                f'<div class="{risk_css}">'
                f'<h2 style="margin:0">{risk_emoji} {risk_label} RISK</h2>'
                f'<p style="margin:4px 0 0 0;font-size:1.05rem;">'
                f'Ensemble: {ensemble_score:.2f} · Confidence: {confidence}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.write("")
            c1, c2, c3 = st.columns(3)
            c1.metric("ML Score", f"{ml_score:.3f}")
            c2.metric("LSTM Score", f"{lstm_score:.3f}")
            c3.metric("Agreement", agreement)

            reliability = results.get("reliability")
            if reliability:
                r1, r2 = st.columns(2)
                r1.metric("Uncertainty", f"{uncertainty:.4f}")
                r2.metric("Reliability", reliability.get("reliability_label", "N/A"))
    except Exception:
        st.markdown(
            f'<div class="{risk_css}">'
            f'<h2 style="margin:0">{risk_emoji} {risk_label} RISK</h2>'
            f'<p style="margin:4px 0 0 0">Ensemble: {ensemble_score:.2f}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ML Score", f"{ml_score:.2f}")
        c2.metric("LSTM Score", f"{lstm_score:.2f}")
        c3.metric("Agreement", agreement)
        c4.metric("Uncertainty", f"{uncertainty:.4f}")

    # Clinical scores
    _render_clinical_scores(results)


def _render_clinical_scores(results: Dict[str, Any]) -> None:
    """Render qSOFA and SOFA score cards."""
    qsofa = results.get("qsofa")
    sofa = results.get("sofa")
    if not qsofa and not sofa:
        return

    st.write("")
    cs1, cs2 = st.columns(2)
    if qsofa:
        with cs1:
            st.metric("qSOFA", f"{qsofa.get('score', 'N/A')}/3")
            st.caption(qsofa.get("interpretation", ""))
            for c in qsofa.get("criteria_met", []):
                st.markdown(f"&nbsp;&nbsp;• {c}")
    if sofa:
        with cs2:
            st.metric("SOFA", f"{sofa.get('total_score', 'N/A')}/24")
            st.caption(sofa.get("interpretation", ""))
            for s in sofa.get("organ_dysfunction", []):
                st.markdown(f"&nbsp;&nbsp;⚠️ {s.capitalize()}")


def _score_to_label(score: float) -> str:
    if score >= 0.7:
        return "HIGH"
    if score >= 0.4:
        return "MODERATE"
    return "LOW"
