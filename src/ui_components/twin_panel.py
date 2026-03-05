"""
twin_panel.py — Digital twin intervention comparison panel.
"""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st


# Intervention colours
_COLORS = {
    "baseline":             "#fc8181",
    "fluid_resuscitation":  "#63b3ed",
    "early_antibiotics":    "#68d391",
}


def render_twin_panel(results: Dict[str, Any]) -> None:
    """Render the digital twin intervention comparison panel.

    Uses src.digital_twin.risk_twin output.
    Gracefully degrades if twin data is unavailable.
    """
    from src.ui_components.icu_layout import icu_panel

    twin = results.get("twin_result")
    if not twin:
        return

    icu_panel("🧬 Digital Twin — Intervention Simulation")

    summary = twin.get("summary", {})
    comparisons = summary.get("comparisons", {})
    recommended = summary.get("recommended_intervention", "")

    if not comparisons:
        st.info("No simulation data available.")
        return

    # Recommendation badge
    if recommended:
        rec_label = summary.get("recommended_label", recommended)
        st.markdown(
            f'<div class="icu-alert-info">'
            f'💡 <strong>Recommended:</strong> {rec_label}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Comparison metrics
    cols = st.columns(len(comparisons))
    for col, (key, comp) in zip(cols, comparisons.items()):
        with col:
            label = comp.get("label", key)
            endpoint = comp.get("endpoint_risk", 0)
            trend = comp.get("trend", "UNKNOWN")
            is_rec = key == recommended

            border = "2px solid #63b3ed" if is_rec else "1px solid #2d3748"
            st.markdown(
                f'<div class="twin-card" style="border:{border}">'
                f'<div class="twin-card-title">{"⭐ " if is_rec else ""}{label}</div>'
                f'<div class="vital-value" style="font-size:1.3rem;color:#e2e8f0">'
                f'{endpoint:.3f}</div>'
                f'<div class="twin-card-desc">{trend}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Plotly overlay chart
    _render_twin_chart(twin)


def _render_twin_chart(twin: Dict[str, Any]) -> None:
    """Render overlaid forecast trajectories for each intervention."""
    try:
        import plotly.graph_objects as go

        fig = go.Figure()

        for key, entry in twin.items():
            if key == "summary":
                continue

            forecast = entry.get("forecast", {})
            predictions = forecast.get("forecast", [])
            current = forecast.get("current_risk", 0)

            if not predictions:
                continue

            hours = [0] + [p["hour"] for p in predictions]
            risks = [current] + [p["risk"] for p in predictions]
            color = _COLORS.get(key, "#a0aec0")
            label = entry.get("label", key)

            fig.add_trace(go.Scatter(
                x=hours, y=risks,
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=2),
                marker=dict(size=6, color=color),
            ))

        # Risk zones
        fig.add_hrect(y0=0.7, y1=1.0, fillcolor="rgba(252,129,129,0.06)", line_width=0)
        fig.add_hrect(y0=0.4, y1=0.7, fillcolor="rgba(246,173,85,0.04)", line_width=0)

        fig.update_layout(
            height=300,
            xaxis_title="Hours from Now",
            yaxis_title="Sepsis Risk",
            yaxis=dict(range=[0, 1]),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#a0aec0"),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1,
            ),
            margin=dict(l=40, r=20, t=30, b=40),
        )
        fig.update_xaxes(gridcolor="rgba(45,55,72,0.5)")
        fig.update_yaxes(gridcolor="rgba(45,55,72,0.5)")

        st.plotly_chart(fig, use_container_width=True)

    except Exception:
        st.caption("Chart unavailable — install plotly.")
