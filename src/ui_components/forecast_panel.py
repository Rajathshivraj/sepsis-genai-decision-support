"""
forecast_panel.py — Risk forecast trajectory panel.
"""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st


def render_forecast_panel(results: Dict[str, Any]) -> None:
    """Render the risk forecast panel with a future trajectory chart.

    Uses src.forecasting.risk_forecast output.
    Gracefully degrades if forecast data is unavailable.
    """
    from src.ui_components.icu_layout import icu_panel

    forecast = results.get("forecast_result")
    if not forecast:
        return

    icu_panel("🔮 Risk Forecast (LSTM-Based)")

    current = forecast.get("current_risk", 0)
    trend = forecast.get("trend", "STABLE")
    predictions = forecast.get("forecast", [])

    if not predictions:
        st.info("No forecast data available.")
        return

    # Trend badge
    trend_emoji = {"INCREASING": "📈", "DECREASING": "📉", "STABLE": "➡️"}.get(trend, "➡️")
    trend_color = {"INCREASING": "#fc8181", "DECREASING": "#68d391", "STABLE": "#63b3ed"}.get(trend, "#63b3ed")

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Current Risk", f"{current:.3f}")
    mc2.metric("Trend", f"{trend_emoji} {trend}")
    endpoint = predictions[-1]["risk"] if predictions else current
    mc3.metric(f"{predictions[-1]['hour']}h Forecast", f"{endpoint:.3f}")

    # Plotly forecast chart
    try:
        import plotly.graph_objects as go

        hours = [0] + [p["hour"] for p in predictions]
        risks = [current] + [p["risk"] for p in predictions]

        fig = go.Figure()

        # High-risk zone
        fig.add_hrect(
            y0=0.7, y1=1.0,
            fillcolor="rgba(252,129,129,0.08)",
            line_width=0,
            annotation_text="HIGH", annotation_position="top left",
            annotation=dict(font_size=10, font_color="#fc8181"),
        )
        # Moderate zone
        fig.add_hrect(
            y0=0.4, y1=0.7,
            fillcolor="rgba(246,173,85,0.05)",
            line_width=0,
        )

        # Forecast line
        fig.add_trace(go.Scatter(
            x=hours, y=risks,
            mode="lines+markers",
            name="Predicted Risk",
            line=dict(color=trend_color, width=2.5),
            marker=dict(size=8, color=trend_color),
            fill="tozeroy",
            fillcolor=f"rgba({_hex_rgb(trend_color)},0.1)",
        ))

        # Current point
        fig.add_trace(go.Scatter(
            x=[0], y=[current],
            mode="markers",
            name="Now",
            marker=dict(size=14, color="#63b3ed", symbol="diamond"),
        ))

        fig.update_layout(
            height=280,
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
        # Fallback: simple table
        st.table(
            [{"Hour": f"t+{p['hour']}h", "Risk": f"{p['risk']:.3f}"}
             for p in predictions]
        )


def _hex_rgb(h: str) -> str:
    h = h.lstrip("#")
    return ",".join(str(int(h[i:i+2], 16)) for i in (0, 2, 4))
