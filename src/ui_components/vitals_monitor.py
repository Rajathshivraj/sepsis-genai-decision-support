"""
vitals_monitor.py — Real-time vital signs monitor with Plotly sparklines.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import streamlit as st


# Normal ranges for colour-coding
_RANGES = {
    "HR":      {"low": 60,  "high": 100, "unit": "bpm",     "label": "Heart Rate"},
    "MAP":     {"low": 65,  "high": 105, "unit": "mmHg",    "label": "MAP"},
    "Resp":    {"low": 12,  "high": 20,  "unit": "/min",    "label": "Resp Rate"},
    "Temp":    {"low": 36.1,"high": 38.0,"unit": "°C",      "label": "Temp"},
    "Lactate": {"low": 0.5, "high": 2.0, "unit": "mmol/L",  "label": "Lactate"},
    "O2Sat":   {"low": 94,  "high": 100, "unit": "%",       "label": "SpO₂"},
}


def render_vitals_monitor(
    df: pd.DataFrame,
    vitals: Dict[str, Any],
) -> None:
    """Render a multi-vital ICU monitor with sparkline trends.

    Parameters
    ----------
    df : pd.DataFrame
        Patient time-series data.
    vitals : dict
        Latest vitals dict.
    """
    from src.ui_components.icu_layout import icu_panel
    icu_panel("🫀 Vital Signs Monitor")

    cols = st.columns(len(_RANGES))

    for col, (key, info) in zip(cols, _RANGES.items()):
        with col:
            val = vitals.get(key)
            _render_vital_card(key, val, info, df)


def _render_vital_card(
    key: str,
    value: Any,
    info: Dict[str, Any],
    df: pd.DataFrame,
) -> None:
    """Render a single vital-sign card with status colour and sparkline."""

    if value is None or (isinstance(value, float) and np.isnan(value)):
        st.markdown(
            f'<div class="vital-card">'
            f'<div class="vital-label">{info["label"]}</div>'
            f'<div class="vital-value" style="color:#4a5568">—</div>'
            f'<div class="vital-unit">{info["unit"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        return

    val = float(value)
    low, high = info["low"], info["high"]

    # Determine status
    if val < low * 0.85 or val > high * 1.15:
        status_class = "vital-crit"
        color = "#fc8181"
    elif val < low or val > high:
        status_class = "vital-warn"
        color = "#f6ad55"
    else:
        status_class = "vital-normal"
        color = "#68d391"

    st.markdown(
        f'<div class="vital-card {status_class}">'
        f'<div class="vital-label">{info["label"]}</div>'
        f'<div class="vital-value" style="color:{color}">{val:.1f}</div>'
        f'<div class="vital-unit">{info["unit"]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Sparkline
    if key in df.columns and len(df) > 1:
        try:
            import plotly.graph_objects as go

            series = df[key].dropna().values[-24:]  # last 24 hours
            if len(series) > 1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=series,
                    mode="lines",
                    line=dict(color=color, width=1.5),
                    fill="tozeroy",
                    fillcolor=f"rgba({_hex_to_rgb(color)},0.08)",
                    showlegend=False,
                ))
                # Threshold bands
                fig.add_hline(y=low, line=dict(color="#4a5568", width=0.5, dash="dot"))
                fig.add_hline(y=high, line=dict(color="#4a5568", width=0.5, dash="dot"))

                fig.update_layout(
                    height=60, margin=dict(l=0, r=0, t=0, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                )
                st.plotly_chart(fig, use_container_width=True, key=f"spark_{key}")
        except Exception:
            pass


def _hex_to_rgb(hex_color: str) -> str:
    """Convert #rrggbb to 'r,g,b' string."""
    h = hex_color.lstrip("#")
    return ",".join(str(int(h[i:i+2], 16)) for i in (0, 2, 4))
