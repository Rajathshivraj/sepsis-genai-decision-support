"""
charts.py — Plotly chart generators for the Sepsis AI dashboard.

Charts: Risk Gauge, Risk Timeline, Feature Importance (SHAP).
"""

from __future__ import annotations
from typing import Any, Dict, List
from src.utils.logger import setup_logger

logger = setup_logger("charts")


def create_risk_gauge(score: float, title: str = "Ensemble Sepsis Risk") -> Any:
    """Semi-circular risk gauge. score in [0,1]."""
    import plotly.graph_objects as go
    color = _risk_color(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score * 100,
        number={"suffix": "%", "font": {"size": 28, "color": "#e2e8f0"}},
        title={"text": title, "font": {"size": 16, "color": "#a0aec0"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#4a5568",
                     "dtick": 20, "tickfont": {"color": "#a0aec0"}},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "#1a1f2e", "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "rgba(72,187,120,0.15)"},
                {"range": [40, 70], "color": "rgba(237,137,54,0.15)"},
                {"range": [70, 100], "color": "rgba(245,101,101,0.15)"},
            ],
            "threshold": {"line": {"color": "#e2e8f0", "width": 2},
                          "thickness": 0.8, "value": score * 100},
        },
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def create_risk_timeline(trajectory: List[Dict[str, Any]],
                         title: str = "Risk Trajectory Over Time") -> Any:
    """Line chart of per-hour risk scores."""
    import plotly.graph_objects as go
    hours = [t["hour"] for t in trajectory]
    scores = [t["risk_score"] for t in trajectory]
    fig = go.Figure()
    fig.add_hrect(y0=0, y1=0.4, fillcolor="rgba(72,187,120,0.08)", line_width=0)
    fig.add_hrect(y0=0.4, y1=0.7, fillcolor="rgba(237,137,54,0.08)", line_width=0)
    fig.add_hrect(y0=0.7, y1=1.0, fillcolor="rgba(245,101,101,0.08)", line_width=0)
    fig.add_trace(go.Scatter(
        x=hours, y=scores, mode="lines+markers", name="Risk Score",
        line=dict(color="#f6ad55", width=3),
        marker=dict(size=6, color=[_risk_color(s) for s in scores]),
        hovertemplate="Hour %{x}<br>Risk: %{y:.2%}<extra></extra>",
    ))
    fig.add_hline(y=0.4, line_dash="dash", line_color="#ed8936", opacity=0.5,
                  annotation_text="Moderate", annotation_position="bottom right",
                  annotation_font_color="#ed8936")
    fig.add_hline(y=0.7, line_dash="dash", line_color="#f56565", opacity=0.5,
                  annotation_text="High", annotation_position="bottom right",
                  annotation_font_color="#f56565")
    fig.update_layout(
        title=dict(text=title, font=dict(color="#e2e8f0", size=14)),
        xaxis=dict(title="Hour", color="#a0aec0", gridcolor="#2d3748",
                   dtick=max(1, len(hours) // 10)),
        yaxis=dict(title="Risk Score", range=[0, 1.05], color="#a0aec0",
                   gridcolor="#2d3748"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=300, margin=dict(l=40, r=20, t=40, b=40), showlegend=False,
    )
    return fig


def create_feature_importance_chart(top_features: List[Dict[str, Any]],
                                    title: str = "Top Risk-Driving Features (SHAP)",
                                    n: int = 8) -> Any:
    """Horizontal bar chart of SHAP feature importance."""
    import plotly.graph_objects as go
    features = list(reversed(top_features[:n]))
    names = [f["feature"] for f in features]
    values = [f["shap_value"] for f in features]
    colors = ["#f56565" if v > 0 else "#48bb78" for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h", marker_color=colors,
        text=[f"{v:+.4f}" for v in values], textposition="outside",
        textfont=dict(color="#e2e8f0", size=11),
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#e2e8f0", size=14)),
        xaxis=dict(title="SHAP Value", color="#a0aec0", gridcolor="#2d3748",
                   zeroline=True, zerolinecolor="#4a5568"),
        yaxis=dict(color="#a0aec0"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=max(200, 40 * len(features) + 80),
        margin=dict(l=120, r=60, t=40, b=40), showlegend=False,
    )
    return fig


def _risk_color(score: float) -> str:
    if score >= 0.7: return "#f56565"
    if score >= 0.4: return "#ed8936"
    return "#48bb78"
