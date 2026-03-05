"""
alerts_panel.py — Patient status strip and clinical alerts.
"""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st


def render_patient_status_strip(
    patient_id: str,
    results: Dict[str, Any],
    timestamp: str = "Just now"
) -> None:
    """Render the persistent top status strip."""
    ensemble_score = results.get("ensemble_score", 0.0)
    reasoning = results.get("reasoning", {})
    confidence = reasoning.get("confidence", "N/A")

    if ensemble_score >= 0.7:
        risk_label = "HIGH"
        color = "#fc8181"
    elif ensemble_score >= 0.4:
        risk_label = "MODERATE"
        color = "#f6ad55"
    else:
        risk_label = "LOW"
        color = "#68d391"

    st.markdown(
        f'''
        <div class="icu-status-bar">
            <div class="icu-status-item">
                PID: <span class="icu-status-value">{patient_id}</span>
            </div>
            <div class="icu-status-item">
                RISK: <span class="icu-status-value" style="color:{color}">{risk_label}</span>
            </div>
            <div class="icu-status-item">
                CONFIDENCE: <span class="icu-status-value">{confidence}</span>
            </div>
            <div class="icu-status-item" style="margin-left:auto">
                UPDATED: <span class="icu-status-value">{timestamp}</span>
            </div>
        </div>
        ''',
        unsafe_allow_html=True
    )


def render_alerts_panel(results: Dict[str, Any], vitals: Dict[str, Any]) -> None:
    """Render active clinical alerts based on current vitals and scores."""
    alerts = []

    # High risk alert
    ensemble_score = results.get("ensemble_score", 0.0)
    if ensemble_score >= 0.7:
        alerts.append((
            "critical",
            f"CRITICAL: High Sepsis Risk Detected (Score: {ensemble_score:.2f}). Imminent intervention recommended."
        ))

    # Vitals alerts
    map_val = vitals.get("MAP")
    if map_val is not None and map_val < 65:
        alerts.append(("critical", f"CRITICAL: Hypotension (MAP {map_val:.1f} < 65 mmHg)."))
    
    lac_val = vitals.get("Lactate")
    if lac_val is not None and lac_val > 2.0:
        alerts.append(("warning", f"WARNING: Elevated Lactate ({lac_val:.1f} > 2.0 mmol/L)."))

    qsofa = results.get("qsofa")
    if qsofa and qsofa.get("score", 0) >= 2:
        alerts.append(("warning", f"WARNING: qSOFA >= 2 ({qsofa['score']}/3). Positive screening for sepsis."))

    if not alerts:
        alerts.append(("info", "INFO: Patient within stable limits. Routine monitoring."))

    for level, msg in alerts:
        st.markdown(f'<div class="icu-alert-{level}">{msg}</div>', unsafe_allow_html=True)
