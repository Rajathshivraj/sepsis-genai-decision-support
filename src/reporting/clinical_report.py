"""
clinical_report.py — Professional PDF clinical report generator with embedded Plotly charts.
"""

from __future__ import annotations

import io
import html
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
import plotly.graph_objects as go

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, 
    Paragraph, 
    Spacer, 
    Table, 
    TableStyle, 
    HRFlowable,
    Image,
    KeepTogether
)


def _hex_rgb(h: str) -> str:
    h = h.lstrip("#")
    return ",".join(str(int(h[i:i+2], 16)) for i in (0, 2, 4))


# ---------------------------------------------------------------------------
# Image Generators using Matplotlib
# ---------------------------------------------------------------------------

def _generate_trajectory_chart(forecast_result: Dict[str, Any]) -> Optional[io.BytesIO]:
    """Generate risk trajectory line chart as a PNG image buffer using matplotlib."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        current = forecast_result.get("current_risk", 0)
        predictions = forecast_result.get("forecast", [])
        if not predictions:
            return None

        hours = [0] + [p["hour"] for p in predictions]
        risks = [current] + [p["risk"] for p in predictions]
        trend = forecast_result.get("trend", "STABLE")
        
        trend_color = {"INCREASING": "#fc8181", "DECREASING": "#68d391", "STABLE": "#63b3ed"}.get(trend, "#63b3ed")

        fig, ax = plt.subplots(figsize=(6, 3.5))
        
        # Risk zones
        ax.axhspan(0.7, 1.0, color="#fc8181", alpha=0.15, label="HIGH RISK")
        ax.axhspan(0.4, 0.7, color="#f6ad55", alpha=0.10, label="MODERATE RISK")

        ax.plot(hours, risks, marker='o', color=trend_color, linewidth=2, markersize=6)
        ax.plot([0], [current], marker='D', color="#63b3ed", markersize=10, label="Now")
        
        ax.set_ylim([0, 1.0])
        ax.set_ylabel("Sepsis Risk", fontsize=10, color="#4a5568")
        ax.set_xlabel("Hours from Now", fontsize=10, color="#4a5568")
        
        # Cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cbd5e0')
        ax.spines['bottom'].set_color('#cbd5e0')
        ax.tick_params(colors='#4a5568')
        ax.grid(axis='y', linestyle='--', alpha=0.5, color='#cbd5e0')
        
        plt.tight_layout()
        
        img_bytes = io.BytesIO()
        plt.savefig(img_bytes, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        img_bytes.seek(0)
        return img_bytes
    except Exception:
        return None


def _generate_twin_chart(twin_result: Dict[str, Any]) -> Optional[io.BytesIO]:
    """Generate digital twin comparison chart as a PNG image buffer using matplotlib."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        fig, ax = plt.subplots(figsize=(6, 3.5))
        _COLORS = {
            "baseline": "#fc8181",
            "fluid_resuscitation": "#63b3ed",
            "early_antibiotics": "#68d391",
        }
        
        has_data = False
        for key, entry in twin_result.items():
            if key == "summary": 
                continue
            forecast = entry.get("forecast", {})
            predictions = forecast.get("forecast", [])
            current = forecast.get("current_risk", 0)
            if not predictions: 
                continue
            
            has_data = True
            hours = [0] + [p["hour"] for p in predictions]
            risks = [current] + [p["risk"] for p in predictions]
            color = _COLORS.get(key, "#a0aec0")
            label = entry.get("label", key)
            
            ax.plot(hours, risks, marker='o', color=color, linewidth=2, markersize=5, label=label)
            
        if not has_data: 
            return None

        # Risk zones
        ax.axhspan(0.7, 1.0, color="#fc8181", alpha=0.1)
        ax.axhspan(0.4, 0.7, color="#f6ad55", alpha=0.05)
        
        ax.set_ylim([0, 1.0])
        ax.set_ylabel("Sepsis Risk", fontsize=10, color="#4a5568")
        ax.set_xlabel("Hours from Now", fontsize=10, color="#4a5568")
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
        
        # Cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cbd5e0')
        ax.spines['bottom'].set_color('#cbd5e0')
        ax.tick_params(colors='#4a5568')
        ax.grid(axis='y', linestyle='--', alpha=0.5, color='#cbd5e0')
        
        plt.tight_layout()
        
        img_bytes = io.BytesIO()
        plt.savefig(img_bytes, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        img_bytes.seek(0)
        return img_bytes
    except Exception:
        return None


def _generate_shap_chart(shap_features: List[Dict[str, Any]]) -> Optional[io.BytesIO]:
    """Generate SHAP feature importance bar chart as a PNG image buffer using matplotlib."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        top_features = shap_features[:8]
        if not top_features: 
            return None
        
        names = [f["feature"] for f in top_features][::-1]
        vals = [f["shap_value"] for f in top_features][::-1]
        colors = ["#fc8181" if v > 0 else "#63b3ed" for v in vals]
        
        fig, ax = plt.subplots(figsize=(6, 3.5))
        
        ax.barh(names, vals, color=colors)
        
        ax.set_xlabel("SHAP Value (Impact on Risk)", fontsize=10, color="#4a5568")
        
        # Cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cbd5e0')
        ax.spines['bottom'].set_color('#cbd5e0')
        ax.tick_params(colors='#4a5568')
        ax.grid(axis='x', linestyle='--', alpha=0.5, color='#cbd5e0')
        
        plt.tight_layout()
        
        img_bytes = io.BytesIO()
        plt.savefig(img_bytes, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        img_bytes.seek(0)
        return img_bytes
    except Exception:
        return None


# ---------------------------------------------------------------------------
# PDF Report Generator
# ---------------------------------------------------------------------------

def generate_clinical_report(
    patient_data: Dict[str, Any],
    ml_score: float,
    lstm_score: float,
    ensemble_score: float,
    agreement: str,
    uncertainty: float,
    retrieved_cases: List[Dict[str, Any]],
    llm_reasoning: str,
    confidence: str,
    qsofa_result: Optional[Dict[str, Any]] = None,
    sofa_result: Optional[Dict[str, Any]] = None,
    shap_features: Optional[List[Dict[str, Any]]] = None,
    trajectory_summary: Optional[Dict[str, Any]] = None,
    reliability: Optional[Dict[str, Any]] = None,
    forecast_result: Optional[Dict[str, Any]] = None,
    twin_result: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> bytes:
    """Generate a structured, professional clinical PDF report with charts."""
    
    buffer = io.BytesIO()

    # Full page width layout
    doc = SimpleDocTemplate(
        buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Title"], fontSize=20, 
        textColor=colors.HexColor("#1A365D"), alignment=1, spaceAfter=8
    )
    
    subtitle_style = ParagraphStyle(
        "ReportSubtitle", parent=styles["Normal"], fontSize=11, 
        textColor=colors.HexColor("#4A5568"), alignment=1, spaceAfter=4
    )
    
    heading_style = ParagraphStyle(
        "ReportHeading", parent=styles["Heading2"], fontSize=14, 
        textColor=colors.HexColor("#2B6CB0"), spaceBefore=18, spaceAfter=10
    )
    
    normal_style = styles["Normal"]
    
    reasoning_style = ParagraphStyle(
        "Reasoning", parent=styles["Normal"], fontSize=10, leading=14, 
        backColor=colors.HexColor("#F7FAFC"), borderColor=colors.HexColor("#E2E8F0"), 
        borderWidth=1, borderPadding=12, spaceAfter=10
    )
    
    disclaimer_style = ParagraphStyle(
        "Disclaimer", parent=styles["Normal"], fontSize=8, 
        textColor=colors.HexColor("#718096"), alignment=1, spaceBefore=20
    )

    def make_table(data: List[List[Any]], col_widths: List[int]) -> Table:
        t = Table(data, colWidths=col_widths)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1A202C")),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
        ]))
        return t

    story = []

    # ── 1. REPORT HEADER ──────────────────────────────────────────────────
    story.append(Paragraph("AI Sepsis Risk Assessment Report", title_style))
    story.append(Paragraph("Generated by Sepsis GenAI Clinical Decision Support System", subtitle_style))
    story.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", subtitle_style))
    story.append(Spacer(1, 15))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2B6CB0"), spaceAfter=20))

    # ── 2. PATIENT CLINICAL DATA ──────────────────────────────────────────
    story.append(Paragraph("1. Patient Clinical Data", heading_style))
    vitals_keys = ["HR", "Temp", "MAP", "Resp", "O2Sat", "Lactate", "Creatinine", "WBC"]
    vitals_data = [["Vital / Lab", "Value"]]
    for k in vitals_keys:
        val = patient_data.get(k)
        val_str = f"{val:.2f}" if isinstance(val, float) else str(val) if val is not None else "N/A"
        vitals_data.append([k, val_str])
    story.append(make_table(vitals_data, [200, 300]))

    # ── 3. MODEL RISK ASSESSMENT ──────────────────────────────────────────
    story.append(Paragraph("2. Model Risk Assessment", heading_style))
    rel_label = reliability.get("reliability_label", "N/A") if reliability else "N/A"
    
    if ensemble_score >= 0.7:
        risk_interp = "HIGH"
    elif ensemble_score >= 0.4:
        risk_interp = "MODERATE"
    else:
        risk_interp = "LOW"
        
    risk_data = [
        ["Metric", "Value"],
        ["ML Risk Score", f"{ml_score:.3f}"],
        ["LSTM Risk Score", f"{lstm_score:.3f}"],
        ["Ensemble Score", f"{ensemble_score:.3f} ({risk_interp})"],
        ["Model Agreement", str(agreement)],
        ["Uncertainty Score", f"{uncertainty:.4f}"],
        ["Clinical Reliability", str(rel_label)]
    ]
    story.append(make_table(risk_data, [200, 300]))

    # ── 4. CLINICAL SCORING SYSTEMS ───────────────────────────────────────
    story.append(Paragraph("3. Clinical Scoring Systems", heading_style))
    if not qsofa_result and not sofa_result:
        story.append(Paragraph("No clinical score available.", normal_style))
    else:
        score_data = [["Scoring System", "Score", "Interpretation"]]
        if qsofa_result:
            score_data.append([
                "qSOFA", f"{qsofa_result.get('score', 'N/A')}/3", 
                str(qsofa_result.get("interpretation", "N/A"))
            ])
        if sofa_result:
            score_data.append([
                "SOFA", f"{sofa_result.get('total_score', 'N/A')}/24", 
                str(sofa_result.get("interpretation", "N/A"))
            ])
        story.append(make_table(score_data, [100, 100, 300]))

    # ── 5. RISK TRAJECTORY FORECAST ───────────────────────────────────────
    story.append(Paragraph("4. Risk Trajectory Forecast", heading_style))
    if forecast_result and "forecast" in forecast_result and forecast_result["forecast"]:
        forecast_list = forecast_result["forecast"]
        traj_data = [["Hour", "Predicted Risk"]]
        for p in forecast_list:
            hr = p.get('hour', 'N/A')
            r = p.get('risk', 0.0)
            traj_data.append([f"t+{hr}h", f"{r:.3f}"])
        story.append(make_table(traj_data, [100, 200]))
        story.append(Spacer(1, 10))
        
        img_io = _generate_trajectory_chart(forecast_result)
        if img_io:
            story.append(KeepTogether([Image(img_io, width=350, height=210)]))
    else:
        story.append(Paragraph("No trajectory data available.", normal_style))

    # ── 6. DIGITAL TWIN SIMULATION ────────────────────────────────────────
    story.append(Paragraph("5. Digital Twin Simulation", heading_style))
    if twin_result and "summary" in twin_result and "comparisons" in twin_result["summary"]:
        comps = twin_result["summary"]["comparisons"]
        twin_data = [["Intervention", "Projected Risk"]]
        for k in ["baseline", "fluid_resuscitation", "early_antibiotics"]:
            if k in comps:
                lbl = comps[k].get("label", k)
                risk = comps[k].get("endpoint_risk", "N/A")
                if isinstance(risk, float): risk = f"{risk:.3f}"
                twin_data.append([str(lbl), str(risk)])
        
        if len(twin_data) == 1:
            story.append(Paragraph("No simulation data available.", normal_style))
        else:
            story.append(make_table(twin_data, [250, 150]))
            story.append(Spacer(1, 10))
            
            img_io = _generate_twin_chart(twin_result)
            if img_io:
                story.append(KeepTogether([Image(img_io, width=400, height=240)]))
    else:
        story.append(Paragraph("No simulation data available.", normal_style))

    # ── 7. FEATURE IMPORTANCE ─────────────────────────────────────────────
    story.append(Paragraph("6. Feature Importance", heading_style))
    if not shap_features:
        story.append(Paragraph("No feature importance data available.", normal_style))
    else:
        shap_data = [["Feature", "Impact"]]
        for f in shap_features[:5]:
            feat = f.get("feature", "N/A")
            val = f.get("shap_value", 0.0)
            direction = f.get("direction", "")
            sgn = "+" if val >= 0 else ""
            shap_data.append([str(feat), f"{sgn}{val:.4f} ({direction})"])
        story.append(make_table(shap_data, [150, 200]))
        story.append(Spacer(1, 10))
        
        img_io = _generate_shap_chart(shap_features)
        if img_io:
            story.append(KeepTogether([Image(img_io, width=350, height=210)]))

    # ── 8. SIMILAR ICU CASES ──────────────────────────────────────────────
    story.append(Paragraph("7. Similar ICU Cases", heading_style))
    if not retrieved_cases:
        story.append(Paragraph("No similar cases available.", normal_style))
    else:
        cases_data = [["Patient ID", "Outcome", "Similarity Score"]]
        for case in retrieved_cases[:5]:
            pid = case.get("patient_id", "N/A")
            sim = case.get("score", case.get("similarity", 0.0))
            sim_str = f"{sim:.3f}" if isinstance(sim, float) else str(sim)
            text = str(case.get("case", ""))
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            outcome = "Unknown"
            for ln in lines:
                if "Outcome" in ln:
                    outcome = ln.replace("Outcome:", "").replace("Outcome", "").strip()
                    break
            cases_data.append([str(pid), outcome, sim_str])
        story.append(make_table(cases_data, [100, 300, 100]))

    # ── 9. AI CLINICAL REASONING ──────────────────────────────────────────
    story.append(Paragraph("8. AI Clinical Reasoning", heading_style))
    if not llm_reasoning or str(llm_reasoning).strip() == "":
        story.append(Paragraph("No clinical reasoning available.", normal_style))
    else:
        safe_text = html.escape(str(llm_reasoning)).replace("\n", "<br/>")
        story.append(Paragraph(safe_text, reasoning_style))

    # ── 10. SYSTEM DISCLAIMER ─────────────────────────────────────────────
    story.append(Spacer(1, 30))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E0"), spaceAfter=10))
    disclaimer = "This AI-generated report is intended for clinical decision support only and must not replace physician judgement."
    story.append(Paragraph(disclaimer, disclaimer_style))

    doc.build(story)
    return buffer.getvalue()
