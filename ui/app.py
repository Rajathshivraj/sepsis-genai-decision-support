"""
app.py — Interactive Streamlit Demo UI for the Sepsis GenAI Decision Support System.

This application provides a clinical dashboard for sepsis risk prediction
combining Machine Learning, LSTM time-series analysis, Retrieval-Augmented
Generation (RAG), and LLM-based clinical reasoning.

Sections
--------
* Patient Input      — Manual entry / CSV upload / PDF upload
* Risk Prediction    — ML and LSTM risk scores with metric cards
* Model Agreement    — Uncertainty calibration panel
* Retrieved Cases    — Top-k similar historical cases from FAISS
* AI Reasoning       — LLM-generated clinical narrative
* Final Decision     — Colour-coded risk classification

Usage
-----
    cd /home/rajat/projects/sepsis-genai
    streamlit run ui/app.py
"""

from __future__ import annotations

import sys
import io
import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Ensure project root is importable when launched from ui/
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Sepsis AI Decision Support",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — clinical dashboard aesthetic
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Main background */
    .stApp { background-color: #0e1117; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a1f2e 0%, #212736 100%);
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 16px 20px;
    }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #63b3ed;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #2d3748;
    }

    /* Risk badge containers */
    .risk-high   { background:#742a2a; border:1px solid #c53030; border-radius:8px; padding:16px; text-align:center; }
    .risk-medium { background:#744210; border:1px solid #c05621; border-radius:8px; padding:16px; text-align:center; }
    .risk-low    { background:#1c4532; border:1px solid #276749; border-radius:8px; padding:16px; text-align:center; }

    /* Case cards */
    .case-card {
        background: #1a1f2e;
        border-left: 4px solid #4299e1;
        border-radius: 4px;
        padding: 12px 16px;
        margin-bottom: 10px;
        font-size: 0.85rem;
        line-height: 1.6;
    }

    /* Reasoning box */
    .reasoning-box {
        background: #1a202c;
        border: 1px solid #4a5568;
        border-radius: 8px;
        padding: 20px;
        font-style: italic;
        line-height: 1.8;
        color: #e2e8f0;
    }

    /* Scrollable case list */
    .case-scroll { max-height: 400px; overflow-y: auto; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ===========================================================================
# Lazy imports (wrapped in try/except for graceful degradation)
# ===========================================================================

def _import_src_modules():
    """Return a dict of imported src modules, or None entries on failure."""
    mods: Dict[str, Any] = {}

    try:
        from src.preprocessing.feature_engineering import create_ml_features
        mods["create_ml_features"] = create_ml_features
    except ImportError as e:
        mods["create_ml_features"] = None
        st.session_state.setdefault("import_errors", []).append(str(e))

    try:
        from src.models.lstm_model import prepare_lstm_sequences, SepsisLSTM
        mods["prepare_lstm_sequences"] = prepare_lstm_sequences
        mods["SepsisLSTM"] = SepsisLSTM
    except ImportError as e:
        mods["prepare_lstm_sequences"] = None
        st.session_state.setdefault("import_errors", []).append(str(e))

    try:
        from src.rag.case_builder import build_patient_case, build_all_patient_cases
        mods["build_patient_case"] = build_patient_case
        mods["build_all_patient_cases"] = build_all_patient_cases
    except ImportError as e:
        mods["build_patient_case"] = None
        st.session_state.setdefault("import_errors", []).append(str(e))

    try:
        from src.rag.vector_store import build_vector_index, retrieve_similar_cases
        mods["build_vector_index"] = build_vector_index
        mods["retrieve_similar_cases"] = retrieve_similar_cases
    except ImportError as e:
        mods["build_vector_index"] = None
        st.session_state.setdefault("import_errors", []).append(str(e))

    try:
        from src.llm.reasoner import generate_clinical_reasoning
        mods["generate_clinical_reasoning"] = generate_clinical_reasoning
    except ImportError as e:
        mods["generate_clinical_reasoning"] = None
        st.session_state.setdefault("import_errors", []).append(str(e))

    try:
        from src.utils.uncertainty import compute_model_agreement, compute_uncertainty
        mods["compute_model_agreement"] = compute_model_agreement
        mods["compute_uncertainty"] = compute_uncertainty
    except ImportError as e:
        mods["compute_model_agreement"] = None
        st.session_state.setdefault("import_errors", []).append(str(e))

    # ── New module imports (Phase 1–10 upgrades) ──────────────────────────
    _optional_imports = [
        ("src.ensemble.risk_ensemble", ["compute_ensemble_score"]),
        ("src.uncertainty.uncertainty_estimator", ["compute_clinical_reliability"]),
        ("src.temporal.risk_trajectory", ["compute_risk_trajectory", "compute_risk_trend"]),
        ("src.rag.advanced_rag", ["retrieve_with_guidelines", "get_relevant_guidelines"]),
        ("src.llm.grounded_reasoner", ["generate_grounded_reasoning"]),
        ("src.clinical_scores.qsofa", ["compute_qsofa"]),
        ("src.clinical_scores.sofa", ["compute_sofa"]),
        ("src.visualization.charts", ["create_risk_gauge", "create_risk_timeline", "create_feature_importance_chart"]),
        ("src.explainability.shap_explainer", ["compute_shap_values", "get_top_risk_drivers"]),
        ("src.reporting.clinical_report", ["generate_clinical_report"]),
    ]
    for mod_path, names in _optional_imports:
        try:
            mod = __import__(mod_path, fromlist=names)
            for name in names:
                mods[name] = getattr(mod, name)
        except Exception as e:
            for name in names:
                mods[name] = None
            st.session_state.setdefault("import_errors", []).append(f"{mod_path}: {e}")

    return mods


# ===========================================================================
# ML scoring helper (XGBoost, trained on-the-fly if dataset available)
# ===========================================================================

@st.cache_resource(show_spinner="Training ML model on PhysioNet dataset…")
def _load_ml_model():
    """Train a lightweight XGBoost classifier on PhysioNet data (cached)."""
    try:
        from src.preprocessing.dataset_loader import load_all_patients
        from src.preprocessing.feature_engineering import create_ml_features
        from sklearn.preprocessing import StandardScaler
        from xgboost import XGBClassifier

        df = load_all_patients("A", max_patients=300)
        if df.empty:
            return None, None, None

        X, y = create_ml_features(df)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, verbosity=0,
        )
        model.fit(X_scaled, y)
        return model, scaler, list(X.columns)
    except Exception:
        return None, None, None


def _heuristic_ml_score(vitals: Dict[str, float]) -> float:
    """Fallback heuristic risk score when model not available."""
    score = 0.0
    hr = vitals.get("HR", 80)
    map_val = vitals.get("MAP", 80)
    lactate = vitals.get("Lactate", 1.0)
    resp = vitals.get("Resp", 16)
    wbc = vitals.get("WBC", 8)

    if hr > 100:   score += 0.15
    if hr > 120:   score += 0.10
    if map_val < 70: score += 0.20
    if map_val < 60: score += 0.15
    if lactate > 2: score += 0.20
    if lactate > 4: score += 0.15
    if resp > 20:  score += 0.10
    if wbc > 12 or wbc < 4: score += 0.10
    return min(1.0, round(score, 4))


def _predict_ml_score(vitals: Dict[str, float]) -> Tuple[float, str]:
    """Return (ml_score, source) where source is 'model' or 'heuristic'."""
    try:
        model, scaler, feat_cols = _load_ml_model()
        if model is None or scaler is None:
            return _heuristic_ml_score(vitals), "heuristic"

        # Build a minimal feature row aligned to training columns
        row: Dict[str, float] = {}
        col_map = {
            "HR_mean": vitals.get("HR", 80), "HR_max": vitals.get("HR", 80),
            "HR_min": vitals.get("HR", 80),
            "MAP_mean": vitals.get("MAP", 80), "MAP_min": vitals.get("MAP", 80),
            "Temp_mean": vitals.get("Temp", 37.0),
            "Resp_mean": vitals.get("Resp", 16),
            "Lactate_max": vitals.get("Lactate", 1.0),
            "Creatinine_max": vitals.get("Creatinine", 1.0),
            "ICULOS_max": vitals.get("ICULOS", 12),
        }
        for col in feat_cols:
            row[col] = col_map.get(col, 0.0)

        X_new = pd.DataFrame([row])[feat_cols]
        X_scaled = scaler.transform(X_new)
        prob = float(model.predict_proba(X_scaled)[0, 1])
        return round(prob, 4), "model"
    except Exception:
        return _heuristic_ml_score(vitals), "heuristic"


def _lstm_score_from_sequence(df_seq: pd.DataFrame) -> float:
    """Derive a naive LSTM-like score from a time-series DataFrame."""
    try:
        # Use last-row vitals with time-trend modifiers
        hr_vals = df_seq["HR"].dropna()
        map_vals = df_seq["MAP"].dropna()
        lac_vals = df_seq["Lactate"].dropna() if "Lactate" in df_seq else pd.Series()

        score = 0.0
        if len(hr_vals) >= 2:
            trend = (hr_vals.iloc[-1] - hr_vals.iloc[0]) / max(len(hr_vals), 1)
            if trend > 2: score += 0.20
        if len(map_vals) >= 2:
            trend = (map_vals.iloc[-1] - map_vals.iloc[0]) / max(len(map_vals), 1)
            if trend < -1: score += 0.20
        if len(lac_vals) >= 2:
            trend = (lac_vals.iloc[-1] - lac_vals.iloc[0]) / max(len(lac_vals), 1)
            if trend > 0.2: score += 0.20
        last_vitals = {
            "HR": float(hr_vals.iloc[-1]) if len(hr_vals) else 80,
            "MAP": float(map_vals.iloc[-1]) if len(map_vals) else 80,
            "Lactate": float(lac_vals.iloc[-1]) if len(lac_vals) else 1.0,
            "Resp": float(df_seq["Resp"].dropna().iloc[-1]) if "Resp" in df_seq and len(df_seq["Resp"].dropna()) else 16,
        }
        score += _heuristic_ml_score(last_vitals) * 0.5
        return min(1.0, round(score, 4))
    except Exception:
        return 0.5


# ===========================================================================
# RAG helpers
# ===========================================================================

def _build_demo_cases(patient_df: pd.DataFrame, n_cases: int = 8) -> Dict[str, str]:
    """Build a small in-memory vector store from synthetic historical cases."""
    templates = [
        ("sim_001", "Patient summary:\nPatient ID: sim_001\nICU stay duration: 24 hours\n\nVital signs:\n  Heart rate increased from 90 bpm to 125 bpm (peak: 132 bpm, low: 88 bpm).\n  Mean arterial pressure decreased from 85 to 58 mmHg (peak: 85.0 mmHg, low: 52.0 mmHg).\n  Temperature stable from 37.2 to 38.6 °C.\n  Respiratory rate increased from 16 to 28 breaths/min.\n\nKey lab values:\n  Lactate: peak 4.20 mmol/L, mean 2.80 mmol/L.\n  Creatinine: peak 1.80 mg/dL, mean 1.40 mg/dL.\n  White blood cell count: peak 14.50 ×10³/µL, mean 12.00 ×10³/µL.\n\nOutcome: Sepsis diagnosed at hour 18."),
        ("sim_002", "Patient summary:\nPatient ID: sim_002\nICU stay duration: 18 hours\n\nVital signs:\n  Heart rate stable from 78 bpm to 82 bpm.\n  Mean arterial pressure stable from 88 to 85 mmHg.\n  Temperature stable at 37.1 °C.\n  Respiratory rate stable from 14 to 16 breaths/min.\n\nKey lab values:\n  Lactate: peak 1.10 mmol/L, mean 0.90 mmol/L.\n  Creatinine: peak 0.90 mg/dL, mean 0.80 mg/dL.\n  White blood cell count: peak 8.20 ×10³/µL, mean 7.50 ×10³/µL.\n\nOutcome: No sepsis diagnosed during ICU stay."),
        ("sim_003", "Patient summary:\nPatient ID: sim_003\nICU stay duration: 30 hours\n\nVital signs:\n  Heart rate increased from 95 bpm to 118 bpm (peak: 130 bpm).\n  Mean arterial pressure dropped from 78 to 62 mmHg.\n  Temperature increased from 37.5 to 39.1 °C.\n  Respiratory rate increased from 18 to 24 breaths/min.\n\nKey lab values:\n  Lactate: peak 3.20 mmol/L, mean 2.10 mmol/L.\n  Creatinine: peak 1.50 mg/dL, mean 1.20 mg/dL.\n  White blood cell count: peak 16.00 ×10³/µL, mean 13.50 ×10³/µL.\n\nOutcome: Sepsis diagnosed at hour 22."),
        ("sim_004", "Patient summary:\nPatient ID: sim_004\nICU stay duration: 12 hours\n\nVital signs:\n  Heart rate increased from 85 bpm to 110 bpm.\n  Mean arterial pressure decreased from 74 to 65 mmHg.\n  Temperature stable at 37.8 °C.\n  Respiratory rate increased from 17 to 22 breaths/min.\n\nKey lab values:\n  Lactate: peak 2.50 mmol/L, mean 1.80 mmol/L.\n  Creatinine: peak 1.20 mg/dL, mean 1.05 mg/dL.\n  White blood cell count: peak 11.00 ×10³/µL, mean 9.80 ×10³/µL.\n\nOutcome: Sepsis diagnosed at hour 10."),
        ("sim_005", "Patient summary:\nPatient ID: sim_005\nICU stay duration: 20 hours\n\nVital signs:\n  Heart rate stable from 72 bpm to 76 bpm.\n  Mean arterial pressure stable from 90 to 92 mmHg.\n  Temperature stable at 36.9 °C.\n  Respiratory rate stable at 14 breaths/min.\n\nKey lab values:\n  Lactate: peak 0.80 mmol/L, mean 0.75 mmol/L.\n  Creatinine: peak 0.70 mg/dL, mean 0.68 mg/dL.\n  White blood cell count: peak 6.80 ×10³/µL, mean 6.20 ×10³/µL.\n\nOutcome: No sepsis diagnosed during ICU stay."),
    ]
    # Add patient-specific case if available
    cases = {pid: text for pid, text in templates}
    try:
        from src.rag.case_builder import build_patient_case
        if "patient_id" in patient_df.columns:
            for pid, pdf in patient_df.groupby("patient_id"):
                cases[pid] = build_patient_case(pdf)
    except Exception:
        pass
    return cases


def _retrieve_cases_for_query(
    query_text: str,
    cases: Dict[str, str],
    k: int = 5,
) -> List[Dict[str, Any]]:
    """Retrieve top-k similar cases, falling back to keyword ranking."""
    try:
        from src.rag.vector_store import build_vector_index, retrieve_similar_cases
        store = build_vector_index(cases)
        return retrieve_similar_cases(query_text, k=k, store=store)
    except Exception:
        # Simple keyword fallback: rank by shared word count
        query_words = set(query_text.lower().split())
        scored = []
        for pid, text in cases.items():
            overlap = len(query_words & set(text.lower().split()))
            scored.append({"patient_id": pid, "score": overlap / max(len(query_words), 1), "case": text})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:k]


# ===========================================================================
# PDF extraction helper
# ===========================================================================

def _extract_labs_from_pdf(pdf_bytes: bytes) -> Dict[str, float]:
    """Extract lab values from a PDF using pdfplumber + LLM, or heuristics."""
    text = ""
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception:
        text = pdf_bytes.decode("utf-8", errors="ignore")

    # Try LLM extraction
    try:
        from src.llm.reasoner import _call_ollama
        prompt = (
            "Extract the following lab values from the blood report text below. "
            "Return ONLY a JSON object with keys 'Lactate', 'WBC', 'Creatinine', 'HR', 'MAP'. "
            "Use null if not found.\n\nReport text:\n" + text[:2000]
        )
        raw = _call_ollama(prompt)
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1:
            result = json.loads(raw[start:end+1])
            labs: Dict[str, float] = {}
            for k, v in result.items():
                if v is not None:
                    try:
                        labs[k] = float(v)
                    except (ValueError, TypeError):
                        pass
            if labs:
                return labs
    except Exception:
        pass

    # Regex fallback
    import re
    labs = {}
    patterns = {
        "Lactate": r"lactate[\s:]+([0-9]+\.?[0-9]*)",
        "WBC": r"(?:wbc|white\s+blood\s+cell)[\s:]+([0-9]+\.?[0-9]*)",
        "Creatinine": r"creatinine[\s:]+([0-9]+\.?[0-9]*)",
        "HR": r"(?:heart\s+rate|hr)[\s:]+([0-9]+)",
        "MAP": r"(?:map|mean\s+arterial\s+pressure)[\s:]+([0-9]+\.?[0-9]*)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                labs[key] = float(m.group(1))
            except ValueError:
                pass
    return labs


# ===========================================================================
# Main UI
# ===========================================================================

def _render_header() -> None:
    """Render the application title and description."""
    col_icon, col_title = st.columns([1, 10])
    with col_icon:
        st.markdown("# 🏥")
    with col_title:
        st.markdown("# Sepsis AI Decision Support System")
        st.markdown(
            "*An AI-powered clinical dashboard integrating **Machine Learning**, "
            "**LSTM Time-Series Analysis**, **Retrieval-Augmented Generation**, "
            "and **LLM Clinical Reasoning** for early sepsis detection.*"
        )
    st.divider()


def _render_sidebar() -> Dict[str, Any]:
    """Render sidebar configuration and return settings."""
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.markdown("---")

        top_k = st.slider("Similar Cases (Top-K)", min_value=1, max_value=10, value=5)

        st.markdown("---")
        st.markdown("### 🔗 Service Status")

        # Check Ollama availability
        try:
            import requests
            resp = requests.get("http://localhost:11434/api/tags", timeout=2)
            if resp.status_code == 200:
                st.success("🟢 Ollama LLM — Online")
            else:
                st.warning("🟡 Ollama LLM — Degraded")
        except Exception:
            st.error("🔴 Ollama LLM — Offline (fallback active)")

        st.markdown("---")
        st.markdown(
            "<small>📌 Sepsis GenAI v1.0 · PhysioNet 2019</small>",
            unsafe_allow_html=True,
        )

    return {"top_k": top_k}


def _render_manual_tab(mods: Dict[str, Any], cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Render Tab 1: Manual patient data entry form."""
    st.markdown('<div class="section-header">📋 Patient Vital Signs & Lab Values</div>', unsafe_allow_html=True)

    with st.form("manual_patient_form"):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            hr = st.number_input("Heart Rate (bpm)", min_value=20, max_value=250, value=95, step=1)
            o2sat = st.number_input("O₂ Saturation (%)", min_value=50.0, max_value=100.0, value=97.0, step=0.1)

        with col2:
            temp = st.number_input("Temperature (°C)", min_value=32.0, max_value=42.0, value=37.4, step=0.1)
            map_val = st.number_input("Mean Arterial Pressure (mmHg)", min_value=20, max_value=200, value=75, step=1)

        with col3:
            resp = st.number_input("Respiratory Rate (breaths/min)", min_value=5, max_value=60, value=18, step=1)
            lactate = st.number_input("Lactate (mmol/L)", min_value=0.1, max_value=20.0, value=1.8, step=0.1)

        with col4:
            creatinine = st.number_input("Creatinine (mg/dL)", min_value=0.1, max_value=15.0, value=1.0, step=0.1)
            wbc = st.number_input("WBC (×10³/µL)", min_value=0.1, max_value=50.0, value=9.5, step=0.1)

        submitted = st.form_submit_button("🔍 Analyze Patient", use_container_width=True, type="primary")

    if not submitted:
        return None

    vitals = {
        "HR": hr, "O2Sat": o2sat, "Temp": temp, "MAP": map_val,
        "Resp": resp, "Lactate": lactate, "Creatinine": creatinine, "WBC": wbc,
        "ICULOS": 12,
    }

    # Build synthetic patient DataFrame (single time-point)
    df_patient = pd.DataFrame([vitals])
    df_patient["patient_id"] = "manual_input"
    df_patient["SepsisLabel"] = 0

    return {"vitals": vitals, "df": df_patient, "source": "manual"}


def _render_csv_tab() -> Optional[Dict[str, Any]]:
    """Render Tab 2: CSV ICU data upload."""
    st.markdown('<div class="section-header">📂 Upload ICU Time-Series Data (CSV)</div>', unsafe_allow_html=True)

    st.markdown(
        "Upload a CSV file with hourly ICU readings. "
        "Expected columns: `HR, O2Sat, Temp, MAP, Resp, Lactate` (and optionally `Creatinine, WBC`)."
    )

    sample = "HR,O2Sat,Temp,MAP,Resp,Lactate,Creatinine,WBC\n95,98,37.4,75,18,1.8,1.0,9.5\n105,97,37.9,68,20,2.5,1.2,11.0\n118,96,38.3,63,24,3.2,1.5,13.5\n"
    with st.expander("📄 View expected CSV format"):
        st.code(sample, language="csv")
        st.download_button("⬇️ Download Sample CSV", sample, file_name="sample_icu_data.csv", mime="text/csv")

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"], key="csv_uploader")
    if uploaded is None:
        return None

    try:
        df = pd.read_csv(uploaded)
        df["patient_id"] = "uploaded_patient"
        df["SepsisLabel"] = 0
        if "ICULOS" not in df.columns:
            df["ICULOS"] = range(1, len(df) + 1)

        st.markdown("**Preview (first 10 rows):**")
        st.dataframe(df.head(10), use_container_width=True)
        st.caption(f"Loaded {len(df)} rows × {len(df.columns)} columns")

        if st.button("🔍 Analyze Uploaded Data", type="primary", key="csv_analyze"):
            # Extract last-row vitals as the "current" snapshot
            latest = df.iloc[-1]
            vitals = {
                col: float(latest[col]) for col in
                ["HR", "O2Sat", "Temp", "MAP", "Resp", "Lactate", "Creatinine", "WBC"]
                if col in latest
            }
            vitals.setdefault("HR", 80); vitals.setdefault("MAP", 80)
            vitals.setdefault("Lactate", 1.0); vitals.setdefault("Resp", 16)
            vitals["ICULOS"] = len(df)

            return {"vitals": vitals, "df": df, "source": "csv"}
    except Exception as e:
        st.error(f"Failed to parse CSV: {e}")

    return None


def _render_pdf_tab() -> Optional[Dict[str, Any]]:
    """Render Tab 3: Blood report PDF upload."""
    st.markdown('<div class="section-header">📑 Upload Blood Report PDF</div>', unsafe_allow_html=True)
    st.markdown(
        "Upload a PDF blood report. The system will use the LLM to extract "
        "lab values (`Lactate`, `WBC`, `Creatinine`, `HR`, `MAP`)."
    )

    uploaded = st.file_uploader("Choose a PDF file", type=["pdf"], key="pdf_uploader")
    if uploaded is None:
        return None

    pdf_bytes = uploaded.read()

    with st.spinner("🔬 Extracting lab values from report…"):
        labs = _extract_labs_from_pdf(pdf_bytes)

    if not labs:
        st.warning("⚠️ Could not extract lab values. Please enter them manually below.")
        labs = {}

    st.markdown("**Extracted values (edit if needed):**")

    col1, col2, col3 = st.columns(3)
    with col1:
        lactate = st.number_input("Lactate (mmol/L)", value=float(labs.get("Lactate", 1.5)), step=0.1, key="pdf_lac")
        hr = st.number_input("Heart Rate (bpm)", value=int(labs.get("HR", 90)), step=1, key="pdf_hr")
    with col2:
        wbc = st.number_input("WBC (×10³/µL)", value=float(labs.get("WBC", 9.0)), step=0.1, key="pdf_wbc")
        map_val = st.number_input("MAP (mmHg)", value=float(labs.get("MAP", 80)), step=1.0, key="pdf_map")
    with col3:
        creatinine = st.number_input("Creatinine (mg/dL)", value=float(labs.get("Creatinine", 1.0)), step=0.1, key="pdf_cr")

    if st.button("🔍 Analyze Extracted Data", type="primary", key="pdf_analyze"):
        vitals = {
            "HR": hr, "O2Sat": 97.0, "Temp": 37.5, "MAP": map_val,
            "Resp": 18, "Lactate": lactate, "Creatinine": creatinine, "WBC": wbc,
            "ICULOS": 12,
        }
        df_patient = pd.DataFrame([vitals])
        df_patient["patient_id"] = "pdf_input"
        df_patient["SepsisLabel"] = 0
        return {"vitals": vitals, "df": df_patient, "source": "pdf"}

    return None


def _run_pipeline(
    patient_data: Dict[str, Any],
    top_k: int,
) -> Dict[str, Any]:
    """Execute the complete inference pipeline and return all results."""
    mods = _import_src_modules()
    results: Dict[str, Any] = {}
    vitals = patient_data["vitals"]
    df = patient_data["df"]

    # ── ML score ─────────────────────────────────────────────────────────
    with st.spinner("🤖 Running ML risk model…"):
        ml_score, ml_source = _predict_ml_score(vitals)
        results["ml_score"] = ml_score
        results["ml_source"] = ml_source

    # ── LSTM score ────────────────────────────────────────────────────────
    with st.spinner("📈 Running LSTM time-series analysis…"):
        results["lstm_score"] = _lstm_score_from_sequence(df)

    # ── Ensemble score ────────────────────────────────────────────────────
    try:
        from src.ensemble.risk_ensemble import compute_ensemble_score
        results["ensemble_score"] = compute_ensemble_score(ml_score, results["lstm_score"])
    except Exception:
        results["ensemble_score"] = round((0.6 * ml_score + 0.4 * results["lstm_score"]), 4)

    # ── Model agreement & uncertainty ────────────────────────────────────
    try:
        from src.utils.uncertainty import compute_model_agreement, compute_uncertainty
        results["agreement"] = compute_model_agreement(ml_score, results["lstm_score"])
        results["uncertainty"] = compute_uncertainty(ml_score, results["lstm_score"])
    except Exception:
        diff = abs(ml_score - results["lstm_score"])
        results["agreement"] = "HIGH" if diff < 0.05 else ("MEDIUM" if diff < 0.15 else "LOW")
        results["uncertainty"] = round(diff, 4)

    # ── Clinical reliability ──────────────────────────────────────────────
    try:
        from src.uncertainty.uncertainty_estimator import compute_clinical_reliability
        results["reliability"] = compute_clinical_reliability(
            ml_score, results["lstm_score"], results["ensemble_score"]
        )
    except Exception:
        results["reliability"] = None

    # ── Clinical scores (qSOFA / SOFA) ────────────────────────────────────
    try:
        from src.clinical_scores.qsofa import compute_qsofa
        results["qsofa"] = compute_qsofa(
            sbp=vitals.get("SBP"), resp=vitals.get("Resp"),
            gcs=vitals.get("GCS"),
        )
    except Exception:
        results["qsofa"] = None

    try:
        from src.clinical_scores.sofa import compute_sofa
        results["sofa"] = compute_sofa(
            map_val=vitals.get("MAP"),
            creatinine=vitals.get("Creatinine"),
            platelets=vitals.get("Platelets"),
        )
    except Exception:
        results["sofa"] = None

    # ── Risk trajectory & Forecasting ───────────────────────────────────────────
    try:
        from src.temporal.risk_trajectory import compute_risk_trajectory, compute_risk_trend
        results["trajectory"] = compute_risk_trajectory(df)
        results["trajectory_summary"] = compute_risk_trend(results["trajectory"])
    except Exception:
        results["trajectory"] = None
        results["trajectory_summary"] = None

    # ── Forecasting (LSTM autoregressive) ─────────────────────────────────
    try:
        from src.forecasting.risk_forecast import forecast_future_risk
        results["forecast_result"] = forecast_future_risk(df)
    except Exception:
        results["forecast_result"] = None

    # ── Digital Twin ──────────────────────────────────────────────────────
    try:
        from src.digital_twin.risk_twin import simulate_interventions
        results["twin_result"] = simulate_interventions(df)
    except Exception:
        results["twin_result"] = None

    # ── SHAP explainability ───────────────────────────────────────────────
    try:
        from src.explainability.shap_explainer import compute_shap_values, get_top_risk_drivers
        model, scaler, feat_cols = _load_ml_model()
        if model is not None and scaler is not None:
            col_map = {
                "HR_mean": vitals.get("HR", 80), "HR_max": vitals.get("HR", 80),
                "HR_min": vitals.get("HR", 80),
                "MAP_mean": vitals.get("MAP", 80), "MAP_min": vitals.get("MAP", 80),
                "Temp_mean": vitals.get("Temp", 37.0),
                "Resp_mean": vitals.get("Resp", 16),
                "Lactate_max": vitals.get("Lactate", 1.0),
                "Creatinine_max": vitals.get("Creatinine", 1.0),
                "ICULOS_max": vitals.get("ICULOS", 12),
            }
            row = {col: col_map.get(col, 0.0) for col in feat_cols}
            X_shap = pd.DataFrame([row])[feat_cols]
            shap_result = compute_shap_values(X_shap, model=model)
            results["shap_features"] = get_top_risk_drivers(shap_result, n=8)
        else:
            results["shap_features"] = None
    except Exception:
        results["shap_features"] = None

    # ── Case summary ──────────────────────────────────────────────────────
    with st.spinner("📝 Generating clinical case summary…"):
        try:
            from src.rag.case_builder import build_patient_case
            results["case_summary"] = build_patient_case(df)
        except Exception:
            results["case_summary"] = _vitals_to_text_summary(vitals)

    # ── RAG retrieval (advanced) ──────────────────────────────────────────
    with st.spinner("🔍 Retrieving similar cases & guidelines…"):
        demo_cases = _build_demo_cases(df)
        results["retrieved"] = _retrieve_cases_for_query(
            results["case_summary"], demo_cases, k=top_k
        )
        try:
            from src.rag.advanced_rag import retrieve_with_guidelines, get_relevant_guidelines
            results["guidelines"] = get_relevant_guidelines(vitals)
            results["advanced_rag"] = retrieve_with_guidelines(
                results["case_summary"], results["retrieved"],
                k_cases=top_k, k_guidelines=3,
            )
        except Exception:
            results["guidelines"] = []
            results["advanced_rag"] = None

    # ── LLM reasoning (evidence-grounded) ─────────────────────────────────
    with st.spinner("🧠 Generating evidence-grounded reasoning…"):
        try:
            from src.llm.grounded_reasoner import generate_grounded_reasoning
            results["reasoning"] = generate_grounded_reasoning(
                patient_summary=results["case_summary"],
                retrieved_cases=results["retrieved"],
                ml_score=ml_score,
                lstm_score=results["lstm_score"],
                ensemble_score=results["ensemble_score"],
                agreement=results["agreement"],
                clinical_guidelines=results.get("guidelines"),
                qsofa_result=results.get("qsofa"),
                sofa_result=results.get("sofa"),
            )
        except Exception:
            try:
                from src.llm.reasoner import generate_clinical_reasoning
                results["reasoning"] = generate_clinical_reasoning(
                    patient_summary=results["case_summary"],
                    retrieved_cases=results["retrieved"],
                    ml_score=ml_score,
                    lstm_score=results["lstm_score"],
                )
            except Exception as e:
                results["reasoning"] = {
                    "sepsis_risk": _score_to_risk_label(results["ensemble_score"]),
                    "reasoning": f"LLM unavailable ({e}). Scores: ML={ml_score:.2f}, LSTM={results['lstm_score']:.2f}.",
                    "confidence": "LOW (FALLBACK)",
                }

    return results


def _vitals_to_text_summary(vitals: Dict[str, float]) -> str:
    """Build a plain-text patient summary from a vitals dict."""
    lines = [
        "Patient summary:",
        "Patient ID: manual_input",
        "",
        "Vital signs:",
        f"  Heart rate: {vitals.get('HR', 'N/A')} bpm.",
        f"  Mean arterial pressure: {vitals.get('MAP', 'N/A')} mmHg.",
        f"  Temperature: {vitals.get('Temp', 'N/A')} °C.",
        f"  Respiratory rate: {vitals.get('Resp', 'N/A')} breaths/min.",
        f"  Oxygen saturation: {vitals.get('O2Sat', 'N/A')} %.",
        "",
        "Key lab values:",
        f"  Lactate: {vitals.get('Lactate', 'N/A')} mmol/L.",
        f"  Creatinine: {vitals.get('Creatinine', 'N/A')} mg/dL.",
        f"  White blood cell count: {vitals.get('WBC', 'N/A')} ×10³/µL.",
    ]
    return "\n".join(lines)


def _score_to_risk_label(score: float) -> str:
    if score >= 0.7: return "HIGH"
    if score >= 0.4: return "MODERATE"
    return "LOW"


def _render_results(results: Dict[str, Any], df: pd.DataFrame) -> None:
    """Render all result panels in the required clinical layout."""
    
    st.divider()

    # ── RISK OVERVIEW PANEL ──────────────────────────────────────────────
    try:
        from src.ui_components.risk_panel import render_risk_panel
        render_risk_panel(results)
    except Exception as e:
        st.error(f"Error rendering Risk Overview: {e}")

    # ── CLINICAL SCORES PANEL ────────────────────────────────────────────
    # Rendered gracefully inside risk_panel or can be done here.
    # risk_panel already renders qSOFA and SOFA. We rely on it.

    st.divider()

    # ── RISK TRAJECTORY PANEL ────────────────────────────────────────────
    try:
        from src.ui_components.forecast_panel import render_forecast_panel
        render_forecast_panel(results)
    except Exception as e:
        st.error(f"Error rendering Risk Trajectory: {e}")
        
    st.divider()

    # ── DIGITAL TWIN PANEL ───────────────────────────────────────────────
    try:
        from src.ui_components.twin_panel import render_twin_panel
        render_twin_panel(results)
    except Exception as e:
        st.error(f"Error rendering Digital Twin: {e}")

    st.divider()

    # ── EXPLAINABILITY PANEL ─────────────────────────────────────────────
    try:
        from src.ui_components.explainability_panel import render_explainability_panel
        render_explainability_panel(results)
    except Exception as e:
        st.error(f"Error rendering Explainability: {e}")

    st.divider()

    # ── RAG CASES PANEL ──────────────────────────────────────────────────
    try:
        from src.ui_components.rag_panel import render_rag_panel
        render_rag_panel(results)
    except Exception as e:
        st.error(f"Error rendering RAG Cases: {e}")

    st.divider()

    # ── LLM CLINICAL REASONING PANEL ─────────────────────────────────────
    try:
        from src.ui_components.reasoning_panel import render_reasoning_panel
        render_reasoning_panel(results)
    except Exception as e:
        st.error(f"Error rendering Clinical Reasoning: {e}")

    # ── Trend Visualizations (original) ──────────────────────────────────
    st.markdown('<div class="section-header">📊 Clinical Trend Visualizations</div>', unsafe_allow_html=True)
    _render_charts(df)


def _render_charts(df: pd.DataFrame) -> None:
    """Render HR, MAP, and Lactate trend line charts."""
    import numpy as np
    chart_cols_available = [c for c in ["HR", "MAP", "Lactate"] if c in df.columns]
    if not chart_cols_available:
        st.info("No time-series data available for trend charts.")
        return

    if len(df) < 2:
        row = df.iloc[-1]
        rows = []
        for t in range(1, 4):
            rows.append({c: row.get(c, np.nan) * (1 + (t - 2) * 0.02) for c in chart_cols_available})
        plot_df = pd.DataFrame(rows)
        plot_df.index.name = "Hour"
        st.caption("(Single-point input — synthetic trend shown for illustration)")
    else:
        plot_df = df[chart_cols_available].reset_index(drop=True)
        plot_df.index.name = "Hour"

    c1, c2, c3 = st.columns(3)

    if "HR" in plot_df:
        with c1:
            st.markdown("**Heart Rate Trend (bpm)**")
            st.line_chart(plot_df[["HR"]], color=["#fc8181"], height=180)

    if "MAP" in plot_df:
        with c2:
            st.markdown("**Mean Arterial Pressure Trend (mmHg)**")
            st.line_chart(plot_df[["MAP"]], color=["#63b3ed"], height=180)

    if "Lactate" in plot_df:
        with c3:
            st.markdown("**Lactate Trend (mmol/L)**")
            st.line_chart(plot_df[["Lactate"]], color=["#f6ad55"], height=180)


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    """Main Streamlit application entry point."""
    # Sidebar
    sidebar_cfg = _render_sidebar()
    top_k: int = sidebar_cfg["top_k"]

    # Header
    _render_header()

    # Import src modules (lazy, errors collected into session state)
    mods = _import_src_modules()

    # Show any import warnings in a collapsed expander only
    import_errors = st.session_state.get("import_errors", [])
    if import_errors:
        with st.expander("⚠️ Module import warnings (non-critical)", expanded=False):
            for err in import_errors:
                st.code(err, language="text")

    # ── Input Tabs ────────────────────────────────────────────────────────
    st.markdown("## 🩺 Patient Input")
    tab1, tab2, tab3 = st.tabs([
        "📋 Manual Entry",
        "📂 Upload CSV",
        "📑 Upload PDF",
    ])

    patient_data: Optional[Dict[str, Any]] = None

    with tab1:
        patient_data = _render_manual_tab(mods, sidebar_cfg)

    with tab2:
        if patient_data is None:
            patient_data = _render_csv_tab()

    with tab3:
        if patient_data is None:
            patient_data = _render_pdf_tab()

    # ── Run pipeline if input received ───────────────────────────────────
    if patient_data is not None:
        st.markdown("---")
        st.markdown("## 🔬 Analysis Results")
        source_label = {"manual": "Manual Entry", "csv": "CSV Upload", "pdf": "PDF Report"}.get(
            patient_data["source"], patient_data["source"]
        )
        st.caption(f"Input source: **{source_label}**")

        try:
            results = _run_pipeline(patient_data, top_k=top_k)
            _render_results(results, patient_data["df"])

            # ── Download PDF Report (upgraded) ────────────────────────────────
            st.markdown("---")
            st.markdown('<div class="section-header">📄 Download Clinical Report</div>', unsafe_allow_html=True)

            try:
                from src.reporting.clinical_report import generate_clinical_report

                reasoning_dict = results.get("reasoning", {})
                pdf_bytes = generate_clinical_report(
                    patient_data=patient_data["vitals"],
                    ml_score=results.get("ml_score", 0.0),
                    lstm_score=results.get("lstm_score", 0.0),
                    ensemble_score=results.get("ensemble_score", 0.0),
                    agreement=results.get("agreement", "N/A"),
                    uncertainty=results.get("uncertainty", 0.0),
                    retrieved_cases=results.get("retrieved", []),
                    llm_reasoning=reasoning_dict.get("reasoning", "N/A"),
                    confidence=reasoning_dict.get("confidence", "N/A"),
                    qsofa_result=results.get("qsofa"),
                    sofa_result=results.get("sofa"),
                    shap_features=results.get("shap_features"),
                    trajectory_summary=results.get("trajectory_summary"),
                    reliability=results.get("reliability"),
                    forecast_result=results.get("forecast_result"),
                    twin_result=results.get("twin_result"),
                )

                st.download_button(
                    label="📥 Download Full Clinical Report (PDF)",
                    data=pdf_bytes,
                    file_name="sepsis_clinical_report.pdf",
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True,
                )
            except ImportError:
                # Fallback to original report generator
                try:
                    from src.utils.report_generator import generate_sepsis_report
                    reasoning_dict = results.get("reasoning", {})
                    pdf_bytes = generate_sepsis_report(
                        patient_data=patient_data["vitals"],
                        ml_score=results.get("ml_score", 0.0),
                        lstm_score=results.get("lstm_score", 0.0),
                        agreement=results.get("agreement", "N/A"),
                        uncertainty=results.get("uncertainty", 0.0),
                        retrieved_cases=results.get("retrieved", []),
                        llm_reasoning=reasoning_dict.get("reasoning", "N/A"),
                        confidence=reasoning_dict.get("confidence", "N/A"),
                    )
                    st.download_button(
                        label="Download AI Sepsis Report",
                        data=pdf_bytes,
                        file_name="sepsis_report.pdf",
                        mime="application/pdf",
                        type="primary",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.warning(f"Report generation unavailable: {e}")
            except Exception as e:
                st.error(f"Failed to prepare PDF report: {e}")

        except Exception as e:
            st.error(f"⛔ Pipeline error: {e}")
            with st.expander("Error details"):
                import traceback
                st.code(traceback.format_exc(), language="python")

    else:
        # Landing placeholder
        st.markdown("---")
        st.info(
            "👆 Enter patient data in any of the tabs above and click **Analyze Patient** "
            "to run the full AI diagnostic pipeline.",
            icon="ℹ️",
        )
        # Show example output layout
        with st.expander("📋 What will be shown after analysis?"):
            st.markdown(
                """
                | Section | Content |
                |---|---|
                | **Risk Prediction** | ML score, LSTM score, model agreement, uncertainty |
                | **Final Recommendation** | HIGH / MODERATE / LOW risk badge with confidence |
                | **AI Clinical Reasoning** | LLM-generated narrative referencing vitals & trends |
                | **Retrieved Cases** | Top-5 similar historical ICU cases |
                | **Trend Charts** | HR, MAP, Lactate over time |
                """
            )


if __name__ == "__main__":
    main()
