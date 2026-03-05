"""
icu_layout.py — ICU-grade dashboard CSS and layout primitives.
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# ICU Dashboard CSS
# ---------------------------------------------------------------------------

ICU_CSS = """
<style>
/* ── ICU Dashboard Overrides ─────────────────────────────────────────── */

/* Panel containers */
.icu-panel {
    background: linear-gradient(145deg, #0f1419 0%, #171d28 100%);
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

.icu-panel-header {
    font-size: 0.85rem;
    font-weight: 700;
    color: #63b3ed;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(99,179,237,0.2);
}

/* Status bar */
.icu-status-bar {
    background: linear-gradient(90deg, #1a202c 0%, #171d28 100%);
    border: 1px solid #2d3748;
    border-radius: 8px;
    padding: 12px 20px;
    display: flex;
    align-items: center;
    gap: 24px;
    margin-bottom: 8px;
}

.icu-status-item {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.82rem;
    color: #a0aec0;
}

.icu-status-value {
    font-weight: 700;
    color: #e2e8f0;
}

/* Alert banners */
.icu-alert-critical {
    background: linear-gradient(135deg, #742a2a 0%, #5c1a1a 100%);
    border-left: 4px solid #fc8181;
    border-radius: 6px;
    padding: 10px 16px;
    margin-bottom: 6px;
    font-size: 0.85rem;
    color: #fed7d7;
    animation: icu-pulse 2s ease-in-out infinite;
}

.icu-alert-warning {
    background: linear-gradient(135deg, #744210 0%, #5c3510 100%);
    border-left: 4px solid #f6ad55;
    border-radius: 6px;
    padding: 10px 16px;
    margin-bottom: 6px;
    font-size: 0.85rem;
    color: #fefcbf;
}

.icu-alert-info {
    background: linear-gradient(135deg, #1a365d 0%, #153e75 100%);
    border-left: 4px solid #63b3ed;
    border-radius: 6px;
    padding: 10px 16px;
    margin-bottom: 6px;
    font-size: 0.85rem;
    color: #bee3f8;
}

@keyframes icu-pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.85; }
}

/* Vital card */
.vital-card {
    background: #1a1f2e;
    border: 1px solid #2d3748;
    border-radius: 8px;
    padding: 10px 14px;
    text-align: center;
}

.vital-label {
    font-size: 0.7rem;
    color: #718096;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.vital-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 2px 0;
}

.vital-unit {
    font-size: 0.7rem;
    color: #4a5568;
}

.vital-normal { border-color: #276749; }
.vital-warn   { border-color: #c05621; }
.vital-crit   { border-color: #c53030; }

/* Twin comparison cards */
.twin-card {
    background: #1a1f2e;
    border: 1px solid #2d3748;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 8px;
}

.twin-card-title {
    font-size: 0.82rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 4px;
}

.twin-card-desc {
    font-size: 0.72rem;
    color: #718096;
    line-height: 1.4;
}
</style>
"""


def inject_icu_css() -> None:
    """Inject the ICU dashboard CSS into the Streamlit page."""
    st.markdown(ICU_CSS, unsafe_allow_html=True)


def icu_panel(title: str) -> None:
    """Render a panel header with ICU styling."""
    st.markdown(
        f'<div class="icu-panel-header">{title}</div>',
        unsafe_allow_html=True,
    )
