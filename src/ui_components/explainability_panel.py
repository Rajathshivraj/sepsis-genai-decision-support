"""
explainability_panel.py — SHAP feature importance panel.
"""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st


def render_explainability_panel(results: Dict[str, Any]) -> None:
    """Render the SHAP risk drivers panel."""
    from src.ui_components.icu_layout import icu_panel
    icu_panel("🔬 Key Risk Drivers (SHAP)")

    shap_features = results.get("shap_features")

    if not shap_features:
        st.info("Feature importance data unavailable.")
        return

    try:
        from src.visualization.charts import create_feature_importance_chart
        st.plotly_chart(
            create_feature_importance_chart(shap_features),
            use_container_width=True
        )
    except Exception:
        for f in shap_features[:5]:
            st.markdown(
                f"• **{f['feature']}**: {f['shap_value']:+.4f} ({f['direction']})"
            )
