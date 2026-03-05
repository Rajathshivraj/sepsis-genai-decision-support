"""
grounded_reasoner.py — Evidence-grounded LLM clinical reasoning.

Extends the existing reasoner.py with an enhanced prompt that includes
clinical guidelines, ensures consistency with model predictions, and
produces more structured, evidence-based reasoning.

This module does NOT modify reasoner.py.  It provides a new function
``generate_grounded_reasoning`` that wraps the existing Ollama call.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import requests

from configs.config import cfg
from src.utils.logger import setup_logger

logger = setup_logger("grounded_reasoner")


# ---------------------------------------------------------------------------
# Enhanced prompt template
# ---------------------------------------------------------------------------

_GROUNDED_PROMPT = """\
You are a clinical decision-support AI specialising in sepsis screening.

IMPORTANT: Your risk assessment MUST be consistent with the prediction scores.
- If ensemble score > 0.7 → you MUST classify as HIGH risk.
- If ensemble score 0.4–0.7 → you MUST classify as MODERATE risk.
- If ensemble score < 0.4 → you MUST classify as LOW risk.

═══════════════════════════════════════
PATIENT STATE
═══════════════════════════════════════
{patient_summary}

═══════════════════════════════════════
PREDICTION SCORES
═══════════════════════════════════════
ML Baseline Risk Score:   {ml_score:.4f}
LSTM Time-Series Score:   {lstm_score:.4f}
Ensemble Score:           {ensemble_score:.4f}
Model Agreement:          {agreement}

═══════════════════════════════════════
CLINICAL SCORES
═══════════════════════════════════════
{clinical_scores_section}

═══════════════════════════════════════
EVIDENCE BASE
═══════════════════════════════════════
{evidence_context}

═══════════════════════════════════════
TASK
═══════════════════════════════════════
Based on ALL the above information:

1. State the sepsis risk level (must match the ensemble score tier).
2. Explain your clinical reasoning referencing specific vitals, lab values,
   trends, and the clinical guidelines provided.
3. Cite which guidelines or evidence support your assessment.
4. Provide a confidence level.

Respond in EXACTLY this JSON format:

{{
  "sepsis_risk": "HIGH|MODERATE|LOW",
  "reasoning": "your clinical explanation citing evidence",
  "confidence": "your confidence level",
  "cited_guidelines": ["guideline names referenced"]
}}
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_grounded_reasoning(
    patient_summary: str,
    retrieved_cases: List[Dict[str, Any]],
    ml_score: float,
    lstm_score: float,
    ensemble_score: float,
    agreement: str,
    clinical_guidelines: Optional[List[Dict[str, str]]] = None,
    qsofa_result: Optional[Dict[str, Any]] = None,
    sofa_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate evidence-grounded clinical reasoning via Ollama.

    Parameters
    ----------
    patient_summary : str
        Natural-language patient case summary.
    retrieved_cases : list[dict]
        Similar historical cases.
    ml_score, lstm_score, ensemble_score : float
        Model risk scores.
    agreement : str
        Model agreement level.
    clinical_guidelines : list[dict], optional
        Matched clinical guidelines.
    qsofa_result, sofa_result : dict, optional
        Clinical scoring results.

    Returns
    -------
    dict
        Keys: sepsis_risk, reasoning, confidence, cited_guidelines.
    """
    # Build clinical scores section
    scores_section = _format_clinical_scores(qsofa_result, sofa_result)

    # Build evidence context
    evidence = _format_evidence(retrieved_cases, clinical_guidelines)

    # Build prompt
    prompt = _GROUNDED_PROMPT.format(
        patient_summary=patient_summary,
        ml_score=ml_score,
        lstm_score=lstm_score,
        ensemble_score=ensemble_score,
        agreement=agreement,
        clinical_scores_section=scores_section,
        evidence_context=evidence,
    )

    # Call Ollama
    raw = _call_ollama(prompt)

    # Parse response
    result = _parse_response(raw, ensemble_score)

    logger.info(
        "Grounded reasoning — risk=%s, confidence=%s, guidelines=%s",
        result.get("sepsis_risk"),
        result.get("confidence"),
        result.get("cited_guidelines"),
    )
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _format_clinical_scores(qsofa, sofa) -> str:
    parts = []
    if qsofa:
        parts.append(
            f"qSOFA: {qsofa.get('score', 'N/A')}/3 — "
            f"{qsofa.get('interpretation', '')}"
        )
    if sofa:
        parts.append(
            f"SOFA: {sofa.get('total_score', 'N/A')}/24 — "
            f"{sofa.get('interpretation', '')}"
        )
    return "\n".join(parts) if parts else "No clinical scores available."


def _format_evidence(cases, guidelines) -> str:
    parts = []
    if cases:
        for i, c in enumerate(cases[:5], 1):
            pid = c.get("patient_id", "?")
            score = c.get("score", 0)
            text = c.get("case", "")[:300]
            parts.append(f"[Case {i} — {pid}, sim={score:.3f}]\n{text}")
    if guidelines:
        parts.append("\n--- Clinical Guidelines ---")
        for g in guidelines:
            parts.append(f"[{g.get('title', '')}]\n{g.get('text', '')}")
    return "\n\n".join(parts) if parts else "No evidence available."


def _call_ollama(prompt: str) -> str:
    endpoint = f"{cfg.OLLAMA_BASE_URL}/api/generate"
    payload = {"model": cfg.OLLAMA_MODEL_NAME, "prompt": prompt, "stream": False}
    try:
        resp = requests.post(endpoint, json=payload, timeout=cfg.LLM_TIMEOUT)
        resp.raise_for_status()
        return resp.json().get("response", "")
    except Exception as exc:
        logger.error("Ollama error: %s", exc)
        return ""


def _parse_response(raw: str, ensemble_score: float) -> Dict[str, Any]:
    # Determine expected risk from ensemble score
    if ensemble_score >= 0.7:
        expected = "HIGH"
    elif ensemble_score >= 0.4:
        expected = "MODERATE"
    else:
        expected = "LOW"

    result = {
        "sepsis_risk": expected,
        "reasoning": "Evidence-grounded reasoning unavailable.",
        "confidence": "N/A",
        "cited_guidelines": [],
    }

    if not raw.strip():
        return result

    # Try JSON
    try:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1:
            parsed = json.loads(raw[start:end + 1])
            result.update(parsed)
    except Exception:
        # Regex fallback
        m = re.search(r'"reasoning"\s*:\s*"([^"]+)"', raw)
        if m:
            result["reasoning"] = m.group(1)

    # Enforce consistency: override LLM risk with model-derived risk
    result["sepsis_risk"] = expected
    return result
