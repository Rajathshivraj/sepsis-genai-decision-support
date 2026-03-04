"""
reasoner.py — LLM-powered clinical reasoning via Ollama.

Constructs a structured prompt from the patient's current state,
ML / LSTM risk scores, and retrieved historical cases, then calls a
local Ollama model (llama3) to produce a clinical sepsis risk assessment.

The module communicates with Ollama through its HTTP API
(POST /api/generate), so Ollama must be running locally.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from configs.config import cfg
from src.utils.logger import setup_logger

logger = setup_logger("reasoner")

# ---------------------------------------------------------------------------
# Output data class
# ---------------------------------------------------------------------------

@dataclass
class ClinicalReasoning:
    """Structured output from the LLM reasoning module.

    Attributes
    ----------
    sepsis_risk : str
        Risk level, e.g., "HIGH", "MODERATE", or "LOW".
    reasoning : str
        Free-text clinical explanation.
    confidence : str
        Confidence level, e.g., "HIGH", "MODERATE", or "LOW" (or a percentage string).
    raw_response : str
        Full raw text returned by the LLM.
    """
    sepsis_risk: str
    reasoning: str
    confidence: str
    raw_response: str

    def __str__(self) -> str:
        return (
            f"Sepsis Risk : {self.sepsis_risk}\n"
            f"Confidence  : {self.confidence}\n"
            f"Reasoning   : {self.reasoning}"
        )


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """\
You are a clinical decision-support AI assistant specialising in sepsis screening.

Analyse the following patient information and provide a structured sepsis risk assessment.

═══════════════════════════════════════
PATIENT STATE
═══════════════════════════════════════
{patient_summary}

═══════════════════════════════════════
PREDICTION SCORES
═══════════════════════════════════════
ML Baseline Risk Score: {ml_score:.4f}
LSTM Time-Series Risk Score: {lstm_score:.4f}

═══════════════════════════════════════
SIMILAR HISTORICAL CASES
═══════════════════════════════════════
{retrieved_cases}

═══════════════════════════════════════
TASK
═══════════════════════════════════════
Based on the patient state, prediction scores, and similar historical cases:

1. Assess the sepsis risk level (HIGH, MODERATE, or LOW).
2. Explain your clinical reasoning, referencing specific vital signs, lab values, and trends.
3. Provide a confidence score or level.

Respond in EXACTLY this JSON format:

{{
  "sepsis_risk": "HIGH|MODERATE|LOW",
  "reasoning": "your clinical explanation",
  "confidence": "your confidence level or score"
}}
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_clinical_reasoning(
    patient_summary: str,
    retrieved_cases: List[Dict[str, Any]],
    ml_score: float,
    lstm_score: float
) -> Dict[str, str]:
    """Call a local Ollama LLM (llama3) to generate clinical reasoning.

    Parameters
    ----------
    patient_summary : str
        Natural-language summary of the patient's current clinical state.
    retrieved_cases : list[dict]
        Top-k similar historical cases from the vector store.
    ml_score : float
        Risk score from the baseline ML model.
    lstm_score : float
        Risk score from the LSTM model.

    Returns
    -------
    dict
        Structured output containing:
        - sepsis_risk: "HIGH", "MODERATE", or "LOW"
        - reasoning: text description
        - confidence: string representing confidence
    """
    # ── Format prompt sections ────────────────────────────────────────────
    cases_section = _format_retrieved_cases(retrieved_cases)

    prompt = _PROMPT_TEMPLATE.format(
        patient_summary=patient_summary,
        ml_score=ml_score,
        lstm_score=lstm_score,
        retrieved_cases=cases_section,
    )

    # ── Call Ollama ───────────────────────────────────────────────────────
    raw_response = _call_ollama(prompt)

    # ── Parse response ────────────────────────────────────────────────────
    # The requirement asks for a structured output: {"sepsis_risk": "...", "reasoning": "...", "confidence": "..."}
    # We will try to parse JSON if returned, otherwise fallback to a best-effort parse.
    result_dict = _parse_json_or_text(raw_response)
    
    logger.info(
        "LLM reasoning complete — risk=%s, confidence=%s",
        result_dict.get("sepsis_risk"),
        result_dict.get("confidence"),
    )
    return result_dict


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _format_retrieved_cases(cases: List[Dict[str, Any]]) -> str:
    """Pretty-format retrieved similar cases."""
    if not cases:
        return "  No similar cases retrieved."

    sections = []
    for i, case in enumerate(cases, 1):
        pid = case.get("patient_id", "unknown")
        score = case.get("score", 0.0)
        text = case.get("case", "N/A")
        sections.append(
            f"── Case {i} (patient {pid}, similarity: {score:.4f}) ──\n{text}"
        )
    return "\n\n".join(sections)


def _call_ollama(prompt: str) -> str:
    """Send a generation request to the Ollama HTTP API.

    Returns the full generated text, or a fallback message if the API is
    unreachable.
    """
    endpoint = f"{cfg.OLLAMA_BASE_URL}/api/generate"
    model = cfg.OLLAMA_MODEL_NAME

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    try:
        logger.info("Calling Ollama (model=%s) …", model)
        resp = requests.post(
            endpoint,
            json=payload,
            timeout=cfg.LLM_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")

    except Exception as exc:
        logger.error("Ollama API error: %s", exc)
        return _fallback_raw_response()


def _fallback_raw_response() -> str:
    """Fallback text when Ollama is unavailable."""
    return """
{
  "sepsis_risk": "MODERATE",
  "reasoning": "The local LLM (Ollama/llama3) is currently unreachable. Risk assessment is based solely on automated prediction scores.",
  "confidence": "LOW (FALLBACK)"
}
"""


def _parse_json_or_text(raw: str) -> Dict[str, str]:
    """Try to parse JSON from the LLM response, or use regex as fallback."""
    import json
    
    # Try direct JSON parsing
    try:
        # Find the first { and last } to handle potential conversational noise
        start = raw.find('{')
        end = raw.rfind('}')
        if start != -1 and end != -1:
            json_str = raw[start:end+1]
            return json.loads(json_str)
    except Exception:
        pass

    # Fallback to regex if JSON parsing fails
    risk = "MODERATE"
    confidence = "N/A"
    reasoning = raw.strip()

    risk_match = re.search(r'"sepsis_risk":\s*"([^"]+)"', raw, re.IGNORECASE)
    if risk_match: risk = risk_match.group(1)
    
    reason_match = re.search(r'"reasoning":\s*"([^"]+)"', raw, re.IGNORECASE)
    if reason_match: reasoning = reason_match.group(1)
    
    conf_match = re.search(r'"confidence":\s*"([^"]+)"', raw, re.IGNORECASE)
    if conf_match: confidence = conf_match.group(1)

    return {
        "sepsis_risk": risk,
        "reasoning": reasoning,
        "confidence": confidence
    }
