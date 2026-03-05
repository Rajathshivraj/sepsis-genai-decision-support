"""
advanced_rag.py — Advanced RAG retrieval with multi-source support.

Extends the existing RAG system to support retrieval from multiple
knowledge sources:
  1. ICU patient cases (existing FAISS index)
  2. Clinical guidelines (sepsis management protocols)

This is a standalone module that does NOT modify vector_store.py.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.utils.logger import setup_logger

logger = setup_logger("advanced_rag")

# ---------------------------------------------------------------------------
# Built-in clinical guidelines (embedded to avoid external file deps)
# ---------------------------------------------------------------------------

CLINICAL_GUIDELINES: Dict[str, str] = {
    "sepsis_3_definition": (
        "Sepsis-3 Definition: Sepsis is defined as life-threatening organ "
        "dysfunction caused by a dysregulated host response to infection. "
        "Organ dysfunction is identified as an acute change in total SOFA "
        "score ≥2 points consequent to the infection. Septic shock is a "
        "subset with circulatory and cellular/metabolic dysfunction "
        "associated with higher mortality (vasopressor requirement to "
        "maintain MAP ≥65 mmHg and serum lactate >2 mmol/L)."
    ),
    "surviving_sepsis_1hr_bundle": (
        "Surviving Sepsis Campaign 1-Hour Bundle: (1) Measure lactate level, "
        "re-measure if initial lactate >2 mmol/L. (2) Obtain blood cultures "
        "before administering antibiotics. (3) Administer broad-spectrum "
        "antibiotics. (4) Begin rapid administration of 30 mL/kg crystalloid "
        "for hypotension or lactate ≥4 mmol/L. (5) Apply vasopressors if "
        "hypotensive during or after fluid resuscitation to maintain MAP ≥65 mmHg."
    ),
    "qsofa_screening": (
        "qSOFA Screening: The quick SOFA (qSOFA) score uses three bedside "
        "criteria: respiratory rate ≥22/min, altered mentation (GCS <15), "
        "and systolic blood pressure ≤100 mmHg. A qSOFA score ≥2 identifies "
        "patients at risk of poor outcomes from sepsis outside the ICU."
    ),
    "lactate_management": (
        "Lactate-Guided Resuscitation: Elevated lactate (>2 mmol/L) indicates "
        "tissue hypoperfusion. Serial lactate measurements guide resuscitation "
        "adequacy. Target lactate normalisation (clearance >10-20% per 2 hours). "
        "Persistent elevation suggests ongoing shock."
    ),
    "antibiotic_timing": (
        "Antibiotic Timing: Each hour of delay in antibiotic administration "
        "is associated with increased mortality in septic shock. Broad-spectrum "
        "antibiotics should be administered within 1 hour of sepsis recognition. "
        "De-escalation should follow once pathogen and sensitivities are known."
    ),
    "fluid_resuscitation": (
        "Fluid Resuscitation: Initial crystalloid bolus of 30 mL/kg within "
        "the first 3 hours for sepsis-induced hypoperfusion. Reassess fluid "
        "responsiveness using dynamic measures (pulse pressure variation, "
        "passive leg raise). Avoid fluid overload."
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve_with_guidelines(
    query_text: str,
    patient_cases: Optional[List[Dict[str, Any]]] = None,
    k_cases: int = 5,
    k_guidelines: int = 3,
) -> Dict[str, Any]:
    """Multi-source retrieval: patient cases + clinical guidelines.

    Parameters
    ----------
    query_text : str
        Patient case summary or clinical query.
    patient_cases : list[dict], optional
        Pre-retrieved patient cases from existing RAG.
    k_cases : int
        Number of patient cases to include.
    k_guidelines : int
        Number of guidelines to match.

    Returns
    -------
    dict
        Keys: ``patient_cases``, ``clinical_guidelines``,
        ``combined_context``.
    """
    if patient_cases is None:
        patient_cases = []

    # Retrieve relevant clinical guidelines
    matched_guidelines = _match_guidelines(query_text, k=k_guidelines)

    # Trim patient_cases to k
    cases_trimmed = patient_cases[:k_cases]

    # Build combined context for LLM prompt
    combined = _build_combined_context(cases_trimmed, matched_guidelines)

    logger.info(
        "Advanced RAG — %d patient cases, %d guidelines matched",
        len(cases_trimmed), len(matched_guidelines),
    )

    return {
        "patient_cases": cases_trimmed,
        "clinical_guidelines": matched_guidelines,
        "combined_context": combined,
    }


def get_relevant_guidelines(
    vitals: Dict[str, float],
) -> List[Dict[str, str]]:
    """Return guidelines relevant to the patient's current vitals.

    Parameters
    ----------
    vitals : dict
        Patient vitals dict.

    Returns
    -------
    list[dict]
        Each entry: ``{"guideline_id", "title", "text"}``.
    """
    relevant = []

    lactate = vitals.get("Lactate", 0)
    map_val = vitals.get("MAP", 80)
    resp = vitals.get("Resp", 16)

    # Always include sepsis definition
    relevant.append({
        "guideline_id": "sepsis_3_definition",
        "title": "Sepsis-3 Definition",
        "text": CLINICAL_GUIDELINES["sepsis_3_definition"],
    })

    if lactate > 2.0:
        relevant.append({
            "guideline_id": "lactate_management",
            "title": "Lactate-Guided Resuscitation",
            "text": CLINICAL_GUIDELINES["lactate_management"],
        })

    if map_val < 65 or lactate > 4.0:
        relevant.append({
            "guideline_id": "surviving_sepsis_1hr_bundle",
            "title": "1-Hour Bundle",
            "text": CLINICAL_GUIDELINES["surviving_sepsis_1hr_bundle"],
        })
        relevant.append({
            "guideline_id": "fluid_resuscitation",
            "title": "Fluid Resuscitation",
            "text": CLINICAL_GUIDELINES["fluid_resuscitation"],
        })

    if resp >= 22:
        relevant.append({
            "guideline_id": "qsofa_screening",
            "title": "qSOFA Screening",
            "text": CLINICAL_GUIDELINES["qsofa_screening"],
        })

    return relevant


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _match_guidelines(query: str, k: int = 3) -> List[Dict[str, str]]:
    """Simple keyword-based guideline matching."""
    query_lower = query.lower()
    scored = []
    for gid, text in CLINICAL_GUIDELINES.items():
        q_words = set(query_lower.split())
        g_words = set(text.lower().split())
        overlap = len(q_words & g_words)
        scored.append((overlap, gid, text))
    scored.sort(reverse=True)

    results = []
    for _, gid, text in scored[:k]:
        results.append({
            "guideline_id": gid,
            "title": gid.replace("_", " ").title(),
            "text": text,
        })
    return results


def _build_combined_context(
    cases: List[Dict[str, Any]],
    guidelines: List[Dict[str, str]],
) -> str:
    """Build a combined context string for LLM prompt injection."""
    parts = []

    if cases:
        parts.append("═══ SIMILAR HISTORICAL CASES ═══")
        for i, c in enumerate(cases, 1):
            pid = c.get("patient_id", "unknown")
            score = c.get("score", 0)
            text = c.get("case", "")
            parts.append(f"Case {i} (ID: {pid}, sim: {score:.3f}):\n{text}")

    if guidelines:
        parts.append("\n═══ CLINICAL GUIDELINES ═══")
        for g in guidelines:
            parts.append(f"[{g['title']}]\n{g['text']}")

    return "\n\n".join(parts)
