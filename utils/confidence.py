"""Confidence scoring helper for answer cards."""
from __future__ import annotations

def compute_confidence(intent_conf: float, chunk_scores):
    """Combine intent confidence and retrieval quality into one score."""
    if not chunk_scores:
        return round(intent_conf * 100, 2)
    retrieval = sum(chunk_scores) / len(chunk_scores)
    return round(max(0.0, min(1.0, 0.6 * intent_conf + 0.4 * retrieval)) * 100, 2)
