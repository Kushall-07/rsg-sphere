"""Hybrid retrieval utilities combining BM25 and vector search."""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, List
from rank_bm25 import BM25Okapi

def bm25_search(query: str, chunks: List[Dict], k: int = 10) -> List[Dict]:
    """Score chunks by BM25 relevance for lexical matching signals."""
    if not chunks:
        return []
    bm25 = BM25Okapi([c["text"].lower().split() for c in chunks])
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted([{**chunk, "bm25_score": float(scores[i])} for i, chunk in enumerate(chunks)], key=lambda x: x["bm25_score"], reverse=True)
    return ranked[:k]

def hybrid_fusion(semantic_chunks: List[Dict], bm25_chunks: List[Dict], alpha: float = 0.7) -> List[Dict]:
    """Fuse semantic and lexical scores into a single ranking."""
    merged = defaultdict(dict)
    sem_max = max((c.get("semantic_score", 0.0) for c in semantic_chunks), default=1.0) or 1.0
    bm_max = max((c.get("bm25_score", 0.0) for c in bm25_chunks), default=1.0) or 1.0
    for c in semantic_chunks:
        key = f"{c.get('filename')}::{c.get('page_number')}::{hash(c.get('text'))}"
        merged[key].update(c); merged[key]["semantic_norm"] = c.get("semantic_score", 0.0) / sem_max
    for c in bm25_chunks:
        key = f"{c.get('filename')}::{c.get('page_number')}::{hash(c.get('text'))}"
        merged[key].update(c); merged[key]["bm25_norm"] = c.get("bm25_score", 0.0) / bm_max
    fused = []
    for item in merged.values():
        item["fusion_score"] = alpha * item.get("semantic_norm", 0.0) + (1 - alpha) * item.get("bm25_norm", 0.0)
        fused.append(item)
    return sorted(fused, key=lambda x: x["fusion_score"], reverse=True)
