"""SVM-based reranking for retrieved chunks."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import joblib
import numpy as np

class SVMReranker:
    """Compute handcrafted features and rerank top context chunks."""
    def __init__(self, model_path: str = "models/reranker_svm.pkl"):
        """Load optional trained reranker model from disk if present."""
        self.model_path = Path(model_path)
        self.model = joblib.load(self.model_path) if self.model_path.exists() else None
    def extract_features(self, query: str, chunk: Dict, position_idx: int = 0) -> List[float]:
        """Build six reranking features from query-chunk pair."""
        q_words = query.lower().split(); c_words = chunk.get("text", "").lower().split()
        overlap = len(set(q_words) & set(c_words)); term_freq = sum(c_words.count(w) for w in q_words)
        chunk_len = len(c_words); length_score = 1.0 / (1.0 + abs(120 - chunk_len)); position_score = 1.0 / (1.0 + position_idx)
        return [float(chunk.get("semantic_score", 0.0)), float(chunk.get("bm25_score", 0.0)), float(term_freq), float(position_score), float(length_score), float(overlap)]
    def rerank(self, query: str, chunks: List[Dict], top_k: int = 3) -> List[Dict]:
        """Rank chunks by SVM decision score or robust fallback heuristic."""
        if not chunks:
            return []
        arr = np.array([self.extract_features(query, c, i) for i, c in enumerate(chunks)])
        scores = np.array(self.model.decision_function(arr)).reshape(-1) if (self.model is not None and hasattr(self.model, "decision_function")) else (arr[:, 0] * 0.45 + arr[:, 1] * 0.25 + arr[:, 2] * 0.1 + arr[:, 5] * 0.2)
        ranked = []
        for i, chunk in enumerate(chunks):
            item = dict(chunk); item["rerank_score"] = float(scores[i]); ranked.append(item)
        return sorted(ranked, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
